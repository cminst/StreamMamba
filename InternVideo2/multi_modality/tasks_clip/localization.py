import logging
import os
import time
import datetime
import math
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from dataset import create_dataset, create_loader, create_stateful_sampler
from dataset.serialize import local_broadcast_process_authkey
from tasks_clip.shared_utils import setup_model, get_media_types
from utils.scheduler import create_scheduler

from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
from models import *
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

logger = logging.getLogger(__name__)


def localization_collate_fn(batch):
    """Pad variable length videos and return tensors."""
    vids, caps, starts, ends, lens, fpss = [], [], [], [], [], []
    max_len = max(v.shape[0] for v, _, _, _, _ in batch)
    for video, cap, s, e, fps in batch:
        pad = torch.zeros(max_len, *video.shape[1:], dtype=video.dtype)
        pad[: video.shape[0]] = video
        vids.append(pad)
        caps.append(cap)
        starts.append(int(s * fps))
        ends.append(int(e * fps))
        lens.append(video.shape[0])
        fpss.append(fps)
    return (
        torch.stack(vids),
        caps,
        torch.tensor(starts, dtype=torch.long),
        torch.tensor(ends, dtype=torch.long),
        torch.tensor(lens, dtype=torch.long),
        torch.tensor(fpss, dtype=torch.float),
    )


def setup_dataloaders(config):
    logger.info("Creating localization dataset")
    train_datasets = create_dataset("loc_train", config)
    media_types = get_media_types(train_datasets)

    batch_size = [config.inputs.batch_size[m] for m in media_types]
    samplers = create_stateful_sampler(train_datasets, batch_size)

    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size=batch_size,
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=[localization_collate_fn] * len(media_types),
    )

    return train_loaders, media_types


def train(model, train_loaders, optimizer, tokenizer, epoch, global_step, device, scheduler, scaler, config, data_type):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    metric_logger.add_meter("total_loss", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("pred_loss", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("skip_loss", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("current_lambda", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("alpha_pred_loss", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("is_phase1", SmoothedValue(window=1, fmt="{value:.0f}"))
    metric_logger.add_meter("global_step", SmoothedValue(window=1, fmt="{value:.0f}"))
    metric_logger.add_meter("loc_loss", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("bce_loss", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("ce_loss", SmoothedValue(window=1, fmt="{value:.4f}"))
    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)

    assert len(train_loaders) > 0, "There must be at least one train DataLoader"

    # Only use the first dataset
    train_loader = train_loaders[0]

    model_without_ddp = model.module if config.distributed else model
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    num_batches = len(train_loader)

    spfs_cfg = getattr(config, "SPFS_CONFIG", {})
    phase1_epochs = spfs_cfg.get("phase1_epochs", 0)
    phase1_steps = phase1_epochs * num_batches
    ramp_up_steps = max(1, int(0.25 * num_batches))

    # Set phase-based parameter freezing at the start of each epoch
    current_phase = 1 if epoch < phase1_epochs else 2

    # Only call set_phase_freezing if SPFS is enabled
    if spfs_cfg.get("enabled", False):
        model_without_ddp.set_phase_freezing(current_phase, spfs_cfg)
        logger.info(f"Epoch {epoch}: Training in Phase {current_phase} (Phase 1 epochs: {phase1_epochs})")
    else:
        logger.info(f"Epoch {epoch}: SPFS disabled, training normally")

    for i, (videos, text, start_times, end_times, lengths, fpss) in enumerate(iterator):
        is_phase1 = epoch < phase1_epochs

        videos = videos.to(device, non_blocking=True)
        text_input = tokenizer(text, padding=True, return_tensors="pt").to(device)

        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            text_feat = model_without_ddp.encode_text(text_input)
            gamma, beta = model_without_ddp.streaming_vision_encoder.rnn.prepare_prompt(text_feat)

            vit = model_without_ddp.streaming_vision_encoder.vit_lite
            rnn = model_without_ddp.streaming_vision_encoder.rnn

            state = model_without_ddp.streaming_vision_encoder.init_hidden(videos.size(0), device)
            scores_list = []
            pred_losses = []
            skip_probs = []

            feat, _ = vit.extract_features(videos[:, 0])
            out, state = rnn(feat, state, gamma, beta)
            if model_without_ddp.config.model.use_streaming_vision_align:
                out = model_without_ddp.streaming_vision_align(out)
            else:
                out = model_without_ddp.vision_align(out)
            out = out / out.norm(dim=-1, keepdim=True)
            scores_list.append(torch.nn.functional.cosine_similarity(out, text_feat, dim=-1))

            for t in range(1, videos.shape[1]):
                mu_t, logv_t = rnn.predict_next_feat()
                gt_feat, _ = vit.extract_features(videos[:, t])

                pred_losses.append(0.5 * ((gt_feat - mu_t).pow(2) * torch.exp(-logv_t) + logv_t).mean())

                conf = torch.exp(-logv_t)
                skip_prob = torch.sigmoid(spfs_cfg.get("skip_decision_k", 10.0) * (conf - spfs_cfg.get("skip_decision_theta", 0.7)))
                s_t = (skip_prob > 0.5).float().detach() + skip_prob - skip_prob.detach()
                skip_probs.append(s_t)

                if not is_phase1 and spfs_cfg.get("enabled", False):
                    p_pred = 1.0 - math.exp(-global_step / spfs_cfg.get("scheduled_sampling_I0", 20000.0))
                    use_pred = torch.rand(1, device=device) < p_pred
                    frame_in = mu_t.detach() if use_pred else gt_feat
                else:
                    frame_in = gt_feat

                out, state = rnn(frame_in, state, gamma, beta)
                if config.model.get('use_streaming_vision_align', False):
                    out = model_without_ddp.streaming_vision_align(out)
                else:
                    out = model_without_ddp.vision_align(out)
                out = out / out.norm(dim=-1, keepdim=True)
                scores_list.append(torch.nn.functional.cosine_similarity(out, text_feat, dim=-1))

            scores = torch.stack(scores_list, dim=1)

            time_vector = torch.arange(scores.shape[1], device=device)[None, :]
            start_frames = start_times.to(device)[:, None]
            end_frames = torch.min(end_times.to(device), lengths.to(device))[:, None]
            labels = ((time_vector >= start_frames) & (time_vector < end_frames)).float()
            valid_mask = time_vector < lengths.to(device)[:, None]

            bce = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels, reduction="none")
            bce = (bce * valid_mask.float()).sum() / valid_mask.sum()

            scores_masked = scores.masked_fill(~valid_mask, float("-inf"))
            peak = ((start_times + end_times) // 2)

            # Perform min on CPU tensors first, then move the result to GPU
            peak = torch.min(peak, lengths - 1).to(device, dtype=torch.long)
            ce = torch.nn.functional.cross_entropy(scores_masked, peak)

            loc_loss = bce + ce
            pred_loss_all = torch.stack(pred_losses).mean() if pred_losses else torch.tensor(0.0, device=device)
            skip_loss = torch.stack(skip_probs).mean() if skip_probs else torch.tensor(0.0, device=device)

            if is_phase1 or not spfs_cfg.get("enabled", False):
                total_loss = pred_loss_all
                current_lambda = 0.0
                alpha_pred_loss = 0.0
            else:
                current_lambda = spfs_cfg.get("lambda_skip_loss", 0.01) * min(1.0, max(0.0, (global_step - phase1_steps) / ramp_up_steps))
                alpha_pred_loss = spfs_cfg.get("alpha_pred_loss", 0.2)
                total_loss = loc_loss + alpha_pred_loss * pred_loss_all + current_lambda * skip_loss

        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            model.backward(total_loss)
            model.step()
        else:
            optimizer.zero_grad()
            if config.use_half_precision:
                scaler.scale(total_loss).backward()
                if config.optimizer.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if config.optimizer.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                optimizer.step()
            if not is_phase1 and spfs_cfg.get("enabled", False):
                scheduler.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(total_loss=total_loss.item())
        metric_logger.update(pred_loss=pred_loss_all.item())
        metric_logger.update(skip_loss=skip_loss.item())
        metric_logger.update(current_lambda=current_lambda)
        metric_logger.update(alpha_pred_loss=alpha_pred_loss)
        metric_logger.update(is_phase1=float(is_phase1))
        metric_logger.update(global_step=global_step)

        if not is_phase1 and spfs_cfg.get("enabled", False):
            metric_logger.update(loc_loss=loc_loss.item())
            metric_logger.update(bce_loss=bce.item())
            metric_logger.update(ce_loss=ce.item())

        global_step += 1

        if (i + 1) % log_freq == 0:
            if is_main_process():
                log_payload = {
                    "lr": optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss.item(),
                    "pred_loss": pred_loss_all.item(),
                    "skip_loss": skip_loss.item(),
                }
                if not is_phase1 and spfs_cfg.get("enabled", False):
                    log_payload["loc_loss"] = loc_loss.item()

                logger.info(f"{header} [Step {i}] {metric_logger}")

            if is_main_process() and config.wandb.enable:
                # Log the running averages to wandb at each step
                log_dict_to_wandb(
                    metric_logger.get_global_avg_dict(),
                    step=global_step,
                    prefix="train/"
                )

        if config.debug and (i + 1) % 5 == 0:
            break

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats for Epoch [{epoch}]: {metric_logger.global_avg()}")
    if is_main_process() and config.wandb.enable:
        log_dict_to_wandb(metric_logger.get_global_avg_dict(), step=global_step, prefix=f"epoch_{epoch}/")

    return global_step


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    train_loaders, media_types = setup_dataloaders(config)
    num_steps_per_epoch = sum(len(d) for d in train_loaders)

    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs

    model_cls = eval(config.model.get('model_cls', 'InternVideo2_CLIP'))
    (
        model,
        model_without_ddp,
        _,
        _,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        pretrain=True,
        find_unused_parameters=True,
        num_steps_per_epoch=num_steps_per_epoch,
    )

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=config.optimizer.lr, betas=tuple(config.optimizer.opt_betas), weight_decay=config.optimizer.weight_decay)
    scheduler = create_scheduler(config.scheduler, optimizer)

    ckpt = config.model.get('cross_mamba_film_ckpt', '')
    if ckpt:
        logger.info(f"Loading cross mamba FiLM weights from {ckpt}")
        state = torch.load(ckpt, map_location='cpu')
        if 'model' in state:
            state = state['model']
        msg = model_without_ddp.load_state_dict(state, strict=False)
        logger.info(msg)
    if is_main_process() and config.wandb.enable:
        import wandb
        wandb.watch(model)

    data_type = torch.bfloat16 if config.get('use_bf16', True) else torch.float16

    logger.info("Start training")
    start_time = time.time()

    # Set initial phase freezing if SPFS is enabled
    spfs_cfg = getattr(config, "SPFS_CONFIG", {})
    if spfs_cfg.get("enabled", False):
        phase1_epochs = spfs_cfg.get("phase1_epochs", 0)
        initial_phase = 1 if start_epoch < phase1_epochs else 2
        model_without_ddp.set_phase_freezing(initial_phase, spfs_cfg)
        logger.info(f"Initialized training with Phase {initial_phase} parameter freezing")

    for epoch in range(start_epoch, config.scheduler.epochs):
        global_step = train(
            model,
            train_loaders,
            optimizer,
            tokenizer,
            epoch,
            global_step,
            device,
            scheduler,
            scaler,
            config,
            data_type,
        )
        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            tag = f"ckpt_{epoch:02d}.pth" if not config.get("save_latest", False) else "ckpt_latest.pth"
            model.save_checkpoint(config.output_dir, tag=tag, save_latest=False, exclude_frozen_parameters=True)
        elif is_main_process():
            state_dict = model_without_ddp.state_dict()
            param_grad_dict = {k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()}
            for k in list(state_dict.keys()):
                if k in param_grad_dict.keys() and not param_grad_dict[k]:
                    del state_dict[k]
            save_obj = {
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "global_step": global_step,
            }
            torch.save(save_obj, os.path.join(config.output_dir, f"ckpt_{epoch:02d}.pth"))
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
