import logging
import os
import time
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from dataset import create_dataset, create_loader, create_stateful_sampler
from dataset.serialize import local_broadcast_process_authkey
from tasks_clip.shared_utils import setup_model, get_media_types

from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
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
    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)

    train_loader = train_loaders[0] if len(train_loaders) == 1 else None
    if train_loader is None:
        train_loader = train_loaders[0]


    model_without_ddp = model.module if config.distributed else model
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    num_batches = len(train_loader)


    for i, (videos, text, start_times, end_times, lengths, fpss) in enumerate(iterator):
        videos = videos.to(device, non_blocking=True)
        text_input = tokenizer(text, padding=True, return_tensors="pt").to(device)

        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            text_feat = model_without_ddp.encode_text(text_input)
            gamma, beta = model_without_ddp.streaming_vision_encoder.rnn.prepare_prompt(text_feat)

            state = model_without_ddp.streaming_vision_encoder.init_hidden(videos.size(0), device)
            all_scores = []

            for t in range(videos.shape[1]):
                frame = videos[:, t]
                vfeat, state = model_without_ddp.get_streaming_vid_feat(frame, state, gamma=gamma, beta=beta)
                score = torch.nn.functional.cosine_similarity(vfeat, text_feat, dim=-1)
                all_scores.append(score)

            scores = torch.stack(all_scores, dim=1)

            # Create labels from the localization data.
            # A time vector is broadcasted against start and end times.
            #
            # The labels represent binary masks, which indicate whether each time step
            # corresponds to an action occurrence in the video. The mask has shape [B, T],
            # where:
            # - B = batch size
            # - T = number of time steps (video frames)
            #
            # For each video in the batch, we create a binary sequence where:
            # 1.0 = time step is within the ground-truth action interval
            # 0.0 = time step is outside the action interval
            #
            # This is done by broadcasting comparisons between:
            # - time_vector: [1, T] tensor containing all time indices
            # - start_frames: [B, 1] tensor of ground-truth start times
            # - end_frames: [B, 1] tensor of ground-truth end times (clamped to video length)
            #
            # The final labels tensor is used for binary cross-entropy loss calculation.
            time_vector = torch.arange(scores.shape[1], device=device)[None, :]
            start_frames = start_times.to(device)[:, None]

            # Clamp end times to the actual video length to avoid applying labels to padding.
            # This is so that we don't create labels for time steps beyond the actual video duration.
            end_frames = torch.min(end_times.to(device), lengths.to(device))[:, None]
            labels = ((time_vector >= start_frames) & (time_vector < end_frames)).float()

            # Create a mask that is True for valid frames and False for padded positions.
            valid_mask = time_vector < lengths.to(device)[:, None]

            # Compute BCE loss only over the valid frames to avoid back-propagating
            # through the padded regions of the videos.
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                scores,
                labels,
                reduction="none",
            )
            bce = (bce * valid_mask.float()).sum() / valid_mask.sum()

            # For cross entropy we also need to ignore logits corresponding to
            # padded frames so that the softmax normalisation is computed only
            # over valid time steps for each video in the batch.
            scores_masked = scores.masked_fill(~valid_mask, float("-inf"))

            peak = ((start_times + end_times) // 2)
            peak = torch.min(peak, lengths.to(device) - 1).to(device, dtype=torch.long)
            ce = torch.nn.functional.cross_entropy(scores_masked, peak)

            loss = bce + ce


        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            model.backward(loss)
            model.step()
        else:
            optimizer.zero_grad()
            if config.use_half_precision:
                scaler.scale(loss).backward()
                if config.optimizer.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.optimizer.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                optimizer.step()
            scheduler.step()


        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        global_step += 1

        if config.debug and (i + 1) % 5 == 0:
            break


    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")
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
        optimizer,
        scheduler,
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
