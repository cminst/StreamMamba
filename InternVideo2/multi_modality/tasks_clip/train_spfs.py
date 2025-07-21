import datetime
import logging
import os
import sys
import time
import random
import numpy as np
import cv2
import argparse

from os.path import join

# Check for dataset path
if not os.environ.get("DATASET_ROOT"):
    raise RuntimeError(
        "DATASET_ROOT environment variable is not set. "
        "Please export DATASET_ROOT to point to your dataset root before running this script."
    )

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from PIL import Image
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from dataset import (
    MetaLoader_rs,
    create_dataset,
    create_loader,
    create_stateful_sampler,
    add_precomputed_embeddings,
)
from dataset.serialize import local_broadcast_process_authkey
from models import *
from tasks_clip.shared_utils import get_media_types, setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

logger = logging.getLogger(__name__)

# Main training function
def train(
    model,
    train_loaders,
    optimizer,
    _,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
    data_type,
    skip_num=0,
    test_video_path=None,
):
    """
    Hybrid SPFS training loop.
    - Phase 1 (epoch 0): Warms up the prediction head and also feeds video
      through the Mamba so the hidden state is updated correctly.
    - Phase 2 (epoch > 0): Jointly fine-tunes all components.
    """

    model_without_ddp = model.module if config.distributed else model
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.10f}"))
    metric_logger.add_meter("different_lr", SmoothedValue(window=1, fmt="{value:.10f}"))
    metric_logger.add_meter("L_pred", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("L_calib", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("L_skip", SmoothedValue(window=1, fmt="{value:.4f}"))

    media_types = get_media_types(train_loaders)
    if config.distributed:
        for loader in train_loaders:
            loader.sampler.set_epoch(epoch)

    seed = config.seed + epoch
    train_loader_agg = MetaLoader_rs(
        name2loader=dict(zip(media_types, train_loaders)),
        skip_num=skip_num,
        seed=seed,
    )

    progress_bar = tqdm(
        train_loader_agg,
        total=len(train_loader_agg),
        desc=f"Training: [Epoch {epoch}]",
        disable=not is_main_process(),
    )

    # Load test video for SPFS testing (if provided)
    test_frames = None
    if is_main_process() and test_video_path is not None and os.path.exists(test_video_path):
        try:
            # Import utility functions
            sys.path.append(os.getcwd())
            from demo.utils import _frame_from_video, frames2tensor
            test_cap = cv2.VideoCapture(test_video_path)
            test_frames = [x for x in _frame_from_video(test_cap)]
            logger.info(f"Loaded {len(test_frames)} frames from test video: {test_video_path}")
        except Exception as e:
            logger.error(f"Failed to load test video: {e}")

    # Loss fns and hyper-param
    cosine_loss_fn = lambda pred, target: 1 - torch.nn.functional.cosine_similarity(
        pred, target.detach(), dim=-1
    ).mean()
    bce_loss_fn = BCEWithLogitsLoss()
    mse_loss_fn = MSELoss()

    calibration_loss_type = config.get("calibration_loss_fn", "bce").lower()
    if calibration_loss_type not in ["bce", "mse"]:
        raise ValueError(
            f"Unsupported calibration_loss_fn {calibration_loss_type}. Choose 'bce' or 'mse'."
        )

    lambda_calib = config.get("lambda_calib", 1.0)
    lambda_skip = config.get("lambda_skip", 0.1)
    MODEL_MAX_FRAMES = config.num_frames

    for i, data_pair in enumerate(progress_bar):
        batch = data_pair[1]
        if len(batch) == 4:
            image, _, _, _ = batch
        else:
            image, _, _ = batch

        image = image.to(device, non_blocking=True)
        image = image.permute(0, 2, 1, 3, 4)

        B, C, T, H, W = image.shape
        assert T >= MODEL_MAX_FRAMES, f"Video batch contains sequences shorter than {MODEL_MAX_FRAMES} frames."

        with torch.amp.autocast('cuda', enabled=config.use_half_precision, dtype=data_type):
            # Warm up hidden state
            h = model.streaming_vision_encoder.init_hidden(batch_size=B, device=device)

            for t in range(MODEL_MAX_FRAMES - 1):
                frame = image[:, :, t, :, :].unsqueeze(2) # [B, C, 1, H, W]
                with torch.no_grad():
                    _, h, _ = model.streaming_vision_encoder(frame, h)

            num_steps = T - (MODEL_MAX_FRAMES - 1)
            loss_pred_acc = loss_calib_acc = loss_skip_acc = 0.0

            for step in range(num_steps):
                idx_curr = (MODEL_MAX_FRAMES - 1) + step
                frame_curr = image[:, :, idx_curr, :, :].unsqueeze(2)

                # Forward through Mamba
                out_t, new_h, _ = model.streaming_vision_encoder(frame_curr, h)
                out_t = model_without_ddp.vision_align(out_t)

                # Teacher targets
                with torch.no_grad():
                    # Next-frame target (MobileCLIP)
                    if idx_curr + 1 < T:
                        next_frame = image[:, :, idx_curr + 1, :, :]
                    else:
                        next_frame = image[:, :, idx_curr, :, :]

                    target_next, _ = model_without_ddp.streaming_vision_encoder.vit_lite.extract_features(next_frame)

                # ----------

                mu_t, logvar = model.streaming_vision_encoder.rnn.predict_next_feat()
                conf_logit = -logvar.squeeze(-1)

                L_pred = torch.tensor(0.0, device=device)

                if epoch == 0:
                    # Phase-1: only train predictor head (primary & other losses = 0)
                    loss = L_pred = cosine_loss_fn(mu_t, target_next)
                    L_calib = L_skip = torch.tensor(0.0, device=device) # Set others to 0
                else:
                    # Phase-2: joint fine-tuning
                    with torch.no_grad():
                        sim_score = torch.nn.functional.cosine_similarity(
                            mu_t, target_next, dim=-1
                        )

                    if calibration_loss_type == "bce":
                        target_c = (sim_score >= 0.85).float()
                        L_calib = bce_loss_fn(conf_logit.squeeze(-1), target_c)
                    else:  # mse
                        L_calib = mse_loss_fn(
                            torch.sigmoid(conf_logit.squeeze(-1)), sim_score
                        )

                    # Logging actual similarity and predicted confidence
                    if i % config.log_freq == 0 and step == 0:  # Log once per batch, at first step
                        sim_mean = sim_score.mean().item()
                        conf_mean = torch.sigmoid(conf_logit).mean().item()
                        if is_main_process() and config.wandb.enable:
                            wandb.log({
                                "train/similarity_score": sim_mean,
                                "train/predicted_confidence": conf_mean,
                            }, step=global_step)
                            logger.info(
                                f"similarity_score={sim_mean:.4f}, "
                                f"predicted_confidence={conf_mean:.4f} @ step {global_step}"
                            )

                    L_skip = -(torch.log(torch.sigmoid(conf_logit) + 1e-8)).mean()

                    loss = lambda_calib * L_calib + lambda_skip * L_skip

                if hasattr(config, "deepspeed") and config.deepspeed.enable:
                    model.backward(loss)
                    model.step()
                else:  # standard AMP path
                    optimizer.zero_grad()
                    if config.use_half_precision:
                        scaler.scale(loss).backward()
                        if config.optimizer.max_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), config.optimizer.max_grad_norm
                            )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if config.optimizer.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), config.optimizer.max_grad_norm
                            )
                        optimizer.step()
                    scheduler.step()

                h = new_h # update hidden for BPTT

                loss_pred_acc += L_pred.item()
                loss_calib_acc += L_calib.item()
                loss_skip_acc += L_skip.item()

        step_count = max(num_steps, 1)
        metric_logger.update(L_pred=loss_pred_acc / step_count)
        metric_logger.update(L_calib=loss_calib_acc / step_count)
        metric_logger.update(L_skip=loss_skip_acc / step_count)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        diff_lr = next(
            (pg["lr"] for pg in optimizer.param_groups if pg.get("different_lr", False)),
            optimizer.param_groups[0]["lr"],
        )
        metric_logger.update(different_lr=diff_lr)

        global_step += step_count
        if i % config.log_freq == 0:
            progress_bar.set_postfix(
                L_pred=f"{metric_logger.meters['L_pred'].value:.4f}",
                L_calib=f"{metric_logger.meters['L_calib'].value:.4f}",
                L_skip=f"{metric_logger.meters['L_skip'].value:.4f}",
            )
            if is_main_process():
                logger.info(f"Training: [Epoch {epoch}] [Step {i}] {metric_logger}")
            if is_main_process() and config.wandb.enable:
                log_dict_to_wandb(
                    metric_logger.get_value_dict(),
                    step=global_step,
                    prefix="train/",
                )

        # Save checkpoint
        global_iter = global_step // step_count

        # Run test inference on test video every 100 iterations
        if is_main_process() and test_frames is not None and global_iter % 100 == 0 and global_iter > 0:
            run_test_inference(model_without_ddp, test_frames, device, global_iter)

        if is_main_process() and config.get("save_iter", False) and global_iter % config.save_iter == 0:
            state_dict = model_without_ddp.state_dict()

            save_obj = {
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "config": config,
                "epoch": epoch,
                "global_step": global_iter,
            }
            torch.save(save_obj, join(config.output_dir, f"ckpt_step_{global_iter}.pt"))

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats for Epoch [{epoch}]: {metric_logger.global_avg()}")
    if is_main_process() and config.wandb.enable:
        log_dict_to_wandb(
            metric_logger.get_value_dict(),
            step=global_step,
            prefix=f"epoch_{epoch}/",
        )

    return global_step

def run_test_inference(model, frames, device, step):
    """Run test inference on a video using the current model state"""
    model.eval()
    logger.info(f"Running test inference at step {step}")

    # Import locally to avoid circular imports
    from demo.utils import frames2tensor

    # Use a fixed number of frames to test
    num_frames = min(100, len(frames))

    # First warm up the model with 8 frames
    hidden = model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)

    for i in range(8):
        frame_tensor = frames2tensor([frames[i]], fnum=1, target_size=(224, 224), device=device)
        with torch.no_grad():
            _, hidden, _ = model.streaming_vision_encoder(
                frame_tensor.permute(0, 2, 1, 3, 4),
                prev_hidden_state=hidden,
                confidence_threshold=1.0,  # Force no skip during warm-up
                max_consecutive_skips=0
            )

    logger.info("Warm-up complete, running inference with SPFS")

    # Track skipped frames and confidence
    total_skipped = 0
    confidence_values = []
    similarity_values = []

    # Process remaining frames with SPFS enabled
    for i in range(8, num_frames - 1):  # -1 to leave room for next frame
        frame_tensor = frames2tensor([frames[i]], fnum=1, target_size=(224, 224), device=device)
        next_frame_tensor = frames2tensor([frames[i+1]], fnum=1, target_size=(224, 224), device=device)

        with torch.no_grad():
            # Get feature for next frame (for gt comparison)
            next_feat, _ = model.streaming_vision_encoder.vit_lite.extract_features(next_frame_tensor.squeeze(1))

            # Process current frame
            _, hidden, spfs_info = model.streaming_vision_encoder(
                frame_tensor.permute(0, 2, 1, 3, 4),
                prev_hidden_state=hidden,
                confidence_threshold=0.5,  # Lower threshold to see if skipping works
                max_consecutive_skips=6,
                teacher_frame_feature=next_feat
            )

            # Track metrics
            if spfs_info.skipped:
                total_skipped += 1

            confidence_values.append(spfs_info.confidence)
            similarity_values.append(spfs_info.gt_cos)

            # Log every 10 frames
            if i % 10 == 8:  # Start at frame 8, then every 10th
                logger.info(f"Frame {i}: skipped={spfs_info.skipped}, "
                           f"confidence={spfs_info.confidence:.4f}, "
                           f"similarity={spfs_info.gt_cos:.4f}")

    # Log summary statistics
    if confidence_values:
        avg_conf = sum(confidence_values) / len(confidence_values)
        avg_sim = sum(similarity_values) / len(similarity_values)
        logger.info(f"Test inference summary: "
                   f"frames={num_frames-8}, "
                   f"skipped={total_skipped}, "
                   f"skip_rate={total_skipped/(num_frames-8):.2f}, "
                   f"avg_conf={avg_conf:.4f}, "
                   f"avg_sim={avg_sim:.4f}")

    model.train()

def clone_collate_fn(batch):
    def clone_item(x):
        if isinstance(x, torch.Tensor):
            return x.clone()
        elif isinstance(x, (list, tuple)):
            return type(x)(clone_item(y) for y in x)
        elif isinstance(x, dict):
            return {k: clone_item(v) for k, v in x.items()}
        else:
            return x

    batch = [clone_item(sample) for sample in batch]
    return default_collate(batch)

def setup_dataloaders(config, mode="pt"):
    logger.info(f"Creating dataset for {mode}")
    train_datasets = create_dataset(f"{mode}_train", config)
    train_datasets = add_precomputed_embeddings(train_datasets, getattr(config, "teacher_embedding_dir", None))
    media_types = get_media_types(train_datasets)

    if not config.distributed:
        raise NotImplementedError("Non-distributed training path might need adjustments for samplers.")

    # one GPU-batch size per media type
    batch_size = [config.inputs.batch_size[k] for k in media_types]
    samplers   = create_stateful_sampler(train_datasets, batch_size)

    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size   = batch_size,
        num_workers  = [config.num_workers] * len(media_types),
        is_trains    = [True] * len(media_types),
        collate_fns  = [clone_collate_fn] * len(media_types),
    )

    # =============================================================

    test_datasets, test_dataset_names = create_dataset(f"{mode}_eval", config)
    test_loaders = create_loader(
        test_datasets,
        [None] * len(test_datasets),
        batch_size   = [config.inputs.batch_size_test[d.media_type] for d in test_datasets],
        num_workers  = [config.num_workers] * len(test_datasets),
        is_trains    = [False] * len(test_datasets),
        collate_fns  = [None]   * len(test_datasets),
    )

    test_name2loaders = dict(zip(test_dataset_names, test_loaders))
    return train_loaders, test_name2loaders, media_types

def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    is_pretrain = config.mode == "pt"

    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, _, train_media_types = setup_dataloaders(
        config, mode=config.mode
    )
    num_steps_per_epoch = sum(len(d) for d in train_loaders) * 247 # Using each individual frame for training

    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    cudnn.benchmark = len(train_media_types) == 1

    if getattr(config, "resume", False) and not getattr(config, "pretrained_path", ""):
        ckpt_dir = os.path.join(config.output_dir, "ckpt_00.pth")
        if not os.path.exists(ckpt_dir):
            for fname in sorted(os.listdir(config.output_dir)):
                if fname.startswith("ckpt_") and fname.endswith(".pth"):
                    ckpt_dir = os.path.join(config.output_dir, fname)
                    break
        if os.path.exists(ckpt_dir):
            logger.info(f"Auto-resume checkpoint from {ckpt_dir}")
            config.pretrained_path = ckpt_dir
        else:
            logger.warning(
                f"Resume flag set but no checkpoint found in {config.output_dir}"
            )

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
        pretrain=is_pretrain,
        find_unused_parameters=True,
        num_steps_per_epoch=num_steps_per_epoch,
    )

    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    if config.get('use_bf16', True):
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    logger.info("Start training")
    logger.info(f"Epoch: {start_epoch}")
    start_time = time.time()

    for epoch in range(start_epoch, config.scheduler.epochs):
        if epoch == 0:
            model.freeze_mamba()
            model.freeze_confidence_head()
            model.unfreeze_prediction_head()
        else:
            model.freeze_prediction_head()
            model.unfreeze_confidence_head()

        if not config.evaluate:
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
                test_video_path=config.get("test_video_path", None),
            )

        # Save checkpoint before next epoch
        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            if config.get("save_latest", False):
                tag = "ckpt_latest.pth"
            else:
                tag = f"ckpt_{epoch:02d}.pth"
            model.save_checkpoint(config.output_dir, tag=tag, save_latest=False, exclude_frozen_parameters=True)

        elif is_main_process():
            state_dict = model_without_ddp.state_dict()
            param_grad_dict = {
                k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
            }
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
            if config.get("save_latest", False):
                torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
            else:
                torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()

    test_video_path = os.path.join(os.environ['DATASET_ROOT'], 'testing_spfs_video.mp4')

    # Add test video path to config
    if test_video_path:
        cfg.test_video_path = test_video_path
        logger.info(f"Will test SPFS on video: {cfg.test_video_path}")
    local_broadcast_process_authkey()
    main(cfg)
