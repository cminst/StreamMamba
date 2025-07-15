# Standard library imports
import datetime
import logging
import os
import pickle
import time
import random
import numpy as np

from os.path import join

# Early check for dataset path
if not os.environ.get("DATASET_ROOT"):
    raise RuntimeError(
        "DATASET_ROOT environment variable is not set. "
        "Please export DATASET_ROOT to point to your dataset root before running this script."
    )

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.nn import BCEWithLogitsLoss, MSELoss, CosineEmbeddingLoss
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

# Local application imports
from dataset import MetaLoader_rs, create_dataset, create_loader, create_stateful_sampler
from dataset.serialize import local_broadcast_process_authkey
from models import *
from tasks_clip.shared_utils import get_media_types, setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb
from utils.optimizer import (
    add_different_lr,
    create_optimizer_params_group,
    extend_optimizer_with_param_groups,
)

logger = logging.getLogger(__name__)


def get_rng_state():
    return {
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_state": torch.get_rng_state(),
        "cuda_state": torch.cuda.get_rng_state_all(),
    }


def set_rng_state(state):
    if not state:
        return
    try:
        if "random_state" in state and state["random_state"] is not None:
            random.setstate(state["random_state"])
        if "numpy_state" in state and state["numpy_state"] is not None:
            np.random.set_state(state["numpy_state"])
        if "torch_state" in state and state["torch_state"] is not None:
            torch.set_rng_state(state["torch_state"])
        if "cuda_state" in state and state["cuda_state"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_state"])
    except Exception as e:
        logger.warning(f"Failed to restore RNG state: {e}")


def unfreeze_mobileclip_vision(model, optimizer, scheduler, config):
    """Unfreeze MobileCLIP vision encoder and add its params to the optimizer."""
    logger.info("Unfreezing MobileCLIP vision encoder parameters")

    submodule = model.streaming_vision_encoder.vit_lite
    for p in submodule.parameters():
        p.requires_grad = True

    weight_decay = config.optimizer.weight_decay
    named_param_tuples = []
    for name, param in submodule.named_parameters():
        full_name = f"streaming_vision_encoder.vit_lite.{name}"
        if len(param.shape) == 1 or full_name.endswith(".bias"):
            wd = 0
        else:
            wd = weight_decay
        named_param_tuples.append([full_name, param, wd])

    if hasattr(config.optimizer, "different_lr") and config.optimizer.different_lr.enable:
        diff_names = config.optimizer.different_lr.module_names
        diff_lr = config.optimizer.different_lr.lr
    else:
        diff_names = []
        diff_lr = None

    named_param_tuples = add_different_lr(named_param_tuples, diff_names, diff_lr, config.optimizer.lr)
    param_groups = create_optimizer_params_group(named_param_tuples, config.optimizer.lr)
    extend_optimizer_with_param_groups(optimizer, scheduler, param_groups)

    # Mark as unfrozen so we don't unfreeze again
    config.model.freeze_mobileclip_vision = False

def save_debug_step_data(output_dir, global_step, frame_idx,
                         new_frame_input, # Input to streaming_vision_encoder
                         current_hidden_state_input, # Hidden state input to streaming_vision_encoder
                         actual_window_input, # Input to vision_encoder (full model)
                         stream_embedding_output, # Output of streaming pipeline
                         target_embedding_output, # Output of target pipeline
                         model_state_dict,
                         config=None): # Optional: save config for completeness
    """
    Saves all relevant tensors and model state for a single debug step.
    """
    step_dir = os.path.join(output_dir, f"debug_step_{global_step}_frame_{frame_idx}")
    os.makedirs(step_dir, exist_ok=True)

    # Save tensors
    torch.save(new_frame_input.cpu(), os.path.join(step_dir, "new_frame_input.pt"))
    torch.save(actual_window_input.cpu(), os.path.join(step_dir, "actual_window_input.pt"))
    torch.save(stream_embedding_output.cpu(), os.path.join(step_dir, "stream_embedding_output.pt"))
    torch.save(target_embedding_output.cpu(), os.path.join(step_dir, "target_embedding_output.pt"))

    if isinstance(current_hidden_state_input, tuple):
        cpu_hidden_state = tuple(h.cpu() for h in current_hidden_state_input)
    elif isinstance(current_hidden_state_input, torch.Tensor):
        cpu_hidden_state = current_hidden_state_input.cpu()
    else:
        cpu_hidden_state = current_hidden_state_input

    with open(os.path.join(step_dir, "current_hidden_state_input.pkl"), "wb") as f:
        pickle.dump(cpu_hidden_state, f)

    torch.save(model_state_dict, os.path.join(step_dir, "model_state_dict.pth"))

    if config:
        with open(os.path.join(step_dir, "config.pkl"), "wb") as f:
            pickle.dump(config, f)

    print(f"Saved debug data for global_step {global_step}, frame_idx {frame_idx} to {step_dir}")

# Helper to read frames from video
def _frame_from_video(video_cap):
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break
        yield frame # BGR numpy array

def get_inference_transform(img_size):
     return transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

# Preprocess single frame for model input
def preprocess_frame(frame_bgr_np, transform, device):
    """
    Preprocesses a single frame (BGR numpy array) for model inference.
    Output: [1, C, H, W] tensor on specified device, normalized [0, 1]
    """
    frame_rgb_np = cv2.cvtColor(frame_bgr_np, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb_np)

    # Apply the transform pipeline
    # Output shape: [C, H, W]
    transformed_tensor_chw = transform(pil_image)

    # Add batch dimension and move to device
    frame_tensor_batch = transformed_tensor_chw.unsqueeze(0).to(device) # [1, C, H, W]

    return frame_tensor_batch

# Model evaluation
def evaluate_streaming_similarity(
    model,
    device,
    streaming_transform, # The preprocessing transform
    video_path,
    model_max_frames,
    output_dir,
    global_step, # Current training step for filename
    config
):
    """
    Evaluates the cosine similarity between streaming and full window features
    for a specific video and saves a plot.

    Returns the average cosine similarity over the comparable frames.
    """

    regular_transform = transforms.Compose(
        [
            transforms.Resize(
                (model.config.model.vision_encoder.img_size, model.config.model.vision_encoder.img_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Ensure model is in evaluation mode and on the correct device
    model.eval()
    model.to(device) # Ensure model is on device, though it should be already

    cosine_similarities = []
    frame_indices_for_plot = []
    avg_similarity = -1.0 # Default value if no frames processed

    logger.info(f"Starting evaluation on video: {video_path}")

    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        logger.error(f"Error: Could not open video {video_path} for evaluation.")
        return avg_similarity # Return default if video can't be opened

    all_frames_raw = list(_frame_from_video(video_cap)) # List of numpy arrays (H, W, C, BGR)
    video_cap.release()

    if len(all_frames_raw) < model_max_frames:
        logger.warning(f"Evaluation video {video_path} has {len(all_frames_raw)} frames, less than MODEL_MAX_FRAMES ({model_max_frames}). Skipping evaluation.")
        return avg_similarity # Return default if video is too short

    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        curr_hidden_state_streaming = model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)

        logger.info(f"Warming up streaming model for evaluation with first {model_max_frames - 1} frames...")

        for i in range(model_max_frames - 1):
            frame_data = all_frames_raw[i]
            frame_tensor_batch = preprocess_frame(frame_data, streaming_transform, device)
            frame_tensor_streaming_input = frame_tensor_batch.unsqueeze(2)
            raw_stream_embedding_dummy, curr_hidden_state_streaming = model.streaming_vision_encoder(
                frame_tensor_streaming_input,
                curr_hidden_state_streaming
            )
        logger.info(f"Warm-up complete for evaluation.")

        logger.info(f"Processing and comparing from frame {model_max_frames - 1} onwards...")

        for frame_idx in range(model_max_frames - 1, len(all_frames_raw)):
            current_frame_data_streaming = all_frames_raw[frame_idx]

            frame_tensor_batch = preprocess_frame(current_frame_data_streaming, streaming_transform, device)

            frame_tensor_streaming_input = frame_tensor_batch.unsqueeze(2)

            raw_stream_embedding, new_hidden_state = model.streaming_vision_encoder(
                frame_tensor_streaming_input,
                curr_hidden_state_streaming
            )

            # Align and Normalize the raw streaming embedding
            if config.model.use_streaming_vision_align:
                aligned_stream_embedding = model.streaming_vision_align(raw_stream_embedding)
            else:
                aligned_stream_embedding = model.vision_align(raw_stream_embedding)
            stream_embedding = aligned_stream_embedding / (aligned_stream_embedding.norm(dim=-1, keepdim=True) + 1e-9)

            # Update the hidden state for the next frame
            curr_hidden_state_streaming = new_hidden_state

            # Full Model Feature for the corresponding window
            window_start_idx = frame_idx - model_max_frames + 1
            window_end_idx = frame_idx + 1 # Slicing is exclusive at the end
            current_window_frames_data = all_frames_raw[window_start_idx : window_end_idx] # List of BGR numpy arrays

            # Preprocess all frames in the window and stack them
            list_of_frame_tensors = [preprocess_frame(f, regular_transform, device) for f in current_window_frames_data]
            window_tensor_full = torch.stack(list_of_frame_tensors, dim=2) # Shape: [B=1, C, T, H, W]

            # Pass the full window tensor to the full vision encoder
            raw_target_embedding = model.vision_encoder(window_tensor_full)

            # Align and Normalize the raw target embedding
            aligned_target_embedding = model.vision_align(raw_target_embedding)
            target_embedding = aligned_target_embedding / (aligned_target_embedding.norm(dim=-1, keepdim=True) + 1e-9)

            # Cosine sim
            similarity = torch.nn.functional.cosine_similarity(stream_embedding, target_embedding, dim=1)

            sim_value = similarity.item()
            cosine_similarities.append(sim_value)
            frame_indices_for_plot.append(frame_idx)

        # Evaluation Complete
        if cosine_similarities:
            avg_similarity = sum(cosine_similarities) / len(cosine_similarities)
            logger.info(f"Evaluation complete. Average Cosine Similarity: {avg_similarity:.4f}")

            # Plot and save
            plt.figure(figsize=(12, 6))
            plt.plot(frame_indices_for_plot, cosine_similarities, 'g-', label='Cosine Similarity (Streaming vs Full Window)')
            plt.xlabel(f'Frame Number (Window of {model_max_frames} frames ending at this frame)')
            plt.ylabel('Cosine Similarity')
            plt.title(f'Feature Similarity Over Time - Video: {os.path.basename(video_path)}\nTraining Step: {global_step}')
            plt.legend()
            plt.grid(True)
            plt.ylim(-0.1, 1.1) # Cosine similarity range
            plt.axhline(y=avg_similarity, color='b', linestyle='--', label=f'Average: {avg_similarity:.4f}')
            plt.legend()

            graph_save_dir = join(output_dir, 'cosine_sim_graphs')
            os.makedirs(graph_save_dir, exist_ok=True)
            graph_filename = f'graph_step_{global_step:07d}.png' # Use padded step number
            graph_save_path = join(graph_save_dir, graph_filename)

            plt.savefig(graph_save_path)
            logger.info(f"Saved evaluation plot to {graph_save_path}")

            plt.close('all')
        else:
            logger.warning("No cosine similarities were calculated during evaluation.")


    # Set model back to training mode
    model.train()
    logger.info("Evaluation complete. Model set back to train() mode.")

    return avg_similarity


# Main training function
# Main training function

def train(
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
    skip_num=0,
    log_debug=False,
):
    """Simplified SPFS training loop following pretrain scheduler logic."""

    model_without_ddp = model.module if config.distributed else model
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    metric_logger.add_meter("different_lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    metric_logger.add_meter("L_primary", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("L_pred", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("L_calib", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("L_skip", SmoothedValue(window=1, fmt="{value:.4f}"))

    media_types = get_media_types(train_loaders)
    if config.distributed:
        for loader in train_loaders:
            loader.sampler.set_epoch(epoch)

    seed = config.seed + epoch
    train_loader_agg = MetaLoader_rs(
        name2loader=dict(list(zip(media_types, train_loaders))),
        skip_num=skip_num,
        seed=seed,
    )

    progress_bar = tqdm(
        train_loader_agg,
        total=len(train_loader_agg),
        desc=f"Training: [Epoch {epoch}]",
        disable=not is_main_process(),
    )

    cosine_loss_fn = lambda pred, target: 1 - torch.nn.functional.cosine_similarity(pred, target.detach(), dim=-1).mean()
    bce_loss_fn = BCEWithLogitsLoss()
    primary_loss_fn = MSELoss()

    lambda_calib = getattr(config, "lambda_calib", 1.0)
    lambda_skip = getattr(config, "lambda_skip", 0.1)
    MODEL_MAX_FRAMES = config.num_frames

    for i, data_pair in enumerate(progress_bar):
        media_type, (image, text, idx) = data_pair
        image = image.to(device, non_blocking=True)
        image = image.permute(0, 2, 1, 3, 4)
        B, C, T, H, W = image.shape
        assert T >= MODEL_MAX_FRAMES

        h = model.streaming_vision_encoder.init_hidden(batch_size=B, device=device)
        with torch.no_grad():
            for t in range(MODEL_MAX_FRAMES - 1):
                frame = image[:, :, t, :, :].unsqueeze(2)
                _, h = model.streaming_vision_encoder(frame, h)

        num_steps = T - (MODEL_MAX_FRAMES - 1)

        loss_primary = loss_pred = loss_calib = loss_skip = 0.0

        for step in range(num_steps):
            idx_curr = (MODEL_MAX_FRAMES - 1) + step
            frame_curr = image[:, :, idx_curr, :, :].unsqueeze(2)
            out_t, new_h = model.streaming_vision_encoder(frame_curr, h)

            window_start = idx_curr - MODEL_MAX_FRAMES + 1
            window_end = idx_curr + 1
            curr_window = image[:, :, window_start:window_end, :, :]
            with torch.no_grad():
                target_curr = model_without_ddp.vision_encoder(curr_window)
                target_curr = model_without_ddp.vision_align(target_curr)

            next_window_start = window_start + 1
            next_window_end = window_end + 1
            next_window = image[:, :, next_window_start:next_window_end, :, :]
            with torch.no_grad():
                target_next = model_without_ddp.vision_encoder(next_window)
                target_next = model_without_ddp.vision_align(target_next)

            mu_t, logvar = model.streaming_vision_encoder.rnn.predict_next_feat(new_h)
            conf_logit = -logvar.squeeze(-1)

            L_primary = primary_loss_fn(out_t, target_curr)
            L_pred = cosine_loss_fn(mu_t, target_next)
            with torch.no_grad():
                pred_quality = torch.nn.functional.cosine_similarity(mu_t, target_next, dim=-1)
                target_c = (pred_quality > 0.98).float()
            L_calib = bce_loss_fn(conf_logit.squeeze(), target_c)
            L_skip = -(torch.log(torch.sigmoid(conf_logit) + 1e-8)).mean()

            loss = L_primary + L_pred + lambda_calib * L_calib + lambda_skip * L_skip

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

            h = tuple(hh.detach().clone() for hh in new_h)
            loss_primary += L_primary.item()
            loss_pred += L_pred.item()
            loss_calib += L_calib.item()
            loss_skip += L_skip.item()

        step_count = max(num_steps, 1)
        metric_logger.update(L_primary=loss_primary / step_count)
        metric_logger.update(L_pred=loss_pred / step_count)
        metric_logger.update(L_calib=loss_calib / step_count)
        metric_logger.update(L_skip=loss_skip / step_count)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        diff_lr = None
        for pg in optimizer.param_groups:
            if pg.get("different_lr", False):
                diff_lr = pg["lr"]
                break
        metric_logger.update(different_lr=diff_lr if diff_lr is not None else optimizer.param_groups[0]["lr"])

        global_step += 1
        if i % config.log_freq == 0:
            progress_bar.set_postfix({
                "L_primary": metric_logger.meters["L_primary"].avg,
                "L_pred": metric_logger.meters["L_pred"].avg,
                "L_calib": metric_logger.meters["L_calib"].avg,
                "L_skip": metric_logger.meters["L_skip"].avg,
            })
            if is_main_process():
                logger.info(f"Training: [Epoch {epoch}] [Step {i}] {metric_logger}")
            if is_main_process() and config.wandb.enable:
                log_dict_to_wandb(metric_logger.get_global_avg_dict(), step=global_step, prefix="train/")

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats for Epoch [{epoch}]: {metric_logger.global_avg()}")
    if is_main_process() and config.wandb.enable:
        log_dict_to_wandb(metric_logger.get_global_avg_dict(), step=global_step, prefix=f"epoch_{epoch}/")

    return global_step


def clone_collate_fn(batch):
    # Recursively clone every Tensor in the sample so its storage is fresh
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
    media_types = get_media_types(train_datasets)

    if not config.distributed:
        raise NotImplementedError("Non-distributed training path might need adjustments for samplers.")

    # one GPU-batch size per media type
    batch_size = [config.inputs.batch_size[k] for k in media_types]
    samplers   = create_stateful_sampler(train_datasets, batch_size)

    train_loaders = create_loader(
        train_datasets,
        samplers, # Use samplers specific to train_datasets
        batch_size   = batch_size,
        num_workers  = [config.num_workers] * len(media_types),
        is_trains    = [True] * len(media_types),
        collate_fns  = [clone_collate_fn] * len(media_types),
    )

    # =============================================================

    # eval side stays the same
    test_datasets, test_dataset_names = create_dataset(f"{mode}_eval", config)
    test_loaders = create_loader(
        test_datasets,
        [None] * len(test_datasets), # Eval loaders typically don't need samplers in DDP if evaluating on all data
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

    train_loaders, test_name2loaders, train_media_types = setup_dataloaders(
        config, mode=config.mode
    )
    num_steps_per_epoch = sum(len(d) for d in train_loaders) * 247 # Using each individual frame for training

    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    cudnn.benchmark = len(train_media_types) == 1

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

    # restore RNG state from checkpoint if available
    if config.resume and os.path.isfile(config.pretrained_path):
        try:
            ckpt = torch.load(config.pretrained_path, map_location="cpu")
            set_rng_state(ckpt.get("rng_state"))
        except Exception as e:
            logger.warning(f"Failed to load RNG state from checkpoint: {e}")
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    if config.get('use_bf16', True):
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    logger.info("Start training")
    logger.info(f"Epoch: {start_epoch}")
    start_time = time.time()
    start_step = start_epoch * num_steps_per_epoch

    for epoch in range(start_epoch, config.scheduler.epochs):
        if epoch == 0:
            model.freeze_mamba()
            model.freeze_confidence_head()
            model.unfreeze_prediction_head()
        else:
            model.unfreeze_mamba()
            model.unfreeze_confidence_head()
            model.unfreeze_prediction_head()

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
                skip_num = global_step - start_step,
                log_debug = True
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
                "rng_state": get_rng_state(),
            }
            if config.get("save_latest", False):
                torch.save(save_obj, join(config.output_dir, "ckpt_latest.pth"))
            else:
                torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))

        # -- End of Epoch --
        start_step = global_step
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
