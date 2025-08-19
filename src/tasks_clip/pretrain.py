# Standard library imports
import datetime
import logging
import os
import pickle
import time
import random
import numpy as np

from os.path import join

if not os.environ.get("DATASET_ROOT"):
    raise RuntimeError(
        "DATASET_ROOT environment variable is not set. "
        "Please export DATASET_ROOT to point to your dataset root before running this script."
    )

import cv2
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from huggingface_hub import hf_hub_download
from PIL import Image
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
import torch.nn.functional as F
from dataset.serialize import local_broadcast_process_authkey
from models import *
from tasks_clip.shared_utils import get_media_types, setup_model
from utils.basic_utils import (
    MetricLogger,
    SmoothedValue,
    setup_seed,
    info_nce_loss,
    cosine_sim_loss
)
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb
from utils.optimizer import (
    add_different_lr,
    create_optimizer_params_group,
    extend_optimizer_with_param_groups,
)
from easydict import EasyDict as edict

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

def _frame_from_video(video_cap):
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break
        yield frame

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

    transformed_tensor_chw = transform(pil_image)

    frame_tensor_batch = transformed_tensor_chw.unsqueeze(0).to(device) # [1, C, H, W]

    return frame_tensor_batch

# Model evaluation
def evaluate_streaming_similarity(
    model,
    device,
    streaming_transform,
    video_path,
    model_max_frames,
    output_dir,
    global_step,
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

    model.eval()
    model.to(device)

    cosine_similarities = []
    frame_indices_for_plot = []
    avg_similarity = -1.0

    logger.info(f"Starting evaluation on video: {video_path}")

    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        logger.error(f"Error: Could not open video {video_path} for evaluation.")
        return avg_similarity

    all_frames_raw = list(_frame_from_video(video_cap))
    video_cap.release()

    if len(all_frames_raw) < model_max_frames:
        logger.warning(f"Evaluation video {video_path} has {len(all_frames_raw)} frames, less than MODEL_MAX_FRAMES ({model_max_frames}). Skipping evaluation.")
        return avg_similarity

    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        curr_hidden_state_streaming = model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)

        logger.info(f"Warming up streaming model for evaluation with first {model_max_frames - 1} frames...")

        for i in range(model_max_frames - 1):
            frame_data = all_frames_raw[i]
            frame_tensor_batch = preprocess_frame(frame_data, streaming_transform, device)
            frame_tensor_streaming_input = frame_tensor_batch.unsqueeze(2)
            _, curr_hidden_state_streaming, _ = model.streaming_vision_encoder(
                frame_tensor_streaming_input,
                curr_hidden_state_streaming
            )

        logger.info("Warm-up complete for evaluation.")

        logger.info(f"Processing and comparing from frame {model_max_frames - 1} onwards...")

        for frame_idx in range(model_max_frames - 1, len(all_frames_raw)):
            current_frame_data_streaming = all_frames_raw[frame_idx]

            frame_tensor_batch = preprocess_frame(current_frame_data_streaming, streaming_transform, device)

            frame_tensor_streaming_input = frame_tensor_batch.unsqueeze(2)

            raw_stream_embedding, new_hidden_state, _ = model.streaming_vision_encoder(
                frame_tensor_streaming_input,
                curr_hidden_state_streaming
            )

            if config.model.use_streaming_vision_align:
                aligned_stream_embedding = model.streaming_vision_align(raw_stream_embedding)
            else:
                aligned_stream_embedding = model.vision_align(raw_stream_embedding)
            stream_embedding = aligned_stream_embedding / (aligned_stream_embedding.norm(dim=-1, keepdim=True) + 1e-9)

            curr_hidden_state_streaming = new_hidden_state

            window_start_idx = frame_idx - model_max_frames + 1
            window_end_idx = frame_idx + 1
            current_window_frames_data = all_frames_raw[window_start_idx : window_end_idx] # List of BGR numpy arrays

            list_of_frame_tensors = [preprocess_frame(f, regular_transform, device) for f in current_window_frames_data]
            window_tensor_full = torch.stack(list_of_frame_tensors, dim=2) # Shape: [B=1, C, T, H, W]

            raw_target_embedding = model.vision_encoder(window_tensor_full)

            aligned_target_embedding = model.vision_align(raw_target_embedding)
            target_embedding = aligned_target_embedding / (aligned_target_embedding.norm(dim=-1, keepdim=True) + 1e-9)

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
def train(
    model,
    train_loaders,
    optimizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
    data_type,
    skip_num=0
):
    """
    Performs one epoch of training with periodic evaluation.
    """
    model_without_ddp = model.module if config.distributed else model

    model.train()

    mobileclip_transform = transforms.Compose(
        [
            transforms.Resize(
                config.inputs.image_res,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(config.inputs.image_res),
            transforms.ToTensor(),
        ]
    )

    EVAL_FREQ_STEPS = config.eval_freq_steps
    logger.info(f"Getting evaluation video from {config.eval_video_repo_id} ({config.eval_video_filename})")
    EVAL_VIDEO_PATH = hf_hub_download(repo_id=config.eval_video_repo_id, filename=config.eval_video_filename, repo_type="dataset")
    EVAL_PLOT_OUTPUT_DIR = config.eval_plot_output_dir
    os.makedirs(EVAL_PLOT_OUTPUT_DIR, exist_ok=True)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    metric_logger.add_meter("different_lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("video_stream_target_loss", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("video_stream_target_sim", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("video_stream_nce_loss", SmoothedValue(window=1, fmt="{value:.4f}"))
    metric_logger.add_meter("eval_avg_sim", SmoothedValue(window=1, fmt="{value:.4f}"))

    header = f"Training: [Epoch {epoch}]"
    log_freq = config.log_freq

    media_types = get_media_types(train_loaders) # Defined here for use in MetaLoader_rs

    if config.distributed:
        for loader in train_loaders:
            loader.sampler.set_epoch(epoch)

    # Aggregate loaders
    seed = config.seed + epoch
    train_loader_agg = MetaLoader_rs(
        name2loader=dict(list(zip(media_types, train_loaders))),
        skip_num=skip_num,
        seed=seed,
    )

    num_batches_train = len(train_loader_agg)
    logger.info(f"Training loader set up, {num_batches_train} batches.")
    enable_mobileclip_ft = config.get("enable_mobileclip_ft", False)
    unfreeze_step = 0

    if enable_mobileclip_ft:
        unfreeze_step = int(num_batches_train * config.unfreeze_mobileclip_pct)

    if config.model.use_streaming_vision_align:
        vision_align = model_without_ddp.streaming_vision_align
    else:
        vision_align = model_without_ddp.vision_align

    progress_bar = tqdm(
        train_loader_agg,
        total=num_batches_train,
        desc=header,
        disable=not is_main_process()
    )

    MODEL_MAX_FRAMES = config.num_frames

    for i, data_pair in enumerate(progress_bar):
        if (
            enable_mobileclip_ft
            and epoch == 0
            and i >= unfreeze_step
            and config.model.freeze_mobileclip_vision
        ):
            unfreeze_mobileclip_vision(model_without_ddp, optimizer, scheduler, config)

        batch = data_pair[1]
        if len(batch) == 4:
            image, text, _, precomputed_emb = batch
        else:
            image, text, _ = batch
            precomputed_emb = None

        image = image.to(device, non_blocking=True)

        nce_lambda = 0.0

        if config.get('enable_contrastive_distillation', False):
            nce_start = int(num_batches_train * config.contrastive_warmup_pct)
            if i >= nce_start:
                ramp = min((i - nce_start) / float(config.contrastive_ramp_iters), 1.0)
                nce_lambda = config.contrastive_lambda * ramp

        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            image = image.permute(0, 2, 1, 3, 4)
            B, C, T, H, W = image.shape

            assert T >= MODEL_MAX_FRAMES, f"Video has {T} frames, needs {MODEL_MAX_FRAMES}."

            curr_hidden_state = model.streaming_vision_encoder.init_hidden(batch_size=B, device=device)
            with torch.no_grad(): # Warm-up phase does not require gradients
                for frame_idx in range(MODEL_MAX_FRAMES - 1):
                    initial_frame_mc = image[:, :, frame_idx, :, :].unsqueeze(2)
                    _, curr_hidden_state, _ = model.streaming_vision_encoder(initial_frame_mc, curr_hidden_state)

            num_sliding_windows = T - (MODEL_MAX_FRAMES - 1)

            assert num_sliding_windows >= 1, "Number of sliding windows must be at least 1."

            # Initialize accumulators for the entire batch item
            total_loss_for_item = 0.0
            batch_total_cosine_loss_item = 0.0
            batch_total_nce_loss_item = 0.0
            batch_total_sim_item = 0.0

            for window_idx in range(num_sliding_windows):
                frame_idx = (MODEL_MAX_FRAMES - 1) + window_idx
                curr_frame = image[:, :, frame_idx, :, :].unsqueeze(2)

                # Get stream embedding
                stream_embedding, new_hidden_state, _ = model.streaming_vision_encoder(curr_frame, curr_hidden_state)
                stream_embedding = vision_align(stream_embedding)
                stream_embedding = stream_embedding / (stream_embedding.norm(dim=-1, keepdim=True) + 1e-9)

                # Get target embedding
                window_start = frame_idx - MODEL_MAX_FRAMES + 1
                window_end = frame_idx + 1
                current_window_frames_orig = image[:, :, window_start:window_end, :, :]

                with torch.no_grad(): # Target is always detached
                    if precomputed_emb is not None:
                        target_embedding = precomputed_emb[:, window_idx, :].to(device)
                    else:
                        target_embedding = model_without_ddp.vision_encoder(current_window_frames_orig)
                        target_embedding = model_without_ddp.vision_align(target_embedding)
                        target_embedding = target_embedding / (target_embedding.norm(dim=-1, keepdim=True) + 1e-9)

                # Calculate loss for this step
                cosine_loss_val = cosine_sim_loss(stream_embedding, target_embedding)
                info_nce_val = info_nce_loss(
                    stream_embedding, stream_embedding.detach(), text,
                    temperature=config.get('contrastive_temperature', 0.07)
                ) if nce_lambda > 0 else torch.tensor(0.0, device=device)

                step_loss = cosine_loss_val + nce_lambda * info_nce_val

                # Accumulate the loss tensor
                total_loss_for_item += step_loss

                # Accumulate for logging
                with torch.no_grad():
                    current_sim = F.cosine_similarity(stream_embedding, target_embedding, dim=1).mean()
                    batch_total_cosine_loss_item += cosine_loss_val.item()
                    batch_total_nce_loss_item += info_nce_val.item()
                    batch_total_sim_item += current_sim.item()

                # Update state without detaching
                curr_hidden_state = new_hidden_state

            # Average the loss over the number of steps
            final_loss = total_loss_for_item / num_sliding_windows

        # Single backward pass and optimizer step for the whole sequence
        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            model.backward(final_loss)
            model.step()
        else:
            optimizer.zero_grad()
            if config.use_half_precision:
                scaler.scale(final_loss).backward()
                if config.optimizer.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                final_loss.backward()
                if config.optimizer.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                optimizer.step()

            scheduler.step()

        # Averages for logging
        final_batch_cosine_loss = batch_total_cosine_loss_item / num_sliding_windows
        final_batch_nce_loss = batch_total_nce_loss_item / num_sliding_windows
        average_cosine_sim = batch_total_sim_item / num_sliding_windows

        learning_rate = optimizer.param_groups[0]["lr"]

        # Search for different LR values in the parameter groups
        different_lr_value = next((pg["lr"] for pg in optimizer.param_groups if pg.get("different_lr", False)), learning_rate)

        temperature = model_without_ddp.temp.item() if hasattr(model_without_ddp, 'temp') else None

        log_update_dict = edict(
            video_stream_target_loss=final_batch_cosine_loss,
            video_stream_nce_loss=final_batch_nce_loss,
            video_stream_target_sim=average_cosine_sim,
            lr=learning_rate,
            different_lr=different_lr_value,
            temperature=temperature,
        )

        metric_logger.update(**log_update_dict)

        global_step += 1
        if global_step % EVAL_FREQ_STEPS == 0 and is_main_process():
            logger.info(f"Performing evaluation at global step {global_step}...")
            avg_sim = evaluate_streaming_similarity(
                model=model_without_ddp, device=device, streaming_transform=mobileclip_transform,
                video_path=EVAL_VIDEO_PATH, model_max_frames=MODEL_MAX_FRAMES, output_dir=config.output_dir,
                global_step=global_step, config = config
            )
            metric_logger.update(eval_avg_sim=avg_sim)
            logger.info(f"Evaluation at step {global_step} complete. Avg Sim: {avg_sim:.4f}")
            model.train()

        # Log to console and W&B
        if i % log_freq == 0:
            if global_step > 0 and global_step % EVAL_FREQ_STEPS == 0 and is_main_process():
                if 'eval_avg_sim' in metric_logger.meters and hasattr(metric_logger.meters['eval_avg_sim'], 'value'): # Check if eval was run
                    log_update_dict["eval_sim"] = metric_logger.meters['eval_avg_sim'].value

            progress_bar.set_postfix(**log_update_dict)

            if is_main_process():
                logger.info(f"{header} [Step {i}] {metric_logger}")

            if is_main_process() and config.wandb.enable:
                averaged_logs_for_wandb = metric_logger.get_value_dict()
                log_dict_to_wandb(averaged_logs_for_wandb, step=global_step, prefix="train/")


        if config.get('save_iter', 0) > 0 and global_step % config.save_iter == 0:
            if is_main_process() and not config.deepspeed.enable:
                logger.info(f"Saving checkpoint at global step {global_step}")
                save_obj = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if config.use_half_precision else None,
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                    "rng_state": get_rng_state(),
                }
                checkpoint_filename = join(config.output_dir, f"ckpt_iter{global_step:07d}.pth")
                torch.save(save_obj, checkpoint_filename)
                logger.info(f"Saved iteration checkpoint to {checkpoint_filename}")
            else:
                logger.info(f"Saving checkpoint at global step {global_step}")
                tag = f"ckpt_iter{global_step:07d}"
                client_state = {"epoch": epoch, "global_step": global_step}

                model.save_checkpoint(config.output_dir, tag=tag, client_state=client_state)

                logger.info(f"Saved iteration checkpoint to {config.output_dir}")

        # Debugging
        if config.debug and global_step >= 20:
            logger.info("Debug mode: breaking training loop early.")
            break

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats for Epoch [{epoch}]: {metric_logger.global_avg()}")
    if is_main_process() and config.wandb.enable:
        log_dict_to_wandb(metric_logger.get_value_dict(), step=global_step, prefix=f"epoch_{epoch}/")

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
    train_datasets = add_precomputed_embeddings(train_datasets, config.get("teacher_embedding_dir", None))
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
        if not config.evaluate:
            global_step = train(
                model,
                train_loaders,
                optimizer,
                epoch,
                global_step,
                device,
                scheduler,
                scaler,
                config,
                data_type,
                skip_num = global_step - start_step
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
