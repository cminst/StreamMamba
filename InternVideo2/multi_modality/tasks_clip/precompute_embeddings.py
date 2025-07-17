import datetime
import logging
import os
import time
import copy
import numpy as np
import concurrent.futures
from os.path import join

if not os.environ.get("DATASET_ROOT"):
    raise RuntimeError(
        "DATASET_ROOT environment variable is not set. "
        "Please export DATASET_ROOT to point to your dataset root before running this script."
    )

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from tqdm import tqdm

from dataset import MetaLoader_rs, create_dataset, create_loader, create_stateful_sampler
from dataset.serialize import local_broadcast_process_authkey
from models import *
from tasks_clip.shared_utils import get_media_types, setup_model
from utils.basic_utils import setup_seed
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process, get_world_size

logger = logging.getLogger(__name__)

def compute_embeddings_worker(gpu_id, model_replica, image_batch, step_indices, config):
    """
    Computes embeddings for a given set of step indices on a specific GPU.
    """
    device = f"cuda:{gpu_id}"
    data_type = torch.bfloat16 if config.use_bf16 else torch.float16
    MODEL_MAX_FRAMES = config.num_frames

    model_replica.to(device)
    image_on_gpu = image_batch.to(device, non_blocking=True)

    results = {}
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=config.use_half_precision, dtype=data_type):
        for step in step_indices:
            idx_curr = (MODEL_MAX_FRAMES - 1) + step
            window_start = idx_curr - MODEL_MAX_FRAMES + 1
            window_end = idx_curr + 1

            curr_window = image_on_gpu[:, :, window_start:window_end, :, :]

            embedding = model_replica.vision_align(
                model_replica.vision_encoder(curr_window)
            )

            results[step] = embedding.cpu()

    return results

def precompute(model, train_loaders, config):
    """
    Pre-computes embeddings using a sliding window approach, parallelized across multiple GPUs.
    Only the main process (rank 0) performs the computation and saves files.
    """
    model_without_ddp = model.module if config.distributed else model

    vision_encoder_template = model_without_ddp.vision_encoder
    vision_align_template = model_without_ddp.vision_align

    vision_encoder_template.eval()
    vision_align_template.eval()

    num_gpus = config.get("num_gpus_for_computation", 1)
    if is_main_process():
        logger.info(f"Creating {num_gpus} model replicas for parallel computation.")
        model_replicas = [
            torch.nn.ModuleDict({
                'vision_encoder': copy.deepcopy(vision_encoder_template),
                'vision_align': copy.deepcopy(vision_align_template)
            }) for _ in range(num_gpus)
        ]
    else:
        model_replicas = []

    media_types = get_media_types(train_loaders)
    if config.distributed:
        for loader in train_loaders:
            loader.sampler.set_epoch(0)

    train_loader_agg = MetaLoader_rs(
        name2loader=dict(zip(media_types, train_loaders)),
        skip_num=0,
        seed=config.seed,
    )

    total_batches = len(train_loader_agg)
    padding_len = len(str(total_batches))

    progress_bar = tqdm(
        train_loader_agg,
        total=total_batches,
        desc="Pre-computing Embeddings",
        disable=not is_main_process(),
    )

    for i, data_pair in enumerate(progress_bar):
        if not is_main_process():
            continue

        _, (image, _, _) = data_pair

        # B, T, C, H, W -> permute to B, C, T, H, W
        image = image.permute(0, 2, 1, 3, 4)

        B, C, T, H, W = image.shape
        MODEL_MAX_FRAMES = config.num_frames

        if T < MODEL_MAX_FRAMES:
            logger.warning(f"Skipping batch {i} due to short video length ({T} < {MODEL_MAX_FRAMES})")
            continue

        num_steps = T - (MODEL_MAX_FRAMES - 1)
        step_indices = np.arange(num_steps)

        # Split the steps among the available computation GPUs
        step_chunks = np.array_split(step_indices, num_gpus)

        all_embeddings = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            future_to_gpu = {
                executor.submit(compute_embeddings_worker, gpu_id, model_replicas[gpu_id], image, chunks, config): gpu_id
                for gpu_id, chunks in enumerate(step_chunks) if len(chunks) > 0
            }

            for future in concurrent.futures.as_completed(future_to_gpu):
                try:
                    # Collect results from completed threads
                    result_dict = future.result()
                    all_embeddings.update(result_dict)
                except Exception as exc:
                    gpu_id = future_to_gpu[future]
                    logger.error(f'GPU {gpu_id} generated an exception: {exc}')
                    raise exc

        sorted_embeddings = [all_embeddings[s] for s in sorted(all_embeddings.keys())]

        if not sorted_embeddings:
            logger.warning(f"Batch {i} produced no embeddings. Skipping save.")
            continue

        final_tensor = torch.stack(sorted_embeddings, dim=1)

        output_path = join(config.output_dir, f"embed_b{str(i).zfill(padding_len)}.pt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(final_tensor, output_path)

        if i % config.log_freq == 0:
            progress_bar.set_postfix(
                last_saved=f"embed_b{str(i).zfill(padding_len)}.pt",
                shape=f"{list(final_tensor.shape)}"
            )

    dist.barrier()


def main(config):
    setup_seed(config.seed + get_rank())
    cudnn.benchmark = True

    if is_main_process():
        os.makedirs(config.output_dir, exist_ok=True)

    logger.info("Setting up dataloaders...")
    train_loaders, _, _ = setup_dataloaders(config, mode=config.mode)

    logger.info("Setting up model to extract encoder weights...")
    model_cls = eval(config.model.get('model_cls', 'InternVideo2_CLIP'))
    (
        model,
        model_without_ddp,
        _, _, _, _, _, _,
    ) = setup_model(
        config,
        model_cls=model_cls,
        pretrain=True,
        find_unused_parameters=False,
    )

    logger.info("Starting embedding pre-computation...")
    start_time = time.time()

    precompute(model, train_loaders, config)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if is_main_process():
        logger.info(f"Embedding computation finished in {total_time_str}")
        logger.info(f"Embeddings saved at {config.output_dir}")

    if config.distributed:
        dist.destroy_process_group()


def setup_dataloaders(config, mode="pt"):
    logger.info(f"Creating dataset for {mode}")
    train_datasets = create_dataset(f"{mode}_train", config)
    media_types = get_media_types(train_datasets)

    if get_world_size() > 1 and not config.distributed:
        raise RuntimeError("Distributed training is required for multi-GPU operations.")

    batch_size = [config.inputs.batch_size[k] for k in media_types]
    samplers   = create_stateful_sampler(train_datasets, batch_size)

    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size   = batch_size,
        num_workers  = [config.num_workers] * len(media_types),
        is_trains    = [True] * len(media_types),
        collate_fns  = [None] * len(media_types),
    )

    return train_loaders, None, media_types


if __name__ == "__main__":
    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
