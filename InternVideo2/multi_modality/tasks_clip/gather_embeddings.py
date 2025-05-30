import os
import logging
from os.path import join

import torch
import asyncio
import aiohttp
from tqdm import tqdm

from dataset.serialize import local_broadcast_process_authkey
from dataset import create_dataset, create_loader, create_stateful_sampler
from dataset import MetaLoader_rs
from tasks_clip.shared_utils import get_media_types
from utils.basic_utils import setup_seed
from utils.config_utils import setup_main
from utils.distributed import get_rank

logging.getLogger().setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def dummy_internvideo6b_api(video_tensor):
    """Simulate a call to an external InternVideo2-6B model.

    Args:
        video_tensor (Tensor): shape [B, C, T, H, W]

    Returns:
        Tensor: random embeddings with shape [B, 768]
    """
    B = video_tensor.size(0)
    return torch.randn(B, 768, device=video_tensor.device)

async def _infer_windows(windows, endpoints):
    """Query embedding servers for a list of windows.

    Args:
        windows (list[Tensor]):
            - A list of 4-frame tensors
            - Each tensor would be in the shape [B, C, 4, H, W]

    Returns:
        embeds (list[Tensor]): A list of embeddings of those windows.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, win in enumerate(windows):
            logger.info(f"Window tensor shape: {win.shape}, size in MB: {win.element_size() * win.nelement() / (1024*1024)}")
            endpoint = endpoints[i % len(endpoints)]
            payload = {"window_tensor": win.tolist()}
            tasks.append(session.post(endpoint, json=payload))

        logger.info(f"Added tasks")
        responses = await asyncio.gather(*tasks)
        # same as responses = await asyncio.gather(tasks[0], tasks[1], etc...)
        embeds = []
        for resp in responses:
            data = await resp.json()
            embeds.append(torch.tensor(data["embeddings"]))
        return embeds


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
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)


def setup_dataloaders(config, mode="pt"):
    logger.info(f"Creating dataset for {mode}")
    train_datasets = create_dataset(f"{mode}_train", config)
    media_types = get_media_types(train_datasets)

    if not config.distributed:
        raise NotImplementedError("Non-distributed training path might need adjustments for samplers.")

    batch_size = [config.inputs.batch_size[k] for k in media_types]
    samplers = create_stateful_sampler(train_datasets, batch_size)

    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size=batch_size,
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=[clone_collate_fn] * len(media_types),
    )

    return train_loaders, media_types


def _get_resume_step(output_dir):
    """Return the step index from which to resume.

    The function scans ``output_dir`` for subfolders following the
    ``step-<idx>`` pattern and checks whether ``embeddings.pt`` exists in
    each folder. The next step after the largest completed one is returned.
    """

    if not os.path.isdir(output_dir):
        return 0

    completed = []
    for d in os.listdir(output_dir):
        if not d.startswith("step-"):
            continue
        try:
            step = int(d.split("-", 1)[1])
        except ValueError:
            continue
        if os.path.isfile(os.path.join(output_dir, d, "embeddings.pt")):
            completed.append(step)
    return max(completed) + 1 if completed else 0


import ray
import logging
from tqdm import tqdm
import os
from os.path import join
import numpy as np

# Setup logging
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Connect to Ray cluster - replace with your server's IP
ray.init(address="ray://192.168.68.130:10001")

# Rest of your imports and setup functions

def gather_embeddings(train_loaders, media_types, device, output_dir, resume=True):
    os.makedirs(output_dir, exist_ok=True)

    start_step = _get_resume_step(output_dir) if resume else 0
    if start_step > 0:
        logger.info(f"Resuming from step {start_step}")

    loader = MetaLoader_rs(
        name2loader=dict(list(zip(media_types, train_loaders))), skip_num=start_step
    )

    total_steps = start_step + len(loader)
    global_step = start_step
    progress_bar = tqdm(loader, total=total_steps, initial=start_step)

    # Get a handle to the remote service
    services = ray.get_actor("InternVideo2Service")

    # Define a window batch size to process windows in chunks
    window_batch_size = 32  # Adjust based on your memory

    for media_type, (images, text, idx) in progress_bar:
        images = images.to(device, non_blocking=True)
        images = images.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        num_frames = images.size(2)
        if num_frames < 4:
            global_step += 1
            continue

        logger.info(f"Processing {idx}")
        step_dir = os.path.join(output_dir, f"step-{global_step}")
        os.makedirs(step_dir, exist_ok=True)
        save_path = os.path.join(step_dir, "embeddings.pt")

        save_dict = {int(i.item()): {} for i in idx}

        # Process windows in batches
        all_windows = [images[:, :, i - 3 : i + 1] for i in range(3, num_frames)]
        num_windows = len(all_windows)
        logger.info(f"Total windows to process: {num_windows}")

        for i in range(0, num_windows, window_batch_size):
            window_batch = all_windows[i : i + window_batch_size]
            logger.info(f"Processing window batch {i//window_batch_size + 1}/{(num_windows + window_batch_size - 1)//window_batch_size}...")

            # Send tensors to remote Ray service and get results
            # Convert to numpy for better serialization
            futures = []
            for window in window_batch:
                # Submit tasks in parallel
                futures.append(services.embed_video.remote(window.cpu().numpy()))

            # Get results
            embeddings_batch_list = ray.get(futures)

            # Store embeddings
            start_frame_idx = i + 3
            for win_batch_idx, embeddings in enumerate(embeddings_batch_list):
                current_frame_idx = start_frame_idx + win_batch_idx
                for vid_id, emb in zip(idx, embeddings):
                    save_dict[int(vid_id.item())][current_frame_idx] = torch.tensor(emb)

        torch.save(save_dict, save_path)
        global_step += 1


def main(config):
    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, media_types = setup_dataloaders(config, mode=config.mode)
    output_dir = join(config.output_dir, "kinetics-embeddings")

    resume = getattr(config, "resume", True)
    api_endpoints = getattr(config, "embedding_endpoints", [])

    logger.info(f"Using API endpoints: {api_endpoints}")

    gather_embeddings(train_loaders, media_types, device, output_dir, resume) #api_endpoints, resume)

if __name__ == "__main__":
    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
