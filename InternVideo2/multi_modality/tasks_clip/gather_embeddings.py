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
            endpoint = endpoints[i % len(endpoints)]
            payload = {"window_tensor": win.tolist()}
            tasks.append(session.post(endpoint, json=payload))
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


def gather_embeddings(train_loaders, media_types, device, output_dir, api_endpoints=None, resume=True):
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
    for media_type, (images, text, idx) in progress_bar:
        images = images.to(device, non_blocking=True)
        images = images.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        num_frames = images.size(2)
        if num_frames < 4:
            # Should not go here because of dataset filtering.
            global_step += 1
            continue

        # Directory to save embeddings for this step
        step_dir = os.path.join(output_dir, f"step-{global_step}")
        os.makedirs(step_dir, exist_ok=True)
        # Path for the embeddings file
        save_path = os.path.join(step_dir, "embeddings.pt")

        # Initialize dictionary to store embeddings.
        # Keys are video IDs, values are dictionaries mapping frame index to embedding.
        save_dict = {int(i.item()): {} for i in idx}

        # Extract sliding windows of 4 frames each.
        # Each window is like [frame i-3, frame i-2, frame i-1, frame i].
        # The loop starts from i=3 because we need 4 frames (0, 1, 2, 3).
        windows = [images[:, :, i - 3 : i + 1] for i in range(3, num_frames)]

        # Get embeddings for the windows using the API or a dummy function.
        if api_endpoints: # api_endpoints is provided in config.api_endpoints
            # Use external API endpoints for inference if provided
            embeddings_list = asyncio.run(_infer_windows(windows, api_endpoints))
        else:
            # Use dummy function for inference (e.g., during development or testing)
            embeddings_list = [dummy_internvideo6b_api(w).cpu() for w in windows]

        # Store embeddings in the save_dict.
        # The frame index in the save_dict is the *last* frame index in the window.
        # Since windows are [i-3, i-2, i-1, i] and the loop for windows is range(3, num_frames),
        # the window index `frame_idx` (starting from 0 in enumerate) corresponds to
        # the frame index `i` in the original loop `range(3, num_frames)`.
        # So the last frame index is `i`, which is `frame_idx + 3`.
        for frame_idx, embeddings in enumerate(embeddings_list, start=3): # Start=3 aligns with 'i' from window creation
            for vid_id, emb in zip(idx, embeddings):
                # Store embedding keyed by video ID and the last frame index in the window
                save_dict[int(vid_id.item())][frame_idx] = emb # corrected frame index

        torch.save(save_dict, save_path)
        global_step += 1

def main(config):
    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, media_types = setup_dataloaders(config, mode=config.mode)
    output_dir = join(config.output_dir, "kinetics-embeddings")

    resume = getattr(config, "resume", True)
    api_endpoints = getattr(config, "api_endpoints", [])
    gather_embeddings(train_loaders, media_types, device, output_dir, api_endpoints, resume)

if __name__ == "__main__":
    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
