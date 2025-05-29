import os
import logging
from os.path import join

import torch
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
logger.setLevel(logging.DEBUG) # Add this line to explicitly set the level


def dummy_internvideo6b_api(video_tensor):
    """Simulate a call to an external InternVideo2-6B model.

    Args:
        video_tensor (Tensor): shape [B, C, T, H, W]

    Returns:
        Tensor: random embeddings with shape [B, 768]
    """
    B = video_tensor.size(0)
    return torch.randn(B, 768, device=video_tensor.device)


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


def gather_embeddings(train_loaders, media_types, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    loader = MetaLoader_rs(name2loader=dict(list(zip(media_types, train_loaders))))

    global_step = 0
    progress_bar = tqdm(loader, total=len(loader))
    for media_type, (images, text, idx) in progress_bar:
        images = images.to(device, non_blocking=True)
        images = images.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        num_frames = images.size(2)
        if num_frames < 4:
            global_step += 1
            continue

        step_dir = os.path.join(output_dir, f"step-{global_step}")
        os.makedirs(step_dir, exist_ok=True)
        save_path = os.path.join(step_dir, "embeddings.pt")

        save_dict = {int(i.item()): {} for i in idx}

        for frame_idx in range(3, num_frames):
            window = images[:, :, frame_idx - 3 : frame_idx + 1]
            embeddings = dummy_internvideo6b_api(window).cpu()
            for vid_id, emb in zip(idx, embeddings):
                save_dict[int(vid_id.item())][frame_idx + 1] = emb

        torch.save(save_dict, save_path)
        global_step += 1



def main(config):
    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, media_types = setup_dataloaders(config, mode=config.mode)
    output_dir = join(config.output_dir, "kinetics-embeddings")

    gather_embeddings(train_loaders, media_types, device, output_dir)


if __name__ == "__main__":
    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
