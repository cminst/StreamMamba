import logging
import os
import torch
import pickle

from dataset import MetaLoader_rs, create_dataset, create_loader, create_stateful_sampler
from tasks_clip.shared_utils import get_media_types
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from tqdm import tqdm

logger = logging.getLogger(__name__)

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

def setup_dataloaders(config, mode="pt", samplers_state=None):
    logger.info(f"Creating dataset for {mode}")
    train_datasets = create_dataset(f"{mode}_train", config)
    media_types = get_media_types(train_datasets)

    if not config.distributed:
        raise NotImplementedError("Non-distributed training path might need adjustments for samplers.")

    batch_size = [config.inputs.batch_size[k] for k in media_types]
    samplers = create_stateful_sampler(train_datasets, batch_size)

    if samplers_state:
        for sampler, state in zip(samplers, samplers_state):
            sampler.load_state_dict(state)

    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size   = batch_size,
        num_workers  = [config.num_workers] * len(media_types),
        is_trains    = [True] * len(media_types),
        collate_fns  = [clone_collate_fn] * len(media_types),
    )

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
    return train_loaders, test_name2loaders, media_types, samplers

def generate_sampler_states(target_global_step, output_path):
    cfg = setup_main()
    
    # Ensure distributed setup is consistent if original training was distributed
    # This part might need adjustment based on how your distributed setup is typically initialized
    # For now, assuming it's handled by setup_main or not strictly necessary for sampler state generation
    
    train_loaders, _, _, samplers = setup_dataloaders(cfg, mode=cfg.mode)
    
    media_types = get_media_types(train_loaders)
    train_loader_agg = MetaLoader_rs(
        name2loader=dict(list(zip(media_types, train_loaders))),
        skip_num=0, # We will iterate manually
        seed=cfg.seed + (cfg.get('epoch', 0) if hasattr(cfg, 'epoch') else 0), # Use the same seed logic as pretrain.py
    )

    logger.info(f"Iterating through data loader to reach global step {target_global_step}...")
    progress_bar = tqdm(
        train_loader_agg,
        total=target_global_step, # Iterate up to the target step
        desc="Generating sampler states",
        disable=not is_main_process()
    )

    current_step = 0
    for i, _ in enumerate(progress_bar):
        current_step += 1
        if current_step >= target_global_step:
            break
    
    if current_step < target_global_step:
        logger.warning(f"Reached end of data loader at step {current_step}, but target was {target_global_step}. Sampler states might not be accurate.")

    sampler_states = [s.state_dict() for s in samplers]
    
    with open(output_path, 'wb') as f:
        pickle.dump(sampler_states, f)
    
    logger.info(f"Sampler states saved to {output_path}")

if __name__ == "__main__":
    import argparse
    import sys
    from torch.utils.data._utils.collate import default_collate # Added import

    # Temporarily store original sys.argv
    original_argv = sys.argv
    
    # Parse arguments for generate_sampler_state.py itself
    parser = argparse.ArgumentParser(description="Generate sampler states for resuming training.")
    parser.add_argument("config_file", type=str, help="Path to the training configuration file.")
    parser.add_argument("--global_step", type=int, required=True, help="The global step at which to capture sampler states.")
    parser.add_argument("--output_path", type=str, default="sampler_states.pkl", help="Path to save the generated sampler states.")
    
    # Parse only the arguments specific to this script first
    # This allows setup_main to parse the config_file and opts later
    args, unknown = parser.parse_known_args()

    # Reconstruct sys.argv for setup_main to parse the config_file and opts
    # The first element is always the script name
    sys.argv = [original_argv[0], args.config_file] + unknown

    logging.basicConfig(level=logging.INFO) # Basic logging setup
    generate_sampler_states(args.global_step, args.output_path)

    # Restore original sys.argv to avoid side effects
    sys.argv = original_argv
