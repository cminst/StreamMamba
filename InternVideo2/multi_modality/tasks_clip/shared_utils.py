import copy
import logging
import os
import os.path as osp
from os.path import join

import torch
import deepspeed
from torch.utils.data import ConcatDataset, DataLoader

from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler

logger = logging.getLogger(__name__)


def get_media_types(datasources):
    """get the media types for for all the dataloaders.

    Args:
        datasources (List): List of dataloaders or datasets.

    Returns: List. The media_types.

    """
    if isinstance(datasources[0], DataLoader):
        datasets = [dataloader.dataset for dataloader in datasources]
    else:
        datasets = datasources
    media_types = [
        (
            dataset.datasets[0].media_type
            if isinstance(dataset, ConcatDataset)
            else dataset.media_type
        )
        for dataset in datasets
    ]

    return media_types


def setup_model(
    config,
    model_cls,
    pretrain=False,
    find_unused_parameters=False,
    num_steps_per_epoch=-1,
):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    model = model_cls(config=config, is_pretrain=pretrain)

    model = model.to(torch.device(config.device))
    if config.use_half_precision:
        if config.get("bf16", True):
            logger.info("Change to bfloat16 for model")
            model = model.to(torch.bfloat16)
        else:
            logger.info("Change to float16 for model")
            model = model.half()
    tokenizer = model.tokenizer
    model_without_ddp = model

    start_epoch = 0
    global_step = 0

    # These will be populated differently depending on whether we use DeepSpeed
    optimizer = None
    scheduler = None
    scaler = None

    samplers_state = None
    if hasattr(config, "deepspeed") and config.deepspeed.enable:
        logger.info("Initializing model with DeepSpeed")
        optimizer_params = create_optimizer(config.optimizer, model, return_group=True)
        # DeepSpeed handles the scheduler, so we pass a lambda
        model, optimizer, _, scheduler = deepspeed.initialize(
            args=config,
            model=model,
            model_parameters=optimizer_params,
            dist_init_required=not config.distributed,
            lr_scheduler=lambda opt: create_scheduler(config.scheduler, opt),
        )

        logger.info(f"DEBUG: Initial LR (before resume attempt): {optimizer.param_groups[0]['lr']}")

        # Resume deepspeed training
        if config.resume and config.pretrained_path:
            logger.info(f"Attempting to resume DeepSpeed training from: {config.pretrained_path}")
            # model.load_checkpoint handles finding the latest checkpoint in the directory
            # It also correctly loads model, optimizer, and scheduler states.
            # The second return value is the client_state dictionary we saved.
            load_path, client_state = model.load_checkpoint(config.pretrained_path)

            if load_path is not None:
                logger.info(f"DEBUG: DeepSpeed load_path: {load_path}")
                if client_state:
                    logger.info(f"DEBUG: DeepSpeed client_state: {client_state}")
                    # We saved epoch and global_step, so we load them back.
                    start_epoch = client_state.get('epoch', 0)
                    global_step = client_state.get('global_step', 0)
                    samplers_state = client_state.get('data_sampler', None)
                    rng_state = client_state.get('rng_state', None)
                    if rng_state:
                        logger.info(f"DEBUG: DeepSpeed found RNG state in client_state.")
                        # The actual setting of RNG state happens in pretrain.py's main function
                        # after setup_model returns.
                    logger.info(
                        f"Successfully resumed from checkpoint. "
                        f"Loaded client state: epoch={start_epoch}, global_step={global_step}"
                    )
                else:
                    logger.warning(
                        "Resumed checkpoint but no client_state found. "
                        "Starting from epoch 0, global_step 0."
                    )
                logger.info(f"DEBUG: LR after DeepSpeed resume attempt: {optimizer.param_groups[0]['lr']}")
            else:
                logger.warning(
                    f"Could not find a valid checkpoint to resume from in {config.pretrained_path}. "
                    "Starting from scratch."
                )
        else:
            logger.info("No resume checkpoint provided or resume is disabled. Starting from scratch.")

    else: # Fallback for standard DDP (non-DeepSpeed)
        if config.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config.gpu],
                find_unused_parameters=find_unused_parameters,
            )
        model_without_ddp = model.module if config.distributed else model

        optimizer = create_optimizer(config.optimizer, model)
        scheduler = create_scheduler(config.scheduler, optimizer)
        scaler = torch.cuda.amp.GradScaler(enabled=config.use_half_precision)

        if config.resume and config.pretrained_path:
            logger.info(f"Resuming from non-DeepSpeed checkpoint: {config.pretrained_path}")
            checkpoint = torch.load(config.pretrained_path, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if checkpoint.get("scaler") is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", 0)
            samplers_state = checkpoint.get('data_sampler', None)
            logger.info(f"Resumed from epoch {start_epoch}, global_step {global_step}")

    logger.info(
        f"Cuda memory after create model: {torch.cuda.memory_allocated() // 1024**2}M, Max mem: {torch.cuda.max_memory_allocated() // 1024**2}M"
    )

    return (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
        samplers_state,
    )
