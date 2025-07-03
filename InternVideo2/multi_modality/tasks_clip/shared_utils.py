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


def resume_from_checkpoint(checkpoint_dir, model, optimizer, scheduler):
    """Load DeepSpeed model and optimizer states from ``checkpoint_dir``.

    Returns the start epoch and global step restored from the checkpoint.
    """

    start_epoch = 0
    global_step = 0

    model_state_file = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
    optim_state_file = os.path.join(
        checkpoint_dir, "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
    )

    if osp.isfile(model_state_file):
        state = torch.load(model_state_file, map_location="cpu", weights_only=False)
        logger.info(f"Loaded model state from {model_state_file}")
        for key in state.keys():
            logger.info(f" - {key}")

        if "module" in state:
            model.load_state_dict(state["module"], strict=False)
        if "lr_scheduler" in state and scheduler is not None:
            scheduler.load_state_dict(state["lr_scheduler"])
        start_epoch = state.get("epoch", start_epoch)
        global_step = state.get("global_step", state.get("global_steps", global_step))
    else:
        logger.warning(f"Model state file not found: {model_state_file}")

    if osp.isfile(optim_state_file):
        opt_state = torch.load(optim_state_file, map_location="cpu", weights_only=False)
        logger.info(f"Loaded optimizer state from {optim_state_file}")
        logger.info(f"Optimizer state keys: {list(opt_state.keys())}")
        opt_sd = opt_state.get("optimizer_state_dict")
        if opt_sd is not None:
            try:
                optimizer.load_state_dict(opt_sd)
            except KeyError:
                if isinstance(opt_sd, dict) and 0 in opt_sd:
                    optimizer.load_state_dict(opt_sd[0])
                elif isinstance(opt_sd, list) and len(opt_sd) > 0:
                    optimizer.load_state_dict(opt_sd[0])
                else:
                    logger.error(f"Unexpected optimizer state format: {type(opt_sd)}")
    else:
        logger.warning(f"Optimizer state file not found: {optim_state_file}")

    return start_epoch, global_step


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

    if hasattr(config, "deepspeed") and config.deepspeed.enable:
        optimizer_params = create_optimizer(config.optimizer, model, return_group=True)
        scheduler = None
        scaler = None
    else:
        if config.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config.gpu],
                find_unused_parameters=find_unused_parameters,  # `False` for image-only task
            )

        optimizer = create_optimizer(config.optimizer, model)
        scheduler = create_scheduler(config.scheduler, optimizer)
        scaler = torch.cuda.amp.GradScaler(
            enabled=config.use_half_precision
        )  # This is never used actually if we fixed bf16

    start_epoch = 0
    global_step = 0

    # initialize deepspeed engine when enabled
    if hasattr(config, "deepspeed") and config.deepspeed.enable:
        logger.info("Use deepspeed to initialize model")
        model = model_without_ddp
        model, optimizer, _, _ = deepspeed.initialize(
            args=config,
            model=model,
            model_parameters=optimizer_params,
            dist_init_required=not config.distributed,
            lr_scheduler=lambda opt: create_scheduler(config.scheduler, opt),
        )

    # ----- New resume logic -----
    if config.resume and config.pretrained_path:
        logger.info(f"Resuming training from {config.pretrained_path}")
        start_epoch, global_step = resume_from_checkpoint(
            config.pretrained_path, model, optimizer, scheduler
        )
    else:
        logger.info("No resume checkpoint provided, starting from scratch")

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
    )
