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
        dataset.datasets[0].media_type
        if isinstance(dataset, ConcatDataset)
        else dataset.media_type
        for dataset in datasets
    ]

    return media_types


def setup_model(
    config, model_cls, pretrain=False, find_unused_parameters=False, num_steps_per_epoch=-1,
):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    model = model_cls(config=config, is_pretrain=pretrain)

    model = model.to(torch.device(config.device))
    if config.use_half_precision:
        if config.get('bf16', True):
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
        scaler = torch.cuda.amp.GradScaler(enabled=config.use_half_precision) # This is never used actually if we fixed bf16

    start_epoch = 0
    global_step = 0

    # auto resume the latest checkpoint
    if config.get("auto_resume", False):
        logger.info("Auto resuming")
        model_latest = join(config.output_dir, "ckpt_latest.pth")
        model_best = join(config.output_dir, "ckpt_best.pth")

        large_step_num = -1
        large_step_path = None
        large_epoch_num = -1
        large_epoch_path = None
        for fname in os.listdir(config.output_dir):
            if fname.startswith("ckpt_iter"):
                step_str = fname[len("ckpt_iter"):]
                if step_str.endswith(".pth"):
                    step_str_nosuffix = step_str[:-4]
                else:
                    step_str_nosuffix = step_str
                if step_str_nosuffix.isdigit():
                    step_num = int(step_str_nosuffix)
                    if step_num > large_step_num:
                        large_step_num = step_num
                        large_step_path = join(config.output_dir, fname)
            elif fname.startswith("ckpt_"):
                epoch_str = fname[len("ckpt_"):]
                if epoch_str.endswith(".pth"):
                    epoch_str_nosuffix = epoch_str[:-4]
                else:
                    epoch_str_nosuffix = epoch_str
                if epoch_str_nosuffix.isdigit():
                    epoch_num = int(epoch_str_nosuffix)
                    if epoch_num > large_epoch_num:
                        large_epoch_num = epoch_num
                        large_epoch_path = join(config.output_dir, fname)

        if large_step_num != -1:
            logger.info(f"Load the latest step: {large_step_num}")
            candidate = large_step_path
            if candidate and (osp.isfile(candidate) or osp.isdir(candidate)):
                model_latest = candidate

        if large_epoch_num != -1 and (large_epoch_num + 1) * num_steps_per_epoch > large_step_num:
            logger.info(f"Load the latest epoch: {large_epoch_num}")
            candidate = large_epoch_path
            if candidate and (osp.isfile(candidate) or osp.isdir(candidate)):
                model_latest = candidate

        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            if osp.isfile(model_latest) or osp.isdir(model_latest):
                config.pretrained_path = model_latest
                config.resume = True
            elif osp.isfile(model_best) or osp.isdir(model_best):
                config.pretrained_path = model_best
                config.resume = True
            else:
                logger.info(f"Not found checkpoint in {config.output_dir}")
        else:
            if osp.isfile(model_latest):
                config.pretrained_path = model_latest
                config.resume = True
            elif osp.isfile(model_best):
                config.pretrained_path = model_best
                config.resume = True
            else:
                logger.info(f"Not found checkpoint in {config.output_dir}")

    # load pretrained model
    if hasattr(config, "deepspeed") and config.deepspeed.enable:
        logger.info('Use deepspeed to initialize model!!!')
        model = model_without_ddp
        model, optimizer, _, _ = deepspeed.initialize(
            args=config, model=model, model_parameters=optimizer_params, dist_init_required=not config.distributed,
            lr_scheduler=lambda opt: create_scheduler(config.scheduler, opt)
        )
        if osp.isdir(config.pretrained_path):
            logger.info(f"Load pretrained model from {config.pretrained_path}")
            output_dir, tag = os.path.split(config.pretrained_path)
            if config.resume:
                _, client_state = model.load_checkpoint(output_dir, tag=tag, load_module_strict=False)
                global_step = model.global_steps
                assert num_steps_per_epoch > 0, "Please provide num_steps_per_epoch"
                start_epoch = global_step // num_steps_per_epoch
            else:
                _, client_state = model.load_checkpoint(
                    output_dir, tag=tag, load_module_strict=False, 
                    load_optimizer_states=False, load_lr_scheduler_states=False,
                    load_module_only=True
                )
    else:
        if osp.isfile(config.pretrained_path):
            checkpoint = torch.load(config.pretrained_path, map_location="cpu")
            logger.info(f"Load pretrained model from {config.pretrained_path}")
            logger.info(f"Checkpoint contains keys: {list(checkpoint.keys())}")
            if 'model' in checkpoint.keys():
                state_dict = checkpoint["model"]
            elif 'module' in checkpoint.keys():
                state_dict = checkpoint["module"]
            else:
                state_dict = checkpoint
            # resume optimizer
            if config.resume:
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    logger.info("Loaded optimizer state from checkpoint")
                else:
                    logger.warning("Optimizer state not found in checkpoint")
                if "scheduler" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                    logger.info("Loaded scheduler state from checkpoint")
                else:
                    logger.warning("Scheduler state not found in checkpoint")
                if "scaler" in checkpoint and scaler is not None:
                    scaler.load_state_dict(checkpoint["scaler"])
                    logger.info("Loaded scaler state from checkpoint")
                elif scaler is not None:
                    logger.warning("Scaler state not found in checkpoint")

                global_step = checkpoint.get("global_step", 0)
                start_epoch = checkpoint.get("epoch", 0)
                
                if num_steps_per_epoch > 0 and global_step % num_steps_per_epoch == 0:
                    start_epoch += 1

            msg = model_without_ddp.load_state_dict(state_dict, strict=False)
            logger.info(msg)
            if hasattr(msg, "missing_keys") and hasattr(msg, "unexpected_keys"):
                if msg.missing_keys:
                    logger.info(f"Missing keys when loading model: {msg.missing_keys}")
                if msg.unexpected_keys:
                    logger.info(f"Unexpected keys when loading model: {msg.unexpected_keys}")
            # Handle optional streaming student checkpoint
            if isinstance(checkpoint, dict):
                student_key = None
                if "streaming_student" in checkpoint:
                    student_key = "streaming_student"
                elif "streaming_vision_encoder" in checkpoint:
                    student_key = "streaming_vision_encoder"
                if student_key is not None:
                    try:
                        model_without_ddp.streaming_vision_encoder.load_state_dict(
                            checkpoint[student_key]
                        )
                        logger.info("Loaded streaming student model from checkpoint")
                    except Exception as e:
                        logger.warning(f"Failed to load streaming student model: {e}")
            logger.info(f"Loaded checkpoint from {config.pretrained_path}")
        else:
            logger.warning("No pretrained checkpoint provided, training from scratch")
    
    logger.info(f"Cuda memory after create model: {torch.cuda.memory_allocated() // 1024**2}M, Max mem: {torch.cuda.max_memory_allocated() // 1024**2}M")

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
