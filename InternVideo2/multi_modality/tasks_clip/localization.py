import logging
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from dataset import create_dataset, create_loader, create_stateful_sampler
from dataset.serialize import local_broadcast_process_authkey
from tasks_clip.shared_utils import setup_model, get_media_types
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

logger = logging.getLogger(__name__)


def localization_collate_fn(batch):
    videos, captions, starts, ends = [], [], [], []
    for video, cap, s, e in batch:
        videos.append(video)
        captions.append(cap)
        starts.append(s)
        ends.append(e)
    return videos, captions, torch.tensor(starts, dtype=torch.float), torch.tensor(ends, dtype=torch.float)


def setup_dataloaders(config):
    logger.info("Creating localization dataset")
    train_datasets = create_dataset("loc_train", config)
    media_types = get_media_types(train_datasets)

    batch_size = [config.inputs.batch_size[m] for m in media_types]
    samplers = create_stateful_sampler(train_datasets, batch_size)

    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size=batch_size,
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=[localization_collate_fn] * len(media_types),
    )

    return train_loaders, media_types


def train(model, train_loaders, optimizer, tokenizer, epoch, global_step, device, scheduler, scaler, config, data_type):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)
    train_loader = train_loaders[0] if len(train_loaders) == 1 else None
    if train_loader is None:
        train_loader = train_loaders[0]

    model_without_ddp = model.module if config.distributed else model
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    for i, (videos, text, start_times, end_times) in enumerate(iterator):
        videos = [v.to(device, non_blocking=True) for v in videos]
        text_input = tokenizer(text).to(device)

        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            # TODO: replace with actual loss computation
            loss = torch.stack([v.float().mean() for v in videos]).mean()

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

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        global_step += 1

        if config.debug and (i + 1) % 5 == 0:
            break

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")
    if is_main_process() and config.wandb.enable:
        log_dict_to_wandb(metric_logger.get_global_avg_dict(), step=global_step, prefix=f"epoch_{epoch}/")

    return global_step


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    train_loaders, media_types = setup_dataloaders(config)
    num_steps_per_epoch = sum(len(d) for d in train_loaders)

    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs

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
        pretrain=True,
        find_unused_parameters=True,
        num_steps_per_epoch=num_steps_per_epoch,
    )
    if is_main_process() and config.wandb.enable:
        import wandb
        wandb.watch(model)

    data_type = torch.bfloat16 if config.get('use_bf16', True) else torch.float16

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, config.scheduler.epochs):
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
        )
        if hasattr(config, "deepspeed") and config.deepspeed.enable:
            tag = f"ckpt_{epoch:02d}.pth" if not config.get("save_latest", False) else "ckpt_latest.pth"
            model.save_checkpoint(config.output_dir, tag=tag, save_latest=False, exclude_frozen_parameters=True)
        elif is_main_process():
            state_dict = model_without_ddp.state_dict()
            param_grad_dict = {k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()}
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
            }
            torch.save(save_obj, os.path.join(config.output_dir, f"ckpt_{epoch:02d}.pth"))
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
