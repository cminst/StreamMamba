import logging
import os
import time
import datetime

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data._utils.collate import default_collate

from dataset import create_dataset, create_loader, create_stateful_sampler
from tasks_clip.shared_utils import setup_model, get_media_types
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
from utils.distributed import is_main_process, get_rank

logger = logging.getLogger(__name__)


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
    return default_collate(batch)


def setup_dataloaders(config):
    logger.info("Creating dataset for distillation training")
    train_datasets = create_dataset("pt_train", config)
    media_types = get_media_types(train_datasets)

    if not config.distributed:
        raise NotImplementedError("Only distributed training is supported")

    batch_size = [config.inputs.batch_size[m] for m in media_types]
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


def train_one_epoch(model, loader, optimizer, scaler, device, config):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", SmoothedValue(window=1, fmt="{value:.4f}"))

    header = "Train"
    iterator = metric_logger.log_every(loader, config.log_freq, header)

    vit = model.streaming_vision_encoder.vit_lite
    rnn = model.streaming_vision_encoder.rnn

    for videos, _, _ in iterator:
        videos = videos.to(device, non_blocking=True)
        videos = videos.permute(0, 2, 1, 3, 4)  # B,C,T,H,W

        state = model.streaming_vision_encoder.init_hidden(videos.size(0), device)
        with torch.no_grad():
            feat0, _ = vit.extract_features(videos[:, :, 0])
        _, state = rnn(feat0, state)

        pred_losses = []
        for t in range(1, videos.shape[2]):
            mu, logv = rnn.predict_next_feat()
            with torch.no_grad():
                gt_feat, _ = vit.extract_features(videos[:, :, t])
            pred_losses.append(0.5 * ((gt_feat - mu).pow(2) * torch.exp(-logv) + logv).mean())
            _, state = rnn(gt_feat, state)

        loss = torch.stack(pred_losses).mean()
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
        if hasattr(config, "scheduler"):
            config.scheduler.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"], loss=loss.item())

    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: " + str(metric_logger.global_avg()))


def main(config):
    if is_main_process() and config.wandb.enable:
        import wandb
        wandb.init(entity=config.wandb.entity, project=config.wandb.project, config=config)

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, media_types = setup_dataloaders(config)
    train_loader = train_loaders[0]
    num_steps_per_epoch = len(train_loader)

    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    cudnn.benchmark = True

    model_cls = eval(config.model.get("model_cls", "InternVideo2_CLIP_small"))
    (model, model_without_ddp, optimizer, scheduler, scaler, tokenizer, start_epoch, _,) = setup_model(
        config,
        model_cls=model_cls,
        pretrain=False,
        find_unused_parameters=True,
        num_steps_per_epoch=num_steps_per_epoch,
    )
    config.scheduler = scheduler

    if config.get("use_bf16", True):
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config.scheduler.epochs):
        if config.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, optimizer, scaler, device, config)
        if is_main_process():
            state = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "epoch": epoch + 1,
            }
            torch.save(state, os.path.join(config.output_dir, f"ckpt_{epoch:02d}.pth"))
        torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")

    if is_main_process() and config.wandb.enable:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
