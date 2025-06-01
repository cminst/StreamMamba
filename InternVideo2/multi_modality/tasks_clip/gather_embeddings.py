import os
import logging
import torch
from torch.utils.data import DataLoader
from utils.basic_utils import MetricLogger
from huggingface_hub import hf_hub_download

from dataset import create_dataset
from utils.basic_utils import setup_seed
from utils.config_utils import setup_main
from demo.config import Config, eval_dict_leaf
from demo.utils import setup_internvideo2

logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    dataset = create_dataset(f"{mode}_train", config)
    # `create_dataset` always returns a list, even when there is only one
    # dataset. DataLoader expects a single `Dataset` object, so grab the first
    # dataset in the list.  Warn the user if there are multiple datasets as this
    # script currently supports a single dataset only.
    if isinstance(dataset, list):
        if len(dataset) > 1:
            logger.warning(
                f"Multiple datasets returned for {mode}_train, using the first one only"
            )
        dataset = dataset[0]

    loader = DataLoader(
        dataset,
        batch_size=config.inputs.batch_size["video"],
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=clone_collate_fn,
        pin_memory=True,
    )
    return loader

MODEL = None
CFG = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model():
    global MODEL, CFG
    if MODEL is not None:
        return MODEL

    # Log the base directory from which config_path is computed
    base_dir = os.path.dirname(__file__)
    logger.info(f"Base directory for config calculation: {base_dir}")

    config_path = "scripts/pretraining/stage2/6B/config.py"
    # Log the computed config path
    logger.info(f"Computed config path: {config_path}")

    ckpt_path = os.environ.get("IV2_6B_CKPT")
    if not ckpt_path:
        repo_id = "qingy2024/InternVideo2_S2_6B_Vision"
        filename = "InternVideo2_S2_6B_vision.pt"
        logger.info(f"IV2_6B_CKPT not set, downloading {filename} from {repo_id}")
        try:
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
            logger.info(f"Downloaded model to {ckpt_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model from {repo_id}/{filename}: {e}") from e

    # Log before loading config from file
    logger.info(f"Attempting to load config from file: {config_path}")
    CFG = Config.from_file(config_path, log=False)
    # Log after loading config
    logger.info(f"Config loaded successfully from file.")

    CFG = eval_dict_leaf(CFG)
    CFG.model.vision_ckpt_path = ckpt_path
    CFG.model.vision_encoder.pretrained = ckpt_path
    CFG.pretrained_path = ckpt_path
    CFG.device = str(DEVICE)

    logger.info(f"Final config for InternVideo2 Stage2 6b: {CFG.model}")

    MODEL, _ = setup_internvideo2(CFG, no_text=True)
    MODEL.eval()
    return MODEL

def gather_embeddings(loader, device, output_dir, log_freq=50):
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    model = _load_model()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Processing"
    iterator = metric_logger.log_every(loader, log_freq, header)

    for step, (images, text, idx) in enumerate(iterator):
        images = images.to(device, non_blocking=True)
        images = images.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        num_frames = images.size(2)
        if num_frames < 4:
            continue

        step_dir = os.path.join(output_dir, f"step-{step}")
        os.makedirs(step_dir, exist_ok=True)
        save_path = os.path.join(step_dir, f"embeddings.pt")

        save_dict = {int(i.item()): {} for i in idx}
        all_windows = [images[:, :, i - 3 : i + 1] for i in range(3, num_frames)]

        with torch.no_grad():
            for win_idx, window in enumerate(all_windows):
                embeddings = model.get_vid_feat(window.permute(0, 2, 1, 3, 4))
                for vid_id, emb in zip(idx, embeddings.cpu()):
                    save_dict[int(vid_id.item())][win_idx + 3] = emb

        torch.save(save_dict, save_path)

def main(config):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    setup_seed(config.seed)

    # Set up dataloader
    loader = setup_dataloaders(config, mode=config.mode)
    print("Dataloaders set up!")

    # Process data
    output_dir = os.path.join(config.output_dir, "kinetics-embeddings")
    gather_embeddings(loader, device, output_dir, log_freq=config.log_freq)

if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
