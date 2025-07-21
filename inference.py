"""
Single-file demo for InternVideo2-B14 with SPFS
------------------------------------------------
1. Downloads ONE checkpoint file: ckpt_step_12500.pt
2. Loads the whole model in one line
3. Processes a video with SPFS after an 8-frame warm-up
"""
import os
import sys
from pathlib import Path

import torch
from easydict import EasyDict as edict
from huggingface_hub import hf_hub_download
from cv2 import VideoCapture

# ------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------
sys.path.append(os.getcwd())

from demo.config import Config, eval_dict_leaf
from demo.utils import _frame_from_video, frames2tensor
from dataset import get_train_transform
from models import InternVideo2_CLIP_small

# ------------------------------------------------------------------
# config
# ------------------------------------------------------------------
args = edict(
    hf_repo='qingy2024/InternVideo2-B14',
    config_dir="scripts/spfs/clip/B14/",
    video_path='photography-model/data/act75/1.mp4',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    seed=42,
)

device = torch.device(args.device)

# ------------------------------------------------------------------
# helper: load model & checkpoint
# ------------------------------------------------------------------
def build_model(config_dir: str, ckpt_path: str, device: torch.device):
    """Build model and load full checkpoint"""
    config = Config.from_file(os.path.join(config_dir, "config.py"))
    config = eval_dict_leaf(config)
    config.model.streaming_vision_encoder.rnn_type = 'mamba_spfs'

    model = InternVideo2_CLIP_small(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)['model']
    a, b = model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


# ------------------------------------------------------------------
# helper: video handling
# ------------------------------------------------------------------
def load_video_frames(video_path: str):
    """Return list of PIL images (RGB)"""
    cap = VideoCapture(video_path)
    return [img for img in _frame_from_video(cap)]


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main(confidence_threshold = 0.9, max_consecutive_skips = 6,):
    # 1. download single checkpoint
    ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename="spfs_r64/ckpt_step_14500.pt")
    print(f"Downloaded checkpoint to {ckpt_path}")

    # 2. build model
    model = build_model(args.config_dir, ckpt_path, device)

    # 3. load video
    frames = load_video_frames(args.video_path)
    print(f"Loaded {len(frames)} frames from {args.video_path}")

    # 4. warm-up: process first 8 frames without skipping
    hidden = model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)
    for i in range(8):
        x = frames2tensor([frames[i]], fnum=1, target_size=(224, 224), device=device)
        x = x.permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            _, hidden, _ = model.streaming_vision_encoder(
                x,
                prev_hidden_state=hidden,
                confidence_threshold=1.0,
                max_consecutive_skips=0,
            )
    print("Warm-up phase completed.")

    skipped_frames = 0

    # 5. main loop with SPFS
    for idx in range(8, len(frames)):
        x = frames2tensor([frames[idx]], fnum=1, target_size=(224, 224), device=device)
        x = x.permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            _, hidden, spfs_info = model.streaming_vision_encoder(
                x,
                prev_hidden_state=hidden,
                confidence_threshold=confidence_threshold,
                max_consecutive_skips=max_consecutive_skips,
            )

        if idx % 10 == 0:
            print(f"SPFS Info: {spfs_info}")

        if spfs_info.confidence > confidence_threshold:
            skipped_frames += 1

    print(f"Total Skipped Frames: {skipped_frames}/{len(frames)} ({skipped_frames/len(frames) * 100:.2f}%)")

if __name__ == "__main__":
    main(
        confidence_threshold = 0.85,
        max_consecutive_skips = 5,
    )
