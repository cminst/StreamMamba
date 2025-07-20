import sys
import os

sys.path.append(os.getcwd())
import numpy as np
import os
import io
import cv2

import torch

from demo.config import (Config,
                    eval_dict_leaf)

from demo.utils import *

from iv2_utils.iv2 import *
from models import *

from huggingface_hub import hf_hub_download
from collections import OrderedDict
from utils.basic_utils import *
from easydict import EasyDict as edict

args = edict(
    hf_repo='qingy2024/InternVideo2-B14',
    config_dir="scripts/spfs/clip/B14/"
)

device = torch.device('cuda')

print("Downloading Mamba checkpoint from Hugging Face Hub...")
mamba_checkpoint_path = hf_hub_download(repo_id=args.hf_repo, filename="mamba_mobileclip_ckpt.pt")
print(f"Downloaded Mamba checkpoint to {mamba_checkpoint_path}")

print("Downloading SPFS checkpoint from Hugging Face Hub...")
spfs_checkpoint_path = hf_hub_download(repo_id=args.hf_repo, filename="spfs_ckpt.pt")
print(f"Downloaded SPFS checkpoint to {spfs_checkpoint_path}")

mamba_ckpt = torch.load(mamba_checkpoint_path, map_location=device)
processed_mamba = process_state_dict(mamba_ckpt)

spfs_ckpt = torch.load(spfs_checkpoint_path, map_location=device, weights_only = False)
processed_spfs = process_state_dict(spfs_ckpt)

merged_state_dict = merge_state_dicts([processed_mamba, processed_spfs], override=True)

ckpt_keys = list(merged_state_dict.keys())

config = Config.from_file(os.path.join(args.config_dir, "config.py"))

config = eval_dict_leaf(config)

# Set rnn_type to mamba_spfs
config.model.streaming_vision_encoder.rnn_type = 'mamba_spfs'

intern_model = InternVideo2_CLIP_small(config)
intern_model.to(device)

for n, p in intern_model.named_parameters():
    if n not in ckpt_keys:
        print(n)

video_path = 'sunset.mp4'

from cv2 import VideoCapture
from demo.utils import _frame_from_video

frames = [x for x in _frame_from_video(VideoCapture(video_path))]

import PIL
from PIL import Image

frame_tensor = frames2tensor([frames[75]], fnum=1, target_size=(224,224), device=device)

# Get the ground truth feature for the reference frame using the same encoder as the SPFS module
with torch.no_grad():
    frame_embed, _ = intern_model.streaming_vision_encoder.vit_lite.extract_features(frame_tensor.squeeze(1))
    frame_embed = intern_model.vision_align(frame_embed)

text_embed = intern_model.get_txt_feat('A person splashing and make big splash in water')

from torch.nn import functional as F

F.cosine_similarity(frame_embed, text_embed)

_, _ = intern_model.load_state_dict(merged_state_dict, strict=False)
intern_model.eval()

hidden = intern_model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)
for frame in range(50, 99):
    frame_tensor = frames2tensor([frames[frame]], fnum=1, target_size=(224,224), device=device)
    next_frame_tensor = frames2tensor([frames[frame+1]], fnum=1, target_size=(224,224), device=device)

    with torch.no_grad():
        teacher_frame_feature, _ = intern_model.streaming_vision_encoder.vit_lite.extract_features(next_frame_tensor.squeeze(1))

    # Permute from [B, T, C, H, W] to [B, C, T, H, W] to match training
    input_tensor = frame_tensor.permute(0, 2, 1, 3, 4)
    embed, hidden, spfs_info = intern_model.streaming_vision_encoder(input_tensor, prev_hidden_state = hidden, confidence_threshold = 0.9, max_consecutive_skips=6, teacher_frame_feature=teacher_frame_feature)

    embed = intern_model.vision_align(embed)

    # Now, the 'gt_cos' from spfs_info should be meaningful
    print(f"SPFS Info: {spfs_info}")
