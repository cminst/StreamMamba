import os
import json
import logging
from os.path import join

import torch
from torch.utils.data import Dataset
from decord import VideoReader

from dataset.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def _read_video_full(video_path):
    """Read an entire video with decord and return frames and fps."""
    num_threads = 1 if video_path.endswith('.webm') else 0
    vr = VideoReader(video_path, num_threads=num_threads)
    vlen = len(vr)
    fps = vr.get_avg_fps()
    frames = vr.get_batch(range(vlen))  # (T, H, W, C), uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
    return frames, float(fps)


class LocalizationDataset(BaseDataset):
    media_type = "video"

    def __init__(self, ann_file, transform, video_reader_type="decord"):
        super().__init__()
        if isinstance(ann_file, dict):
            ann_path = ann_file["anno_path"]
            self.data_root = ann_file.get("data_root", "")
            self.data_root_prefix = ann_file.get("data_root_prefix", "")
        else:
            ann_path = ann_file
            self.data_root = ""
            self.data_root_prefix = ""
        logger.info(f"Loading annotations from {ann_path}")
        with open(ann_path, "r") as f:
            self.anno = json.load(f)
        self.transform = transform
        self.video_reader_type = video_reader_type

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        item = self.anno[index]
        caption = item["caption"]
        video_path = item["video"]
        if self.data_root:
            video_path = join(self.data_root, video_path)
        if self.data_root_prefix:
            video_path = self.data_root_prefix + video_path
        start_time = float(item.get("start_time", 0.0))
        end_time = float(item.get("end_time", 0.0))

        frames, fps = _read_video_full(video_path)
        frames = self.transform(frames)

        return frames, caption, start_time, end_time, fps
