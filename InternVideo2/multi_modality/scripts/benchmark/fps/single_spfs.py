import argparse
import glob
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
from utils.basic_utils import merge_state_dicts, process_state_dict

def ensure_dependencies():
    try:
        import einops  # noqa: F401
    except Exception:
        print("Installing...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "einops",
            "peft",
            "open_clip_torch",
            "protobuf",
            "sentencepiece",
            "iv2-utils",
            "matplotlib",
            "huggingface_hub",
        ])
    print("Installed packages")


def parse_args():
    parser = argparse.ArgumentParser(description="Measure streaming inference FPS with SPFS")
    parser.add_argument(
        "config_dir",
        help="Path to training config directory, e.g. scripts/pretraining/clip/B14",
    )
    parser.add_argument(
        "--config-name",
        default="delta",
        help="Configuration name",
    )
    parser.add_argument(
        "--hf-repo",
        default="qingy2024/InternVideo2-B14",
        help="HuggingFace repository name to load checkpoint from",
    )
    parser.add_argument(
        "--mamba-weights",
        default=None,
        help="Path to the Mamba weights checkpoint file. If not specified, downloads from Hugging Face.",
    )
    parser.add_argument(
        "--spfs-weights",
        default=None,
        help="Path to the SPFS prediction/confidence head weights checkpoint file. If not specified, downloads from Hugging Face.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for skipping frames",
    )
    parser.add_argument(
        "--max-consecutive-skips",
        type=int,
        default=4,
        help="Maximum number of consecutive frames to skip",
    )
    parser.add_argument(
        "--output-json",
        default="fps_results_spfs.json",
        help="Path to output JSON with FPS results",
    )
    parser.add_argument(
        "--output-graph",
        default="fps_graph_spfs.png",
        help="Path to output PNG graph of FPS",
    )
    return parser.parse_args()


def main():
    ensure_dependencies()
    args = parse_args()

    sys.path.append(os.getcwd())

    from demo.config import Config, eval_dict_leaf
    from demo.utils import _frame_from_video, frames2tensor
    from models.internvideo2_clip_small import InternVideo2_CLIP_small
    from iv2_utils.iv2 import json_read, json_write
    import torch
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "photography-model" not in os.listdir('.'):
        subprocess.check_call(["git", "clone", "https://github.com/ruo2019/photography-model.git"])

    config = Config.from_file(os.path.join(args.config_dir, "config.py"))
    config = eval_dict_leaf(config)

    # Set rnn_type to mamba_spfs
    config.model.streaming_vision_encoder.rnn_type = 'mamba_spfs'

    intern_model = InternVideo2_CLIP_small(config)
    intern_model.to(device)

    from huggingface_hub import hf_hub_download
    from collections import OrderedDict

    if args.mamba_weights:
        mamba_checkpoint_path = args.mamba_weights
    else:
        print("Downloading Mamba checkpoint from Hugging Face Hub...")
        mamba_checkpoint_path = hf_hub_download(repo_id=args.hf_repo, filename="mamba_mobileclip_ckpt.pt")
        print(f"Downloaded Mamba checkpoint to {mamba_checkpoint_path}")

    if args.spfs_weights:
        spfs_checkpoint_path = args.spfs_weights
    else:
        print("Downloading SPFS checkpoint from Hugging Face Hub...")
        spfs_checkpoint_path = hf_hub_download(repo_id=args.hf_repo, filename="spfs_ckpt.pt")
        print(f"Downloaded SPFS checkpoint to {spfs_checkpoint_path}")

    # Load checkpoints
    mamba_ckpt = torch.load(mamba_checkpoint_path, map_location=device)
    processed_mamba = process_state_dict(mamba_ckpt)

    spfs_ckpt = torch.load(spfs_checkpoint_path, map_location=device)
    processed_spfs = process_state_dict(spfs_ckpt)

    merged_state_dict = merge_state_dicts([processed_mamba, processed_spfs], override=True)

    # Load the merged state dict into the model
    missing_keys, unexpected_keys = intern_model.load_state_dict(merged_state_dict, strict=False)

    if unexpected_keys:
        print("\nERROR: Unexpected keys in merged state_dict:")
        for k in unexpected_keys:
            print(f"  - {k}")

    if missing_keys:
        print("\nINFO: Missing keys in merged state_dict:")
        for k in missing_keys[:5]:
            print(f"  - {k}")
        if len(missing_keys) > 5:
            print(f"  - ... and {len(missing_keys) - 5} more")

    print("\nMerged state_dict loaded successfully.")

    intern_model.eval()

    act75_data = json_read('photography-model/data/ACT75.json')

    results = []
    total_skipped_frames = 0
    total_frames = 0
    size_t = config.get('size_t', 224)

    for video_path, _, _ in act75_data:
        cap = cv2.VideoCapture('photography-model/' + video_path)
        frames = [x for x in _frame_from_video(cap)]
        if not frames:
            continue
        h, w = frames[0].shape[:2]
        pixels = w * h
        total_time = 0.0
        skipped_frames = 0

        hidden = intern_model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)
        # Don't use SPFS for the first 2 frames
        for i in range(2):
            f = frames[i]
            tensor = frames2tensor([f], fnum=1, target_size=(size_t, size_t), device=device).squeeze(0)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _, hidden, _ = intern_model.encode_streaming_vision(
                tensor,
                hidden,
                confidence_threshold=1.0,  # Force no skip
                max_consecutive_skips=0
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            total_time += end - start

        for f in frames[2:]:
            tensor = frames2tensor([f], fnum=1, target_size=(size_t, size_t), device=device).squeeze(0)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _, hidden, skipped = intern_model.encode_streaming_vision(
                tensor,
                hidden,
                confidence_threshold=args.confidence_threshold,
                max_consecutive_skips=args.max_consecutive_skips
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            total_time += end - start
            if skipped:
                skipped_frames += 1

        total_frames += len(frames)
        total_skipped_frames += skipped_frames
        fps = len(frames) / total_time if total_time > 0 else 0.0
        results.append({"video": video_path, "resolution": f"{w}x{h}", "pixels": pixels, "fps": fps, "skipped_frames": skipped_frames})

    json_write(results, args.output_json)

    results_sorted = sorted(results, key=lambda r: r["pixels"])
    x = [r["pixels"] for r in results_sorted]
    y = [r["fps"] for r in results_sorted]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("pixels (w*h)")
    plt.ylabel("fps")
    plt.title("Streaming FPS vs image size (with SPFS)")
    plt.grid(True)
    plt.savefig(args.output_graph)
    print(f"Saved FPS results to {args.output_json}")
    print(f"Saved FPS graph to {args.output_graph}")

    skip_percentage = (total_skipped_frames / total_frames) * 100 if total_frames > 0 else 0
    avg_fps = sum(r['fps'] for r in results) / len(results) if results else 0
    print(f"Total frames skipped: {total_skipped_frames}")
    print(f"Percentage of frames skipped: {skip_percentage:.2f}%")
    print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
