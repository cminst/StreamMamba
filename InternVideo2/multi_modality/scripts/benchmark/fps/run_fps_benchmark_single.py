import argparse
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download


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
        "--checkpoint",
        default=None,
        help="Path to a full SPFS checkpoint. If not provided, will download from HF.",
    )
    parser.add_argument(
        "--hf-repo",
        default="qingy2024/InternVideo2-B14",
        help="HuggingFace repo from which to download the checkpoint",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="spfs_r64/ckpt_step_24500.pt",
        help="Checkpoint filename within the HF repo",
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
    parser.add_argument(
        "--no-spfs",
        action="store_true",
        help="Disable SPFS (use plain model)",
    )
    return parser.parse_args()


def main():
    ensure_dependencies()
    args = parse_args()

    # Determine mode
    rnn_type = 'mamba' if args.no_spfs else 'mamba_spfs'
    mode = "non-SPFS" if args.no_spfs else "SPFS"
    print(f"Running in {mode} mode.")

    # Folder setup
    if args.no_spfs:
        folder_name = f"results_{rnn_type}"
    else:
        folder_name = f"results_{rnn_type}_ct_{args.confidence_threshold}_mcs_{args.max_consecutive_skips}"
    os.makedirs(folder_name, exist_ok=True)

    fps_json_path = os.path.join(folder_name, os.path.basename(args.output_json))
    fps_graph_path = os.path.join(folder_name, os.path.basename(args.output_graph))

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

    # Set rnn_type dynamically
    config.model.streaming_vision_encoder.rnn_type = rnn_type

    intern_model = InternVideo2_CLIP_small(config)
    intern_model.to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        print(f"Downloading {args.checkpoint_file} from Hugging Face...")
        ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.checkpoint_file)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    missing_keys, unexpected_keys = intern_model.load_state_dict(state_dict, strict=False)

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

    print("\nCheckpoint loaded successfully.")

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

        # Warm-up with first seven frames
        for i in range(7):
            f = frames[i]
            tensor = frames2tensor([f], fnum=1, target_size=(size_t, size_t), device=device).squeeze(0)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _, hidden, _ = intern_model.encode_streaming_vision(
                tensor,
                hidden,
                confidence_threshold=1.0,
                max_consecutive_skips=0
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            total_time += end - start

        # Process remaining frames
        for f in frames[7:]:
            tensor = frames2tensor([f], fnum=1, target_size=(size_t, size_t), device=device).squeeze(0)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()

            if args.no_spfs:
                _, hidden, _ = intern_model.encode_streaming_vision(
                    tensor,
                    hidden,
                    confidence_threshold=1.0,
                    max_consecutive_skips=0
                )
            else:
                _, hidden, spfs_info = intern_model.encode_streaming_vision(
                    tensor,
                    hidden,
                    confidence_threshold=args.confidence_threshold,
                    max_consecutive_skips=args.max_consecutive_skips
                )
                if spfs_info.skipped:
                    skipped_frames += 1

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            total_time += end - start

        total_frames += len(frames)
        total_skipped_frames += skipped_frames
        fps = len(frames) / total_time if total_time > 0 else 0.0
        results.append({
            "video": video_path,
            "resolution": f"{w}x{h}",
            "pixels": pixels,
            "fps": fps,
            "skipped_frames": skipped_frames
        })

    results_sorted = sorted(results, key=lambda r: r["pixels"])
    x = [r["pixels"] for r in results_sorted]
    y = [r["fps"] for r in results_sorted]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("pixels (w*h)")
    plt.ylabel("fps")
    plt.title("Streaming FPS vs image size (with SPFS)" if not args.no_spfs else "Streaming FPS vs image size")
    plt.grid(True)

    from iv2_utils.iv2 import json_write
    json_write(results, fps_json_path)
    plt.savefig(fps_graph_path)

    print(f"Saved FPS results to {fps_json_path}")
    print(f"Saved FPS graph to {fps_graph_path}")

    skip_percentage = (total_skipped_frames / total_frames) * 100 if total_frames > 0 else 0
    avg_fps = sum(r['fps'] for r in results) / len(results) if results else 0
    print(f"Total frames skipped: {total_skipped_frames}")
    print(f"Percentage of frames skipped: {skip_percentage:.2f}%")
    print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
