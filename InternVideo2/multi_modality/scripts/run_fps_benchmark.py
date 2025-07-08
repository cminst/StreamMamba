import argparse
import glob
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    parser = argparse.ArgumentParser(description="Measure streaming inference FPS")
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
        "--branch",
        default=None,
        help="Git branch to checkout before evaluation",
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Directory that contains the checkpoint file",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Run in no-stream mode without loading checkpoint",
    )
    parser.add_argument(
        "--output-json",
        default="fps_results.json",
        help="Path to output JSON with FPS results",
    )
    parser.add_argument(
        "--output-graph",
        default="fps_graph.png",
        help="Path to output PNG graph of FPS",
    )
    return parser.parse_args()


def find_checkpoint(ckpt_dir: str) -> str:
    pattern = os.path.join(ckpt_dir, "**", "mp_rank_00_model_states.pt")
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"Unable to find checkpoint under {ckpt_dir}")
    return matches[0]


def main():
    ensure_dependencies()
    args = parse_args()

    if args.branch:
        subprocess.check_call(["git", "checkout", args.branch])

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

    if "delta" not in args.config_name:
        config.model.text_ckpt_path = config.model.mobileclip_ckpt_path

    intern_model = InternVideo2_CLIP_small(config)
    intern_model.to(device)

    intern_model.eval()

    act75_data = json_read('photography-model/data/ACT75.json')

    results = []
    size_t = config.get('size_t', 224)

    for video_path, _, _ in act75_data:
        cap = cv2.VideoCapture('photography-model/' + video_path)
        frames = [x for x in _frame_from_video(cap)]
        if not frames:
            continue
        h, w = frames[0].shape[:2]
        pixels = w * h
        total_time = 0.0

        if not args.no_stream:
            hidden = intern_model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)
            for f in frames:
                tensor = frames2tensor([f], fnum=1, target_size=(size_t, size_t), device=device).squeeze(0)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                _, hidden = intern_model.streaming_vision_encoder(tensor, hidden)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
                total_time += end - start
            fps = len(frames) / total_time if total_time > 0 else 0.0
        else:
            num_frames = config.get('num_frames', 8)
            num_chunks = len(frames) // num_frames
            for i in range(num_chunks):
                chunk = frames[i * num_frames:(i + 1) * num_frames]
                tensor = frames2tensor(chunk, fnum=num_frames, target_size=(size_t, size_t), device=device)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                intern_model.encode_vision(tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
                total_time += end - start
                
            fps = num_chunks / total_time if total_time > 0 and num_chunks > 0 else 0.0

        results.append({"video": video_path, "resolution": f"{w}x{h}", "pixels": pixels, "fps": fps})

    json_write(results, args.output_json)

    results_sorted = sorted(results, key=lambda r: r["pixels"])
    x = [r["pixels"] for r in results_sorted]
    y = [r["fps"] for r in results_sorted]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("pixels (w*h)")
    plt.ylabel("fps")
    plt.title("Streaming FPS vs image size")
    plt.grid(True)
    plt.savefig(args.output_graph)
    print(f"Saved FPS results to {args.output_json}")
    print(f"Saved FPS graph to {args.output_graph}")


if __name__ == "__main__":
    main()
