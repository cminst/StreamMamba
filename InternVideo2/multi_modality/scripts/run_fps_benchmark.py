import argparse
import glob
import os
import subprocess
import sys
import time
import json

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


branch_switch = {
    "patch_cache": "benchmark-cache2",
    "main": "benchmark-main",
    "adapter": "adapter",
    "window_v3": "window_v3",
    "recycle": "recycle",
    "delta": "delta",
    "delta_mamba": "delta_mamba",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Measure streaming inference FPS")
    parser.add_argument(
        "config_dir",
        help="Path to training config directory, e.g. scripts/pretraining/clip/B14",
    )
    parser.add_argument(
        "--config-name",
        default="delta",
        choices=list(branch_switch.keys()),
        help="Configuration name",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory that contains the checkpoint file",
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


def checkout_branch(name: str):
    branch = branch_switch[name]
    subprocess.check_call(["git", "checkout", branch])


def find_checkpoint(ckpt_dir: str) -> str:
    pattern = os.path.join(ckpt_dir, "**", "mp_rank_00_model_states.pt")
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"Unable to find checkpoint under {ckpt_dir}")
    return matches[0]


def main():
    ensure_dependencies()
    args = parse_args()

    if args.config_name not in branch_switch:
        raise ValueError(f"Invalid config name: {args.config_name}")

    checkout_branch(args.config_name)

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

    ckpt_path = find_checkpoint(args.checkpoint_dir)

    file_path = config.model.vision_ckpt_path
    clip_path = config.model.extra_ckpt_path

    subprocess.call([
        "wget",
        "-q",
        "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt",
    ])

    config.model.vision_ckpt_path = file_path
    if "delta" in args.config_name:
        config.model.mobileclip_ckpt_path = "mobileclip_blt.pt"
    else:
        config.model.text_ckpt_path = "mobileclip_blt.pt"
    config.model.extra_ckpt_path = clip_path

    intern_model = InternVideo2_CLIP_small(config)
    intern_model.to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)

    if "module" in checkpoint:
        state_dict = checkpoint["module"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
        else:
            raise KeyError(f"Could not find model state_dict in checkpoint. Keys: {checkpoint.keys()}")

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    intern_model.load_state_dict(new_state_dict, strict=False)
    intern_model.eval()

    act75_data = json_read('photography-model/data/ACT75.json')

    results = []
    size_t = 224
    for video_path, _, _ in act75_data:
        cap = cv2.VideoCapture('photography-model/' + video_path)
        frames = [x for x in _frame_from_video(cap)]
        if not frames:
            continue
        h, w = frames[0].shape[:2]
        pixels = w * h
        hidden = intern_model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)
        start = time.time()
        for f in frames:
            tensor = frames2tensor([f], fnum=1, target_size=(size_t, size_t), device=device).squeeze(0)
            _, hidden = intern_model.streaming_vision_encoder(tensor, hidden)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.time() - start
        fps = len(frames) / elapsed if elapsed > 0 else 0.0
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
