import argparse
import glob
import logging
import os
import subprocess
import sys
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    parser = argparse.ArgumentParser(description="Run streaming benchmarks")
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
        "--output-graph",
        default="accuracy_graph.png",
        help="Path to output PNG graph of accuracy",
    )
    parser.add_argument(
        "--eval_again",
        action="store_true",
        help="Recalculate accuracy using existing predictions without running the model",
    )
    return parser.parse_args()


def checkout_branch(name: str):
    branch = branch_switch[name]
    subprocess.check_call(["git", "checkout", branch])


def find_streaming_checkpoints(base_dir: str, model_name: str) -> list[str]:
    """Return a sorted list of all streaming checkpoints for ``model_name``."""
    search_root = os.path.join(base_dir, model_name)
    pattern = os.path.join(search_root, "**", "mp_rank_00_model_states.pt")
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"Unable to find streaming checkpoint under {search_root}"
        )
    return matches


def find_closest(pred, truths):
    """Return the truth value closest to ``pred``."""
    if not truths:
        return pred
    return min(truths, key=lambda x: abs(x - pred))


def calculate_mse(preds_with_offset, data):
    """Calculate mean squared error for a list of predictions."""
    errors = []
    for idx, p in enumerate(preds_with_offset):
        truth_peaks = data[idx][2]
        if not truth_peaks:
            continue
        closest_t = find_closest(p, truth_peaks)
        errors.append((p - closest_t) ** 2)
    return np.mean(errors) if errors else float("inf")


def find_best_offset(preds, data, search_range=(-30, 30)):
    """Find the integer offset that minimises MSE."""
    best_offset = 0
    best_mse = float("inf")
    for off in range(search_range[0], search_range[1] + 1):
        shifted = [p + off for p in preds]
        mse = calculate_mse(shifted, data)
        if mse < best_mse:
            best_mse = mse
            best_offset = off
    return best_offset


def offset_predictions(preds, data):
    best = find_best_offset(preds, data)
    return [p + best for p in preds]


def compute_accuracy(preds: list[int], dataset: list) -> dict:
    """Return MAE percentages within various frame offsets using global offset."""
    thresholds = [2, 4, 8, 16, 32]
    preds_adj = offset_predictions(preds, dataset)
    totals = {t: 0 for t in thresholds}

    for pred, entry in zip(preds_adj, dataset):
        gt_frames = entry[2]
        if not gt_frames:
            continue
        diff = min(abs(pred - f) for f in gt_frames)
        for t in thresholds:
            if diff <= t:
                totals[t] += 1

    n = len(preds_adj)
    percentages = {f"within_{t}": totals[t] * 100.0 / n for t in thresholds}
    percentages["average"] = sum(percentages.values()) / len(thresholds)
    return percentages


def checkpoint_step(path: str) -> int:
    """Extract numeric step from checkpoint path."""
    dirname = os.path.basename(os.path.dirname(path))
    digits = re.findall(r"\d+", dirname)
    return int(digits[-1]) if digits else 0


def main():
    ensure_dependencies()
    args = parse_args()

    if args.config_name not in branch_switch:
        raise ValueError(f"Invalid config name: {args.config_name}")

    checkout_branch(args.config_name)

    sys.path.append(os.getcwd())

    from demo.config import Config, eval_dict_leaf
    from demo.utils import _frame_from_video
    if args.config_name in ["delta", "recycle", "delta_mamba"]:
        from demo.utils import retrieve_text_streaming, frames2tensor

    model_name = os.path.basename(os.path.normpath(args.config_dir))

    config = Config.from_file(os.path.join(args.config_dir, "config.py"))
    config = eval_dict_leaf(config)

    file_path = config.model.vision_ckpt_path
    clip_path = config.model.extra_ckpt_path
    streaming_vit_paths = find_streaming_checkpoints(args.config_dir, model_name)

    subprocess.call([
        "wget",
        "-q",
        "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt",
    ])

    print(f"{model_name} downloaded to: {file_path}")
    print(f"CLIP downloaded to: {clip_path}")
    print("Downloaded MobileCLIP to mobileclip_blt.pt")

    config.model.vision_ckpt_path = file_path
    if "delta" in args.config_name:
        config.model.mobileclip_ckpt_path = "mobileclip_blt.pt"
    else:
        config.model.text_ckpt_path = "mobileclip_blt.pt"
    config.model.extra_ckpt_path = clip_path

    from models.internvideo2_clip_small import InternVideo2_CLIP_small
    from collections import OrderedDict
    import torch
    import numpy as np
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from iv2_utils.iv2 import json_read, json_write

    if "photography-model" not in os.listdir('.'):
        subprocess.check_call(["git", "clone", "https://github.com/ruo2019/photography-model.git"])

    act75_data = json_read('photography-model/data/ACT75.json')

    def evaluate_checkpoint(checkpoint_path: str, dataset, eval_again: bool = False):
        out_dir = os.path.dirname(checkpoint_path)
        preds_file = os.path.join(out_dir, "t8.json")
        logits_file = os.path.join(out_dir, "logits-act75.json")
        acc_file = os.path.join(out_dir, "accuracy.json")

        if eval_again:
            if os.path.exists(preds_file):
                preds = json_read(preds_file)
                metrics = compute_accuracy(preds, dataset)
                json_write(metrics, acc_file)
                print(f"Recalculated metrics for {checkpoint_path}")
                return metrics.get("average", 0.0)
            else:
                print(f"Predictions not found for {checkpoint_path}")
                return 0.0
        if os.path.exists(preds_file) and os.path.exists(logits_file) and os.path.exists(acc_file):
            metrics = json_read(acc_file)
            print(f"Skipping {checkpoint_path}, results already exist")
            return metrics.get("average", 0.0)

        intern_model = InternVideo2_CLIP_small(config)
        intern_model.to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Loaded checkpoint from {checkpoint_path}. Keys: {checkpoint.keys()}")

        if "module" in checkpoint:
            state_dict_from_checkpoint = checkpoint["module"]
        elif "model" in checkpoint:
            state_dict_from_checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict_from_checkpoint = checkpoint["state_dict"]
        else:
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict_from_checkpoint = checkpoint
            else:
                raise KeyError(
                    f"Could not find model state_dict in checkpoint. Available keys: {checkpoint.keys()}"
                )

        new_state_dict = OrderedDict()
        is_ddp_inner = any(k.startswith("module.") for k in state_dict_from_checkpoint)

        if is_ddp_inner:
            for k, v in state_dict_from_checkpoint.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            state_dict_to_load = new_state_dict
        else:
            state_dict_to_load = state_dict_from_checkpoint

        missing_keys, unexpected_keys = intern_model.load_state_dict(
            state_dict_to_load, strict=False
        )

        if unexpected_keys:
            print("\nERROR: Unexpected keys in state_dict:")
            for k in unexpected_keys:
                print(f"  - {k}")

        if missing_keys:
            print("\nINFO: Missing keys in state_dict:")
            for k in missing_keys[:5]:
                print(f"  - {k}")
            if len(missing_keys) > 5:
                print(f"  - ... and {len(missing_keys) - 5} more")

        print("\nModel state_dict loaded.")

        intern_model.eval()
        print("Model set to evaluation mode.")

        logging.basicConfig(
            format='%(asctime)s %(levelname)s: %(message)s',
            level=logging.DEBUG,
            datefmt='%I:%M:%S'
        )
        logging.debug("Logging working!")

        from tqdm import tqdm

        logits = []
        preds = []

        size_t = 224

        intern_model.to(device)

        for video_path, phrase, frames in dataset:
            frames = [x for x in _frame_from_video(cv2.VideoCapture('photography-model/' + video_path))]

            logit_curr = []
            pbar = tqdm(range(len(frames) - 8))

            curr_hidden_state = intern_model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)

            for frame_idx in range(7):
                initial_frame_mc = frames2tensor(
                    [frames[frame_idx]],
                    fnum=1,
                    target_size=(size_t, size_t),
                    device=device
                ).squeeze(0).to(device)

                _, curr_hidden_state = intern_model.streaming_vision_encoder(initial_frame_mc, curr_hidden_state)

            for j in pbar:
                texts, probs, curr_hidden_state = retrieve_text_streaming(
                    frames[j+8],
                    [phrase],
                    intern_model,
                    curr_hidden_state,
                    topk=1,
                    config=config
                )
                logit_curr.append(probs.item())
                if len(logit_curr) > 0:
                    pbar.set_description(str(np.argmax(logit_curr) + 1))

            preds.append(np.argmax(logit_curr) + 1)
            logits.append(list(zip(logit_curr, range(1, len(logit_curr) + 1))))

        preds = [int(x) for x in preds]
        logits_v2 = [[(float(l[0]), l[1]) for l in x] for x in logits]

        out_dir = os.path.dirname(checkpoint_path)
        json_write(preds, os.path.join(out_dir, 't8.json'))
        json_write(logits_v2, os.path.join(out_dir, 'logits-act75.json'))

        metrics = compute_accuracy(preds, dataset)
        json_write(metrics, os.path.join(out_dir, 'accuracy.json'))

        return metrics["average"]

    results = {}
    for checkpoint_path in streaming_vit_paths:
        avg = evaluate_checkpoint(checkpoint_path, act75_data, eval_again=args.eval_again)
        results[checkpoint_path] = avg

    if results:
        best_ckpt, best_score = max(results.items(), key=lambda x: x[1])
        print(f"Best checkpoint: {best_ckpt} (average {best_score:.2f})")

        steps = []
        accs = []
        for ckpt, score in sorted(results.items(), key=lambda x: checkpoint_step(x[0])):
            steps.append(checkpoint_step(ckpt))
            accs.append(score)

        plt.figure()
        plt.plot(steps, accs, marker="o")
        plt.xlabel("checkpoint step")
        plt.ylabel("mean accuracy")
        plt.title("Mean accuracy over time")
        plt.grid(True)
        plt.savefig(args.output_graph)
        print(f"Saved accuracy graph to {args.output_graph}")


if __name__ == "__main__":
    main()
