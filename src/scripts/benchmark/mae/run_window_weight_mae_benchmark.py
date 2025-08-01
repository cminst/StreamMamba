import argparse
import os
import sys
import subprocess
from typing import List

try:
    import numpy as np
except Exception:  # pragma: no cover - install at runtime
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy"])
    import numpy as np

import torch
from tqdm import tqdm


def ensure_dependencies():
    try:
        import einops  # noqa: F401
    except Exception:
        print("Installing dependencies...")
        packages = [
            "einops",
            "peft",
            "open_clip_torch",
            "protobuf",
            "sentencepiece",
            "iv2-utils",
            "matplotlib",
            "huggingface_hub",
            "tabulate",
            "tqdm",
        ]
        # Using tqdm for progress bar during installation
        for package in tqdm(packages, desc="Installing packages"):
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                package,
            ])
    print("Dependencies installed/verified")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark StreamMamba with different center frame weights",
    )
    parser.add_argument(
        "config_dir",
        help="Path to training config directory, e.g. scripts/pretraining/clip/B14",
    )
    parser.add_argument(
        "--output-json",
        default="center_weight_results.json",
        help="Path to output JSON with results",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint file. If not provided, will download from HF.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="spfs_r64/ckpt_step_24500.pt",
        help="Checkpoint filename within the HF repo",
    )
    parser.add_argument(
        "--hf-repo",
        default="qingy2024/InternVideo2-B14",
        help="HuggingFace repo to download checkpoint from",
    )
    parser.add_argument(
        "--center-weights",
        default="0.3333,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma separated list of center weights to evaluate (between 0 and 1)",
    )
    return parser.parse_args()


def find_closest(pred, truths):
    if not truths:
        return pred
    return min(truths, key=lambda x: abs(x - pred))


def calculate_mse(preds_with_offset, data):
    errors = []
    for idx, p in enumerate(preds_with_offset):
        truth_peaks = data[idx][2]
        if not truth_peaks:
            continue
        closest_t = find_closest(p, truth_peaks)
        errors.append((p - closest_t) ** 2)
    return np.mean(errors) if errors else float("inf")


def find_best_offset(preds, data, search_range=(-30, 30)):
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


def compute_accuracy(preds: List[int], dataset: List) -> dict:
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


def streammamba_predict(frames, phrase, model, device, size_t, center_weight):
    from demo.utils import frames2tensor

    hidden = model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)
    embeddings = []
    logits = []
    text_feat = model.get_txt_feat(phrase)
    window_size = 10  # Fixed window size

    for frame in frames:
        tensor = frames2tensor([frame], fnum=1, target_size=(size_t, size_t), device=device)
        tensor = tensor.squeeze(0)
        emb, hidden, _ = model.get_streaming_vid_feat(
            tensor,
            hidden,
            confidence_threshold=1.0,
            max_consecutive_skips=0,
        )
        embeddings.append(emb)

        if len(embeddings) <= 7:
            logits.append(0.0)
            continue

        k = max(1, window_size - 7)  # k = 3 for window_size = 10
        if k == 1:
            use_emb = embeddings[-1]
        else:
            # Weighted averaging: center gets center_weight, others split (1 - center_weight) evenly
            recent_embs = embeddings[-k:]  # Last k embeddings
            if len(recent_embs) == 3:
                # Center is the middle one (index 1), outer ones are indices 0 and 2
                outer_weight = (1 - center_weight) / 2
                weights = torch.tensor([outer_weight, center_weight, outer_weight], device=device)
                weights = weights / weights.sum()  # Normalize to ensure they sum to 1

                weighted_embs = torch.stack([w * emb for w, emb in zip(weights, recent_embs)])
                use_emb = weighted_embs.sum(dim=0)
            else:
                # Fallback to simple mean if we don't have exactly 3 embeddings
                use_emb = torch.stack(recent_embs).mean(dim=0)

        probs, _ = model.predict_label(use_emb, text_feat, top=1)
        logits.append(probs.item())

    return int(np.argmax(logits) + 1)


def main():
    ensure_dependencies()
    args = parse_args()

    sys.path.append(os.getcwd())

    from demo.config import Config, eval_dict_leaf
    from demo.utils import _frame_from_video
    from models.internvideo2_clip_small import InternVideo2_CLIP_small
    from huggingface_hub import hf_hub_download
    from iv2_utils.iv2 import json_read, json_write
    from tabulate import tabulate
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "photography-model" not in os.listdir("."):
        subprocess.check_call([
            "git",
            "clone",
            "https://github.com/ruo2019/photography-model.git",
        ])

    act75_data = json_read("photography-model/data/ACT75.json")

    config_path = os.path.join(args.config_dir, "config.py")
    config = Config.from_file(config_path)
    config = eval_dict_leaf(config)

    model = InternVideo2_CLIP_small(config)
    model.to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.checkpoint_file)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model") or ckpt.get("module")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    size_t = config.get("size_t", 224)

    center_weights = [float(x) for x in args.center_weights.split(",") if x.strip()]

    # Validate center weights
    for weight in center_weights:
        if not (0 <= weight <= 1):
            raise ValueError(f"Center weight {weight} must be between 0 and 1")

    results = {}

    # Progress bar for center weights
    for center_weight in tqdm(center_weights, desc="Processing center weights"):
        preds_stream = []
        # Progress bar for video processing
        for video_path, phrase, _ in tqdm(act75_data, desc=f"Processing videos for center weight {center_weight:.4f}", leave=False):
            frames = [
                f
                for f in _frame_from_video(
                    cv2.VideoCapture(os.path.join("photography-model", video_path))
                )
            ]
            preds_stream.append(
                streammamba_predict(frames, phrase, model, device, size_t, center_weight)
            )

        metrics_stream = compute_accuracy(preds_stream, act75_data)
        results[center_weight] = {
            "streammamba": metrics_stream,
        }

    # Create table with center weights and accuracies
    table = [
        (center_weight, results[center_weight]["streammamba"]["average"])
        for center_weight in center_weights
    ]
    print("\nResults for different center frame weights (window size = 10):")
    print(tabulate(table, headers=["Center Weight", "StreamMamba Accuracy"], floatfmt=".4f"))

    json_write(results, args.output_json)
    print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
