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


def parse_args():
    parser = argparse.ArgumentParser(description="Run streaming benchmarks with SPFS")
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
        "--mamba-weights",
        required=True,
        help="Path to the Mamba weights checkpoint file",
    )
    parser.add_argument(
        "--spfs-weights",
        required=True,
        help="Path to the SPFS prediction/confidence head weights checkpoint file",
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
        "--use-film",
        action="store_true",
        help="Indicate that the model uses FiLM conditioning",
    )
    parser.add_argument(
        "--output-graph",
        default="accuracy_graph_spfs.png",
        help="Path to output PNG graph of accuracy",
    )
    parser.add_argument(
        "--eval_again",
        action="store_true",
        help="Recalculate accuracy using existing predictions without running the model",
    )
    return parser.parse_args()


def checkout_branch(name: str | None):
    if name:
        subprocess.check_call(["git", "checkout", name])


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

    checkout_branch(args.branch)

    sys.path.append(os.getcwd())

    from demo.config import Config, eval_dict_leaf
    from demo.utils import _frame_from_video
    if args.config_name in ["delta", "recycle", "delta_mamba"]:
        from demo.utils import retrieve_text_streaming, frames2tensor

    model_name = os.path.basename(os.path.normpath(args.config_dir))

    config = Config.from_file(os.path.join(args.config_dir, "config.py"))
    config = eval_dict_leaf(config)

    # Set rnn_type to mamba_spfs
    config.model.streaming_vision_encoder.rnn_type = 'mamba_spfs'

    if "delta" not in args.config_name:
        config.model.text_ckpt_path = config.model.mobileclip_ckpt_path

    streaming_vit_paths = find_streaming_checkpoints(args.config_dir, model_name)

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
        preds_file = os.path.join(out_dir, "t8_spfs.json")
        logits_file = os.path.join(out_dir, "logits-act75_spfs.json")
        acc_file = os.path.join(out_dir, "accuracy_spfs.json")

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

        # Load Mamba weights
        mamba_ckpt = torch.load(args.mamba_weights, map_location=device)
        intern_model.streaming_vision_encoder.rnn.load_state_dict(mamba_ckpt, strict=False)

        # Load SPFS weights
        spfs_ckpt = torch.load(args.spfs_weights, map_location=device)
        intern_model.streaming_vision_encoder.rnn.load_state_dict(spfs_ckpt, strict=False)

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
        total_skipped_frames = 0
        total_frames = 0

        size_t = 224

        intern_model.to(device)

        for video_path, phrase, frames in dataset:
            frames = [x for x in _frame_from_video(cv2.VideoCapture('photography-model/' + video_path))]
            total_frames += len(frames)
            skipped_frames = 0

            logit_curr = []
            pbar = tqdm(range(len(frames) - 8))

            gamma = beta = None
            if args.use_film and getattr(intern_model.streaming_vision_encoder, "rnn_type", "") == "cross_mamba_film":
                text_input = intern_model.tokenizer(
                    phrase,
                    padding="max_length",
                    truncation=True,
                    max_length=config.max_txt_l,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    prompt_vec = intern_model.encode_text(text_input)
                gamma, beta = intern_model.streaming_vision_encoder.rnn.prepare_prompt(prompt_vec)

            curr_hidden_state = intern_model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)

            # Don't use SPFS for the first 2 frames
            for frame_idx in range(2):
                initial_frame_mc = frames2tensor(
                    [frames[frame_idx]],
                    fnum=1,
                    target_size=(size_t, size_t),
                    device=device
                ).squeeze(0).to(device)

                _, curr_hidden_state, _ = intern_model.encode_streaming_vision(
                    initial_frame_mc,
                    curr_hidden_state,
                    confidence_threshold=1.0,  # Force no skip
                    max_consecutive_skips=0,
                    gamma=gamma,
                    beta=beta,
                )

            for j in pbar:
                texts, probs, curr_hidden_state, skipped = retrieve_text_streaming(
                    frames[j+8],
                    [phrase],
                    intern_model,
                    prev_hidden_state = curr_hidden_state,
                    topk=1,
                    config=config,
                    confidence_threshold=args.confidence_threshold,
                    max_consecutive_skips=args.max_consecutive_skips,
                    gamma=gamma,
                    beta=beta,
                )
                if skipped:
                    skipped_frames += 1
                logit_curr.append(probs.item())
                if len(logit_curr) > 0:
                    pbar.set_description(str(np.argmax(logit_curr) + 1))

            preds.append(np.argmax(logit_curr) + 1)
            logits.append(list(zip(logit_curr, range(1, len(logit_curr) + 1))))
            total_skipped_frames += skipped_frames

        preds = [int(x) for x in preds]
        logits_v2 = [[(float(l[0]), l[1]) for l in x] for x in logits]

        out_dir = os.path.dirname(checkpoint_path)
        json_write(preds, os.path.join(out_dir, 't8_spfs.json'))
        json_write(logits_v2, os.path.join(out_dir, 'logits-act75_spfs.json'))

        metrics = compute_accuracy(preds, dataset)
        json_write(metrics, os.path.join(out_dir, 'accuracy_spfs.json'))

        skip_percentage = (total_skipped_frames / total_frames) * 100 if total_frames > 0 else 0
        print(f"Total frames skipped: {total_skipped_frames}")
        print(f"Percentage of frames skipped: {skip_percentage:.2f}%")

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
        plt.title("Mean accuracy over time (with SPFS)")
        plt.grid(True)
        plt.savefig(args.output_graph)
        print(f"Saved accuracy graph to {args.output_graph}")


if __name__ == "__main__":
    main()
