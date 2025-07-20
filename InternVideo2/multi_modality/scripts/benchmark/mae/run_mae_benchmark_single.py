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
from huggingface_hub import hf_hub_download
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
    parser = argparse.ArgumentParser(description="Run streaming benchmarks for a single model")
    parser.add_argument(
        "--config-name",
        default="delta",
        help="Configuration name",
    )
    parser.add_argument(
        "--use-film",
        action="store_true",
        help="Indicate that the model uses FiLM conditioning",
    )
    parser.add_argument(
        "--output-graph",
        default="accuracy_graph_single.png",
        help="Path to output PNG graph of accuracy",
    )
    parser.add_argument(
        "--output-json",
        default="mae_results_single.json",
        help="Path to output JSON with MAE results",
    )
    parser.add_argument(
        "--mamba-weights",
        default=None,
        help="Path to mamba_mobileclip_ckpt.pt. If not provided, will download from HF.",
    )
    parser.add_argument(
        "--spfs-weights",
        default=None,
        help="Path to spfs_ckpt.pt. If not provided, will download from HF.",
    )
    return parser.parse_args()


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


def main():
    ensure_dependencies()
    args = parse_args()

    sys.path.append(os.getcwd())

    from demo.config import Config, eval_dict_leaf
    from demo.utils import _frame_from_video
    from demo.utils import retrieve_text_streaming, frames2tensor

    config_path = os.path.join(os.getcwd(), "InternVideo2/multi_modality/configs", "med_config_fusion.json") # Assuming a default config for single model
    config = Config.from_file(config_path)
    config = eval_dict_leaf(config)

    if "delta" not in args.config_name:
        config.model.text_ckpt_path = config.model.mobileclip_ckpt_path

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

    if args.mamba_weights is None:
        print("Downloading mamba_mobileclip_ckpt.pt from Hugging Face...")
        args.mamba_weights = hf_hub_download(repo_id="qingy2024/InternVideo2-B14", filename="mamba_mobileclip_ckpt.pt")
    if args.spfs_weights is None:
        print("Downloading spfs_ckpt.pt from Hugging Face...")
        args.spfs_weights = hf_hub_download(repo_id="qingy2024/InternVideo2-B14", filename="spfs_ckpt.pt")

    intern_model = InternVideo2_CLIP_small(config)
    intern_model.to(device)

    # Load checkpoints
    mamba_ckpt = torch.load(args.mamba_weights, map_location=device)
    processed_mamba = process_state_dict(mamba_ckpt)

    spfs_ckpt = torch.load(args.spfs_weights, map_location=device, weights_only = False)
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

    for video_path, phrase, frames in act75_data:
        frames = [x for x in _frame_from_video(cv2.VideoCapture('photography-model/' + video_path))]

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

        for frame_idx in range(7):
            initial_frame_mc = frames2tensor(
                [frames[frame_idx]],
                fnum=1,
                target_size=(size_t, size_t),
                device=device
            ).squeeze(0).to(device)

            _, curr_hidden_state = intern_model.streaming_vision_encoder(
                initial_frame_mc,
                curr_hidden_state,
                gamma,
                beta,
            )

        for j in pbar:
            texts, probs, curr_hidden_state = retrieve_text_streaming(
                frames[j+8],
                [phrase],
                intern_model,
                curr_hidden_state,
                topk=1,
                config=config,
                gamma=gamma,
                beta=beta,
            )
            logit_curr.append(probs.item())
            if len(logit_curr) > 0:
                pbar.set_description(str(np.argmax(logit_curr) + 1))

        preds.append(np.argmax(logit_curr) + 1)
        logits.append(list(zip(logit_curr, range(1, len(logit_curr) + 1))))

    preds = [int(x) for x in preds]
    logits_v2 = [[(float(l[0]), l[1]) for l in x] for x in logits]

    json_write(preds, args.output_json)

    metrics = compute_accuracy(preds, act75_data)
    json_write(metrics, args.output_json.replace(".json", "_metrics.json"))

    print(f"Saved MAE results to {args.output_json}")
    print(f"Saved MAE metrics to {args.output_json.replace('.json', '_metrics.json')}")

    # Plotting is not directly applicable for a single run, but we can plot the distribution of errors if needed.
    # For now, just print the average accuracy.
    print(f"Average MAE accuracy: {metrics['average']:.2f}")


if __name__ == "__main__":
    main()
