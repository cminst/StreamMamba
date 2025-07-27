import argparse
import os
import sys
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download

def ensure_dependencies():
    try:
        import einops  # noqa: F401
    except Exception:
        print("Installing...")
        subprocess.check_call(
            [
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
            ]
        )
    print("Installed packages")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run streaming benchmarks for a single model"
    )
    parser.add_argument(
        "config_dir",
        help="Path to training config directory, e.g. scripts/pretraining/clip/B14",
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
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for SPFS frame skipping",
    )
    parser.add_argument(
        "--max-consecutive-skips",
        type=int,
        default=0,
        help="Maximum number of consecutive frames to skip",
    )
    parser.add_argument(
        "--mode",
        default="streammamba_spfs",
        choices=[
            "streammamba_dense",
            "streammamba_spfs",
            "streammamba_spfs_uniform",
            "streammamba_skip",
            "lstm",
        ],
        help="Streaming configuration variant",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=2,
        help="Sampling rate for *_uniform modes",
    )
    parser.add_argument(
        "--no-spfs",
        action="store_true",
        help="(Deprecated) Disable SPFS; equivalent to --mode streammamba_dense",
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

def retrieve_text_streaming_spfs(
    new_frame,
    texts,
    model,
    prev_hidden_state=None,
    *,
    topk: int = 5,
    config: dict,
    device,
    confidence_threshold: float,
    max_consecutive_skips: int,
    reuse_state_on_skip: bool,
    frames2tensor_func,
):
    """Lightweight inline implementation of ``retrieve_text_streaming`` with SPFS support."""

    size_t = config.get("size_t", 224)

    frame_tensor = frames2tensor_func(
        [new_frame], fnum=1, target_size=(size_t, size_t), device=device
    )
    if frame_tensor.ndim == 5:
        frame_tensor = frame_tensor.squeeze(1)

    vid_feat, new_hidden_state, spfs_info = model.get_streaming_vid_feat(
        frame_tensor,
        prev_hidden_state,
        confidence_threshold=confidence_threshold,
        max_consecutive_skips=max_consecutive_skips,
        reuse_state_on_skip=reuse_state_on_skip,
    )

    text_feats = torch.cat([model.get_txt_feat(t) for t in texts], dim=0)

    probs, idxs = model.predict_label(vid_feat, text_feats, top=topk)

    ret_texts = [texts[i] for i in idxs.long().cpu().numpy()[0].tolist()]
    return ret_texts, probs.float().cpu().numpy()[0], new_hidden_state, spfs_info

def main():
    ensure_dependencies()
    args = parse_args()

    # ---------- Setup ----------

    sys.path.append(os.getcwd())

    from demo.config import Config, eval_dict_leaf
    from demo.utils import _frame_from_video
    from demo.utils import frames2tensor
    from tqdm import tqdm
    from models.internvideo2_clip_small import InternVideo2_CLIP_small
    import torch
    import cv2
    from iv2_utils.iv2 import json_read, json_write

    # ---------- Configuration ----------

    config_path = os.path.join(args.config_dir, "config.py")
    config = Config.from_file(config_path)
    config = eval_dict_leaf(config)

    if args.no_spfs and args.mode != "lstm":
        args.mode = "streammamba_dense"

    if args.mode == "lstm":
        use_spfs = False
        expected_rnn_type = "lstm"
        args.checkpoint_file = "lstm_ckpt.pt"
    else:
        use_spfs = args.mode in [
            "streammamba_spfs",
            "streammamba_spfs_uniform",
            "streammamba_skip",
        ]
        expected_rnn_type = "mamba_spfs" if use_spfs else "mamba"
    current_rnn_type = config.model.streaming_vision_encoder.rnn_type

    if current_rnn_type != expected_rnn_type:
        print(
            f"Warning: Overriding RNN type from '{current_rnn_type}' to '{expected_rnn_type}'."
        )
        config.model.streaming_vision_encoder.rnn_type = expected_rnn_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Data Preparation ----------

    if "photography-model" not in os.listdir("."):
        subprocess.check_call(
            ["git", "clone", "https://github.com/ruo2019/photography-model.git"]
        )

    act75_data = json_read("photography-model/data/ACT75.json")

    # ---------- Model Loading ----------

    intern_model = InternVideo2_CLIP_small(config)
    intern_model.to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        print(f"Downloading {args.checkpoint_file} from Hugging Face...")
        ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.checkpoint_file)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "model" in ckpt.keys():
        state_dict = ckpt["model"]
    elif "module" in ckpt.keys():
        state_dict = ckpt["module"]
    else:
        print("ERROR: Checkpoint state_dict does not contain 'model' or 'module' keys.")
        sys.exit(1)

    missing_keys, unexpected_keys = intern_model.load_state_dict(
        state_dict, strict=False
    )

    if unexpected_keys:
        print("\nERROR: Unexpected keys in checkpoint state_dict:")
        for k in unexpected_keys:
            print(f"  - {k}")

    if missing_keys:
        print("\nINFO: Missing keys in checkpoint state_dict:")
        for k in missing_keys[:5]:
            print(f"  - {k}")
        if len(missing_keys) > 5:
            print(f"  - ... and {len(missing_keys) - 5} more")

    print("\nCheckpoint loaded successfully.")

    intern_model.eval()
    print("Model set to evaluation mode.")

    # ---------- Prediction ----------

    logits = []

    preds = []

    size_t = config.get("size_t", 224)

    intern_model.to(device)

    for video_path, phrase, frames in act75_data:
        frames = [
            x
            for x in _frame_from_video(
                cv2.VideoCapture("photography-model/" + video_path)
            )
        ]

        logit_curr = []
        pbar = tqdm(range(7, len(frames)))

        curr_hidden_state = intern_model.streaming_vision_encoder.init_hidden(
            batch_size=1, device=device
        )

        for frame_idx in range(7):
            initial_frame_mc = (
                frames2tensor(
                    [frames[frame_idx]],
                    fnum=1,
                    target_size=(size_t, size_t),
                    device=device,
                )
                .squeeze(0)
                .to(device)
            )

            _, curr_hidden_state, _ = intern_model.streaming_vision_encoder(
                initial_frame_mc,
                curr_hidden_state,
                confidence_threshold=1.0,
                max_consecutive_skips=0,
            )
            logit_curr.append(0.0)

        for j in pbar:
            force_skip = False
            if args.mode == "streammamba_spfs_uniform":
                sampling_rate = args.sampling_rate
                if sampling_rate > 1:
                    force_skip = (j - 7) % sampling_rate != (sampling_rate - 1)
                else:
                    force_skip = False
            threshold = (
                -1e6
                if force_skip
                else (
                    args.confidence_threshold
                    if use_spfs
                    else 1.0
                )
            )
            max_skip = (
                1000 if force_skip else (args.max_consecutive_skips if use_spfs else 0)
            )

            _, probs, curr_hidden_state, _ = retrieve_text_streaming_spfs(
                frames[j],
                [phrase],
                intern_model,
                curr_hidden_state,
                topk=1,
                config=config,
                device=device,
                confidence_threshold=threshold,
                max_consecutive_skips=max_skip,
                reuse_state_on_skip=(args.mode == "streammamba_skip"),
                frames2tensor_func=frames2tensor,
            )
            logit_curr.append(probs.item())
            if len(logit_curr) > 0:
                pbar.set_description(str(np.argmax(logit_curr) + 1))

        preds.append(np.argmax(logit_curr) + 1)
        logits.append(list(zip(logit_curr, range(1, len(logit_curr) + 1))))

    preds = [int(x) for x in preds]

    reformatted_logits = [[(float(l[0]), l[1]) for l in x] for x in logits]

    if args.mode == "streammamba_skip":
        root_folder = "results_skip"
    elif args.mode == "streammamba_spfs_uniform":
        root_folder = "results_uniform"
    else:
        root_folder = "results"

    rnn_type = "lstm" if args.mode == "lstm" else ("mamba_spfs" if use_spfs else "mamba")
    folder_name = f"{root_folder}/{rnn_type}_ct_{args.confidence_threshold}_mcs_{args.max_consecutive_skips}"

    # Add sampling rate to folder name if using uniform mode
    if "uniform" in args.mode:
        folder_name += f"_sr_{args.sampling_rate}"

    logits_dir = os.path.join(folder_name, "logits")
    preds_dir = os.path.join(folder_name, "predictions", "act75")
    metrics_file = os.path.join(folder_name, "metrics.json")

    os.makedirs(logits_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    json_write(reformatted_logits, os.path.join(logits_dir, "act75.json"))
    json_write(preds, os.path.join(preds_dir, "8.json"))

    metrics = compute_accuracy(preds, act75_data)

    run_details = {
        "confidence_threshold": args.confidence_threshold,
        "max_consecutive_skips": args.max_consecutive_skips,
        "mode": args.mode,
        "rnn_type": rnn_type,
        "model_config_path": args.config_dir,
        "command": " ".join(sys.argv),
        "performance": metrics,
    }
    json_write(run_details, metrics_file)

    print(f"Saved MAE results to {os.path.join(preds_dir, '8.json')}")
    print(f"Saved logits to {os.path.join(logits_dir, 'act75.json')}")
    print(f"Saved MAE metrics to {metrics_file}")
    print(f"Average MAE accuracy: {metrics['average']:.2f}")

if __name__ == "__main__":
    main()
