import argparse
import os
import sys
import subprocess
import numpy as np
import torch
from huggingface_hub import hf_hub_download

def ensure_dependencies():
    """Ensure required packages are installed."""
    required_packages = {"einops", "peft""sentencepiece", "iv2-utils", "huggingface_hub"}

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing required packages: {missing}")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run streaming benchmarks for a single model"
    )
    parser.add_argument(
        "config_dir",
        help="Path to training config directory, e.g. scripts/pretraining/clip/B14",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Path to a checkpoint file. If not provided, will download from HF.",
    )
    parser.add_argument(
        "--hf-checkpoint-file",
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
        "--dataset-name",
        default="act75",
        choices=[
            "act75",
            "flash",
        ],
        help="Dataset to evaluate on",
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
    for offset in range(search_range[0], search_range[1] + 1):
        shifted = [p + offset for p in preds]
        mse = calculate_mse(shifted, data)
        if mse < best_mse:
            best_mse = mse
            best_offset = offset
    return best_offset

def offset_predictions(preds, data):
    best = find_best_offset(preds, data)
    return [p + best for p in preds]

def compute_frame_accuracy(preds: list[int], dataset: list) -> dict:
    """Return accuracy percentages within various frame offset thresholds."""
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

def forward_streaming_spfs(
    new_frame_tensor,
    texts,
    model,
    confidence_threshold: float = 0.8,
    max_consecutive_skips: int = 8,
    prev_hidden_state=None,
):
    """Lightweight implementation of ``retrieve_text_streaming`` with SPFS support."""

    if new_frame_tensor.ndim == 5:
        print("NDIM IS 5")
        new_frame_tensor = new_frame_tensor.squeeze(1)

    vid_feat, new_hidden_state, spfs_info = model.get_streaming_vid_feat(
        new_frame_tensor,
        prev_hidden_state,
        confidence_threshold=confidence_threshold,
        max_consecutive_skips=max_consecutive_skips,
    )

    text_feats = torch.cat([model.get_txt_feat(t) for t in texts], dim=0)

    probs, idxs = model.predict_label(vid_feat, text_feats, top=1)

    ret_texts = [texts[i] for i in idxs.long().cpu().numpy()[0].tolist()]
    return ret_texts, probs.float().cpu().numpy()[0], new_hidden_state, spfs_info

def normalize_dataset(args, raw_dataset):
    """
    Normalize dataset entries into a common format:
    For ACT75: expected as [video_path, phrase, gt_frames]
    For FLASH: input format is [video_path, [peaks...]] per video.
    We expand each peak into an entry:
      (video_path, caption, relative_gt_peak_frames, build_up, drop_off)
    """
    dataset = []
    if args.dataset_name.lower() == "flash":
        for item in raw_dataset:
            # Each item: [video_path, [ {build_up, peak_start, peak_end, drop_off, caption}, ... ]]
            video_path, peaks = item

            for peak in peaks:
                build_up, peak_start, peak_end, drop_off = peak["build_up"], peak["peak_start"], peak["peak_end"], peak["drop_off"]
                caption = str(peak.get("caption", "")).strip()

                # Compute relative GT frames within [build_up, drop_off] inclusive
                # Prediction indices are 1-based relative to the segment start
                rel_start = peak_start - build_up + 1
                rel_end = peak_end - build_up + 1
                if rel_end < 1 or rel_start > (drop_off - build_up + 1):
                    # Peak window falls entirely outside the evaluated segment
                    rel_gt = []
                else:
                    rel_start = max(rel_start, 1)
                    rel_end = min(rel_end, drop_off - build_up + 1)
                    rel_gt = list(range(rel_start, rel_end + 1))

                dataset.append((video_path, caption, rel_gt, build_up, drop_off))
    else:
        # Assume already in [video_path, phrase, gt_frames]
        dataset = raw_dataset

    return dataset

def should_force_skip(frame_idx, args):
    if args.mode == "streammamba_spfs_uniform" and args.sampling_rate > 1:
        return (frame_idx - 7) % args.sampling_rate != (args.sampling_rate - 1)
    else:
        return False

def get_skip_parameters(frame_idx, args, use_spfs):
    """Determine skip parameters for current frame."""
    if should_force_skip(frame_idx, args):
        return -1e6, 10**6  # Force skip
    elif use_spfs:
        return args.confidence_threshold, args.max_consecutive_skips
    else:
        return 1.0, 0  # No skipping

def get_output_folder(args, rnn_type):
    """Generate output folder name based on configuration."""
    base = "results_uniform" if "uniform" in args.mode else "results"

    spfs_template = f"{base}/results_{rnn_type}_ct_{args.confidence_threshold}_mcs_{args.max_consecutive_steps}"
    uniform_sampling_template = f"{base}/results_{rnn_type}_{args.sampling_rate}"

    if "uniform" in args.mode:
        return uniform_sampling_template
    else:
        return spfs_template

def get_checkpoint_weights(ckpt_path):
    """
    Load the checkpoint weights from a given file path.

    Args:
        ckpt_path (str): Path to the checkpoint file.

    Returns:
        dict: The state dictionary of the model weights.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model" in ckpt.keys():
        state_dict = ckpt["model"]
    elif "module" in ckpt.keys():
        state_dict = ckpt["module"]
    else:
        print("ERROR: Checkpoint state_dict does not contain 'model' or 'module' keys.")
        sys.exit(1)

    return state_dict

def load_checkpoint(args, model):
    """
    Load a model checkpoint from a specified file or download it from Hugging Face if not provided.

    Args:
        args: Parsed command-line arguments containing checkpoint information.
        model: The model to load the checkpoint into.

    This function handles downloading the checkpoint from Hugging Face if a local file is not provided.
    It also checks for missing or unexpected keys in the checkpoint and prints relevant information.
    """
    ckpt_path = args.checkpoint_file
    if ckpt_path is None:
        print(f"Downloading {args.hf_checkpoint_file} from Hugging Face...")
        ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_checkpoint_file)

    missing_keys, unexpected_keys = model.load_state_dict(get_checkpoint_weights(ckpt_path), strict=False)

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

    model.eval()

def main():
    ensure_dependencies()
    args = parse_args()
    sys.path.append(os.getcwd())

    # ------------- Setup ---------------

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

    if args.mode == "lstm":
        use_spfs = False
        expected_rnn_type = "lstm"
        args.hf_checkpoint_file = "lstm_ckpt.pt" # Download LSTM checkpoint from HF Repo
    else:
        use_spfs = args.mode in [
            "streammamba_spfs",
            "streammamba_spfs_uniform",
        ]
        expected_rnn_type = "mamba_spfs" if use_spfs else "mamba"

    current_rnn_type = config.model.streaming_vision_encoder.rnn_type

    if current_rnn_type != expected_rnn_type:
        print(f"Warning: Overriding RNN type from '{current_rnn_type}' to '{expected_rnn_type}'.")
        config.model.streaming_vision_encoder.rnn_type = expected_rnn_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Helper for simplifying frame -> tensor conversion
    def get_frame_tensor(frame):
        frame_size = config.get("size_t", 224)
        return frames2tensor([frame], fnum=1, target_size=(frame_size, frame_size), device=device)

    # -------- Data Preparation ---------

    if "peakframe-toolkit" not in os.listdir("."):
        subprocess.check_call(["git", "clone", "https://github.com/cminst/peakframe-toolkit.git"])

    # Load dataset JSON
    json_path = f"peakframe-toolkit/data/{args.dataset_name.upper().replace('-', '_')}.json"
    dataset = normalize_dataset(args, json_read(json_path))

    # ----------- Prediction ------------

    model = InternVideo2_CLIP_small(config).to(device)

    load_checkpoint(args, model)

    print("\nCheckpoint loaded.")

    logits = []

    preds = []

    for entry in dataset:
        # Unpack depending on dataset variant
        if len(entry) >= 5:
            video_path, phrase, gt_frames, seg_start, seg_end = entry[:5]
        elif len(entry) == 3:
            video_path, phrase, gt_frames = entry
            seg_start, seg_end = None, None
        else:
            # Unexpected format, skip
            continue

        # Load frames from video
        video_capture = cv2.VideoCapture("peakframe-toolkit/" + video_path)
        all_frames = [x for x in _frame_from_video(video_capture)]

        # For FLASH dataset, crop to the [build_up, drop_off] segment
        if seg_start is not None and seg_end is not None:
            start_i = max(0, int(seg_start))
            end_i = min(len(all_frames) - 1, int(seg_end))

            frames = all_frames[start_i : end_i + 1]
        else:
            frames = all_frames

        # Need at least 8 frames to warm up and score
        assert len(frames) >= 8, f"Video must have at least 8 frames, but only found {len(frames)}."

        logits_list_curr = []
        frame_progress_bar = tqdm(range(7, len(frames)))

        curr_hidden_state = model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)

        for frame_idx in range(7):
            initial_frame_tensor = get_frame_tensor(frames[frame_idx]).squeeze(0).to(device)

            _, curr_hidden_state, _ = model.streaming_vision_encoder(
                initial_frame_tensor,
                curr_hidden_state,
                confidence_threshold=1.0,
                max_consecutive_skips=0,
            )

            logits_list_curr.append(0.0)

        for frame_idx in frame_progress_bar:
            # Determine if the current frame should be skipped (forcefully) in uniform sampling mode
            confidence_threshold, max_consecutive_skips = get_skip_parameters(frame_idx, args, use_spfs)

            _, probs, curr_hidden_state, _ = forward_streaming_spfs(
                get_frame_tensor(frames[frame_idx]),
                [phrase],
                model,
                confidence_threshold=confidence_threshold,
                max_consecutive_skips=max_consecutive_skips,
                prev_hidden_state=curr_hidden_state,
            )

            logits_list_curr.append(float(probs.item()))

            frame_progress_bar.set_description(f"Current Best Frame: {np.argmax(logits_list_curr) + 1}")

        preds.append(int(np.argmax(logits_list_curr) + 1))
        logits.append(zip(logits_list_curr, range(1, len(logits_list_curr) + 1)))

    folder_name = get_output_folder(args, expected_rnn_type)

    logits_dir = os.path.join(folder_name, "logits")
    logits_path = os.path.join(logits_dir, f"{args.dataset_name}.json")

    preds_dir = os.path.join(folder_name, "predictions", args.dataset_name)
    preds_path = os.path.join(preds_dir, "8.json")

    metrics_file = os.path.join(folder_name, "metrics.json")

    os.makedirs(logits_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    json_write(logits, logits_path)
    json_write(preds, preds_path)

    metrics = compute_frame_accuracy(preds, dataset)

    run_details = {
        "confidence_threshold": args.confidence_threshold,
        "max_consecutive_skips": args.max_consecutive_skips,
        "mode": args.mode,
        "rnn_type": expected_rnn_type,
        "model_config_path": args.config_dir,
        "command": " ".join(sys.argv),
        "performance": metrics,
    }
    json_write(run_details, metrics_file)

    print(f"Saved MAE results to {preds_path}")
    print(f"Saved logits to {logits_path}")
    print(f"Saved MAE metrics to {metrics_file}")
    print(f"Average MAE accuracy: {metrics['average']:.2f}")

if __name__ == "__main__":
    main()
