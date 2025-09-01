import argparse
import os
import subprocess
import sys
import time
import json
from huggingface_hub import hf_hub_download


def ensure_dependencies():
    try:
        import einops
    except Exception:
        print("Installing dependencies...")
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
            ]
        )
    print("Installed packages")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure streaming inference FPS with SPFS"
    )
    parser.add_argument(
        "config_dir",
        help="Path to training config directory, e.g. scripts/pretraining/clip/B14",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a full SPFS checkpoint. If not provided, will download from HF.",
    )
    parser.add_argument(
        "--hf-repo",
        default="cminst/StreamMamba",
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
        default=0.9,
        help="Confidence threshold for skipping frames",
    )
    parser.add_argument(
        "--max-consecutive-skips",
        type=int,
        default=0,
        help="Maximum number of consecutive frames to skip",
    )
    parser.add_argument(
        "--output-json",
        default="fps_results_spfs.json",
        help="Path to output JSON with FPS results",
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
        help="Sampling rate for *_uniform modes.",
    )
    return parser.parse_args()


def main():
    ensure_dependencies()
    args = parse_args()

    if args.mode == "lstm":
        use_spfs = False
        rnn_type = "lstm"
        args.checkpoint_file = "lstm_ckpt.pt"
    else:
        use_spfs = args.mode in [
            "streammamba_spfs",
            "streammamba_spfs_uniform",
        ]
        rnn_type = "mamba_spfs" if use_spfs else "mamba"
    print(f"Running in {args.mode} mode.")

    if use_spfs:
        folder_name = f"results_{rnn_type}_ct_{args.confidence_threshold}_mcs_{args.max_consecutive_skips}"
    else:
        folder_name = f"results_{rnn_type}"

    # Add sampling rate to folder name if using uniform mode
    if "uniform" in args.mode:
        folder_name += f"_sr_{args.sampling_rate}"

    # Determine root folder based on mode
    if args.mode == "streammamba_spfs_uniform":
        root_folder = "results_uniform"
    else:
        root_folder = "results"

    folder_path = os.path.join(root_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    fps_json_path = os.path.join(folder_path, os.path.basename(args.output_json))

    sys.path.append(os.getcwd())

    from demo.config import Config, eval_dict_leaf
    from demo.utils import _frame_from_video, frames2tensor
    from models.internvideo2_clip_small import InternVideo2_CLIP_small
    import torch
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "peakframe-toolkit" not in os.listdir("."):
        subprocess.check_call(
            ["git", "clone", "https://github.com/cminst/peakframe-toolkit.git"]
        )

    config = Config.from_file(os.path.join(args.config_dir, "config.py"))
    config = eval_dict_leaf(config)

    # Set rnn_type dynamically
    config.model.streaming_vision_encoder.rnn_type = rnn_type
    config.model.text_ckpt_path = config.model.mobileclip_ckpt_path

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

    with open("peakframe-toolkit/data/ACT75.json", "r") as f:
        act75_data = json.load(f)

    results = []
    total_skipped_frames = 0
    total_frames = 0
    size_t = config.get("size_t", 224)

    for video_path, _, _ in act75_data:
        cap = cv2.VideoCapture("peakframe-toolkit/" + video_path)
        frames = [x for x in _frame_from_video(cap)]
        if not frames:
            continue
        h, w = frames[0].shape[:2]
        pixels = w * h
        total_time = 0.0
        skipped_frames = 0

        hidden = intern_model.streaming_vision_encoder.init_hidden(
            batch_size=1, device=device
        )

        for i in range(7):
            f = frames[i]
            tensor = frames2tensor(
                [f], fnum=1, target_size=(size_t, size_t), device=device
            ).squeeze(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _, hidden, _ = intern_model.encode_streaming_vision(
                tensor, hidden, confidence_threshold=1.0, max_consecutive_skips=0
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            total_time += end - start

        for idx, f in enumerate(frames[7:], start=7):
            force_skip = False
            if args.mode == "streammamba_spfs_uniform":
                sampling_rate = args.sampling_rate
                if sampling_rate > 1:
                    # Process 1 frame every 'sampling_rate' frames (last in cycle)
                    force_skip = (idx - 7) % sampling_rate != (sampling_rate - 1)
                else:
                    force_skip = False  # Process all if rate is 1 or less

            tensor = frames2tensor(
                [f], fnum=1, target_size=(size_t, size_t), device=device
            ).squeeze(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            if args.mode in ["streammamba_dense", "lstm"]:
                _, hidden, _ = intern_model.encode_streaming_vision(
                    tensor,
                    hidden,
                    confidence_threshold=1.0,
                    max_consecutive_skips=0,
                )
            elif args.mode == "streammamba_spfs":
                _, hidden, spfs_info = intern_model.encode_streaming_vision(
                    tensor,
                    hidden,
                    confidence_threshold=args.confidence_threshold,
                    max_consecutive_skips=args.max_consecutive_skips,
                )
                if spfs_info.skipped:
                    skipped_frames += 1
            else:  # streammamba_spfs_uniform
                threshold = -1e6 if force_skip else 1.0
                max_skip = 1000 if force_skip else 0  # Allow many consecutive skips for uniform mode
                _, hidden, spfs_info = intern_model.encode_streaming_vision(
                    tensor,
                    hidden,
                    confidence_threshold=threshold,
                    max_consecutive_skips=max_skip,
                )
                if spfs_info.skipped:
                    skipped_frames += 1

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            total_time += end - start

        total_frames += len(frames)
        total_skipped_frames += skipped_frames
        fps = len(frames) / total_time if total_time > 0 else 0.0
        results.append(
            {
                "video": video_path,
                "resolution": f"{w}x{h}",
                "pixels": pixels,
                "fps": fps,
                "skipped_frames": skipped_frames,
            }
        )

    with open(fps_json_path, "w") as f:
        json.dump(results, f)

    skip_percentage = (
        (total_skipped_frames / total_frames) * 100 if total_frames > 0 else 0
    )
    avg_fps = sum(r["fps"] for r in results) / len(results) if results else 0
    print(f"Saved FPS results to {fps_json_path}")
    print(f"Total frames skipped: {total_skipped_frames}")
    print(f"Percentage of frames skipped: {skip_percentage:.2f}%")
    print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
