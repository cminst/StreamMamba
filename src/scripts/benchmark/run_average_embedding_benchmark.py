import argparse
from tqdm import tqdm
import os
import sys
import subprocess
import torch


def ensure_dependencies():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("Installing required dependencies...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch==2.2.1+cpu",
            "torchvision==0.17.1+cpu",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
            "einops",
            "peft",
            "open_clip_torch",
            "protobuf",
            "sentencepiece",
            "iv2-utils",
            "matplotlib",
            "opencv-python",
            "huggingface_hub",
        ])
        print("Dependencies installed.")
    else:
        print("Dependencies already available.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark average StreamMamba embedding for longer windows"
    )
    parser.add_argument(
        "config_dir",
        help="Path to training config directory, e.g. scripts/pretraining/clip/B14"
    )
    parser.add_argument(
        "--output-json",
        default="average_embedding_results.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint file. If not provided, will download from HF"
    )
    parser.add_argument(
        "--checkpoint-file",
        default="spfs_r64/ckpt_step_24500.pt",
        help="Checkpoint filename within the HF repo"
    )
    parser.add_argument(
        "--hf-repo",
        default="qingy2024/InternVideo2-B14",
        help="HuggingFace repo to download checkpoint from"
    )
    parser.add_argument(
        "--window-sizes",
        default="10,16,24,30",
        help="Comma separated list of window sizes to evaluate"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output for debugging"
    )
    return parser.parse_args()


def cosine_distance(a, b):
    with torch.no_grad():
        return 1 - torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()


def preprocess_frames_to_tensors(frames, size, device):
    """Preprocess all frames to tensors once to avoid repeated conversion."""
    from demo.utils import frames2tensor
    tensors = []
    with torch.no_grad():
        for frame in frames:
            tensor = frames2tensor([frame], fnum=1, target_size=(size, size), device=device).squeeze(0)
            tensors.append(tensor)
    return tensors


def streammamba_embedding_from_tensors(tensors, model, device):
    """StreamMamba embedding using pre-processed tensors."""
    with torch.no_grad():
        hidden = model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)
        emb = None
        for tensor in tensors:
            emb, hidden, _ = model.streaming_vision_encoder(
                tensor,
                hidden,
                confidence_threshold=1.0,
                max_consecutive_skips=0,
            )

        vision_embeds_aligned = model.vision_align(emb)
        vision_embeds_aligned /= vision_embeds_aligned.norm(dim=-1, keepdim=True)
        return vision_embeds_aligned.squeeze(0).cpu()


def teacher_embedding(frames, model, device, size_t):
    """Teacher embedding (unchanged for compatibility)."""
    from demo.utils import frames2tensor
    with torch.no_grad():
        tensor = frames2tensor(frames, fnum=8, target_size=(size_t, size_t), device=device)
        emb = model.get_vid_feat(tensor).squeeze(0)
        return emb.cpu()


def process_video_optimized(frames, model, device, size_t, window_sizes, results, verbose=False):
    """Process a single video with optimized computation and progress tracking."""
    if not frames:
        return

    if verbose:
        print(f"Processing video with {len(frames)} frames")

    # Pre-process all frames to tensors once
    frame_tensors = preprocess_frames_to_tensors(frames, size_t, device)

    # Outer loop: window sizes
    for N in tqdm(window_sizes, desc="Window sizes", leave=False, disable=not verbose):
        if len(frames) < N:
            if verbose:
                print(f"Skipping window size {N} (not enough frames: {len(frames)})")
            continue

        # Inner loop: sliding window
        for start in tqdm(range(len(frames) - N + 1), desc=f"Sliding N={N}", leave=False, disable=not verbose):
            clip_frames = frames[start:start + N]
            clip_tensors = frame_tensors[start:start + N]

            # Compute teacher embedding
            teacher = teacher_embedding(clip_frames, model, device, size_t)

            # Compute all StreamMamba embeddings for 8-frame windows
            avg_embeds = []
            for k in range(N - 7):
                sub_tensors = clip_tensors[k:k + 8]
                embed = streammamba_embedding_from_tensors(sub_tensors, model, device)
                avg_embeds.append(embed)

            # Average embedding
            with torch.no_grad():
                avg_embed = torch.stack(avg_embeds).mean(dim=0)

            # Tail embedding is just the last computed embedding
            tail_embed = avg_embeds[-1]

            # Compute distances
            d_avg = cosine_distance(avg_embed, teacher)
            d_tail = cosine_distance(tail_embed, teacher)

            # Update results
            results[N]["d_avg"] += d_avg
            results[N]["d_tail"] += d_tail
            results[N]["count"] += 1

            if verbose:
                print(f"  Window {start+1}/{len(frames)-N+1}, d_avg={d_avg:.4f}, d_tail={d_tail:.4f}")


def main():
    ensure_dependencies()
    args = parse_args()

    sys.path.append(os.getcwd())

    from demo.config import Config, eval_dict_leaf
    from demo.utils import _frame_from_video
    from models.internvideo2_clip_small import InternVideo2_CLIP_small
    from huggingface_hub import hf_hub_download
    from iv2_utils.iv2 import json_read, json_write
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Clone repo if not exists
    repo_path = "photography-model"
    if not os.path.exists(repo_path):
        print("Cloning photography-model repository...")
        subprocess.check_call([
            "git", "clone", "https://github.com/ruo2019/photography-model.git"
        ])
        print("Repository cloned.")

    # Load config
    config_path = os.path.join(args.config_dir, "config.py")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    print(f"Loading config from: {config_path}")
    config = Config.from_file(config_path)
    config = eval_dict_leaf(config)

    # Initialize model
    print("Initializing model...")
    model = InternVideo2_CLIP_small(config)
    model.to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        print(f"Downloading checkpoint '{args.checkpoint_file}' from Hugging Face ({args.hf_repo})...")
        ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.checkpoint_file)
        print(f"Checkpoint downloaded: {ckpt_path}")

    print("Loading model checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("module"))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded and set to eval mode.")

    # Clean up memory
    del ckpt, state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load dataset
    data_path = os.path.join(repo_path, "data/ACT75.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    print(f"Loading dataset: {data_path}")
    act75_data = json_read(data_path)

    size_t = config.get("size_t", 224)
    requested_window_sizes = sorted([int(x) for x in args.window_sizes.split(",")])
    print(f"Requested window sizes: {requested_window_sizes}")

    # Load existing results if file exists
    existing_results = {}
    if os.path.exists(args.output_json):
        print(f"Loading existing results from: {args.output_json}")
        existing_results = json_read(args.output_json)
        
        # Filter window sizes to only process those not already in results
        window_sizes = [N for N in requested_window_sizes if str(N) not in existing_results]
        if window_sizes:
            print(f"Skipping already processed window sizes: {[N for N in requested_window_sizes if str(N) in existing_results]}")
            print(f"Processing new window sizes: {window_sizes}")
        else:
            print("All requested window sizes already processed. Nothing to do.")
            return
    else:
        window_sizes = requested_window_sizes
        print(f"No existing results found. Processing all window sizes: {window_sizes}")

    # Initialize results for new window sizes
    results = {N: {"d_avg": 0.0, "d_tail": 0.0, "count": 0} for N in window_sizes}

    # Main video processing loop with tqdm
    print(f"Processing {len(act75_data)} videos...")
    for item in tqdm(act75_data, desc="Videos", unit="video"):
        video_path, _, _ = item
        full_video_path = os.path.join(repo_path, video_path)

        if not os.path.exists(full_video_path):
            tqdm.write(f"Video not found: {full_video_path}")
            continue

        try:
            cap = cv2.VideoCapture(full_video_path)
            frames = [x for x in _frame_from_video(cap)]
            cap.release()

            if args.verbose:
                tqdm.write(f"Loaded {len(frames)} frames from {video_path}")

            if len(frames) == 0:
                tqdm.write(f"No frames extracted from {video_path}")
                continue

            # Process video with progress bars inside
            process_video_optimized(frames, model, device, size_t, window_sizes, results, verbose=args.verbose)

            # Optional: free memory
            del frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            tqdm.write(f"Error processing {video_path}: {e}")
            continue

    # Finalize results for new window sizes
    print("\nNew Results:")
    for N in window_sizes:
        if results[N]["count"] > 0:
            results[N]["d_avg"] /= results[N]["count"]
            results[N]["d_tail"] /= results[N]["count"]
            print(f"  Window {N}: {results[N]['count']} samples | "
                  f"d_avg = {results[N]['d_avg']:.4f} | d_tail = {results[N]['d_tail']:.4f}")
        else:
            print(f"  Window {N}: No valid samples processed.")

    # Merge new results with existing ones
    final_results = existing_results.copy()
    for N, data in results.items():
        if data["count"] > 0:  # Only include results with actual data
            final_results[str(N)] = {
                "d_avg": data["d_avg"],
                "d_tail": data["d_tail"],
                "count": data["count"]
            }

    # Sort final results by window size
    final_results = dict(sorted(final_results.items(), key=lambda x: int(x[0])))

    # Save merged results
    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    json_write(final_results, args.output_json)
    print(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()