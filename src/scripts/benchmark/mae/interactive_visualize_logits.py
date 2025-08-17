import argparse
import os
import sys
import subprocess
import json
from typing import List, Tuple, Optional

import numpy as np
import torch
import cv2
import pygame
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# --- Dependency Management ---

def ensure_dependencies():
    """Ensure required packages are installed."""
    required_packages = {
        "einops", "peft", "sentencepiece", "iv2-utils",
        "huggingface_hub", "tqdm", "pygame", "opencv-python"
    }
    missing = []
    for package in required_packages:
        try:
            # Special case for opencv-python
            if package == "opencv-python":
                __import__("cv2")
            else:
                __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing required packages: {missing}")
        print("Please install them using: pip install " + " ".join(missing))
        sys.exit(1)

# --- Argument Parsing ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively visualize model logits for a video and a custom prompt."
    )
    # --- Model & Inference Arguments (from inference script) ---
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
        help="Checkpoint filename within the HF repo.",
    )
    parser.add_argument(
        "--hf-repo",
        default="qingy2024/InternVideo2-B14",
        help="HuggingFace repo to download checkpoint from.",
    )
    parser.add_argument(
        "--mode",
        default="streammamba_spfs",
        choices=[
            "streammamba_dense", "streammamba_spfs", "streammamba_spfs_uniform",
            "lstm", "mobileclip", "internvideo2"
        ],
        help="Model/streaming configuration variant.",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.9,
        help="Confidence threshold for SPFS frame skipping."
    )
    parser.add_argument(
        "--max-consecutive-skips", type=int, default=0,
        help="Maximum number of consecutive frames to skip with SPFS."
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=2,
        help="Sampling rate for 'uniform' modes."
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Override evaluation FPS by skipping frames."
    )
    # --- Visualization & Data Arguments (from visualization script) ---
    parser.add_argument(
        "--dataset-name", default="flash", choices=["flash", "act75"],
        help="Dataset name to source video and metadata from."
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Zero-based index of the example within the dataset to visualize."
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Custom text prompt to use. If not provided, uses the caption from the dataset."
    )
    parser.add_argument(
        "--video-root", default="peakframe-toolkit",
        help="Root folder containing dataset videos and data JSON (default: peakframe-toolkit)."
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Scale factor for the display window (e.g., 0.75 to shrink)."
    )
    return parser.parse_args()

# --- Model Loading & Utilities (from inference script) ---

def get_checkpoint_weights(ckpt_path):
    """Load the checkpoint weights from a given file path."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        return ckpt["model"]
    if "module" in ckpt:
        return ckpt["module"]
    print("ERROR: Checkpoint state_dict does not contain 'model' or 'module' keys.")
    sys.exit(1)

def load_checkpoint(args, model):
    """Load a model checkpoint from a file or download from Hugging Face."""
    ckpt_path = args.checkpoint_file
    if ckpt_path is None:
        print(f"Downloading {args.hf_checkpoint_file} from {args.hf_repo}...")
        ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_checkpoint_file)

    print(f"Loading checkpoint from: {ckpt_path}")
    missing_keys, unexpected_keys = model.load_state_dict(get_checkpoint_weights(ckpt_path), strict=False)
    if unexpected_keys:
        print("\nWarning: Unexpected keys in checkpoint state_dict:")
        for k in unexpected_keys[:5]: print(f"  - {k}")
        if len(unexpected_keys) > 5: print(f"  - ... and {len(unexpected_keys) - 5} more")
    if missing_keys:
        print("\nWarning: Missing keys in checkpoint state_dict:")
        for k in missing_keys[:5]: print(f"  - {k}")
        if len(missing_keys) > 5: print(f"  - ... and {len(missing_keys) - 5} more")
    model.eval()
    print("Checkpoint loaded successfully.")

def forward_streaming_spfs(new_frame_tensor, text, model, confidence_threshold, max_consecutive_skips, prev_hidden_state):
    """Lightweight implementation of `retrieve_text_streaming` with SPFS support."""
    vid_feat, new_hidden_state, _ = model.get_streaming_vid_feat(
        new_frame_tensor,
        prev_hidden_state,
        confidence_threshold=confidence_threshold,
        max_consecutive_skips=max_consecutive_skips,
    )
    text_feat = model.get_txt_feat(text)
    probs, _ = model.predict_label(vid_feat, text_feat, top=1)
    return probs.float().cpu().numpy()[0], new_hidden_state

def get_skip_parameters(frame_idx, args, use_spfs):
    """Determine skip parameters for the current frame."""
    if "uniform" in args.mode and args.sampling_rate > 1:
        if (frame_idx - 7) % args.sampling_rate != (args.sampling_rate - 1):
            return -1e6, 10**6  # Force skip
    if use_spfs:
        return args.confidence_threshold, args.max_consecutive_skips
    return 1.0, 0  # No skipping

# --- Data Handling & Visualization Utilities (from visualization script) ---

def json_read(path: str):
    with open(path, "r") as f:
        return json.load(f)

def normalize_dataset(dataset_name: str, raw_dataset: list) -> list:
    """Normalize dataset entries into a common format."""
    dataset = []
    if dataset_name.lower() == "flash":
        for item in raw_dataset:
            video_path, peaks = item
            for peak in peaks:
                build_up, peak_start, peak_end, drop_off = peak["build_up"], peak["peak_start"], peak["peak_end"], peak["drop_off"]
                caption = str(peak.get("caption", "")).strip()
                rel_start = int(peak_start) - int(build_up) + 1
                rel_end = int(peak_end) - int(build_up) + 1
                segment_len = int(drop_off) - int(build_up) + 1
                if rel_end < 1 or rel_start > segment_len:
                    rel_gt = []
                else:
                    rel_start = max(rel_start, 1)
                    rel_end = min(rel_end, segment_len)
                    rel_gt = list(range(rel_start, rel_end + 1))
                dataset.append((video_path, caption, rel_gt, build_up, drop_off))
    else: # act75
        dataset = raw_dataset
    return dataset

def load_video_frames(video_path: str) -> Tuple[Optional[List[pygame.Surface]], int, int, int]:
    """Load all frames from a video file as Pygame surfaces."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None, 0, 0, 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, 0, 0
    frames: List[pygame.Surface] = []
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame_bgr = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        surface = pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")
        frames.append(surface)
    cap.release()
    return frames, w, h, fps

def replace_leading_zeros(logits: List[float]) -> List[float]:
    """Replace leading zeros (from warmup) with the first non-zero logit value."""
    first_non_zero = next((val for val in logits if val != 0), None)
    if first_non_zero is None: return logits

    new_logits, replaced = [], False
    for val in logits:
        if not replaced and val == 0:
            new_logits.append(first_non_zero)
        else:
            new_logits.append(val)
            replaced = True
    return new_logits

def draw_graph(screen, rect, logits, current_index_1based, gt_frames_1based=None):
    """Draws the logit graph UI."""
    pygame.draw.rect(screen, (30, 30, 30), rect)
    pygame.draw.rect(screen, (70, 70, 70), rect, 1)

    points = [(i + 1, float(v)) for i, v in enumerate(logits)]
    if not points: return

    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if abs(y_max - y_min) > 1e-6 else 1.0

    def scale_point(x, y):
        px = int(rect.left + ((x - x_min) / x_range) * rect.width)
        py = int(rect.bottom - ((y - y_min) / y_range) * rect.height)
        return px, py

    if gt_frames_1based:
        gmin_x, _ = scale_point(min(gt_frames_1based), y_min)
        gmax_x, _ = scale_point(max(gt_frames_1based), y_min)
        band = pygame.Surface((max(1, gmax_x - gmin_x + 1), rect.height), pygame.SRCALPHA)
        band.fill((0, 200, 0, 60))
        screen.blit(band, (gmin_x, rect.top))

    scaled_points = [scale_point(x, y) for x, y in points]
    if len(scaled_points) >= 2:
        pygame.draw.lines(screen, (80, 160, 255), False, scaled_points, 2)

    ci = max(0, min(current_index_1based - 1, len(scaled_points) - 1))
    cx, cy = scaled_points[ci]
    pygame.draw.circle(screen, (255, 60, 60), (cx, cy), 5)


# --- Core Inference Logic ---

def calculate_logits_for_video(model, frames_cv2, prompt, args, config, device):
    """
    Run inference for a single video and prompt to generate a time-series of logits.

    Args:
        model: The loaded InternVideo2 model.
        frames_cv2: A list of video frames (in OpenCV BGR format).
        prompt (str): The text prompt to score against the video.
        args: Parsed command-line arguments.
        config: Model configuration dictionary.
        device: The torch device to run on.

    Returns:
        A list of float logit values, one for each frame.
    """
    from demo.utils import frames2tensor, retrieve_text

    logits = []

    # Helper to convert a single CV2 frame to a tensor
    def get_frame_tensor(frame):
        frame_size = config.get("size_t", 224)
        return frames2tensor([frame], fnum=1, target_size=(frame_size, frame_size), device=device).squeeze(0)

    # Determine frame skipping for FPS override
    eval_indices = set(range(len(frames_cv2)))
    if args.fps is not None:
        video_capture = cv2.VideoCapture(args.video_path_internal) # A bit of a hack to get FPS
        orig_fps = float(video_capture.get(cv2.CAP_PROP_FPS))
        video_capture.release()
        if orig_fps > 0 and args.fps > 0:
            stride = max(1, int(round(orig_fps / float(args.fps))))
            eval_indices = set(range(0, len(frames_cv2), stride))
            print(f"Original FPS: {orig_fps:.2f}, Target FPS: {args.fps:.2f}. Evaluating every {stride} frames.")

    # Model-specific setup
    if args.mode == "mobileclip":
        vit = model.streaming_vision_encoder.vit_lite
        txt_encoder = model.text_encoder
        text_emb = txt_encoder.encode_text(model.tokenizer(prompt).to(device)).squeeze(0).float()
        text_emb = text_emb / text_emb.norm().clamp(min=1e-12)
    elif args.mode != "internvideo2": # StreamMamba / LSTM
        curr_hidden_state = model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)

    # Main inference loop
    last_logit = 0.0
    progress_bar = tqdm(range(len(frames_cv2)), desc=f"Scoring frames for '{prompt}'", unit="frame")

    for frame_idx in progress_bar:
        skip_for_fps = frame_idx not in eval_indices
        logit = 0.0

        if args.mode == "mobileclip":
            if skip_for_fps:
                logit = last_logit
            else:
                raw_feat, _ = vit.extract_features(get_frame_tensor(frames_cv2[frame_idx]))
                img_emb = vit.classifier(raw_feat).squeeze(0)
                img_emb = img_emb / img_emb.norm().clamp(min=1e-12)
                logit = float(torch.dot(img_emb.float(), text_emb.float()).item())

        elif args.mode == "internvideo2":
            if frame_idx < 7: # Warmup
                logit = 0.0
            elif skip_for_fps:
                logit = last_logit
            else:
                window_frames = frames_cv2[frame_idx - 7 : frame_idx + 1]
                _, probs = retrieve_text(window_frames, [prompt], model=model, topk=1, config=config, device=device)
                logit = float(probs[0])
        else: # StreamMamba / LSTM
            if frame_idx < 7: # Warmup
                initial_frame_tensor = get_frame_tensor(frames_cv2[frame_idx])
                _, curr_hidden_state, _ = model.streaming_vision_encoder(
                    initial_frame_tensor, curr_hidden_state, confidence_threshold=1.0, max_consecutive_skips=0
                )
                logit = 0.0
            elif skip_for_fps:
                logit = last_logit
            else:
                use_spfs = "spfs" in args.mode
                conf, skips = get_skip_parameters(frame_idx, args, use_spfs)
                probs, curr_hidden_state = forward_streaming_spfs(
                    get_frame_tensor(frames_cv2[frame_idx]), [prompt], model, conf, skips, curr_hidden_state
                )
                logit = float(probs.item())

        logits.append(logit)
        last_logit = logit
        progress_bar.set_postfix_str(f"best_frame={np.argmax(logits) + 1}")

    return logits


# --- Main Application Logic ---

def main():
    ensure_dependencies()
    args = parse_args()
    sys.path.append(os.getcwd())

    # Import after path append
    from demo.config import Config, eval_dict_leaf
    from models.internvideo2_clip_small import InternVideo2_CLIP_small

    # --- 1. Setup Model and Config ---
    config_path = os.path.join(args.config_dir, "config.py")
    config = eval_dict_leaf(Config.from_file(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override RNN type in config based on --mode argument
    mode_to_rnn = {"lstm": "lstm", "streammamba_dense": "mamba", "streammamba_spfs": "mamba_spfs", "streammamba_spfs_uniform": "mamba_spfs"}
    if args.mode in mode_to_rnn:
        expected_rnn = mode_to_rnn[args.mode]
        if config.model.streaming_vision_encoder.rnn_type != expected_rnn:
            print(f"Overriding config RNN type from '{config.model.streaming_vision_encoder.rnn_type}' to '{expected_rnn}'.")
            config.model.streaming_vision_encoder.rnn_type = expected_rnn

    # --- 2. Load Model ---
    model = InternVideo2_CLIP_small(config).to(device)
    load_checkpoint(args, model)

    # --- 3. Load Dataset and Video Frames ---
    if "peakframe-toolkit" not in os.listdir("."):
        print("Cloning peakframe-toolkit for dataset access...")
        subprocess.check_call(["git", "clone", "https://github.com/cminst/peakframe-toolkit.git"])

    ds_json_path = os.path.join(args.video_root, "data", f"{args.dataset_name.upper()}.json")
    if not os.path.exists(ds_json_path):
        print(f"Error: Dataset JSON not found at '{ds_json_path}'. Please check --video-root.")
        sys.exit(1)

    dataset = normalize_dataset(args.dataset_name, json_read(ds_json_path))
    if not (0 <= args.index < len(dataset)):
        print(f"Error: Index {args.index} is out of range for dataset with {len(dataset)} items.")
        sys.exit(1)

    entry = dataset[args.index]
    video_rel, caption, gt_frames, seg_start, seg_end = (list(entry) + [None, None])[:5]
    video_path = os.path.join(args.video_root, video_rel)
    args.video_path_internal = video_path # For FPS calculation

    # Load frames with OpenCV for model, Pygame for display
    cap = cv2.VideoCapture(video_path)
    frames_cv2 = [frame for ret, frame in iter(lambda: cap.read(), (False, None))]
    cap.release()

    pygame.init()
    frames_pygame, f_w, f_h, _ = load_video_frames(video_path)
    if not frames_pygame:
        print("Failed to load video frames for visualization.")
        pygame.quit()
        sys.exit(1)

    # Crop frames based on dataset segment (e.g., for FLASH)
    if seg_start is not None and seg_end is not None:
        start_i, end_i = max(0, int(seg_start)), min(len(frames_cv2) - 1, int(seg_end))
        frames_cv2 = frames_cv2[start_i : end_i + 1]
        frames_pygame = frames_pygame[start_i : end_i + 1]

    if args.mode != "mobileclip" and len(frames_cv2) < 8:
        print(f"Error: Mode '{args.mode}' requires at least 8 frames for warmup, but video segment has only {len(frames_cv2)}.")
        sys.exit(1)

    # --- 4. Run Inference ---
    active_prompt = args.prompt if args.prompt is not None else caption
    logits = calculate_logits_for_video(model, frames_cv2, active_prompt, args, config, device)
    logits = replace_leading_zeros(logits)

    # --- 5. Setup Visualization UI ---
    scale = max(0.1, args.scale)
    disp_w, disp_h = int(f_w * scale), int(f_h * scale)
    screen_w, screen_h = disp_w * 2, disp_h
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.key.set_repeat(250, 50)  # Wait 250ms, then repeat every 50ms
    pygame.display.set_caption(f"'{active_prompt}' - {os.path.basename(video_path)} [{args.dataset_name}] (#{args.index})")
    try: font = pygame.font.Font(None, 36)
    except: font = pygame.font.SysFont("sans", 30)

    scaled_frames = [pygame.transform.smoothscale(f, (disp_w, disp_h)) for f in frames_pygame]
    pad = 20
    graph_rect = pygame.Rect(disp_w + pad, pad, disp_w - 2 * pad, screen_h - 2 * pad)

    # --- 6. Main Visualization Loop ---
    print("\nStarting visualization. Use arrow keys to navigate. Press ESC to quit.")
    clock = pygame.time.Clock()
    current_idx_1based = 1
    running = True
    needs_redraw = True
    total_frames = min(len(logits), len(scaled_frames))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT: current_idx_1based = min(total_frames, current_idx_1based + 1)
                elif event.key == pygame.K_LEFT: current_idx_1based = max(1, current_idx_1based - 1)
                elif event.key == pygame.K_DOWN: current_idx_1based = min(total_frames, current_idx_1based + 10)
                elif event.key == pygame.K_UP: current_idx_1based = max(1, current_idx_1based - 10)
                elif event.key == pygame.K_END: current_idx_1based = total_frames
                elif event.key == pygame.K_HOME: current_idx_1based = 1
                needs_redraw = True

        if needs_redraw:
            screen.fill((0, 0, 0))
            # Video frame
            screen.blit(scaled_frames[current_idx_1based - 1], (0, 0))
            # Frame counter overlay
            text_surf = font.render(f"Frame: {current_idx_1based} / {total_frames}", True, (255, 255, 0))
            bg_rect = pygame.Rect(5, 5, text_surf.get_width() + 10, text_surf.get_height() + 6)
            pygame.draw.rect(screen, (0, 0, 0, 180), bg_rect)
            screen.blit(text_surf, (10, 8))
            # Logit graph
            draw_graph(screen, graph_rect, logits, current_idx_1based, gt_frames)

            pygame.display.flip()
            needs_redraw = False

        clock.tick(60)

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
