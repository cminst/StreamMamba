import argparse
import os
import sys
import subprocess
import numpy as np
import torch
from huggingface_hub import hf_hub_download

import cv2
import pygame
from typing import List, Tuple, Optional, Dict, Any

# --- Dependency Check (from inference script) ---
def ensure_dependencies():
    """Ensure required packages are installed."""
    required_packages = {"einops", "peft", "sentencepiece", "iv2-utils", "huggingface_hub", "tqdm"}

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing required packages: {missing}")
        print("Please install them using: pip install " + " ".join(missing))
        sys.exit(1)

# --- UI Helper Classes ---

class TextInputBox:
    """A simple UI class for a text input box."""
    def __init__(self, rect, font, initial_text="", text_color=(255, 255, 255), box_color=(70, 70, 70), active_color=(100, 100, 200)):
        self.rect = pygame.Rect(rect)
        self.font = font
        self.text = initial_text
        self.text_surface = self.font.render(self.text, True, text_color)
        self.text_color = text_color
        self.box_color = box_color
        self.active_color = active_color
        self.active = False
        self.enabled = True

    def handle_event(self, event):
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.text_surface = self.font.render(self.text, True, self.text_color)
                return True # Indicates text has changed
        return False

    def draw(self, screen):
        color = self.active_color if self.active else self.box_color
        if not self.enabled:
            color = (40, 40, 40)
        pygame.draw.rect(screen, color, self.rect, 2)
        screen.blit(self.text_surface, (self.rect.x + 5, self.rect.y + 5))

class Button:
    """A simple UI class for a clickable button."""
    def __init__(self, rect, text, font, text_color=(255, 255, 255), button_color=(100, 100, 100), hover_color=(120, 120, 120)):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.text_color = text_color
        self.button_color = button_color
        self.hover_color = hover_color
        self.text_surface = self.font.render(self.text, True, self.text_color)
        self.text_rect = self.text_surface.get_rect(center=self.rect.center)
        self.enabled = True

    def handle_event(self, event) -> bool:
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, screen):
        color = self.button_color
        if self.enabled and self.rect.collidepoint(pygame.mouse.get_pos()):
            color = self.hover_color
        if not self.enabled:
            color = (40, 40, 40)

        pygame.draw.rect(screen, color, self.rect)
        screen.blit(self.text_surface, self.text_rect)

# --- Core Inference and Model Logic (from inference script) ---
# Note: Functions are slightly adapted for interactive use.

def get_checkpoint_weights(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in ckpt: return ckpt["model"]
    if "module" in ckpt: return ckpt["module"]
    raise KeyError("Checkpoint state_dict does not contain 'model' or 'module' keys.")

def load_model_and_checkpoint(args, config):
    """Loads the model and checkpoint, returning the model object."""
    from models.internvideo2_clip_small import InternVideo2_CLIP_small

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InternVideo2_CLIP_small(config).to(device)

    ckpt_path = args.checkpoint_file
    if ckpt_path is None:
        print(f"Downloading {args.hf_checkpoint_file} from Hugging Face...")
        ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_checkpoint_file)

    print("Loading checkpoint weights...")
    state_dict = get_checkpoint_weights(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected: print(f"Warning: Unexpected keys in state_dict: {unexpected[:5]}...")
    if missing: print(f"Warning: Missing keys in state_dict: {missing[:5]}...")

    model.eval()
    print("Model loaded successfully.")
    return model, device

def get_skip_parameters(frame_idx: int, args, use_spfs: bool) -> Tuple[float, int]:
    """Determine skip parameters for current frame."""
    if "uniform" in args.mode and args.sampling_rate > 1:
        if (frame_idx - 7) % args.sampling_rate != (args.sampling_rate - 1):
            return -1e6, 10**6  # Force skip
    if use_spfs:
        return args.confidence_threshold, args.max_consecutive_skips
    return 1.0, 0 # No skipping

def run_inference_generator(model, device, frames, phrase, config, args, mode_config):
    """
    A generator function that runs inference and yields the updated logits list at each frame.
    This allows the UI to update in real-time without freezing.
    """
    from demo.utils import retrieve_text, frames2tensor

    use_spfs = mode_config["use_spfs"]
    logits_list_curr = []

    def get_frame_tensor(frame):
        frame_size = config.get("size_t", 224)
        return frames2tensor([frame], fnum=1, target_size=(frame_size, frame_size), device=device).squeeze(0)

    # Model-specific setup
    if args.mode == "mobileclip":
        vit = model.streaming_vision_encoder.vit_lite
        txt_encoder = model.text_encoder
        text_emb = txt_encoder.encode_text(model.tokenizer(phrase).to(device)).squeeze(0)
        text_emb = text_emb / (text_emb.norm() + 1e-12)
    elif args.mode != "internvideo2": # StreamMamba / LSTM
        curr_hidden_state = model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)

    # Main inference loop
    for frame_idx in range(len(frames)):
        logit = 0.0

        if args.mode == "mobileclip":
            raw_feat, _ = vit.extract_features(get_frame_tensor(frames[frame_idx]))
            img_emb = vit.classifier(raw_feat).squeeze(0)
            img_emb = img_emb / (img_emb.norm() + 1e-12)
            logit = float(torch.dot(img_emb, text_emb).item())

        elif args.mode == "internvideo2":
            if frame_idx >= 7:
                window_frames = frames[frame_idx - 7 : frame_idx + 1]
                _, probs = retrieve_text(window_frames, [phrase], model=model, topk=1, config=config, device=device)
                logit = float(probs[0])

        else: # StreamMamba / LSTM
            if frame_idx < 7: # Warmup phase
                _, curr_hidden_state, _ = model.streaming_vision_encoder(
                    get_frame_tensor(frames[frame_idx]), curr_hidden_state, 1.0, 0)
            else:
                conf, mcs = get_skip_parameters(frame_idx, args, use_spfs)
                vid_feat, curr_hidden_state, _ = model.get_streaming_vid_feat(
                    get_frame_tensor(frames[frame_idx]), curr_hidden_state, conf, mcs)
                text_feat = model.get_txt_feat(phrase)
                probs, _ = model.predict_label(vid_feat, text_feat, top=1)
                logit = float(probs.item())

        logits_list_curr.append(logit)
        yield list(logits_list_curr) # Yield a copy of the list so far

# --- Visualization and Data Handling (from viz script) ---

def json_read(path: str):
    import json
    with open(path, "r") as f: return json.load(f)

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

def normalize_dataset(dataset_name: str, raw_dataset: list) -> list:
    dataset = []
    if dataset_name.lower() == "flash":
        for item in raw_dataset:
            video_path, peaks = item
            for peak in peaks:
                build_up, peak_start, peak_end, drop_off = int(peak["build_up"]), int(peak["peak_start"]), int(peak["peak_end"]), int(peak["drop_off"])
                caption = str(peak.get("caption", "")).strip()
                rel_start = max(1, peak_start - build_up + 1)
                rel_end = min(drop_off - build_up + 1, peak_end - build_up + 1)
                rel_gt = list(range(rel_start, rel_end + 1)) if rel_end >= rel_start else []
                dataset.append((video_path, caption, rel_gt, build_up, drop_off))
    else: # act75
        dataset = raw_dataset
    return dataset

def load_video_frames(video_path: str) -> Tuple[Optional[List[pygame.Surface]], int, int, int]:
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None, 0, 0, 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return None, 0, 0, 0

    frames, width, height = [], int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame_bgr = cap.read()
        if not ret: break
        # We need the original CV2 frames for the model, and PyGame surfaces for display
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames, width, height, len(frames)

def scale_points(points: List[Tuple[int, float]], rect: pygame.Rect) -> List[Tuple[int, int]]:
    if not points: return []
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min: x_max += 1
    if abs(y_max - y_min) < 1e-6: y_max += 1.0

    return [
        (int(rect.left + (x - x_min) / (x_max - x_min) * rect.width),
         int(rect.bottom - (y - y_min) / (y_max - y_min) * rect.height))
        for x, y in points
    ]

def draw_graph(screen, rect, logits, current_idx_1based, gt_frames_1based=None):
    pygame.draw.rect(screen, (30, 30, 30), rect)
    pygame.draw.rect(screen, (70, 70, 70), rect, 1)

    if not logits: return

    points = [(i + 1, float(v)) for i, v in enumerate(logits)]
    scaled_points = scale_points(points, rect)

    if gt_frames_1based:
        gmin, gmax = min(gt_frames_1based), max(gt_frames_1based)
        # Use the same scaling logic to find the x-pixel coordinates for the GT band
        x_coords = [p[0] for p in scaled_points]
        gx1 = x_coords[max(0, gmin - 1)]
        gx2 = x_coords[min(len(x_coords) - 1, gmax - 1)]
        band = pygame.Surface((max(1, gx2 - gx1), rect.height), pygame.SRCALPHA)
        band.fill((0, 200, 0, 60))
        screen.blit(band, (gx1, rect.top))

    if len(scaled_points) >= 2:
        pygame.draw.lines(screen, (80, 160, 255), False, scaled_points, 2)

    ci = max(0, min(current_idx_1based - 1, len(scaled_points) - 1))
    cx, cy = scaled_points[ci]
    pygame.draw.circle(screen, (255, 60, 60), (cx, cy), 5)

# --- Argument Parsing (Combined) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Interactively visualize and re-run model logits on video clips.")

    # Model/Inference arguments
    parser.add_argument("config_dir", help="Path to training config directory (e.g., scripts/pretraining/clip/B14).")
    parser.add_argument("--checkpoint-file", default=None, help="Path to a checkpoint file. If not provided, will download from HF.")
    parser.add_argument("--hf-checkpoint-file", default="spfs_r64/ckpt_step_24500.pt", help="Checkpoint filename within the HF repo.")
    parser.add_argument("--hf-repo", default="qingy2024/InternVideo2-B14", help="HF repo to download checkpoint from.")
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--max-consecutive-skips", type=int, default=0)
    parser.add_argument("--mode", default="streammamba_spfs", choices=["streammamba_dense", "streammamba_spfs", "streammamba_spfs_uniform", "lstm", "mobileclip", "internvideo2"])
    parser.add_argument("--sampling-rate", type=int, default=2)

    # Visualization/Data arguments
    parser.add_argument("--dataset-name", default="flash", choices=["flash", "act75"])
    parser.add_argument("--index", type=int, default=0, help="Zero-based index of the example within the dataset to visualize.")
    parser.add_argument("--video-root", default="peakframe-toolkit", help="Root folder for dataset videos and JSON.")
    parser.add_argument("--scale", type=float, default=0.8, help="Scale factor for display.")

    return parser.parse_args()

# --- Main Application Logic ---
def main():
    ensure_dependencies()
    args = parse_args()
    sys.path.append(os.getcwd())

    from demo.config import Config, eval_dict_leaf

    # --- Setup: Model and Config ---
    config_path = os.path.join(args.config_dir, "config.py")
    config = eval_dict_leaf(Config.from_file(config_path))

    mode_configs = {
        "lstm": {"use_spfs": False, "rnn_type": "lstm", "checkpoint": "lstm_ckpt.pt"},
        "streammamba_spfs": {"use_spfs": True, "rnn_type": "mamba_spfs"},
        "streammamba_spfs_uniform": {"use_spfs": True, "rnn_type": "mamba_spfs"},
        "streammamba_dense": {"use_spfs": False, "rnn_type": "mamba"},
        "mobileclip": {"use_spfs": False, "rnn_type": "mobileclip"},
        "internvideo2": {"use_spfs": False, "rnn_type": "internvideo2"}
    }
    mode_config = mode_configs[args.mode]
    if "checkpoint" in mode_config: args.hf_checkpoint_file = mode_config["checkpoint"]
    if "rnn_type" in mode_config and mode_config["rnn_type"] in ["lstm", "mamba", "mamba_spfs"]:
        config.model.streaming_vision_encoder.rnn_type = mode_config["rnn_type"]

    model, device = load_model_and_checkpoint(args, config)

    # --- Setup: Data ---
    if not os.path.exists(args.video_root):
        print(f"Video root '{args.video_root}' not found. Cloning peakframe-toolkit...")
        subprocess.check_call(["git", "clone", "https://github.com/cminst/peakframe-toolkit.git"])

    ds_json_path = os.path.join(args.video_root, "data", f"{args.dataset_name.upper()}.json")
    dataset = normalize_dataset(args.dataset_name, json_read(ds_json_path))
    entry = dataset[args.index]

    video_rel, caption, gt_frames, seg_start, seg_end = (entry + (None, None))[:5]
    video_path = os.path.join(args.video_root, video_rel)

    cv2_frames, f_w, f_h, _ = load_video_frames(video_path)
    if seg_start is not None and seg_end is not None:
        cv2_frames = cv2_frames[int(seg_start) : int(seg_end) + 1]

    if not cv2_frames: sys.exit(1)
    if args.mode != "mobileclip" and len(cv2_frames) < 8:
        print(f"Error: Video must have at least 8 frames for this mode, but found {len(cv2_frames)}.")
        sys.exit(1)

    # --- Setup: PyGame UI ---
    pygame.init()
    try:
        font_small = pygame.font.Font(None, 24)
        font_large = pygame.font.Font(None, 36)
    except pygame.error:
        font_small = pygame.font.SysFont("sans", 20)
        font_large = pygame.font.SysFont("sans", 30)

    scale = max(0.1, float(args.scale))
    vid_w, vid_h = int(f_w * scale), int(f_h * scale)
    graph_w = vid_w

    # Create a control panel area above the graph
    control_h = 80
    screen_w = vid_w + graph_w
    screen_h = vid_h + control_h

    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption(f"Interactive Logits: {os.path.basename(video_path)} - {caption}")

    # Pre-convert CV2 frames to PyGame surfaces for faster blitting
    print("Preparing display frames...")
    pg_frames = [pygame.transform.smoothscale(pygame.image.frombuffer(f.tobytes(), (f_w, f_h), "RGB"), (vid_w, vid_h)) for f in cv2_frames]

    pad = 20
    graph_rect = pygame.Rect(vid_w + pad, control_h + pad, graph_w - 2 * pad, vid_h - 2*pad)

    # UI Elements
    input_box = TextInputBox((vid_w + pad, 10, graph_w - 140 - pad, 35), font_small, initial_text=caption)
    rerun_button = Button((screen_w - 120 - pad, 10, 120, 35), "Rerun Inference", font_small)

    # --- Main Loop ---
    print("\nControls:\n  Right/Left: Next/Prev Frame | Up/Down: Jump +/- 10 Frames | Home/End: Go to Start/End | Esc: Quit")

    clock = pygame.time.Clock()
    current_idx = 1  # 1-based
    logits: List[float] = []

    running = True
    needs_redraw = True
    is_running_inference = False
    inference_generator = None

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

            # Pass events to UI elements
            input_box.handle_event(event)
            if rerun_button.handle_event(event) and not is_running_inference:
                is_running_inference = True
                rerun_button.enabled = False
                input_box.enabled = False
                logits = [] # Clear previous results
                print(f"\nStarting inference for prompt: '{input_box.text}'")
                inference_generator = run_inference_generator(model, device, cv2_frames, input_box.text, config, args, mode_config)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_RIGHT: current_idx = min(len(pg_frames), current_idx + 1)
                elif event.key == pygame.K_LEFT: current_idx = max(1, current_idx - 1)
                elif event.key == pygame.K_DOWN: current_idx = min(len(pg_frames), current_idx + 10)
                elif event.key == pygame.K_UP: current_idx = max(1, current_idx - 10)
                elif event.key == pygame.K_HOME: current_idx = 1
                elif event.key == pygame.K_END: current_idx = len(pg_frames)
                needs_redraw = True

        # --- Live Inference Update ---
        if is_running_inference and inference_generator:
            try:
                logits = next(inference_generator)
                # Update the current frame to follow the inference progress
                current_idx = len(logits)
                needs_redraw = True
            except StopIteration:
                print("Inference finished.")
                is_running_inference = False
                inference_generator = None
                rerun_button.enabled = True
                input_box.enabled = True
                # Post-process logits after they are all generated
                logits = replace_leading_zeros(logits)
                needs_redraw = True

        # --- Drawing ---
        if needs_redraw:
            screen.fill((20, 20, 20))

            # Top Control Panel background
            pygame.draw.rect(screen, (40, 40, 40), (vid_w, 0, graph_w, control_h))

            # Left: Video Frame
            screen.blit(pg_frames[current_idx - 1], (0, control_h))

            # Frame index overlay
            text_surf = font_large.render(f"{current_idx}/{len(pg_frames)}", True, (255, 255, 0))
            bg_rect = pygame.Rect(5, control_h + 5, text_surf.get_width() + 10, text_surf.get_height() + 6)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 180))
            screen.blit(bg_surf, bg_rect.topleft)
            screen.blit(text_surf, (10, control_h + 8))

            # Right: UI and Graph
            input_box.draw(screen)
            rerun_button.draw(screen)
            draw_graph(screen, graph_rect, logits, current_idx, gt_frames)

            pygame.display.flip()
            needs_redraw = False

        clock.tick(60)

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
