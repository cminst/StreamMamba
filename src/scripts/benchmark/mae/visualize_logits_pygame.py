import argparse
import json
import os
import sys
from typing import List, Tuple, Optional
import cv2
import pygame

def json_read(path: str):
    with open(path, "r") as f:
        return json.load(f)

def replace_leading_zeros(logits: List[float]) -> List[float]:
    """Replace leading zeros with the first non-zero logit value."""
    # Find first non-zero value
    first_non_zero = None
    for val in logits:
        if val != 0:
            first_non_zero = val
            break
    
    # If no non-zero values found, return original logits
    if first_non_zero is None:
        return logits
    
    # Replace leading zeros
    new_logits = []
    replaced = False
    for val in logits:
        if not replaced and val == 0:
            new_logits.append(first_non_zero)
        else:
            new_logits.append(val)
            replaced = True
    return new_logits

def normalize_dataset(dataset_name: str, raw_dataset: list) -> list:
    """
    Normalize dataset entries into a common format:
    - ACT75: expected as [video_path, phrase, gt_frames]
    - FLASH: input format is [video_path, [peaks...]] per video.
      We expand each peak into an entry:
        (video_path, caption, relative_gt_peak_frames, build_up, drop_off)
    """
    dataset = []
    if dataset_name.lower() == "flash":
        for item in raw_dataset:
            # Each item: [video_path, [ {build_up, peak_start, peak_end, drop_off, caption}, ... ]]
            video_path, peaks = item
            for peak in peaks:
                build_up = int(peak["build_up"])     # inclusive
                peak_start = int(peak["peak_start"]) # inclusive
                peak_end = int(peak["peak_end"])     # inclusive
                drop_off = int(peak["drop_off"])     # inclusive
                caption = str(peak.get("caption", "")).strip()
                # Compute relative GT frames within [build_up, drop_off] inclusive
                # Prediction indices are 1-based relative to the segment start
                rel_start = peak_start - build_up + 1
                rel_end = peak_end - build_up + 1
                if rel_end < 1 or rel_start > (drop_off - build_up + 1):
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

def load_video_frames(video_path: str) -> Tuple[Optional[List[pygame.Surface]], int, int, int]:
    """Load all frames from a video file as Pygame surfaces (RGB)."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None, 0, 0, 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return None, 0, 0, 0
    frames: List[pygame.Surface] = []
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        surface = pygame.image.frombuffer(frame_rgb.tobytes(), (frame_width, frame_height), "RGB")
        frames.append(surface)
    cap.release()
    if not frames:
        print("Error: No frames were loaded from the video.")
        return None, 0, 0, 0
    return frames, frame_width, frame_height, len(frames)

def scale_points(points: List[Tuple[int, float]], rect: pygame.Rect) -> List[Tuple[int, int]]:
    """
    Scale time-series points (x=index starting at 1, y=logit) to fit within rect.
    Returns a list of (px, py) in integer pixels.
    """
    if not points:
        return []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # Avoid zero ranges
    if x_max == x_min:
        x_max += 1
    if abs(y_max - y_min) < 1e-6:
        y_max += 1.0
    scaled: List[Tuple[int, int]] = []
    for x, y in points:
        nx = (x - x_min) / (x_max - x_min)
        ny = (y - y_min) / (y_max - y_min)
        px = int(rect.left + nx * rect.width)
        py = int(rect.bottom - ny * rect.height)  # y grows downward
        scaled.append((px, py))
    return scaled

def draw_graph(
    screen: pygame.Surface,
    rect: pygame.Rect,
    logits: List[float],
    current_index_1based: int,
    gt_frames_1based: Optional[List[int]] = None,
):
    """Draw logits line (blue), current point (red), and GT band (green)."""
    # Background
    pygame.draw.rect(screen, (30, 30, 30), rect)
    pygame.draw.rect(screen, (70, 70, 70), rect, 1)
    # Prepare points (1-based x)
    points = [(i + 1, float(v)) for i, v in enumerate(logits)]
    # Draw GT band if provided
    if gt_frames_1based:
        gmin = min(gt_frames_1based)
        gmax = max(gt_frames_1based)
        # Scale x using same mapping as scale_points
        xs = [p[0] for p in points]
        ys = [p[1] for p in points] if points else [0.0, 1.0]
        x_min, x_max = (min(xs), max(xs)) if xs else (1, 2)
        y_min, y_max = (min(ys), max(ys)) if ys else (0.0, 1.0)
        if x_max == x_min:
            x_max += 1
        if abs(y_max - y_min) < 1e-6:
            y_max += 1.0
        def map_x(xv: int) -> int:
            nx = (xv - x_min) / (x_max - x_min)
            return int(rect.left + nx * rect.width)
        gx1 = map_x(gmin)
        gx2 = map_x(gmax)
        band = pygame.Surface((max(1, gx2 - gx1 + 1), rect.height), pygame.SRCALPHA)
        band.fill((0, 200, 0, 60))  # semi-transparent green
        screen.blit(band, (gx1, rect.top))
    # Draw line
    if points:
        scaled = scale_points(points, rect)
        if len(scaled) >= 2:
            pygame.draw.lines(screen, (80, 160, 255), False, scaled, 2)
        # Draw current red point
        ci = max(1, min(current_index_1based, len(points)))
        cx, cy = scaled[ci - 1]
        pygame.draw.circle(screen, (255, 60, 60), (cx, cy), 5)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize logits over time alongside video frames (FLASH/ACT75)."
    )
    parser.add_argument("logits_json", help="Path to logits JSON file (from benchmark run).")
    parser.add_argument("dataset_name", choices=["flash", "act75"], help="Dataset name.")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based index of the example within the logits JSON to visualize.",
    )
    parser.add_argument(
        "--video-root",
        default="peakframe-toolkit",
        help="Root folder containing dataset videos and data JSON (default: peakframe-toolkit).",
    )
    parser.add_argument(
        "--dataset-json",
        default=None,
        help="Optional explicit path to dataset JSON; overrides default path.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for display (e.g., 0.75 to shrink).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    # Load logits JSON
    all_logits = json_read(args.logits_json)
    if not isinstance(all_logits, list) or not all_logits:
        print("Error: logits JSON must be a non-empty list.")
        sys.exit(1)
    if args.index < 0 or args.index >= len(all_logits):
        print(f"Error: index {args.index} out of range [0, {len(all_logits)-1}].")
        sys.exit(1)
    # Extract the selected logits sequence; structure is list of [logit, frame_idx]
    sample_logits_pairs = all_logits[args.index]
    logits = [float(p[0]) for p in sample_logits_pairs]
    
    # Replace leading zeros with first non-zero value
    logits = replace_leading_zeros(logits)
    
    # Load dataset JSON and normalize entries
    if args.dataset_json is None:
        ds_json_name = args.dataset_name.upper().replace('-', '_') + ".json"
        args.dataset_json = os.path.join(args.video_root, "data", ds_json_name)
    if not os.path.exists(args.dataset_json):
        print(f"Error: dataset JSON not found at '{args.dataset_json}'.")
        print("Hint: Set --dataset-json explicitly, or ensure peakframe-toolkit is present.")
        sys.exit(1)
    raw_dataset = json_read(args.dataset_json)
    dataset = normalize_dataset(args.dataset_name, raw_dataset)
    if args.index < 0 or args.index >= len(dataset):
        print(f"Error: index {args.index} out of range for dataset with {len(dataset)} items.")
        sys.exit(1)
    entry = dataset[args.index]
    # Unpack entry depending on dataset variant
    if len(entry) >= 5:
        video_rel, caption, gt_frames, seg_start, seg_end = entry[:5]
    elif len(entry) == 3:
        video_rel, caption, gt_frames = entry
        seg_start, seg_end = None, None
    else:
        print("Error: Unexpected dataset entry format.")
        sys.exit(1)
    # Resolve video path
    video_path = video_rel
    if not os.path.isabs(video_path):
        video_path = os.path.join(args.video_root, video_rel)
    # Initialize pygame
    pygame.init()
    try:
        font = pygame.font.Font(None, 36)
    except pygame.error:
        font = pygame.font.SysFont("sans", 30)
    # Load frames
    frames, f_w, f_h, f_n = load_video_frames(video_path)
    if not frames:
        pygame.quit()
        sys.exit(1)
    # If FLASH with segment, crop frames
    if seg_start is not None and seg_end is not None:
        start_i = max(0, int(seg_start))
        end_i = min(f_n - 1, int(seg_end))
        frames = frames[start_i : end_i + 1]
        f_n = len(frames)
    # Sanity check: align lengths (logits and frames often match by construction)
    if len(logits) != f_n:
        print(f"Warning: logits length ({len(logits)}) != frame count ({f_n}). Graph will use logits length.")
    # Layout: left = video, right = graph of equal height
    scale = max(0.1, float(args.scale))
    left_w = int(f_w * scale)
    left_h = int(f_h * scale)
    right_w = left_w  # symmetric layout
    screen_w = left_w + right_w
    screen_h = left_h
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.key.set_repeat(250, 50)  # Wait 250ms, then repeat every 50ms
    pygame.display.set_caption(f"{caption} - {os.path.basename(video_path)} [{args.dataset_name}] (# {args.index})")
    # Pre-scale frames to target size for faster blits
    scaled_frames = [
        pygame.transform.smoothscale(frm, (left_w, left_h)) for frm in frames
    ]
    # Graph area on the right with some padding
    pad = 20
    graph_rect = pygame.Rect(left_w + pad, pad, right_w - 2 * pad, screen_h - 2 * pad)
    # Controls
    print("\nControls:")
    print("  Right/Left: next/prev frame")
    print("  Up/Down: +/- 10 frames")
    print("  Home/End: jump to start/end")
    print("  Esc or Close: quit")
    clock = pygame.time.Clock()
    current_idx = 1  # 1-based
    running = True
    needs_redraw = True
    total_frames_display = min(len(logits), len(scaled_frames))
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RIGHT:
                    if current_idx < total_frames_display:
                        current_idx += 1
                        needs_redraw = True
                elif event.key == pygame.K_LEFT:
                    if current_idx > 1:
                        current_idx -= 1
                        needs_redraw = True
                elif event.key == pygame.K_UP:
                    new_idx = max(1, current_idx - 10)
                    if new_idx != current_idx:
                        current_idx = new_idx
                        needs_redraw = True
                elif event.key == pygame.K_DOWN:
                    new_idx = min(total_frames_display, current_idx + 10)
                    if new_idx != current_idx:
                        current_idx = new_idx
                        needs_redraw = True
                elif event.key == pygame.K_HOME:
                    if current_idx != 1:
                        current_idx = 1
                        needs_redraw = True
                elif event.key == pygame.K_END:
                    if current_idx != total_frames_display:
                        current_idx = total_frames_display
                        needs_redraw = True
        if needs_redraw:
            screen.fill((0, 0, 0))
            # Left: current frame + overlay index
            frame_surface = scaled_frames[current_idx - 1]
            screen.blit(frame_surface, (0, 0))
            text_content = f"Frame: {current_idx} / {total_frames_display}"
            text_surface = font.render(text_content, True, (255, 255, 0))
            text_bg_rect = pygame.Rect(5, 5, text_surface.get_width() + 10, text_surface.get_height() + 6)
            bg_surface = pygame.Surface((text_bg_rect.width, text_bg_rect.height), pygame.SRCALPHA)
            bg_surface.fill((0, 0, 0, 180))
            screen.blit(bg_surface, (text_bg_rect.left, text_bg_rect.top))
            screen.blit(text_surface, (10, 8))
            # Right: graph
            draw_graph(
                screen,
                graph_rect,
                logits[:total_frames_display],
                current_idx,
                gt_frames,
            )
            pygame.display.flip()
            needs_redraw = False
        clock.tick(60)
    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
