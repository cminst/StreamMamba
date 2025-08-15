import cv2
import pygame
import sys
import os
import argparse

def load_video_frames(video_path):
    """
    Loads all frames from a video file into a list of Pygame surfaces.
    Returns a list of frames, width, height, and total frame count.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None, 0, 0, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'. Check codec support.")
        return None, 0, 0, 0

    frames = []
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Loading video frames... This might take a while for large videos.")
    frame_count = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pygame_surface = pygame.image.frombuffer(frame_rgb.tobytes(), (frame_width, frame_height), "RGB")
        frames.append(pygame_surface)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Loaded {frame_count} frames...")

    cap.release()
    print(f"Successfully loaded {frame_count} frames.")

    if not frames:
        print("Error: No frames were loaded from the video.")
        return None, 0, 0, 0

    return frames, frame_width, frame_height, len(frames)

def main():
    parser = argparse.ArgumentParser(description="Display video frames with navigation.")
    parser.add_argument("video_path", help="Path to the MP4 or MOV video file.")
    args = parser.parse_args()

    video_path = args.video_path

    # Initialize Pygame
    pygame.init()

    # Enable key repeat:
    # pygame.key.set_repeat(delay, interval)
    # delay: time in milliseconds before the first KEYDOWN event is sent
    # interval: time in milliseconds between subsequent KEYDOWN events
    pygame.key.set_repeat(250, 50)  # Wait 250ms, then repeat every 50ms

    # Load frames
    frames, frame_width, frame_height, total_frames = load_video_frames(video_path)

    if not frames:
        pygame.quit()
        sys.exit("Exiting due to video loading error.")

    # Set up the display
    screen_height_adjusted = frame_height # Adjust if you want more space for UI later
    screen = pygame.display.set_mode((frame_width, screen_height_adjusted))
    pygame.display.set_caption(f"Frame Viewer - {os.path.basename(video_path)}")

    # Font for displaying frame number
    try:
        font = pygame.font.Font(None, 36)
    except pygame.error:
        print("Warning: Default font not found. Using fallback.")
        font = pygame.font.SysFont("sans", 30)

    current_frame_index = 0
    running = True
    needs_redraw = True

    clock = pygame.time.Clock()

    print("\nControls:")
    print("  Right Arrow (hold for continuous): Next Frame")
    print("  Left Arrow (hold for continuous): Previous Frame")
    print("  Escape or Close Window: Quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN: # This event will now repeat if key is held
                if event.key == pygame.K_RIGHT:
                    if current_frame_index < total_frames - 1:
                        current_frame_index += 1
                        needs_redraw = True
                elif event.key == pygame.K_LEFT:
                    if current_frame_index > 0:
                        current_frame_index -= 1
                        needs_redraw = True
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if needs_redraw:
            screen.blit(frames[current_frame_index], (0, 0))

            text_content = f"Frame: {current_frame_index + 1} / {total_frames}"
            text_surface = font.render(text_content, True, (255, 255, 0))

            text_bg_rect = pygame.Rect(5, 5, text_surface.get_width() + 10, text_surface.get_height() + 6)
            bg_surface = pygame.Surface((text_bg_rect.width, text_bg_rect.height), pygame.SRCALPHA)
            bg_surface.fill((0, 0, 0, 180))
            screen.blit(bg_surface, (text_bg_rect.left, text_bg_rect.top))
            screen.blit(text_surface, (10, 8))

            pygame.display.flip()
            needs_redraw = False

        clock.tick(60) # Increased tick rate slightly for smoother repeated key response

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
