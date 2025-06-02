import os
from typing import Iterable
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.10")
    .run_commands(
        "apt-get update -y",
        "apt-get install -y git curl ffmpeg libsm6 libxext6",
        "git clone https://github.com/qingy1337/IV2.git /app/IV2",  # Clone the repo
    )
    .run_commands(
        "curl -s -o reqs.txt https://raw.githubusercontent.com/qingy1337/IV2/refs/heads/main/reqs.txt && pip install -r reqs.txt",
        "rm reqs.txt"
    )
)

app = modal.App(name="Slim-Kinetics Embeddings", image=image)

slim_volume = modal.Volume.from_name("slim-kinetics", create_if_missing=True)

@app.function(gpu="A100-40GB:1", volumes={"/data": slim_volume}, memory=48000, timeout=6000)
def embed_video(video_path: str):
    """Embed a single video file and store result alongside it."""
    import torch
    import decord
    from torchvision import transforms # Though not used in this snippet, kept from original
    from pathlib import Path
    import os # Added for os.chdir

    original_cwd = os.getcwd()
    # Change to the directory where model loading expects to be
    # This path assumes the repo is cloned into /app/IV2 as specified in the image definition
    intern_video_multi_modality_path = "/app/IV2/InternVideo2/multi_modality/"
    os.chdir(intern_video_multi_modality_path)

    # Now import _load_model, which should work from the new CWD
    # The import path is relative to the new CWD
    from tasks_clip.gather_embeddings import _load_model

    model = _load_model()

    # Restore the original CWD after model loading
    os.chdir(original_cwd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Ensure model is on the correct device after loading

    # Video processing logic (paths should be absolute, so CWD change doesn't affect them)
    vr = decord.VideoReader(video_path)
    frames = vr.get_batch(range(len(vr)))
    frames = frames.permute(0, 3, 1, 2)  # T H W C -> T C H W
    frames = frames.float() / 255.0
    # Take sliding windows of 4 frames as in gather_embeddings
    windows = [frames[i - 3:i + 1] for i in range(3, len(frames))]
    save_dict = {}
    with torch.no_grad():
        for idx, window in enumerate(windows):
            window = window.unsqueeze(0).to(device)
            feat = model.get_vid_feat(window)
            save_dict[idx + 3] = feat.cpu()

    # Output paths are absolute, so they are not affected by CWD changes
    out_dir = Path("/data/embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / (Path(video_path).stem + ".pt")
    torch.save(save_dict, save_path)

@app.local_entrypoint()
def main(json_path: str):
    import json
    from pathlib import Path
    from typing import List
    from tqdm import tqdm

    json_file_path = Path(json_path)
    if not json_file_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return
    except Exception as e:
        print(f"Error reading file {json_path}: {e}")
        return

    video_files: List[str] = []

    # The JSON file lists paths relative to the directory where the videos are stored.
    # Assuming this video data directory structure is present within the /data volume mount
    # inside the Modal container.
    for entry in tqdm(data):
        if isinstance(entry, dict) and "video" in entry and isinstance(entry["video"], str):
            relative_video_path = entry["video"]
            # Construct the path as accessed within the container's /data mount
            container_video_path = str(Path("/data") / "kinetics-dataset" / 'k600' / 'train' / 'train' / relative_video_path)
            video_files.append(container_video_path)
        else:
             print(f"Warning: Skipping invalid entry in JSON: {entry}")


    if not video_files:
        print("No video files found in the JSON.")
        return

    print(f"Found {len(video_files)} videos to process.")
    embed_video.spawn_map(video_files)
