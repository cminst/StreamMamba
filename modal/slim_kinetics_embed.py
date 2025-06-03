import os
from pathlib import Path
import modal

# --- Modal Image Definition ---
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.10")
    .run_commands(
        "apt-get update -y",
        "apt-get install -y git curl ffmpeg libsm6 libxext6",
        "git clone https://github.com/qingy1337/IV2.git /app/IV2",
    )
    .run_commands(
        "cd /app/IV2 && git checkout delta_6b_server"
    )
    .run_commands(
        "curl -s -o reqs.txt https://raw.githubusercontent.com/qingy1337/IV2/refs/heads/main/reqs.txt && pip install -r reqs.txt",
        "rm reqs.txt"
    )
    .run_commands(
        "huggingface-cli download qingy2024/InternVideo2_S2_6B_Vision InternVideo2_S2_6B_vision.pt"
    )
    .pip_install(
        'librosa',
        'decord',
        'imageio',
        'easydict',
        'open-clip-torch',
        'torch',
        'av',
        'opencv-python',
        'torchaudio',
    )
)

app = modal.App(name="Slim-Kinetics Embeddings", image=image)
slim_volume = modal.Volume.from_name("slim-kinetics", create_if_missing=False)

@app.function(gpu="A100-40GB:1", volumes={"/data": slim_volume}, memory=48000, timeout=6000)
def embed_video(video_path: str):
    """
    Embed a single video file (path) and store result alongside it.

    ====================================================================

    Output Folder Structure (data is the slim kinetics volume)

    /data/embeddings
        video1.pt
        ├─ 3: {dict}
        │   ├─ "raw": tensor[...] (raw features from the model)
        │   ├─ "proj": tensor[...] (projected features)
        │   └─ "final": tensor[...] (normalized features)
        ├─ 4: {dict}
        │   ├─ "raw": tensor[...]
        │   ├─ "proj": tensor[...]
        │   └─ "final": tensor[...]
        └─ ... (continues for each frame index)
        video2.pt
        ├─ 3: {dict}
        │   ├─ "raw": tensor[...]
        │   ├─ "proj": tensor[...]
        │   └─ "final": tensor[...]
        └─ ... (similar structure as video1.pt)
        video3.pt
        └─ ... (similar structure)
        ...
    """
    import torch
    import cv2
    import decord
    import sys
    import numpy as np

    # Change to the directory where model loading expects to be
    intern_video_multi_modality_path = "/app/IV2/InternVideo2/multi_modality/"
    original_cwd = os.getcwd()
    os.chdir(intern_video_multi_modality_path)

    sys.path.append(os.getcwd()) # Allow imports from here

    from tasks_clip.gather_embeddings import _load_model
    from demo.utils import _frame_from_video, frames2tensor

    model = _load_model()
    os.chdir(original_cwd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    frames = [x for x in _frame_from_video(cv2.VideoCapture(video_path))]
    save_dict = {}

    with torch.no_grad():
        for frame_idx in range(4, len(frames) + 1):
            window = frames[frame_idx - 4 : frame_idx]
            frames_tensor = frames2tensor(window, fnum=4, target_size=(224, 224), device=device)
            raw_features = model.encode_vision(frames_tensor)
            projected_features = model.vision_proj(raw_features)
            final_features = projected_features / projected_features.norm(dim=-1, keepdim=True)
            save_dict[frame_idx - 1] = {
                "raw": raw_features.cpu(),
                "proj": projected_features.cpu(),
                "final": final_features.cpu(),
            }

    out_dir = Path("/data/embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / (Path(video_path).stem + ".pt")
    torch.save(save_dict, save_path)

@app.local_entrypoint()
def main(json_path: str):
    """
    Entrypoint for local execution. Reads a JSON file listing video paths and spawns embedding jobs.
    """
    import json
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

    video_files = []
    for entry in tqdm(data):
        if isinstance(entry, dict) and "video" in entry and isinstance(entry["video"], str):
            relative_video_path = entry["video"]
            container_video_path = str(Path("/data/k600/train/train") / relative_video_path)
            video_files.append(container_video_path)
        else:
            print(f"Warning: Skipping invalid entry in JSON: {entry}")

    if not video_files:
        print("No video files found in the JSON.")
        return

    video_files = [video_files[0]]
    print(f"===== Found {len(video_files)} videos to process. =====")
    print(f"Starting the SPAWN")
    embed_video.spawn_map(video_files)
