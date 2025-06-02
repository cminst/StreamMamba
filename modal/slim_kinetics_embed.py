import os
from typing import Iterable
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.10")
    .run_commands(
        "apt-get update -y",
        "apt-get install -y git curl ffmpeg",
    )
    .run_commands(
        "curl -s -o reqs.txt https://raw.githubusercontent.com/qingy1337/IV2/refs/heads/main/reqs.txt && pip install -r reqs.txt",
        "rm reqs.txt"
    )
)

app = modal.App(name="Slim-Kinetics Embeddings", image=image)

slim_volume = modal.Volume.from_name("slim-kinetics", create_if_missing=True)

@app.function(gpu="A100-40GB:1", volumes={"/data": slim_volume}, timeout=600)
def embed_video(video_path: str):
    """Embed a single video file and store result alongside it."""
    from InternVideo2.multi_modality.tasks_clip.gather_embeddings import _load_model
    import torch
    import decord
    from torchvision import transforms
    from pathlib import Path

    model = _load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    out_dir = Path("/data/embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / (Path(video_path).stem + ".pt")
    torch.save(save_dict, save_path)

@app.local_entrypoint()
def main(data_dir: str = "/data"):
    video_files: Iterable[str] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                video_files.append(os.path.join(root, f))

    embed_video.spawn_map(video_files)
