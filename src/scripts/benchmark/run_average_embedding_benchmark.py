import argparse
from tqdm import tqdm
import os
import sys
import subprocess


def ensure_dependencies():
    try:
        import torch  # noqa: F401
    except Exception:
        print("Installing dependencies...")
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
    print("Installed packages")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark average StreamMamba embedding for longer windows",
    )
    parser.add_argument(
        "config_dir",
        help="Path to training config directory, e.g. scripts/pretraining/clip/B14",
    )
    parser.add_argument(
        "--output-json",
        default="average_embedding_results.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint file. If not provided, will download from HF",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="spfs_r64/ckpt_step_24500.pt",
        help="Checkpoint filename within the HF repo",
    )
    parser.add_argument(
        "--hf-repo",
        default="qingy2024/InternVideo2-B14",
        help="HuggingFace repo to download checkpoint from",
    )
    parser.add_argument(
        "--window-sizes",
        default="10,16,24,30",
        help="Comma separated list of window sizes to evaluate",
    )
    return parser.parse_args()



def cosine_distance(a, b):
    import torch
    return 1 - torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()


def streammamba_embedding(frames, model, device, size):
    from demo.utils import frames2tensor
    hidden = model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)
    emb = None
    for f in frames:
        tensor = frames2tensor([f], fnum=1, target_size=(size, size), device=device).squeeze(0)
        emb, hidden, _ = model.streaming_vision_encoder(
            tensor,
            hidden,
            confidence_threshold=1.0,
            max_consecutive_skips=0,
        )

    vision_embeds_aligned = model.vision_align(emb)
    
    vision_embeds_aligned /= vision_embeds_aligned.norm(dim=-1, keepdim=True)
    return vision_embeds_aligned.squeeze(0).cpu()


def teacher_embedding(frames, model, device, size):
    from demo.utils import frames2tensor
    tensor = frames2tensor(frames, fnum=8, target_size=(size, size), device=device)
    emb = model.get_vid_feat(tensor).squeeze(0)
    return emb.cpu()


def main():
    ensure_dependencies()
    args = parse_args()

    sys.path.append(os.getcwd())
    
    output_dir = os.path.join(os.getcwd(), "results_average")
    args.output_json = os.path.join(output_dir, os.path.basename(args.output_json))
    os.makedirs(output_dir, exist_ok=True)

    from demo.config import Config, eval_dict_leaf
    from demo.utils import _frame_from_video
    from models.internvideo2_clip_small import InternVideo2_CLIP_small
    from huggingface_hub import hf_hub_download
    from iv2_utils.iv2 import json_read, json_write
    import torch
    import cv2
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "photography-model" not in os.listdir("."):
        subprocess.check_call([
            "git",
            "clone",
            "https://github.com/ruo2019/photography-model.git",
        ])

    config = Config.from_file(os.path.join(args.config_dir, "config.py"))
    config = eval_dict_leaf(config)

    model = InternVideo2_CLIP_small(config)
    model.to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        print(f"Downloading {args.checkpoint_file} from Hugging Face...")
        ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.checkpoint_file)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("module"))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    del ckpt
    del state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    act75_data = json_read("photography-model/data/ACT75.json")
    size_t = config.get("size_t", 224)
    window_sizes = [int(x) for x in args.window_sizes.split(",")]

    results = {N: {"d_avg": 0.0, "d_tail": 0.0, "count": 0} for N in window_sizes}

    with torch.no_grad():
        for video_path, _, _ in tqdm(act75_data):
            cap = cv2.VideoCapture(os.path.join("photography-model", video_path))
            frames = [x for x in _frame_from_video(cap)]
            if not frames:
                continue
            for N in window_sizes:
                if len(frames) < N:
                    continue
                for start in range(len(frames) - N + 1):
                    clip = frames[start:start + N]
                    teacher = teacher_embedding(clip, model, device, size_t)
                    avg_embeds = []
                    for k in range(N - 7):
                        sub = clip[k:k + 8]
                        avg_embeds.append(streammamba_embedding(sub, model, device, size_t))
                    avg_embed = torch.stack(avg_embeds).mean(dim=0)
                    tail_embed = streammamba_embedding(clip[-8:], model, device, size_t)
                    d_avg = cosine_distance(avg_embed, teacher)
                    d_tail = cosine_distance(tail_embed, teacher)
                    results[N]["d_avg"] += d_avg
                    results[N]["d_tail"] += d_tail
                    results[N]["count"] += 1

    for N in window_sizes:
        if results[N]["count"] > 0:
            results[N]["d_avg"] /= results[N]["count"]
            results[N]["d_tail"] /= results[N]["count"]

    json_write(results, args.output_json)
    print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
