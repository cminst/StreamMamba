import os
import time
import secrets
from random import randint
import subprocess
import pathlib
import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.10"
    )
    .pip_install(
        "jupyterlab",
        "jupyter",
        "ipywidgets",
    )
    .pip_install(
        "openai",
        "httpx",
        "tqdm",
        "nest_asyncio",
    )
    .pip_install("hf_transfer")
    .pip_install(
        "packaging",
        "ninja",
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "wandb",
    )
    .pip_install(
        "bitsandbytes",
        "accelerate",
        "peft",
        "trl",
        "triton",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "datasets",
    )
    .run_commands("pip install xformers==0.0.29")
    .run_commands(
        "apt-get update -y",
        "apt-get install -y git",
        "apt-get install -y curl",
        "apt-get install -y ffmpeg"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_TOKEN": os.environ['HF_TOKEN']
    })
    .run_commands(
        "pip uninstall -y torch torchvision"
    )
    .run_commands(
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        'deepspeed',
        'timm',
        'open_clip_torch',
        'scipy',
        'librosa',
        'av',
        'opencv-python',
        'decord',
        'imageio',
        'easydict',
        'termcolor',
        'wheel'
    )
    .run_commands(
        "pip install torchaudio --no-deps --index-url https://download.pytorch.org/whl/cu121"
    )
    .run_commands(
        'pip install flash-attn --no-build-isolation'
    )
    .run_commands(
        f"echo {time.time()}"
    )
    .run_commands(
        "cd /root && git clone https://github.com/qingy1337/IV2.git",
        "cd /root/IV2 && git checkout window",
    )
    .run_commands(
        "huggingface-cli download OpenGVLab/InternVideo2_distillation_models stage1/B14/B14_dist_1B_stage2/pytorch_model.bin --local-dir /root/StreamMamba/models",
        "huggingface-cli download OpenGVLab/InternVideo2_distillation_models clip/B14/pytorch_model.bin --local-dir /root/StreamMamba/models",
        "curl -L https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt -o /root/StreamMamba/models/mobileclip_blt.pt"
    )
)

k600_volume = modal.Volume.from_name("k600")

app = modal.App(image=image, name="Window InternVideo2 Training")

@app.function(volumes={"/root/k600": k600_volume}, gpu="A100-80GB:1", timeout=86_400)
def runwithgpu():
    token = secrets.token_urlsafe(13)
    with modal.forward(8888) as tunnel:
        url = f"{tunnel.url}/?token={token}"
        print("-" * 50 + f"\n{url}\n" + "-" * 50)

        subprocess.run(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                "--port=8888",
                "--LabApp.allow_origin='*'",
                "--LabApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
            stderr=subprocess.DEVNULL,
        )

@app.local_entrypoint()
def main():
    runwithgpu.remote()
