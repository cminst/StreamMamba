import os
import secrets
from random import randint
import subprocess
import pathlib
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.10")
    .pip_install("jupyterlab")
    .pip_install("ipywidgets")
    .pip_install("hf_transfer")
    .pip_install("jupyter")
    .pip_install('packaging')
    .pip_install('ninja')
    .run_commands(
        "apt-get update -y",
        "apt-get install git -y",
        "apt-get install curl -y"
    )
    .pip_install('torch')
    .pip_install('torchvision')
    .pip_install('numpy')
    .pip_install('wandb')
    .pip_install('pandas')
    .pip_install('tensorboard')
    .run_commands('pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.7.4.post1+cu126torch2.7-cp310-cp310-linux_x86_64.whl')
    .pip_install('flash-attn')
    .pip_install('huggingface_hub')
)

image = image.env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_TOKEN": os.environ['HF_TOKEN'],
    "WANDB_API_KEY": os.environ['WANDB_API_KEY']
})

image = image.run_commands(
    "apt-get update -y",
    "apt-get install git curl -y",
)

image = image.run_commands(
    "huggingface-cli download OpenGVLab/InternVideo2-Stage2_1B-224p-f4 --local-dir /root/",
)

image = image.run_commands(
    "cd /root/ && git clone https://github.com/qingy1337/IV2.git",
)

image = image.run_commands(
    "pip install opencv-python tabulate",
    "apt-get update",
    "apt-get install ffmpeg libsm6 libxext6 -y"
)

app = modal.App(image=image, name="InternVideo2 Experiments")

@app.function(gpu="A100-80GB:1", timeout=86400)
def runwithgpu():
    import os
    import subprocess

    os.chdir("/root/IV2")

    commands = """
    git pull
    git checkout fix-iv2-1b
    pip install -r reqs.txt
    """.strip().splitlines()

    for line in commands:
        subprocess.run(line.strip().split(), check=True)

    pass

    token = 'jupyter'
    with modal.forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        print('-' * 50 + '\n' + f"{url}\n" + '-' * 50)
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
