import os
import time
import secrets
from random import randint
import subprocess
import pathlib
import modal

# ── Image ─────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim()
    # core Jupyter stack
    .pip_install(
        "httpx",
        "tqdm",
        "hf_transfer",
        "packaging",
        "ninja",
    )
    # OS‑level bits
    .run_commands(
        "apt-get update -y"
    )
    .apt_install(
        "git",
        "curl",
        "wget",
        "ffmpeg",
        "aria2"
    )
    .pip_install("huggingface_hub")
)

# ── Modal App / GPU entrypoint ───────────────────────────────────────────
app = modal.App(image=image, name="CPU I/O ops")

@app.function(timeout=86_400)
def download_k600():
    import os
    import subprocess

    os.chdir("/root/")

    # Generate a secure random token for JupyterLab access authentication.
    token = 'internvideo2'

    with modal.forward(8888) as tunnel:
        # Construct the URL to access JupyterLab, including the generated token.
        url = tunnel.url + "/?token=" + token
        print('-'*50 + '\n' + f"{url}\n"+'-'*50) # Print the URL to the console, making it easy to access JupyterLab in a browser.
        # Start JupyterLab server with specific configurations.
        subprocess.run(
            [
                "jupyter", # Command to execute JupyterLab.
                "lab", # Start JupyterLab interface.
                "--no-browser", # Prevent JupyterLab from trying to open a browser automatically.
                "--allow-root", # Allow JupyterLab to be run as root user inside the container.
                "--ip=0.0.0.0", # Bind JupyterLab to all network interfaces, making it accessible externally.
                "--port=8888", # Specify the port for JupyterLab to listen on.
                "--LabApp.allow_origin='*'", # Allow requests from any origin (for easier access from different networks).
                "--LabApp.allow_remote_access=1", # Allow remote connections to JupyterLab.
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"}, # Set environment variables, including the authentication token and shell.
            stderr=subprocess.DEVNULL, # Suppress standard error output from JupyterLab for cleaner logs.
        )

@app.local_entrypoint()
def main():
    download_k600.remote()
