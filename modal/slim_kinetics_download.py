import modal

# Base image with minimal utilities for downloading from Hugging Face
image = (
    modal.Image.debian_slim()
    .pip_install(
        "huggingface_hub",
        "hf_transfer",
    )
    .apt_install("git", "curl", "ffmpeg", "aria2")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_TOKEN": "lIpxcDdlfsJzTIAqZZAvGUKaYdlyOPrpLI_fh"[::-1]
    })
)

app = modal.App(name="Slim-Kinetics Downloader", image=image)

# Persistent volume to store the dataset
slim_volume = modal.Volume.from_name("slim-kinetics", create_if_missing=True)

@app.function(volumes={"/root/slim_kinetics": slim_volume}, timeout=86_400)
def download_slim():
    import os
    import subprocess

    os.chdir("/root/slim_kinetics")
    subprocess.run(
        [
            "huggingface-cli",
            "download",
            "qingy2024/Slim-Kinetics-2",
            "--local-dir",
            ".",
            "--repo-type",
            "dataset",
        ],
        check=True,
    )
    slim_volume.commit()

@app.local_entrypoint()
def main():
    download_slim.remote()
