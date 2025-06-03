import modal

# Base image with minimal utilities for downloading from Hugging Face
image = (
    modal.Image.debian_slim()
    .pip_install(
        "huggingface_hub",
        "hf_transfer",
        "pillow",
        "iv2_utils"
    )
    .apt_install("git", "curl", "ffmpeg", "aria2")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_TOKEN": "lIpxcDdlfsJzTIAqZZAvGUKaYdlyOPrpLI_fh"[::-1]
    })
)

app = modal.App(name="Slim-Kinetics Extractor (from .gz)", image=image)

# Persistent volume to store the dataset
slim_volume = modal.Volume.from_name("slim-kinetics")

@app.function(volumes={"/root/slim_kinetics": slim_volume}, timeout=86_400)
def extract_kinetics():
    import os
    import subprocess
    from pathlib import Path
    from iv2_utils.iv2 import json_read
    import gzip
    import shutil

    home_dir = "/root/slim_kinetics" # kinetics-dataset/ folder is in here

    os.chdir(home_dir)

    kinetics_base_path = Path(home_dir) / 'kinetics-dataset' / 'k600' / 'train' / 'train'

    actions = list(filter(lambda x: x.endswith('.gz') and not x.startswith('.'), os.listdir(kinetics_base_path)))

    for idx, action_class in enumerate(actions):
        # Create an extraction directory for the action class
        # More robust extension handling
        if action_class.endswith('.tar.gz'):
            action_class_name = action_class.replace('.tar.gz', '')
        else:
            action_class_name = action_class.replace('.gz', '')

        extraction_dir = kinetics_base_path / action_class_name

        # Make sure the directory exists
        extraction_dir.mkdir(exist_ok=True, parents=True)

        # Extract the tar/folder content directly to the directory
        # Use subprocess to extract the gzipped archive
        subprocess.run(['tar', '-xzf', kinetics_base_path / action_class, '-C', str(extraction_dir)], check=True)

        print(f"[{idx+1}/{len(actions)}] [{((idx+1)/len(actions))*100:.2f}%] Extracted {action_class} to {extraction_dir}.")

    slim_volume.commit()

@app.local_entrypoint()
def main():
    extract_kinetics.remote()
