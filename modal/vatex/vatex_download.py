import modal
import subprocess
import os

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.10"
    )
    .pip_install(
        "httpx",
        "tqdm",
        "huggingface_hub",
        "hf_transfer",
        "packaging",
        "ninja",
        "jsonlines",
    )
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
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_TOKEN": os.environ['HF_TOKEN']
    })
)


# Create the modal app + volume for the dataset
app = modal.App(image=image, name="vatex-dataset-download")
vatex_volume = modal.Volume.from_name("vatex-dataset", create_if_missing=True)

@app.function(
    volumes={"/data": vatex_volume},
    timeout=86_400,
    cpu=16.0
)
def download_and_process_vatex(cookies_texts: list[str], hf_dataset_name: str):
    """Download VaTeX dataset using video2dataset and upload to HuggingFace."""

    json_url = "https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json"
    local_json_path = "/data/vatex_training_v1.0.json"

    print(f"Downloading {json_url} to {local_json_path}...")
    subprocess.run(["wget", "-O", local_json_path, json_url], check=True)
    print("Download complete.")

    cookies_paths = []
    for i, text in enumerate(cookies_texts):
        cookies_path = f"/data/cookies_{i}.txt"
        with open(cookies_path, "w") as f:
            f.write(text)
        cookies_paths.append(cookies_path)
    cookies_arg = ",".join(cookies_paths)
    print(f"Wrote cookies to {cookies_arg}")

    print("Cloning and installing video2dataset...")
    subprocess.run(["git", "clone", "https://github.com/qingy1337/video2dataset"], check=True)
    os.chdir("video2dataset")
    subprocess.run(["pip", "install", "-e", "."], check=True)
    print("video2dataset installed.")

    output_folder_in_volume = "/data/dataset"
    print(f"Running video2dataset, outputting to {output_folder_in_volume}...")
    video2dataset_command = [
        "video2dataset",
        f"--url_list={local_json_path}",
        "--input_format=json",
        "--url_col=videoID",
        "--caption_col=enCap",
        f"--output_folder={output_folder_in_volume}",
        f"--cookies_file={cookies_arg}",
    ]
    subprocess.run(video2dataset_command, check=True)
    print("video2dataset command finished.")

    print("Committing vatex-dataset volume...")
    vatex_volume.commit()
    print("Volume committed. VaTeX dataset is now persisted.")

    print("Uploading dataset to Huggingface")
    huggingface_upload_command = [
        "huggingface-cli",
        "upload",
        hf_dataset_name,
        output_folder_in_volume,
        "--repo-type=dataset"
    ]
    subprocess.run(huggingface_upload_command, check=True)
    print(f"\n\n{'-'*50}\n\nFinished uploading dataset: https://huggingface.co/datasets/{hf_dataset_name}")

@app.local_entrypoint()
def main(cookies_file_paths: str, hf_dataset_name: str):
    print("Starting script!")
    print(f"  > Cookies file paths: {cookies_file_paths}")
    print(f"  > Output HF dataset name: {hf_dataset_name}")

    paths = [p.strip() for p in cookies_file_paths.split(',') if p.strip()]
    cookies_texts = []
    for path in paths:
        with open(path, 'r') as f:
            cookies_texts.append(f.read())

    download_and_process_vatex.remote(cookies_texts, hf_dataset_name)
