import os
import re
import time
import shutil
from huggingface_hub import login, create_repo, upload_folder, HfFolder
from pathlib import Path

BASE_MODEL_NAME = "OpenGVLab/InternVideo2_distillation_models"
TARGET_REPO_NAME = "qingy2024/Window-IV2-CKPT"

TOTAL_STEPS = 24912

CHECKPOINT_DIR_PATTERN = re.compile(r"^ckpt_iter(\d+)\.pth$")
POLL_INTERVAL_SECONDS = 30
PRE_UPLOAD_DELAY_SECONDS = 10

uploaded_checkpoints = set()

def get_huggingface_token():
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token:
        print("Using Hugging Face token from HUGGINGFACE_TOKEN environment variable.")
        return token
    token = HfFolder.get_token()
    if token:
        print("Using Hugging Face token from saved credentials.")
        return token
    raise ValueError("Hugging Face token not found. Set HUGGINGFACE_TOKEN environment variable or login using `huggingface-cli login`.")

def find_new_checkpoint(current_dir: Path = Path('.')) -> tuple[int, Path] | None:
    new_checkpoints = []
    try:
        for item in current_dir.iterdir():
            if item.is_dir():
                match = CHECKPOINT_DIR_PATTERN.match(item.name)
                if match and item not in uploaded_checkpoints:
                    checkpoint_number = int(match.group(1))
                    new_checkpoints.append((checkpoint_number, item))
    except FileNotFoundError:
        print(f"Error: Directory not found: {current_dir}")
        return None
    except Exception as e:
        print(f"Error scanning directory {current_dir}: {e}")
        return None

    if new_checkpoints:
        new_checkpoints.sort(key=lambda x: x[0], reverse=True)
        return new_checkpoints[0]
    return None

def upload_checkpoint_to_hf(folder_path: Path, checkpoint_number: int, repo_id: str):
    print(f"\nAttempting to upload {folder_path.name} to Hugging Face repository: {repo_id}...")

    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"Repository {repo_id} exists or was created.")

        upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            commit_message=f"Upload checkpoint {checkpoint_number}",
            repo_type="model"
        )
        print(f"Successfully uploaded contents of {folder_path.name} to {repo_id}.")

        try:
            shutil.rmtree(folder_path)
            print(f"Successfully deleted local folder: {folder_path}")
            return True
        except OSError as e:
            print(f"Error deleting local folder {folder_path}: {e}. Please delete manually.")
            return True

    except Exception as e:
        print(f"ERROR during Hugging Face upload for {folder_path.name}: {e}")
        print("Upload failed. Local folder will not be deleted.")
        return False

def main():
    try:
        hf_token = get_huggingface_token()
        login(hf_token)
        print("\nSuccessfully logged into Hugging Face Hub.")
    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during Hugging Face login: {e}")
        return

    print("\nStarting checkpoint monitor...")
    print(f"Will check for new checkpoints matching '{CHECKPOINT_DIR_PATTERN.pattern}' every {POLL_INTERVAL_SECONDS} seconds.")
    print(f"Target repository: {TARGET_REPO_NAME}")
    print(f"Found checkpoints will be tracked (not re-uploaded): {uploaded_checkpoints or 'None yet'}")
    print("-" * 30)

    while True:
        new_checkpoint_info = find_new_checkpoint()

        if new_checkpoint_info:
            checkpoint_number, folder_path = new_checkpoint_info
            print(f"\nFound new checkpoint: {folder_path.name} (Step {checkpoint_number})")

            print(f"Waiting {PRE_UPLOAD_DELAY_SECONDS} seconds before processing...")
            time.sleep(PRE_UPLOAD_DELAY_SECONDS)

            upload_successful = upload_checkpoint_to_hf(
                folder_path=folder_path,
                checkpoint_number=checkpoint_number,
                repo_id=TARGET_REPO_NAME
            )

            if upload_successful:
                uploaded_checkpoints.add(folder_path)
                print(f"Added {folder_path.name} to the set of processed checkpoints.")

            print("-" * 30)

        else:
            print(f"\rNo new checkpoints found. Checking again in {POLL_INTERVAL_SECONDS} seconds... ", end="")

        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
