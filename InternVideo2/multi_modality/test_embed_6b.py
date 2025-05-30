#!/usr/bin/env python
# coding: utf-8
# Cleaned and enhanced InternVideo2 6B evaluation script with structured logging
# Source:

import os
import sys
import subprocess
import logging
import json
import argparse
from pathlib import Path

import torch
# Removed: numpy, cv2, tqdm
from huggingface_hub import hf_hub_download, HfApi, login


def setup_logging(log_level=logging.INFO, log_file=None):
    handlers = []
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=log_level, format=fmt, handlers=handlers)
    logging.info("Logging initialized.")


def run_command(cmd, cwd=None):
    logging.debug(f"Running command: {cmd} (cwd={cwd})")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Command failed: {cmd}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        raise RuntimeError(f"Command '{cmd}' failed (exit code {result.returncode})")
    logging.debug(f"Command succeeded, output: {result.stdout.strip()}")
    return result.stdout.strip()


def download_checkpoint(repo_id: str, filename: str) -> str:
    logging.info(f"Downloading {filename} from {repo_id}...")
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    logging.info(f"Downloaded vision checkpoint to {path}")
    return path


def load_config(config_path: str, vision_ckpt_path: str):
    from demo.config import Config, eval_dict_leaf
    logging.info(f"Loading config from {config_path}")
    cfg = Config.from_file(config_path)
    cfg = eval_dict_leaf(cfg)
    cfg.model.vision_ckpt_path = vision_ckpt_path
    cfg.model.vision_encoder.pretrained = vision_ckpt_path
    cfg.pretrained_path = vision_ckpt_path
    logging.debug(f"Config loaded: {cfg}")
    return cfg


def process_videos(
    tensor_path: str,
    model,
    config
):
    """
    Load pre-processed tensor, run inference, write outputs.
    """
    from demo.utils import retrieve_text_with_tensor

    logging.info("\n--- Starting processing with pre-processed tensor ---")

    # Load the pre-processed tensor instead of video frames
    try:
        frames_tensor = torch.load(tensor_path)
        logging.info(f"Loaded tensor from {tensor_path} with shape {frames_tensor.shape}")
    except FileNotFoundError:
        logging.error(f"{tensor_path} not found. Please ensure it exists in the specified path.")
        raise

    # Use the retrieve_text_with_tensor function
    _, probs = retrieve_text_with_tensor(
        frames_tensor, [phrase],
        model=model, topk=1, config=config,
        log=True # Add log=True for more verbosity if needed
    )
    score = probs[0] # Probability for the single phrase

    # Since we are processing a single tensor instead of sliding windows,
    # we represent the result as a prediction for index 1 (the whole tensor/video segment).
    best_idx = 1
    preds.append(best_idx)
    # Store the score associated with the predicted index (which is 1)
    logits.append([(float(score), best_idx)])

    logging.info(f"Processed tensor result: score {score:.4f}, prediction index {best_idx}\n")

    # Write results to files
    MODEL_NAME = '6B'
    prefix = f"ACT75-V5-InternVideo-{MODEL_NAME}-single-tensor"
    preds_file = f"{prefix}.json"
    logits_file = f"{prefix}-logits.json"
    logging.info(f"Writing predictions to {preds_file}")
    Path(preds_file).write_text(json.dumps(preds, indent=2))
    logging.info(f"Writing logits to {logits_file}")
    Path(logits_file).write_text(json.dumps(logits, indent=2))

    return preds_file, logits_file


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate InternVideo2 sliding-window retrieval using pre-processed tensor."
    )
    parser.add_argument('--tensor_path', type=str, default="example_input.pt",
                      help="Path to the pre-processed tensor file")
    args = parser.parse_args()

    setup_logging()

    # ensure IV2 repo
    iv2_path = Path('~/IV2').expanduser()
    if not iv2_path.exists():
        logging.info("Cloning IV2 repository...")
        run_command('git clone https://github.com/qingy1337/IV2.git', cwd=str(Path('~').expanduser())) # Clone into user home

    # Change directory into the multi_modality folder within the cloned repo
    target_cwd = iv2_path / 'InternVideo2' / 'multi_modality'
    if not target_cwd.exists():
         logging.error(f"Target directory does not exist: {target_cwd}")
         sys.exit(1)

    logging.info(f"Changing directory to {target_cwd}")
    os.chdir(target_cwd)

    vision_ckpt = download_checkpoint(
        repo_id="OpenGVLab/InternVideo2-Stage2_6B-224p-f4",
        filename="internvideo2-s2_6b-224p-f4.pt"
    )
    config = load_config('scripts/pretraining/stage2/6B/config.py', vision_ckpt)
    # Ensure setup_internvideo2 is imported after changing directory
    from demo.utils import setup_internvideo2
    model, tokenizer = setup_internvideo2(config)

    preds_file, logits_file = process_videos(
        args.tensor_path,
        model, config
    )


if __name__ == '__main__':
    main()
