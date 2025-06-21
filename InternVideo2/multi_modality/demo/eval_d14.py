import os
import sys
import cv2
import torch
import shutil
import logging
from iv2_utils.iv2 import *
from os import system as run
from collections import OrderedDict
from huggingface_hub import hf_hub_download

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

def install_dependencies():
    """Install required dependencies if they're not already installed."""
    try:
        import einops
        logging.info("Dependencies already installed.")
    except ImportError:
        logging.info("Installing dependencies...")
        run('pip install -q einops peft opencv-python open_clip_torch protobuf sentencepiece iv2-utils')
        logging.info("Dependencies installed successfully.")

def setup_environment():
    """Setup the project environment."""
    os.chdir(os.path.expanduser('~/'))

    if 'IV2' not in os.listdir('.'):
        logging.info("Cloning IV2 repository...")
        run('git clone https://github.com/qingy1337/IV2.git')

    os.chdir('IV2/InternVideo2/multi_modality')
    run('git checkout delta')
    sys.path.append(os.getcwd())
    logging.info("Environment setup complete.")

def download_model_components(hf_token, model_name="B14"):
    """Download necessary model components from Hugging Face.

    Args:
        hf_token (str): HuggingFace token for authentication
        model_name (str): Model name/version to download

    Returns:
        dict: Paths to downloaded model components
    """
    logging.info(f"Downloading {model_name} model components...")

    # Download model components
    file_path = hf_hub_download(
        repo_id="OpenGVLab/InternVideo2_distillation_models",
        filename=f"stage1/{model_name}/{model_name}_dist_1B_stage2/pytorch_model.bin",
        token=hf_token
    )

    clip_path = hf_hub_download(
        repo_id="OpenGVLab/InternVideo2_distillation_models",
        filename=f"clip/{model_name}/pytorch_model.bin",
        token=hf_token
    )

    streaming_vit_path = hf_hub_download(
        repo_id="qingy2024/InternVideo2-R16-ckpt",
        filename="ckpt_v6_d14_2k.pt",
        token=hf_token
    )

    run('wget -q https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt')
    mobileclip_path = "mobileclip_blt.pt"

    logging.info(f"{model_name} downloaded to: {file_path}")
    logging.info(f"CLIP downloaded to: {clip_path}")
    logging.info(f"Streaming ViT weights downloaded to: {streaming_vit_path}")
    logging.info(f"Downloaded MobileCLIP to {mobileclip_path}")

    return {
        "model_path": file_path,
        "clip_path": clip_path,
        "streaming_vit_path": streaming_vit_path,
        "mobileclip_path": mobileclip_path
    }

def configure_model(model_paths, model_name="B14"):
    """Configure the model with downloaded components.

    Args:
        model_paths (dict): Paths to model components
        model_name (str): Model name/version

    Returns:
        tuple: Configured model and device
    """
    # Import after sys.path.append in setup_environment
    from models.internvideo2_clip_small import InternVideo2_CLIP_small
    from demo.config import Config, eval_dict_leaf

    logging.info(f"Configuring {model_name} model...")

    config = Config.from_file(f'scripts/pretraining/clip/{model_name}/config.py')
    config = eval_dict_leaf(config)

    config.model.vision_ckpt_path = model_paths["model_path"]
    config.model.mobileclip_ckpt_path = model_paths["mobileclip_path"]
    config.model.extra_ckpt_path = model_paths["clip_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    intern_model = InternVideo2_CLIP_small(config)
    intern_model.to(device)

    return intern_model, device

def load_checkpoint(model, checkpoint_path, device):
    """Load and apply checkpoint to the model.

    Args:
        model: The model to load weights into
        checkpoint_path (str): Path to the checkpoint file
        device: Device to load the checkpoint to

    Returns:
        model: The model with loaded weights
    """
    logging.info(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    logging.debug(f"Loaded checkpoint. Keys: {checkpoint.keys()}")

    # Extract state dictionary from checkpoint
    if 'module' in checkpoint:
        logging.info("Found 'module' key in checkpoint, assuming it contains the state_dict.")
        state_dict_from_checkpoint = checkpoint['module']
    elif 'model' in checkpoint:
        logging.info("Found 'model' key in checkpoint, assuming it contains the state_dict.")
        state_dict_from_checkpoint = checkpoint['model']
    elif 'state_dict' in checkpoint:
        logging.info("Found 'state_dict' key in checkpoint, assuming it contains the state_dict.")
        state_dict_from_checkpoint = checkpoint['state_dict']
    else:
        # If none of the common keys are found, maybe the checkpoint IS the state_dict
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            logging.info("Checkpoint keys don't match common patterns, but values are tensors. Assuming checkpoint IS the state_dict.")
            state_dict_from_checkpoint = checkpoint
        else:
            raise KeyError(f"Could not find model state_dict in checkpoint. Tried 'module', 'model', 'state_dict', and direct. Available keys: {checkpoint.keys()}")

    # Handle 'module.' prefix in state_dict keys
    new_state_dict = OrderedDict()
    is_ddp_inner_checkpoint = any(key.startswith('module.') for key in state_dict_from_checkpoint.keys())

    if is_ddp_inner_checkpoint:
        logging.info("State_dict contains keys prefixed with 'module.'. Removing prefix.")
        for k, v in state_dict_from_checkpoint.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict_to_load = new_state_dict
    else:
        logging.info("State_dict keys do not start with 'module.', using as is.")
        state_dict_to_load = state_dict_from_checkpoint

    # Load state dictionary into model
    logging.info("Loading state_dict into model. Using strict=False as checkpoint may only contain trainable parameters.")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_to_load, strict=False)

    report_checkpoint_errors(unexpected_keys, missing_keys)

    return model

def report_checkpoint_errors(unexpected_keys, missing_keys):
    """Report any issues with checkpoint loading.

    Args:
        unexpected_keys (list): Keys found in checkpoint but not in model
        missing_keys (list): Keys found in model but not in checkpoint
    """
    # Report unexpected keys
    if unexpected_keys:
        logging.error("Unexpected keys in state_dict (present in checkpoint but not in current model definition):")
        for key in unexpected_keys:
            logging.error(f"  - {key}")
        logging.error("This likely indicates a mismatch between the saved model architecture and the current one.")

    # Report missing keys
    if missing_keys:
        logging.info("Missing keys in state_dict (present in current model definition but not in checkpoint):")
        potentially_expected_missing = []
        actually_unexpected_missing = []

        for key in missing_keys:
            # Determine if missing key is expected to be frozen
            is_expected_frozen = True
            if 'streaming' in key:
                is_expected_frozen = False

            if is_expected_frozen:
                potentially_expected_missing.append(key)
            else:
                actually_unexpected_missing.append(key)

        if actually_unexpected_missing:
            logging.warning("Some TRAINABLE parts of the model are missing from the checkpoint:")
            for key in actually_unexpected_missing:
                logging.warning(f"    - {key}")
        else:
            logging.info("All missing keys appear to be for parameters that were likely frozen and not saved, which is expected.")
            if potentially_expected_missing:
                logging.debug("    (These include parts like the base InternVideo2 blocks if they were frozen)")
                for key in potentially_expected_missing[:5]:
                    logging.debug(f"      - {key}")
                if len(potentially_expected_missing) > 5:
                    logging.debug(f"      - ... and {len(potentially_expected_missing) - 5} more.")

def get_photography_model_data():
    """Clone the photography-model repo, load a sample video, and extract frames."""
    repo_dir = 'photography-model'

    if not os.path.isdir(repo_dir):
        logging.info(f"Cloning photography-model into {repo_dir}...")
        run('git clone https://github.com/ruo2019/photography-model.git')

    video_path = os.path.join(repo_dir, 'data', 'act75', '1.mp4')
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")

    # Import here so _frame_from_video is defined
    from demo.utils import _frame_from_video

    logging.info(f"Opening video file: {video_path}")
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise IOError(f"Unable to open video: {video_path}")

    # Use the helper to extract frames
    frames = list(_frame_from_video(video))
    video.release()

    if not frames:
        raise RuntimeError("No frames extracted from the video.")

    # Show shape of first frame for sanity check
    logging.info(f"Extracted {len(frames)} frames; first frame shape: {frames[0].shape}")
    return frames

def evaluate_streaming_similarity(
    model,
    device,
    streaming_transform, # The preprocessing transform
    video_path,
    model_max_frames,
    output_dir,
    global_step, # Current training step for filename
    config
):
    """
    Evaluates the cosine similarity between streaming and full window features
    for a specific video and saves a plot.

    Returns the average cosine similarity over the comparable frames.
    """

    regular_transform = transforms.Compose(
        [
            transforms.Resize(
                (model.config.model.vision_encoder.img_size, model.config.model.vision_encoder.img_size),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Ensure model is in evaluation mode and on the correct device
    model.eval()
    model.to(device) # Ensure model is on device, though it should be already

    cosine_similarities = []
    frame_indices_for_plot = []
    avg_similarity = -1.0 # Default value if no frames processed

    logging.info(f"Starting evaluation on video: {video_path}")

    # 1) Read all frames for the current video
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        logging.error(f"Error: Could not open video {video_path} for evaluation.")
        return avg_similarity # Return default if video can't be opened

    all_frames_raw = list(_frame_from_video(video_cap)) # List of numpy arrays (H, W, C, BGR)
    video_cap.release()

    if len(all_frames_raw) < model_max_frames:
        logging.warning(f"Evaluation video {video_path} has {len(all_frames_raw)} frames, less than MODEL_MAX_FRAMES ({model_max_frames}). Skipping evaluation.")
        return avg_similarity # Return default if video is too short

    # Use torch.no_grad() for inference
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        # 2) Initialize streaming model's hidden state for this evaluation run
        # Batch size is 1 for single video inference
        curr_hidden_state_streaming = model.streaming_vision_encoder.init_hidden(batch_size=1, device=device)

        # 3) Warm-up streaming model with the first (MODEL_MAX_FRAMES - 1) frames
        # Process frames from index 0 up to MODEL_MAX_FRAMES - 2
        logging.info(f"Warming up streaming model for evaluation with first {model_max_frames - 1} frames...")
        for i in range(model_max_frames - 1):
            frame_data = all_frames_raw[i] # Get BGR numpy array

            # Preprocess single frame -> [1, C, H, W] tensor on device
            frame_tensor_batch = preprocess_frame(frame_data, streaming_transform, device) # [1, C, H, W]

            # Add temporal dimension (T=1) for streaming encoder input [B, C, T=1, H, W]
            frame_tensor_streaming_input = frame_tensor_batch.unsqueeze(2) # [1, C, 1, H, W]

            # Pass the frame and the previous hidden state to the streaming encoder
            raw_stream_embedding_dummy, curr_hidden_state_streaming = model.streaming_vision_encoder(
                frame_tensor_streaming_input, # Input is [1, C, 1, H, W]
                curr_hidden_state_streaming
            )
        logging.info(f"Warm-up complete for evaluation.")


        # 4) Slide through the rest of the video, frame by frame
        #    Loop from the index of the frame that completes the first window (MODEL_MAX_FRAMES - 1)
        #    Up to the last frame of the video
        logging.info(f"Processing and comparing from frame {model_max_frames - 1} onwards...")
        # No tqdm here to avoid interfering with training progress bar
        for frame_idx in range(model_max_frames - 1, len(all_frames_raw)):
            # --- Streaming Model Feature (using the *current* frame and state) ---
            current_frame_data_streaming = all_frames_raw[frame_idx] # BGR numpy array

            # Preprocess the *current* frame for the streaming encoder
            frame_tensor_batch = preprocess_frame(current_frame_data_streaming, streaming_transform, device) # [1, C, H, W]

            # Add temporal dimension (T=1) for streaming encoder input [B, C, T=1, H, W]
            frame_tensor_streaming_input = frame_tensor_batch.unsqueeze(2) # [1, C, 1, H, W]

            # Pass the current frame and the previous hidden state to the streaming encoder
            raw_stream_embedding, new_hidden_state = model.streaming_vision_encoder(
                frame_tensor_streaming_input, # Input is [1, C, 1, H, W]
                curr_hidden_state_streaming
            )

            # Align and Normalize the raw streaming embedding
            if config.model.use_streaming_vision_align:
                aligned_stream_embedding = model.streaming_vision_align(raw_stream_embedding)
            else:
                aligned_stream_embedding = model.vision_align(raw_stream_embedding)
            stream_embedding = aligned_stream_embedding / (aligned_stream_embedding.norm(dim=-1, keepdim=True) + 1e-9)

            # Update the hidden state for the next frame
            curr_hidden_state_streaming = new_hidden_state

            # --- Full Model Feature for the corresponding window ---
            # The window of MODEL_MAX_FRAMES frames ends at the current frame_idx
            window_start_idx = frame_idx - model_max_frames + 1
            window_end_idx = frame_idx + 1 # Slicing is exclusive at the end
            current_window_frames_data = all_frames_raw[window_start_idx : window_end_idx] # List of BGR numpy arrays

            # Preprocess all frames in the window and stack them
            # List of [1, C, H, W] tensors -> Stack -> [MODEL_MAX_FRAMES, 1, C, H, W]
            list_of_frame_tensors = [preprocess_frame(f, regular_transform, device) for f in current_window_frames_data]
            stacked_window_tensor_T_B_C_H_W = torch.stack(list_of_frame_tensors, dim=0) # Shape: [T, B=1, C, H, W]

            # Reshape for the full vision encoder [B, C, T, H, W]
            window_tensor_full = stacked_window_tensor_T_B_C_H_W.unsqueeze(0).squeeze(2).permute(0, 2, 1, 3, 4) # Shape: [1, C, MODEL_MAX_FRAMES, H, W]

            # Pass the full window tensor to the full vision encoder
            raw_target_embedding = model.vision_encoder(window_tensor_full)

            # Align and Normalize the raw target embedding
            aligned_target_embedding = model.vision_align(raw_target_embedding)
            target_embedding = aligned_target_embedding / (aligned_target_embedding.norm(dim=-1, keepdim=True) + 1e-9)

            # --- Cosine Similarity ---
            similarity = torch.nn.functional.cosine_similarity(stream_embedding, target_embedding, dim=1)

            sim_value = similarity.item()
            cosine_similarities.append(sim_value)
            frame_indices_for_plot.append(frame_idx) # Store the actual frame index

        # --- Evaluation Complete ---
        if cosine_similarities:
            avg_similarity = sum(cosine_similarities) / len(cosine_similarities)
            logging.info(f"Evaluation complete. Average Cosine Similarity: {avg_similarity:.4f}")

            # --- Plotting and Saving ---
            plt.figure(figsize=(12, 6))
            plt.plot(frame_indices_for_plot, cosine_similarities, 'g-', label='Cosine Similarity (Streaming vs Full Window)')
            plt.xlabel(f'Frame Number (Window of {model_max_frames} frames ending at this frame)')
            plt.ylabel('Cosine Similarity')
            plt.title(f'Feature Similarity Over Time - Video: {os.path.basename(video_path)}\nTraining Step: {global_step}')
            plt.legend()
            plt.grid(True)
            plt.ylim(-0.1, 1.1) # Cosine similarity range
            plt.axhline(y=avg_similarity, color='b', linestyle='--', label=f'Average: {avg_similarity:.4f}')
            plt.legend()

            # Define save path
            graph_save_dir = join(output_dir, 'cosine_sim_graphs')
            os.makedirs(graph_save_dir, exist_ok=True)
            graph_filename = f'graph_step_{global_step:07d}.png' # Use padded step number
            graph_save_path = join(graph_save_dir, graph_filename)

            plt.savefig(graph_save_path)
            logging.info(f"Saved evaluation plot to {graph_save_path}")

            # Close the plot figure to free memory
            plt.close('all')
        else:
            logging.warning("No cosine similarities were calculated during evaluation.")


    # Set model back to training mode
    model.train()
    logging.info("Evaluation complete. Model set back to train() mode.")

    return avg_similarity

def main():
    """Main function to orchestrate the entire process."""
    # Install dependencies
    install_dependencies()

    # Setup environment
    setup_environment()

    # Now we can import project-specific modules after setting up the path
    from demo.utils import (
        retrieve_text,
        setup_internvideo2,
        retrieve_text_streaming,
        frames2tensor
    )

    # Get HuggingFace token from environment
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not found. Please set it with your HuggingFace token.")

    # Set model name
    model_name = "B14"

    # Download model components
    model_paths = download_model_components(hf_token, model_name)

    # Configure model
    intern_model, device = configure_model(model_paths, model_name)

    # Load checkpoint
    intern_model = load_checkpoint(intern_model, model_paths["streaming_vit_path"], device)

    # Prepare model for inference
    intern_model.eval()
    logging.info("Model set to evaluation mode and ready for inference.")

    get_photography_model_data()

    return intern_model

if __name__ == "__main__":
    model = main()
