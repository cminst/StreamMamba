import os
import torch
import ray

from demo.config import Config, eval_dict_leaf
from demo.utils import setup_internvideo2

# Initialize Ray as the head node
ray.init(dashboard_host="0.0.0.0")
print(f"Ray dashboard available at: http://{ray.dashboard.serve.get_webui_url()}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once and make it available globally
def load_model():
    config_path = os.environ.get(
        "IV2_6B_CONFIG",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "pretraining",
            "stage2",
            "6B",
            "config.py",
        ),
    )
    ckpt_path = os.environ.get("IV2_6B_CKPT")
    if not ckpt_path:
        raise RuntimeError("IV2_6B_CKPT environment variable must point to the model weights")

    cfg = Config.from_file(config_path)
    cfg = eval_dict_leaf(cfg)
    cfg.model.vision_ckpt_path = ckpt_path
    cfg.model.vision_encoder.pretrained = ckpt_path
    cfg.pretrained_path = ckpt_path
    cfg.device = str(DEVICE)

    model, _ = setup_internvideo2(cfg)
    model.eval()
    return model

# Create a remote function for inference
@ray.remote(num_gpus=1)
class InternVideo2Service:
    def __init__(self):
        self.model = load_model()

    def embed_video(self, video_tensor):
        """Compute embeddings using InternVideo2-6B."""
        # Convert list to tensor if needed
        if isinstance(video_tensor, list):
            video_tensor = torch.tensor(video_tensor)

        video_tensor = video_tensor.to(DEVICE)
        # Convert from [B, C, T, H, W] -> [B, T, C, H, W]
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            embeddings = self.model.get_vid_feat(video_tensor)

        return embeddings.cpu().numpy().tolist()

# Start the service
service = InternVideo2Service.remote()
print(f"\n\nInternVideo2 service started! Address: {service}")

# Keep the server running
import time
while True:
    time.sleep(1)
