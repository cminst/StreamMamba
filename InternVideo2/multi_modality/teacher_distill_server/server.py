import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from demo.config import Config, eval_dict_leaf
from demo.utils import setup_internvideo2

# Define Pydantic models for request and response
class InferRequest(BaseModel):
    window_tensor: list  # Representing the tensor data

class InferResponse(BaseModel):
    embeddings: list[list[float]]

MODEL = None
CFG = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
    global MODEL, CFG
    if MODEL is not None:
        return

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

    CFG = Config.from_file(config_path)
    CFG = eval_dict_leaf(CFG)
    CFG.model.vision_ckpt_path = ckpt_path
    CFG.model.vision_encoder.pretrained = ckpt_path
    CFG.pretrained_path = ckpt_path
    CFG.device = str(DEVICE)

    MODEL, _ = setup_internvideo2(CFG)
    MODEL.eval()


def embed_video_6b(video_tensor: torch.Tensor):
    """Compute embeddings using InternVideo2-6B."""
    _load_model()
    video_tensor = video_tensor.to(DEVICE)
    # Convert from [B, C, T, H, W] -> [B, T, C, H, W]
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        embeddings = MODEL.get_vid_feat(video_tensor)
    return embeddings.cpu().tolist()



# Create a FastAPI instance
app = FastAPI()

@app.post("/infer", response_model=InferResponse)
async def infer(request_data: InferRequest):
    """
    FastAPI endpoint for simulating video inference.

    This endpoint accepts a POST request with a JSON body containing
    video data represented as a tensor (`window_tensor`).
    It runs the InternVideo2 model on the window tensor and returns
    the resulting embeddings in a JSON response.

    Request JSON body:
        {"window_tensor": list}

    Response JSON body:
        On success:
        {
            "embeddings": list[list[float]]  # Video embeddings
        }
        On failure:
        {
            "detail": str  # Error message
        }, 400

    Validates the input tensor shape to ensure it has at least one dimension (batch size).
    """
    try:
        # Extract the 'window_tensor' list and convert it to a PyTorch tensor.
        window_tensor_input = torch.tensor(request_data.window_tensor)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}")

    # Validate the shape: ensure it's not None and has at least one dimension (batch size).
    shape = window_tensor_input.shape
    if not shape or len(shape) < 1:
         raise HTTPException(status_code=400, detail='invalid shape: tensor must have at least one dimension')

    batch_size = shape[0]

    # Compute embeddings with the InternVideo2 model.
    embeddings = embed_video_6b(window_tensor_input)

    # Return the embeddings using the Pydantic response model.
    return InferResponse(embeddings=embeddings)

if __name__ == '__main__':
    # Run the FastAPI application using uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8008)
