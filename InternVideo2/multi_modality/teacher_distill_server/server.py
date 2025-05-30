import os
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn
import tempfile

from demo.config import Config, eval_dict_leaf
from demo.utils import setup_internvideo2

# Define Pydantic model for response
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
async def infer(window_tensor: UploadFile = File(...)):
    """
    FastAPI endpoint for video inference using file uploads.

    This endpoint accepts a POST request with a file upload containing
    a PyTorch tensor (`window_tensor.pt`) saved with `torch.save`.
    It loads the tensor, runs the InternVideo2 model, and returns the
    resulting embeddings in a JSON response.

    Request:
        - Form-data with key `window_tensor` and value as a `.pt` file.

    Response JSON body:
        On success:
        {
            "embeddings": list[list[float]]  # Video embeddings
        }
        On failure:
        {
            "detail": str  # Error message
        }, 400
    """
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            content = await window_tensor.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load tensor from the temporary file
        try:
            window_tensor_input = torch.load(tmp_path)
        finally:
            os.unlink(tmp_path)  # Clean up the temporary file

        # Validate the tensor shape
        shape = window_tensor_input.shape
        if len(shape) != 5:
            raise HTTPException(status_code=400, detail="Invalid tensor shape: expected [B, C, T, H, W]")

        # Compute embeddings
        embeddings = embed_video_6b(window_tensor_input)

        # Return the embeddings
        return InferResponse(embeddings=embeddings)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
