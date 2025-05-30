import random
import torch
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
import uvicorn

# Define Pydantic models for request and response
class InferRequest(BaseModel):
    window_tensor: list # Representing the tensor data
    texts: list[str] # Representing the text data

class InferResponse(BaseModel):
    vision_embeddings: list[list[float]]
    text_embeddings: list[list[float]]

def embed_video_6b(video_tensor):
    """
    Implements the InternVideo-6B API.

    This function takes a PyTorch tensor representing video data and
    returns a list of lists representing embeddings.

    Args:
        video_tensor (torch.Tensor): Input tensor representing video data.
                                     Expected shape is [B, C, 4, H, W].

    Returns:
        list[list[float]]: A list of lists, where each inner list is a
                           embedding vector of size 768.
    """
    # The dummy implementation only needs the batch size from the input tensor's shape.
    batch = video_tensor.shape[0]
    # Generate random embeddings of size [batch, 768] and convert to a list.
    return torch.randn(batch, 768).tolist()

def embed_text(text):
    """
    Dummy function to embed text.

    Args:
        text (str): Input text string.

    Returns:
        list[list[float]]: A list of lists, where each inner list is a
                           embedding vector of size 768.
    """
    # Generate a random embedding of size [1, 768] and convert to a list.
    return torch.randn(1, 768).tolist()


# Create a FastAPI instance
app = FastAPI()

@app.post("/infer", response_model=InferResponse)
async def infer(request_data: InferRequest):
    """
    FastAPI endpoint for simulating video inference.

    This endpoint accepts a POST request with a JSON body containing
    video data represented as a tensor (`window_tensor`) and a text string.
    It simulates processing this data by calling dummy embedding functions and
    returns the resulting dummy embeddings in a JSON response.

    Request JSON body:
        {
            "window_tensor": list  # List representing a tensor (e.g., [[...]])
            "texts": list[str] # list of text queries, one for each batch
        }

    Response JSON body:
        On success:
        {
            "vision_embeddings": list[list[float]]  # List of dummy video embedding vectors
            "text_embeddings": list[list[float]]  # List of dummy text embedding vectors
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
        texts_input = request_data.texts
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}")

    # Validate the shape: ensure it's not None and has at least one dimension (batch size).
    shape = window_tensor_input.shape
    if not shape or len(shape) < 1:
         raise HTTPException(status_code=400, detail='invalid shape: tensor must have at least one dimension')

    batch_size = shape[0]

    if len(texts_input) != batch_size:
        raise HTTPException(status_code=400, detail='The number of texts must match the batch size of the video tensor.')

    # Call the dummy API to get simulated embeddings.
    vision_embeddings = embed_video_6b(window_tensor_input)
    text_embeddings = [embed_text(text) for text in texts_input]

    # Return the embeddings using the Pydantic response model.
    return InferResponse(vision_embeddings=vision_embeddings, text_embeddings=text_embeddings)

if __name__ == '__main__':
    # Run the FastAPI application using uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8008)
