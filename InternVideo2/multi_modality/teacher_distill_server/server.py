import random
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)


def dummy_internvideo6b_api(video_tensor):
    """Return dummy embeddings for the input tensor."""
    batch = video_tensor.shape[0]
    return torch.randn(batch, 768).tolist()


@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json(force=True)
    shape = data.get('shape')
    if shape is None or len(shape) < 1:
        return jsonify({'error': 'invalid shape'}), 400
    dummy_tensor = torch.empty(shape)
    embeddings = dummy_internvideo6b_api(dummy_tensor)
    return jsonify({'embeddings': embeddings})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008)
