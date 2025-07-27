import os
import torch
from torch.utils.data import Dataset

class PrecomputedEmbeddingDataset(Dataset):
    """Wraps another dataset and adds precomputed teacher embeddings."""
    def __init__(self, base_dataset, embedding_root, suffix=".pt"):
        self.base_dataset = base_dataset
        self.embedding_root = embedding_root
        self.suffix = suffix
        self.media_type = "video"
        self._digits = len(str(len(base_dataset)))

    def __len__(self):
        return len(self.base_dataset)

    def _path(self, idx):
        fname = f"{idx:0{self._digits}d}{self.suffix}"
        return os.path.join(self.embedding_root, fname)

    def __getitem__(self, idx):
        media, caption, index = self.base_dataset[idx]
        emb_path = self._path(index)
        embedding = torch.load(emb_path) if os.path.isfile(emb_path) else None
        return media, caption, index, embedding
