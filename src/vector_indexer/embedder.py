from sentence_transformers import SentenceTransformer
import torch
from typing import List
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

    def encode_query(self, query: str) -> np.ndarray:
        query_with_prompt = f"Represent this sentence for similarity search: {query}"
        return self.encode([query_with_prompt])[0]