import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector_indexer.embedder import Embedder


class TestEmbedder:
    @pytest.fixture
    def embedder(self):
        return Embedder(model_name="BAAI/bge-small-zh-v1.5")

    def test_initialization(self, embedder):
        assert embedder.model is not None
        assert embedder.embedding_dim > 0
        assert embedder.device in ["cpu", "cuda"]

    def test_encode_single_text(self, embedder):
        text = "这是一个测试句子"
        embedding = embedder.encode([text])[0]
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == embedder.embedding_dim
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6

    def test_encode_batch(self, embedder):
        texts = ["第一个句子", "第二个句子", "第三个句子"]
        embeddings = embedder.encode(texts)
        assert embeddings.shape == (3, embedder.embedding_dim)
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 1e-6

    def test_encode_query(self, embedder):
        query = "查询句子"
        embedding = embedder.encode_query(query)
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == embedder.embedding_dim
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6

    def test_batch_size(self, embedder):
        texts = ["句子" + str(i) for i in range(10)]
        embeddings = embedder.encode(texts, batch_size=4)
        assert embeddings.shape == (10, embedder.embedding_dim)

    def test_normalize_option(self, embedder):
        texts = ["测试句子"]
        embeddings_normalized = embedder.encode(texts, normalize=True)
        embeddings_raw = embedder.encode(texts, normalize=False)
        norm_norm = np.linalg.norm(embeddings_normalized[0])
        norm_raw = np.linalg.norm(embeddings_raw[0])
        assert abs(norm_norm - 1.0) < 1e-6
        assert abs(norm_raw - 1.0) < 1e-6