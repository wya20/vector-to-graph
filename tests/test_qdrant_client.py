import pytest
from unittest.mock import MagicMock, patch
from qdrant_client.http.models import Distance, VectorParams

import sys
sys.path.insert(0, "c:/Users/w/Desktop/trae/vector-to-graph/src")

from vector_indexer.qdrant_client import QdrantVectorStore


class TestQdrantVectorStore:
    @patch("vector_indexer.qdrant_client.QdrantClient")
    def test_init_creates_collection_if_not_exists(self, mock_qdrant_client):
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collection.side_effect = Exception("Collection not found")

        store = QdrantVectorStore(host="localhost", port=6333, collection_name="test", vector_dim=512)

        mock_client_instance.create_collection.assert_called_once_with(
            collection_name="test",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )

    @patch("vector_indexer.qdrant_client.QdrantClient")
    def test_init_does_not_create_collection_if_exists(self, mock_qdrant_client):
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collection.return_value = MagicMock()

        store = QdrantVectorStore(host="localhost", port=6333, collection_name="test", vector_dim=512)

        mock_client_instance.create_collection.assert_not_called()

    @patch("vector_indexer.qdrant_client.QdrantClient")
    def test_upsert_calls_client_upsert(self, mock_qdrant_client):
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collection.return_value = MagicMock()

        store = QdrantVectorStore(host="localhost", port=6333, collection_name="test")
        points = [{"id": "1", "vector": [0.1, 0.2], "payload": {"text": "hello"}}]
        store.upsert(points)

        mock_client_instance.upsert.assert_called_once_with(
            collection_name="test",
            points=points
        )

    @patch("vector_indexer.qdrant_client.QdrantClient")
    def test_search_returns_formatted_results(self, mock_qdrant_client):
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collection.return_value = MagicMock()

        mock_hit = MagicMock()
        mock_hit.id = "1"
        mock_hit.score = 0.95
        mock_hit.payload = {"text": "hello"}
        mock_client_instance.search.return_value = [mock_hit]

        store = QdrantVectorStore(host="localhost", port=6333, collection_name="test")
        results = store.search(query_vector=[0.1, 0.2], limit=5, score_threshold=0.5)

        assert len(results) == 1
        assert results[0]["id"] == "1"
        assert results[0]["score"] == 0.95
        assert results[0]["payload"] == {"text": "hello"}

        mock_client_instance.search.assert_called_once_with(
            collection_name="test",
            query_vector=[0.1, 0.2],
            limit=5,
            score_threshold=0.5
        )

    @patch("vector_indexer.qdrant_client.QdrantClient")
    def test_delete_calls_client_delete(self, mock_qdrant_client):
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collection.return_value = MagicMock()

        store = QdrantVectorStore(host="localhost", port=6333, collection_name="test")
        store.delete(["1", "2", "3"])

        mock_client_instance.delete.assert_called_once()

    @patch("vector_indexer.qdrant_client.QdrantClient")
    def test_get_collection_info(self, mock_qdrant_client):
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collection.return_value = MagicMock(
            vectors_count=100,
            points_count=100
        )

        store = QdrantVectorStore(host="localhost", port=6333, collection_name="test")
        info = store.get_collection_info()

        assert info["name"] == "test"
        assert info["vectors_count"] == 100
        assert info["points_count"] == 100