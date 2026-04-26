import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

with patch('src.api.server.init_components'):
    from src.api.server import (
        app,
        IndexRequest,
        QueryRequest,
        IndexResponse,
        QueryResponse,
        GraphResponse,
        StatusResponse,
        indexing_tasks,
        current_graph
    )

client = TestClient(app)


class TestRootEndpoint:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Vector-to-Graph API"
        assert data["version"] == "0.1.0"


class TestIndexEndpoint:
    @patch('src.api.server.QdrantVectorStore')
    @patch('src.api.server.Embedder')
    @patch('src.api.server.Chunker')
    @patch('src.api.server.CommunityDetector')
    @patch('src.api.server.GraphExporter')
    def test_index_returns_task_id(
        self, mock_exporter, mock_detector, mock_chunker, mock_embedder, mock_qdrant
    ):
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk_file.return_value = []
        mock_chunker.return_value = mock_chunker_instance

        response = client.post("/index", json={"path": "/fake/path.py"})
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "indexing"
        assert "Indexing started" in data["message"]


class TestQueryEndpoint:
    @patch('src.api.server.qdrant_store')
    @patch('src.api.server.embedder')
    @patch('src.api.server.router')
    def test_query_direct_response(self, mock_router, mock_embedder, mock_store):
        from src.query_router.router import QueryType
        mock_router.classify.return_value = QueryType.DIRECT
        mock_router.get_weight.return_value = (0.0, 0.0)

        response = client.post("/query", json={"question": "你好"})
        assert response.status_code == 200
        data = response.json()
        assert "Vector-to-Graph" in data["answer"]
        assert data["query_type"] == "direct"

    @patch('src.api.server.merger')
    @patch('src.api.server.qdrant_store')
    @patch('src.api.server.embedder')
    @patch('src.api.server.router')
    def test_query_vector_search(self, mock_router, mock_embedder, mock_store, mock_merger):
        from src.query_router.router import QueryType
        from src.result_merger.merger import VectorResult, ResultSource, MergedResult
        import numpy as np

        mock_router.classify.return_value = QueryType.VECTOR_ONLY
        mock_router.get_weight.return_value = (1.0, 0.0)

        mock_embedder.encode_query.return_value = np.array([0.1] * 512)

        mock_store.search.return_value = [
            {
                "id": "test-id-1",
                "score": 0.95,
                "payload": {
                    "text": "test text",
                    "metadata": {"source": "test.py"}
                }
            }
        ]

        mock_result = MergedResult(
            id="test-id-1",
            source=ResultSource.VECTOR,
            text="test text",
            score=0.95,
            normalized_score=0.95,
            metadata={}
        )
        mock_merger.merge.return_value = [mock_result]
        mock_merger.build_context.return_value = "context string"

        response = client.post("/query", json={"question": "这是什么", "limit": 5})
        assert response.status_code == 200
        data = response.json()
        assert data["query_type"] == "vector_only"
        assert len(data["sources"]) > 0


class TestGraphEndpoint:
    @patch('src.api.server.graph_exporter')
    @patch('src.api.server.community_detector')
    def test_graph_returns_empty_on_no_data(self, mock_detector, mock_exporter):
        mock_exporter.load.side_effect = FileNotFoundError()

        response = client.get("/graph")
        assert response.status_code == 200
        data = response.json()
        assert data["nodes"] == []
        assert data["edges"] == []

    @patch('src.api.server.graph_exporter')
    @patch('src.api.server.community_detector')
    def test_graph_returns_data(self, mock_detector, mock_exporter):
        mock_exporter.load.return_value = {
            "nodes": [{"id": "1", "label": "test"}],
            "edges": [{"source": "1", "target": "2"}],
            "community_map": {"1": 0, "2": 0}
        }
        mock_detector.get_community_stats.return_value = [
            {"community_id": 0, "size": 2, "nodes": ["1", "2"]}
        ]

        response = client.get("/graph")
        assert response.status_code == 200
        data = response.json()
        assert len(data["nodes"]) == 1
        assert len(data["communities"]) == 1


class TestStatusEndpoint:
    @patch('src.api.server.qdrant_store')
    @patch('src.api.server.current_graph', {"nodes": [], "edges": []})
    def test_status_healthy(self, mock_store):
        mock_store.get_collection_info.return_value = {
            "points_count": 100,
            "vectors_count": 100
        }

        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["doc_count"] == 100

    @patch('src.api.server.qdrant_store')
    def test_status_degraded_on_error(self, mock_store):
        mock_store.get_collection_info.side_effect = Exception("Connection failed")

        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["doc_count"] == 0


class TestRequestModels:
    def test_index_request_model(self):
        req = IndexRequest(path="/test/path.py")
        assert req.path == "/test/path.py"
        assert req.collection_name is None

    def test_query_request_model(self):
        req = QueryRequest(question="测试问题", limit=5)
        assert req.question == "测试问题"
        assert req.limit == 5

    def test_query_request_default_limit(self):
        req = QueryRequest(question="测试问题")
        assert req.limit == 10

    def test_index_response_model(self):
        resp = IndexResponse(
            task_id="test-123",
            status="completed",
            message="Done"
        )
        assert resp.task_id == "test-123"
        assert resp.status == "completed"

    def test_query_response_model(self):
        resp = QueryResponse(
            answer="Test answer",
            sources=[],
            query_type="vector_only",
            context=""
        )
        assert resp.answer == "Test answer"
        assert resp.query_type == "vector_only"

    def test_graph_response_model(self):
        resp = GraphResponse(
            nodes=[{"id": "1"}],
            edges=[{"source": "1", "target": "2"}],
            communities=[{"community_id": 0, "size": 2}]
        )
        assert len(resp.nodes) == 1
        assert len(resp.edges) == 1
        assert len(resp.communities) == 1

    def test_status_response_model(self):
        resp = StatusResponse(
            status="healthy",
            doc_count=10,
            node_count=5,
            edge_count=3
        )
        assert resp.status == "healthy"
        assert resp.doc_count == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])