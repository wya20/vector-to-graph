import pytest

from src.result_merger.merger import (
    ResultMerger,
    ResultSource,
    VectorResult,
    GraphResult,
    MergedResult,
)


class TestNormalizeScores:
    def test_normalize_vector_scores(self):
        vector_results = [
            VectorResult(id="v1", text="text1", score=0.2, metadata={}),
            VectorResult(id="v2", text="text2", score=0.8, metadata={}),
        ]
        merger = ResultMerger()
        v_norm, g_norm = merger.normalize_scores(vector_results, [])

        assert len(v_norm) == 2
        assert v_norm[0] == 0.0
        assert v_norm[1] == 1.0

    def test_normalize_graph_scores(self):
        graph_results = [
            GraphResult(id="g1", label="label1", node_type="type1", confidence=0.3, relations=[], metadata={}),
            GraphResult(id="g2", label="label2", node_type="type2", confidence=0.9, relations=[], metadata={}),
        ]
        merger = ResultMerger()
        v_norm, g_norm = merger.normalize_scores([], graph_results)

        assert len(g_norm) == 2
        assert g_norm[0] == 0.0
        assert g_norm[1] == 1.0

    def test_normalize_empty_results(self):
        merger = ResultMerger()
        v_norm, g_norm = merger.normalize_scores([], [])
        assert v_norm == []
        assert g_norm == []

    def test_normalize_identical_scores(self):
        vector_results = [
            VectorResult(id="v1", text="text1", score=0.5, metadata={}),
            VectorResult(id="v2", text="text2", score=0.5, metadata={}),
        ]
        merger = ResultMerger()
        v_norm, _ = merger.normalize_scores(vector_results, [])

        assert v_norm[0] == 1.0
        assert v_norm[1] == 1.0


class TestMerge:
    def test_merge_both_empty(self):
        merger = ResultMerger()
        result = merger.merge([], [])
        assert result == []

    def test_merge_vector_only(self):
        vector_results = [
            VectorResult(id="v1", text="text1", score=0.5, metadata={"key": "val"}),
            VectorResult(id="v2", text="text2", score=0.8, metadata={}),
        ]
        merger = ResultMerger()
        result = merger.merge(vector_results, [])

        assert len(result) == 2
        assert all(r.source == ResultSource.VECTOR for r in result)
        assert result[0].id == "v1"
        assert result[0].normalized_score == 0.0
        assert result[1].id == "v2"
        assert result[1].normalized_score == 1.0

    def test_merge_graph_only(self):
        graph_results = [
            GraphResult(id="g1", label="label1", node_type="type1", confidence=0.2, relations=[], metadata={}),
        ]
        merger = ResultMerger()
        result = merger.merge([], graph_results)

        assert len(result) == 1
        assert result[0].source == ResultSource.GRAPH
        assert result[0].text == "label1"

    def test_merge_both_results_weighted(self):
        vector_results = [
            VectorResult(id="v1", text="text1", score=0.2, metadata={}),
            VectorResult(id="v2", text="text2", score=0.8, metadata={}),
        ]
        graph_results = [
            GraphResult(id="g1", label="label1", node_type="type1", confidence=0.3, relations=[], metadata={}),
            GraphResult(id="g2", label="label2", node_type="type2", confidence=0.9, relations=[], metadata={}),
        ]
        merger = ResultMerger(alpha=0.4, beta=0.6)
        result = merger.merge(vector_results, graph_results)

        assert len(result) == 4
        assert result[0].normalized_score >= result[-1].normalized_score

    def test_merge_deduplication(self):
        vector_results = [
            VectorResult(id="shared", text="text1", score=0.5, metadata={}),
        ]
        graph_results = [
            GraphResult(id="shared", label="label1", node_type="type1", confidence=0.5, relations=[], metadata={}),
        ]
        merger = ResultMerger()
        result = merger.merge(vector_results, graph_results)

        assert len(result) == 1
        assert result[0].id == "shared"

    def test_merge_custom_alpha_beta(self):
        vector_results = [
            VectorResult(id="v1", text="text1", score=0.5, metadata={}),
        ]
        merger = ResultMerger(alpha=0.3, beta=0.7)
        result = merger.merge(vector_results, [])

        assert result[0].normalized_score == 1.0


class TestBuildContext:
    def test_build_context_basic(self):
        results = [
            MergedResult(id="1", source=ResultSource.VECTOR, text="text1", score=0.5, normalized_score=0.8, metadata={}),
            MergedResult(id="2", source=ResultSource.GRAPH, text="text2", score=0.6, normalized_score=0.9, metadata={}),
        ]
        merger = ResultMerger()
        context = merger.build_context(results)

        assert "[向量检索] text1" in context
        assert "[图谱检索] text2" in context

    def test_build_context_max_items(self):
        results = [
            MergedResult(id=str(i), source=ResultSource.VECTOR, text=f"text{i}", score=0.5, normalized_score=0.5, metadata={})
            for i in range(15)
        ]
        merger = ResultMerger()
        context = merger.build_context(results, max_items=5)

        assert context.count("text") == 5

    def test_build_context_empty(self):
        merger = ResultMerger()
        context = merger.build_context([])
        assert context == ""


class TestResultSource:
    def test_result_source_enum(self):
        assert ResultSource.VECTOR.value == "vector"
        assert ResultSource.GRAPH.value == "graph"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])