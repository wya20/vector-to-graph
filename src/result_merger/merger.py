from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ResultSource(Enum):
    VECTOR = "vector"
    GRAPH = "graph"


@dataclass
class VectorResult:
    id: str
    text: str
    score: float
    metadata: dict


@dataclass
class GraphResult:
    id: str
    label: str
    node_type: str
    confidence: float
    relations: List[dict]
    metadata: dict


@dataclass
class MergedResult:
    id: str
    source: ResultSource
    text: str
    score: float
    normalized_score: float
    metadata: dict


class ResultMerger:
    def __init__(self, alpha: float = 0.4, beta: float = 0.6):
        self.alpha = alpha
        self.beta = beta

    def normalize_scores(
        self,
        vector_results: List[VectorResult],
        graph_results: List[GraphResult],
    ) -> Tuple[List[float], List[float]]:
        if not vector_results:
            vector_normalized = []
        else:
            v_scores = [r.score for r in vector_results]
            v_min, v_max = min(v_scores), max(v_scores)
            if v_max - v_min > 0:
                vector_normalized = [(s - v_min) / (v_max - v_min) for s in v_scores]
            else:
                vector_normalized = [1.0 for _ in v_scores]

        if not graph_results:
            graph_normalized = []
        else:
            g_scores = [r.confidence for r in graph_results]
            g_min, g_max = min(g_scores), max(g_scores)
            if g_max - g_min > 0:
                graph_normalized = [(s - g_min) / (g_max - g_min) for s in g_scores]
            else:
                graph_normalized = [1.0 for _ in g_scores]

        return vector_normalized, graph_normalized

    def merge(
        self,
        vector_results: List[VectorResult],
        graph_results: List[GraphResult],
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> List[MergedResult]:
        alpha = alpha or self.alpha
        beta = beta or self.beta

        if not vector_results and not graph_results:
            return []

        if not graph_results:
            v_norm, _ = self.normalize_scores(vector_results, [])
            return [
                MergedResult(
                    id=r.id,
                    source=ResultSource.VECTOR,
                    text=r.text,
                    score=r.score,
                    normalized_score=v_norm[i],
                    metadata=r.metadata,
                )
                for i, r in enumerate(vector_results)
            ]

        if not vector_results:
            _, g_norm = self.normalize_scores([], graph_results)
            return [
                MergedResult(
                    id=r.id,
                    source=ResultSource.GRAPH,
                    text=r.label,
                    score=r.confidence,
                    normalized_score=g_norm[i],
                    metadata=r.metadata,
                )
                for i, r in enumerate(graph_results)
            ]

        v_norm, g_norm = self.normalize_scores(vector_results, graph_results)

        merged = []

        for i, r in enumerate(vector_results):
            merged.append(
                MergedResult(
                    id=r.id,
                    source=ResultSource.VECTOR,
                    text=r.text,
                    score=r.score,
                    normalized_score=alpha * v_norm[i],
                    metadata={**r.metadata, "source": "vector"},
                )
            )

        for i, r in enumerate(graph_results):
            merged.append(
                MergedResult(
                    id=r.id,
                    source=ResultSource.GRAPH,
                    text=r.label,
                    score=r.confidence,
                    normalized_score=beta * g_norm[i],
                    metadata={**r.metadata, "source": "graph", "relations": r.relations},
                )
            )

        merged.sort(key=lambda x: x.normalized_score, reverse=True)

        seen = set()
        unique_results = []
        for r in merged:
            if r.id not in seen:
                seen.add(r.id)
                unique_results.append(r)

        return unique_results

    def build_context(self, results: List[MergedResult], max_items: int = 10) -> str:
        context_parts = []
        for r in results[:max_items]:
            source_tag = "[向量检索]" if r.source == ResultSource.VECTOR else "[图谱检索]"
            part = f"{source_tag} {r.text}"
            if r.source == ResultSource.GRAPH and r.metadata.get("relations"):
                relations = r.metadata.get("relations", [])
                rel_strs = []
                for rel in relations[:5]:
                    if "target" in rel:
                        rel_strs.append(f"→{rel['target']}({rel['relation']})")
                    elif "source" in rel:
                        rel_strs.append(f"{rel['source']}→({rel['relation']})")
                if rel_strs:
                    part += f" [关系: {', '.join(rel_strs)}]"
            context_parts.append(part)

        return "\n\n".join(context_parts)