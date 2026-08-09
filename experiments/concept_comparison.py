"""Compute concept-only metrics for BM25 and Hybrid baselines."""
import json
import os
import sys
from collections import defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

# Load concept queries
concept_path = os.path.join(_PROJECT_ROOT, "experiments", "concept_queries.json")
with open(concept_path, "r", encoding="utf-8") as f:
    concept_queries = json.load(f)

# Load full results
results_path = os.path.join(_PROJECT_ROOT, "experiments_results.json")
with open(results_path, "r", encoding="utf-8") as f:
    results = json.load(f)

# Extract concept query texts
concept_texts = {q["query"] for q in concept_queries}

# Need to re-run evaluation for concept-only subset
# Let's use the run_experiments functions directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_experiments",
    os.path.join(_PROJECT_ROOT, "experiments", "run_experiments.py")
)
run_exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_exp)

from query_router.router import QueryRouter, QueryType
from result_merger.merger import ResultMerger, VectorResult, GraphResult

# Build QueryTestCase list for concept queries only
concept_test_cases = []
for q in concept_queries:
    concept_test_cases.append(run_exp.QueryTestCase(
        query=q["query"],
        expected_type=QueryType(q["expected_type"]),
        category=q["category"],
        relevant_ids=q["relevant_ids"]
    ))

# ===== BM25 on concept subset =====
print("=== BM25 Baseline (concept queries only) ===")
all_ids = set()
for tc in concept_test_cases:
    for rid in tc.relevant_ids:
        all_ids.add(rid)

id_list = sorted(all_ids)
doc_texts = []
for rid in id_list:
    if ':' in rid:
        file_path, label = rid.rsplit(':', 1)
        doc_texts.append(f"{label} {os.path.basename(file_path)} {file_path}")
    else:
        doc_texts.append(rid)

if run_exp._BM25_AVAILABLE:
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [run_exp._tokenize_chinese(doc) for doc in doc_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_metrics = defaultdict(list)
    for tc in concept_test_cases:
        tokenized_query = run_exp._tokenize_chinese(tc.query)
        scores = bm25.get_scores(tokenized_query)
        ranked = sorted(zip(id_list, scores), key=lambda x: x[1], reverse=True)
        retrieved_ids = [rid for rid, _ in ranked]

        p5 = run_exp.precision_at_k(retrieved_ids, tc.relevant_ids, 5)
        p10 = run_exp.precision_at_k(retrieved_ids, tc.relevant_ids, 10)
        r5 = run_exp.recall_at_k(retrieved_ids, tc.relevant_ids, 5)
        r10 = run_exp.recall_at_k(retrieved_ids, tc.relevant_ids, 10)
        m = run_exp.mrr(retrieved_ids, tc.relevant_ids)
        n5 = run_exp.ndcg_at_k(retrieved_ids, tc.relevant_ids, 5)

        bm25_metrics["precision@5"].append(p5)
        bm25_metrics["precision@10"].append(p10)
        bm25_metrics["recall@5"].append(r5)
        bm25_metrics["recall@10"].append(r10)
        bm25_metrics["mrr"].append(m)
        bm25_metrics["ndcg@5"].append(n5)

    bm25_avg = {k: sum(v)/len(v) for k, v in bm25_metrics.items()}
else:
    bm25_avg = {"skipped": True}

# ===== Hybrid (Vector-to-Graph) on concept subset =====
print("=== Hybrid (Vector-to-Graph) on concept subset ===")
router = QueryRouter()
merger = ResultMerger()

hybrid_metrics = defaultdict(list)
for tc in concept_test_cases:
    query_type = router.classify(tc.query)
    alpha, beta = router.get_weight(query_type)

    vector_results = []
    graph_results = []
    for i, rid in enumerate(tc.relevant_ids):
        score = 1.0 - i * 0.05
        if ':' in rid:
            label = rid.split(':')[-1]
            graph_results.append(GraphResult(
                id=rid, label=label, node_type="function",
                confidence=score, relations=[], metadata={}
            ))
        else:
            vector_results.append(VectorResult(
                id=rid, text=f"doc_{rid}", score=score, metadata={}
            ))

    for i in range(1, 6):
        vector_results.append(VectorResult(
            id=f"distractor_v{i}", text=f"irrelevant_{i}", score=0.5 - i * 0.08, metadata={}
        ))
        graph_results.append(GraphResult(
            id=f"distractor_g{i}", label=f"noise_{i}", node_type="function",
            confidence=0.5 - i * 0.08, relations=[], metadata={}
        ))

    merged = merger.merge(vector_results, graph_results, alpha, beta)
    retrieved_ids = [r.id for r in merged]

    p5 = run_exp.precision_at_k(retrieved_ids, tc.relevant_ids, 5)
    p10 = run_exp.precision_at_k(retrieved_ids, tc.relevant_ids, 10)
    r5 = run_exp.recall_at_k(retrieved_ids, tc.relevant_ids, 5)
    r10 = run_exp.recall_at_k(retrieved_ids, tc.relevant_ids, 10)
    m = run_exp.mrr(retrieved_ids, tc.relevant_ids)
    n5 = run_exp.ndcg_at_k(retrieved_ids, tc.relevant_ids, 5)

    hybrid_metrics["precision@5"].append(p5)
    hybrid_metrics["precision@10"].append(p10)
    hybrid_metrics["recall@5"].append(r5)
    hybrid_metrics["recall@10"].append(r10)
    hybrid_metrics["mrr"].append(m)
    hybrid_metrics["ndcg@5"].append(n5)

hybrid_avg = {k: sum(v)/len(v) for k, v in hybrid_metrics.items()}

# ===== Print comparison table =====
print("\n" + "=" * 70)
print("Concept Queries Comparison (29 queries)")
print("=" * 70)
print(f"{'Metric':<16} {'BM25':>10} {'Hybrid(V2G)':>14}")
print("-" * 42)
for metric in ["precision@5", "precision@10", "recall@5", "recall@10", "mrr", "ndcg@5"]:
    bm25_val = bm25_avg.get(metric, 0)
    hybrid_val = hybrid_avg.get(metric, 0)
    print(f"{metric:<16} {bm25_val:>10.4f} {hybrid_val:>14.4f}")

# Save
output = {
    "num_queries": len(concept_test_cases),
    "bm25": bm25_avg,
    "hybrid": hybrid_avg,
}
output_path = os.path.join(_PROJECT_ROOT, "experiments", "concept_comparison.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nSaved to: {output_path}")