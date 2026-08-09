"""Generate final three-way comparison table for concept queries."""
import json
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load results from all three baselines
concept_path = os.path.join(_PROJECT_ROOT, "experiments", "concept_comparison.json")
bge_path = os.path.join(_PROJECT_ROOT, "experiments", "bge_vector_results.json")

with open(concept_path, "r") as f:
    concept = json.load(f)

with open(bge_path, "r") as f:
    bge = json.load(f)

bm25 = concept["bm25"]
hybrid = concept["hybrid"]
bge_vec = bge["average_metrics"]

metrics = ["precision@5", "precision@10", "recall@5", "recall@10", "mrr", "ndcg@5"]

print("=" * 75)
print("Concept Queries Comparison (29 queries, same BGE model)")
print("=" * 75)
print(f"{'Metric':<16} {'BM25':>10} {'BGE Vector':>12} {'V2G Hybrid':>12} {'Hybrid/BM25':>13}")
print("-" * 65)

for m in metrics:
    bm25_v = bm25.get(m, 0)
    bge_v = bge_vec.get(m, 0)
    hyb_v = hybrid.get(m, 0)
    ratio = hyb_v / bm25_v if bm25_v > 0 else 0
    print(f"{m:<16} {bm25_v:>10.4f} {bge_v:>12.4f} {hyb_v:>12.4f} {ratio:>12.1f}x")

print("-" * 65)

# Winner analysis
print("\nBest performer per metric:")
for m in metrics:
    vals = {"BM25": bm25.get(m, 0), "BGE": bge_vec.get(m, 0), "Hybrid": hybrid.get(m, 0)}
    best = max(vals, key=vals.get)
    print(f"  {m}: {best} ({vals[best]:.4f})")

# Save combined results
output = {
    "num_queries": 29,
    "query_type": "concept_only",
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "bm25": bm25,
    "bge_vector": bge_vec,
    "v2g_hybrid": hybrid,
}
output_path = os.path.join(_PROJECT_ROOT, "experiments", "final_comparison.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nSaved to: {output_path}")