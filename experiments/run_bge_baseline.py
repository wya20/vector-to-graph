"""BGE Local Embedding baseline: pure vector search using the same
BAAI/bge-small-zh-v1.5 model that Vector-to-Graph uses internally.

This isolates the contribution of graph fusion: same embeddings,
but without the knowledge graph re-ranking step.
"""
import json
import os
import sys
import math
import numpy as np
from collections import defaultdict
from typing import List

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from vector_indexer.embedder import Embedder

# Reuse IR metrics
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run_experiments",
    os.path.join(_PROJECT_ROOT, "experiments", "run_experiments.py")
)
run_exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_exp)
precision_at_k = run_exp.precision_at_k
recall_at_k = run_exp.recall_at_k
mrr = run_exp.mrr
ndcg_at_k = run_exp.ndcg_at_k

# Load concept queries
concept_path = os.path.join(_PROJECT_ROOT, "experiments", "concept_queries.json")
with open(concept_path, "r", encoding="utf-8") as f:
    concept_queries = json.load(f)

# Collect all unique document IDs
all_ids = set()
for q in concept_queries:
    for rid in q["relevant_ids"]:
        all_ids.add(rid)

id_list = sorted(all_ids)


def doc_id_to_text(rid: str) -> str:
    """Convert a document ID to searchable text."""
    if ':' in rid:
        file_path, label = rid.rsplit(':', 1)
        basename = os.path.basename(file_path)
        return f"{label} in {basename}"
    return rid


doc_texts = [doc_id_to_text(rid) for rid in id_list]
print(f"Corpus: {len(id_list)} documents")
print(f"Queries: {len(concept_queries)}")

# Load local BGE model (same as Vector-to-Graph)
print("Loading BAAI/bge-small-zh-v1.5 (CPU)...")
embedder = Embedder(model_name="BAAI/bge-small-zh-v1.5", device="cpu")
print(f"  Embedding dimension: {embedder.embedding_dim}")

# Embed documents
print("Embedding documents...")
doc_embeddings = embedder.encode(doc_texts, batch_size=32, normalize=True)
print(f"  Shape: {doc_embeddings.shape}")

# Embed queries (using the same query-prompt format as the project)
print("Embedding queries...")
query_texts = [f"Represent this sentence for similarity search: {q['query']}" for q in concept_queries]
query_embeddings = embedder.encode(query_texts, batch_size=32, normalize=True)
print(f"  Shape: {query_embeddings.shape}")

# Evaluate: cosine similarity (already normalized, so dot product = cosine)
print("\nEvaluating...")
all_metrics = defaultdict(list)

for qi, q in enumerate(concept_queries):
    # Dot product with normalized vectors = cosine similarity
    scores = np.dot(doc_embeddings, query_embeddings[qi])
    # Rank by similarity descending
    ranked_indices = np.argsort(scores)[::-1]
    retrieved_ids = [id_list[i] for i in ranked_indices]

    p5 = precision_at_k(retrieved_ids, q["relevant_ids"], 5)
    p10 = precision_at_k(retrieved_ids, q["relevant_ids"], 10)
    r5 = recall_at_k(retrieved_ids, q["relevant_ids"], 5)
    r10 = recall_at_k(retrieved_ids, q["relevant_ids"], 10)
    m = mrr(retrieved_ids, q["relevant_ids"])
    n5 = ndcg_at_k(retrieved_ids, q["relevant_ids"], 5)

    all_metrics["precision@5"].append(p5)
    all_metrics["precision@10"].append(p10)
    all_metrics["recall@5"].append(r5)
    all_metrics["recall@10"].append(r10)
    all_metrics["mrr"].append(m)
    all_metrics["ndcg@5"].append(n5)

avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}

print(f"\n=== BGE Pure Vector Search (same model, no graph) ===")
for k, v in avg_metrics.items():
    print(f"  {k}: {v:.4f}")

# Save
output = {
    "model": "BAAI/bge-small-zh-v1.5",
    "corpus_size": len(id_list),
    "num_queries": len(concept_queries),
    "average_metrics": avg_metrics,
}
output_path = os.path.join(_PROJECT_ROOT, "experiments", "bge_vector_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nSaved to: {output_path}")