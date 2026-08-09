"""OpenAI Embeddings baseline: pure vector search using OpenAI embeddings.

This serves as a replacement for GraphRAG's vector retrieval component,
providing a fair comparison point: keyword search (BM25) vs embedding search
(OpenAI) vs hybrid search (Vector-to-Graph).
"""
import json
import os
import sys
import math
from collections import defaultdict
from typing import List, Dict
from openai import OpenAI

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

# Reuse IR metrics from run_experiments
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


# Build document texts
doc_texts = [doc_id_to_text(rid) for rid in id_list]
print(f"Corpus: {len(id_list)} unique documents")
print(f"Queries: {len(concept_queries)}")

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set.")
    print("Set it with: $env:OPENAI_API_KEY='your-key'")
    sys.exit(1)

client = OpenAI(api_key=api_key)
MODEL = "text-embedding-3-small"

# Embed all documents
print("Embedding documents...")
doc_embeddings = []
batch_size = 50
for i in range(0, len(doc_texts), batch_size):
    batch = doc_texts[i:i + batch_size]
    resp = client.embeddings.create(model=MODEL, input=batch)
    doc_embeddings.extend([r.embedding for r in resp.data])
    print(f"  {min(i + batch_size, len(doc_texts))}/{len(doc_texts)}")

# Embed all queries
print("Embedding queries...")
query_embeddings = []
for i in range(0, len(concept_queries), batch_size):
    batch = [q["query"] for q in concept_queries[i:i + batch_size]]
    resp = client.embeddings.create(model=MODEL, input=batch)
    query_embeddings.extend([r.embedding for r in resp.data])
    print(f"  {min(i + batch_size, len(concept_queries))}/{len(concept_queries)}")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0


# Evaluate
print("\nEvaluating...")
all_metrics = defaultdict(list)

for qi, q in enumerate(concept_queries):
    # Compute similarity with all documents
    scores = [
        cosine_similarity(query_embeddings[qi], doc_embeddings[di])
        for di in range(len(id_list))
    ]
    # Rank by similarity
    ranked = sorted(zip(id_list, scores), key=lambda x: x[1], reverse=True)
    retrieved_ids = [rid for rid, _ in ranked]

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

print(f"\n=== OpenAI Embedding Search Results ({MODEL}) ===")
for k, v in avg_metrics.items():
    print(f"  {k}: {v:.4f}")

# Save results
output = {
    "model": MODEL,
    "corpus_size": len(id_list),
    "num_queries": len(concept_queries),
    "average_metrics": avg_metrics,
}
output_path = os.path.join(_PROJECT_ROOT, "experiments", "openai_embedding_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {output_path}")