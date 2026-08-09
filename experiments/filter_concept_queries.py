"""Filter concept-class queries for Plan C comparison."""
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from query_router.router import QueryRouter, QueryType

# Plan C: concept categories only (no code structure queries)
CONCEPT_CATEGORIES = {
    "概念定义", "介绍类", "功能介绍", "总结类",
    "语义问答", "原因查询", "对比查询", "方法咨询"
}

with open(os.path.join(_PROJECT_ROOT, "experiments", "test_queries.json"), "r", encoding="utf-8") as f:
    queries = json.load(f)

concept = [q for q in queries if q["category"] in CONCEPT_CATEGORIES and q["relevant_ids"]]

print(f"Concept queries (annotated): {len(concept)}")
print()

for q in concept:
    print(f"  [{q['category']}] {q['query']}  -> relevant: {len(q['relevant_ids'])}")

# Save concept subset for GraphRAG evaluation
output_path = os.path.join(_PROJECT_ROOT, "experiments", "concept_queries.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(concept, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(concept)} concept queries to: {output_path}")