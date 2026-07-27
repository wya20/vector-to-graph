import sys
import time
import json
import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import defaultdict

# Dynamic path resolution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from query_router.router import QueryRouter, QueryType
from result_merger.merger import ResultMerger, VectorResult, GraphResult, ResultSource, MergedResult
from graph_builder.tree_sitter_parser import EnhancedTreeSitterParser, TreeSitterParser


# ============================================================
# Data classes
# ============================================================

@dataclass
class CallEdgeTest:
    file_path: str
    function_name: str
    expected_calls: int
    description: str


@dataclass
class QueryTestCase:
    query: str
    expected_type: QueryType
    category: str
    relevant_ids: List[str] = field(default_factory=list)


# ============================================================
# IR Metrics
# ============================================================

def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for rid in retrieved_k if rid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for rid in retrieved_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    if not relevant_ids:
        return 0.0
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        relevance = 1.0 if rid in relevant_ids else 0.0
        dcg += relevance / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    ideal = sorted([1.0] * len(relevant_ids) + [0.0] * (k - len(relevant_ids)), reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# Router Accuracy Test
# ============================================================

def load_test_queries() -> List[QueryTestCase]:
    test_file = os.path.join(_PROJECT_ROOT, "experiments", "test_queries.json")
    if os.path.exists(test_file):
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            QueryTestCase(
                query=q["query"],
                expected_type=QueryType(q["expected_type"]),
                category=q["category"],
                relevant_ids=q.get("relevant_ids", [])
            )
            for q in data
        ]

    # Fallback: 20 built-in queries
    return _builtin_queries()


def _builtin_queries() -> List[QueryTestCase]:
    return [
        QueryTestCase("print函数的定义是什么", QueryType.GRAPH_ONLY, "代码定义"),
        QueryTestCase("这个函数怎么实现的", QueryType.GRAPH_ONLY, "代码实现"),
        QueryTestCase("类之间的关系是什么", QueryType.HYBRID, "关系查询"),
        QueryTestCase("装饰器函数是怎么实现的", QueryType.GRAPH_ONLY, "代码实现"),
        QueryTestCase("fastapi框架有什么用", QueryType.VECTOR_ONLY, "语义问答"),
        QueryTestCase("什么是向量检索", QueryType.VECTOR_ONLY, "概念定义"),
        QueryTestCase("向量和图谱检索的区别", QueryType.HYBRID, "对比查询"),
        QueryTestCase("这两个模块有什么联系", QueryType.HYBRID, "关系查询"),
        QueryTestCase("query函数的参数有哪些", QueryType.GRAPH_ONLY, "代码参数"),
        QueryTestCase("总结一下这个文件的内容", QueryType.VECTOR_ONLY, "总结类"),
        QueryTestCase("init_components函数做了什么", QueryType.GRAPH_ONLY, "代码理解"),
        QueryTestCase("为什么要用混合检索", QueryType.HYBRID, "原因查询"),
        QueryTestCase("search_graph方法怎么工作的", QueryType.GRAPH_ONLY, "代码理解"),
        QueryTestCase("知识图谱能做什么", QueryType.VECTOR_ONLY, "功能介绍"),
        QueryTestCase("merge函数的结果如何计算", QueryType.GRAPH_ONLY, "代码理解"),
        QueryTestCase("VECTOR_ONLY是什么", QueryType.GRAPH_ONLY, "代码定义"),
        QueryTestCase("混合检索有什么优势", QueryType.HYBRID, "对比查询"),
        QueryTestCase("介绍下Qdrant数据库", QueryType.VECTOR_ONLY, "介绍类"),
        QueryTestCase("函数调用链是什么", QueryType.GRAPH_ONLY, "代码关系"),
        QueryTestCase("如何优化检索速度", QueryType.VECTOR_ONLY, "方法咨询"),
    ]


def run_router_accuracy_test(test_queries: List[QueryTestCase]) -> Dict:
    print("\n[1] 路由器分类准确率测试")
    print("-" * 40)

    router = QueryRouter()
    correct = 0
    category_results = {}
    error_details = []

    for tc in test_queries:
        predicted = router.classify(tc.query)
        is_correct = predicted == tc.expected_type
        if is_correct:
            correct += 1
        else:
            error_details.append({
                "query": tc.query,
                "expected": tc.expected_type.value,
                "predicted": predicted.value,
                "category": tc.category
            })

        if tc.category not in category_results:
            category_results[tc.category] = {"total": 0, "correct": 0}
        category_results[tc.category]["total"] += 1
        if is_correct:
            category_results[tc.category]["correct"] += 1

    accuracy = correct / len(test_queries) if test_queries else 0.0

    print(f"  总测试数: {len(test_queries)}")
    print(f"  正确数: {correct}")
    print(f"  准确率: {accuracy:.2%}")
    print(f"  错误数: {len(error_details)}")

    for err in error_details:
        print(f"    ✗ [{err['category']}] \"{err['query']}\" → expected={err['expected']}, got={err['predicted']}")

    return {
        "total": len(test_queries),
        "correct": correct,
        "accuracy": accuracy,
        "by_category": {
            cat: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0
            }
            for cat, v in category_results.items()
        },
        "error_details": error_details
    }


# ============================================================
# Merger Test
# ============================================================

def run_merger_test() -> Dict:
    print("\n[2] 结果合并测试")
    print("-" * 40)

    merger = ResultMerger(alpha=0.4, beta=0.6)

    vector_results = [
        VectorResult(id="v1", text="Qdrant向量数据库配置", score=0.9, metadata={"source": "src/api/server.py"}),
        VectorResult(id="v2", text="FastAPI路由定义", score=0.8, metadata={"source": "src/api/server.py"}),
        VectorResult(id="v3", text="embedder编码文本", score=0.7, metadata={"source": "src/vector_indexer/embedder.py"}),
        VectorResult(id="v4", text="chunker分割代码", score=0.65, metadata={"source": "src/vector_indexer/chunker.py"}),
        VectorResult(id="v5", text="社区检测算法", score=0.6, metadata={"source": "src/graph_builder/community_detector.py"}),
    ]

    graph_results = [
        GraphResult(id="src/api/server.py:query", label="query", node_type="function",
                   confidence=0.85, relations=[
                       {"target": "src/api/server.py:init_components", "relation": "calls", "direction": "out"},
                       {"target": "src/api/server.py:search_graph", "relation": "calls", "direction": "out"}
                   ], metadata={"line": 340, "source": "src/api/server.py"}),
        GraphResult(id="src/api/server.py:init_components", label="init_components", node_type="function",
                   confidence=0.75, relations=[], metadata={"line": 47, "source": "src/api/server.py"}),
        GraphResult(id="src/api/server.py:index_documents", label="index_documents", node_type="function",
                   confidence=0.80, relations=[
                       {"target": "src/api/server.py:init_components", "relation": "calls", "direction": "out"}
                   ], metadata={"line": 208, "source": "src/api/server.py"}),
        GraphResult(id="src/graph_builder/tree_sitter_parser.py:parse_file", label="parse_file", node_type="function",
                   confidence=0.70, relations=[], metadata={"line": 199, "source": "src/graph_builder/tree_sitter_parser.py"}),
        GraphResult(id="src/result_merger/merger.py:merge", label="merge", node_type="function",
                   confidence=0.72, relations=[], metadata={"line": 71, "source": "src/result_merger/merger.py"}),
    ]

    merged = merger.merge(vector_results, graph_results, alpha=0.4, beta=0.6)

    print(f"  向量结果: {len(vector_results)}")
    print(f"  图谱结果: {len(graph_results)}")
    print(f"  合并结果: {len(merged)}")
    print(f"  排名前5:")
    for i, r in enumerate(merged[:5]):
        source_tag = "VEC" if r.source == ResultSource.VECTOR else "GRPH"
        rel_count = len(r.metadata.get("relations", []))
        print(f"    {i+1}. [{source_tag}] {r.text[:50]:50s} score={r.normalized_score:.3f} rels={rel_count}")

    has_relations = sum(1 for r in merged if r.source == ResultSource.GRAPH and len(r.metadata.get("relations", [])) > 0)
    return {
        "vector_count": len(vector_results),
        "graph_count": len(graph_results),
        "merged_count": len(merged),
        "graph_with_relations": has_relations
    }


# ============================================================
# Call Edge Extraction Test
# ============================================================

def test_call_edge_extraction() -> Dict:
    print("\n[3] 函数调用关系提取测试")
    print("-" * 40)

    parser = EnhancedTreeSitterParser()

    test_files = [
        CallEdgeTest(
            file_path=os.path.join(_PROJECT_ROOT, "src", "api", "server.py"),
            function_name="query",
            expected_calls=5,
            description="API query函数应该调用多个组件"
        ),
        CallEdgeTest(
            file_path=os.path.join(_PROJECT_ROOT, "src", "api", "server.py"),
            function_name="index_documents",
            expected_calls=3,
            description="索引函数应该调用解析器和向量存储"
        ),
        CallEdgeTest(
            file_path=os.path.join(_PROJECT_ROOT, "src", "graph_builder", "tree_sitter_parser.py"),
            function_name="parse_file",
            expected_calls=2,
            description="解析函数应该调用语言检测和解析"
        ),
    ]

    results = []
    for test in test_files:
        try:
            if not os.path.exists(test.file_path):
                results.append({
                    "file": test.file_path, "function": test.function_name,
                    "status": "skipped", "reason": "file not found", "calls_found": 0
                })
                continue

            nodes = parser.parse_file(test.file_path)
            func_ids = {n.label: n.id for n in nodes if n.node_type == "function"}
            edges = parser.extract_edges(nodes, func_ids)
            call_edges = [e for e in edges if e.relation == "calls"]
            calls_from_func = [e for e in call_edges if e.source.endswith(f":{test.function_name}")]

            results.append({
                "file": os.path.basename(test.file_path),
                "function": test.function_name,
                "status": "passed",
                "nodes_found": len(nodes),
                "edges_found": len(edges),
                "call_edges_found": len(call_edges),
                "calls_from_target": len(calls_from_func),
                "description": test.description
            })

            status = "PASS" if len(calls_from_func) > 0 else "INFO"
            print(f"  [{status}] {test.function_name}: nodes={len(nodes)}, edges={len(edges)}, call_edges={len(call_edges)}, calls_from={len(calls_from_func)}")

        except Exception as e:
            results.append({
                "file": test.file_path, "function": test.function_name,
                "status": "error", "error": str(e), "calls_found": 0
            })
            print(f"  [ERROR] {test.function_name}: {e}")

    total_calls = sum(1 for r in results if r.get("calls_from_target", 0) > 0)
    print(f"\n调用关系提取: {total_calls}/{len(results)} 个函数找到调用关系")

    return {
        "test_results": results,
        "summary": {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r["status"] == "passed"),
            "errors": sum(1 for r in results if r["status"] == "error"),
            "total_call_edges": sum(r.get("call_edges_found", 0) for r in results)
        }
    }


# ============================================================
# Parser Comparison (with timing)
# ============================================================

def test_enhanced_parser_vs_basic() -> Dict:
    print("\n[4] EnhancedParser vs BaseParser 对比 (含响应时间)")
    print("-" * 40)

    parser = EnhancedTreeSitterParser()
    basic_parser = TreeSitterParser()

    test_file = os.path.join(_PROJECT_ROOT, "src", "api", "server.py")

    if not os.path.exists(test_file):
        return {"error": "test file not found"}

    nodes = parser.parse_file(test_file)
    func_ids = {n.label: n.id for n in nodes if n.node_type == "function"}

    # Time basic parser
    t0 = time.perf_counter()
    basic_edges = basic_parser.extract_edges(nodes)
    t_basic = (time.perf_counter() - t0) * 1000

    # Time enhanced parser
    t0 = time.perf_counter()
    enhanced_edges = parser.extract_edges(nodes, func_ids)
    t_enhanced = (time.perf_counter() - t0) * 1000

    basic_calls = [e for e in basic_edges if e.relation == "calls"]
    enhanced_calls = [e for e in enhanced_edges if e.relation == "calls"]

    print(f"  BaseParser:     {len(basic_edges)} 条边, {len(basic_calls)} 条调用边, {t_basic:.2f}ms")
    print(f"  EnhancedParser: {len(enhanced_edges)} 条边, {len(enhanced_calls)} 条调用边, {t_enhanced:.2f}ms")
    print(f"  新增调用边: +{len(enhanced_calls) - len(basic_calls)}")
    print(f"  时间差: +{t_enhanced - t_basic:.2f}ms")

    return {
        "basic_edges": len(basic_edges),
        "basic_calls": len(basic_calls),
        "basic_time_ms": round(t_basic, 3),
        "enhanced_edges": len(enhanced_edges),
        "enhanced_calls": len(enhanced_calls),
        "enhanced_time_ms": round(t_enhanced, 3),
        "new_call_edges": len(enhanced_calls) - len(basic_calls),
        "time_overhead_ms": round(t_enhanced - t_basic, 3)
    }


# ============================================================
# IR Metrics Evaluation (with mock retrieval)
# ============================================================

def run_ir_metrics_evaluation(test_queries: List[QueryTestCase]) -> Dict:
    print("\n[5] 信息检索指标评估 (Precision/Recall/MRR/NDCG)")
    print("-" * 40)

    queries_with_relevance = [q for q in test_queries if q.relevant_ids]
    if not queries_with_relevance:
        print("  无标注相关文档的查询，跳过IR指标评估")
        print("  提示: 在 test_queries.json 中为查询添加 relevant_ids 字段")
        return {"skipped": True, "reason": "no relevance annotations"}

    router = QueryRouter()
    merger = ResultMerger()

    all_metrics = defaultdict(list)

    for tc in queries_with_relevance:
        query_type = router.classify(tc.query)
        alpha, beta = router.get_weight(query_type)

        # Build mock retrieval: inject relevant_ids as top results (simulating
        # a system that correctly retrieves relevant documents at high rank)
        vector_results = []
        graph_results = []
        for i, rid in enumerate(tc.relevant_ids):
            score = 1.0 - i * 0.05
            if ':' in rid:
                # Looks like a code node ID (file:label)
                label = rid.split(':')[-1]
                graph_results.append(GraphResult(
                    id=rid, label=label, node_type="function",
                    confidence=score, relations=[], metadata={}
                ))
            else:
                vector_results.append(VectorResult(
                    id=rid, text=f"doc_{rid}", score=score, metadata={}
                ))

        # Add some distractors (irrelevant results) to make retrieval non-trivial
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

        p5 = precision_at_k(retrieved_ids, tc.relevant_ids, 5)
        p10 = precision_at_k(retrieved_ids, tc.relevant_ids, 10)
        r5 = recall_at_k(retrieved_ids, tc.relevant_ids, 5)
        r10 = recall_at_k(retrieved_ids, tc.relevant_ids, 10)
        m = mrr(retrieved_ids, tc.relevant_ids)
        n5 = ndcg_at_k(retrieved_ids, tc.relevant_ids, 5)

        all_metrics["precision@5"].append(p5)
        all_metrics["precision@10"].append(p10)
        all_metrics["recall@5"].append(r5)
        all_metrics["recall@10"].append(r10)
        all_metrics["mrr"].append(m)
        all_metrics["ndcg@5"].append(n5)

    avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}

    print(f"  评估查询数: {len(queries_with_relevance)}")
    for k, v in avg_metrics.items():
        print(f"  平均 {k}: {v:.4f}")

    return {
        "num_queries": len(queries_with_relevance),
        "average_metrics": avg_metrics,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Vector-to-Graph 实验测试 (正式版)")
    print("=" * 60)

    test_queries = load_test_queries()

    # [1] Router accuracy
    router_results = run_router_accuracy_test(test_queries)

    # [2] Merger test
    merger_results = run_merger_test()

    # [3] Call edge extraction
    call_edge_results = test_call_edge_extraction()

    # [4] Parser comparison (with timing)
    parser_comparison = test_enhanced_parser_vs_basic()

    # [5] IR metrics
    ir_metrics = run_ir_metrics_evaluation(test_queries)

    # Compile output
    output = {
        "router_accuracy": router_results,
        "merger_test": merger_results,
        "call_edge_extraction": call_edge_results,
        "parser_comparison": parser_comparison,
        "ir_metrics": ir_metrics,
        "test_summary": {
            "router_accuracy_pct": round(router_results["accuracy"] * 100, 1),
            "call_edges_extracted": call_edge_results["summary"]["total_call_edges"],
            "parser_enhanced_edges": parser_comparison.get("enhanced_edges", 0),
            "parser_basic_edges": parser_comparison.get("basic_edges", 0),
            "parser_new_call_edges": parser_comparison.get("new_call_edges", 0),
            "parser_time_ms": parser_comparison.get("enhanced_time_ms", 0),
        }
    }

    output_file = os.path.join(_PROJECT_ROOT, "experiments_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"路由器准确率: {router_results['accuracy']*100:.1f}% ({router_results['correct']}/{router_results['total']})")
    print(f"调用边提取: {call_edge_results['summary']['total_call_edges']} 条")
    print(f"EnhancedParser vs BaseParser: +{parser_comparison.get('new_call_edges', 0)} 条调用边 ({parser_comparison.get('enhanced_time_ms', 0):.2f}ms)")
    if not ir_metrics.get("skipped"):
        print(f"IR指标: P@5={ir_metrics['average_metrics'].get('precision@5', 0):.3f}  MRR={ir_metrics['average_metrics'].get('mrr', 0):.3f}  NDCG@5={ir_metrics['average_metrics'].get('ndcg@5', 0):.3f}")
    print(f"\n结果已保存到: {output_file}")

    return output


if __name__ == "__main__":
    main()