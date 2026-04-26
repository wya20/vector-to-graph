import sys
sys.path.insert(0, r"c:\Users\w\Desktop\trae\vector-to-graph\src")

from query_router.router import QueryRouter, QueryType


def test_classify_vector_only():
    router = QueryRouter()
    queries = [
        "这本书讲了什么",
        "总结一下内容",
        "介绍一下Python",
        "什么是向量数据库",
        "怎么使用这个API",
        "这个概念的定义"
    ]
    for query in queries:
        result = router.classify(query)
        assert result == QueryType.VECTOR_ONLY, f"Query '{query}' expected VECTOR_ONLY, got {result}"


def test_classify_graph_only():
    router = QueryRouter()
    queries = [
        "这段代码怎么写",
        "print函数的用法",
        "定义一个Animal类",
        "实现某个接口",
        "哪个模块调用了这个方法"
    ]
    for query in queries:
        result = router.classify(query)
        assert result == QueryType.GRAPH_ONLY, f"Query '{query}' expected GRAPH_ONLY, got {result}"


def test_classify_hybrid():
    router = QueryRouter()
    queries = [
        "向量和图谱的区别",
        "它们之间有什么关系",
        "为什么要用混合搜索",
        "怎么做到的"
    ]
    for query in queries:
        result = router.classify(query)
        assert result == QueryType.HYBRID, f"Query '{query}' expected HYBRID, got {result}"


def test_classify_direct():
    router = QueryRouter()
    queries = [
        "你好",
        "HELLO",
        "Hi there",
        "谢谢你的帮助",
        "Thanks"
    ]
    for query in queries:
        result = router.classify(query)
        assert result == QueryType.DIRECT, f"Query '{query}' expected DIRECT, got {result}"


def test_get_weight():
    router = QueryRouter()

    alpha, beta = router.get_weight(QueryType.VECTOR_ONLY)
    assert (alpha, beta) == (1.0, 0.0), f"VECTOR_ONLY expected (1.0, 0.0), got ({alpha}, {beta})"

    alpha, beta = router.get_weight(QueryType.GRAPH_ONLY)
    assert (alpha, beta) == (0.0, 1.0), f"GRAPH_ONLY expected (0.0, 1.0), got ({alpha}, {beta})"

    alpha, beta = router.get_weight(QueryType.HYBRID)
    assert (alpha, beta) == (0.4, 0.6), f"HYBRID expected (0.4, 0.6), got ({alpha}, {beta})"

    alpha, beta = router.get_weight(QueryType.DIRECT)
    assert (alpha, beta) == (0.0, 0.0), f"DIRECT expected (0.0, 0.0), got ({alpha}, {beta})"


def test_default_query():
    router = QueryRouter()
    result = router.classify("随便输入一些内容")
    assert result == QueryType.VECTOR_ONLY


if __name__ == "__main__":
    test_classify_vector_only()
    test_classify_graph_only()
    test_classify_hybrid()
    test_classify_direct()
    test_get_weight()
    test_default_query()
    print("All tests passed!")