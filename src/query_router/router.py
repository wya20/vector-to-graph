from enum import Enum
from typing import Literal

class QueryType(Enum):
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"
    DIRECT = "direct"

class QueryRouter:
    def __init__(self):
        self.vector_keywords = ["是什么", "讲了什么", "总结", "介绍", "定义",
                                "描述", "解释", "什么是", "怎么用", "如何使用"]
        self.graph_keywords = ["代码", "函数", "类", "调用", "继承", "实现",
                               "接口", "模块", "方法", "变量", "参数", "返回"]
        self.hybrid_keywords = ["关系", "区别", "联系", "为什么", "怎么做到的",
                                "有什么区别", "有什么关系", "为什么是", "如何实现"]
        self.direct_keywords = ["你好", "hello", "hi", "谢谢", "thanks"]

    def classify(self, query: str) -> QueryType:
        """
        根据关键词判断查询类型

        返回：
        - VECTOR_ONLY: 语义问答类
        - GRAPH_ONLY: 结构关系类
        - HYBRID: 混合类
        - DIRECT: 直接回答类
        """
        query_lower = query.lower()

        if any(kw in query_lower for kw in self.direct_keywords):
            return QueryType.DIRECT

        if any(kw in query for kw in self.hybrid_keywords):
            return QueryType.HYBRID

        if any(kw in query for kw in self.graph_keywords):
            return QueryType.GRAPH_ONLY

        if any(kw in query for kw in self.vector_keywords):
            return QueryType.VECTOR_ONLY

        return QueryType.VECTOR_ONLY

    def get_weight(self, query_type: QueryType) -> tuple[float, float]:
        """
        获取向量和图谱的权重

        返回 (alpha, beta)：
        - alpha: 向量权重
        - beta: 图谱权重
        """
        weights = {
            QueryType.VECTOR_ONLY: (1.0, 0.0),
            QueryType.GRAPH_ONLY: (0.0, 1.0),
            QueryType.HYBRID: (0.4, 0.6),
            QueryType.DIRECT: (0.0, 0.0)
        }
        return weights.get(query_type, (0.5, 0.5))