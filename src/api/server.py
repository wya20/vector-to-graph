from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Set
import uuid
import numpy as np
from pathlib import Path
from collections import defaultdict

from ..vector_indexer.embedder import Embedder
from ..vector_indexer.chunker import Chunker
from ..vector_indexer.qdrant_client import QdrantVectorStore
from ..graph_builder.tree_sitter_parser import EnhancedTreeSitterParser
from ..graph_builder.community_detector import CommunityDetector
from ..graph_builder.graph_exporter import GraphExporter
from ..query_router.router import QueryRouter, QueryType
from ..result_merger.merger import ResultMerger, VectorResult, GraphResult
from ..sync_manager.transaction_log import TransactionLogger, OperationType
from ..sync_manager.file_watcher import FileWatcher

app = FastAPI(title="Vector-to-Graph API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = None
chunker = None
qdrant_store = None
parser = None
community_detector = None
graph_exporter = None
router = None
merger = None

indexing_tasks: Dict[str, dict] = {}
current_graph: Dict = {"nodes": [], "edges": [], "community_map": {}, "node_map": {}}
global_func_ids: Dict[str, str] = {}
indexed_files: Set[str] = set()
tx_logger: TransactionLogger = None
file_watcher: FileWatcher = None


def init_components():
    global embedder, chunker, qdrant_store, parser, community_detector, graph_exporter, router, merger, tx_logger
    embedder = Embedder()
    chunker = Chunker()
    qdrant_store = QdrantVectorStore(host="qdrant", port=6333)
    parser = EnhancedTreeSitterParser()
    community_detector = CommunityDetector()
    graph_exporter = GraphExporter()
    router = QueryRouter()
    merger = ResultMerger(alpha=0.4, beta=0.6)
    tx_logger = TransactionLogger()
    _restore_graph_from_disk()


def _restore_graph_from_disk():
    global current_graph, global_func_ids, indexed_files
    try:
        graph_data = graph_exporter.load()
        if graph_data.get("nodes"):
            current_graph["nodes"] = graph_data["nodes"]
            current_graph["edges"] = graph_data["edges"]
            current_graph["community_map"] = graph_data.get("community_map", {})
            current_graph["node_map"] = {n["id"]: n for n in graph_data["nodes"]}
            for node in graph_data["nodes"]:
                if node.get("type") == "function":
                    global_func_ids[node.get("label", "")] = node["id"]
            indexed_files = {n.get("metadata", {}).get("source", n.get("file", "")) 
                          for n in graph_data["nodes"]}
            indexed_files.discard("")
    except FileNotFoundError:
        pass


def search_graph(query_vector: List[float], top_k: int = 10, query_text: str = "") -> List[GraphResult]:
    if not current_graph.get("nodes"):
        return []

    nodes = current_graph.get("nodes", [])
    edges = current_graph.get("edges", [])
    node_map = current_graph.get("node_map", {})

    query_np = np.array(query_vector)
    query_lower = query_text.lower()

    keyword_match_nodes = []
    for node in nodes:
        if node.get("type") in ["function", "class"]:
            label = node.get("label", "").lower()
            if label and (label in query_lower or query_lower in label):
                keyword_match_nodes.append(node)

    if not keyword_match_nodes:
        for node in nodes:
            if node.get("type") in ["function", "class"]:
                label = node.get("label", "")
                if label:
                    label_vec = embedder.encode_query(label)
                    label_np = np.array(label_vec)
                    norm_product = np.linalg.norm(query_np) * np.linalg.norm(label_np)
                    if norm_product > 0:
                        score = np.dot(query_np, label_np) / norm_product
                        if score > 0.5:
                            node["_match_score"] = score
                            keyword_match_nodes.append(node)

    results = []
    visited = set()

    for start_node in keyword_match_nodes:
        if start_node["id"] in visited:
            continue

        queue = [start_node]
        visited.add(start_node["id"])

        while queue and len(results) < top_k * 2:
            current = queue.pop(0)
            current_id = current["id"]

            relations = []
            for edge in edges:
                if edge["source"] == current_id:
                    relations.append({"target": edge["target"], "relation": edge["relation"], "direction": "out"})
                    if edge["target"] not in visited:
                        visited.add(edge["target"])
                        target_node = node_map.get(edge["target"])
                        if target_node:
                            queue.append(target_node)
                elif edge["target"] == current_id:
                    relations.append({"source": edge["source"], "relation": edge["relation"], "direction": "in"})
                    if edge["source"] not in visited:
                        visited.add(edge["source"])
                        source_node = node_map.get(edge["source"])
                        if source_node:
                            queue.append(source_node)

            match_score = current.get("_match_score", 0.8)
            call_count = sum(1 for r in relations if r.get("relation") == "calls")

            results.append(GraphResult(
                id=current_id,
                label=current.get("label", ""),
                node_type=current.get("type", ""),
                confidence=float(match_score + call_count * 0.05),
                relations=relations,
                metadata=current.get("metadata", {})
            ))

    results.sort(key=lambda x: x.confidence, reverse=True)
    return results[:top_k]


class IndexRequest(BaseModel):
    path: str
    collection_name: Optional[str] = None


class IndexResponse(BaseModel):
    task_id: str
    status: str
    message: str


class QueryRequest(BaseModel):
    question: str
    limit: Optional[int] = 10


class SourceItem(BaseModel):
    type: str
    id: str
    score: float
    text: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    query_type: str
    context: str


class GraphResponse(BaseModel):
    nodes: List[dict]
    edges: List[dict]
    communities: List[dict]


class StatusResponse(BaseModel):
    status: str
    doc_count: int
    node_count: int
    edge_count: int
    last_update: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    init_components()


@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    indexing_tasks[task_id] = {"status": "pending", "path": request.path}

    async def index_task():
        global global_func_ids, indexed_files, tx_logger
        try:
            indexing_tasks[task_id]["status"] = "running"
            if tx_logger:
                tx_logger.log(OperationType.UPSERT_VECTOR, request.path, "running")
                tx_logger.log(OperationType.UPSERT_GRAPH, request.path, "running")

            code_nodes = parser.parse_file(request.path)

            with open(request.path, "r", encoding="utf-8") as f:
                content = f.read()
            lines = content.split("\n")

            chunks = []
            for node in code_nodes:
                if node.node_type in ["function", "class"]:
                    start_line = node.metadata.get("start_line", node.line)
                    end_line = node.metadata.get("end_line", node.line)
                    if start_line > 0 and end_line >= start_line:
                        chunk_lines = lines[start_line - 1 : end_line]
                        chunk_text = "\n".join(chunk_lines)
                        chunks.append({
                            "id": str(uuid.uuid4()),
                            "text": chunk_text,
                            "node_id": node.id,
                            "metadata": {
                                "source": request.path,
                                "chunk_type": "code_class" if node.node_type == "class" else "code_function",
                                "language": Path(request.path).suffix[1:],
                                "line_start": start_line,
                                "line_end": end_line,
                                "function_name" if node.node_type == "function" else "class_name": node.label
                            }
                        })

            texts = [chunk["text"] for chunk in chunks]
            embeddings = embedder.encode(texts)

            vectors = []
            for i, chunk in enumerate(chunks):
                vectors.append({
                    "id": chunk["id"],
                    "vector": embeddings[i].tolist(),
                    "payload": {
                        "text": chunk["text"],
                        "metadata": chunk["metadata"]
                    }
                })

            qdrant_store.upsert(vectors)

            nodes, edges = [], []
            try:
                new_nodes = [{
                    "id": n.id,
                    "label": n.label,
                    "type": n.node_type,
                    "file": n.file,
                    "metadata": {"line": n.line, "source": request.path, **n.metadata}
                } for n in code_nodes]

                for node in new_nodes:
                    if node["type"] == "function":
                        global_func_ids[node["label"]] = node["id"]

                new_edges = [{
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "confidence": e.confidence
                } for e in parser.extract_edges(code_nodes, global_func_ids)]

                node_map = current_graph.get("node_map", {})

                existing_node_ids = {n["id"] for n in current_graph["nodes"]}
                existing_edge_hashes = {
                    (e["source"], e["target"]) for e in current_graph["edges"]
                }

                for node in new_nodes:
                    if node["id"] not in existing_node_ids:
                        current_graph["nodes"].append(node)
                        node_map[node["id"]] = node

                for edge in new_edges:
                    if (edge["source"], edge["target"]) not in existing_edge_hashes:
                        current_graph["edges"].append(edge)

                current_graph["node_map"] = node_map
                indexed_files.add(request.path)

                if current_graph["nodes"]:
                    community_map = community_detector.detect_communities(
                        community_detector.build_graph(
                            current_graph["nodes"], 
                            current_graph["edges"]
                        )
                    )
                    current_graph["community_map"] = community_map
                    graph_exporter.export(
                        current_graph["nodes"], 
                        current_graph["edges"], 
                        community_map
                    )

            except Exception:
                pass

            indexing_tasks[task_id] = {"status": "completed", "path": request.path}
            if tx_logger:
                tx_logger.mark_completed(OperationType.UPSERT_VECTOR, request.path)
                tx_logger.mark_completed(OperationType.UPSERT_GRAPH, request.path)
        except Exception as e:
            indexing_tasks[task_id] = {"status": "failed", "error": str(e)}
            if tx_logger:
                tx_logger.mark_failed(OperationType.UPSERT_VECTOR, request.path, str(e))
                tx_logger.mark_failed(OperationType.UPSERT_GRAPH, request.path, str(e))

    background_tasks.add_task(index_task)

    return IndexResponse(
        task_id=task_id,
        status="indexing",
        message=f"Indexing started for {request.path}"
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    query_type = router.classify(request.question)
    alpha, beta = router.get_weight(query_type)

    if query_type == QueryType.DIRECT:
        return QueryResponse(
            answer="您好！我是 Vector-to-Graph 混合检索助手。请问有什么可以帮助您的？",
            sources=[],
            query_type=query_type.value,
            context=""
        )

    try:
        query_vector = embedder.encode_query(request.question)
        vector_results = qdrant_store.search(query_vector.tolist(), limit=request.limit or 10)

        vr = [
            VectorResult(
                id=r["id"],
                text=r["payload"]["text"],
                score=r["score"],
                metadata=r["payload"].get("metadata", {})
            )
            for r in vector_results
        ]

        gr = search_graph(query_vector.tolist(), top_k=request.limit or 10, query_text=request.question)
        merged = merger.merge(vr, gr, alpha, beta)

        sources = [
            SourceItem(
                type=r.source.value,
                id=r.id,
                score=r.normalized_score,
                text=r.text
            )
            for r in merged[:5]
        ]

        context = merger.build_context(merged, max_items=5)

        return QueryResponse(
            answer=f"[基于 {query_type.value} 检索] 请参考以下来源回答您的问题：",
            sources=sources,
            query_type=query_type.value,
            context=context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph", response_model=GraphResponse)
async def get_graph():
    try:
        graph_data = graph_exporter.load()
        communities = community_detector.get_community_stats(
            graph_data.get("community_map", {})
        )
        return GraphResponse(
            nodes=graph_data.get("nodes", []),
            edges=graph_data.get("edges", []),
            communities=communities
        )
    except FileNotFoundError:
        return GraphResponse(nodes=[], edges=[], communities=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def get_status():
    try:
        info = qdrant_store.get_collection_info()
        return StatusResponse(
            status="healthy",
            doc_count=info.get("points_count", 0),
            node_count=len(current_graph.get("nodes", [])),
            edge_count=len(current_graph.get("edges", [])),
            last_update=None
        )
    except Exception:
        return StatusResponse(
            status="degraded",
            doc_count=0,
            node_count=0,
            edge_count=0
        )


@app.get("/watch/{path:path}")
async def watch_directory(path: str):
    global file_watcher
    try:
        if file_watcher is not None:
            file_watcher.stop_watching()

        file_watcher = FileWatcher(root_path=path)

        def on_change(event_type: str, file_path: str):
            import asyncio
            asyncio.create_task(index_documents(IndexRequest(path=file_path), BackgroundTasks()))

        file_watcher.start_watching(on_change)
        return {"status": "watching", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/watch/stop")
async def stop_watching():
    global file_watcher
    if file_watcher:
        file_watcher.stop_watching()
        file_watcher = None
    return {"status": "stopped"}


@app.get("/")
async def root():
    return {"message": "Vector-to-Graph API", "version": "0.1.0"}