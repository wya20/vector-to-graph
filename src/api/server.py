from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import asyncio
import uuid
from pathlib import Path

from ..vector_indexer.embedder import Embedder
from ..vector_indexer.chunker import Chunker
from ..vector_indexer.qdrant_client import QdrantVectorStore
from ..graph_builder.tree_sitter_parser import TreeSitterParser
from ..graph_builder.community_detector import CommunityDetector
from ..graph_builder.graph_exporter import GraphExporter
from ..query_router.router import QueryRouter, QueryType
from ..result_merger.merger import ResultMerger, VectorResult, GraphResult
from ..utils.config import config

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
current_graph: Dict = {"nodes": [], "edges": [], "community_map": {}}


def init_components():
    global embedder, chunker, qdrant_store, parser, community_detector, graph_exporter, router, merger
    embedder = Embedder()
    chunker = Chunker()
    qdrant_store = QdrantVectorStore(host="qdrant", port=6333)
    parser = TreeSitterParser()
    community_detector = CommunityDetector()
    graph_exporter = GraphExporter()
    router = QueryRouter()
    merger = ResultMerger(alpha=0.4, beta=0.6)


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
        try:
            indexing_tasks[task_id]["status"] = "running"
            Path(request.path)

            chunks = chunker.chunk_file(request.path)

            vectors = []
            for chunk in chunks:
                embedding = embedder.encode([chunk.text])[0]
                vectors.append({
                    "id": chunk.id,
                    "vector": embedding.tolist(),
                    "payload": {
                        "text": chunk.text,
                        "metadata": chunk.metadata
                    }
                })

            qdrant_store.upsert(vectors)

            nodes, edges = [], []
            for chunk in chunks:
                if chunk.metadata.get("chunk_type") in ["code_function", "code_class"]:
                    nodes.append({
                        "id": chunk.id,
                        "label": chunk.metadata.get("function_name") or chunk.metadata.get("class_name"),
                        "type": chunk.metadata.get("chunk_type"),
                        "metadata": chunk.metadata
                    })

            if nodes:
                community_map = community_detector.detect_communities(
                    community_detector.build_graph(nodes, edges)
                )
                graph_exporter.export(nodes, edges, community_map)

            indexing_tasks[task_id] = {"status": "completed", "path": request.path}
        except Exception as e:
            indexing_tasks[task_id] = {"status": "failed", "error": str(e)}

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

        gr = []
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
    except Exception as e:
        return StatusResponse(
            status="degraded",
            doc_count=0,
            node_count=0,
            edge_count=0
        )


@app.get("/")
async def root():
    return {"message": "Vector-to-Graph API", "version": "0.1.0"}