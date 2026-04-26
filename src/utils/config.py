import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Config(BaseModel):
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "vector-to-graph")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    embedding_dim: int = 512
    chunk_size: int = 512
    graph_storage_path: str = os.getenv("GRAPH_STORAGE_PATH", "./data/graphs")
    vector_storage_path: str = os.getenv("VECTOR_STORAGE_PATH", "./data/vectors")
    raw_data_path: str = os.getenv("RAW_DATA_PATH", "./data/raw")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

config = Config()
