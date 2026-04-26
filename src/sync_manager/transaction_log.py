import json
import time
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class OperationType(Enum):
    UPSERT_VECTOR = "upsert_vector"
    DELETE_VECTOR = "delete_vector"
    UPSERT_GRAPH = "upsert_graph"
    DELETE_GRAPH = "delete_graph"


@dataclass
class TransactionLog:
    timestamp: float
    operation: str
    file_path: str
    status: str
    error: Optional[str] = None


class TransactionLogger:
    def __init__(self, log_dir: str = "./data/graphs/.transactions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self.log_dir / f"transactions_{int(time.time())}.jsonl"

    def log(self, operation: OperationType, file_path: str, status: str = "pending", error: Optional[str] = None):
        entry = TransactionLog(
            timestamp=time.time(),
            operation=operation.value,
            file_path=file_path,
            status=status,
            error=error
        )
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(entry)) + '\n')

    def mark_completed(self, operation: OperationType, file_path: str):
        self.log(operation, file_path, status="completed")

    def mark_failed(self, operation: OperationType, file_path: str, error: str):
        self.log(operation, file_path, status="failed", error=error)

    def get_pending_transactions(self) -> List[TransactionLog]:
        pending = []
        for log_file in self.log_dir.glob("transactions_*.jsonl"):
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry['status'] == "pending":
                        pending.append(TransactionLog(**entry))
        return pending