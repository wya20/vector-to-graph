import hashlib
from pathlib import Path
from typing import List, Set, Optional, Dict, Callable
from dataclasses import dataclass, field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent


@dataclass
class FileState:
    path: str
    sha256: str
    mtime: float
    indexed: bool = False


@dataclass
class FileWatcher:
    root_path: str
    extensions: Set[str] = field(default_factory=lambda: {'.py', '.js', '.ts', '.jsx', '.tsx', '.md'})
    ignore_dirs: Set[str] = field(default_factory=lambda: {'__pycache__', '.git', 'node_modules', 'venv', '.venv'})

    def __post_init__(self):
        self.file_states: Dict[str, FileState] = {}
        self.observer: Optional[Observer] = None

    def compute_sha256(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def scan_directory(self) -> List[FileState]:
        states = []
        for ext in self.extensions:
            for file_path in Path(self.root_path).rglob(f"*{ext}"):
                if any(ignore in str(file_path) for ignore in self.ignore_dirs):
                    continue
                try:
                    stat = file_path.stat()
                    sha256 = self.compute_sha256(str(file_path))
                    state = FileState(
                        path=str(file_path),
                        sha256=sha256,
                        mtime=stat.st_mtime
                    )
                    states.append(state)
                    self.file_states[str(file_path)] = state
                except Exception:
                    continue
        return states

    def check_changes(self) -> List[FileState]:
        changed = []

        for ext in self.extensions:
            for file_path in Path(self.root_path).rglob(f"*{ext}"):
                if any(ignore in str(file_path) for ignore in self.ignore_dirs):
                    continue
                try:
                    stat = file_path.stat()
                    sha256 = self.compute_sha256(str(file_path))
                    current_state = FileState(
                        path=str(file_path),
                        sha256=sha256,
                        mtime=stat.st_mtime
                    )

                    old_state = self.file_states.get(str(file_path))
                    if old_state is None:
                        changed.append(current_state)
                    elif old_state.sha256 != sha256:
                        changed.append(current_state)
                except Exception:
                    continue

        return changed

    def start_watching(self, callback: Callable[[str, str], None]):
        event_handler = FileChangeHandler(self, callback)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.root_path, recursive=True)
        self.observer.start()

    def stop_watching(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, watcher: FileWatcher, callback: Callable[[str, str], None]):
        self.watcher = watcher
        self.callback = callback

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and event.src_path.endswith(tuple(self.watcher.extensions)):
            self.callback('modified', event.src_path)

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and event.src_path.endswith(tuple(self.watcher.extensions)):
            self.callback('created', event.src_path)

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory and event.src_path.endswith(tuple(self.watcher.extensions)):
            self.callback('deleted', event.src_path)