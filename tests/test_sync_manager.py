import os
import tempfile
import time
import pytest
import shutil
from pathlib import Path

from src.sync_manager.file_watcher import FileWatcher, FileState, FileChangeHandler
from src.sync_manager.transaction_log import TransactionLogger, TransactionLog, OperationType


class TestFileWatcher:
    def test_file_state_creation(self):
        state = FileState(path="/test.py", sha256="abc123", mtime=123456.0, indexed=False)
        assert state.path == "/test.py"
        assert state.sha256 == "abc123"
        assert state.mtime == 123456.0
        assert state.indexed is False

    def test_file_watcher_initialization(self):
        watcher = FileWatcher(root_path="/tmp", extensions={'.py'}, ignore_dirs={'__pycache__'})
        assert watcher.root_path == "/tmp"
        assert '.py' in watcher.extensions
        assert '__pycache__' in watcher.ignore_dirs
        assert len(watcher.file_states) == 0
        assert watcher.observer is None

    def test_compute_sha256(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')")
            temp_path = f.name

        try:
            watcher = FileWatcher(root_path="/tmp")
            sha = watcher.compute_sha256(temp_path)
            assert isinstance(sha, str)
            assert len(sha) == 64
        finally:
            os.unlink(temp_path)

    def test_scan_directory_finds_python_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "test.py")
            with open(py_file, "w") as f:
                f.write("def hello(): pass")

            md_file = os.path.join(tmpdir, "readme.md")
            with open(md_file, "w") as f:
                f.write("# Title")

            txt_file = os.path.join(tmpdir, "data.txt")
            with open(txt_file, "w") as f:
                f.write("ignored")

            watcher = FileWatcher(root_path=tmpdir, extensions={'.py', '.md'})
            states = watcher.scan_directory()

            assert len(states) == 2
            paths = [s.path for s in states]
            assert py_file in paths
            assert md_file in paths
            assert txt_file not in paths

    def test_scan_directory_ignores_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "__pycache__")
            os.makedirs(subdir)
            py_file = os.path.join(subdir, "cached.py")
            with open(py_file, "w") as f:
                f.write("# cached")

            main_py = os.path.join(tmpdir, "main.py")
            with open(main_py, "w") as f:
                f.write("# main")

            watcher = FileWatcher(root_path=tmpdir, ignore_dirs={'__pycache__'})
            states = watcher.scan_directory()

            paths = [s.path for s in states]
            assert main_py in paths
            assert py_file not in paths

    def test_check_changes_detects_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher(root_path=tmpdir)
            watcher.scan_directory()

            new_file = os.path.join(tmpdir, "new.py")
            with open(new_file, "w") as f:
                f.write("# new file")

            changed = watcher.check_changes()
            assert len(changed) == 1
            assert changed[0].path == new_file

    def test_check_changes_detects_modified_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "test.py")
            with open(py_file, "w") as f:
                f.write("v1")

            watcher = FileWatcher(root_path=tmpdir)
            watcher.scan_directory()

            time.sleep(0.1)
            with open(py_file, "w") as f:
                f.write("v2")

            changed = watcher.check_changes()
            assert len(changed) == 1
            assert changed[0].path == py_file

    def test_check_changes_no_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "test.py")
            with open(py_file, "w") as f:
                f.write("content")

            watcher = FileWatcher(root_path=tmpdir)
            watcher.scan_directory()

            changed = watcher.check_changes()
            assert len(changed) == 0

    def test_stop_watching_without_start(self):
        watcher = FileWatcher(root_path="/tmp")
        watcher.stop_watching()
        assert watcher.observer is None


class TestFileChangeHandler:
    def test_callback_on_modified(self):
        events = []

        def callback(event_type, path):
            events.append((event_type, path))

        watcher = FileWatcher(root_path="/tmp")
        handler = FileChangeHandler(watcher, callback)
        handler.on_modified(type('Event', (), {'is_directory': False, 'src_path': '/test.py', 'src_path.endswith': lambda s, x: s.endswith(x)})())

        assert len(events) == 1
        assert events[0] == ('modified', '/test.py')

    def test_callback_on_created(self):
        events = []

        def callback(event_type, path):
            events.append((event_type, path))

        watcher = FileWatcher(root_path="/tmp")
        handler = FileChangeHandler(watcher, callback)
        handler.on_created(type('Event', (), {'is_directory': False, 'src_path': '/new.py', 'src_path.endswith': lambda s, x: s.endswith(x)})())

        assert len(events) == 1
        assert events[0] == ('created', '/new.py')

    def test_callback_on_deleted(self):
        events = []

        def callback(event_type, path):
            events.append((event_type, path))

        watcher = FileWatcher(root_path="/tmp")
        handler = FileChangeHandler(watcher, callback)
        handler.on_deleted(type('Event', (), {'is_directory': False, 'src_path': '/deleted.py', 'src_path.endswith': lambda s, x: s.endswith(x)})())

        assert len(events) == 1
        assert events[0] == ('deleted', '/deleted.py')

    def test_ignores_directory_events(self):
        events = []

        def callback(event_type, path):
            events.append((event_type, path))

        watcher = FileWatcher(root_path="/tmp")
        handler = FileChangeHandler(watcher, callback)
        handler.on_modified(type('Event', (), {'is_directory': True, 'src_path': '/some_dir'})())

        assert len(events) == 0


class TestTransactionLogger:
    def test_transaction_log_creation(self):
        log = TransactionLog(
            timestamp=123456.0,
            operation="upsert_vector",
            file_path="/test.py",
            status="pending",
            error=None
        )
        assert log.timestamp == 123456.0
        assert log.operation == "upsert_vector"
        assert log.status == "pending"

    def test_operation_type_values(self):
        assert OperationType.UPSERT_VECTOR.value == "upsert_vector"
        assert OperationType.DELETE_VECTOR.value == "delete_vector"
        assert OperationType.UPSERT_GRAPH.value == "upsert_graph"
        assert OperationType.DELETE_GRAPH.value == "delete_graph"

    def test_transaction_logger_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TransactionLogger(log_dir=tmpdir)
            logger.log(OperationType.UPSERT_VECTOR, "/test.py", status="pending")

            log_files = list(Path(tmpdir).glob("transactions_*.jsonl"))
            assert len(log_files) == 1

            with open(log_files[0], 'r') as f:
                line = f.readline()
                entry = eval(line.replace('true', 'True').replace('false', 'False').replace('null', 'None'))

            assert entry['operation'] == "upsert_vector"
            assert entry['file_path'] == "/test.py"
            assert entry['status'] == "pending"

    def test_mark_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TransactionLogger(log_dir=tmpdir)
            logger.mark_completed(OperationType.UPSERT_GRAPH, "/test.py")

            log_files = list(Path(tmpdir).glob("transactions_*.jsonl"))
            with open(log_files[0], 'r') as f:
                line = f.readline()
                entry = eval(line.replace('true', 'True').replace('false', 'False').replace('null', 'None'))

            assert entry['status'] == "completed"
            assert entry['operation'] == "upsert_graph"

    def test_mark_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TransactionLogger(log_dir=tmpdir)
            logger.mark_failed(OperationType.DELETE_VECTOR, "/test.py", "File not found")

            log_files = list(Path(tmpdir).glob("transactions_*.jsonl"))
            with open(log_files[0], 'r') as f:
                line = f.readline()
                entry = eval(line.replace('true', 'True').replace('false', 'False').replace('null', 'None'))

            assert entry['status'] == "failed"
            assert entry['error'] == "File not found"

    def test_get_pending_transactions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger1 = TransactionLogger(log_dir=tmpdir)
            logger1.log(OperationType.UPSERT_VECTOR, "/test1.py", status="pending")
            logger1.log(OperationType.UPSERT_GRAPH, "/test2.py", status="completed")

            logger2 = TransactionLogger(log_dir=tmpdir)
            logger2.log(OperationType.DELETE_VECTOR, "/test3.py", status="pending")

            pending = logger2.get_pending_transactions()
            assert len(pending) == 2

            pending_paths = [p.file_path for p in pending]
            assert "/test1.py" in pending_paths
            assert "/test3.py" in pending_paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])