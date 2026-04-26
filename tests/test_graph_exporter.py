import json
import pytest
import tempfile
import shutil
from pathlib import Path
from src.graph_builder.graph_exporter import GraphExporter


class TestGraphExporter:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = GraphExporter(output_dir=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_output_dir(self):
        temp_dir = tempfile.mkdtemp()
        try:
            exporter = GraphExporter(output_dir=temp_dir)
            assert Path(temp_dir).exists()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_export_basic(self):
        nodes = [{'id': '1', 'label': 'A'}]
        edges = [{'source': '1', 'target': '2'}]
        path = self.exporter.export(nodes, edges)
        assert path == str(Path(self.temp_dir) / "graph.json")
        assert Path(path).exists()

    def test_export_with_community_map(self):
        nodes = [{'id': '1'}]
        edges = []
        community_map = {'1': 0, '2': 0}
        path = self.exporter.export(nodes, edges, community_map=community_map)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert 'community_map' in data
        assert data['community_map'] == community_map

    def test_export_metadata(self):
        nodes = [{'id': '1'}, {'id': '2'}]
        edges = [{'source': '1', 'target': '2'}]
        path = self.exporter.export(nodes, edges)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data['metadata']['node_count'] == 2
        assert data['metadata']['edge_count'] == 1

    def test_export_custom_filename(self):
        nodes = []
        edges = []
        path = self.exporter.export(nodes, edges, filename="custom.json")
        assert path == str(Path(self.temp_dir) / "custom.json")

    def test_export_content_structure(self):
        nodes = [{'id': '1', 'metadata': {'label': 'A'}}]
        edges = [{'source': '1', 'target': '2', 'weight': 1.0}]
        path = self.exporter.export(nodes, edges)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert 'nodes' in data
        assert 'edges' in data
        assert 'metadata' in data
        assert len(data['nodes']) == 1
        assert len(data['edges']) == 1

    def test_load(self):
        nodes = [{'id': '1'}]
        edges = [{'source': '1', 'target': '2'}]
        self.exporter.export(nodes, edges, filename="test.json")
        data = self.exporter.load("test.json")
        assert data['metadata']['node_count'] == 1
        assert data['metadata']['edge_count'] == 1

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            self.exporter.load("nonexistent.json")