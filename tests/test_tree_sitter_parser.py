import pytest
import os
import tempfile
from pathlib import Path

from src.graph_builder.tree_sitter_parser import (
    TreeSitterParser,
    EnhancedTreeSitterParser,
    CodeNode,
    CodeEdge
)


class TestTreeSitterParser:
    def setup_method(self):
        self.parser = TreeSitterParser()

    def test_parse_python_function(self):
        code = """
def foo():
    pass
"""
        nodes = self.parser.parse_content(code, "test.py", "python")
        assert len(nodes) >= 1
        func_nodes = [n for n in nodes if n.node_type == "function"]
        assert len(func_nodes) == 1
        assert func_nodes[0].label == "foo"
        assert func_nodes[0].file == "test.py"

    def test_parse_python_class(self):
        code = """
class MyClass:
    def method(self):
        pass
"""
        nodes = self.parser.parse_content(code, "test.py", "python")
        class_nodes = [n for n in nodes if n.node_type == "class"]
        assert len(class_nodes) == 1
        assert class_nodes[0].label == "MyClass"

    def test_parse_python_import(self):
        code = """
import os
from sys import path
"""
        nodes = self.parser.parse_content(code, "test.py", "python")
        import_nodes = [n for n in nodes if n.node_type == "import"]
        assert len(import_nodes) == 2

    def test_parse_javascript_function(self):
        code = """
function foo() {
    return 1;
}
"""
        nodes = self.parser.parse_content(code, "test.js", "javascript")
        func_nodes = [n for n in nodes if n.node_type == "function"]
        assert len(func_nodes) == 1
        assert func_nodes[0].label == "foo"

    def test_parse_javascript_class(self):
        code = """
class MyClass {
    constructor() {}
}
"""
        nodes = self.parser.parse_content(code, "test.js", "javascript")
        class_nodes = [n for n in nodes if n.node_type == "class"]
        assert len(class_nodes) == 1
        assert class_nodes[0].label == "MyClass"

    def test_parse_javascript_import(self):
        code = """
import foo from 'bar';
export const x = 1;
"""
        nodes = self.parser.parse_content(code, "test.js", "javascript")
        import_nodes = [n for n in nodes if n.node_type == "import"]
        assert len(import_nodes) >= 1

    def test_extract_edges(self):
        code = """
def foo():
    pass
"""
        nodes = self.parser.parse_content(code, "test.py", "python")
        edges = self.parser.extract_edges(nodes)
        assert len(edges) >= 1
        defines_edges = [e for e in edges if e.relation == "defines"]
        assert len(defines_edges) == 1

    def test_build_call_graph(self):
        code = """
def foo():
    pass
"""
        nodes = self.parser.parse_content(code, "test.py", "python")
        edges = self.parser.extract_edges(nodes)
        call_graph = self.parser.build_call_graph(nodes, edges)
        assert "nodes" in call_graph
        assert "edges" in call_graph
        assert "call_graph" in call_graph
        assert len(call_graph["nodes"]) == len(nodes)

    def test_detect_language(self):
        assert self.parser._detect_language("test.py") == "python"
        assert self.parser._detect_language("test.js") == "javascript"
        assert self.parser._detect_language("test.mjs") == "javascript"
        assert self.parser._detect_language("test.unknown") is None

    def test_parse_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test_func():\n    pass\n")
            temp_path = f.name

        try:
            nodes = self.parser.parse_file(temp_path)
            assert len(nodes) >= 1
            func_nodes = [n for n in nodes if n.node_type == "function"]
            assert len(func_nodes) == 1
            assert func_nodes[0].label == "test_func"
        finally:
            os.unlink(temp_path)

    def test_code_node_dataclass(self):
        node = CodeNode(
            id="test.py:foo",
            label="foo",
            node_type="function",
            file="test.py",
            line=1,
            metadata={"key": "value"}
        )
        assert node.id == "test.py:foo"
        assert node.label == "foo"
        assert node.node_type == "function"
        assert node.metadata["key"] == "value"

    def test_code_edge_dataclass(self):
        edge = CodeEdge(
            source="test.py:foo",
            target="test.py:bar",
            relation="calls",
            confidence=0.95,
            evidence="Call at line 5"
        )
        assert edge.source == "test.py:foo"
        assert edge.target == "test.py:bar"
        assert edge.relation == "calls"
        assert edge.confidence == 0.95

    def test_unsupported_language(self):
        with pytest.raises(ValueError):
            self.parser._get_parser("unsupported")

    def test_parse_multiple_functions(self):
        code = """
def foo():
    pass

def bar():
    foo()
"""
        nodes = self.parser.parse_content(code, "test.py", "python")
        func_nodes = [n for n in nodes if n.node_type == "function"]
        assert len(func_nodes) == 2
        labels = {n.label for n in func_nodes}
        assert labels == {"foo", "bar"}


class TestEnhancedTreeSitterParser:
    def setup_method(self):
        self.parser = EnhancedTreeSitterParser()

    def test_enhanced_extract_edges_with_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def foo():\n    pass\n")
            temp_path = f.name

        try:
            nodes = self.parser.parse_file(temp_path)
            edges = self.parser.extract_edges(nodes)
            assert len(edges) >= 1
        finally:
            os.unlink(temp_path)

    def test_example_from_requirements(self):
        code = """def foo():
    bar()
"""
        nodes = self.parser.parse_content(code, "file1.py", "python")
        func_nodes = [n for n in nodes if n.node_type == "function"]
        assert len(func_nodes) == 1
        assert func_nodes[0].label == "foo"

        edges = self.parser.extract_edges(nodes)
        assert len(edges) >= 1

    def test_calls_edge_extraction(self):
        code = """def foo():
    bar()
"""
        nodes = self.parser.parse_content(code, "test.py", "python")
        edges = self.parser.extract_edges(nodes)

        defines_edges = [e for e in edges if e.relation == "defines"]
        assert len(defines_edges) >= 1

        func_edges = [e for e in edges if e.relation == "calls"]
        assert len(func_edges) == 0

    def test_cross_file_calls_edge(self):
        code1 = """def foo():
    pass
"""
        code2 = """def bar():
    foo()
"""
        nodes1 = self.parser.parse_content(code1, "file1.py", "python")
        nodes2 = self.parser.parse_content(code2, "file2.py", "python")

        global_func_ids = {}
        for n in nodes1:
            if n.node_type == "function":
                global_func_ids[n.label] = n.id
        for n in nodes2:
            if n.node_type == "function":
                global_func_ids[n.label] = n.id

        edges2 = self.parser.extract_edges(nodes2, global_func_ids, code2.encode('utf-8'))

        calls_edges = [e for e in edges2 if e.relation == "calls"]
        assert len(calls_edges) == 1
        assert calls_edges[0].relation == "calls"
        assert "bar" in calls_edges[0].source or "bar" in str(calls_edges[0].source)
        assert "foo" in calls_edges[0].target or "foo" in str(calls_edges[0].target)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
