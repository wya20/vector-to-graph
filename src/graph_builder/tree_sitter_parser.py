from pydantic import BaseModel
from typing import List, Optional, Literal
from dataclasses import dataclass, field
import uuid
import os

try:
    import tree_sitter
    from tree_sitter import Language, Parser
except ImportError:
    raise ImportError("tree-sitter is not installed. Run: pip install tree-sitter")

try:
    import tree_sitter_python
except ImportError:
    tree_sitter_python = None

try:
    import tree_sitter_javascript
except ImportError:
    tree_sitter_javascript = None


@dataclass
class CodeNode:
    id: str
    label: str
    node_type: Literal["function", "class", "import", "module"]
    file: str
    line: int
    metadata: dict = field(default_factory=dict)


@dataclass
class CodeEdge:
    source: str
    target: str
    relation: Literal["defines", "imports", "calls", "contains"]
    confidence: float = 1.0
    evidence: Optional[str] = None


class TreeSitterParser:
    _parsers: dict = {}
    _languages: dict = {}

    def __init__(self):
        self._init_languages()

    def _init_languages(self):
        if tree_sitter_python is not None:
            self._languages['python'] = Language(tree_sitter_python.language())
        if tree_sitter_javascript is not None:
            self._languages['javascript'] = Language(tree_sitter_javascript.language())

    def _get_parser(self, language: str) -> Parser:
        if language not in self._parsers:
            if language not in self._languages:
                raise ValueError(f"Unsupported language: {language}")
            parser = Parser()
            parser.language = self._languages[language]
            self._parsers[language] = parser
        return self._parsers[language]

    def _detect_language(self, file_path: str) -> Optional[str]:
        ext = os.path.splitext(file_path)[1].lower()
        mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.mjs': 'javascript',
            '.cjs': 'javascript',
        }
        return mapping.get(ext)

    def _generate_node_id(self, file_path: str, label: str, node_type: str) -> str:
        return f"{file_path}:{label}"

    def _extract_python_nodes(self, tree: tree_sitter.Tree, content: bytes, file_path: str) -> List[CodeNode]:
        nodes = []

        def walk(node):
            if node.type == 'function_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    label = content[name_node.start_byte:name_node.end_byte].decode('utf-8')
                    nodes.append(CodeNode(
                        id=self._generate_node_id(file_path, label, 'function'),
                        label=label,
                        node_type='function',
                        file=file_path,
                        line=name_node.start_point[0] + 1,
                        metadata={
                            'start_line': node.start_point[0] + 1,
                            'end_line': node.end_point[0] + 1
                        }
                    ))
            elif node.type == 'class_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    label = content[name_node.start_byte:name_node.end_byte].decode('utf-8')
                    nodes.append(CodeNode(
                        id=self._generate_node_id(file_path, label, 'class'),
                        label=label,
                        node_type='class',
                        file=file_path,
                        line=name_node.start_point[0] + 1,
                        metadata={
                            'start_line': node.start_point[0] + 1,
                            'end_line': node.end_point[0] + 1
                        }
                    ))
            elif node.type == 'import_statement':
                imp_text = content[node.start_byte:node.end_byte].decode('utf-8')
                nodes.append(CodeNode(
                    id=str(uuid.uuid4()),
                    label=imp_text,
                    node_type='import',
                    file=file_path,
                    line=node.start_point[0] + 1,
                    metadata={'import_statement': imp_text}
                ))
            elif node.type == 'import_from_statement':
                imp_text = content[node.start_byte:node.end_byte].decode('utf-8')
                nodes.append(CodeNode(
                    id=str(uuid.uuid4()),
                    label=imp_text,
                    node_type='import',
                    file=file_path,
                    line=node.start_point[0] + 1,
                    metadata={'import_statement': imp_text}
                ))

            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return nodes

    def _extract_javascript_nodes(self, tree: tree_sitter.Tree, content: bytes, file_path: str) -> List[CodeNode]:
        nodes = []

        def walk(node):
            if node.type == 'function_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    label = content[name_node.start_byte:name_node.end_byte].decode('utf-8')
                    nodes.append(CodeNode(
                        id=self._generate_node_id(file_path, label, 'function'),
                        label=label,
                        node_type='function',
                        file=file_path,
                        line=name_node.start_point[0] + 1,
                        metadata={
                            'start_line': node.start_point[0] + 1,
                            'end_line': node.end_point[0] + 1
                        }
                    ))
            elif node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    label = content[name_node.start_byte:name_node.end_byte].decode('utf-8')
                    nodes.append(CodeNode(
                        id=self._generate_node_id(file_path, label, 'class'),
                        label=label,
                        node_type='class',
                        file=file_path,
                        line=name_node.start_point[0] + 1,
                        metadata={
                            'start_line': node.start_point[0] + 1,
                            'end_line': node.end_point[0] + 1
                        }
                    ))
            elif node.type == 'import_statement':
                imp_text = content[node.start_byte:node.end_byte].decode('utf-8')
                nodes.append(CodeNode(
                    id=str(uuid.uuid4()),
                    label=imp_text,
                    node_type='import',
                    file=file_path,
                    line=node.start_point[0] + 1,
                    metadata={'import_statement': imp_text}
                ))
            elif node.type == 'export_statement':
                exp_text = content[node.start_byte:node.end_byte].decode('utf-8')
                nodes.append(CodeNode(
                    id=str(uuid.uuid4()),
                    label=exp_text,
                    node_type='import',
                    file=file_path,
                    line=node.start_point[0] + 1,
                    metadata={'export_statement': exp_text}
                ))

            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return nodes

    def parse_file(self, file_path: str) -> List[CodeNode]:
        language = self._detect_language(file_path)
        if not language:
            raise ValueError(f"Cannot detect language for file: {file_path}")

        with open(file_path, 'rb') as f:
            content = f.read()

        return self.parse_content(content.decode('utf-8'), file_path, language)

    def parse_content(self, content: str, file_path: str, language: str) -> List[CodeNode]:
        content_bytes = content.encode('utf-8')
        parser = self._get_parser(language)
        tree = parser.parse(content_bytes)

        if language == 'python':
            return self._extract_python_nodes(tree, content_bytes, file_path)
        elif language == 'javascript':
            return self._extract_javascript_nodes(tree, content_bytes, file_path)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def extract_edges(self, nodes: List[CodeNode]) -> List[CodeEdge]:
        edges = []
        func_nodes = {n.label: n for n in nodes if n.node_type == 'function'}
        class_nodes = {n.label: n for n in nodes if n.node_type == 'class'}

        for node in nodes:
            if node.node_type == 'function':
                edges.append(CodeEdge(
                    source=node.file,
                    target=node.id,
                    relation='defines',
                    confidence=1.0,
                    evidence=f"Function defined at line {node.line}"
                ))
            elif node.node_type == 'class':
                edges.append(CodeEdge(
                    source=node.file,
                    target=node.id,
                    relation='defines',
                    confidence=1.0,
                    evidence=f"Class defined at line {node.line}"
                ))
            elif node.node_type == 'import':
                edges.append(CodeEdge(
                    source=node.file,
                    target=node.label,
                    relation='imports',
                    confidence=0.9,
                    evidence=f"Import at line {node.line}"
                ))

        return edges

    def build_call_graph(self, nodes: List[CodeNode], edges: List[CodeEdge]) -> dict:
        call_graph = {}

        for edge in edges:
            if edge.relation == 'calls':
                if edge.source not in call_graph:
                    call_graph[edge.source] = []
                call_graph[edge.source].append(edge.target)

        return {
            'nodes': [
                {
                    'id': n.id,
                    'label': n.label,
                    'type': n.node_type,
                    'file': n.file,
                    'line': n.line,
                    **n.metadata
                }
                for n in nodes
            ],
            'edges': [
                {
                    'source': e.source,
                    'target': e.target,
                    'relation': e.relation,
                    'confidence': e.confidence,
                    'evidence': e.evidence
                }
                for e in edges
            ],
            'call_graph': call_graph
        }


def _extract_function_calls_python(tree: tree_sitter.Tree, content: bytes, file_path: str, all_nodes: List[CodeNode]) -> List[CodeEdge]:
    edges = []
    func_ids = {n.label: n.id for n in all_nodes if n.node_type == 'function'}

    current_func = None

    def walk(node):
        nonlocal current_func

        if node.type == 'function_definition':
            name_node = node.child_by_field_name('name')
            if name_node:
                old_func = current_func
                current_func = content[name_node.start_byte:name_node.end_byte].decode('utf-8')
                for child in node.children:
                    walk(child)
                current_func = old_func
                return

        elif node.type == 'call' and current_func:
            func_name_node = node.child_by_field_name('function')
            if func_name_node:
                called_name = content[func_name_node.start_byte:func_name_node.end_byte].decode('utf-8')
                if called_name in func_ids:
                    edges.append(CodeEdge(
                        source=func_ids[current_func],
                        target=func_ids[called_name],
                        relation='calls',
                        confidence=1.0,
                        evidence=f"Call at line {node.start_point[0] + 1}"
                    ))

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return edges


def _extract_function_calls_javascript(tree: tree_sitter.Tree, content: bytes, file_path: str, all_nodes: List[CodeNode]) -> List[CodeEdge]:
    edges = []
    func_ids = {n.label: n.id for n in all_nodes if n.node_type == 'function'}

    current_func = None

    def walk(node):
        nonlocal current_func

        if node.type == 'function_declaration':
            name_node = node.child_by_field_name('name')
            if name_node:
                old_func = current_func
                current_func = content[name_node.start_byte:name_node.end_byte].decode('utf-8')
                for child in node.children:
                    walk(child)
                current_func = old_func
                return

        elif node.type == 'call_expression' and current_func:
            func_name_node = node.child_by_field_name('function')
            if func_name_node:
                called_name = content[func_name_node.start_byte:func_name_node.end_byte].decode('utf-8')
                if called_name in func_ids:
                    edges.append(CodeEdge(
                        source=func_ids[current_func],
                        target=func_ids[called_name],
                        relation='calls',
                        confidence=1.0,
                        evidence=f"Call at line {node.start_point[0] + 1}"
                    ))

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return edges


class EnhancedTreeSitterParser(TreeSitterParser):
    def extract_edges(self, nodes: List[CodeNode]) -> List[CodeEdge]:
        edges = super().extract_edges(nodes)

        if not nodes:
            return edges

        file_path = nodes[0].file if nodes else ""
        language = self._detect_language(file_path) if file_path else None

        if not language or not file_path:
            return edges

        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            parser = self._get_parser(language)
            tree = parser.parse(content)

            if language == 'python':
                call_edges = _extract_function_calls_python(tree, content, file_path, nodes)
            elif language == 'javascript':
                call_edges = _extract_function_calls_javascript(tree, content, file_path, nodes)
            else:
                call_edges = []

            edges.extend(call_edges)
        except Exception:
            pass

        return edges
