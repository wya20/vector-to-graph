import re
import uuid
from pathlib import Path
from typing import List, Optional, Literal

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    source: str
    chunk_type: Literal["code_function", "code_class", "markdown_paragraph"]
    language: Optional[str] = None
    line_start: int
    line_end: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None


class Chunk(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata


class Chunker:
    MAX_TOKENS = 500

    def __init__(self, max_tokens: int = 500):
        self.max_tokens = max_tokens

    def chunk_file(self, file_path: str) -> List[Chunk]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        source = str(path)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if suffix == ".py":
            return self.chunk_code(content, source, "python")
        elif suffix in [".js", ".ts", ".jsx", ".tsx"]:
            return self.chunk_code(content, source, "javascript")
        elif suffix == ".md":
            return self.chunk_markdown(content, source)
        else:
            return []

    def chunk_code(self, content: str, source: str, language: str) -> List[Chunk]:
        if language == "python":
            return self._chunk_python(content, source)
        elif language == "javascript":
            return self._chunk_javascript(content, source)
        return []

    def chunk_markdown(self, content: str, source: str) -> List[Chunk]:
        chunks = []
        lines = content.split("\n")
        paragraphs = self._split_markdown_paragraphs(lines)

        for para_idx, (para_lines, para_start_line) in enumerate(paragraphs):
            para_text = "\n".join(para_lines)
            if not para_text.strip():
                continue

            para_end_line = para_start_line + len(para_lines) - 1
            estimated_tokens = len(para_text) // 4

            if estimated_tokens <= self.max_tokens:
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    text=para_text,
                    metadata=ChunkMetadata(
                        source=source,
                        chunk_type="markdown_paragraph",
                        language="markdown",
                        line_start=para_start_line,
                        line_end=para_end_line,
                    ),
                )
                chunks.append(chunk)
            else:
                sub_chunks = self._split_large_paragraph(para_text, source, para_start_line)
                chunks.extend(sub_chunks)

        return chunks

    def _split_markdown_paragraphs(self, lines: List[str]) -> List[tuple]:
        paragraphs = []
        current_para = []
        current_start = 1

        for i, line in enumerate(lines):
            stripped = line.strip()
            is_heading = stripped.startswith("#") and (stripped.startswith("##") is False or stripped.startswith("###"))
            is_code_block = stripped.startswith("```")
            is_empty = stripped == ""

            if is_empty and current_para:
                paragraphs.append((current_para, current_start))
                current_para = []
                current_start = i + 2
            elif is_heading and current_para:
                paragraphs.append((current_para, current_start))
                current_para = [line]
                current_start = i + 1
            elif is_heading and not current_para:
                current_para = [line]
                current_start = i + 1
            elif is_code_block:
                if current_para:
                    paragraphs.append((current_para, current_start))
                    current_para = []
                    current_start = i + 2
                code_block_lines = [line]
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("```"):
                    code_block_lines.append(lines[j])
                    j += 1
                if j < len(lines):
                    code_block_lines.append(lines[j])
                paragraphs.append((code_block_lines, i + 1))
                current_para = []
                current_start = j + 2
                i = j
            elif not is_empty:
                if not current_para:
                    current_start = i + 1
                current_para.append(line)

        if current_para:
            paragraphs.append((current_para, current_start))

        return paragraphs

    def _split_large_paragraph(self, text: str, source: str, start_line: int) -> List[Chunk]:
        chunks = []
        sentences = re.split(r"(?<=[。.!?；;])\s+", text)
        current_chunk_lines = []
        current_chunk_tokens = 0
        current_line = start_line

        for sentence in sentences:
            sentence_tokens = len(sentence) // 4
            if current_chunk_tokens + sentence_tokens <= self.max_tokens:
                current_chunk_lines.append(sentence)
                current_chunk_tokens += sentence_tokens
            else:
                if current_chunk_lines:
                    chunk_text = " ".join(current_chunk_lines)
                    end_line = current_line + len(current_chunk_lines) - 1
                    chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            text=chunk_text,
                            metadata=ChunkMetadata(
                                source=source,
                                chunk_type="markdown_paragraph",
                                language="markdown",
                                line_start=current_line,
                                line_end=end_line,
                            ),
                        )
                    )
                current_chunk_lines = [sentence]
                current_chunk_tokens = sentence_tokens
                current_line = start_line + len(sentences)

        if current_chunk_lines:
            chunk_text = " ".join(current_chunk_lines)
            end_line = current_line + len(current_chunk_lines) - 1
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        source=source,
                        chunk_type="markdown_paragraph",
                        language="markdown",
                        line_start=current_line,
                        line_end=end_line,
                    ),
                )
            )

        return chunks

    def _chunk_python(self, content: str, source: str) -> List[Chunk]:
        chunks = []
        lines = content.split("\n")
        classes = self._find_python_classes(lines)
        functions = self._find_python_functions(lines)

        all_defs = []
        for cls in classes:
            all_defs.append((cls["line"], "class", cls))
        for func in functions:
            all_defs.append((func["line"], "function", func))
        all_defs.sort(key=lambda x: x[0])

        for i, (line_num, kind, definition) in enumerate(all_defs):
            if kind == "class":
                start_line = definition["line"]
                end_line = definition["end_line"]
                class_name = definition["name"]
                next_start = all_defs[i + 1][0] if i + 1 < len(all_defs) else len(lines) + 1
                end_line = min(end_line, next_start - 1)
                chunk_lines = lines[start_line - 1 : end_line]
                chunk_text = "\n".join(chunk_lines)

                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        text=chunk_text,
                        metadata=ChunkMetadata(
                            source=source,
                            chunk_type="code_class",
                            language="python",
                            line_start=start_line,
                            line_end=end_line,
                            class_name=class_name,
                        ),
                    )
                )
            else:
                func = definition
                start_line = func["line"]
                end_line = func["end_line"]
                func_name = func["name"]
                next_start = all_defs[i + 1][0] if i + 1 < len(all_defs) else len(lines) + 1
                end_line = min(end_line, next_start - 1)
                chunk_lines = lines[start_line - 1 : end_line]
                chunk_text = "\n".join(chunk_lines)

                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        text=chunk_text,
                        metadata=ChunkMetadata(
                            source=source,
                            chunk_type="code_function",
                            language="python",
                            line_start=start_line,
                            line_end=end_line,
                            function_name=func_name,
                        ),
                    )
                )

        return chunks

    def _chunk_javascript(self, content: str, source: str) -> List[Chunk]:
        chunks = []
        lines = content.split("\n")
        classes = self._find_js_classes(lines)
        functions = self._find_js_functions(lines)

        all_defs = []
        for cls in classes:
            all_defs.append((cls["line"], "class", cls))
        for func in functions:
            all_defs.append((func["line"], "function", func))
        all_defs.sort(key=lambda x: x[0])

        for i, (line_num, kind, definition) in enumerate(all_defs):
            start_line = definition["line"]
            end_line = definition["end_line"]
            name = definition["name"]

            if kind == "class":
                chunk_type = "code_class"
                class_name = name
                function_name = None
            else:
                chunk_type = "code_function"
                class_name = None
                function_name = name

            next_start = all_defs[i + 1][0] if i + 1 < len(all_defs) else len(lines) + 1
            end_line = min(end_line, next_start - 1)
            chunk_lines = lines[start_line - 1 : end_line]
            chunk_text = "\n".join(chunk_lines)

            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        source=source,
                        chunk_type=chunk_type,
                        language="javascript",
                        line_start=start_line,
                        line_end=end_line,
                        function_name=function_name,
                        class_name=class_name,
                    ),
                )
            )

        return chunks

    def _find_python_classes(self, lines: List[str]) -> List[dict]:
        classes = []
        in_docstring = False
        docstring_char = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_char = stripped[:3]
                    if stripped.count(docstring_char) >= 2:
                        in_docstring = False
                    else:
                        in_docstring = True
                elif stripped.startswith("class "):
                    match = re.match(r"class\s+(\w+)", stripped)
                    if match:
                        class_name = match.group(1)
                        line_num = i + 1
                        end_line = self._find_python_indent_block(lines, i)
                        classes.append({"name": class_name, "line": line_num, "end_line": end_line})
            else:
                if docstring_char and docstring_char in stripped:
                    in_docstring = False

        return classes

    def _find_python_functions(self, lines: List[str]) -> List[dict]:
        functions = []
        in_docstring = False
        docstring_char = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_char = stripped[:3]
                    if stripped.count(docstring_char) >= 2:
                        in_docstring = False
                    else:
                        in_docstring = True
                elif stripped.startswith("def "):
                    match = re.match(r"def\s+(\w+)", stripped)
                    if match:
                        func_name = match.group(1)
                        line_num = i + 1
                        end_line = self._find_python_indent_block(lines, i)
                        functions.append({"name": func_name, "line": line_num, "end_line": end_line})
            else:
                if docstring_char and docstring_char in stripped:
                    in_docstring = False

        return functions

    def _find_python_indent_block(self, lines: List[str], start_idx: int) -> int:
        if start_idx >= len(lines):
            return start_idx + 1

        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        if base_indent == len(lines[start_idx]):
            base_indent = 0

        end_line = start_idx + 1
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if line.strip():
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent:
                    break
                end_line = i + 1

        return end_line

    def _find_js_classes(self, lines: List[str]) -> List[dict]:
        classes = []
        in_block_comment = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            if in_block_comment:
                if "*/" in stripped:
                    in_block_comment = False
                continue

            if stripped.startswith("/*"):
                in_block_comment = True
                continue

            if stripped.startswith("//"):
                continue

            if stripped.startswith("class "):
                match = re.match(r"class\s+(\w+)", stripped)
                if match:
                    class_name = match.group(1)
                    line_num = i + 1
                    end_line = self._find_js_block(lines, i)
                    classes.append({"name": class_name, "line": line_num, "end_line": end_line})

        return classes

    def _find_js_functions(self, lines: List[str]) -> List[dict]:
        functions = []
        in_block_comment = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            if in_block_comment:
                if "*/" in stripped:
                    in_block_comment = False
                continue

            if stripped.startswith("/*"):
                in_block_comment = True
                continue

            if stripped.startswith("//"):
                continue

            if stripped.startswith("function ") or stripped.startswith("async function "):
                match = re.match(r"(?:async\s+)?function\s+(\w+)", stripped)
                if match:
                    func_name = match.group(1)
                    line_num = i + 1
                    end_line = self._find_js_block(lines, i)
                    functions.append({"name": func_name, "line": line_num, "end_line": end_line})

            elif re.match(r"^\w+\s*\([^)]*\)\s*{", stripped) or re.match(r"^\w+\s*:\s*function", stripped):
                if ":" in stripped and "function" not in stripped.split(":")[0]:
                    continue
                match = re.match(r"(\w+)\s*\(", stripped)
                if match:
                    possible_name = match.group(1)
                    if possible_name not in ["if", "else", "for", "while", "switch", "catch", "return"]:
                        func_name = possible_name
                        line_num = i + 1
                        end_line = self._find_js_block(lines, i)
                        functions.append({"name": func_name, "line": line_num, "end_line": end_line})

        return functions

    def _find_js_block(self, lines: List[str], start_idx: int) -> int:
        if start_idx >= len(lines):
            return start_idx + 1

        brace_count = 0
        found_first = False

        for i in range(start_idx, len(lines)):
            for char in lines[i]:
                if char == "{":
                    brace_count += 1
                    found_first = True
                elif char == "}":
                    brace_count -= 1

            if found_first and brace_count == 0:
                return i + 1

        return len(lines)
