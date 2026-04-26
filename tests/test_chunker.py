import os
import tempfile
import pytest

from src.vector_indexer.chunker import Chunker, Chunk, ChunkMetadata


class TestChunkerPython:
    def test_chunk_python_class(self):
        code = '''class MyClass:
    def __init__(self):
        pass

    def my_method(self):
        return 42
'''
        chunker = Chunker()
        chunks = chunker.chunk_code(code, "test.py", "python")

        class_chunks = [c for c in chunks if c.metadata.chunk_type == "code_class"]
        assert len(class_chunks) == 1
        assert class_chunks[0].metadata.class_name == "MyClass"

    def test_chunk_python_functions(self):
        code = '''def func_one():
    pass

def func_two():
    pass
'''
        chunker = Chunker()
        chunks = chunker.chunk_code(code, "test.py", "python")

        func_chunks = [c for c in chunks if c.metadata.chunk_type == "code_function"]
        assert len(func_chunks) == 2
        func_names = [c.metadata.function_name for c in func_chunks]
        assert "func_one" in func_names
        assert "func_two" in func_names

    def test_chunk_python_mixed(self):
        code = '''class MyClass:
    def method_one(self):
        pass

def standalone_func():
    pass
'''
        chunker = Chunker()
        chunks = chunker.chunk_code(code, "test.py", "python")

        assert len(chunks) == 3

        class_chunks = [c for c in chunks if c.metadata.chunk_type == "code_class"]
        assert len(class_chunks) == 1

        func_chunks = [c for c in chunks if c.metadata.chunk_type == "code_function"]
        assert len(func_chunks) == 2


class TestChunkerJavaScript:
    def test_chunk_js_class(self):
        code = '''class MyClass {
    constructor() {
    }

    myMethod() {
        return 42;
    }
}
'''
        chunker = Chunker()
        chunks = chunker.chunk_code(code, "test.js", "javascript")

        class_chunks = [c for c in chunks if c.metadata.chunk_type == "code_class"]
        assert len(class_chunks) == 1
        assert class_chunks[0].metadata.class_name == "MyClass"

    def test_chunk_js_function(self):
        code = '''function funcOne() {
    return 1;
}

function funcTwo() {
    return 2;
}
'''
        chunker = Chunker()
        chunks = chunker.chunk_code(code, "test.js", "javascript")

        func_chunks = [c for c in chunks if c.metadata.chunk_type == "code_function"]
        assert len(func_chunks) == 2
        func_names = [c.metadata.function_name for c in func_chunks]
        assert "funcOne" in func_names
        assert "funcTwo" in func_names

    def test_chunk_js_arrow_function_not_supported(self):
        code = '''const myFunc = () => {
    return 1;
};
'''
        chunker = Chunker()
        chunks = chunker.chunk_code(code, "test.js", "javascript")

        func_chunks = [c for c in chunks if c.metadata.chunk_type == "code_function"]
        assert len(func_chunks) == 0


class TestChunkerMarkdown:
    def test_chunk_markdown_basic(self):
        content = '''# Heading 1

This is a paragraph.

## Heading 2

Another paragraph here.
'''
        chunker = Chunker()
        chunks = chunker.chunk_markdown(content, "test.md")

        assert len(chunks) == 4
        para_chunks = [c for c in chunks if "paragraph" in c.text.lower()]
        assert len(para_chunks) == 2

    def test_chunk_markdown_heading_as_own_chunk(self):
        content = '''# Title

First paragraph content.

## Subtitle

Second paragraph content.
'''
        chunker = Chunker()
        chunks = chunker.chunk_markdown(content, "test.md")

        assert len(chunks) == 4

    def test_chunk_markdown_empty_lines(self):
        content = '''# Header


First paragraph.


Second paragraph.
'''
        chunker = Chunker()
        chunks = chunker.chunk_markdown(content, "test.md")

        para_chunks = [c for c in chunks if "paragraph" in c.text.lower()]
        assert len(para_chunks) == 2


class TestChunkerFile:
    def test_chunk_file_python(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('''def hello():
    print("hello")

class MyClass:
    def world(self):
        print("world")
''')
            temp_path = f.name

        try:
            chunker = Chunker()
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) > 0
            assert all(c.metadata.language == "python" for c in chunks)
        finally:
            os.unlink(temp_path)

    def test_chunk_file_javascript(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write('''function hello() {
    console.log("hello");
}

class MyClass {
    world() {
        console.log("world");
    }
}
''')
            temp_path = f.name

        try:
            chunker = Chunker()
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) > 0
            assert all(c.metadata.language == "javascript" for c in chunks)
        finally:
            os.unlink(temp_path)

    def test_chunk_file_markdown(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write('''# Title

This is a paragraph.
''')
            temp_path = f.name

        try:
            chunker = Chunker()
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) > 0
        finally:
            os.unlink(temp_path)

    def test_chunk_file_unsupported(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some text")
            temp_path = f.name

        try:
            chunker = Chunker()
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) == 0
        finally:
            os.unlink(temp_path)


class TestChunkDataStructure:
    def test_chunk_has_id(self):
        chunker = Chunker()
        chunks = chunker.chunk_code("def f(): pass", "test.py", "python")
        assert len(chunks) == 1
        assert chunks[0].id is not None
        assert len(chunks[0].id) > 0

    def test_chunk_has_text(self):
        chunker = Chunker()
        code = "def hello(): pass"
        chunks = chunker.chunk_code(code, "test.py", "python")
        assert len(chunks) == 1
        assert "hello" in chunks[0].text

    def test_chunk_metadata_source(self):
        chunker = Chunker()
        chunks = chunker.chunk_code("def f(): pass", "my_source.py", "python")
        assert chunks[0].metadata.source == "my_source.py"

    def test_chunk_metadata_line_numbers(self):
        code = '''def first():
    pass

def second():
    pass
'''
        chunker = Chunker()
        chunks = chunker.chunk_code(code, "test.py", "python")

        func_chunks = [c for c in chunks if c.metadata.function_name == "first"]
        assert len(func_chunks) == 1
        assert func_chunks[0].metadata.line_start == 1
        assert func_chunks[0].metadata.line_end == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
