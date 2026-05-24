import pytest

from omnimem.compression.micro import microcompact
from omnimem.compression.line_compress import structured_line_compress
from omnimem.compression.pipeline import CompressionPipeline


class TestMicrocompact:
    def test_removes_duplicate_lines(self):
        lines = ["hello", "hello", "world"]
        result = microcompact(lines)
        assert len(result) == 2
        assert result[0] == "hello"
        assert result[1] == "world"

    def test_removes_noise(self):
        lines = ["---", "===", "valid content"]
        result = microcompact(lines)
        assert len(result) == 1
        assert "valid content" in result[0]

    def test_preserves_key_markers(self):
        lines = ["DECISION: use Python", "normal line"]
        result = microcompact(lines)
        assert any("DECISION" in l for l in result)

    def test_preserves_empty_line_structure(self):
        lines = ["line1", "", "", "line2"]
        result = microcompact(lines)
        assert "" in result

    def test_empty_input(self):
        result = microcompact([])
        assert result == []

    def test_short_lines_filtered(self):
        lines = ["ab", "this is long enough"]
        result = microcompact(lines)
        assert len(result) == 1


class TestStructuredLineCompress:
    def test_merges_similar_lines(self):
        lines = [
            "\u7528\u6237\u559c\u6b22Python",
            "\u7528\u6237\u504f\u597dPython",
            "\u7528\u6237\u7231\u597d\u7f16\u7a0b",
        ]
        result = structured_line_compress(lines)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_empty_lines_preserved(self):
        lines = ["line1", "", "line2"]
        result = structured_line_compress(lines)
        assert any("line1" in l for l in result)
        assert any("line2" in l for l in result)

    def test_empty_input(self):
        result = structured_line_compress([])
        assert result == []

    def test_long_line_truncated(self):
        long_line = "a" * 300
        lines = [long_line]
        result = structured_line_compress(lines)
        assert len(result[0]) <= 200

    def test_redundant_phrases_removed(self):
        lines = ["I think this is important", "basically it works"]
        result = structured_line_compress(lines)
        combined = " ".join(result)
        assert "I think " not in combined
        assert "basically " not in combined


class TestCompressionPipeline:
    def test_pipeline_runs(self):
        pipeline = CompressionPipeline()
        result = pipeline.compress("\u6d4b\u8bd5\u6587\u672c " * 100)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_pipeline_without_llm(self):
        pipeline = CompressionPipeline(llm_call_fn=None)
        result = pipeline.compress("some content to compress")
        assert isinstance(result, str)

    def test_pipeline_short_content(self):
        pipeline = CompressionPipeline()
        result = pipeline.compress("short")
        assert isinstance(result, str)
