"""Task 1: Mermaid 符号化压缩 — 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestMermaidCanvas:
    """测试 MermaidCanvas：日志卸载、Mermaid 生成、refs 清理、索引持久化。"""

    @pytest.fixture
    def canvas(self, omni_tmp_path):
        from omnimem.compression.mermaid_canvas import MermaidCanvas
        return MermaidCanvas(omni_tmp_path)

    def test_offload_and_compress(self, canvas):
        """工具日志应生成 Mermaid 图谱 + refs 文件。"""
        logs = [
            {"tool_name": "search_files", "status": "success"},
            {"tool_name": "read_file", "status": "success"},
            {"tool_name": "patch", "status": "error"},
        ]
        mermaid, mapping = canvas.offload_and_compress(logs, "s1")

        assert "graph TD" in mermaid
        assert "T1" in mermaid
        assert "T2" in mermaid
        assert "T3" in mermaid
        assert len(mapping) == 3

    def test_recover_by_node_id(self, canvas):
        """按 node_id 应能恢复原文。"""
        logs = [{"tool_name": "test", "status": "ok", "data": "hello"}]
        _, mapping = canvas.offload_and_compress(logs, "s1")

        text = canvas.recover_by_node_id("T1")
        assert text is not None
        assert "test" in text  # tool_name should be in the JSON

    def test_recover_missing_node(self, canvas):
        """不存在的 node_id 应返回 None。"""
        assert canvas.recover_by_node_id("nonexistent") is None

    def test_stale_refs_cleanup(self, omni_tmp_path):
        """过期 refs 文件应被清理。"""
        from omnimem.compression.mermaid_canvas import MermaidCanvas
        config = {"max_refs_age_days": 1}  # 1天过期
        canvas = MermaidCanvas(omni_tmp_path, config=config)

        # 创建一个旧文件
        old_file = canvas._refs_dir / "old.md"
        old_file.write_text("old data")
        import os
        os.utime(old_file, (0, 0))  # 设置时间为 epoch

        # 创建一个新文件
        new_file = canvas._refs_dir / "new.md"
        new_file.write_text("new data")

        canvas._cleanup_stale_refs()
        assert not old_file.exists()
        assert new_file.exists()

    def test_index_persistence(self, canvas):
        """映射索引应持久化到 _index.json。"""
        logs = [{"tool_name": "test", "status": "ok"}]
        canvas.offload_and_compress(logs, "s1")

        index_path = canvas._refs_dir / "_index.json"
        assert index_path.exists()
        index = json.loads(index_path.read_text())
        assert "T1" in index

    def test_is_tool_log(self, canvas):
        """应识别工具日志内容。"""
        assert canvas.is_tool_log("tool_call: search_files") is True
        assert canvas.is_tool_log("command: ls -la") is True
        assert canvas.is_tool_log("工具调用: 读取文件") is True
        assert canvas.is_tool_log("这是一个普通对话") is False

    def test_is_tool_log_custom_pattern(self, omni_tmp_path):
        """自定义 pattern 应生效。"""
        from omnimem.compression.mermaid_canvas import MermaidCanvas
        config = {"mermaid_tool_log_patterns": [r"MY_TOOL_TAG"]}
        canvas = MermaidCanvas(omni_tmp_path, config=config)
        assert canvas.is_tool_log("MY_TOOL_TAG: something") is True
        assert canvas.is_tool_log("tool_call: normal") is False

    def test_empty_logs(self, canvas):
        """空日志应返回空结果。"""
        mermaid, mapping = canvas.offload_and_compress([], "s1")
        assert mermaid == ""
        assert mapping == {}


class TestCompressionPipelineMermaid:
    """测试 CompressionPipeline 的 Mermaid 分支。"""

    def test_tool_log_goes_mermaid(self, omni_tmp_path):
        """工具日志应走 Mermaid 路径。"""
        from omnimem.compression.pipeline import CompressionPipeline
        from omnimem.compression.mermaid_canvas import MermaidCanvas

        canvas = MermaidCanvas(omni_tmp_path)
        pipeline = CompressionPipeline(
            llm_call_fn=None,
            mermaid_canvas=canvas,
            session_key="test",
        )

        content = "tool_call: search_files\nresult: found 3 files"
        result = pipeline.compress(content)
        assert "graph TD" in result

    def test_normal_content_skips_mermaid(self, omni_tmp_path):
        """普通内容不应走 Mermaid 路径。"""
        from omnimem.compression.pipeline import CompressionPipeline
        from omnimem.compression.mermaid_canvas import MermaidCanvas

        canvas = MermaidCanvas(omni_tmp_path)
        pipeline = CompressionPipeline(
            llm_call_fn=None,
            mermaid_canvas=canvas,
            session_key="test",
        )

        content = "用户喜欢简洁的回答"
        result = pipeline.compress(content)
        # 普通内容走标准压缩管线，不含 Mermaid
        assert "graph TD" not in result

    def test_no_canvas_fallback(self):
        """没有 canvas 时应走标准路径。"""
        from omnimem.compression.pipeline import CompressionPipeline

        pipeline = CompressionPipeline(llm_call_fn=None)
        content = "tool_call: search_files\nresult: found 3 files"
        result = pipeline.compress(content)
        # 不崩溃，走标准路径
        assert isinstance(result, str)
