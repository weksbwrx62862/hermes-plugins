"""Task 4: node_id 全链路溯源 — 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestTraceChain:
    """测试 TraceChain：记录派生、下钻、上钻、防循环。"""

    @pytest.fixture
    def chain(self, omni_tmp_path):
        from omnimem.core.trace_chain import TraceChain
        tc = TraceChain(omni_tmp_path)
        yield tc
        tc.close()

    def test_record_and_get_node(self, chain):
        """记录节点后应可查询。"""
        chain.record_derivation(
            parent_node_ids=["conv-s1-turn-1"],
            child_node_id="mem-001",
            child_layer="L1",
        )
        node = chain._get_node("mem-001")
        assert node is not None
        assert node["node_id"] == "mem-001"
        assert node["layer"] == "L1"
        assert "conv-s1-turn-1" in json.loads(node["parent_ids_json"])

    def test_drill_down_basic(self, chain):
        """下钻应返回从高层到低层的完整链路。"""
        # L0 → L1 → L2
        chain.record_derivation(
            parent_node_ids=["conv-1-turn-1"],
            child_node_id="atom-001",
            child_layer="L1",
        )
        chain.record_derivation(
            parent_node_ids=["atom-001"],
            child_node_id="scenario-001",
            child_layer="L2",
        )
        result = chain.drill_down("scenario-001")
        assert len(result) == 3  # scenario → atom → conv
        assert result[0]["node_id"] == "scenario-001"
        assert result[1]["node_id"] == "atom-001"
        assert result[2]["node_id"] == "conv-1-turn-1"

    def test_drill_down_circular_ref(self, chain):
        """循环引用不应无限递归。"""
        # 手动创建循环
        chain._conn.execute(
            "INSERT INTO trace_nodes (node_id, layer, parent_ids_json, child_ids_json, ref_path, metadata_json, created_at) "
            "VALUES ('A', 'L1', '[\"B\"]', '[\"B\"]', '', '{}', 0)"
        )
        chain._conn.execute(
            "INSERT INTO trace_nodes (node_id, layer, parent_ids_json, child_ids_json, ref_path, metadata_json, created_at) "
            "VALUES ('B', 'L1', '[\"A\"]', '[\"A\"]', '', '{}', 0)"
        )
        chain._conn.commit()

        result = chain.drill_down("A", max_depth=5)
        assert len(result) == 2  # A → B, 然后 B→A 已在 visited 中，停止
        ids = [r["node_id"] for r in result]
        assert "A" in ids
        assert "B" in ids

    def test_drill_down_max_depth(self, chain):
        """超过 max_depth 应停止。"""
        # 创建 15 层链
        for i in range(15):
            parent = [f"node-{i-1}"] if i > 0 else ["root"]
            chain.record_derivation(
                parent_node_ids=parent,
                child_node_id=f"node-{i}",
                child_layer=f"L{i % 4}",
            )

        result = chain.drill_down("node-14", max_depth=5)
        # 最多 6 个节点（0 到 5）
        assert len(result) <= 6

    def test_drill_up(self, chain):
        """上钻应返回所有引用该节点的上层摘要。"""
        chain.record_derivation(
            parent_node_ids=["conv-1-turn-1"],
            child_node_id="atom-001",
            child_layer="L1",
        )
        chain.record_derivation(
            parent_node_ids=["atom-001"],
            child_node_id="scenario-001",
            child_layer="L2",
        )
        chain.record_derivation(
            parent_node_ids=["atom-001"],
            child_node_id="scenario-002",
            child_layer="L2",
        )

        result = chain.drill_up("atom-001")
        ids = [r["node_id"] for r in result]
        assert "scenario-001" in ids
        assert "scenario-002" in ids

    def test_recover_full_text(self, chain):
        """应能按 node_id 恢复原文。"""
        # 创建临时文件
        ref_dir = Path(chain._chain_db).parent / "refs"
        ref_dir.mkdir(exist_ok=True)
        ref_file = ref_dir / "test.md"
        ref_file.write_text("Hello, this is the original text.", encoding="utf-8")

        chain.record_derivation(
            parent_node_ids=["conv-1-turn-1"],
            child_node_id="mem-001",
            child_layer="L1",
            ref_path=str(ref_file),
        )
        text = chain.recover_full_text("mem-001")
        assert text == "Hello, this is the original text."

    def test_recover_full_text_missing(self, chain):
        """不存在的 ref_path 应返回 None。"""
        chain.record_derivation(
            parent_node_ids=["conv-1-turn-1"],
            child_node_id="mem-001",
            child_layer="L1",
            ref_path="/nonexistent/file.md",
        )
        text = chain.recover_full_text("mem-001")
        assert text is None

    def test_get_ref_path(self, chain):
        """get_ref_path 应返回正确的文件路径。"""
        chain.record_derivation(
            parent_node_ids=["conv-1-turn-1"],
            child_node_id="mem-001",
            child_layer="L1",
            ref_path="/some/path.md",
        )
        assert chain.get_ref_path("mem-001") == "/some/path.md"
        assert chain.get_ref_path("nonexistent") is None

    def test_node_count(self, chain):
        """节点计数应正确。"""
        assert chain.get_node_count() == 0
        chain.record_derivation(
            parent_node_ids=["conv-1-turn-1"],
            child_node_id="mem-001",
            child_layer="L1",
        )
        assert chain.get_node_count() == 2  # parent + child

    def test_duplicate_record_ignored(self, chain):
        """重复记录应被忽略（INSERT OR IGNORE）。"""
        chain.record_derivation(
            parent_node_ids=["conv-1-turn-1"],
            child_node_id="mem-001",
            child_layer="L1",
        )
        chain.record_derivation(
            parent_node_ids=["conv-1-turn-1"],
            child_node_id="mem-001",
            child_layer="L1",
        )
        assert chain.get_node_count() == 2  # 不应重复
