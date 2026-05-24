"""OpenViking 优化 — 单元测试。

测试目录递归检索、三层渐进式披露、可视化检索轨迹、目录导航 API。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestCatalogRetriever:
    """测试目录递归检索通道（内化 OpenViking find()）。"""

    @pytest.fixture
    def mock_index(self):
        """模拟 ThreeLevelIndex。"""
        class MockIndex:
            def search_l0(self, wing="", hall=""):
                if hall == "preferences":
                    return ["偏好", "饮食"]
                if wing == "personal":
                    return ["docker", "python", "偏好", "饮食"]
                return []
            
            def search_by_directory(self, wing="", hall="", room=""):
                if hall == "preferences":
                    return [
                        {"memory_id": "m1", "wing": "personal", "hall": "preferences", "room": "偏好"},
                        {"memory_id": "m2", "wing": "personal", "hall": "preferences", "room": "饮食"},
                    ]
                return []
        
        return MockIndex()

    @pytest.fixture
    def mock_wing_room(self):
        """模拟 WingRoomManager。"""
        class MockWingRoom:
            def list_wings(self):
                return ["personal", "team"]
            
            def list_halls(self, wing):
                return ["facts", "preferences"]
            
            def list_rooms(self, wing, hall):
                return ["docker", "python", "称呼"]
        
        return MockWingRoom()

    @pytest.fixture
    def mock_vector(self):
        """模拟 VectorRetriever。"""
        class MockVector:
            def search(self, query, top_k=10):
                return [
                    {"memory_id": "m1", "score": 0.9, "content": "用户喜欢被称呼为老板"},
                    {"memory_id": "m2", "score": 0.8, "content": "用户喜欢吃鸡胸肉"},
                    {"memory_id": "m3", "score": 0.7, "content": "用户使用 Docker"},
                ]
        
        return MockVector()

    @pytest.fixture
    def mock_bm25(self):
        """模拟 BM25Retriever。"""
        class MockBM25:
            def search(self, query, top_k=10):
                return [
                    {"memory_id": "m1", "score": 0.85, "content": "用户喜欢被称呼为老板"},
                ]
        
        return MockBM25()

    def test_infer_hall(self, mock_index, mock_wing_room, mock_vector, mock_bm25):
        """测试从查询中推断 Hall。"""
        from omnimem.retrieval.catalog import CatalogRetriever
        
        catalog = CatalogRetriever(
            index=mock_index,
            wing_room=mock_wing_room,
            vector_retriever=mock_vector,
            bm25_retriever=mock_bm25,
        )
        
        # 测试类型关键词
        assert catalog._infer_hall("用户的偏好是什么") == "preferences"
        assert catalog._infer_hall("记住这个事实") == "facts"
        assert catalog._infer_hall("纠正错误") == "corrections"
        assert catalog._infer_hall("普通查询") == ""

    def test_infer_wing(self, mock_index, mock_wing_room, mock_vector, mock_bm25):
        """测试从查询中推断 Wing。"""
        from omnimem.retrieval.catalog import CatalogRetriever
        
        catalog = CatalogRetriever(
            index=mock_index,
            wing_room=mock_wing_room,
            vector_retriever=mock_vector,
            bm25_retriever=mock_bm25,
        )
        
        # 测试隐私关键词
        assert catalog._infer_wing("我的私人信息") == "personal"
        assert catalog._infer_wing("团队共享的内容") == "team"
        assert catalog._infer_wing("普通查询") == ""

    def test_locate_directories(self, mock_index, mock_wing_room, mock_vector, mock_bm25):
        """测试目录定位。"""
        from omnimem.retrieval.catalog import CatalogRetriever
        
        catalog = CatalogRetriever(
            index=mock_index,
            wing_room=mock_wing_room,
            vector_retriever=mock_vector,
            bm25_retriever=mock_bm25,
        )
        
        # 测试类型+隐私关键词
        dirs = catalog._locate_directories("偏好", "", "")
        assert len(dirs) > 0
        assert dirs[0]["hall"] == "preferences"

    def test_search(self, mock_index, mock_wing_room, mock_vector, mock_bm25):
        """测试完整检索流程。"""
        from omnimem.retrieval.catalog import CatalogRetriever
        
        catalog = CatalogRetriever(
            index=mock_index,
            wing_room=mock_wing_room,
            vector_retriever=mock_vector,
            bm25_retriever=mock_bm25,
        )
        
        # 测试带目录线索的查询
        results = catalog.search("偏好", top_k=5)
        assert len(results) > 0
        # 验证结果带有 catalog 标记
        for r in results:
            assert "_source" in r
            assert "catalog" in r["_source"]

    def test_search_no_directory(self, mock_index, mock_wing_room, mock_vector, mock_bm25):
        """测试无法定位目录时的降级。"""
        from omnimem.retrieval.catalog import CatalogRetriever
        
        catalog = CatalogRetriever(
            index=mock_index,
            wing_room=mock_wing_room,
            vector_retriever=mock_vector,
            bm25_retriever=mock_bm25,
        )
        
        # 测试无目录线索的查询
        results = catalog.search("普通查询", top_k=5)
        assert results == []  # 定位失败返回空


class TestThreeLevelDisclosure:
    """测试三层渐进式披露（内化 OpenViking overview()）。"""

    def test_refine_overview_short_content(self):
        """测试短内容直接返回。"""
        from omnimem.context.manager import ContextManager
        
        content = "用户喜欢简洁的回答"
        overview = ContextManager.refine_overview(content, max_chars=200)
        assert overview == content

    def test_refine_overview_with_signal_words(self):
        """测试含信号词的概览提取。"""
        from omnimem.context.manager import ContextManager
        
        content = "用户喜欢简洁的回答。但是不喜欢太长的回复。如果可能的话，尽量用一句话回答。因为用户很忙。"
        overview = ContextManager.refine_overview(content, max_chars=100)
        
        # 应该优先提取含信号词的句子
        assert "但是" in overview or "如果" in overview or "因为" in overview

    def test_refine_overview_truncation(self):
        """测试超长内容截断。"""
        from omnimem.context.manager import ContextManager
        
        content = "这是一段很长的内容。" * 100
        overview = ContextManager.refine_overview(content, max_chars=50)
        
        assert len(overview) <= 50

    def test_context_budget_l1(self):
        """测试 ContextBudget 增加 L1 配置。"""
        from omnimem.context.manager import ContextBudget
        
        budget = ContextBudget()
        assert budget.max_overview_chars == 200

    def test_refined_item_overview(self):
        """测试 RefinedItem 增加 overview 字段。"""
        from omnimem.context.manager import RefinedItem
        
        item = RefinedItem(
            summary="用户偏好",
            memory_id="m1",
            memory_type="preference",
            confidence=0.9,
            overview="用户喜欢简洁的回答，但是不喜欢太长的回复",
        )
        
        assert item.overview == "用户喜欢简洁的回答，但是不喜欢太长的回复"


class TestSearchTrace:
    """测试可视化检索轨迹（内化 OpenViking trajectory）。"""

    def test_trace_step(self):
        """测试轨迹步骤记录。"""
        from omnimem.retrieval.trace import SearchTrace
        
        trace = SearchTrace(query="用户偏好")
        
        # 记录步骤
        trace.add_step("channel_search", channel="vector", output_count=5)
        trace.add_step("channel_search", channel="bm25", output_count=3)
        trace.add_step("rrf_fuse", input_count=8, output_count=6)
        
        assert len(trace.steps) == 3
        assert trace.steps[0].action == "channel_search"
        assert trace.steps[0].channel == "vector"
        assert trace.steps[0].output_count == 5

    def test_trace_to_dict(self):
        """测试轨迹序列化。"""
        from omnimem.retrieval.trace import SearchTrace
        
        trace = SearchTrace(query="用户偏好")
        trace.add_step("channel_search", channel="vector", output_count=5)
        
        result = trace.to_dict()
        
        assert result["query"] == "用户偏好"
        assert result["total_steps"] == 1
        assert result["steps"][0]["action"] == "channel_search"
        assert result["steps"][0]["channel"] == "vector"

    def test_trace_context_manager(self):
        """测试轨迹上下文管理器（自动计时）。"""
        from omnimem.retrieval.trace import SearchTrace
        
        trace = SearchTrace(query="测试")
        
        with trace.step("channel_search", channel="vector") as s:
            s.output_count = 10
        
        assert len(trace.steps) == 1
        assert trace.steps[0].elapsed_ms >= 0
        assert trace.steps[0].output_count == 10


class TestDirectoryNavigation:
    """测试目录导航 API（内化 OpenViking ls/tree/grep）。"""

    @pytest.fixture
    def wing_room(self, omni_tmp_path):
        """创建测试用的 WingRoomManager。"""
        from omnimem.memory.wing_room import WingRoomManager
        
        # 创建测试目录结构
        palace_dir = omni_tmp_path / "palace"
        palace_dir.mkdir()
        
        # personal/facts/docker
        (palace_dir / "personal" / "facts" / "docker" / "drawer").mkdir(parents=True)
        (palace_dir / "personal" / "facts" / "python" / "drawer").mkdir(parents=True)
        
        # personal/preferences/称呼
        (palace_dir / "personal" / "preferences" / "称呼" / "drawer").mkdir(parents=True)
        
        # team/skills/部署
        (palace_dir / "team" / "skills" / "部署" / "drawer").mkdir(parents=True)
        
        return WingRoomManager(palace_dir)

    def test_tree(self, wing_room):
        """测试目录树展示。"""
        tree = wing_room.tree()
        
        assert "personal" in tree
        assert "team" in tree
        assert "facts" in tree["personal"]
        assert "preferences" in tree["personal"]
        assert "docker" in tree["personal"]["facts"]
        assert "python" in tree["personal"]["facts"]

    def test_tree_with_filter(self, wing_room):
        """测试带过滤的目录树。"""
        tree = wing_room.tree(wing="personal")
        
        assert "personal" in tree
        assert "team" not in tree

    def test_grep_rooms(self, wing_room):
        """测试 Room 名称搜索。"""
        results = wing_room.grep_rooms("python")
        
        assert len(results) > 0
        assert results[0]["room"] == "python"
        assert results[0]["wing"] == "personal"
        assert results[0]["hall"] == "facts"

    def test_grep_rooms_case_insensitive(self, wing_room):
        """测试大小写不敏感搜索。"""
        results = wing_room.grep_rooms("Python")
        
        assert len(results) > 0
        assert results[0]["room"] == "python"

    def test_count_memories(self, wing_room):
        """测试记忆数量统计。"""
        # 创建测试记忆文件
        drawer_path = wing_room._palace_dir / "personal" / "facts" / "docker" / "drawer"
        (drawer_path / "m1.md").write_text("test1")
        (drawer_path / "m2.md").write_text("test2")
        
        counts = wing_room.count_memories()
        
        assert "personal/facts/docker" in counts
        assert counts["personal/facts/docker"] == 2


class TestConfigSchema:
    """测试配置项是否正确添加。"""

    def test_openviking_configs(self):
        """测试 OpenViking 相关配置项。"""
        from omnimem.config._config import _CONFIG_SCHEMA
        
        # 目录递归检索
        assert "enable_catalog" in _CONFIG_SCHEMA
        assert _CONFIG_SCHEMA["enable_catalog"]["default"] is True
        
        assert "catalog_weight" in _CONFIG_SCHEMA
        assert _CONFIG_SCHEMA["catalog_weight"]["default"] == 2.0
        
        # 三层渐进式披露
        assert "max_overview_chars" in _CONFIG_SCHEMA
        assert _CONFIG_SCHEMA["max_overview_chars"]["default"] == 200
        
        # 可视化检索轨迹
        assert "enable_trace_by_default" in _CONFIG_SCHEMA
        assert _CONFIG_SCHEMA["enable_trace_by_default"]["default"] is False
