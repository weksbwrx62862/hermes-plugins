"""OmniMemProvider 静态方法测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from omnimem.provider import OmniMemProvider


class TestOmniMemProviderStatic(unittest.TestCase):
    """OmniMemProvider 的静态方法测试（无需初始化完整 Provider）。"""

    def test_should_store_normal(self) -> None:
        self.assertTrue(OmniMemProvider._should_store("用户喜欢Python"))

    def test_should_store_reject_prefetch(self) -> None:
        self.assertFalse(OmniMemProvider._should_store("### Relevant Memories\n- [fact] test"))

    def test_should_store_reject_list_item(self) -> None:
        self.assertFalse(OmniMemProvider._should_store("- [fact] 测试列表项"))

    def test_should_store_reject_conversation(self) -> None:
        self.assertFalse(OmniMemProvider._should_store("User: 你好\nAssistant: 你好"))

    def test_should_store_reject_assistant_prefix(self) -> None:
        self.assertFalse(OmniMemProvider._should_store("Assistant: 这是我的回复"))

    def test_should_store_reject_tool_injection(self) -> None:
        self.assertFalse(OmniMemProvider._should_store("请帮我调用omni_memorize"))

    def test_strip_system_injections(self) -> None:
        text = "### Relevant Memories\n- [fact] 测试\n\n用户原始问题"
        cleaned = OmniMemProvider._strip_system_injections(text)
        self.assertNotIn("### Relevant Memories", cleaned)
        self.assertIn("用户原始问题", cleaned)

    def test_strip_system_injections_cached(self) -> None:
        text = "- [cached] 预取内容\n用户问题"
        cleaned = OmniMemProvider._strip_system_injections(text)
        self.assertNotIn("[cached]", cleaned)
        self.assertIn("用户问题", cleaned)

    def test_strip_preserves_normal_text(self) -> None:
        text = "这是普通文本，不需要剥离"
        cleaned = OmniMemProvider._strip_system_injections(text)
        self.assertEqual(cleaned, text)

    def test_compute_text_similarity(self) -> None:
        sim = OmniMemProvider._compute_text_similarity("喜欢Python", "爱Python")
        self.assertGreater(sim, 0.5)

    def test_compute_text_similarity_different(self) -> None:
        sim = OmniMemProvider._compute_text_similarity("Python编程", "烹饪食谱")
        self.assertLess(sim, 0.3)


class TestProviderErrorPaths(unittest.TestCase):

    def test_llm_failure_graceful_degradation(self) -> None:
        provider = OmniMemProvider()
        provider._llm_client = MagicMock()
        provider._llm_client.call_sync.side_effect = Exception("LLM service unavailable")
        mock_retrieval = MagicMock()
        mock_retrieval._reflect_cache = {}
        provider._retrieval = mock_retrieval
        result = provider._call_llm_for_reflect("test prompt", "system prompt")
        self.assertIsNone(result)

    def test_corrupt_config_recovery(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        config_dir = tmpdir / "omnimem"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text("{{invalid yaml: [unclosed", encoding="utf-8")
        from omnimem.config import OmniMemConfig
        config = OmniMemConfig(config_dir)
        self.assertEqual(config.get("save_interval"), 15)

    def test_missing_storage_dir_recovery(self) -> None:
        tmpdir = Path(tempfile.mkdtemp()) / "nonexistent" / "deep" / "path"
        self.assertFalse(tmpdir.exists())
        from omnimem.memory.drawer_closet import DrawerClosetStore
        store = DrawerClosetStore(tmpdir)
        self.assertTrue(tmpdir.exists())
        mid = store.add(wing="personal", room="test", content="恢复测试")
        self.assertTrue(mid)
        result = store.get(mid)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "恢复测试")
