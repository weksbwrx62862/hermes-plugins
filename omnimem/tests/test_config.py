"""配置管理模块测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omnimem.config import OmniMemConfig


class TestOmniMemConfig(unittest.TestCase):
    """OmniMemConfig 测试。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def test_defaults(self) -> None:
        config = OmniMemConfig(Path(self.tmpdir))
        self.assertEqual(config.get("save_interval"), 15)
        self.assertEqual(config.get("retrieval_mode"), "rag")
        self.assertEqual(config.get("budget_tokens"), 4000)

    def test_set_and_get(self) -> None:
        config = OmniMemConfig(Path(self.tmpdir))
        config.set("custom_key", "custom_value")
        self.assertEqual(config.get("custom_key"), "custom_value")

    def test_save_and_reload(self) -> None:
        config = OmniMemConfig(Path(self.tmpdir))
        config.save({"save_interval": 30})
        config2 = OmniMemConfig(Path(self.tmpdir))
        self.assertEqual(config2.get("save_interval"), 30)

    def test_get_with_default(self) -> None:
        config = OmniMemConfig(Path(self.tmpdir))
        self.assertEqual(config.get("nonexistent", "default_val"), "default_val")

    def test_values(self) -> None:
        config = OmniMemConfig(Path(self.tmpdir))
        vals = config.values
        self.assertIsInstance(vals, dict)
        self.assertIn("save_interval", vals)
