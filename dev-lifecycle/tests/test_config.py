"""dev-lifecycle 插件 — config.py 配置管理测试。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "/home/xxh/.hermes/plugins/dev-lifecycle")

from config import load_config, merge_lifecycle, LifecycleConfig


class TestMergeLifecycle(unittest.TestCase):

    def test_custom_overrides_default(self):
        default = {"ideate": {"emoji": "💡", "flow": [("a", "A")]}}
        custom = {"ideate": {"flow": [("b", "B")]}}
        merged = merge_lifecycle(default, custom)
        self.assertEqual(merged["ideate"]["flow"], [("b", "B")])
        self.assertEqual(merged["ideate"]["emoji"], "💡")

    def test_custom_adds_new_stage(self):
        default = {"ideate": {"flow": []}}
        custom = {"deploy": {"flow": [("deploy-skill", "部署")]}}
        merged = merge_lifecycle(default, custom)
        self.assertIn("deploy", merged)
        self.assertIn("ideate", merged)


class TestLoadConfig(unittest.TestCase):

    def test_load_default_when_no_file(self):
        with patch("config.CONFIG_PATH", Path("/nonexistent/lifecycle.yaml")):
            config = load_config()
            self.assertIsInstance(config, LifecycleConfig)
            self.assertIn("ideate", config.stages)

    def test_load_custom_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "lifecycle.yaml"
            yaml_path.write_text("stages:\n  custom:\n    flow:\n      - [custom-skill, 自定义]\n")
            with patch("config.CONFIG_PATH", yaml_path):
                config = load_config()
                self.assertIn("custom", config.stages)


if __name__ == "__main__":
    unittest.main()
