"""dev-lifecycle 插件 — 辅助函数测试（_strip_frontmatter, _skill_path, _read_summary, warmup_skill_cache）。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/home/xxh/.hermes/plugins/dev-lifecycle")

from handlers import (
    _read_summary,
    _skill_path,
    _skill_path_cache,
    _strip_frontmatter,
    warmup_skill_cache,
)


class TestStripFrontmatter(unittest.TestCase):

    def test_strip_frontmatter_standard(self):
        content = "---\ntitle: 测试\n---\n正文内容"
        result = _strip_frontmatter(content)
        self.assertEqual(result, "正文内容")

    def test_strip_frontmatter_none(self):
        content = "没有 frontmatter 的纯文本"
        result = _strip_frontmatter(content)
        self.assertEqual(result, "没有 frontmatter 的纯文本")

    def test_strip_frontmatter_body_dashes(self):
        content = "---\ntitle: 测试\n---\n正文中有 --- 分隔线\n更多内容"
        result = _strip_frontmatter(content)
        self.assertIn("--- 分隔线", result)
        self.assertIn("更多内容", result)


class TestSkillPath(unittest.TestCase):

    def setUp(self):
        _skill_path_cache.clear()

    def test_skill_path_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "my-skill"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text("# 测试")
            mtime = skill_file.stat().st_mtime
            _skill_path_cache["my-skill"] = (mtime, skill_file)
            result = _skill_path("my-skill")
            self.assertEqual(result, skill_file)

    def test_skill_path_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "test-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# 测试技能")
            with patch("handlers.SKILLS_DIR", Path(tmpdir)):
                _skill_path_cache.clear()
                result = _skill_path("test-skill")
            self.assertIsNotNone(result)
            self.assertTrue(str(result).endswith("SKILL.md"))

    def test_skill_path_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("handlers.SKILLS_DIR", Path(tmpdir)):
                _skill_path_cache.clear()
                result = _skill_path("nonexistent-skill")
            self.assertIsNone(result)


class TestReadSummary(unittest.TestCase):

    def setUp(self):
        _skill_path_cache.clear()

    def test_read_summary_with_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "summary-skill"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(
                "---\ntitle: 摘要测试\n---\n这是摘要第一行\n第二行内容\n# 标题应被过滤\n第三行内容"
            )
            with patch("handlers.SKILLS_DIR", Path(tmpdir)):
                _skill_path_cache.clear()
                result = _read_summary("summary-skill")
            self.assertIn("摘要第一行", result)
            self.assertIn("第二行内容", result)
            self.assertNotIn("# 标题应被过滤", result)

    def test_read_summary_io_error(self):
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_mtime = 0.0
        mock_path.read_text.side_effect = IOError("磁盘错误")
        _skill_path_cache["io-skill"] = (0.0, mock_path)
        result = _read_summary("io-skill")
        self.assertIn("读取技能文件失败", result)


class TestWarmupSkillCache(unittest.TestCase):

    def setUp(self):
        _skill_path_cache.clear()

    def test_warmup_skill_cache(self):
        with patch("handlers._skill_path") as mock_sp:
            mock_sp.side_effect = lambda name: _skill_path_cache.setdefault(name, (0.0, None)) or None
            warmup_skill_cache()
            self.assertTrue(mock_sp.call_count > 0)
            self.assertTrue(len(_skill_path_cache) > 0)


if __name__ == "__main__":
    unittest.main()
