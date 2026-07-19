import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from plugins.skill_pool import (
    SkillEntry,
    SkillPool,
    register,
    _tool_search_skills,
    _tool_set_core,
    _tool_set_pool,
)


class TestSkillPool(unittest.TestCase):
    """SkillPool 插件单元测试"""

    def test_quick_embed(self):
        """测试 _quick_embed 输出维度为 384 且 L2 归一化。"""
        vec = SkillPool._quick_embed("test")
        self.assertEqual(vec.shape, (384,))
        self.assertAlmostEqual(
            float(np.linalg.norm(vec)), 1.0, delta=1e-6,
        )

    def test_search_empty_index(self):
        """测试空索引时 search 返回空列表。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with patch.object(SkillPool, "INDEX_FILE", tmp_path / ".pool_index.json"):
                pool = SkillPool()
                self.assertEqual(pool.search("anything"), [])

    def test_set_core_and_set_pool(self):
        """测试 set_core / set_pool 正确更新 tier。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with patch.object(SkillPool, "CONFIG_FILE", tmp_path / ".pool_config.json"):
                pool = SkillPool()
                entry = SkillEntry(
                    name="test-skill",
                    description="test",
                    category="test",
                    skill_path="/tmp",
                )
                pool._entries.append(entry)

                pool.set_core("test-skill")
                self.assertEqual(pool.get_tier("test-skill"), "core")
                self.assertEqual(entry.tier, "core")

                pool.set_pool("test-skill")
                self.assertEqual(pool.get_tier("test-skill"), "pool")
                self.assertEqual(entry.tier, "pool")

    def test_register_none_does_not_raise(self):
        """测试 register(None) 不抛异常。"""
        try:
            register(None)
        except Exception as exc:
            self.fail(f"register(None) raised {exc!r}")

    def test_incremental_delete_consistency(self):
        """测试增量更新删除场景下 _entries 与 _embeddings 保持一致。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            skill_a_dir = tmp_path / "skill-a"
            skill_b_dir = tmp_path / "skill-b"
            skill_a_dir.mkdir()
            skill_b_dir.mkdir()

            (skill_a_dir / "SKILL.md").write_text(
                "---\nname: skill-a\ndescription: first test skill\n---\n",
                encoding="utf-8",
            )
            (skill_b_dir / "SKILL.md").write_text(
                "---\nname: skill-b\ndescription: second test skill\n---\n",
                encoding="utf-8",
            )

            with patch.object(SkillPool, "SKILLS_DIR", tmp_path), \
                 patch.object(SkillPool, "INDEX_FILE", tmp_path / ".pool_index.json"), \
                 patch.object(SkillPool, "MTIME_FILE", tmp_path / ".pool_mtimes.json"), \
                 patch.object(SkillPool, "CONFIG_FILE", tmp_path / ".pool_config.json"):
                pool = SkillPool()
                # 避免下载 sentence-transformers 模型
                with patch.object(pool, "_ensure_model_downloaded", return_value=None):
                    pool.build_index(incremental=False, embed_fn=SkillPool._quick_embed)

                self.assertEqual(len(pool._entries), 2)
                self.assertEqual(pool._embeddings.shape[0], 2)

                # 删除 skill-b
                (skill_b_dir / "SKILL.md").unlink()

                with patch.object(pool, "_ensure_model_downloaded", return_value=None):
                    pool.build_index(incremental=True, embed_fn=SkillPool._quick_embed)

                self.assertEqual(len(pool._entries), 1)
                self.assertEqual(pool._embeddings.shape[0], 1)

                results = pool.search("first test skill")
                result_names = {e.name for e in results}
                self.assertNotIn("skill-b", result_names)

    def test_tool_handlers_validation(self):
        """测试工具 handler 缺少必填参数时返回错误 JSON。"""
        search_result = json.loads(_tool_search_skills({}))
        self.assertIn("error", search_result)
        self.assertIn("query", search_result["error"])

        core_result = json.loads(_tool_set_core({}))
        self.assertIn("error", core_result)
        self.assertIn("name", core_result["error"])

        pool_result = json.loads(_tool_set_pool({}))
        self.assertIn("error", pool_result)
        self.assertIn("name", pool_result["error"])


if __name__ == "__main__":
    unittest.main()
