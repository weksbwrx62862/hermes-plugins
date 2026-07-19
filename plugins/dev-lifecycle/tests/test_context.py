"""dev-lifecycle 插件 — context.py 项目上下文感知测试。"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "/home/xxh/.hermes/plugins/dev-lifecycle")

from context import ProjectDetector, ProjectContext, SkillRecommender


class TestProjectDetector(unittest.TestCase):

    def test_nonexistent_path(self):
        det = ProjectDetector()
        ctx = det.detect("/nonexistent/path")
        self.assertEqual(ctx.project_type, "unknown")

    def test_python_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "requirements.txt").write_text("flask")
            det = ProjectDetector()
            ctx = det.detect(tmpdir)
            self.assertEqual(ctx.project_type, "python")
            self.assertIn("python", ctx.languages)

    def test_node_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "package.json").write_text("{}")
            det = ProjectDetector()
            ctx = det.detect(tmpdir)
            self.assertEqual(ctx.project_type, "node")
            self.assertIn("node", ctx.languages)


class TestSkillRecommender(unittest.TestCase):

    def test_python_build_promotes_debugpy(self):
        rec = SkillRecommender()
        ctx = ProjectContext(project_path="/tmp", project_type="python")
        skills = [("prototype", "原型"), ("python-debugpy", "调试"), ("spike", "实验")]
        result = rec.recommend(ctx, "build", skills)
        self.assertEqual(result[0][0], "python-debugpy")

    def test_node_build_promotes_inspector(self):
        rec = SkillRecommender()
        ctx = ProjectContext(project_path="/tmp", project_type="node")
        skills = [("prototype", "原型"), ("node-inspect-debugger", "调试"), ("spike", "实验")]
        result = rec.recommend(ctx, "build", skills)
        self.assertEqual(result[0][0], "node-inspect-debugger")

    def test_unknown_keeps_order(self):
        rec = SkillRecommender()
        ctx = ProjectContext(project_path="/tmp", project_type="unknown")
        skills = [("a", "A"), ("b", "B")]
        result = rec.recommend(ctx, "build", skills)
        self.assertEqual(result[0][0], "a")


if __name__ == "__main__":
    unittest.main()
