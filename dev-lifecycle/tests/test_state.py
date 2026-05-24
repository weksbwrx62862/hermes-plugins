"""dev-lifecycle 插件 — state.py 工作流状态机测试。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "/home/xxh/.hermes/plugins/dev-lifecycle")

from state import WorkflowManager, WorkflowState


class TestWorkflowState(unittest.TestCase):

    def test_workflow_state_defaults(self):
        state = WorkflowState(project_path="/tmp/test", current_stage="ideate")
        self.assertEqual(state.project_path, "/tmp/test")
        self.assertEqual(state.current_stage, "ideate")
        self.assertIsInstance(state.skills_status, dict)
        self.assertGreater(state.created_at, 0)


class TestWorkflowManager(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "workflow.db"
        self.patcher = patch("state.DB_PATH", self.db_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_start_creates_workflow(self):
        mgr = WorkflowManager()
        config = {"stages": {"ideate": {"skills": ["grill-me", "to-prd"]}}}
        state = mgr.start("/tmp/test", config)
        self.assertEqual(state.current_stage, "ideate")
        self.assertEqual(state.skills_status["grill-me"], "pending")

    def test_advance_completes_skill(self):
        mgr = WorkflowManager()
        config = {"stages": {"ideate": {"skills": ["grill-me"]}, "build": {"skills": ["prototype"]}}}
        mgr.start("/tmp/test_adv", config)
        result = mgr.advance(1, "grill-me")
        self.assertEqual(result["current_stage"], "build")

    def test_resume_returns_none_when_empty(self):
        mgr = WorkflowManager()
        result = mgr.resume("/nonexistent")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
