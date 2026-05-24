"""dev-lifecycle 插件 — handle_dev_workflow 和 handle_on_session_start 测试。"""

import json
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/home/xxh/.hermes/plugins/dev-lifecycle")

from handlers import (
    _skill_path_cache,
    handle_dev_workflow,
    handle_on_session_start,
)
from state import WorkflowState


class TestHandleOverview(unittest.TestCase):

    def setUp(self):
        _skill_path_cache.clear()

    def test_handle_overview(self):
        result = handle_dev_workflow({"action": "overview"})
        data = json.loads(result)
        self.assertIn("lifecycle", data)
        self.assertIn("stages", data)
        self.assertIn("aux_skills", data)
        self.assertEqual(len(data["stages"]), 3)


class TestHandleStage(unittest.TestCase):

    def setUp(self):
        _skill_path_cache.clear()

    def test_handle_stage_valid(self):
        result = handle_dev_workflow({"action": "stage", "stage_name": "ideate"})
        data = json.loads(result)
        self.assertEqual(data["stage"], "ideate")
        self.assertIn("flow", data)
        self.assertIsInstance(data["flow"], list)

    def test_handle_stage_invalid(self):
        result = handle_dev_workflow({"action": "stage", "stage_name": "nonexistent"})
        data = json.loads(result)
        self.assertIn("error", data)


class TestHandleSkill(unittest.TestCase):

    def setUp(self):
        _skill_path_cache.clear()

    def test_handle_skill_valid(self):
        with patch("handlers._skill_path", return_value=None):
            result = handle_dev_workflow({"action": "skill", "skill_name": "grill-me"})
        data = json.loads(result)
        self.assertEqual(data["skill"], "grill-me")
        self.assertIn("exists", data)

    def test_handle_skill_not_found(self):
        result = handle_dev_workflow({"action": "skill", "skill_name": "nonexistent-skill"})
        data = json.loads(result)
        self.assertFalse(data["exists"])


class TestHandleUnknownAction(unittest.TestCase):

    def setUp(self):
        _skill_path_cache.clear()

    def test_handle_unknown_action(self):
        result = handle_dev_workflow({"action": "invalid"})
        data = json.loads(result)
        self.assertIn("error", data)


class TestHandleDevWorkflowException(unittest.TestCase):

    def setUp(self):
        _skill_path_cache.clear()

    def test_handle_dev_workflow_exception(self):
        with patch("handlers._handle_overview", side_effect=RuntimeError("boom")):
            result = handle_dev_workflow({"action": "overview"})
        data = json.loads(result)
        self.assertEqual(data["error"], "dev_workflow 内部错误")
        self.assertIn("detail", data)


class TestOnSessionStart(unittest.TestCase):

    def test_on_session_start_injects_context(self):
        mock_ctx = MagicMock()
        with patch("handlers._plugin_ctx", mock_ctx), \
             patch("handlers._workflow_mgr", None):
            handle_on_session_start()
        mock_ctx.inject_context.assert_called_once()
        call_arg = mock_ctx.inject_context.call_args[0][0]
        self.assertIn("dev_workflow", call_arg)

    def test_on_session_start_no_ctx(self):
        with patch("handlers._plugin_ctx", None):
            try:
                handle_on_session_start()
            except Exception:
                self.fail("handle_on_session_start 在 _plugin_ctx 为 None 时不应抛异常")


class TestHandleStart(unittest.TestCase):

    def setUp(self):
        _skill_path_cache.clear()

    def test_handle_start_no_path(self):
        result = handle_dev_workflow({"action": "start"})
        data = json.loads(result)
        self.assertIn("error", data)

    def test_handle_start_with_path(self):
        with patch("handlers._workflow_mgr") as mock_mgr, \
             patch("handlers._project_detector", None), \
             patch("handlers._skill_recommender", None), \
             patch("handlers.init_modules"), \
             patch("config.load_config") as mock_lc:
            mock_lc.return_value.stages = {
                "ideate": {"flow": [("grill-me", "深挖")]},
                "build": {"flow": [("prototype", "原型")]},
                "deliver": {"flow": [("triage", "分类")]},
            }
            mock_state = WorkflowState(
                project_path="/tmp/test",
                current_stage="ideate",
                skills_status={"grill-me": "pending"},
            )
            mock_mgr.start.return_value = mock_state
            mock_mgr.list_active.return_value = [mock_state]
            result = handle_dev_workflow({"action": "start", "project_path": "/tmp/test"})
            data = json.loads(result)
            self.assertEqual(data["status"], "started")


class TestHandleAdvance(unittest.TestCase):

    def setUp(self):
        _skill_path_cache.clear()

    def test_handle_advance_no_skill(self):
        result = handle_dev_workflow({"action": "advance"})
        data = json.loads(result)
        self.assertIn("error", data)

    def test_handle_advance_no_workflow(self):
        with patch("handlers._workflow_mgr", None):
            result = handle_dev_workflow({"action": "advance", "skill_name": "grill-me"})
            data = json.loads(result)
            self.assertIn("error", data)


class TestHandleRollback(unittest.TestCase):

    def setUp(self):
        _skill_path_cache.clear()

    def test_handle_rollback_no_stage(self):
        result = handle_dev_workflow({"action": "rollback"})
        data = json.loads(result)
        self.assertIn("error", data)

    def test_handle_rollback_invalid_stage(self):
        result = handle_dev_workflow({"action": "rollback", "to_stage": "invalid"})
        data = json.loads(result)
        self.assertIn("error", data)


class TestHandleResume(unittest.TestCase):

    def setUp(self):
        _skill_path_cache.clear()

    def test_handle_resume_no_path(self):
        result = handle_dev_workflow({"action": "resume"})
        data = json.loads(result)
        self.assertIn("error", data)


class TestHandleReport(unittest.TestCase):

    def setUp(self):
        _skill_path_cache.clear()

    def test_handle_report(self):
        with patch("handlers._telemetry") as mock_tel, \
             patch("handlers.init_modules"):
            mock_tel.report.return_value = {
                "skill_usage": {},
                "stage_durations": {},
                "skip_rate": {},
                "total_events": 0,
                "top_skills": [],
            }
            result = handle_dev_workflow({"action": "report"})
            data = json.loads(result)
            self.assertIn("total_events", data)


if __name__ == "__main__":
    unittest.main()
