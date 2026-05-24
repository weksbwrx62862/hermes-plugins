"""ActionMemoryService 和 omni_record_action 工具测试。"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from omnimem.core.action_memory import ActionRecord, ActionMemoryService
from omnimem.handlers.record_action import handle_record_action


class TestActionRecord(unittest.TestCase):

    def test_basic_record(self) -> None:
        rec = ActionRecord(
            action_type="tool_call",
            tool_name="terminal",
            tool_args_summary="git status",
            outcome="success",
        )
        self.assertEqual(rec.action_type, "tool_call")
        content = rec.to_content()
        self.assertIn("terminal", content)
        self.assertIn("success", content)

    def test_from_args(self) -> None:
        rec = ActionRecord.from_args({
            "action_type": "tool_call",
            "tool_name": "web_search",
            "outcome": "failure",
            "tool_result_summary": "network timeout",
            "lesson_learned": "需要增加重试逻辑",
            "parent_task_id": "task-001",
            "duration_ms": 1500,
        })
        self.assertEqual(rec.outcome, "failure")
        self.assertEqual(rec.duration_ms, 1500)
        self.assertEqual(rec.parent_task_id, "task-001")

    def test_from_args_truncation(self) -> None:
        rec = ActionRecord.from_args({
            "action_type": "decision",
            "tool_args_summary": "A" * 300,
            "tool_result_summary": "B" * 400,
        })
        self.assertLessEqual(len(rec.tool_args_summary), 200)
        self.assertLessEqual(len(rec.tool_result_summary), 300)

    def test_to_dict(self) -> None:
        rec = ActionRecord(
            action_type="spawn",
            agent_role="orchestrator",
            tool_name="delegate_task",
            outcome="partial",
        )
        d = rec.to_dict()
        self.assertEqual(d["agent_role"], "orchestrator")
        self.assertEqual(d["tool_name"], "delegate_task")

    def test_from_memory_entry(self) -> None:
        entry = {
            "memory_id": "act-001",
            "type": "action",
            "stored_at": "2026-01-01T00:00:00Z",
            "metadata": json.dumps({
                "action_type": "tool_call",
                "tool_name": "browser_navigate",
                "outcome": "success",
                "agent_role": "leaf",
            }),
        }
        rec = ActionRecord.from_memory_entry(entry)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.tool_name, "browser_navigate")

    def test_from_memory_entry_invalid(self) -> None:
        rec = ActionRecord.from_memory_entry({"memory_id": "bad"})
        self.assertIsNone(rec)


class TestActionMemoryService(unittest.TestCase):

    def setUp(self) -> None:
        self.store = MagicMock()
        self.store.add = MagicMock(return_value="act-test-001")
        self.store.flush = MagicMock()
        self.store.get = MagicMock()
        self.index = MagicMock()
        self.index.add = MagicMock()
        self.retriever = MagicMock()
        self.wing_room = MagicMock()
        self.wing_room.resolve_wing.return_value = "personal"
        self.provenance = MagicMock()
        self.forgetting = MagicMock()

        self.svc = ActionMemoryService(
            self.store, self.index, self.retriever,
            self.wing_room, self.provenance, self.forgetting,
        )

    def test_record_action(self) -> None:
        rec = ActionRecord(
            action_type="tool_call",
            tool_name="terminal",
            outcome="success",
            duration_ms=500,
        )
        mid = self.svc.record_action(rec)
        self.assertEqual(mid, "act-test-001")
        self.store.add.assert_called_once()
        self.index.add.assert_called_once()
        self.forgetting.record_access.assert_called_once_with("act-test-001")

    def test_query_actions(self) -> None:
        self.retriever.search.return_value = [
            {"memory_id": "act-1"},
            {"memory_id": "act-2"},
        ]
        self.store.get.side_effect = [
            {
                "memory_id": "act-1",
                "type": "action",
                "stored_at": "2026-01-01T00:00:00Z",
                "metadata": json.dumps({
                    "action_type": "tool_call",
                    "tool_name": "web_search",
                    "outcome": "success",
                    "parent_task_id": "task-001",
                }),
            },
            {
                "memory_id": "act-2",
                "type": "action",
                "stored_at": "2026-01-01T00:01:00Z",
                "metadata": json.dumps({
                    "action_type": "tool_call",
                    "tool_name": "patch",
                    "outcome": "failure",
                    "parent_task_id": "task-001",
                }),
            },
        ]
        results = self.svc.query_actions(parent_task_id="task-001")
        self.assertEqual(len(results), 2)

    def test_query_filter_by_outcome(self) -> None:
        self.retriever.search.return_value = [
            {"memory_id": "act-1"},
            {"memory_id": "act-2"},
        ]
        self.store.get.side_effect = [
            {
                "memory_id": "act-1", "type": "action",
                "stored_at": "", "metadata": json.dumps({
                    "outcome": "success", "tool_name": "a",
                }),
            },
            {
                "memory_id": "act-2", "type": "action",
                "stored_at": "", "metadata": json.dumps({
                    "outcome": "failure", "tool_name": "b",
                }),
            },
        ]
        failures = self.svc.query_actions(outcome="failure")
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].tool_name, "b")

    def test_get_task_chain(self) -> None:
        self.retriever.search.return_value = [
            {"memory_id": "act-3"},
            {"memory_id": "act-1"},
        ]
        self.store.get.side_effect = [
            {
                "memory_id": "act-3", "type": "action",
                "stored_at": "", "metadata": json.dumps({
                    "parent_task_id": "t", "turn_index": 2,
                    "tool_name": "step2", "outcome": "success",
                }),
            },
            {
                "memory_id": "act-1", "type": "action",
                "stored_at": "", "metadata": json.dumps({
                    "parent_task_id": "t", "turn_index": 1,
                    "tool_name": "step1", "outcome": "success",
                }),
            },
        ]
        chain = self.svc.get_task_chain("t")
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0].tool_name, "step1")  # sorted by turn_index

    def test_auto_extract_lesson_timeout(self) -> None:
        rec = ActionRecord(
            tool_name="terminal",
            tool_result_summary="Request timed out after 30s",
        )
        lesson = self.svc._auto_extract_lesson(rec)
        self.assertIn("超时", lesson)

    def test_auto_extract_lesson_permission(self) -> None:
        rec = ActionRecord(
            tool_name="write_file",
            tool_result_summary="Permission denied: /root/secret",
        )
        lesson = self.svc._auto_extract_lesson(rec)
        self.assertIn("权限", lesson)

    def test_auto_extract_lesson_not_found(self) -> None:
        rec = ActionRecord(
            tool_name="read_file",
            tool_result_summary="File not found: /tmp/missing",
        )
        lesson = self.svc._auto_extract_lesson(rec)
        self.assertIn("不存在", lesson)

    def test_learn_from_failures(self) -> None:
        self.retriever.search.return_value = [
            {"memory_id": "fail-1"},
        ]
        self.store.get.side_effect = [
            {
                "memory_id": "fail-1", "type": "action",
                "stored_at": "", "metadata": json.dumps({
                    "outcome": "failure",
                    "tool_name": "browser_navigate",
                    "tool_result_summary": "Connection timed out",
                    "parent_task_id": "task-x",
                }),
            },
        ]
        lessons = self.svc.learn_from_failures()
        self.assertEqual(len(lessons), 1)
        self.assertIn("超时", lessons[0]["lesson"])


class TestHandleRecordAction(unittest.TestCase):

    def setUp(self) -> None:
        self.provider = MagicMock()
        self.provider._action_memory = MagicMock()
        self.provider._action_memory.record_action.return_value = "mem-001"

    def test_basic_record(self) -> None:
        result = json.loads(handle_record_action(self.provider, {
            "action_type": "tool_call",
            "tool_name": "search_files",
            "outcome": "success",
        }))
        self.assertEqual(result["status"], "stored")
        self.assertEqual(result["memory_id"], "mem-001")

    def test_unavailable(self) -> None:
        provider = MagicMock()
        provider._action_memory = None
        result = json.loads(handle_record_action(provider, {
            "action_type": "tool_call",
        }))
        self.assertEqual(result["status"], "unavailable")
