"""ToolRouter 核心调度器单元测试。

覆盖: ToolRouter 路由分发 / handle_compact / handle_detail (list/get/events)
       build_system_prompt / get_config_schema / apply_sync_change
       retry_index_add / retry_retriever_add / l3_recall / run_prefetch
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from omnimem.core.prompt_builder import build_system_prompt
from omnimem.core.tool_names import (
    MEMORY_COMPAT,
    OMNI_COMPACT,
    OMNI_DETAIL,
    OMNI_GOVERN,
    OMNI_MEMORIZE,
    OMNI_RECALL,
    OMNI_RECORD_ACTION,
    OMNI_REFLECT,
)
from omnimem.core.tool_router import (
    ToolRouter,
    apply_sync_change,
    get_config_schema,
    handle_compact,
    handle_detail,
    l3_recall,
    retry_index_add,
    retry_kg_extract,
    retry_retriever_add,
    run_prefetch,
    run_queue_prefetch,
    save_config,
)
from omnimem.provider import OmniMemProvider

# ──────────────────────────────────────────────
# ToolRouter 路由分发
# ──────────────────────────────────────────────

class TestToolRouter(unittest.TestCase):
    """ToolRouter 路由分发测试。"""

    def setUp(self) -> None:
        self.memorize_fn = MagicMock(return_value=json.dumps({"status": "stored", "memory_id": "m1"}))
        self.recall_fn = MagicMock(return_value=json.dumps({"status": "found", "count": 3}))
        self.govern_fn = MagicMock(return_value=json.dumps({"status": "ok"}))
        self.reflect_fn = MagicMock(return_value=json.dumps({"status": "reflected"}))
        self.compact_fn = MagicMock(return_value=json.dumps({"status": "ready"}))
        self.detail_fn = MagicMock(return_value=json.dumps({"status": "ok", "count": 0}))
        self.compat_fn = MagicMock(return_value=json.dumps({"status": "ok"}))
        self.record_action_fn = MagicMock(return_value=json.dumps({"status": "recorded"}))

        self.router = ToolRouter(
            memorize_fn=self.memorize_fn,
            recall_fn=self.recall_fn,
            govern_fn=self.govern_fn,
            reflect_fn=self.reflect_fn,
            compact_fn=self.compact_fn,
            detail_fn=self.detail_fn,
            memory_compat_fn=self.compat_fn,
            record_action_fn=self.record_action_fn,
        )

    def test_route_omni_memorize(self) -> None:
        result = self.router.route(OMNI_MEMORIZE, {"content": "测试"})
        data = json.loads(result)
        self.assertEqual(data["status"], "stored")
        self.memorize_fn.assert_called_once()

    def test_route_omni_recall(self) -> None:
        result = self.router.route(OMNI_RECALL, {"query": "test"})
        data = json.loads(result)
        self.assertEqual(data["count"], 3)
        self.recall_fn.assert_called_once()

    def test_route_omni_govern(self) -> None:
        result = self.router.route(OMNI_GOVERN, {"action": "archive"})
        data = json.loads(result)
        self.assertEqual(data["status"], "ok")
        self.govern_fn.assert_called_once()

    def test_route_omni_reflect(self) -> None:
        result = self.router.route(OMNI_REFLECT, {"query": "思考"})
        data = json.loads(result)
        self.assertEqual(data["status"], "reflected")

    def test_route_omni_compact(self) -> None:
        result = self.router.route(OMNI_COMPACT, {"budget": 2000})
        data = json.loads(result)
        self.assertEqual(data["status"], "ready")

    def test_route_omni_detail(self) -> None:
        result = self.router.route(OMNI_DETAIL, {"action": "list"})
        data = json.loads(result)
        self.assertEqual(data["status"], "ok")

    def test_route_omni_record_action(self) -> None:
        result = self.router.route(OMNI_RECORD_ACTION, {"action_type": "decision"})
        data = json.loads(result)
        self.assertEqual(data["status"], "recorded")
        self.record_action_fn.assert_called_once()

    def test_route_memory_compat(self) -> None:
        result = self.router.route(MEMORY_COMPAT, {"action": "add"})
        data = json.loads(result)
        self.assertEqual(data["status"], "ok")
        self.compat_fn.assert_called_once()

    def test_route_unknown_tool(self) -> None:
        result = self.router.route("nonexistent_tool", {})
        data = json.loads(result)
        self.assertIn("error", data)
        self.assertIn("Unknown tool", data["error"])

    def test_get_tool_names(self) -> None:
        names = self.router.get_tool_names()
        self.assertIn(OMNI_MEMORIZE, names)
        self.assertIn(OMNI_RECALL, names)
        self.assertIn(MEMORY_COMPAT, names)
        self.assertIn(OMNI_RECORD_ACTION, names)
        self.assertEqual(len(names), 8)

    def test_routes_match_tool_name_constants(self) -> None:
        """路由表 key 应与 tool_names 常量完全一致。"""
        names = set(self.router.get_tool_names())
        expected = {
            OMNI_MEMORIZE,
            OMNI_RECALL,
            OMNI_GOVERN,
            OMNI_REFLECT,
            OMNI_COMPACT,
            OMNI_DETAIL,
            OMNI_RECORD_ACTION,
            MEMORY_COMPAT,
        }
        self.assertEqual(names, expected)


# ──────────────────────────────────────────────
# handle_compact
# ──────────────────────────────────────────────

class TestHandleCompact(unittest.TestCase):

    def test_default_budget(self) -> None:
        result = json.loads(handle_compact({}))
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["budget"], 4000)

    def test_custom_budget(self) -> None:
        result = json.loads(handle_compact({"budget": 1500}))
        self.assertEqual(result["budget"], 1500)

    def test_message_present(self) -> None:
        result = json.loads(handle_compact({"budget": 1000}))
        self.assertIn("on_pre_compress", result["message"])


# ──────────────────────────────────────────────
# handle_detail
# ──────────────────────────────────────────────

class TestHandleDetail(unittest.TestCase):

    def setUp(self) -> None:
        self.ctx_mgr = MagicMock()
        self.store = MagicMock()
        self.forgetting = MagicMock()
        self.feedback = MagicMock()

    def test_list_empty(self) -> None:
        self.ctx_mgr.get_injected_items.return_value = []
        result = json.loads(handle_detail(
            {"action": "list"}, self.ctx_mgr, self.store,
            self.forgetting, self.feedback, turn_count=5, last_query=""
        ))
        self.assertEqual(result["status"], "empty")

    def test_list_with_items(self) -> None:
        self.ctx_mgr.get_injected_items.return_value = [
            {"memory_id": "m1", "content": "test"},
        ]
        self.store.get.return_value = {"memory_id": "m1"}
        result = json.loads(handle_detail(
            {"action": "list"}, self.ctx_mgr, self.store,
            self.forgetting, self.feedback, turn_count=5, last_query=""
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["count"], 1)

    def test_list_filters_orphaned(self) -> None:
        """store.get 返回 None 的条目应该被过滤掉。"""
        self.ctx_mgr.get_injected_items.return_value = [
            {"memory_id": "m1", "content": "valid"},
            {"memory_id": "m2", "content": "orphan"},
        ]
        self.store.get.side_effect = lambda mid: {"memory_id": mid} if mid == "m1" else None
        result = json.loads(handle_detail(
            {"action": "list"}, self.ctx_mgr, self.store,
            self.forgetting, self.feedback, turn_count=5, last_query=""
        ))
        self.assertEqual(result["count"], 1)

    def test_get_missing_id(self) -> None:
        result = json.loads(handle_detail(
            {"action": "get"}, self.ctx_mgr, self.store,
            self.forgetting, self.feedback, turn_count=5, last_query=""
        ))
        self.assertEqual(result["status"], "error")
        self.assertIn("memory_id", result["message"])

    def test_get_found_with_archived(self) -> None:
        self.ctx_mgr.get_detail_for.return_value = {"status": "found", "memory_id": "m1"}
        self.forgetting.get_stage.return_value = "archived"
        result = json.loads(handle_detail(
            {"action": "get", "memory_id": "m1"}, self.ctx_mgr, self.store,
            self.forgetting, self.feedback, turn_count=5, last_query="q"
        ))
        self.assertTrue(result.get("archived"))
        self.feedback.record_click.assert_called_once()

    def test_get_found_active(self) -> None:
        self.ctx_mgr.get_detail_for.return_value = {"status": "found", "memory_id": "m2"}
        self.forgetting.get_stage.return_value = "active"
        result = json.loads(handle_detail(
            {"action": "get", "memory_id": "m2"}, self.ctx_mgr, self.store,
            self.forgetting, self.feedback, turn_count=5, last_query=""
        ))
        self.assertFalse(result.get("archived"))

    def test_events_empty(self) -> None:
        self.store.search.return_value = []
        result = json.loads(handle_detail(
            {"action": "events", "from_turn": 0, "to_turn": 10},
            self.ctx_mgr, self.store, self.forgetting, self.feedback,
            turn_count=10, last_query=""
        ))
        self.assertEqual(result["count"], 0)

    def test_events_with_turn_filter(self) -> None:
        self.store.search.return_value = [
            {"memory_id": "e1", "content": "[Turn 3] something", "type": "event",
             "stored_at": "2026-01-01T00:00:00Z"},
            {"memory_id": "e2", "content": "[Turn 8] other", "type": "event",
             "stored_at": "2026-01-01T00:01:00Z"},
            {"memory_id": "e3", "content": "[Turn 12] outside", "type": "event",
             "stored_at": "2026-01-01T00:02:00Z"},
        ]
        result = json.loads(handle_detail(
            {"action": "events", "from_turn": 0, "to_turn": 10},
            self.ctx_mgr, self.store, self.forgetting, self.feedback,
            turn_count=10, last_query=""
        ))
        # Turn 12 outside range, so 2 events
        self.assertEqual(result["count"], 2)

    def test_events_query_filter(self) -> None:
        self.store.search.return_value = [
            {"memory_id": "e1", "content": "[Turn 1] python test", "type": "event",
             "stored_at": "2026-01-01T00:00:00Z"},
            {"memory_id": "e2", "content": "[Turn 2] rust test", "type": "event",
             "stored_at": "2026-01-01T00:01:00Z"},
        ]
        result = json.loads(handle_detail(
            {"action": "events", "from_turn": 0, "to_turn": 10, "query": "python"},
            self.ctx_mgr, self.store, self.forgetting, self.feedback,
            turn_count=10, last_query=""
        ))
        self.assertEqual(result["count"], 1)

    def test_unknown_action(self) -> None:
        result = json.loads(handle_detail(
            {"action": "delete"}, self.ctx_mgr, self.store,
            self.forgetting, self.feedback, turn_count=5, last_query=""
        ))
        self.assertIn("error", result)


# ──────────────────────────────────────────────
# build_system_prompt
# ──────────────────────────────────────────────

class TestBuildSystemPrompt(unittest.TestCase):

    def setUp(self) -> None:
        self.core_block = MagicMock()
        self.core_block.identity_block = "AI助手 v1"
        self.ctx_mgr = MagicMock()
        self.ctx_mgr.max_summary_chars = 200
        self.ctx_mgr.get_injected_fingerprints.return_value = set()
        self.ctx_mgr.add_persistent_fingerprint = MagicMock()
        self.config = {"system_prompt_char_limit": 500}
        self.store = MagicMock()

    def test_cache_hit(self) -> None:
        cached = "cached prompt"
        result, tc, val = build_system_prompt(
            data_dir="/tmp/mem", store=self.store, core_block=self.core_block,
            context_manager=self.ctx_mgr, config=self.config, turn_count=3,
            system_prompt_cache_turn=3, system_prompt_cache_value=cached,
            last_query=""
        )
        self.assertEqual(result, cached)

    def test_empty_store(self) -> None:
        self.store.search.return_value = []
        result, tc, val = build_system_prompt(
            data_dir="/tmp/mem", store=self.store, core_block=self.core_block,
            context_manager=self.ctx_mgr, config=self.config, turn_count=1,
            system_prompt_cache_turn=0, system_prompt_cache_value="",
            last_query=""
        )
        self.assertIn("Identity", result)
        self.assertIn("AI助手 v1", result)

    def test_with_entries(self) -> None:
        self.store.search.side_effect = lambda memory_type, **__: {
            "preference": [
                {"content": "喜欢深色主题", "memory_id": "p1", "type": "preference"}
            ],
            "correction": [],
            "fact": [
                {"content": "项目使用Python", "memory_id": "f1", "type": "fact"}
            ],
        }.get(memory_type, [])

        self.ctx_mgr.refine_content = MagicMock(side_effect=lambda c, _: c)
        self.ctx_mgr._content_fingerprint = MagicMock(return_value="fp1")
        self.ctx_mgr._fingerprint_similarity = MagicMock(return_value=0.0)

        result, tc, val = build_system_prompt(
            data_dir="/tmp/mem", store=self.store, core_block=self.core_block,
            context_manager=self.ctx_mgr, config=self.config, turn_count=1,
            system_prompt_cache_turn=0, system_prompt_cache_value="",
            last_query=""
        )
        self.assertIn("Core Memories", result)
        self.assertIn("喜欢深色主题", result)

    def test_fingerprint_dedup(self) -> None:
        self.store.search.side_effect = lambda memory_type, **__: {
            "preference": [
                {"content": "喜欢Python", "memory_id": "p1", "type": "preference"}
            ],
            "correction": [],
            "fact": [
                {"content": "也喜欢Python", "memory_id": "f1", "type": "fact"}
            ],
        }.get(memory_type, [])

        self.ctx_mgr.refine_content = MagicMock(return_value="喜欢Python")
        self.ctx_mgr._content_fingerprint = MagicMock(return_value="fp_same")
        self.ctx_mgr._fingerprint_similarity = MagicMock(return_value=0.9)
        self.ctx_mgr.get_injected_fingerprints.return_value = set()

        result, tc, val = build_system_prompt(
            data_dir="/tmp/mem", store=self.store, core_block=self.core_block,
            context_manager=self.ctx_mgr, config=self.config, turn_count=1,
            system_prompt_cache_turn=0, system_prompt_cache_value="",
            last_query=""
        )
        # fact entry should be dedup'd (same fingerprint as preference)
        self.assertIn("[preference]", result)
        self.assertNotIn("[fact]", result)


# ──────────────────────────────────────────────
# get_config_schema / save_config
# ──────────────────────────────────────────────

class TestConfigSchema(unittest.TestCase):

    def test_get_non_empty(self) -> None:
        schema = get_config_schema()
        self.assertIsInstance(schema, list)
        self.assertGreater(len(schema), 5)

    def test_all_have_keys(self) -> None:
        schema = get_config_schema()
        for item in schema:
            self.assertIn("key", item)
            self.assertIn("default", item)

    def test_save_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / ".hermes"
            save_config({"test_key": "test_val"}, str(home))
            config_path = home / "omnimem" / "config.yaml"
            self.assertTrue(config_path.exists())
            content = config_path.read_text(encoding="utf-8")
            self.assertIn("test_key", content)


# ──────────────────────────────────────────────
# apply_sync_change
# ──────────────────────────────────────────────

class TestApplySyncChange(unittest.TestCase):

    def setUp(self) -> None:
        self.store = MagicMock()
        self.index = MagicMock()
        self.retriever = MagicMock()
        self.forgetting = MagicMock()

    def test_delete_operation(self) -> None:
        change = {
            "operation": "DELETE",
            "data": {"memory_id": "sync-del-1"}
        }
        result = apply_sync_change(change, self.store, self.index, self.retriever, self.forgetting)
        self.assertTrue(result)
        self.forgetting.archive.assert_called_once_with("sync-del-1")

    def test_insert_operation(self) -> None:
        change = {
            "operation": "INSERT",
            "data": {
                "memory_id": "sync-ins-1",
                "content": "hello sync",
                "type": "fact",
                "wing": "auto",
                "room": "sync",
                "confidence": 3,
                "privacy": "personal",
                "vc": "v1",
                "stored_at": "2026-01-01T00:00:00Z",
            },
            "instance_id": "peer-1",
            "vc": "v1",
        }
        result = apply_sync_change(change, self.store, self.index, self.retriever, self.forgetting)
        self.assertTrue(result)
        self.store.add.assert_called_once()
        self.index.add.assert_called_once()
        self.retriever.add.assert_called_once()

    def test_empty_memory_id(self) -> None:
        change = {"operation": "INSERT", "data": {"memory_id": ""}}
        result = apply_sync_change(change, self.store, self.index, self.retriever, self.forgetting)
        self.assertFalse(result)

    def test_no_data(self) -> None:
        change = {"operation": "INSERT", "data": {}}
        result = apply_sync_change(change, self.store, self.index, self.retriever, self.forgetting)
        self.assertFalse(result)

    def test_store_add_raises(self) -> None:
        self.store.add.side_effect = RuntimeError("db locked")
        change = {
            "operation": "INSERT",
            "data": {"memory_id": "err-1", "content": "test", "type": "fact"}
        }
        result = apply_sync_change(change, self.store, self.index, self.retriever, self.forgetting)
        self.assertFalse(result)


# ──────────────────────────────────────────────
# retry_index_add / retry_retriever_add / retry_kg_extract
# ──────────────────────────────────────────────

class TestRetryHelpers(unittest.TestCase):

    def setUp(self) -> None:
        self.store = MagicMock()
        self.index = MagicMock()
        self.retriever = MagicMock()
        self.kg = MagicMock()

    def test_retry_index_add_success(self) -> None:
        self.store.get.return_value = {
            "memory_id": "r1", "wing": "personal", "room": "test",
            "content": "hello", "type": "fact", "confidence": 3,
            "privacy": "personal", "stored_at": "2026-01-01",
        }
        retry_index_add("r1", self.store, self.index)
        self.index.add.assert_called_once()

    def test_retry_index_add_not_found(self) -> None:
        self.store.get.return_value = None
        with self.assertRaises(RuntimeError):
            retry_index_add("r1", self.store, self.index)

    def test_retry_retriever_add_success(self) -> None:
        self.store.get.return_value = {
            "memory_id": "r1", "wing": "personal", "room": "test",
            "content": "hello", "type": "fact", "confidence": 3,
            "privacy": "personal", "stored_at": "2026-01-01",
        }
        retry_retriever_add("r1", self.store, self.retriever)
        self.retriever.add.assert_called_once()

    def test_retry_retriever_add_not_found(self) -> None:
        self.store.get.return_value = None
        with self.assertRaises(RuntimeError):
            retry_retriever_add("r1", self.store, self.retriever)

    def test_retry_kg_extract_success(self) -> None:
        self.store.get.return_value = {
            "memory_id": "r1", "content": "Python is great", "confidence": 5,
        }
        retry_kg_extract("r1", self.store, self.kg)
        self.kg.extract_and_store.assert_called_once()

    def test_retry_kg_extract_kg_none(self) -> None:
        self.store.get.return_value = {
            "memory_id": "r1", "content": "test", "confidence": 3,
        }
        # Should not raise when kg is None
        retry_kg_extract("r1", self.store, None)

    def test_retry_kg_extract_not_found(self) -> None:
        self.store.get.return_value = None
        with self.assertRaises(RuntimeError):
            retry_kg_extract("r1", self.store, self.kg)


# ──────────────────────────────────────────────
# l3_recall
# ──────────────────────────────────────────────

class TestL3Recall(unittest.TestCase):

    def setUp(self) -> None:
        self.retriever = MagicMock()
        self.store = MagicMock()

    def test_retriever_returns_results(self) -> None:
        results = [{"content": "匹配结果", "memory_id": "r1", "score": 0.9}]
        self.retriever.search.return_value = results
        out = l3_recall("query", self.retriever, self.store)
        self.assertEqual(len(out), 1)

    def test_fallback_to_store(self) -> None:
        self.retriever.search.return_value = []
        self.store.search_by_content.return_value = [
            {"content": "测试内容匹配", "memory_id": "f1"}
        ]
        out = l3_recall("测试内容", self.retriever, self.store)
        self.assertGreaterEqual(len(out), 1)
        self.assertEqual(out[0].get("_source"), "store_fallback")


# ──────────────────────────────────────────────
# run_prefetch
# ──────────────────────────────────────────────

class TestRunPrefetch(unittest.TestCase):

    def setUp(self) -> None:
        self.ctx_mgr = MagicMock()
        self.retriever = MagicMock()
        self.kv_cache = None
        self.kg = None
        self.temporal_decay = MagicMock()
        self.temporal_decay.apply = MagicMock(side_effect=lambda x: x)
        self.privacy = MagicMock()
        self.privacy.filter = MagicMock(side_effect=lambda x, **_kw: x)
        self.lock = MagicMock()

    def test_empty_all(self) -> None:
        self.retriever.search.return_value = []
        result, _ = run_prefetch(
            query="test", session_id="s1", config={}, retriever=self.retriever,
            context_manager=self.ctx_mgr, kv_cache=self.kv_cache,
            knowledge_graph=self.kg, temporal_decay=self.temporal_decay,
            privacy=self.privacy, prefetch_cache="", prefetch_lock=self.lock,
        )
        self.assertEqual(result, "")

    def test_returns_refined(self) -> None:
        self.retriever.search.return_value = [
            {"content": "test result", "type": "fact", "score": 0.8, "memory_id": "m1"}
        ]
        self.ctx_mgr.refine_prefetch_results.return_value = "refined_output"
        result, _ = run_prefetch(
            query="test", session_id="s1", config={}, retriever=self.retriever,
            context_manager=self.ctx_mgr, kv_cache=self.kv_cache,
            knowledge_graph=self.kg, temporal_decay=self.temporal_decay,
            privacy=self.privacy, prefetch_cache="", prefetch_lock=self.lock,
        )
        self.assertEqual(result, "refined_output")

    def test_cached_prefetch(self) -> None:
        cached_data = "___RAW_RESULTS___" + json.dumps([
            {"content": "cached result", "memory_id": "c1"}
        ])
        self.ctx_mgr.refine_prefetch_results.return_value = "cached_output"
        result, _ = run_prefetch(
            query="test", session_id="s1", config={}, retriever=self.retriever,
            context_manager=self.ctx_mgr, kv_cache=self.kv_cache,
            knowledge_graph=self.kg, temporal_decay=self.temporal_decay,
            privacy=self.privacy, prefetch_cache=cached_data, prefetch_lock=self.lock,
        )
        self.assertEqual(result, "cached_output")

    def test_cached_parse_error(self) -> None:
        self.retriever.search.return_value = []
        result, _ = run_prefetch(
            query="test", session_id="s1", config={}, retriever=self.retriever,
            context_manager=self.ctx_mgr, kv_cache=self.kv_cache,
            knowledge_graph=self.kg, temporal_decay=self.temporal_decay,
            privacy=self.privacy, prefetch_cache="___RAW_RESULTS___INVALID_JSON",
            prefetch_lock=self.lock,
        )
        self.assertEqual(result, "")

    def test_kv_cache_hits(self) -> None:
        kv = MagicMock()
        kv.search_cache.return_value = [
            {"content": "kv hit", "memory_id": "kv1"}
        ]
        self.ctx_mgr.refine_prefetch_results.return_value = "kv_output"
        result, _ = run_prefetch(
            query="test", session_id="s1", config={}, retriever=self.retriever,
            context_manager=self.ctx_mgr, kv_cache=kv,
            knowledge_graph=self.kg, temporal_decay=self.temporal_decay,
            privacy=self.privacy, prefetch_cache="", prefetch_lock=self.lock,
        )
        self.assertEqual(result, "kv_output")
        # should not call retriever when kv cache hits
        self.retriever.search.assert_not_called()


# ──────────────────────────────────────────────
# run_queue_prefetch
# ──────────────────────────────────────────────

class TestRunQueuePrefetch(unittest.TestCase):

    def setUp(self) -> None:
        self.retriever = MagicMock()
        self.temporal_decay = MagicMock()
        self.temporal_decay.apply = MagicMock(side_effect=lambda x: x)
        self.privacy = MagicMock()
        self.privacy.filter = MagicMock(side_effect=lambda x, **_kw: x)
        self.lock = MagicMock()

    def test_returns_serialized(self) -> None:
        self.retriever.search.return_value = [
            {"content": "bg result", "memory_id": "bg1"}
        ]
        result = run_queue_prefetch(
            query="bg", session_id="s1", config={}, retriever=self.retriever,
            temporal_decay=self.temporal_decay, privacy=self.privacy,
            prefetch_lock=self.lock,
        )
        self.assertTrue(result.startswith("___RAW_RESULTS___"))

    def test_empty_result(self) -> None:
        self.retriever.search.return_value = []
        result = run_queue_prefetch(
            query="bg", session_id="s1", config={}, retriever=self.retriever,
            temporal_decay=self.temporal_decay, privacy=self.privacy,
            prefetch_lock=self.lock,
        )
        self.assertEqual(result, "")

    def test_exception_returns_empty(self) -> None:
        self.retriever.search.side_effect = RuntimeError("boom")
        result = run_queue_prefetch(
            query="bg", session_id="s1", config={}, retriever=self.retriever,
            temporal_decay=self.temporal_decay, privacy=self.privacy,
            prefetch_lock=self.lock,
        )
        self.assertEqual(result, "")


# ──────────────────────────────────────────────
# Provider 工具名一致性
# ──────────────────────────────────────────────

class TestProviderToolNameConsistency(unittest.TestCase):
    """Provider 注册的工具名应与 tool_names 常量一致。"""

    def test_provider_schema_names_match_constants(self) -> None:
        provider = OmniMemProvider()
        schemas = provider.get_tool_schemas()
        names = {s["name"] for s in schemas}
        expected = {
            OMNI_MEMORIZE,
            OMNI_RECALL,
            OMNI_GOVERN,
            OMNI_REFLECT,
            OMNI_COMPACT,
            OMNI_DETAIL,
            OMNI_RECORD_ACTION,
            MEMORY_COMPAT,
        }
        self.assertEqual(names, expected)
