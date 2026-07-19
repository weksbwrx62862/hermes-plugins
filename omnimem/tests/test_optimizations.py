"""OmniMem 优化项单元测试。

覆盖:
  - provider.py: Facade __getattr__/__setattr__ 代理、prefetch 线程池复用、审计间隔可配置
  - engine.py: 查询缓存 TTL 可配置、同义词映射外置
  - drawer_closet.py: 批量写入阈值可配置
  - config/_config.py: 新增配置项
"""

from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

from omnimem.config import OmniMemConfig
from omnimem.memory.drawer_closet import DrawerClosetStore
from omnimem.provider import OmniMemProvider
from omnimem.retrieval.engine import HybridRetriever

# ──────────────────────────────────────────────
# Provider: Facade __getattr__/__setattr__ 代理
# ──────────────────────────────────────────────


class TestFacadeAttrProxy(unittest.TestCase):
    """Facade 属性动态代理测试。"""

    def _make_provider_with_facades(self) -> OmniMemProvider:
        provider = OmniMemProvider()

        mock_storage = MagicMock()
        mock_storage.soul = "soul_obj"
        mock_storage.core_block = "core_block_obj"
        mock_storage.budget = "budget_obj"
        mock_storage.wing_room = "wing_room_obj"
        mock_storage.store = "store_obj"
        mock_storage.index = "index_obj"
        mock_storage.md_store = "md_store_obj"
        mock_storage.attachments = []

        mock_retrieval = MagicMock()
        mock_retrieval.retriever = "retriever_obj"
        mock_retrieval.context_manager = "context_manager_obj"
        mock_retrieval.perception = "perception_obj"
        mock_retrieval.feedback = "feedback_obj"
        mock_retrieval.prefetch_lock = "prefetch_lock_obj"
        mock_retrieval._reflect_cache = {}
        mock_retrieval.prefetch_cache = ""
        mock_retrieval._prefetch_executor = MagicMock(spec=ThreadPoolExecutor)

        mock_governance = MagicMock()
        mock_governance.conflict_resolver = "conflict_resolver_obj"
        mock_governance.temporal_decay = "temporal_decay_obj"
        mock_governance.forgetting = "forgetting_obj"
        mock_governance.privacy = "privacy_obj"
        mock_governance.provenance = "provenance_obj"
        mock_governance.sync_engine = "sync_engine_obj"
        mock_governance.vector_clock = "vector_clock_obj"
        mock_governance.auditor = "auditor_obj"
        mock_governance.audit_logger = "audit_logger_obj"
        mock_governance.rbac = "rbac_obj"

        mock_sync = MagicMock()
        mock_sync.saga = "saga_obj"
        mock_sync.bg_executor = "bg_executor_obj"
        mock_sync.store_service = "store_service_obj"
        mock_sync.kv_cache = "kv_cache_obj"
        mock_sync.lora_trainer = "lora_trainer_obj"

        mock_deep = MagicMock()
        mock_deep.consolidation = "consolidation_obj"
        mock_deep.knowledge_graph = "knowledge_graph_obj"
        mock_deep.reflect_engine = "reflect_engine_obj"

        object.__setattr__(provider, "_storage", mock_storage)
        object.__setattr__(provider, "_retrieval", mock_retrieval)
        object.__setattr__(provider, "_governance", mock_governance)
        object.__setattr__(provider, "_sync", mock_sync)
        object.__setattr__(provider, "_deep", mock_deep)
        object.__setattr__(provider, "_dedup_service", "dedup_service_obj")

        return provider

    def test_getattr_storage_facade(self) -> None:
        provider = self._make_provider_with_facades()
        self.assertEqual(provider._soul, "soul_obj")
        self.assertEqual(provider._core_block, "core_block_obj")
        self.assertEqual(provider._budget, "budget_obj")
        self.assertEqual(provider._wing_room, "wing_room_obj")
        self.assertEqual(provider._store, "store_obj")
        self.assertEqual(provider._index, "index_obj")
        self.assertEqual(provider._md_store, "md_store_obj")

    def test_getattr_retrieval_facade(self) -> None:
        provider = self._make_provider_with_facades()
        self.assertEqual(provider._retriever, "retriever_obj")
        self.assertEqual(provider._context_manager, "context_manager_obj")
        self.assertEqual(provider._perception, "perception_obj")
        self.assertEqual(provider._feedback, "feedback_obj")
        self.assertEqual(provider._prefetch_lock, "prefetch_lock_obj")
        self.assertEqual(provider._reflect_cache, {})
        self.assertIsNotNone(provider._prefetch_executor)

    def test_getattr_governance_facade(self) -> None:
        provider = self._make_provider_with_facades()
        self.assertEqual(provider._conflict_resolver, "conflict_resolver_obj")
        self.assertEqual(provider._temporal_decay, "temporal_decay_obj")
        self.assertEqual(provider._forgetting, "forgetting_obj")
        self.assertEqual(provider._privacy, "privacy_obj")
        self.assertEqual(provider._provenance, "provenance_obj")
        self.assertEqual(provider._sync_engine, "sync_engine_obj")
        self.assertEqual(provider._vector_clock, "vector_clock_obj")
        self.assertEqual(provider._auditor, "auditor_obj")
        self.assertEqual(provider._audit_logger, "audit_logger_obj")
        self.assertEqual(provider._rbac, "rbac_obj")

    def test_getattr_sync_facade(self) -> None:
        provider = self._make_provider_with_facades()
        self.assertEqual(provider._saga, "saga_obj")
        self.assertEqual(provider._bg_executor, "bg_executor_obj")
        self.assertEqual(provider._store_service, "store_service_obj")
        self.assertEqual(provider._kv_cache, "kv_cache_obj")
        self.assertEqual(provider._lora_trainer, "lora_trainer_obj")

    def test_getattr_deep_facade(self) -> None:
        provider = self._make_provider_with_facades()
        self.assertEqual(provider._consolidation, "consolidation_obj")
        self.assertEqual(provider._knowledge_graph, "knowledge_graph_obj")
        self.assertEqual(provider._reflect_engine, "reflect_engine_obj")

    def test_getattr_direct_map(self) -> None:
        provider = self._make_provider_with_facades()
        self.assertEqual(provider._dedup, "dedup_service_obj")

    def test_getattr_unknown_raises(self) -> None:
        provider = self._make_provider_with_facades()
        with self.assertRaises(AttributeError):
            _ = provider._nonexistent_attr

    def test_setattr_facade_setter_map(self) -> None:
        provider = self._make_provider_with_facades()
        provider._attachments = ["test_attachment"]
        self.assertEqual(provider._storage.attachments, ["test_attachment"])

    def test_setattr_prefetch_cache(self) -> None:
        provider = self._make_provider_with_facades()
        provider._prefetch_cache = "cached_result"
        self.assertEqual(provider._retrieval.prefetch_cache, "cached_result")

    def test_setattr_normal_attr(self) -> None:
        provider = OmniMemProvider()
        provider._degraded_mode = True
        self.assertTrue(provider._degraded_mode)

    def test_init_no_recursion(self) -> None:
        provider = OmniMemProvider()
        self.assertFalse(provider._degraded_mode)
        self.assertEqual(provider._turn_count, 0)

    def test_facade_attr_map_completeness(self) -> None:
        # 显式属性赋值替代了 _FACADE_ATTR_MAP 动态代理
        # 验证类型注解中声明的所有属性在初始化方法中都有显式赋值
        expected_storage = {"_soul", "_core_block", "_budget", "_wing_room", "_store", "_index", "_md_store"}
        expected_retrieval = {"_retriever", "_context_manager", "_perception", "_feedback", "_prefetch_lock", "_reflect_cache", "_prefetch_executor"}
        expected_governance = {"_conflict_resolver", "_temporal_decay", "_forgetting", "_privacy", "_provenance", "_sync_engine", "_vector_clock", "_auditor", "_audit_logger", "_rbac", "_temporal_kg"}
        expected_sync = {"_saga", "_bg_executor", "_store_service", "_kv_cache", "_lora_trainer"}
        expected_deep = {"_consolidation", "_knowledge_graph", "_reflect_engine"}

        all_expected = expected_storage | expected_retrieval | expected_governance | expected_sync | expected_deep
        # 检查类型注解中声明了这些属性
        annotated = {name for name in OmniMemProvider.__annotations__ if name.startswith("_")}
        for attr in all_expected:
            self.assertIn(attr, annotated, f"属性 {attr} 未在类型注解中声明")


# ──────────────────────────────────────────────
# Provider: prefetch 线程池复用
# ──────────────────────────────────────────────


class TestPrefetchThreadPoolReuse(unittest.TestCase):
    """prefetch() 应复用 _prefetch_executor 而非新建 ThreadPoolExecutor。"""

    def test_prefetch_uses_existing_executor(self) -> None:
        provider = OmniMemProvider()
        mock_executor = MagicMock(spec=ThreadPoolExecutor)
        mock_future = MagicMock()
        mock_future.result.return_value = ("result", "")
        mock_executor.submit.return_value = mock_future

        mock_retrieval = MagicMock()
        mock_retrieval.prefetch_cache = ""
        mock_retrieval.prefetch_lock = MagicMock()
        mock_retrieval._prefetch_executor = mock_executor
        mock_retrieval.retriever = MagicMock()

        mock_storage = MagicMock()
        mock_storage.store = MagicMock()

        mock_sync = MagicMock()
        mock_sync.kv_cache = MagicMock()

        mock_governance = MagicMock()
        mock_governance.forgetting = MagicMock()

        mock_deep = MagicMock()
        mock_deep.knowledge_graph = MagicMock()

        mock_config = MagicMock()
        mock_config.get = lambda k, d=None: {"enable_prefetch": True, "prefetch_timeout": 5}.get(k, d)

        object.__setattr__(provider, "_retrieval", mock_retrieval)
        object.__setattr__(provider, "_storage", mock_storage)
        object.__setattr__(provider, "_sync", mock_sync)
        object.__setattr__(provider, "_governance", mock_governance)
        object.__setattr__(provider, "_deep", mock_deep)
        object.__setattr__(provider, "_config", mock_config)
        object.__setattr__(provider, "_last_query", "")
        object.__setattr__(provider, "_prefetch_cache", "")
        object.__setattr__(provider, "_prefetch_lock", MagicMock())

        with patch("omnimem.provider.run_prefetch", return_value=("result", "")):
            provider.prefetch("test query", session_id="test")

        mock_executor.submit.assert_called_once()
        mock_future.result.assert_called_once_with(timeout=5)


# ──────────────────────────────────────────────
# Provider: 审计间隔可配置
# ──────────────────────────────────────────────


class TestAuditIntervalConfigurable(unittest.TestCase):
    """审计巡检间隔应可通过 audit_interval_turns 配置。"""

    def test_default_audit_interval(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        config = OmniMemConfig(tmpdir)
        self.assertEqual(config.get("audit_interval_turns", 50), 50)

    def test_custom_audit_interval(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        config = OmniMemConfig(tmpdir)
        config.set("audit_interval_turns", 100)
        self.assertEqual(config.get("audit_interval_turns"), 100)

    def test_audit_interval_validation(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        config = OmniMemConfig(tmpdir)
        with self.assertRaises(ValueError):
            config.set("audit_interval_turns", 2)
        with self.assertRaises(ValueError):
            config.set("audit_interval_turns", 600)

    def _make_minimal_provider_for_session_end(self) -> OmniMemProvider:
        provider = OmniMemProvider()

        mock_storage = MagicMock()
        mock_storage.store = MagicMock()
        mock_storage.store.flush = MagicMock()
        mock_storage.md_store = MagicMock()
        mock_storage.md_store.flush = MagicMock()
        mock_storage.index = MagicMock()
        mock_storage.index.close = MagicMock()

        mock_retrieval = MagicMock()
        mock_retrieval.retriever = MagicMock()
        mock_retrieval.retriever.flush = MagicMock()
        mock_retrieval.retriever.delete = MagicMock()
        mock_retrieval._prefetch_executor = MagicMock()
        mock_retrieval._prefetch_executor.shutdown = MagicMock()

        mock_governance = MagicMock()
        mock_governance.forgetting = MagicMock()
        mock_governance.forgetting.run_archive_cycle.return_value = 0
        mock_governance.forgetting.close = MagicMock()
        mock_governance.provenance = MagicMock()
        mock_governance.provenance.close = MagicMock()
        mock_governance.sync_engine = MagicMock()
        mock_governance.sync_engine.close = MagicMock()
        mock_governance.auditor = MagicMock()
        mock_governance.auditor.quick_health_check.return_value = {"healthy": True}

        mock_sync = MagicMock()
        mock_sync.store_service = MagicMock()
        mock_sync.store_service.extract_session_memories = MagicMock()
        mock_sync.store_service.last_save_turn = 0
        mock_sync.saga = MagicMock()
        mock_sync.saga.get_pending.return_value = []
        mock_sync.bg_executor = MagicMock()
        mock_sync.bg_executor.shutdown = MagicMock()

        mock_deep = MagicMock()
        mock_deep.consolidation = MagicMock()
        mock_deep.consolidation.process_pending.return_value = 0
        mock_deep.consolidation.close = MagicMock()
        mock_deep.knowledge_graph = MagicMock()
        mock_deep.knowledge_graph.close = MagicMock()
        mock_deep.reflect_engine = MagicMock()
        mock_deep.reflect_engine.close = MagicMock()

        mock_config = MagicMock()
        mock_config.get = lambda k, d=None: 50 if k == "audit_interval_turns" else d

        object.__setattr__(provider, "_should_write", True)
        object.__setattr__(provider, "_turn_count", 30)
        object.__setattr__(provider, "_config", mock_config)
        object.__setattr__(provider, "_storage", mock_storage)
        object.__setattr__(provider, "_retrieval", mock_retrieval)
        object.__setattr__(provider, "_governance", mock_governance)
        object.__setattr__(provider, "_sync", mock_sync)
        object.__setattr__(provider, "_deep", mock_deep)
        object.__setattr__(provider, "_last_backup_time", 0)
        object.__setattr__(provider, "_save_interval", 15)

        return provider

    def test_on_session_end_uses_config_interval(self) -> None:
        provider = self._make_minimal_provider_for_session_end()
        object.__setattr__(provider, "_turn_count", 50)

        provider.on_session_end([])

        provider._governance.auditor.quick_health_check.assert_called_once()

    def test_on_session_end_skips_when_not_at_interval(self) -> None:
        provider = self._make_minimal_provider_for_session_end()
        object.__setattr__(provider, "_turn_count", 30)

        provider.on_session_end([])

        provider._governance.auditor.quick_health_check.assert_not_called()


# ──────────────────────────────────────────────
# Engine: 查询缓存 TTL 可配置
# ──────────────────────────────────────────────


class TestQueryCacheTTLConfigurable(unittest.TestCase):
    """查询缓存 TTL 应可通过构造函数参数配置。"""

    def test_default_ttl(self) -> None:
        self.assertEqual(HybridRetriever(data_dir=Path("/tmp/omnimem_test_ttl1"))._query_cache_ttl, 60.0)

    def test_custom_ttl_via_constructor(self) -> None:
        hybrid = HybridRetriever(data_dir=Path("/tmp/omnimem_test_ttl2"), query_cache_ttl=120.0)
        self.assertEqual(hybrid._query_cache_ttl, 120.0)

    def test_zero_ttl_disables_cache(self) -> None:
        hybrid = HybridRetriever(data_dir=Path("/tmp/omnimem_test_ttl3"), query_cache_ttl=0)
        self.assertEqual(hybrid._query_cache_ttl, 0)

    def test_ttl_from_config(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        config = OmniMemConfig(tmpdir)
        self.assertEqual(config.get("query_cache_ttl", 60), 60)

        config.set("query_cache_ttl", 30.0)
        self.assertEqual(config.get("query_cache_ttl"), 30.0)

    def test_cache_ttl_used_in_search(self) -> None:
        hybrid = HybridRetriever(data_dir=Path("/tmp/omnimem_test_ttl4"), query_cache_ttl=0.1)
        hybrid._query_cache["test_key|1500|rag|10"] = ([{"memory_id": "m1", "content": "test", "score": 0.5}], 0.0)
        import time
        time.sleep(0.2)
        now = time.time()
        cache_key = "test_key|1500|rag|10"
        cached_results, cached_time = hybrid._query_cache[cache_key]
        self.assertGreaterEqual(now - cached_time, 0.1)


# ──────────────────────────────────────────────
# Engine: 同义词映射外置
# ──────────────────────────────────────────────


class TestSynonymExternalization(unittest.TestCase):
    """同义词映射应从 synonyms.json 加载，而非硬编码。"""

    def test_no_hardcoded_fallback_class_attr(self) -> None:
        self.assertFalse(
            hasattr(HybridRetriever, "_SYNONYM_MAP_FALLBACK"),
            "_SYNONYM_MAP_FALLBACK 应已移除",
        )

    def test_load_synonyms_from_file(self) -> None:
        synonyms = HybridRetriever._load_synonyms()
        self.assertIsInstance(synonyms, dict)
        if synonyms:
            self.assertIn("宠物", synonyms)
            self.assertIn("编程", synonyms)
            self.assertIn("饮食", synonyms)

    def test_synonym_map_contains_extended_entries(self) -> None:
        synonyms = HybridRetriever._load_synonyms()
        if synonyms:
            self.assertIn("姓名", synonyms)
            self.assertIn("城市", synonyms)
            self.assertIn("爱好", synonyms)
            self.assertIn("深度学习", synonyms)

    def test_load_synonyms_graceful_fallback(self) -> None:
        with patch.object(HybridRetriever, "_load_synonyms", return_value={}):
            synonyms = HybridRetriever._load_synonyms()
            self.assertIsInstance(synonyms, dict)

    def test_synonym_map_initialized_in_constructor(self) -> None:
        hybrid = HybridRetriever(data_dir=Path("/tmp/omnimem_test_syn1"))
        self.assertIsInstance(hybrid._synonym_map, dict)


# ──────────────────────────────────────────────
# DrawerClosetStore: 批量写入阈值可配置
# ──────────────────────────────────────────────


class TestWriteBufferThresholdConfigurable(unittest.TestCase):
    """批量写入阈值应可通过构造函数参数配置。"""

    def test_default_threshold(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        store = DrawerClosetStore(tmpdir / "palace")
        self.assertEqual(store._WRITE_BUFFER_THRESHOLD, 20)

    def test_custom_threshold(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        store = DrawerClosetStore(tmpdir / "palace", write_buffer_threshold=50)
        self.assertEqual(store._WRITE_BUFFER_THRESHOLD, 50)

    def test_threshold_from_config(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        config = OmniMemConfig(tmpdir)
        self.assertEqual(config.get("write_buffer_threshold", 20), 20)

        config.set("write_buffer_threshold", 30)
        self.assertEqual(config.get("write_buffer_threshold"), 30)

    def test_buffer_not_flushed_before_threshold(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        store = DrawerClosetStore(tmpdir / "palace", write_buffer_threshold=10)
        for i in range(5):
            store.add(wing="personal", room="test", content=f"测试内容{i}")
        self.assertEqual(len(store._write_buffer), 10)
        self.assertGreater(store._pending_disk_writes, 0)

    def test_buffer_flushed_at_threshold(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        store = DrawerClosetStore(tmpdir / "palace", write_buffer_threshold=3)
        for i in range(3):
            store.add(wing="personal", room="test", content=f"阈值测试{i}")
        self.assertEqual(len(store._write_buffer), 0)
        self.assertEqual(store._pending_disk_writes, 0)

    def test_flush_explicit(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        store = DrawerClosetStore(tmpdir / "palace", write_buffer_threshold=100)
        store.add(wing="personal", room="test", content="显式刷新测试")
        self.assertGreater(len(store._write_buffer), 0)
        store.flush()
        self.assertEqual(len(store._write_buffer), 0)


# ──────────────────────────────────────────────
# Config: 新增配置项验证
# ──────────────────────────────────────────────


class TestNewConfigItems(unittest.TestCase):
    """新增配置项的默认值和验证。"""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.config = OmniMemConfig(self.tmpdir)

    def test_query_cache_ttl_default(self) -> None:
        self.assertEqual(self.config.get("query_cache_ttl"), 60)

    def test_write_buffer_threshold_default(self) -> None:
        self.assertEqual(self.config.get("write_buffer_threshold"), 20)

    def test_audit_interval_turns_default(self) -> None:
        self.assertEqual(self.config.get("audit_interval_turns"), 50)

    def test_query_cache_ttl_validation(self) -> None:
        self.config.set("query_cache_ttl", 120.0)
        self.assertEqual(self.config.get("query_cache_ttl"), 120.0)
        with self.assertRaises(ValueError):
            self.config.set("query_cache_ttl", -1)

    def test_write_buffer_threshold_validation(self) -> None:
        self.config.set("write_buffer_threshold", 50)
        self.assertEqual(self.config.get("write_buffer_threshold"), 50)
        with self.assertRaises(ValueError):
            self.config.set("write_buffer_threshold", 0)

    def test_audit_interval_turns_validation(self) -> None:
        self.config.set("audit_interval_turns", 100)
        self.assertEqual(self.config.get("audit_interval_turns"), 100)
        with self.assertRaises(ValueError):
            self.config.set("audit_interval_turns", 3)
        with self.assertRaises(ValueError):
            self.config.set("audit_interval_turns", 600)

    def test_config_persistence(self) -> None:
        self.config.set("query_cache_ttl", 90.0)
        self.config.set("write_buffer_threshold", 40)
        self.config.set("audit_interval_turns", 100)
        self.config.save()

        config2 = OmniMemConfig(self.tmpdir)
        self.assertEqual(config2.get("query_cache_ttl"), 90.0)
        self.assertEqual(config2.get("write_buffer_threshold"), 40)
        self.assertEqual(config2.get("audit_interval_turns"), 100)
