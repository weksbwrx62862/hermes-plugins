"""OmniMem Provider — 五层混合记忆系统，实现 Hermes MemoryProvider ABC。

安装: 将本目录放入 plugins/memory/omnimem/
配置: config.yaml → memory.provider: omnimem

五层架构:
  L0 感知层  — 主动监控 + 信号检测 + 意图预测
  L1 工作记忆 — CoreBlock(常驻上下文) + Attachment(压缩后状态)
  L2 结构化记忆 — Wing/Room 宫殿导航 + Drawer/Closet 双存储
  L3 深层记忆 — Consolidation(事实→观察→心智模型) + 知识图谱
  L4 内化记忆 — KV Cache(高频) + LoRA(深层) [可选]

治理引擎(横切面):
  冲突仲裁 + 时间衰减 + 遗忘曲线 + 隐私分级 + 溯源追踪
"""

from __future__ import annotations

import json
import logging
import tarfile
import threading
import time

# ★ 抑制 ChromaDB 0.6.x telemetry PostHog capture() 签名不兼容的噪音日志
class _ChromaDBTelemetryFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Failed to send telemetry event" in msg:
            return False
        if record.name.startswith("chromadb.telemetry"):
            return False
        return True

_tf = _ChromaDBTelemetryFilter()
for _ln in ("chromadb.telemetry.product.posthog", "chromadb.telemetry"):
    logging.getLogger(_ln).addFilter(_tf)
    logging.getLogger(_ln).setLevel(logging.WARNING)
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider

from omnimem.compression.pipeline import CompressionPipeline
from omnimem.config import OmniMemConfig
from omnimem.core.attachment import build_attachments
from omnimem.core.dedup import SemanticDedupService
from omnimem.core.llm_memory_manager import LLMMemoryManager
from omnimem.core.tool_router import (
    ToolRouter,
    handle_compact,
    handle_reflect,
    handle_detail,
    build_system_prompt,
    run_prefetch,
    run_queue_prefetch,
    l3_recall,
    init_llm_client,
    make_llm_call_fn,
    call_llm_for_reflect,
    retry_index_add,
    retry_retriever_add,
    retry_kg_extract,
    apply_sync_change,
    get_config_schema as _get_config_schema_impl,
    save_config as _save_config_impl,
)
from omnimem.governance.vector_clock import VectorClock
from omnimem.handlers.compat_handler import CompatHandler
from omnimem.handlers.govern import _scan_memory_conflicts as _scan_memory_conflicts_impl
from omnimem.handlers.govern import handle_govern as _handle_govern_impl
from omnimem.handlers.memorize import handle_memorize as _handle_memorize_impl
from omnimem.handlers.recall import handle_recall as _handle_recall_impl
from omnimem.handlers.record_action import handle_record_action as _handle_record_action_impl
from omnimem.handlers.schemas import get_tool_schemas as _get_tool_schemas
from omnimem.utils.security import SecurityValidator
from omnimem.core.memory_monitor import MemoryMonitor
from omnimem.core.distillation import DistillationEngine
from omnimem.core.action_memory import ActionMemoryService
from omnimem.facades import (
    StorageFacade,
    RetrievalFacade,
    GovernanceFacade,
    DeepMemoryFacade,
    SyncFacade,
)

logger = logging.getLogger(__name__)


class OmniMemProvider(MemoryProvider):  # type: ignore[misc]
    """OmniMem: 五层混合记忆系统，实现 Hermes MemoryProvider ABC。

    五层架构:
      L0 感知层  — 主动监控 + 信号检测 + 意图预测
      L1 工作记忆 — CoreBlock(常驻上下文) + Attachment(压缩后状态)
      L2 结构化记忆 — Wing/Room 宫殿导航 + Drawer/Closet 双存储
      L3 深层记忆 — Consolidation(事实→观察→心智模型) + 知识图谱
      L4 内化记忆 — KV Cache(高频) + LoRA(深层) [可选]

    治理引擎(横切面):
      冲突仲裁 + 时间衰减 + 遗忘曲线 + 隐私分级 + 溯源追踪

    暴露给 Agent 的 7 个工具接口:
      omni_memorize — 主动存储记忆
      omni_recall   — 主动检索记忆（RAG/LLM/关键词三种模式）
      omni_compact  — 压缩前准备
      omni_reflect  — L3 深层反思（四步循环 + Disposition 性格修饰）
      omni_govern   — 治理操作（shade/conflict/kv_cache/stats等）
      omni_detail   — 按需拉取记忆细节（lazy provisioning）
      memory        — 兼容内置 memory 工具（add/replace/remove）
    """

    # ─── 类型注解（显式属性，替代 __getattr__ 动态代理） ──────────
    _soul: Any
    _core_block: Any
    _budget: Any
    _wing_room: Any
    _store: Any
    _index: Any
    _md_store: Any
    _retriever: Any
    _context_manager: Any
    _perception: Any
    _feedback: Any
    _prefetch_lock: Any
    _reflect_cache: Any
    _prefetch_executor: Any
    _conflict_resolver: Any
    _temporal_decay: Any
    _forgetting: Any
    _privacy: Any
    _provenance: Any
    _sync_engine: Any
    _vector_clock: Any
    _auditor: Any
    _audit_logger: Any
    _rbac: Any
    _saga: Any
    _bg_executor: Any
    _store_service: Any
    _kv_cache: Any
    _lora_trainer: Any
    _consolidation: Any
    _knowledge_graph: Any
    _reflect_engine: Any

    # ─── MemoryProvider ABC 必须实现 ────────────────────────────

    @property
    def name(self) -> str:
        return "omnimem"

    def __init__(self) -> None:
        self._degraded_mode = False
        self._turn_count = 0
        self._system_prompt_cache_turn = -1
        self._system_prompt_cache_value = ""

    def is_available(self) -> bool:
        core_deps = {"rank_bm25": "rank-bm25", "tiktoken": "tiktoken", "yaml": "pyyaml"}
        for module, pip_name in core_deps.items():
            try:
                __import__(module)
            except ImportError:
                logger.error("OmniMem 核心依赖 %s 缺失，请安装: pip install %s", module, pip_name)
                return False

        optional_deps = {"chromadb": "chromadb", "sentence_transformers": "sentence-transformers"}
        for module, pip_name in optional_deps.items():
            try:
                __import__(module)
            except ImportError:
                logger.warning("OmniMem 可选依赖 %s 缺失，将降级到 BM25-only 模式。安装: pip install %s", module, pip_name)

        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        hermes_home = kwargs.get("hermes_home", str(Path.home() / ".hermes"))
        platform = kwargs.get("platform", "cli")
        agent_context = kwargs.get("agent_context", "primary")

        self._should_write = agent_context == "primary"

        self._data_dir = Path(hermes_home) / "omnimem"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id

        self._config = OmniMemConfig(self._data_dir)

        # ─── 降级模式：跳过向量检索和 ChromaDB，仅 BM25 检索 ───
        if self._degraded_mode:
            logger.warning("OmniMem: 降级模式 — 向量检索和 reranker 不可用，仅 BM25 检索")
            self._init_l1()
            logger.info(
                "OmniMem initialized (degraded): session=%s, platform=%s, data_dir=%s, BM25-only",
                session_id,
                platform,
                self._data_dir,
            )
            return

        # ─── 阶段1: 核心同步初始化（快速返回，让 agent 尽早就绪） ───
        self._init_l1()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._init_store): "store",
                executor.submit(self._init_retrieval): "retrieval",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning("Init %s failed: %s", name, e)

        try:
            self._init_governance_sync_services()
        except Exception as e:
            logger.warning("Init governance_sync_services failed: %s", e)

        logger.info(
            "OmniMem initialized: session=%s, platform=%s, data_dir=%s, L3=enabled, L4=enabled",
            session_id,
            platform,
            self._data_dir,
        )

        self._memory_monitor = MemoryMonitor(
            interval=self._config.get("memory_monitor_interval", 60.0),
            warning_mb=self._config.get("memory_warning_mb", 500.0),
        )
        self._memory_monitor.start()

        # ─── 阶段2: 后台异步预热（重活丢到 worker 线程，不阻塞对话启动） ───
        t_bg = threading.Thread(target=self._background_warmup, daemon=True, name="omnimem_bg_warmup")
        t_bg.start()

        try:
            step_actions = {
                "three_level_index": lambda mid: self._retry_index_add(mid),
                "retriever": lambda mid: self._retry_retriever_add(mid),
                "knowledge_graph": lambda mid: self._retry_kg_extract(mid),
            }
            fixed = self._saga.auto_retry_pending(step_actions)
            if fixed > 0:
                logger.info("OmniMem: auto-retried %d pending saga records", fixed)
        except Exception as e:
            logger.warning("OmniMem: saga auto-retry failed (non-fatal): %s", e)

    def _background_warmup(self) -> None:
        """后台异步预热：L3/L4 初始化 + 数据预热 + 检索引擎预热。

        全部非致命——失败不影响核心记忆功能（只是首次检索可能稍慢）。
        """
        logger.info("OmniMem background warmup: starting...")
        t0 = time.time()

        # L3/L4 并行初始化
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self._init_reflect): "reflect",
                executor.submit(self._init_lora): "lora",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning("BG init %s failed: %s", name, e)

        # 数据预热（索引 + BM25）
        try:
            indexed_entries = self._index.search_l1(limit=2000)
            if indexed_entries:
                self._store.warm_up(indexed_entries[:500])
                rebuilt = self._retriever.rebuild_bm25_from_entries(indexed_entries)
                if rebuilt > 0:
                    logger.info(
                        "OmniMem: warmed up %d entries, rebuilt BM25 with %d entries",
                        min(len(indexed_entries), 500),
                        rebuilt,
                    )
            # ★ OPT: index.db 全量同步 — 直接 SQLite 交叉比对
            try:
                import sqlite3 as _sql
                meta_path = self._store._meta_store._db_path
                index_path = self._index._db_path
                meta_db = _sql.connect(str(meta_path), check_same_thread=False)
                idx_db = _sql.connect(str(index_path), check_same_thread=False)
                # 获取两组 ID
                meta_rows = meta_db.execute("SELECT memory_id, wing, type, room, summary, confidence, privacy, stored_at, content_preview FROM memories").fetchall()
                idx_rows = idx_db.execute("SELECT memory_id FROM memory_index").fetchall()
                meta_ids = {r[0] for r in meta_rows}
                idx_ids = {r[0] for r in idx_rows}
                # 清理 index.db 中失联条目
                stale = idx_ids - meta_ids
                for mid in stale:
                    idx_db.execute("DELETE FROM memory_index WHERE memory_id = ?", (mid,))
                # 补充 index.db 中缺失条目
                missing = meta_ids - idx_ids
                if missing:
                    meta_map = {}
                    for r in meta_rows:
                        mid = r[0]
                        if mid in missing:
                            meta_map[mid] = {
                                'wing': r[1] or 'personal',
                                'hall': r[2] or 'facts',
                                'room': r[3] or 'default',
                                'summary': r[4] or '',
                                'type': r[2] or 'fact',
                                'confidence': r[5] or 3,
                                'privacy': r[6] or 'personal',
                                'stored_at': r[7] or '',
                                'content': r[8] or r[4] or '',
                            }
                    for mid, m in meta_map.items():
                        idx_db.execute(
                            "INSERT INTO memory_index (memory_id, wing, hall, room, summary, content, type, confidence, privacy, scope, stored_at, provenance) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (mid, m['wing'], m['hall'], m['room'], m['summary'], m['content'],
                             m['type'], m['confidence'], m['privacy'], m['wing'], m['stored_at'], '{}'))
                idx_db.commit()
                idx_db.close()
                meta_db.close()
                if stale or missing:
                    logger.info("OmniMem: index.db synced — cleaned %d stale, added %d missing", len(stale), len(missing))
            except Exception as _e:
                logger.warning("OmniMem index.db sync skipped (non-fatal): %s", _e)
        except Exception as e:
            logger.warning("OmniMem warm-up/BM25 rebuild failed (non-fatal): %s", e)

        # 检索引擎预热（SentenceTransformer + ChromaDB）
        try:
            self._retrieval.warmup()
            logger.info("OmniMem: retrieval engine warmup complete")
            try:
                health = self._retriever._check_vector_health()
                vec_count = health.get("vector_count", -1)
                if vec_count == 0 and hasattr(self, '_index') and self._index:
                    indexed_entries = self._index.search_l1(limit=5000)
                    if indexed_entries:
                        logger.warning("OmniMem: vector index empty but %d entries in meta_store, triggering rebuild", len(indexed_entries))
                        result = self._retriever.rebuild_all_from_entries(indexed_entries)
                        logger.info("OmniMem: vector rebuild complete: %s", result)
                        from omnimem.retrieval.vector_store import _emit
                        _emit("[OmniMem] 向量索引已自动重建")
            except Exception as e:
                logger.warning("OmniMem: vector health check failed: %s", e)
        except Exception as e:
            logger.warning("OmniMem retrieval warmup failed (non-fatal): %s", e)

        # ★ P0修复：启动时运行审计+修复，确保 store/index/retriever 一致
        try:
            health = self._auditor.quick_health_check()
            if not health.get("healthy", True):
                audit = self._auditor.run_full_audit(limit=2000)
                if audit.get("total_issues", 0) > 0:
                    fixed = self._auditor.repair(audit)
                    logger.info(
                        "OmniMem startup audit: %d inconsistencies found, %d repaired",
                        audit["total_issues"], fixed,
                    )
        except Exception as e:
            logger.warning("OmniMem startup audit skipped (non-fatal): %s", e)

        elapsed = time.time() - t0
        logger.info("OmniMem background warmup: complete in %.1fs", elapsed)

    def _init_l1(self) -> None:
        self._storage = StorageFacade(self._data_dir, self._config)

    def _init_store(self) -> None:
        self._storage.init_l2()

    def _init_retrieval(self) -> None:
        self._retrieval = RetrievalFacade(self._data_dir, self._config, self._storage)
        if hasattr(self._retrieval.retriever, '_vector_breaker'):
            def _on_circuit_recover():
                try:
                    health = self._retrieval.retriever._check_vector_health()
                    if health.get("vector_count", -1) <= 0:
                        logger.warning("OmniMem: CircuitBreaker recovered but vector still empty, triggering rebuild")
                        if hasattr(self, '_index') and self._index:
                            indexed_entries = self._index.search_l1(limit=5000)
                            if indexed_entries:
                                result = self._retrieval.retriever.rebuild_all_from_entries(indexed_entries)
                                logger.info("OmniMem: vector rebuild after circuit recovery: %s", result)
                except Exception as e:
                    logger.warning("OmniMem: circuit recovery rebuild failed: %s", e)
            self._retrieval.retriever._vector_breaker._on_recover = _on_circuit_recover

    def _init_governance_sync_services(self) -> None:
        self._governance = GovernanceFacade(
            self._data_dir, self._config, self._session_id,
            self._storage, self._retrieval.retriever,
        )
        self._sync = SyncFacade(
            self._data_dir, self._config, self._session_id,
            self._storage, self._retrieval,
        )
        self._sync.bind_provenance(self._governance.provenance)

        self._instance_id = self._governance.instance_id
        self._turn_count = 0
        self._last_save_turn = 0
        self._save_interval = self._config.get("save_interval", 15)
        self._system_prompt_cache_turn = -1
        self._system_prompt_cache_value = ""
        self._init_llm_client()
        # ★ LLM-as-Memory-Manager：初始化 LLM 记忆决策管理器
        self._llm_memory_manager = LLMMemoryManager(
            llm_client=self._llm_client if hasattr(self, "_llm_client") else None,
            config=self._config,
        )
        # ★ OPT: 初始化 MermaidCanvas + CompressionPipeline
        from omnimem.compression.mermaid_canvas import MermaidCanvas
        self._mermaid_canvas = MermaidCanvas(self._data_dir, config=self._config)
        self._compression_pipeline = CompressionPipeline(
            llm_call_fn=self._make_llm_call_fn(),
            config=self._config,
            mermaid_canvas=self._mermaid_canvas,
            session_key=self._session_id,
        )
        self._compat_handler = CompatHandler(
            memorize_fn=self._handle_memorize,
            store=self._store,
            forgetting=self._forgetting,
            extract_core_fact_fn=self._extract_core_fact,
        )
        self._dedup_service = SemanticDedupService(self._store, self._retriever)
        self._action_memory = ActionMemoryService(
            self._store, self._index, self._retriever,
            self._wing_room, self._provenance, self._forgetting,
        )
        self._tool_router = ToolRouter(
            memorize_fn=self._handle_memorize,
            recall_fn=self._handle_recall,
            govern_fn=self._handle_govern,
            reflect_fn=self._handle_reflect,
            compact_fn=self._handle_compact,
            detail_fn=self._handle_detail,
            memory_compat_fn=self._handle_builtin_memory_compat,
            record_action_fn=self._handle_record_action,
        )
        # ★ OPT: 初始化 TraceChain 全链路溯源
        from omnimem.core.trace_chain import TraceChain
        self._trace_chain = TraceChain(self._data_dir)
        # ★ OPT: 初始化 PipelineScheduler（L2/L3 自动调度）
        from omnimem.core.pipeline_scheduler import PipelineScheduler
        self._pipeline_scheduler = PipelineScheduler(
            config=self._config,
            logger=logger,
            bg_executor=self._bg_executor if hasattr(self, '_bg_executor') else None,
            reflect_fn=self._handle_reflect if hasattr(self, '_handle_reflect') else None,
        )

        # ★ 显式属性赋值（替代 __getattr__ 动态代理）
        self._soul = self._storage.soul
        self._core_block = self._storage.core_block
        self._budget = self._storage.budget
        self._wing_room = self._storage.wing_room
        self._store = self._storage.store
        self._index = self._storage.index
        self._md_store = self._storage.md_store
        self._retriever = self._retrieval.retriever
        self._context_manager = self._retrieval.context_manager
        self._perception = self._retrieval.perception
        self._feedback = self._retrieval.feedback
        self._prefetch_lock = self._retrieval.prefetch_lock
        self._reflect_cache = self._retrieval._reflect_cache
        self._prefetch_executor = self._retrieval.prefetch_executor
        self._conflict_resolver = self._governance.conflict_resolver
        self._temporal_decay = self._governance.temporal_decay
        self._forgetting = self._governance.forgetting
        self._privacy = self._governance.privacy
        self._provenance = self._governance.provenance
        self._sync_engine = self._governance.sync_engine
        self._vector_clock = self._governance.vector_clock
        self._auditor = self._governance.auditor
        self._audit_logger = self._governance.audit_logger
        self._rbac = self._governance.rbac
        self._saga = self._sync.saga
        self._bg_executor = self._sync.bg_executor
        self._store_service = self._sync.store_service
        self._kv_cache = self._sync.kv_cache
        self._lora_trainer = self._sync.lora_trainer
        # ★ 质量评估器：检索质量指标计算 + 持久化 + 自动调优建议
        from omnimem.retrieval.quality_eval import RetrievalQualityEvaluator
        self._quality_evaluator = RetrievalQualityEvaluator(
            self._data_dir, config=self._config,
        )

    def _init_reflect(self) -> None:
        self._deep = DeepMemoryFacade(
            self._data_dir, self._config,
            recall_fn=self._l3_recall,
            llm_fn=self._call_llm_for_reflect,
            llm_client=self._llm_client,
        )
        # ★ 显式属性赋值（_deep facade，在 _init_reflect 中初始化后赋值）
        self._consolidation = self._deep.consolidation
        self._knowledge_graph = self._deep.knowledge_graph
        self._reflect_engine = self._deep.reflect_engine

    def _init_distill(self) -> None:
        """初始化 LLM 蒸馏引擎（OPT-1）."""
        self._distillation_engine = DistillationEngine(
            llm_fn=self._call_llm_for_distill,
            store=self._store,
            memorize_fn=self._handle_memorize,
            config=self._config,
        )

    def _init_lora(self) -> None:
        """初始化 L4 内化记忆。"""
        self._sync.init_l4()

    _FACADE_ATTR_MAP: dict[str, tuple[str, str]] = {
        "_soul": ("_storage", "soul"),
        "_core_block": ("_storage", "core_block"),
        "_budget": ("_storage", "budget"),
        "_wing_room": ("_storage", "wing_room"),
        "_store": ("_storage", "store"),
        "_index": ("_storage", "index"),
        "_md_store": ("_storage", "md_store"),
        "_retriever": ("_retrieval", "retriever"),
        "_context_manager": ("_retrieval", "context_manager"),
        "_perception": ("_retrieval", "perception"),
        "_feedback": ("_retrieval", "feedback"),
        "_prefetch_lock": ("_retrieval", "prefetch_lock"),
        "_reflect_cache": ("_retrieval", "_reflect_cache"),
        "_prefetch_executor": ("_retrieval", "_prefetch_executor"),
        "_conflict_resolver": ("_governance", "conflict_resolver"),
        "_temporal_decay": ("_governance", "temporal_decay"),
        "_forgetting": ("_governance", "forgetting"),
        "_privacy": ("_governance", "privacy"),
        "_provenance": ("_governance", "provenance"),
        "_sync_engine": ("_governance", "sync_engine"),
        "_vector_clock": ("_governance", "vector_clock"),
        "_auditor": ("_governance", "auditor"),
        "_audit_logger": ("_governance", "audit_logger"),
        "_rbac": ("_governance", "rbac"),
        "_temporal_kg": ("_governance", "temporal_kg"),
        "_saga": ("_sync", "saga"),
        "_bg_executor": ("_sync", "bg_executor"),
        "_store_service": ("_sync", "store_service"),
        "_kv_cache": ("_sync", "kv_cache"),
        "_lora_trainer": ("_sync", "lora_trainer"),
        "_consolidation": ("_deep", "consolidation"),
        "_knowledge_graph": ("_deep", "knowledge_graph"),
        "_reflect_engine": ("_deep", "reflect_engine"),
    }

    _FACADE_DIRECT_MAP: dict[str, str] = {
        "_dedup": "_dedup_service",
    }

    _FACADE_SETTER_MAP: dict[str, tuple[str, str]] = {
        "_attachments": ("_storage", "attachments"),
        "_prefetch_cache": ("_retrieval", "prefetch_cache"),
    }

    def __getattr__(self, name: str) -> Any:
        if name in self._FACADE_ATTR_MAP:
            logger.warning("OmniMemProvider.__getattr__('%s') is deprecated, use explicit attribute", name)
            facade_attr, sub_attr = self._FACADE_ATTR_MAP[name]
            facade = object.__getattribute__(self, facade_attr)
            return getattr(facade, sub_attr)
        if name in self._FACADE_DIRECT_MAP:
            return object.__getattribute__(self, self._FACADE_DIRECT_MAP[name])
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._FACADE_SETTER_MAP:
            facade_attr, sub_attr = self._FACADE_SETTER_MAP[name]
            facade = object.__getattribute__(self, facade_attr)
            setattr(facade, sub_attr, value)
            return
        object.__setattr__(self, name, value)

    # ─── 异步接口（P2方案五） ───────────────────────────────────

    @property
    def async_provider(self) -> Any:
        """获取异步包装器（延迟初始化）。"""
        if not hasattr(self, "_async_provider"):
            from omnimem.core.async_provider import AsyncOmniMemProvider

            self._async_provider = AsyncOmniMemProvider(self)
        return self._async_provider

    # ─── 上下文注入 ─────────────────────────────────────────────

    def system_prompt_block(self) -> str:
        result, cache_turn, cache_value = build_system_prompt(
            data_dir=str(self._data_dir),
            store=self._store,
            core_block=self._core_block,
            context_manager=self._context_manager,
            config=self._config,
            turn_count=self._turn_count,
            system_prompt_cache_turn=self._system_prompt_cache_turn,
            system_prompt_cache_value=self._system_prompt_cache_value,
            last_query=getattr(self, "_last_query", ""),
        )
        self._system_prompt_cache_turn = cache_turn
        self._system_prompt_cache_value = cache_value
        if hasattr(self, '_retrieval') and hasattr(self._retrieval, 'retriever'):
            try:
                health = self._retrieval.retriever._check_vector_health()
                if health.get("vector_count", -1) <= 0 or health.get("breaker_state") == "open":
                    result += "\n[⚠ OmniMem: 向量检索不可用，已降级到关键词模式]"
            except Exception:
                pass
        return result

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """非阻塞 prefetch：优先返回缓存，超时则返回空字符串。

        优化策略：
        1. enable_prefetch=false → 立即返回空字符串（0ms）
        2. 有缓存 → 立即返回缓存（0ms）
        3. 无缓存 → 带超时搜索（默认5秒，可配置）
        4. 超时/异常 → 返回空字符串，不阻塞对话
        5. 同时触发后台 queue_prefetch 预取下一次查询
        """
        if not self._config.get("enable_prefetch", True):
            return ""

        self._last_query = query

        with self._prefetch_lock:
            cached = self._prefetch_cache
            if cached:
                return cached
            self._prefetch_cache = ""

        prefetch_timeout = self._config.get("prefetch_timeout", 5)
        future = self._prefetch_executor.submit(
            run_prefetch,
            query=query,
            session_id=session_id,
            config=self._config,
            retriever=self._retriever,
            context_manager=self._context_manager,
            kv_cache=self._kv_cache,
            knowledge_graph=self._knowledge_graph,
            temporal_decay=self._temporal_decay,
            privacy=self._privacy,
            prefetch_cache="",
            prefetch_lock=self._prefetch_lock,
            forgetting=self._forgetting,
        )
        try:
            result, new_cache = future.result(timeout=prefetch_timeout)
        except TimeoutError:
            logger.warning(
                "OmniMem prefetch timed out after %ds", prefetch_timeout
            )
            return ""
        except Exception as e:
            logger.warning(
                "OmniMem prefetch failed (non-blocking): %s", e
            )
            self.queue_prefetch(query, session_id=session_id)
            return ""

        with self._prefetch_lock:
            self._prefetch_cache = new_cache
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        def _bg_prefetch() -> None:
            serialized = run_queue_prefetch(
                query=query,
                session_id=session_id,
                config=self._config,
                retriever=self._retriever,
                temporal_decay=self._temporal_decay,
                privacy=self._privacy,
                prefetch_lock=self._prefetch_lock,
            )
            with self._prefetch_lock:
                self._prefetch_cache = serialized

        self._prefetch_executor.submit(_bg_prefetch)

    # ─── 对话同步 ─────────────────────────────────────────────

    # ─── 输入净化：剥离系统注入内容，防止递归膨胀 ───

    @staticmethod
    def _strip_system_injections(text: str) -> str:
        """剥离 prefetch 注入的记忆区块，只保留用户原始输入。

        委托 SecurityValidator 实现，支持 Unicode 归一化和编码绕过检测。
        """
        return SecurityValidator.strip_system_injections(text)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """每轮对话后：感知 → 写入 → 治理。"""
        if not self._should_write:
            return

        # ★ 输入净化：剥离系统注入内容，只对原始用户输入做感知
        clean_user = self._strip_system_injections(user_content)

        # L0 感知
        signals = self._perception.detect_signals(clean_user, assistant_content)

        # 信号驱动的记忆写入
        # ★ 信号互斥：correction > reinforcement > fact
        # 避免同一条信息被多个信号触发重复写入
        if signals.has_correction:
            self._store_service.store_correction(signals, user_content)
        elif signals.has_reinforcement:
            self._store_service.store_reinforcement(signals, user_content)
        elif signals.should_memorize:
            self._store_service.store_fact(signals, user_content)

        # 定期自动存档
        self._turn_count += 1
        self._store_service.turn_count = self._turn_count
        if self._turn_count - self._store_service.last_save_turn >= self._save_interval:
            self._store_service.auto_checkpoint(user_content, self._save_interval)
            self._last_save_turn = self._store_service.last_save_turn

        # OPT-1: 定期 LLM 蒸馏 — 将 auto-captured raw facts 提炼为高质量记忆
        if self._config.get("distill_enabled", True):
            distill = getattr(self, "_distillation_engine", None)
            if distill is None:
                self._init_distill()
                distill = getattr(self, "_distillation_engine", None)
            distill_interval = self._config.get("distill_interval", 15)
            if distill and self._turn_count % distill_interval == 0:
                self._bg_executor.submit(
                    lambda: distill.distill_recent_facts(turn_count=self._turn_count)
                )
                # ★ OPT: 蒸馏完成后延迟触发 L2 场景归纳
                if hasattr(self, '_pipeline_scheduler') and self._pipeline_scheduler:
                    self._pipeline_scheduler.schedule_l2_after_l1(self._session_id)

        # ★ P0方案二：统一后台任务执行器替代每轮新建 threading.Thread
        self._bg_executor.submit(
            self._retriever.index_update,
            user_content,
            assistant_content,
        )

    # ─── 工具暴露 ─────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """OmniMem 暴露 7 个工具给 Agent — 委托到 handlers/schemas.py。"""
        return _get_tool_schemas()

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        try:
            return self._tool_router.route(tool_name, args)
        except Exception as e:
            logger.error("OmniMem tool %s failed: %s", tool_name, e)
            return json.dumps({"error": str(e)})

    # ─── 扩展 Hooks ─────────────────────────────────────────────

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        """每轮开始：配置热重载 + 意图预测 + 预加载 + 分布式同步。"""
        if not self._should_write:
            return
        # ★ 周期性热重载配置（每 10 轮检查一次文件变更）
        if turn_number % 10 == 0:
            self._config.reload()
        # ★ P3：每 15 轮自动触发 Consolidation（长会话不积压）
        if turn_number % 15 == 0 and self._consolidation:
            try:
                processed = self._consolidation.process_pending()
                if processed > 0:
                    logger.info("OmniMem auto-consolidation: %d memories", processed)
            except Exception as e:
                logger.warning("OmniMem auto-consolidation failed: %s", e)
        # ★ P1方案四：每 20 轮更新检索来源权重（基于反馈统计）
        if turn_number % 20 == 0 and hasattr(self, "_feedback") and self._feedback:
            try:
                weights = self._feedback.get_source_weights(window=100)
                if weights:
                    self._retriever.set_source_weights(weights)
                    logger.info("Updated source weights from feedback: %s", weights)
            except Exception as e:
                logger.warning("Feedback weight update failed: %s", e)
        # ★ 分布式同步：每 5 轮从其他实例拉取变更（changelog 模式）
        if (
            turn_number % 5 == 0
            and hasattr(self, "_sync_engine")
            and self._sync_engine
            and self._sync_engine._config.mode == "changelog"
        ):
            try:
                applied = self._sync_engine.sync_from_others(
                    apply_fn=self._apply_sync_change,
                    get_local_fn=lambda mid: self._store.get(mid),
                )
                if applied > 0:
                    logger.info("OmniMem sync: applied %d changes from other instances", applied)
            except Exception as e:
                logger.warning("OmniMem sync failed: %s", e)
        # ★ 质量评估自动调优：每 30 轮检查质量趋势并应用调优建议
        if turn_number % 30 == 0 and hasattr(self, "_quality_evaluator") and self._quality_evaluator:
            try:
                suggestions = self._quality_evaluator.get_auto_tune_suggestions()
                for s in suggestions.get("suggestions", []):
                    param = s.get("parameter", "")
                    action = s.get("action", "")
                    desc = s.get("description", "")
                    suggested_value = s.get("suggested_value", "")
                    logger.info("质量调优建议: %s → %s (%s)", param, action, desc)
                    if param == "min_rrf" and hasattr(self, "_retriever") and self._retriever:
                        try:
                            new_val = float(suggested_value)
                            self._retriever._rrf._min_rrf = new_val
                            logger.info("已应用调优: min_rrf = %.3f", new_val)
                        except (ValueError, AttributeError) as e:
                            logger.warning("应用 min_rrf 调优失败: %s", e)
                    elif param == "rrf_vector_weight" and hasattr(self, "_retriever") and self._retriever:
                        try:
                            new_val = float(suggested_value)
                            if "vector" in self._retriever._channels:
                                retriever, _ = self._retriever._channels["vector"]
                                self._retriever._channels["vector"] = (retriever, new_val)
                                logger.info("已应用调优: vector_weight = %.1f", new_val)
                        except (ValueError, AttributeError) as e:
                            logger.warning("应用 vector_weight 调优失败: %s", e)
                    elif param == "rrf_bm25_weight" and hasattr(self, "_retriever") and self._retriever:
                        try:
                            new_val = float(suggested_value)
                            if "bm25" in self._retriever._channels:
                                retriever, _ = self._retriever._channels["bm25"]
                                self._retriever._channels["bm25"] = (retriever, new_val)
                                logger.info("已应用调优: bm25_weight = %.1f", new_val)
                        except (ValueError, AttributeError) as e:
                            logger.warning("应用 bm25_weight 调优失败: %s", e)
            except Exception as e:
                logger.warning("质量评估自动调优失败: %s", e)
        predicted = self._perception.predict_intent(message)
        if predicted:
            self.queue_prefetch(predicted)

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """会话结束：Consolidation + 治理归档。"""
        if not self._should_write:
            return

        # 关闭后台预取线程池（优雅释放资源）
        if hasattr(self, "_prefetch_executor") and self._prefetch_executor:
            self._prefetch_executor.shutdown(wait=False)

        # 1. 从完整对话中提取遗漏的记忆
        self._store_service.extract_session_memories(
            messages,
            self._strip_system_injections,
            self._should_store,
            lambda args: self._handle_memorize(args),
        )

        # 2. 治理：遗忘曲线归档
        archived_ids = self._forgetting.run_archive_cycle()

        # ★ P3修复：归档后从检索索引+文件系统中清理，防止残留
        if archived_ids > 0:
            try:
                archived_list = self._forgetting.get_archived_ids(limit=500)
                for mid in archived_list:
                    self._retriever.delete(mid)
                    try:
                        self._index.delete(mid)
                    except Exception:
                        pass
                    try:
                        self._store.delete(mid)
                    except Exception:
                        pass
                logger.info("OmniMem: cleaned %d archived entries from retrieval indices + file system", len(archived_list))
            except Exception as e:
                logger.warning("OmniMem archive cleanup skipped: %s", e)

        # 3. L3 Consolidation: 处理待升华的记忆
        if self._consolidation:
            processed = self._consolidation.process_pending()
            if processed > 0:
                logger.info("OmniMem consolidation: processed %d memories", processed)

        # 4. L4: 将心智模型提交到 LoRA 训练队列
        if self._consolidation and self._lora_trainer:
            try:
                models = self._consolidation.get_mental_models(limit=20)
                if models:
                    self._lora_trainer.submit_training_data(models, shade="default")
                    logger.info(
                        "OmniMem L4: submitted %d mental models for LoRA training", len(models)
                    )
            except Exception as e:
                logger.warning("OmniMem L4 submit failed: %s", e)

        # 5. 刷新存储缓冲与索引
        self._store.flush()
        self._retriever.flush()

        # ★ P0方案六：治理巡检（每 10 轮执行一次一致性审计）
        if self._turn_count % self._config.get("audit_interval_turns", 50) == 0 and hasattr(self, "_auditor") and self._auditor:
            try:
                health = self._auditor.quick_health_check()
                if not health["healthy"]:
                    audit = self._auditor.run_full_audit(limit=1000)
                    if audit["total_issues"] > 0:
                        fixed = self._auditor.repair(audit)
                        logger.info(
                            "OmniMem governance audit: %d issues found, %d fixed",
                            audit["total_issues"],
                            fixed,
                        )
            except Exception as e:
                logger.warning("Governance audit failed: %s", e)

        # ★ P0方案二：Saga pending 重试（会话结束前补偿未完成的索引写入）
        if self._saga.get_pending():
            fixed = self._saga.retry_pending(
                {
                    "three_level_index": lambda mid: self._retry_index_add(mid),
                    "retriever": lambda mid: self._retry_retriever_add(mid),
                    "knowledge_graph": lambda mid: self._retry_kg_extract(mid),
                }
            )
            if fixed > 0:
                logger.info("OmniMem saga retry: fixed %d pending records", fixed)

        # ★ P0方案二：关闭统一后台任务执行器
        # ★ OPT: 刷新 PipelineScheduler（L2/L3 未完成任务）
        if hasattr(self, '_pipeline_scheduler') and self._pipeline_scheduler:
            self._pipeline_scheduler.flush_session(self._session_id)
        if hasattr(self, "_bg_executor") and self._bg_executor:
            self._bg_executor.shutdown(wait=True)

        # ★ 自动备份：检查距上次备份是否超过 backup_interval_hours
        try:
            backup_interval_hours = self._config.get("backup_interval_hours", 24)
            if time.time() - self._last_backup_time >= backup_interval_hours * 3600:
                backup_path, backup_size = self._create_backup()
                backup_max_copies = self._config.get("backup_max_copies", 3)
                self._cleanup_old_backups(backup_max_copies)
        except Exception as e:
            logger.warning("OmniMem 自动备份失败: %s", e)

        logger.info("OmniMem session end: processed %d messages", len(messages))

    def _create_backup(self) -> tuple[str, int]:
        """将 omnimem 数据目录打包为 tar.gz 备份。

        备份路径: ~/.hermes/omnimem.bak/YYYYMMDD_HHMMSS.tar.gz
        返回: (备份路径, 备份字节数)
        """
        from datetime import datetime

        backup_dir = Path.home() / ".hermes" / "omnimem.bak"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{timestamp}.tar.gz"

        with tarfile.open(str(backup_path), "w:gz") as tar:
            tar.add(str(self._data_dir), arcname=self._data_dir.name)

        size = backup_path.stat().st_size
        self._last_backup_time = time.time()
        logger.info("OmniMem 备份完成: %s (%.1f KB)", backup_path, size / 1024)
        return str(backup_path), size

    def _cleanup_old_backups(self, max_copies: int = 3) -> None:
        """清理旧备份，保留最近 max_copies 个。"""
        backup_dir = Path.home() / ".hermes" / "omnimem.bak"
        if not backup_dir.exists():
            return

        backups = sorted(backup_dir.glob("*.tar.gz"))
        if len(backups) > max_copies:
            for old in backups[:-max_copies]:
                old.unlink()
                logger.info("OmniMem 清理旧备份: %s", old)

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """压缩前：构建 Attachment + 紧急保存。"""
        self._store.flush()
        saved_context = self._store_service.emergency_save(messages)

        attachments = build_attachments(messages)

        parts = []
        if saved_context:
            parts.append(saved_context)
        if attachments:
            att_text = "\n".join(f"[{a.kind}] {a.title}: {a.body[:200]}" for a in attachments)
            parts.append(f"### Pre-Compression Attachments\n{att_text}")

        result = "\n\n".join(parts)

        if self._config.get("enable_compression", False) and result:
            result = self._compression_pipeline.compress(result)

        return result

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """内置记忆写入时：冲突检测。"""
        if action == "add":
            conflict = self._conflict_resolver.check(content)
            if conflict.has_conflict:
                logger.warning(
                    "OmniMem: conflict detected with existing memory: %s",
                    conflict.existing_memory,
                )

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any
    ) -> None:
        """子 Agent 完成时：记录过程记忆。"""
        if not self._should_write:
            return
        self._store_service.store_delegation(task, result, child_session_id)

    # ─── 配置 ─────────────────────────────────────────────────

    def get_config_schema(self) -> list[dict[str, Any]]:
        return _get_config_schema_impl()

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        _save_config_impl(values, hermes_home)

    def shutdown(self) -> None:
        """清理：刷新所有缓冲到磁盘。

        关闭顺序: retriever → md_store → index → knowledge_graph →
                  consolidation → reflect_engine → kv_cache → lora_trainer →
                  provenance → sync_engine → forgetting → executors
        """
        if hasattr(self, "_memory_monitor") and self._memory_monitor:
            self._memory_monitor.stop()
        if hasattr(self, "_feedback") and self._feedback:
            self._feedback.close()
        if hasattr(self, "_prefetch_executor") and self._prefetch_executor:
            self._prefetch_executor.shutdown(wait=False)
        if hasattr(self, "_bg_executor") and self._bg_executor:
            self._bg_executor.shutdown(wait=True)

        # 1. 存储层
        self._store.flush()
        self._retriever.flush()
        self._md_store.flush()
        self._index.close()
        if hasattr(self, "_perception") and self._perception:
            self._perception.close()
        if self._knowledge_graph:
            self._knowledge_graph.close()
        if self._consolidation:
            self._consolidation.close()
        if hasattr(self, "_reflect_engine") and self._reflect_engine:
            self._reflect_engine.close()
        if hasattr(self, "_kv_cache") and self._kv_cache:
            self._kv_cache.close()
        if hasattr(self, "_lora_trainer") and self._lora_trainer:
            self._lora_trainer.close()
        if hasattr(self, "_provenance") and self._provenance:
            self._provenance.close()
        if hasattr(self, "_sync_engine") and self._sync_engine:
            self._sync_engine.close()
        # OPT-2: 关闭 LLM 客户端
        if hasattr(self, "_llm_client") and self._llm_client:
            self._llm_client.close()
        if hasattr(self, "_distill_llm_client") and self._distill_llm_client:
            self._distill_llm_client.close()
        if hasattr(self, "_distillation_engine") and self._distillation_engine:
            self._distillation_engine.close()
        self._forgetting.close()
        if hasattr(self, "_quality_evaluator") and self._quality_evaluator:
            self._quality_evaluator.close()
        logger.info("OmniMem shutdown complete")

    # ─── 工具实现（委托到 handlers 子模块） ─────────────────────

    def get_next_vc(self) -> VectorClock:
        """获取下一个向量时钟值（递增当前实例计数器）。"""
        self._vector_clock.increment(self._instance_id)
        return self._vector_clock  # type: ignore[no-any-return]

    def _apply_sync_change(self, change: dict[str, Any]) -> bool:
        return apply_sync_change(change, self._store, self._index, self._retriever, self._forgetting)

    def _handle_memorize(self, args: dict[str, Any]) -> str:
        """委托到 handlers/memorize.py。"""
        llm_mgr = getattr(self, "_llm_memory_manager", None)
        return _handle_memorize_impl(self, args, llm_memory_manager=llm_mgr)

    # ─── Saga 重试辅助方法 ────────────────────────────────────

    def _retry_index_add(self, memory_id: str) -> None:
        retry_index_add(memory_id, self._store, self._index)

    def _retry_retriever_add(self, memory_id: str) -> None:
        retry_retriever_add(memory_id, self._store, self._retriever)

    def _retry_kg_extract(self, memory_id: str) -> None:
        retry_kg_extract(memory_id, self._store, self._knowledge_graph)

    def _handle_recall(self, args: dict[str, Any]) -> str:
        """委托到 handlers/recall.py。"""
        result = _handle_recall_impl(self, args)
        # ★ 记录反馈：recall 返回的候选列表
        if hasattr(self, "_feedback") and self._feedback:
            try:
                data = json.loads(result)
                if data.get("status") == "found":
                    self._feedback.record_shown(
                        query=args.get("query", ""),
                        candidates=data.get("memories", []),
                    )
            except Exception as e:
                logger.warning("Feedback recording failed: %s", e)
        # ★ 遗忘曲线：召回命中时记录访问（驱动 recall_count 递增 + 重置遗忘计时器）
        try:
            data = json.loads(result)
            if data.get("status") == "found":
                for mem in data.get("memories", []):
                    mid = mem.get("memory_id", "")
                    if mid:
                        self._forgetting.record_access(mid)
        except Exception as e:
            logger.warning("Record access for forgetting curve failed: %s", e)
        return result

    def _handle_govern(self, args: dict[str, Any]) -> str:
        """委托到 handlers/govern.py。"""
        return _handle_govern_impl(self, args)

    def _scan_memory_conflicts(self) -> list[dict[str, Any]]:
        """委托到 handlers/govern.py。"""
        return _scan_memory_conflicts_impl(self)

    def _handle_compact(self, args: dict[str, Any]) -> str:
        return handle_compact(args)

    def _handle_reflect(self, args: dict[str, Any]) -> str:
        return handle_reflect(args, self._consolidation, self._reflect_engine)

    # ─── omni_detail：按需拉取记忆细节 ─────────────────────────

    def _handle_detail(self, args: dict[str, Any]) -> str:
        return handle_detail(
            args,
            context_manager=self._context_manager,
            store=self._store,
            forgetting=self._forgetting,
            feedback=self._feedback if hasattr(self, "_feedback") else None,
            turn_count=self._turn_count,
            last_query=getattr(self, "_last_query", ""),
            trace_chain=getattr(self, "_trace_chain", None),
        )

    def _handle_builtin_memory_compat(self, args: dict[str, Any]) -> str:
        return self._compat_handler.handle(args)

    def _handle_record_action(self, args: dict[str, Any]) -> str:
        return _handle_record_action_impl(self, args)

    # ─── 反递归防护 ─────────────────────────────────────────────

    @staticmethod
    def _should_store(content: str) -> bool:
        """判断内容是否值得存储，过滤系统注入和递归内容。

        委托 SecurityValidator 实现，支持 Unicode 归一化和编码绕过检测。
        所有存储路径都应经过此检查。
        """
        should_store, reason = SecurityValidator.should_store(content)
        if not should_store and reason:
            logger.warning("SecurityValidator._should_store blocked: %s", reason)
        return should_store

    # ─── 语义去重 ─────────────────────────────────────────────

    def _extract_core_fact(self, text: str) -> str:
        """从原文中提取精简核心事实（委托给感知引擎）。"""
        return str(self._perception._extract_core_fact(text))

    @staticmethod
    def _compute_text_similarity(text_a: str, text_b: str) -> float:
        return SemanticDedupService.compute_text_similarity(text_a, text_b)

    def _semantic_dedup(
        self, content: str, memory_type: str, candidates: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        return self._dedup.semantic_dedup(content, memory_type, candidates)

    # ─── 内部辅助方法 ─────────────────────────────────────────────

    def _unified_candidate_search(self, content: str) -> list[dict[str, Any]]:
        return self._dedup.unified_candidate_search(content)

    def _search_candidates(self, content: str) -> list[dict[str, Any]]:
        return self._dedup.search_candidates(content)

    # ─── 内部辅助方法 ─────────────────────────────────────────

    def _l3_recall(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return l3_recall(query, self._retriever, self._store, limit)

    _REFLECT_CACHE_TTL = 60.0

    def _init_llm_client(self) -> None:
        self._llm_client = init_llm_client(self._config)
        # ★ R25修复ARCH-1：若 LLM 客户端凭证为空，尝试从 Hermes 主配置获取
        if self._llm_client and not getattr(self._llm_client, "_api_key", "").strip():
            try:
                from omnimem.utils.llm_client import AsyncLLMClient
                hermes_creds = AsyncLLMClient.load_credentials_from_hermes_config()
                if hermes_creds.get("api_key") and hermes_creds.get("base_url"):
                    logger.info("OmniMem: using Hermes main config LLM credentials for Reflect")
                    self._llm_client = AsyncLLMClient(
                        api_key=hermes_creds["api_key"],
                        base_url=hermes_creds["base_url"],
                        model=hermes_creds.get("model", "glm-5.1"),
                        max_concurrent=3,
                        timeout=30.0,
                        cache_ttl=self._REFLECT_CACHE_TTL,
                    )
            except Exception as e:
                logger.warning("OmniMem: failed to load Hermes main config LLM credentials: %s", e)

    def _make_llm_call_fn(self):
        return make_llm_call_fn(self._llm_client)

    def _call_llm_for_reflect(self, prompt: str, system: str, max_tokens: int = 800) -> str | None:
        return call_llm_for_reflect(
            prompt, system,
            llm_client=self._llm_client if hasattr(self, "_llm_client") else None,
            reflect_cache=self._reflect_cache,
            max_tokens=max_tokens,
        )

    def _call_llm_for_distill(
        self, prompt: str, system: str, max_tokens: int = 600, model: str | None = None
    ) -> str | None:
        """蒸馏引擎的 LLM 调用入口，支持自定义模型。

        Args:
            prompt: 蒸馏 prompt
            system: 系统提示
            max_tokens: 最大输出 token
            model: 自定义模型名（None=使用主模型）
        """
        client = None
        if model:
            # 为蒸馏模型创建/复用独立客户端
            if not hasattr(self, "_distill_llm_client"):
                self._init_distill_llm_client(model)
            client = getattr(self, "_distill_llm_client", self._llm_client)
        else:
            client = self._llm_client if hasattr(self, "_llm_client") else None

        if not client:
            return None

        try:
            result = client.call_sync(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return result.content if result and result.content else None
        except Exception as e:
            logger.warning("Distill LLM call failed: %s", e)
            return None

    def _init_distill_llm_client(self, model: str) -> None:
        """为蒸馏任务创建专用 LLM 客户端。"""
        from omnimem.utils.llm_client import AsyncLLMClient

        # 使用与主客户端相同的凭证
        if hasattr(self, "_llm_client") and self._llm_client:
            creds_key = getattr(self._llm_client, "_api_key", "")
            creds_url = getattr(self._llm_client, "_base_url", "")
        else:
            creds = AsyncLLMClient.load_credentials_from_hermes_config()
            creds_key = creds.get("api_key", "")
            creds_url = creds.get("base_url", "")

        if creds_key and creds_url:
            self._distill_llm_client = AsyncLLMClient(
                api_key=creds_key,
                base_url=creds_url,
                model=model,
                max_concurrent=1,  # 蒸馏是低频后台任务，1并发足够
                timeout=30.0,
                cache_ttl=0.0,  # 蒸馏不缓存
            )
            logger.info("Distill LLM client created: model=%s", model)
