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
from typing import Any

from agent.memory_provider import MemoryProvider
from omnimem.compat.provider_proxy import ProviderProxyMixin
from omnimem.core.provider_initializer import ProviderInitializerMixin
from omnimem.core.provider_lifecycle import ProviderLifecycleMixin
from omnimem.core.provider_middleware import ProviderMiddlewareMixin
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
    apply_sync_change,
    handle_compact,
    handle_detail,
    handle_reflect,
    l3_recall,
    retry_index_add,
    retry_kg_extract,
    retry_retriever_add,
    run_prefetch,
    run_queue_prefetch,
)
from omnimem.core.tool_router import (
    get_config_schema as _get_config_schema_impl,
)
from omnimem.core.tool_router import (
    save_config as _save_config_impl,
)
from omnimem.governance.vector_clock import VectorClock
from omnimem.handlers.govern import _scan_memory_conflicts as _scan_memory_conflicts_impl
from omnimem.handlers.govern import handle_govern as _handle_govern_impl
from omnimem.handlers.memorize import handle_memorize as _handle_memorize_impl
from omnimem.handlers.recall import handle_recall as _handle_recall_impl
from omnimem.handlers.record_action import handle_record_action as _handle_record_action_impl

logger = logging.getLogger(__name__)


# ★ 抑制 ChromaDB 0.4.x - 0.7.x telemetry PostHog capture() 签名不兼容的噪音日志
# 覆盖已知噪音模式：Failed to send telemetry event / PostHog / capture / telemetry
class _ChromaDBTelemetryFilter(logging.Filter):
    """抑制 ChromaDB telemetry 噪音日志，兼容 0.4.x - 0.7.x。

    过滤策略：
      1. logger 名称前缀匹配：chromadb.telemetry.* 全部抑制
      2. 消息内容匹配：包含已知噪音关键词（PostHog/capture/telemetry）的记录抑制
    """

    # 已知 telemetry 噪音关键词（消息内容匹配）
    _NOISE_KEYWORDS = (
        "Failed to send telemetry event",
        "PostHog",
        "capture",
        "telemetry",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        # 1. logger 名称前缀匹配：chromadb.telemetry.* 全部抑制
        if record.name.startswith("chromadb.telemetry"):
            return False
        # 2. 消息内容匹配：检查已知噪音关键词
        msg = record.getMessage()
        for keyword in self._NOISE_KEYWORDS:
            if keyword in msg:
                return False
        return True


_tf = _ChromaDBTelemetryFilter()
# 覆盖 0.4.x - 0.7.x 已知的 telemetry logger 名称
for _ln in (
    "chromadb.telemetry.product.posthog",
    "chromadb.telemetry.product",
    "chromadb.telemetry",
):
    logging.getLogger(_ln).addFilter(_tf)
    logging.getLogger(_ln).setLevel(logging.WARNING)


class OmniMemProvider(
    ProviderProxyMixin,
    ProviderMiddlewareMixin,
    ProviderLifecycleMixin,
    ProviderInitializerMixin,
    MemoryProvider,  # type: ignore[misc]
):
    __doc__ = f"""OmniMem: 五层混合记忆系统，实现 Hermes MemoryProvider ABC。

    五层架构:
      L0 感知层  — 主动监控 + 信号检测 + 意图预测
      L1 工作记忆 — CoreBlock(常驻上下文) + Attachment(压缩后状态)
      L2 结构化记忆 — Wing/Room 宫殿导航 + Drawer/Closet 双存储
      L3 深层记忆 — Consolidation(事实→观察→心智模型) + 知识图谱
      L4 内化记忆 — KV Cache(高频) + LoRA(深层) [可选]

    治理引擎(横切面):
      冲突仲裁 + 时间衰减 + 遗忘曲线 + 隐私分级 + 溯源追踪

    暴露给 Agent 的 8 个工具接口:
      {OMNI_MEMORIZE}      — 主动存储记忆
      {OMNI_RECALL}        — 主动检索记忆（RAG/LLM/关键词三种模式）
      {OMNI_COMPACT}       — 压缩前准备
      {OMNI_REFLECT}       — L3 深层反思（四步循环 + Disposition 性格修饰）
      {OMNI_GOVERN}        — 治理操作（shade/conflict/kv_cache/stats等）
      {OMNI_DETAIL}        — 按需拉取记忆细节（lazy provisioning）
      {OMNI_RECORD_ACTION} — 记录 agent 动作（决策、子 agent、错误处理等）
      {MEMORY_COMPAT}      — 兼容内置 memory 工具（add/replace/remove）
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
    _temporal_kg: Any
    _saga: Any
    _bg_executor: Any
    _store_service: Any
    _kv_cache: Any
    _lora_trainer: Any
    _consolidation: Any
    _knowledge_graph: Any
    _reflect_engine: Any

    _REFLECT_CACHE_TTL = 60.0

    # ─── MemoryProvider ABC 必须实现 ────────────────────────────

    @property
    def name(self) -> str:
        return "omnimem"

    # ─── 配置 ─────────────────────────────────────────────────

    def get_config_schema(self) -> list[dict[str, Any]]:
        return _get_config_schema_impl()

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        _save_config_impl(values, hermes_home)

    # ─── Prefetch 中间件 ───────────────────────────────────────

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
            cached = self._retrieval.prefetch_cache
            if cached:
                return cached
            self._retrieval.prefetch_cache = ""

        prefetch_timeout = self._config.get("prefetch_timeout", 5)
        max_retries = self._config.get("prefetch_max_retries", 1)
        result = ""
        new_cache = ""
        try:
            for attempt in range(max_retries + 1):
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
                    break
                except TimeoutError:
                    if attempt < max_retries:
                        logger.debug("OmniMem prefetch timed out after %ds (attempt %d/%d), retrying",
                                     prefetch_timeout, attempt + 1, max_retries + 1)
                        continue
                    logger.warning("OmniMem prefetch timed out after %ds (all %d attempts)",
                                   prefetch_timeout, max_retries + 1)
                    return ""
        except Exception as e:
            logger.warning("OmniMem prefetch failed (non-blocking): %s", e)
            self.queue_prefetch(query, session_id=session_id)
            return ""

        with self._prefetch_lock:
            self._retrieval.prefetch_cache = new_cache
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
                self._retrieval.prefetch_cache = serialized

        self._prefetch_executor.submit(_bg_prefetch)

    # ─── 工具实现（委托到 handlers 子模块） ─────────────────────

    def get_next_vc(self) -> "VectorClock | None":
        """获取下一个向量时钟值（递增当前实例计数器）。

        单机模式（sync_mode=none）下返回 None，跳过递增操作。
        """
        if self._vector_clock is None:
            return None
        self._vector_clock.increment(self._instance_id)
        return self._vector_clock  # type: ignore[no-any-return]

    def _apply_sync_change(self, change: dict[str, Any]) -> bool:
        return apply_sync_change(
            change, self._store, self._index, self._retriever, self._forgetting
        )

    def _retry_index_add(self, memory_id: str) -> None:
        retry_index_add(memory_id, self._store, self._index)

    def _retry_retriever_add(self, memory_id: str) -> None:
        retry_retriever_add(memory_id, self._store, self._retriever)

    def _retry_kg_extract(self, memory_id: str) -> None:
        retry_kg_extract(memory_id, self._store, self._knowledge_graph)

    def _handle_memorize(self, args: dict[str, Any]) -> str:
        """委托到 handlers/memorize.py。"""
        llm_mgr = getattr(self, "_llm_memory_manager", None)
        return _handle_memorize_impl(self, args, llm_memory_manager=llm_mgr)

    def _handle_recall(self, args: dict[str, Any]) -> str:
        """委托到 handlers/recall.py。"""
        result = _handle_recall_impl(self, args)
        # 记录反馈：recall 返回的候选列表
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
        # 遗忘曲线：召回命中时记录访问
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
        result = handle_reflect(args, self._consolidation, self._reflect_engine)
        # 标记已反思（更新冷却计时器）
        if self._session_manager:
            self._session_manager.mark_reflected()
        return result

    def evaluate_auto_reflection(self) -> dict[str, Any]:
        """评估是否应自动触发反思。供对话循环调用。"""
        if not self._session_manager:
            return {"should_reflect": False, "reason": "无会话管理器"}
        signal = self._session_manager.evaluate_reflection()
        return {
            "should_reflect": signal.should_reflect,
            "score": signal.score,
            "factors": signal.factors,
            "query_hint": signal.query_hint,
            "reason": signal.reason,
        }

    def record_tool_call_for_reflection(self, tool_name: str) -> None:
        """记录工具调用到反思触发器。"""
        if self._session_manager:
            self._session_manager.record_tool_call(tool_name)

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

    # ─── 内部辅助方法 ─────────────────────────────────────────────

    def _l3_recall(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return l3_recall(query, self._retriever, self._store, limit)
