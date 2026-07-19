from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

# 以下函数已迁移到独立模块，此处 re-export 以保持向后兼容
from omnimem.core.llm_initializer import _REFLECT_CACHE_TTL
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

logger = logging.getLogger(__name__)


class ToolRouter:
    def __init__(
        self,
        memorize_fn: Callable[[dict[str, Any]], str],
        recall_fn: Callable[[dict[str, Any]], str],
        govern_fn: Callable[[dict[str, Any]], str],
        reflect_fn: Callable[[dict[str, Any]], str],
        compact_fn: Callable[[dict[str, Any]], str],
        detail_fn: Callable[[dict[str, Any]], str],
        memory_compat_fn: Callable[[dict[str, Any]], str],
        record_action_fn: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        self._routes: dict[str, Callable[[dict[str, Any]], str]] = {
            OMNI_MEMORIZE: memorize_fn,
            OMNI_RECALL: recall_fn,
            OMNI_GOVERN: govern_fn,
            OMNI_REFLECT: reflect_fn,
            OMNI_COMPACT: compact_fn,
            OMNI_DETAIL: detail_fn,
            MEMORY_COMPAT: memory_compat_fn,
        }
        if record_action_fn is not None:
            self._routes[OMNI_RECORD_ACTION] = record_action_fn

    def route(self, tool_name: str, args: dict[str, Any]) -> str:
        handler = self._routes.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        return handler(args)

    def get_tool_names(self) -> list[str]:
        return list(self._routes.keys())


def handle_compact(args: dict[str, Any]) -> str:
    budget = args.get("budget", 4000)
    return json.dumps(
        {
            "status": "ready",
            "budget": budget,
            "message": (
                "OmniMem will save context before compaction via on_pre_compress. "
                "Trigger compaction normally — OmniMem hooks handle the rest."
            ),
        }
    )


def handle_reflect(
    args: dict[str, Any],
    consolidation: Any,
    reflect_engine: Any,
) -> str:
    query = args["query"]
    disposition = args.get("disposition")

    if consolidation and consolidation.pending_count > 0:
        consolidation.process_pending()

    if reflect_engine is None:
        # Lazy-init: SDK 模式下未传入 reflect_engine
        try:
            from pathlib import Path

            from omnimem.config import OmniMemConfig
            from omnimem.deep.consolidation import ConsolidationEngine
            from omnimem.deep.reflect import ReflectEngine
            data_dir = Path.home() / ".hermes" / "omnimem"
            cfg = OmniMemConfig(data_dir)
            cons = ConsolidationEngine(data_dir / "deep", cfg)
            reflect_engine = ReflectEngine(data_dir / "deep", consolidation_engine=cons)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"ReflectEngine init failed: {e}"}, ensure_ascii=False)

    result = reflect_engine.reflect(
        query=query,
        disposition=disposition,
    )
    return json.dumps(
        {
            "status": "reflected",
            "query": query,
            "observation": result.observation,
            "mental_model": result.mental_model,
            "confidence": result.confidence,
            "reflection_depth": result.reflection_depth,
            "disposition_used": result.disposition_used,
        },
        ensure_ascii=False,
    )


def handle_detail(
    args: dict[str, Any],
    context_manager: Any,
    store: Any,
    forgetting: Any,
    feedback: Any,
    turn_count: int,
    last_query: str,
    trace_chain: Any = None,  # ★ OPT: TraceChain 实例，用于 drill_down
) -> str:
    action = args.get("action", "list")

    if action == "list":
        items = context_manager.get_injected_items()
        if items:
            items = [
                item
                for item in items
                if item.get("memory_id") and store.get(item["memory_id"])
            ]
        if not items:
            return json.dumps(
                {
                    "status": "empty",
                    "message": "No memories injected this turn.",
                }
            )
        return json.dumps(
            {
                "status": "ok",
                "count": len(items),
                "memories": items,
            },
            ensure_ascii=False,
        )

    elif action == "get":
        memory_id = args.get("memory_id", "")
        if not memory_id:
            return json.dumps(
                {
                    "status": "error",
                    "message": "memory_id is required for action='get'",
                }
            )
        result = context_manager.get_detail_for(memory_id, store)
        if result.get("status") == "found" and forgetting:
            stage = forgetting.get_stage(memory_id)
            result["archived"] = stage in ("archived", "forgotten")
        if feedback and result.get("status") == "found":
            feedback.record_click(
                query=last_query,
                memory_id=memory_id,
                source_type=result.get("type", "unknown"),
            )
        return json.dumps(result, ensure_ascii=False)

    elif action == "events":
        from_turn = args.get("from_turn", 0)
        to_turn = args.get("to_turn", turn_count)
        query = args.get("query", "")

        events = []
        try:
            all_events = store.search(memory_type="event", limit=100)
            for evt in all_events:
                evt_content = evt.get("content", "")
                if query and query.lower() not in evt_content.lower():
                    continue
                turn_match = re.search(
                    r"\[Turn (\d+)\]|\[Checkpoint at turn (\d+)\]|\[Emergency save\].*?turn[_ ](\d+)",
                    evt_content,
                )
                if turn_match:
                    turn_num = int(
                        turn_match.group(1) or turn_match.group(2) or turn_match.group(3)
                    )
                else:
                    turn_num = 0
                if from_turn <= turn_num <= to_turn:
                    events.append(
                        {
                            "turn": turn_num,
                            "memory_id": evt.get("memory_id", ""),
                            "content": evt_content,
                            "type": evt.get("type", "event"),
                            "stored_at": evt.get("stored_at", ""),
                        }
                    )
        except Exception as e:
            logger.warning("OmniMem events query failed: %s", e)

        events.sort(key=lambda x: x.get("turn", 0))

        return json.dumps(
            {
                "status": "ok",
                "from_turn": from_turn,
                "to_turn": to_turn,
                "count": len(events),
                "events": events[:20],
            },
            ensure_ascii=False,
        )

    elif action == "drill_down":
        # ★ OPT: 按 node_id 下钻溯源链，恢复完整原文
        node_id = args.get("node_id", "")
        if not node_id:
            return json.dumps(
                {"status": "error", "message": "node_id is required for action='drill_down'"},
                ensure_ascii=False,
            )
        if trace_chain:
            chain = trace_chain.drill_down(node_id)
            full_text = trace_chain.recover_full_text(node_id)
            return json.dumps(
                {
                    "status": "ok",
                    "node_id": node_id,
                    "trace_chain": chain,
                    "full_text": full_text,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {"status": "error", "message": "Trace chain not available"},
            ensure_ascii=False,
        )

    return json.dumps({"error": f"Unknown action: {action}"})


def run_prefetch(
    query: str,
    session_id: str,
    config: Any,
    retriever: Any,
    context_manager: Any,
    kv_cache: Any,
    knowledge_graph: Any,
    temporal_decay: Any,
    privacy: Any,
    prefetch_cache: str,
    prefetch_lock: Any,
    forgetting: Any = None,
) -> tuple[str, str]:
    context_manager.reset_for_new_turn()

    kv_results = []
    if kv_cache:
        kv_results = kv_cache.search_cache(query, limit=5)
        if kv_results:
            for cr in kv_results:
                cr["source_type"] = "kv_cache"

    async_results = []
    with prefetch_lock:
        cached = prefetch_cache
        prefetch_cache = ""
    if cached and cached.startswith("___RAW_RESULTS___"):
        try:
            async_results = json.loads(cached[len("___RAW_RESULTS___"):])
        except Exception as e:
            logger.warning("Async prefetch cache JSON parse failed: %s", e)
            async_results = []

    live_results = []
    if not kv_results and not async_results:
        max_tokens = config.get("max_prefetch_tokens", 300)
        live_results = retriever.search(query, max_tokens=max_tokens)

        # ★ OPT: prefetch 结果记录访问（可配置，驱动遗忘曲线热度分类）
        if config.get("prefetch_record_access", False) and forgetting and live_results:
            for r in live_results:
                mid = r.get("memory_id", "")
                if mid:
                    try:
                        forgetting.record_access(mid)
                    except Exception as e:
                        logger.warning("ToolRouter prefetch record_access failed: %s", e)

        if knowledge_graph:
            try:
                graph_results = knowledge_graph.graph_search(query, max_depth=2, limit=10)
                if graph_results:
                    for gr in graph_results[:5]:
                        gr["content"] = (
                            f"{gr.get('subject', '')} {gr.get('predicate', '')} {gr.get('object', '')}"
                        )
                        gr["type"] = "graph_triple"
                        gr["confidence"] = gr.get("confidence", 0.5)
                    live_results.extend(graph_results[:5])
            except Exception as e:
                logger.warning("OmniMem graph prefetch failed: %s", e)

        live_results = temporal_decay.apply(live_results)
        live_results = privacy.filter(live_results, session_id=session_id)

        if kv_cache and live_results:
            for r in live_results[:3]:
                if r.get("score", 0) > 0.6:
                    kv_cache.check_and_auto_preload(
                        key=r.get("memory_id", ""),
                        content=r.get("content", ""),
                        metadata={"source": "prefetch", "query": query},
                        source_memory_ids=[r.get("memory_id", "")],
                    )

    all_results = kv_results + async_results + live_results

    if not all_results:
        # ★ 智能预取：即使无结果，也预测下一轮查询并预热缓存
        _trigger_smart_prefetch(query, config, retriever)
        return "", prefetch_cache

    # ★ 智能预取：基于 PerceptionEngine 预测下一轮查询，异步回填 MultiLevelCache
    _trigger_smart_prefetch(query, config, retriever)

    return str(context_manager.refine_prefetch_results(all_results)), prefetch_cache


def _trigger_smart_prefetch(query: str, config: Any, retriever: Any) -> None:
    """基于 PerceptionEngine 预测下一轮查询，异步预取并回填 MultiLevelCache。

    使用 daemon 线程在后台执行，不阻塞主流程。
    预取结果通过 retriever.search() 自动写入 MultiLevelCache（search 内部调用 _set_cache）。
    """
    try:
        from omnimem.perception.engine import PerceptionEngine

        # 仅当配置启用智能预取时执行（默认启用）
        if not config.get("enable_smart_prefetch", True):
            return

        engine = PerceptionEngine()
        predicted = engine.predict_intent(query)
        # 预测查询为空、过短或与当前查询相同则跳过
        if not predicted or len(predicted) < 3 or predicted == query:
            return

        # 后台 daemon 线程执行预取（不阻塞主流程）
        thread = threading.Thread(
            target=_smart_prefetch_background,
            args=(predicted, config, retriever),
            daemon=True,
        )
        thread.start()
        logger.debug("Smart prefetch triggered for predicted query: %s", predicted[:50])
    except Exception as e:
        # 智能预取失败不影响主流程
        logger.debug("Smart prefetch trigger failed: %s", e)


def _smart_prefetch_background(predicted_query: str, config: Any, retriever: Any) -> None:
    """后台执行智能预取，结果回填 MultiLevelCache。

    retriever.search() 内部会调用 _set_cache 将结果写入 MultiLevelCache，
    因此此处只需调用 search 即可自动预热缓存。
    """
    try:
        max_tokens = config.get("max_prefetch_tokens", 300)
        # 调用 retriever.search 自动缓存结果到 MultiLevelCache
        results = retriever.search(predicted_query, max_tokens=max_tokens)
        if results:
            logger.debug(
                "Smart prefetch completed: %d results for predicted query: %s",
                len(results),
                predicted_query[:50],
            )
    except Exception as e:
        logger.debug("Smart prefetch background failed: %s", e)


def run_queue_prefetch(
    query: str,
    session_id: str,
    config: Any,
    retriever: Any,
    temporal_decay: Any,
    privacy: Any,
    prefetch_lock: Any,
) -> str:
    try:
        max_tokens = config.get("max_prefetch_tokens", 300)
        result = retriever.search(query, max_tokens=max_tokens)
        result = temporal_decay.apply(result)
        result = privacy.filter(result, session_id=session_id)
        if result:
            serialized = "___RAW_RESULTS___" + json.dumps(result, ensure_ascii=False)
        else:
            serialized = ""
        with prefetch_lock:
            return serialized
    except Exception as e:
        logger.warning("OmniMem background prefetch failed: %s", e)
        return ""


def l3_recall(query: str, retriever: Any, store: Any, limit: int = 20) -> list[dict[str, Any]]:
    results = retriever.search(query, max_tokens=3000)
    if results:
        return results[:limit]

    query_keywords = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", query.lower()))
    if query_keywords:
        seen_ids: set[str] = set()
        for kw in list(query_keywords)[:5]:
            for sf in store.search_by_content(kw, limit=20):
                mid = sf.get("memory_id", "")
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                sf_content = sf.get("content", "").lower()
                keyword_hits = sum(1 for kw2 in query_keywords if kw2 in sf_content)
                if keyword_hits >= 1:
                    sf["_source"] = "store_fallback"
                    sf["score"] = min(0.15 + keyword_hits * 0.05, 0.35)
                    results.append(sf)
                    if len(results) >= limit:
                        break
            if len(results) >= limit:
                break
    return results[:limit]


# ★ 修复1: reflect_cache 并发访问锁（模块级别，保护多线程读写）
_reflect_cache_lock = threading.Lock()


def make_llm_call_fn(llm_client: Any) -> Callable[[str], str] | None:
    if not llm_client:
        return None

    def _llm_call(prompt: str) -> str:
        result = llm_client.call_sync(
            prompt=prompt,
            system="You are a structured summarizer. Respond in JSON only.",
            max_tokens=600,
            temperature=0.3,
        )
        return result.content if result and result.content else ""

    return _llm_call


def call_llm_for_reflect(
    prompt: str,
    system: str,
    llm_client: Any,
    reflect_cache: dict[str, tuple[str, float]],
    max_tokens: int = 800,
) -> str | None:
    max_reflect_cache = 64
    cache_key = prompt[:200]
    now = time.time()
    # ★ 修复1: 缓存清理加锁，不在 LLM 调用期间持锁
    with _reflect_cache_lock:
        if len(reflect_cache) > max_reflect_cache:
            expired_keys = [k for k, (_, t) in reflect_cache.items() if now - t >= _REFLECT_CACHE_TTL]
            for k in expired_keys:
                del reflect_cache[k]
            if len(reflect_cache) > max_reflect_cache:
                sorted_keys = sorted(reflect_cache.keys(), key=lambda k: reflect_cache[k][1])
                for k in sorted_keys[:len(sorted_keys) // 2]:
                    del reflect_cache[k]
    # ★ 修复1: 缓存读取加锁
    with _reflect_cache_lock:
        cache_hit = None
        if cache_key in reflect_cache:
            cached_result, cached_time = reflect_cache[cache_key]
            if now - cached_time < _REFLECT_CACHE_TTL:
                cache_hit = str(cached_result)
    if cache_hit is not None:
        logger.warning("ReflectEngine LLM cache hit")
        return cache_hit

    if llm_client:
        try:
            # ★ 修复1: LLM 调用不在锁内，避免持锁阻塞
            result = llm_client.call_sync(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=0.5,
            )
            if result.content:
                # ★ 修复1: 缓存写入加锁
                with _reflect_cache_lock:
                    reflect_cache[cache_key] = (result.content, now)
                return result.content
        except Exception as e:
            logger.warning("ReflectEngine AsyncLLM failed: %s", e)

    logger.warning("ReflectEngine: LLM client not available, returning None")
    return None


async def async_call_llm_for_reflect(
    prompt: str,
    system: str,
    llm_client: Any,
    reflect_cache: dict[str, tuple[str, float]],
    max_tokens: int = 800,
) -> str | None:
    """异步版本的 call_llm_for_reflect — 使用 asyncio.to_thread 包装 LLM 调用。

    保持与同步版本相同的缓存逻辑（模块级锁 + LRU）：
      1. 缓存清理/读取/写入仍使用 threading.Lock（操作极短，不会阻塞事件循环）
      2. LLM 调用使用 asyncio.to_thread 包装，不持锁，避免阻塞事件循环

    Args:
        prompt: 反思 prompt
        system: 系统提示
        llm_client: LLM 客户端实例（需提供 call_sync 方法）
        reflect_cache: 反思缓存字典（与同步版本共享）
        max_tokens: 最大输出 token

    Returns:
        LLM 响应文本，失败时返回 None
    """
    max_reflect_cache = 64
    cache_key = prompt[:200]
    now = time.time()

    # ★ 缓存清理加锁（与同步版本一致，操作极短）
    with _reflect_cache_lock:
        if len(reflect_cache) > max_reflect_cache:
            expired_keys = [k for k, (_, t) in reflect_cache.items() if now - t >= _REFLECT_CACHE_TTL]
            for k in expired_keys:
                del reflect_cache[k]
            if len(reflect_cache) > max_reflect_cache:
                sorted_keys = sorted(reflect_cache.keys(), key=lambda k: reflect_cache[k][1])
                for k in sorted_keys[:len(sorted_keys) // 2]:
                    del reflect_cache[k]

    # ★ 缓存读取加锁
    with _reflect_cache_lock:
        cache_hit = None
        if cache_key in reflect_cache:
            cached_result, cached_time = reflect_cache[cache_key]
            if now - cached_time < _REFLECT_CACHE_TTL:
                cache_hit = str(cached_result)
    if cache_hit is not None:
        logger.warning("ReflectEngine LLM async cache hit")
        return cache_hit

    if llm_client:
        try:
            # ★ LLM 调用使用 asyncio.to_thread 包装，不持锁，避免阻塞事件循环
            result = await asyncio.to_thread(
                llm_client.call_sync,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=0.5,
            )
            if result and result.content:
                # ★ 缓存写入加锁
                with _reflect_cache_lock:
                    reflect_cache[cache_key] = (result.content, now)
                return result.content
        except Exception as e:
            logger.warning("ReflectEngine async LLM failed: %s", e)

    logger.warning("ReflectEngine: LLM client not available (async), returning None")
    return None


def retry_index_add(memory_id: str, store: Any, index: Any) -> None:
    entry = store.get(memory_id)
    if not entry:
        raise RuntimeError(f"Memory {memory_id} not found in store")
    index.add(
        memory_id=memory_id,
        wing=entry.get("wing", ""),
        hall=entry.get("hall", entry.get("type", "fact")),
        room=entry.get("room", ""),
        content=entry.get("content", ""),
        summary=entry.get("content", "")[:200]
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " "),
        type=entry.get("type", "fact"),
        confidence=entry.get("confidence", 3),
        privacy=entry.get("privacy", "personal"),
        scope=entry.get("privacy", "personal"),
        stored_at=entry.get("stored_at", ""),
        provenance="",
    )


def retry_retriever_add(memory_id: str, store: Any, retriever: Any) -> None:
    entry = store.get(memory_id)
    if not entry:
        raise RuntimeError(f"Memory {memory_id} not found in store")
    retriever.add(
        entry.get("content", ""),
        memory_id=memory_id,
        metadata={
            "memory_id": memory_id,
            "type": entry.get("type", "fact"),
            "confidence": entry.get("confidence", 3),
            "scope": entry.get("privacy", "personal"),
            "privacy": entry.get("privacy", "personal"),
            "wing": entry.get("wing", ""),
            "room": entry.get("room", ""),
            "stored_at": entry.get("stored_at", ""),
        },
    )


def retry_kg_extract(memory_id: str, store: Any, knowledge_graph: Any) -> None:
    entry = store.get(memory_id)
    if not entry:
        raise RuntimeError(f"Memory {memory_id} not found in store")
    if knowledge_graph:
        knowledge_graph.extract_and_store(
            entry.get("content", ""),
            memory_id=memory_id,
            confidence=entry.get("confidence", 3) / 5.0,
        )


def apply_sync_change(change: dict[str, Any], store: Any, index: Any, retriever: Any, forgetting: Any) -> bool:
    data = change.get("data", {})
    op = change.get("operation", "INSERT")
    memory_id = data.get("memory_id", "")
    if not memory_id:
        return False

    if op == "DELETE":
        forgetting.archive(memory_id)
        return True

    try:
        store.add(
            memory_id=memory_id,
            wing=data.get("wing", "auto"),
            room=data.get("room", "sync"),
            content=data.get("content", ""),
            memory_type=data.get("type", "fact"),
            confidence=data.get("confidence", 3),
            privacy=data.get("privacy", "personal"),
            vc=data.get("vc", change.get("vc", "")),
        )
        index.add(
            memory_id=memory_id,
            wing=data.get("wing", "auto"),
            hall=data.get("type", "fact"),
            room=data.get("room", "sync"),
            content=data.get("content", ""),
            summary=data.get("content", "")[:200]
            .replace("\n", " ")
            .replace("\r", " ")
            .replace("\t", " "),
            type=data.get("type", "fact"),
            confidence=data.get("confidence", 3),
            privacy=data.get("privacy", "personal"),
            scope=data.get("privacy", "personal"),
            stored_at=data.get("stored_at", ""),
            provenance=json.dumps({"sync_from": change.get("instance_id", "unknown")}),
        )
        retriever.add(
            data.get("content", ""),
            memory_id=memory_id,
            metadata={
                "memory_id": memory_id,
                "type": data.get("type", "fact"),
                "confidence": data.get("confidence", 3),
                "scope": data.get("privacy", "personal"),
                "privacy": data.get("privacy", "personal"),
                "wing": data.get("wing", "auto"),
                "room": data.get("room", "sync"),
                "stored_at": data.get("stored_at", ""),
            },
        )
        return True
    except Exception as e:
        logger.warning("OmniMem apply_sync_change failed for %s: %s", memory_id, e)
        return False


def get_config_schema() -> list[dict[str, Any]]:
    """从 config._config 动态生成 UI 友好的配置 schema。"""
    from omnimem.config._config import _CONFIG_SCHEMA

    result = []
    for key, spec in _CONFIG_SCHEMA.items():
        entry = {"key": key, "description": spec.get("description", ""), "default": spec.get("default")}
        if "choices" in spec:
            entry["choices"] = spec["choices"]
        result.append(entry)
    return result


def save_config(values: dict[str, Any], hermes_home: str) -> None:
    from pathlib import Path

    config_path = Path(hermes_home) / "omnimem" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(values, f, allow_unicode=True, default_flow_style=False)
    except ImportError:
        logger.warning("yaml not available — config not saved")
