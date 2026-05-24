from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable

from omnimem.context.manager import ContextManager

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
            "omni_memorize": memorize_fn,
            "omni_recall": recall_fn,
            "omni_govern": govern_fn,
            "omni_reflect": reflect_fn,
            "omni_compact": compact_fn,
            "omni_detail": detail_fn,
            "memory": memory_compat_fn,
        }
        if record_action_fn is not None:
            self._routes["omni_record_action"] = record_action_fn

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


def build_system_prompt(
    data_dir: str,
    store: Any,
    core_block: Any,
    context_manager: Any,
    config: Any,
    turn_count: int,
    system_prompt_cache_turn: int,
    system_prompt_cache_value: str,
    last_query: str,
) -> tuple[str, int, str]:
    if system_prompt_cache_turn == turn_count:
        return system_prompt_cache_value, system_prompt_cache_turn, system_prompt_cache_value

    parts = [
        "## OmniMem Memory System (Unified)",
        f"Memory directory: {data_dir}",
        "",
    ]

    boot_entries = []
    fact_entries = []
    for mtype in ("preference", "correction"):
        entries = store.search(memory_type=mtype, limit=10)
        for e in entries:
            e["_mtype"] = mtype
            boot_entries.append(e)
    for e in store.search(memory_type="fact", limit=15):
        e["_mtype"] = "fact"
        fact_entries.append(e)

    if not boot_entries and not fact_entries:
        parts.append("### Identity")
        parts.append(core_block.identity_block)
        result = "\n".join(parts)
        return result, turn_count, result

    refined_lines = []
    total_chars = 0
    base_budget = config.get("system_prompt_char_limit", 500)
    query_kw_count = (
        len(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", last_query.lower()))
        if last_query
        else 0
    )
    char_budget = base_budget + min(query_kw_count * 40, 300)
    max_summary = context_manager.max_summary_chars
    seen_fps = set(context_manager.get_injected_fingerprints())

    def _refine_and_add(entries: list[dict[str, Any]], budget_remaining: int) -> tuple[list[str], int]:
        lines = []
        used = 0
        for entry in entries:
            raw = entry.get("content", "")
            if not raw:
                continue
            summary = ContextManager.refine_content(raw, max_summary)
            if len(summary) < 3:
                continue
            fp = ContextManager._content_fingerprint(summary)
            if fp:
                is_dup = any(
                    ContextManager._fingerprint_similarity(fp, existing) > 0.7
                    for existing in seen_fps
                )
                if is_dup:
                    continue
                seen_fps.add(fp)
                context_manager.add_persistent_fingerprint(fp)
            line = f"- [{entry.get('_mtype', 'fact')}] {summary}"
            if used + len(line) + 1 > budget_remaining:
                break
            lines.append(line)
            used += len(line) + 1
        return lines, used

    boot_lines, boot_used = _refine_and_add(boot_entries, char_budget)
    refined_lines.extend(boot_lines)
    total_chars += boot_used

    remaining = char_budget - total_chars
    if remaining > 50 and fact_entries:
        fact_lines, fact_used = _refine_and_add(fact_entries, remaining)
        refined_lines.extend(fact_lines)
        total_chars += fact_used

    if refined_lines:
        parts.append("### Core Memories (summaries — use omni_detail for full content)")
        parts.extend(refined_lines)
        parts.append("")

    parts.append("### Identity")
    parts.append(core_block.identity_block)

    result = "\n".join(parts)
    return result, turn_count, result


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
        return "", prefetch_cache

    return str(context_manager.refine_prefetch_results(all_results)), prefetch_cache


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


_REFLECT_CACHE_TTL = 60.0


def init_llm_client(config: Any) -> Any:
    from omnimem.utils.llm_client import AsyncLLMClient

    creds = AsyncLLMClient.load_credentials_from_env()
    if not creds.get("api_key") or not creds.get("base_url"):
        creds.update(AsyncLLMClient.load_credentials_from_hermes_env())
    config_creds = AsyncLLMClient.load_credentials_from_hermes_config()
    if not creds.get("base_url"):
        creds["base_url"] = config_creds.get("base_url", "")
    if not creds.get("api_key"):
        creds["api_key"] = config_creds.get("api_key", "")
    # ★ R25修复ARCH-1 + P0-fix：model 选择策略
    # 必须匹配实际 base_url 对应的 provider 的 models 列表，
    # 否则会把 mimo-v2.5-pro 发给 deepseek API（400错误）
    actual_base_url = creds.get("base_url", "")
    default_model = config_creds.get("model") or config.get("default", "glm-5.1")

    # 从 config providers 中找到与实际 base_url 匹配的 provider，取其 models
    matched_models: list[str] = []
    try:
        from pathlib import Path
        import yaml
        cfg_file = Path.home() / ".hermes" / "config.yaml"
        if cfg_file.exists():
            cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            for _pname, pval in (cfg.get("providers") or {}).items():
                if isinstance(pval, dict) and pval.get("base_url") == actual_base_url:
                    matched_models = pval.get("models", [])
                    if matched_models:
                        logger.warning("Matched provider '%s' for base_url %s, models: %s",
                                     _pname, actual_base_url, matched_models)
                        break
    except Exception as e:
        logger.warning("ToolRouter init_llm_client provider match failed: %s", e)

    # 优先用匹配到的 provider models，其次用 config_creds 的 models，最后用 default
    if matched_models:
        model = matched_models[0]
    elif config_creds.get("models"):
        # config_creds.models 来自第一个 provider，可能与 base_url 不匹配
        # 只在 base_url 也来自 config（未被 env 覆盖）时使用
        config_base_url = config_creds.get("base_url", "")
        if config_base_url and config_base_url == actual_base_url:
            model = config_creds["models"][0]
        else:
            model = default_model
    else:
        model = default_model
    logger.warning("Selected model=%s (actual_base_url=%s, default=%s)", model, actual_base_url, default_model)

    # ★ R25修复ARCH-1：凭证有效性检测
    has_api_key = bool(creds.get("api_key", "").strip())
    has_base_url = bool(creds.get("base_url", "").strip())
    if not has_api_key or not has_base_url:
        logger.warning(
            "AsyncLLMClient: LLM 凭证不完整 (api_key=%s, base_url=%s), "
            "Reflect/Recall 的 LLM 功能将不可用，回退到规则归纳",
            "有" if has_api_key else "缺失",
            "有" if has_base_url else "缺失",
        )

    llm_client = AsyncLLMClient(
        api_key=creds.get("api_key", ""),
        base_url=creds.get("base_url", ""),
        model=model,
        max_concurrent=3,
        timeout=30.0,
        cache_ttl=_REFLECT_CACHE_TTL,
    )
    logger.warning("AsyncLLMClient initialized: model=%s, has_creds=%s", model, has_api_key and has_base_url)
    return llm_client


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
    if len(reflect_cache) > max_reflect_cache:
        reflect_cache.clear()
        reflect_cache.update(
            {
                k: (v, t)
                for k, (v, t) in reflect_cache.items()
                if now - t < _REFLECT_CACHE_TTL
            }
        )
    if cache_key in reflect_cache:
        cached_result, cached_time = reflect_cache[cache_key]
        if now - cached_time < _REFLECT_CACHE_TTL:
            logger.warning("ReflectEngine LLM cache hit")
            return str(cached_result)

    if llm_client:
        try:
            result = llm_client.call_sync(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=0.5,
            )
            if result.content:
                reflect_cache[cache_key] = (result.content, now)
                return result.content
        except Exception as e:
            logger.warning("ReflectEngine AsyncLLM failed: %s", e)

    logger.warning("ReflectEngine: LLM client not available, returning None")
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


_CONFIG_SCHEMA: list[dict[str, Any]] = [
    {
        "key": "save_interval",
        "description": "Auto-save every N turns (default: 15)",
        "default": 15,
    },
    {
        "key": "retrieval_mode",
        "description": "Default retrieval mode (rag/llm)",
        "default": "rag",
        "choices": ["rag", "llm"],
    },
    {
        "key": "vector_backend",
        "description": "Vector storage backend",
        "default": "chromadb",
        "choices": ["chromadb", "qdrant", "pgvector"],
    },
    {
        "key": "fact_threshold",
        "description": "Consolidation trigger threshold (default: 10)",
        "default": 10,
    },
    {
        "key": "enable_reranker",
        "description": "Enable Cross-Encoder reranking (needs sentence-transformers)",
        "default": False,
    },
    {
        "key": "enable_compression",
        "description": "Enable 5-layer compression pipeline in on_pre_compress (default: False)",
        "default": False,
    },
    {
        "key": "conflict_strategy",
        "description": "Conflict resolution strategy",
        "default": "latest",
        "choices": ["latest", "confidence", "manual"],
    },
    {
        "key": "budget_tokens",
        "description": "Token budget for working memory (default: 4000)",
        "default": 4000,
    },
    {
        "key": "kv_cache_threshold",
        "description": "KV Cache auto-preload threshold (access count, default: 10)",
        "default": 10,
    },
    {
        "key": "kv_cache_max",
        "description": "KV Cache max entries (default: 100)",
        "default": 100,
    },
    {
        "key": "lora_base_model",
        "description": "LoRA base model name (default: Qwen2.5-7B)",
        "default": "Qwen2.5-7B",
    },
    {
        "key": "sync_mode",
        "description": "Multi-instance sync mode (none/file_lock/changelog)",
        "default": "none",
        "choices": ["none", "file_lock", "changelog"],
    },
    {
        "key": "sync_interval",
        "description": "Sync interval in seconds for changelog mode (default: 30)",
        "default": 30,
    },
    {
        "key": "sync_conflict_resolution",
        "description": "How to resolve sync conflicts",
        "default": "latest_wins",
        "choices": ["latest_wins", "manual"],
    },
]


def get_config_schema() -> list[dict[str, Any]]:
    return list(_CONFIG_SCHEMA)


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
