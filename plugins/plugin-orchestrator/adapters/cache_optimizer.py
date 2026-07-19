"""
deepseek-cache-optimizer → PluginOrchestrator 适配器

解决 model-router 与 cache-optimizer 之间的"盲人摸象"冲突。
使用 sys.modules 加载 context 模块，完全向后兼容。
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ORCHESTRATOR_AVAILABLE: Optional[bool] = None
_CTX_MOD_NAME = "plugin_orchestrator.context"


def _load_context_module():
    if _CTX_MOD_NAME in sys.modules:
        return sys.modules[_CTX_MOD_NAME]
    try:
        import importlib.util, os
        _plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _ctx_spec = importlib.util.spec_from_file_location(
            _CTX_MOD_NAME,
            os.path.join(_plugin_dir, "context.py"),
        )
        _ctx_mod = importlib.util.module_from_spec(_ctx_spec)
        sys.modules[_CTX_MOD_NAME] = _ctx_mod
        _ctx_spec.loader.exec_module(_ctx_mod)
        return _ctx_mod
    except Exception:
        return None


def _check_orchestrator():
    global _ORCHESTRATOR_AVAILABLE
    if _ORCHESTRATOR_AVAILABLE is None:
        _ORCHESTRATOR_AVAILABLE = _load_context_module() is not None
    return _ORCHESTRATOR_AVAILABLE


def _get_ctx_func(func_name: str):
    mod = sys.modules.get(_CTX_MOD_NAME) or _load_context_module()
    if mod is None:
        return None
    return getattr(mod, func_name, None)


def get_real_provider_info(
    session_id: str,
    fallback_provider: str = "",
    fallback_model: str = "",
    fallback_base_url: str = "",
) -> Dict[str, str]:
    if _check_orchestrator() and session_id:
        try:
            get_context = _get_ctx_func("get_context")
            if get_context:
                ctx = get_context(session_id)
                if ctx:
                    routing = ctx.shared_get("routing_decision", {}) or {}
                    metadata = ctx.metadata
                    return {
                        "provider": routing.get("provider") or metadata.get("provider") or fallback_provider,
                        "model": routing.get("name") or metadata.get("model") or fallback_model,
                        "base_url": routing.get("base_url") or metadata.get("base_url") or fallback_base_url,
                    }
        except Exception as exc:
            logger.debug("Failed to get real provider via PluginContext: %s", exc)

    return {"provider": fallback_provider, "model": fallback_model, "base_url": fallback_base_url}


def publish_cache_diagnostics(session_id: str, diagnostics: Dict[str, Any]) -> None:
    if _check_orchestrator() and session_id:
        try:
            get_context = _get_ctx_func("get_context")
            if get_context:
                ctx = get_context(session_id)
                if ctx:
                    ctx.shared_set("cache_diagnostics", diagnostics)
                    ctx.event_bus.publish(
                        "cache_diagnostics_updated",
                        source_plugin="deepseek_cache_optimizer",
                        session_id=session_id,
                        **diagnostics,
                    )
        except Exception:
            pass


def is_tool_result_compressed(session_id: str) -> bool:
    if _check_orchestrator() and session_id:
        try:
            get_context = _get_ctx_func("get_context")
            if get_context:
                ctx = get_context(session_id)
                if ctx:
                    return ctx.shared_get("tool_result_compressed", False)
        except Exception:
            pass
    return False


def mark_tool_result_compressed(session_id: str) -> None:
    if _check_orchestrator() and session_id:
        try:
            get_context = _get_ctx_func("get_context")
            if get_context:
                ctx = get_context(session_id)
                if ctx:
                    ctx.shared_set("tool_result_compressed", True)
                    ctx.plugin_set("deepseek_cache_optimizer", "last_compression_time",
                                   __import__("time").time())
        except Exception:
            pass


def on_model_routed_callback(event: Dict[str, Any]) -> None:
    """model_routed 事件回调，cache-optimizer 在此更新缓存策略。"""
    session_id = event.get("data", {}).get("session_id", "")
    model = event.get("data", {}).get("model", "")
    provider = event.get("data", {}).get("provider", "")
    logger.debug(
        "Cache-optimizer received model_routed event: %s → %s (%s)",
        session_id[:8], model, provider,
    )
