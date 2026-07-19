"""
model-router → PluginOrchestrator 适配器

将 model-router 原有的全局变量通信方式升级为 PluginContext。
通过此适配器，model-router 可以：
  1. 将路由决策发布到共享上下文（而非 _routing_decisions dict）
  2. 接收来自 AMA 等插件的任务权重（通过共享上下文而非 _ama_task_weights）
  3. 发布事件通知其他插件模型切换、预算变更等

如果 PluginOrchestrator 未安装，自动降级到原有的全局变量模式。
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── 可用性检测 ─────────────────────────────────────────────────────

_ORCHESTRATOR_AVAILABLE: Optional[bool] = None
_CTX_MOD_NAME = "plugin_orchestrator.context"


def _load_context_module():
    """加载 context 模块到 sys.modules（如果尚未加载）。返回模块或 None。"""
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
    """从已加载的 context 模块获取函数。"""
    mod = sys.modules.get(_CTX_MOD_NAME)
    if mod is None:
        mod = _load_context_module()
    if mod is None:
        return None
    return getattr(mod, func_name, None)


# ── 增强的路由决策存储 ─────────────────────────────────────────────


def store_routing_decision(
    session_id: str,
    decision: Dict[str, Any],
    *,
    fallback_to_global: bool = True,
) -> None:
    if _check_orchestrator():
        try:
            get_context = _get_ctx_func("get_context")
            if get_context:
                ctx = get_context(session_id)
                if ctx:
                    # 只写入，不消费任何数据（打破循环）
                    ctx.shared_set("routing_decision", decision)
                    ctx.shared_set("model_selection", decision.get("name", ""))
                    ctx.shared_set("model_quality", decision.get("quality", 3))
                    ctx.shared_set("routing_strategy", decision.get("strategy", "auto"))
                    ctx.shared_set("budget_status", decision.get("budget", {}))
                    ctx.update_metadata(
                        model=decision.get("name", ""),
                        provider=decision.get("provider", ""),
                    )
                    # 仅发布事件通知其他插件，不依赖其响应
                    ctx.event_bus.publish(
                        "model_routed",
                        source_plugin="model_router",
                        session_id=session_id,
                        model=decision.get("name", ""),
                        provider=decision.get("provider", ""),
                        strategy=decision.get("strategy", "auto"),
                    )
                    logger.debug("Routing decision stored via PluginContext: %s", session_id[:8])
                    return
        except Exception as exc:
            logger.warning("Failed to store via PluginContext: %s", exc)

    # 回退：使用原有的全局变量
    if fallback_to_global:
        try:
            import plugins.model_router as mr
            if hasattr(mr, '_routing_decisions'):
                with mr._routing_lock:
                    mr._routing_decisions[session_id] = {
                        "decision": decision,
                        "_created_at": __import__("time").time(),
                    }
                logger.debug("Routing decision fallback to global dict: %s", session_id[:8])
        except Exception:
            pass


def get_ama_task_weight(session_id: str) -> Optional[float]:
    """从 PluginContext 获取 AMA 设置的任务权重。不消费任何数据，仅读取（打破循环）"""
    if _check_orchestrator():
        try:
            get_context = _get_ctx_func("get_context")
            if get_context:
                ctx = get_context(session_id)
                if ctx:
                    # 只读取，不写入或产生副作用
                    return ctx.shared_get("ama_task_weight")
        except Exception as exc:
            logger.warning("Failed to read ama_task_weight via PluginContext: %s", exc)
    return None


def notify_budget_warning(budget_percent: float, session_id: str = "") -> None:
    """发布预算告警事件。"""
    if _check_orchestrator() and session_id:
        try:
            get_context = _get_ctx_func("get_context")
            if get_context:
                ctx = get_context(session_id)
                if ctx:
                    ctx.event_bus.publish(
                        "budget_warning",
                        source_plugin="model_router",
                        session_id=session_id,
                        budget_percent=budget_percent,
                    )
        except Exception:
            pass


def notify_provider_cooldown(provider: str, reason: str, duration_seconds: int = 3600) -> None:
    """发布供应商冷却事件。"""
    if _check_orchestrator():
        try:
            list_active_contexts = _get_ctx_func("list_active_contexts")
            get_context = _get_ctx_func("get_context")
            if list_active_contexts and get_context:
                for sid in list_active_contexts():
                    ctx = get_context(sid)
                    if ctx:
                        ctx.event_bus.publish(
                            "provider_cooldown",
                            source_plugin="model_router",
                            provider=provider,
                            reason=reason,
                            duration_seconds=duration_seconds,
                        )
        except Exception:
            pass


def set_task_weight(session_id: str, score: float) -> Optional[str]:
    """设置任务权重（由 AMA 调用），推荐策略。不消费任何数据，仅写入（打破循环）"""
    strategy = "auto"
    if score <= 3:
        strategy = "cheapest"
    elif score >= 7:
        strategy = "smartest"

    if _check_orchestrator():
        try:
            get_context = _get_ctx_func("get_context")
            if get_context:
                ctx = get_context(session_id)
                if ctx:
                    # 只写入，不读取或产生副作用（打破循环）
                    ctx.shared_set("ama_task_weight", score)
                    logger.debug("Task weight stored via PluginContext: %.1f → %s", score, strategy)
                    return strategy
        except Exception as exc:
            logger.warning("Failed to set task weight via PluginContext: %s", exc)

    # 回退：使用原有的全局变量（不修改）
    try:
        import plugins.model_router as mr
        if hasattr(mr, '_ama_task_weights'):
            with mr._ama_task_lock:
                mr._ama_task_weights[session_id] = score
            return strategy
    except ImportError:
        pass
    return strategy