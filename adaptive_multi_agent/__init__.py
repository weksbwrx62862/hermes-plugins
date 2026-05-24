"""Adaptive Multi-Agent (AMA) — 自适应多智能体调度插件。

根据任务复杂度自动选择最佳的 Anthropic 多智能体协作模式，
通过 Hermes delegate_tool 委派子智能体执行，支持智能模式切换、
性能持久化和历史学习。

五种模式:
  - generator_verifier: 生成-验证（复杂度 1-3）
  - orchestrator_subagent: 协调-子代理（复杂度 4-6）
  - agent_teams: 团队协作（复杂度 7-8）
  - message_bus: 事件驱动（复杂度 9）
  - shared_state: 共享状态（复杂度 10）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# 注册 plugins.adaptive_multi_agent 包别名，解决 Hermes 加载时的导入路径问题
sys.modules.setdefault("plugins.adaptive_multi_agent", sys.modules[__name__])

# 缓存的 PluginContext，在 register() 时由 Hermes 设置
_plugin_ctx: Optional[object] = None

from .handlers import (
    handle_ama_assess,
    handle_ama_cancel,
    handle_ama_clarify,
    handle_ama_diagnose,
    handle_ama_execute,
    handle_ama_stats,
    handle_ama_switch_mode,
    handle_on_session_end,
    handle_on_session_start,
    handle_post_tool_call,
)
from .schemas import AMA_TOOL_SCHEMAS

_TOOLS = (
    ("ama_execute", AMA_TOOL_SCHEMAS["ama_execute"], handle_ama_execute, "🔀"),
    ("ama_assess", AMA_TOOL_SCHEMAS["ama_assess"], handle_ama_assess, "📊"),
    ("ama_switch_mode", AMA_TOOL_SCHEMAS["ama_switch_mode"], handle_ama_switch_mode, "🔄"),
    ("ama_stats", AMA_TOOL_SCHEMAS["ama_stats"], handle_ama_stats, "📈"),
    ("ama_cancel", AMA_TOOL_SCHEMAS["ama_cancel"], handle_ama_cancel, "🛑"),
    ("ama_clarify", AMA_TOOL_SCHEMAS["ama_clarify"], handle_ama_clarify, "💬"),
    ("ama_diagnose", AMA_TOOL_SCHEMAS["ama_diagnose"], handle_ama_diagnose, "🔍"),
)


def register(ctx) -> None:
    global _plugin_ctx
    _plugin_ctx = ctx

    # 将 ctx 注入 handlers 模块，供工具处理器使用
    from . import handlers as _h
    _h._plugin_ctx = ctx

    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="ama",
            schema=schema,
            handler=handler,
            emoji=emoji,
        )

    ctx.register_hook("post_tool_call", handle_post_tool_call)
    ctx.register_hook("on_session_start", handle_on_session_start)
    ctx.register_hook("on_session_end", handle_on_session_end)
