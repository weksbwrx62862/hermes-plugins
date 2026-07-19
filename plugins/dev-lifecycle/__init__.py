"""dev-lifecycle — 软件开发生命周期技能包插件。

将 software-development 目录下 21 个技能组织为三段式生命周期：
  ideate (构思) → build (构建) → deliver (交付)

通过 dev_workflow 工具让 agent 导航开发流程，按阶段加载对应技能。
"""

from __future__ import annotations

import sys
from typing import Optional

sys.modules.setdefault("plugins.dev_lifecycle", sys.modules[__name__])

_plugin_ctx: Optional[object] = None

try:
    from .handlers import handle_dev_workflow, handle_on_session_start, warmup_skill_cache
    from .schemas import DEV_WORKFLOW_SCHEMA
except ImportError:
    from handlers import handle_dev_workflow, handle_on_session_start, warmup_skill_cache
    from schemas import DEV_WORKFLOW_SCHEMA

_TOOLS = (
    ("dev_workflow", DEV_WORKFLOW_SCHEMA, handle_dev_workflow, "🔄"),
)


def register(ctx) -> None:
    """Hermes 插件入口 — 注册工具和钩子。"""
    if ctx is None or not hasattr(ctx, "register_tool"):
        import logging
        logging.getLogger(__name__).warning("dev_lifecycle: ctx 无效或缺少 register_tool 方法，跳过注册")
        return
    global _plugin_ctx
    _plugin_ctx = ctx
    try:
        try:
            from . import handlers as _h
        except ImportError:
            import handlers as _h
        _h._plugin_ctx = ctx

        for name, schema, handler, emoji in _TOOLS:
            ctx.register_tool(
                name=name,
                toolset="dev-lifecycle",
                schema=schema,
                handler=handler,
                emoji=emoji,
            )

        ctx.register_hook("on_session_start", handle_on_session_start)

        warmup_skill_cache()

        try:
            from .handlers import init_modules
        except ImportError:
            from handlers import init_modules
        init_modules()
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("dev_lifecycle: register 失败: %s", e)
