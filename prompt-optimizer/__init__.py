"""prompt-optimizer — 提示词优化工作台 Hermes 插件。

核心功能：
  1. prompt_optimize  — 一键优化提示词（六维框架）
  2. prompt_analyze   — 提示词质量分析与评分
  3. prompt_compare   — A/B 对比评估
  4. prompt_garden    — Prompt Garden 资产管理
"""

from __future__ import annotations

import sys
from typing import Optional

sys.modules.setdefault("plugins.prompt_optimizer", sys.modules[__name__])

_plugin_ctx: Optional[object] = None

try:
    from .handlers import (
        handle_prompt_optimize,
        handle_prompt_analyze,
        handle_prompt_compare,
        handle_prompt_garden,
        handle_pre_llm_call,
    )
    from .schemas import (
        PROMPT_OPTIMIZE_SCHEMA,
        PROMPT_ANALYZE_SCHEMA,
        PROMPT_COMPARE_SCHEMA,
        PROMPT_GARDEN_SCHEMA,
    )
except ImportError:
    from handlers import (
        handle_prompt_optimize,
        handle_prompt_analyze,
        handle_prompt_compare,
        handle_prompt_garden,
        handle_pre_llm_call,
    )
    from schemas import (
        PROMPT_OPTIMIZE_SCHEMA,
        PROMPT_ANALYZE_SCHEMA,
        PROMPT_COMPARE_SCHEMA,
        PROMPT_GARDEN_SCHEMA,
    )

_TOOLS = (
    ("prompt_optimize", PROMPT_OPTIMIZE_SCHEMA, handle_prompt_optimize, "🔧"),
    ("prompt_analyze", PROMPT_ANALYZE_SCHEMA, handle_prompt_analyze, "📊"),
    ("prompt_compare", PROMPT_COMPARE_SCHEMA, handle_prompt_compare, "⚖️"),
    ("prompt_garden", PROMPT_GARDEN_SCHEMA, handle_prompt_garden, "🌱"),
)


def register(ctx) -> None:
    """Hermes 插件入口 — 注册工具。"""
    global _plugin_ctx

    # ctx 校验：容忍 None / 非法类型，避免插件加载器崩溃
    if ctx is None or not hasattr(ctx, "register_tool"):
        import logging
        logging.getLogger(__name__).warning(
            "prompt-optimizer: ctx 无效或缺少 register_tool 方法，跳过注册"
        )
        return

    try:
        _plugin_ctx = ctx

        for name, schema, handler, emoji in _TOOLS:
            ctx.register_tool(
                name=name,
                toolset="prompt-optimizer",
                schema=schema,
                handler=handler,
                emoji=emoji,
            )

        # 注册 pre_llm_call hook — 自动检测低质量提示词
        ctx.register_hook("pre_llm_call", handle_pre_llm_call)
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception(
            "prompt-optimizer: register 失败: %s", e
        )
