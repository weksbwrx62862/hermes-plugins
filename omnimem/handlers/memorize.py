"""OmniMem memorize 处理器。

仅保留：schema 校验、依赖注入、调用 MemoryWriteService、结果序列化。
核心业务逻辑已下沉到 services.memory_write_service。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from omnimem.handlers.deps import extract_deps
from omnimem.services.memory_write_service import (
    MemoryWriteService,
    get_background_executor,
    shutdown_background_executor,
)

logger = logging.getLogger(__name__)

# 保持向后兼容：外部模块仍可从本模块导入后台线程池管理函数
__all__ = ["handle_memorize", "get_background_executor", "shutdown_background_executor"]


def _validate_memorize_args(args: dict[str, Any]) -> str | None:
    """对 omni_memorize 参数做轻量 schema 校验。

    返回错误信息，无错误返回 None。
    """
    if not isinstance(args, dict):
        return "args must be a dict"
    if "content" not in args or not isinstance(args["content"], str) or not args["content"].strip():
        return "content is required and must be a non-empty string"

    memory_type = args.get("memory_type", "fact")
    if memory_type not in {
        "fact",
        "preference",
        "correction",
        "skill",
        "procedural",
        "event",
        "action",
        "reasoning",
    }:
        return f"invalid memory_type: {memory_type}"

    confidence = args.get("confidence", 3)
    if not isinstance(confidence, int) or not 1 <= confidence <= 5:
        return f"confidence must be an integer between 1 and 5, got {confidence}"

    privacy = args.get("privacy", "personal")
    if privacy not in {"public", "team", "personal", "secret"}:
        return f"invalid privacy: {privacy}"

    scope = args.get("scope", "personal")
    if scope not in {"personal", "project", "shared"}:
        return f"invalid scope: {scope}"

    return None


def handle_memorize(provider: Any, args: dict[str, Any], llm_memory_manager: Any = None) -> str:
    """处理 omni_memorize 工具调用。

    Args:
        provider: OmniMemProvider 实例，用于访问子组件
        args: 工具调用参数
        llm_memory_manager: 可选 LLM 决策管理器

    Returns:
        JSON 字符串
    """
    validation_error = _validate_memorize_args(args)
    if validation_error:
        return json.dumps({"status": "error", "reason": validation_error})

    deps = extract_deps(provider)
    service = MemoryWriteService(
        deps=deps,
        llm_memory_manager=llm_memory_manager,
        bg_executor=deps.bg_executor,
        turn_count=getattr(provider, "_turn_count", 0),
    )
    result = service.handle(args)
    return json.dumps(result, ensure_ascii=False)
