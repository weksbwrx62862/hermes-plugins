"""OmniMem govern 处理器。

仅保留：schema 校验、依赖注入、调用 GovernanceService、结果序列化。
核心业务逻辑已下沉到 services.governance_service。
"""

from __future__ import annotations

import json
from typing import Any, cast

from omnimem.handlers.deps import extract_deps
from omnimem.services.governance_service import (
    ActionRegistry,
    GovernanceService,
)
from omnimem.services.governance_service import (
    _scan_memory_conflicts as _service_scan_memory_conflicts,
)

# 向后兼容：旧代码仍可通过本模块注册/注销自定义治理动作
_registry = ActionRegistry()


def register_action(name: str, handler: Any) -> None:
    """注册自定义治理动作（已弃用，建议直接扩展 GovernanceService）。"""
    _registry.register(name, handler)


def unregister_action(name: str) -> None:
    """注销自定义治理动作。"""
    _registry.unregister(name)


def _scan_memory_conflicts(provider: Any) -> list[dict[str, Any]]:
    """主动扫描所有记忆，检测同主题的矛盾对。

    为保持向后兼容，仍从 handler 暴露；内部委托给 GovernanceService 的实现。
    """
    deps = extract_deps(provider)
    return cast(list[dict[str, Any]], _service_scan_memory_conflicts(deps))


def _validate_govern_args(args: dict[str, Any]) -> str | None:
    """对 omni_govern 参数做轻量 schema 校验。"""
    if not isinstance(args, dict):
        return "args must be a dict"
    if (
        "action" not in args
        or not isinstance(args["action"], str)
        or not args["action"].strip()
    ):
        return "action is required and must be a non-empty string"
    return None


def handle_govern(provider: Any, args: dict[str, Any]) -> str:
    """处理 omni_govern 工具调用。

    Args:
        provider: OmniMemProvider 实例，用于访问子组件。
        args: 工具调用参数。

    Returns:
        JSON 字符串。
    """
    validation_error = _validate_govern_args(args)
    if validation_error:
        return json.dumps({"status": "error", "reason": validation_error})

    deps = extract_deps(provider)
    service = GovernanceService(deps=deps)
    return cast(str, service.handle(args))
