"""Priority Post-Processing — 优先级确定性后处理。

第五层压缩，零LLM调用（参考 Claw Code 设计）：
  - 按优先级排序
  - 确定性裁剪（无 LLM 再处理）
  - 优先级 0-3 分级

优先级定义:
  0 (关键): 纠正、决策、用户偏好 — 永不裁剪
  1 (重要): 事实、技能 — 优先保留
  2 (一般): 事件、观察 — 预算不足时裁剪
  3 (低): 闲聊、重复 — 首先裁剪
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 记忆类型 → 优先级映射
_TYPE_PRIORITY = {
    "correction": 0,
    "preference": 0,
    "fact": 1,
    "skill": 1,
    "procedural": 1,
    "event": 2,
}

# 关键标记 → 优先级提升
_PRIORITY_MARKERS = {
    "CORRECTION:": 0,
    "REINFORCED:": 0,
    "DECISION:": 0,
    "IMPORTANT:": 0,
    "纠正:": 0,
    "决定:": 0,
}


def priority_compress(
    items: list[dict[str, Any]],
    max_tokens: int = 4000,
) -> list[dict[str, Any]]:
    """优先级确定性后处理。

    Args:
        items: 记忆项列表（每项有 "content", "type" 等字段）
        max_tokens: Token 预算

    Returns:
        按优先级裁剪后的列表
    """
    # 1. 为每项计算优先级
    for item in items:
        item["_priority"] = _compute_priority(item)

    # 2. 按优先级排序（0最高）
    sorted_items = sorted(items, key=lambda x: x.get("_priority", 3))

    # 3. 按预算裁剪
    chars_per_token = 4
    budget = max_tokens
    result = []
    used = 0

    for item in sorted_items:
        content = item.get("content", "")
        est_tokens = max(1, len(content) // chars_per_token)

        # 优先级 0 的项目：即使超预算也保留
        if item.get("_priority") == 0:
            result.append(item)
            used += est_tokens
            continue

        # 其他项目：按预算裁剪
        if used + est_tokens <= budget:
            result.append(item)
            used += est_tokens
        # 预算不足时停止

    # 4. 清理临时字段
    for item in result:
        item.pop("_priority", None)

    return result


def _compute_priority(item: dict[str, Any]) -> int:
    """计算单个记忆项的优先级。"""
    content = item.get("content", "")
    memory_type = item.get("type", "fact")
    confidence = item.get("confidence", 3)

    # 关键标记检测
    for marker, priority in _PRIORITY_MARKERS.items():
        if marker in content:
            return priority

    # 类型优先级
    type_priority = _TYPE_PRIORITY.get(memory_type, 2)

    # 高置信度提升优先级
    if confidence >= 4 and type_priority > 0:
        type_priority -= 1

    # 低置信度降低优先级
    if confidence <= 1 and type_priority < 3:
        type_priority += 1

    return type_priority
