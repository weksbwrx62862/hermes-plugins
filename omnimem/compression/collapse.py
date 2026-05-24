"""Head-Tail Collapse — 首尾折叠压缩。

第二层压缩，零LLM调用：
  - 保留头部（开头几行，通常是上下文/任务描述）
  - 保留尾部（最近几行，通常是最新状态）
  - 中间部分折叠为摘要占位符
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def head_tail_collapse(
    lines: list[str],
    head_lines: int = 5,
    tail_lines: int = 10,
    collapse_marker: str = "[... middle section collapsed ...]",
) -> list[str]:
    """首尾折叠压缩。

    Args:
        lines: 文本行列表
        head_lines: 保留的头部行数
        tail_lines: 保留的尾部行数
        collapse_marker: 折叠标记

    Returns:
        压缩后的行列表
    """
    if len(lines) <= head_lines + tail_lines + 2:
        return lines

    head = lines[:head_lines]
    tail = lines[-tail_lines:]
    collapsed_count = len(lines) - head_lines - tail_lines

    result = head + [f"{collapse_marker} ({collapsed_count} lines)"] + tail
    return result
