"""Microcompact — 微压缩：感知标记 + 去噪。

第一层压缩，零LLM调用：
  - 去除重复行
  - 去除空白/无意义行
  - 保留关键标记（决策、修正、偏好）
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# 关键标记（这些行永远不会被压缩删除）
_KEY_MARKERS = [
    "CORRECTION:",
    "REINFORCED:",
    "DECISION:",
    "PREFERENCE:",
    "IMPORTANT:",
    "NOTE:",
    "WARNING:",
    "TODO:",
    "纠正:",
    "决定:",
    "偏好:",
    "重要:",
    "注意:",
]


def microcompact(lines: list[str]) -> list[str]:
    """微压缩：去噪 + 感知标记。

    Args:
        lines: 文本行列表

    Returns:
        压缩后的行列表
    """
    result: list[str] = []
    seen = set()

    for line in lines:
        stripped = line.strip()

        # 保留空行（结构分隔），但最多连续1个
        if not stripped:
            if result and not result[-1].strip():
                continue
            result.append("")
            continue

        # 关键标记行：始终保留
        if any(marker in stripped for marker in _KEY_MARKERS):
            result.append(line)
            seen.add(stripped)
            continue

        # 去重：跳过重复行
        if stripped in seen:
            continue
        seen.add(stripped)

        # 去噪：跳过纯符号/无意义行
        if _is_noise(stripped):
            continue

        result.append(line)

    return result


def _is_noise(line: str) -> bool:
    """判断一行是否是噪声。"""
    # 纯符号行
    if re.match(r"^[\s\-\*\=\#\|]+$", line):
        return True
    # 太短的行（少于3个有效字符）
    clean = re.sub(r"[\s\-\*\#\|]", "", line)
    return len(clean) < 3
