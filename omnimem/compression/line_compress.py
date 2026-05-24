"""Structured Line Compress — 结构化行压缩。

第三层压缩，零LLM调用：
  - 合并相似行
  - 提取行内关键信息
  - 结构化格式压缩
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def structured_line_compress(lines: list[str]) -> list[str]:
    """结构化行压缩。

    Args:
        lines: 文本行列表

    Returns:
        压缩后的行列表
    """
    result = []
    prev_topic = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue

        # 折叠标记行：直接保留
        if stripped.startswith("[..."):
            result.append(line)
            continue

        # 提取主题
        topic = _extract_topic(stripped)

        # 如果与上一行主题相同，尝试合并
        if topic and topic == prev_topic and result:
            # 合并到上一行
            prev = result[-1].strip()
            if not prev.endswith("..."):
                result[-1] = prev + " | " + _compress_line(stripped)
            continue

        # 压缩单行
        compressed = _compress_line(stripped)
        result.append(compressed)
        prev_topic = topic

    return result


def _extract_topic(line: str) -> str:
    """提取行的主题关键词。"""
    # 提取第一个有意义的词/短语
    # 中文：前2-4字
    zh_match = re.match(r"[\u4e00-\u9fff]{2,4}", line)
    if zh_match:
        return zh_match.group()

    # 英文：第一个词
    en_match = re.match(r"[a-zA-Z]+", line)
    if en_match:
        return en_match.group().lower()

    return ""


def _compress_line(line: str) -> str:
    """压缩单行：去除冗余信息，保留关键内容。"""
    # 去除连续空白
    compressed = re.sub(r"\s+", " ", line)

    # 去除常见冗余短语
    redundant = [
        "I think ",
        "I believe ",
        "在我看来",
        "我觉得",
        "basically ",
        "actually ",
        "实际上",
        "基本上",
    ]
    for phrase in redundant:
        compressed = compressed.replace(phrase, "")

    # 截断过长行
    if len(compressed) > 200:
        compressed = compressed[:197] + "..."

    return compressed.strip()
