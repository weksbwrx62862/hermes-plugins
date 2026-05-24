"""BudgetManager — Token 预算管理。

管理工作记忆的 Token 预算，确保 CoreBlock + Attachment 不超过限制。
"""

from __future__ import annotations

import logging
from typing import Any  # noqa: F401

logger = logging.getLogger(__name__)

# 每个字符约 0.25 Token（中文），0.25 Token（英文平均）
_CHARS_PER_TOKEN = 4


class BudgetManager:
    """Token 预算管理器。"""

    def __init__(self, max_tokens: int = 4000):
        self._max_tokens = max_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def estimate_tokens(self, text: str) -> int:
        """估算文本的 Token 数。简单实现：字符数 / 4。"""
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except (ImportError, Exception):
            return max(1, len(text) // _CHARS_PER_TOKEN)

    def trim_to_budget(self, items: list[dict[str, Any]], max_tokens: int = 0) -> list[dict[str, Any]]:
        """裁剪检索结果到 Token 预算内。

        items 中每项需有 "content" 字段。
        """
        budget = max_tokens or self._max_tokens
        result = []
        used = 0
        for item in items:
            content = item.get("content", "")
            tokens = self.estimate_tokens(content)
            if used + tokens <= budget:
                result.append(item)
                used += tokens
            else:
                # 尝试截断最后一条
                remaining = budget - used
                if remaining > 50:
                    char_budget = remaining * _CHARS_PER_TOKEN
                    item_copy = dict(item)
                    item_copy["content"] = content[:char_budget]
                    result.append(item_copy)
                break
        return result

    def fits(self, text: str, extra_tokens: int = 0) -> bool:
        """检查文本是否在预算内。"""
        return self.estimate_tokens(text) + extra_tokens <= self._max_tokens
