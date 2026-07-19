"""TemporalDecay — 时间衰减。

时间衰减公式: score *= exp(-ln2 * days / half_life)
不同记忆类型有不同的半衰期：
  - fact/correction: 不衰减（半衰期无穷大）
  - preference: 180天
  - event: 90天
  - skill/procedural: 365天
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# 各类型的半衰期（天）
HALF_LIVES = {
    "fact": float("inf"),
    "correction": float("inf"),
    "preference": 180,
    "event": 90,
    "skill": 365,
    "procedural": 365,
    "action": 30,       # 行为记忆快衰减 — 操作时效性强
    "reasoning": 90,    # 经验推理中衰减 — 踩坑经验有时效
}


class TemporalDecay:
    """时间衰减引擎。"""

    def __init__(self, custom_half_lives: dict[str, float] | None = None):
        self._half_lives = {**HALF_LIVES}
        if custom_half_lives:
            self._half_lives.update(custom_half_lives)
        # ★ 时间字符串解析缓存：避免重复解析相同的 stored_at
        self._parse_cache: dict[str, datetime] = {}
        self._parse_cache_max = 512

    def apply(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """对检索结果应用时间衰减。

        Args:
            results: 检索结果列表（每项有 "stored_at", "type", "score" 字段）

        Returns:
            衰减后的结果（按 score 降序）
        """
        now = datetime.now(timezone.utc)

        for r in results:
            stored_at = r.get("stored_at")
            if not stored_at:
                continue

            # 解析时间（带缓存）
            try:
                if isinstance(stored_at, str):
                    stored_dt = self._parse_cache.get(stored_at)
                    if stored_dt is None:
                        stored_dt = datetime.fromisoformat(stored_at.replace("Z", "+00:00"))
                        if len(self._parse_cache) < self._parse_cache_max:
                            self._parse_cache[stored_at] = stored_dt
                elif isinstance(stored_at, datetime):
                    stored_dt = stored_at
                else:
                    continue
            except (ValueError, TypeError):
                continue

            # 计算天数
            if stored_dt.tzinfo is None:
                stored_dt = stored_dt.replace(tzinfo=timezone.utc)
            days = (now - stored_dt).days
            if days <= 0:
                continue

            # 应用衰减
            memory_type = r.get("type", "fact")
            half_life = self._half_lives.get(memory_type, 365)

            if half_life != float("inf"):
                decay_factor = math.exp(-math.log(2) * days / half_life)
                current_score = r.get("score", 1.0)
                r["score"] = current_score * decay_factor
                r["decay_factor"] = decay_factor

        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    def get_half_life(self, memory_type: str) -> float:
        """获取某类型的半衰期。"""
        return self._half_lives.get(memory_type, 365)

    def set_half_life(self, memory_type: str, days: float) -> None:
        """设置某类型的半衰期。"""
        self._half_lives[memory_type] = days
