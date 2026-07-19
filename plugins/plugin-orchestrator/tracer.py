"""
Tracer — 轻量级钩子追踪（OpenTelemetry 风格）

每个钩子调用自动生成 Span，记录：
  - 插件名 + 钩子名
  - 耗时（ms）
  - 成功/失败
  - 返回值大小（transform 钩子）

结果存储在全局 TraceStore 中，供 diag.py 查询。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Span ───────────────────────────────────────────────────────────────

class Span:
    """单个钩子调用的追踪跨度。"""

    __slots__ = (
        "hook_name", "plugin_name", "session_id",
        "_start", "_end", "_success", "_result_size",
        "_error", "_turn",
    )

    def __init__(self, hook_name: str, plugin_name: str, session_id: str = "", turn: int = 0):
        self.hook_name = hook_name
        self.plugin_name = plugin_name
        self.session_id = session_id
        self._start = time.time()
        self._end: Optional[float] = None
        self._success: Optional[bool] = None
        self._result_size: int = 0
        self._error: Optional[str] = None
        self._turn = turn

    def end(self, success: bool = True, result_size: int = 0, error: Optional[str] = None):
        self._end = time.time()
        self._success = success
        self._result_size = result_size
        self._error = error

    @property
    def duration_ms(self) -> float:
        if self._end and self._start:
            return (self._end - self._start) * 1000
        return 0.0

    def dump(self) -> dict:
        return {
            "hook": self.hook_name,
            "plugin": self.plugin_name,
            "session": self.session_id[:12] if self.session_id else "",
            "turn": self._turn,
            "duration_ms": round(self.duration_ms, 2),
            "success": self._success,
            "result_size": self._result_size,
            "error": self._error,
        }


# ── TraceStore ─────────────────────────────────────────────────────────

class TraceStore:
    """全局追踪存储。每个 span 追加到列表，最多保留最近 5000 条。"""

    def __init__(self, max_spans: int = 5000):
        self._spans: List[Span] = []
        self._max_spans = max_spans
        self._lock = threading.Lock()

    def record(self, span: Span):
        with self._lock:
            self._spans.append(span)
            if len(self._spans) > self._max_spans:
                self._spans = self._spans[-self._max_spans:]

    def recent(self, limit: int = 50) -> List[dict]:
        with self._lock:
            return [s.dump() for s in self._spans[-limit:]]

    def stats(self) -> dict:
        """聚合统计：按插件分组，平均耗时、成功率等。"""
        with self._lock:
            if not self._spans:
                return {"total_spans": 0, "by_plugin": {}}

            by_plugin: Dict[str, dict] = {}
            for s in self._spans:
                key = s.plugin_name
                if key not in by_plugin:
                    by_plugin[key] = {
                        "calls": 0, "failures": 0, "total_duration_ms": 0.0,
                        "hooks": set(),
                    }
                info = by_plugin[key]
                info["calls"] += 1
                if s._success is False:
                    info["failures"] += 1
                info["total_duration_ms"] += s.duration_ms
                info["hooks"].add(s.hook_name)

            # 转换为可序列化结构
            serializable = {}
            for plugin, info in by_plugin.items():
                serializable[plugin] = {
                    "calls": info["calls"],
                    "failures": info["failures"],
                    "avg_duration_ms": (
                        round(info["total_duration_ms"] / info["calls"], 2)
                        if info["calls"] else 0
                    ),
                    "total_duration_ms": round(info["total_duration_ms"], 2),
                    "hooks": list(info["hooks"]),
                    "success_rate": round(
                        (info["calls"] - info["failures"]) / info["calls"] * 100, 1
                    ) if info["calls"] else 0,
                }

            return {
                "total_spans": len(self._spans),
                "total_failures": sum(
                    info["failures"] for info in serializable.values()
                ),
                "by_plugin": serializable,
            }

    def clear(self):
        with self._lock:
            self._spans.clear()


# ── 全局单例 ──────────────────────────────────────────────────────────

_global_store: Optional[TraceStore] = None
_store_lock = threading.Lock()


def get_trace_store() -> TraceStore:
    global _global_store
    if _global_store is None:
        with _store_lock:
            if _global_store is None:
                _global_store = TraceStore()
    return _global_store
