"""PipelineScheduler — 自动 Pipeline 调度器。

参考 TencentDB MemoryPipelineManager，适配 OmniMem 现有架构：

适配说明：
- L0 感知已有：sync_turn() → Perception Engine（provider.py:536-547）
- L1 蒸馏已有：sync_turn() → DistillationEngine（provider.py:556-566）
- 本调度器只补 L2 场景归纳 + L3 画像触发的调度逻辑
- 复用 provider 的 _bg_executor，不自建线程池
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class PipelineScheduler:
    """自动 Pipeline 调度器 — 只补 L2/L3，不重复 L0/L1。

    调度流程：
    1. sync_turn → L0 感知（已有） → L1 蒸馏（已有）
    2. L1 完成后延迟触发 L2 场景归纳（新增）
    3. 新记忆达到阈值 → L3 画像生成（新增）

    配置项：
    - pipeline_l2_delay_after_l1_seconds: L1 后延迟触发 L2（默认 90）
    - persona_trigger_every_n: 每 N 条新记忆触发画像（默认 50）
    - persona_min_interval_seconds: L3 画像最小触发间隔（默认 300）
    """

    def __init__(
        self,
        config: Any,
        logger: Any = None,
        bg_executor: Any = None,
        reflect_fn: Callable | None = None,
    ):
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        self._bg_executor = bg_executor  # 复用 provider 的 _bg_executor
        self._reflect_fn = reflect_fn

        # 每 session 的新记忆计数
        self._new_memory_counts: dict[str, int] = {}
        # L3 画像上次触发时间
        self._last_persona_time: dict[str, float] = {}
        # L2 延迟 Timer
        self._l2_timers: dict[str, threading.Timer] = {}

    def on_new_memory(self, session_key: str) -> None:
        """新记忆写入后检查是否触发 L3 画像。

        双条件触发：count >= persona_trigger_every_n AND
                    time_since_last >= persona_min_interval_seconds
        """
        self._new_memory_counts[session_key] = self._new_memory_counts.get(session_key, 0) + 1
        self._last_persona_time.setdefault(session_key, time.time())

        count_trigger = self._config.get("persona_trigger_every_n", 50)
        min_interval = self._config.get("persona_min_interval_seconds", 300)
        now = time.time()

        if (self._new_memory_counts[session_key] >= count_trigger
                and now - self._last_persona_time[session_key] >= min_interval):
            self._schedule_l3_persona(session_key)
            self._new_memory_counts[session_key] = 0
            self._last_persona_time[session_key] = now

    def schedule_l2_after_l1(self, session_key: str) -> None:
        """L1 蒸馏完成后，延迟调度 L2 场景归纳。

        由 DistillationEngine 的回调或 sync_turn 的蒸馏完成后触发。
        """
        delay = self._config.get("pipeline_l2_delay_after_l1_seconds", 90)
        self._set_timer(
            self._l2_timers, session_key, delay,
            lambda: self._schedule_l2(session_key),
        )

    def _schedule_l2(self, session_key: str) -> None:
        """调度 L2 场景归纳（后台执行）。"""
        if self._bg_executor:
            self._bg_executor.submit(self._do_l2_scenario, session_key)
        else:
            self._logger.warning("PipelineScheduler: no bg_executor for L2, skipped")

    def _do_l2_scenario(self, session_key: str) -> None:
        """L2 场景归纳：从近期记忆中提取场景模式。

        当前实现：调用 reflect_fn 触发反思（复用现有 LLM 能力）。
        未来可扩展：独立的场景归纳引擎。
        """
        if self._reflect_fn:
            try:
                self._reflect_fn({"query": "最近的对话场景和模式", "disposition": {"skepticism": 2}})
                self._logger.info("L2 scenario induction completed for session %s", session_key)
            except Exception as e:
                self._logger.warning("L2 scenario induction failed: %s", e)

    def _schedule_l3_persona(self, session_key: str) -> None:
        """调度 L3 画像生成（后台执行）。"""
        if self._bg_executor:
            self._bg_executor.submit(self._do_l3_persona, session_key)
        else:
            self._logger.warning("PipelineScheduler: no bg_executor for L3, skipped")

    def _do_l3_persona(self, session_key: str) -> None:
        """L3 画像生成：调用 reflect 生成用户画像/心智模型。"""
        if self._reflect_fn:
            try:
                self._reflect_fn({"query": "用户画像和偏好模式", "disposition": {"empathy": 4}})
                self._logger.info("L3 persona generation completed for session %s", session_key)
            except Exception as e:
                self._logger.warning("L3 persona generation failed: %s", e)

    def _set_timer(
        self, timers_dict: dict, key: str, delay: float, callback: Callable
    ) -> None:
        """统一的 Timer 设置，带 TTL 保护。"""
        # 取消已有 timer
        old = timers_dict.pop(key, None)
        if old and old.is_alive():
            old.cancel()

        def _wrapper():
            timers_dict.pop(key, None)  # 自清理引用
            callback()

        t = threading.Timer(delay, _wrapper)
        t.daemon = True
        t.start()
        timers_dict[key] = t

    def flush_session(self, session_key: str) -> None:
        """会话结束时刷新所有待处理任务。

        取消所有 timer，执行剩余计数器。
        """
        # 取消 L2 timer
        timer = self._l2_timers.pop(session_key, None)
        if timer and timer.is_alive():
            timer.cancel()

        # 如果有未触发的 L3，立即触发
        count = self._new_memory_counts.get(session_key, 0)
        if count > 0:
            self._logger.info(
                "Session %s ending with %d unprocessed new memories, scheduling L3",
                session_key, count,
            )
            self._schedule_l3_persona(session_key)

        # 清理计数器
        self._new_memory_counts.pop(session_key, None)
        self._last_persona_time.pop(session_key, None)
