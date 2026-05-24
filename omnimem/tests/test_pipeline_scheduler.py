"""Task 3: 自动 Pipeline 调度器 — 单元测试。"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


class TestPipelineScheduler:
    """测试 PipelineScheduler：L3 双条件触发、Timer TTL、flush。"""

    @pytest.fixture
    def scheduler(self):
        from omnimem.core.pipeline_scheduler import PipelineScheduler
        config = {
            "persona_trigger_every_n": 3,
            "persona_min_interval_seconds": 1,  # 1秒便于测试
            "pipeline_l2_delay_after_l1_seconds": 90,
        }
        reflect_fn = MagicMock()
        # 提供一个同步执行的 bg_executor（直接执行任务）
        mock_executor = MagicMock()
        mock_executor.submit = lambda fn, *args, **kwargs: fn(*args, **kwargs)
        return PipelineScheduler(config=config, reflect_fn=reflect_fn, bg_executor=mock_executor)

    def test_on_new_memory_count_trigger(self, scheduler):
        """达到 count 阈值 + min_interval 后应触发 L3。"""
        scheduler.on_new_memory("s1")
        scheduler.on_new_memory("s1")
        # 还没到 3，不应触发
        assert scheduler._reflect_fn.call_count == 0

        scheduler.on_new_memory("s1")
        # 到了 3，但 min_interval 可能未满足
        # 在 fixture 中 min_interval=1，第一次触发时间是 now
        # 所以第 3 次时 interval 可能 < 1s
        # 让我们手动设置 last_persona_time 到过去
        scheduler._last_persona_time["s1"] = time.time() - 10
        scheduler.on_new_memory("s1")  # 第 4 次，count=4 >= 3
        assert scheduler._reflect_fn.call_count == 1

    def test_on_new_memory_min_interval_blocks(self, scheduler):
        """min_interval 未满足时不应触发。"""
        scheduler.on_new_memory("s1")
        scheduler.on_new_memory("s1")
        scheduler.on_new_memory("s1")
        # count=3 >= 3, 但 min_interval=1s 未满足（刚写入）
        # 不应触发
        assert scheduler._reflect_fn.call_count == 0

    def test_flush_session_schedules_l3(self, scheduler):
        """flush_session 应为未处理的新记忆触发 L3。"""
        scheduler.on_new_memory("s1")
        scheduler.on_new_memory("s1")
        scheduler.flush_session("s1")
        # 有 2 条未处理，应触发 L3
        assert scheduler._reflect_fn.call_count == 1

    def test_flush_session_cleans_up(self, scheduler):
        """flush_session 应清理计数器。"""
        scheduler.on_new_memory("s1")
        scheduler.flush_session("s1")
        assert "s1" not in scheduler._new_memory_counts
        assert "s1" not in scheduler._last_persona_time

    def test_flush_session_empty(self, scheduler):
        """空 session flush 不应触发 L3。"""
        scheduler.flush_session("empty")
        assert scheduler._reflect_fn.call_count == 0

    def test_timer_self_cleanup(self, scheduler):
        """Timer 回调后应自动清理引用。"""
        scheduler.schedule_l2_after_l1("s1")
        assert "s1" in scheduler._l2_timers

        # flush 应取消 timer
        scheduler.flush_session("s1")
        assert "s1" not in scheduler._l2_timers

    def test_config_schema_entries(self):
        """验证 Pipeline 配置在 schema 中。"""
        from omnimem.config._config import _CONFIG_SCHEMA

        assert "pipeline_l2_delay_after_l1_seconds" in _CONFIG_SCHEMA
        assert _CONFIG_SCHEMA["pipeline_l2_delay_after_l1_seconds"]["default"] == 90

        assert "persona_trigger_every_n" in _CONFIG_SCHEMA
        assert _CONFIG_SCHEMA["persona_trigger_every_n"]["default"] == 15

        assert "persona_min_interval_seconds" in _CONFIG_SCHEMA
        assert _CONFIG_SCHEMA["persona_min_interval_seconds"]["default"] == 300

    def test_no_bg_executor_graceful(self):
        """没有 bg_executor 时应优雅降级（不崩溃）。"""
        from omnimem.core.pipeline_scheduler import PipelineScheduler
        config = {"persona_trigger_every_n": 1, "persona_min_interval_seconds": 0}
        scheduler = PipelineScheduler(config=config, bg_executor=None)
        scheduler._last_persona_time["s1"] = 0
        # 不应崩溃
        scheduler.on_new_memory("s1")
        scheduler.flush_session("s1")
