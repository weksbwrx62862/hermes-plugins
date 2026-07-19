"""CircuitBreaker 单元测试。

覆盖：
  - 状态机转换 CLOSED → OPEN → HALF_OPEN → CLOSED
  - 连续失败阈值配置
  - 指数退避冷却时间
  - HALF_OPEN 探测成功恢复与探测失败重新 OPEN
  - call() 自动记录成功/失败
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

# 加载被测模块（目录名含连字符，使用 importlib）
_cb_path = Path(__file__).resolve().parent.parent / "circuit_breaker.py"
_spec = importlib.util.spec_from_file_location("circuit_breaker", str(_cb_path))
circuit_breaker = importlib.util.module_from_spec(_spec)
sys.modules["circuit_breaker"] = circuit_breaker
_spec.loader.exec_module(circuit_breaker)

CircuitBreaker = circuit_breaker.CircuitBreaker
CircuitBreakerOpenError = circuit_breaker.CircuitBreakerOpenError


class TestCircuitBreakerStateMachine:
    """熔断器状态机测试。"""

    def test_initial_state(self):
        """初始状态为 CLOSED。"""
        cb = CircuitBreaker()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.can_execute() is True

    def test_closed_to_open_after_threshold(self):
        """连续失败达到阈值后进入 OPEN。"""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.CLOSED
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        assert cb.can_execute() is False

    def test_open_to_half_open_after_cooldown(self):
        """OPEN 冷却结束后进入 HALF_OPEN。"""
        cb = CircuitBreaker(
            failure_threshold=3, base_cooldown=0.05, max_cooldown=1.0
        )
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        assert cb.can_execute() is False
        time.sleep(0.06)
        assert cb.can_execute() is True
        assert cb.state == CircuitBreaker.HALF_OPEN

    def test_half_open_success_to_closed(self):
        """HALF_OPEN 探测成功后恢复 CLOSED。"""
        cb = CircuitBreaker(
            failure_threshold=3, base_cooldown=0.05, max_cooldown=1.0
        )
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.06)
        cb.can_execute()
        assert cb.state == CircuitBreaker.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.can_execute() is True

    def test_half_open_failure_back_to_open(self):
        """HALF_OPEN 探测失败重新进入 OPEN。"""
        cb = CircuitBreaker(
            failure_threshold=3, base_cooldown=0.05, max_cooldown=1.0
        )
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.06)
        cb.can_execute()
        assert cb.state == CircuitBreaker.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

    def test_success_decreases_failure_count(self):
        """CLOSED 状态下成功会逐渐减少失败计数。"""
        cb = CircuitBreaker(failure_threshold=4)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb._failure_count == 3
        cb.record_success()
        assert cb._failure_count == 2


class TestCircuitBreakerBackoff:
    """指数退避测试。"""

    def test_initial_cooldown(self):
        """首次 OPEN 后冷却时间为基础值。"""
        cb = CircuitBreaker(failure_threshold=2, base_cooldown=30.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        assert cb._current_cooldown() == 30.0

    def test_exponential_backoff(self):
        """重新 OPEN 后冷却时间指数翻倍。"""
        cb = CircuitBreaker(
            failure_threshold=2,
            base_cooldown=30.0,
            max_cooldown=300.0,
        )
        # 第一次 OPEN
        cb.record_failure()
        cb.record_failure()
        assert cb._current_cooldown() == 30.0

        # HALF_OPEN 后再失败，额外失败 1 次，冷却 60s
        cb._state = CircuitBreaker.HALF_OPEN
        cb.record_failure()
        assert cb._current_cooldown() == 60.0

        # 再 HALF_OPEN 再失败，额外失败 2 次，冷却 120s
        cb._state = CircuitBreaker.HALF_OPEN
        cb.record_failure()
        assert cb._current_cooldown() == 120.0

    def test_max_cooldown_cap(self):
        """冷却时间不超过上限。"""
        cb = CircuitBreaker(
            failure_threshold=1,
            base_cooldown=30.0,
            max_cooldown=90.0,
        )
        cb.record_failure()
        assert cb._current_cooldown() == 30.0

        cb._state = CircuitBreaker.HALF_OPEN
        cb.record_failure()
        assert cb._current_cooldown() == 60.0

        cb._state = CircuitBreaker.HALF_OPEN
        cb.record_failure()
        assert cb._current_cooldown() == 90.0

        cb._state = CircuitBreaker.HALF_OPEN
        cb.record_failure()
        # 达到上限后不再翻倍
        assert cb._current_cooldown() == 90.0


class TestCircuitBreakerCall:
    """call() 接口测试。"""

    def test_call_success(self):
        """成功调用返回结果并记录成功。"""
        cb = CircuitBreaker(failure_threshold=3)
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb.state == CircuitBreaker.CLOSED

    def test_call_failure(self):
        """异常调用记录失败并重抛异常。"""
        cb = CircuitBreaker(failure_threshold=3)

        def boom():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            cb.call(boom)
        assert cb._failure_count == 1

    def test_call_open_raises(self):
        """OPEN 状态下 call() 直接抛出 CircuitBreakerOpenError。"""
        cb = CircuitBreaker(failure_threshold=1, base_cooldown=10.0)
        cb.record_failure()
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(lambda: 1)

    def test_call_half_open_recovery(self):
        """call() 在半开状态下成功探测后恢复。"""
        cb = CircuitBreaker(
            failure_threshold=2, base_cooldown=0.05, max_cooldown=1.0
        )
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.06)
        result = cb.call(lambda: "ok")
        assert result == "ok"
        assert cb.state == CircuitBreaker.CLOSED


class TestCircuitBreakerConfig:
    """构造参数校验。"""

    def test_invalid_failure_threshold(self):
        with pytest.raises(ValueError):
            CircuitBreaker(failure_threshold=0)

    def test_invalid_cooldown(self):
        with pytest.raises(ValueError):
            CircuitBreaker(base_cooldown=0)
        with pytest.raises(ValueError):
            CircuitBreaker(max_cooldown=0)

    def test_state_summary(self):
        cb = CircuitBreaker(name="test-cb", failure_threshold=2)
        summary = cb.get_state_summary()
        assert summary["name"] == "test-cb"
        assert summary["state"] == CircuitBreaker.CLOSED
        assert summary["failure_threshold"] == 2
