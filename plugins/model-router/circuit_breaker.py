"""Provider 级 CircuitBreaker 实现。

状态机：
    CLOSED ──连续失败达到阈值──→ OPEN
    OPEN ──冷却时间到──→ HALF_OPEN
    HALF_OPEN ──探测成功──→ CLOSED
    HALF_OPEN ──探测失败──→ OPEN

冷却时间使用指数退避：首次打开后等待 base_cooldown，
每次重新打开时翻倍，直至 max_cooldown 上限。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitBreakerOpenError(Exception):
    """熔断器打开时拒绝执行的异常。"""


class CircuitBreaker:
    """Provider 级 CircuitBreaker。

    参数:
        name: 熔断器名称/标识，用于日志与异常信息
        failure_threshold: 连续失败阈值，达到后进入 OPEN，默认 5
        base_cooldown: 首次 OPEN 后的基础冷却时间（秒），默认 30
        max_cooldown: 最大冷却时间（秒），默认 300
        half_open_successes: HALF_OPEN 状态下需要连续成功次数才能关闭，默认 1
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        name: str = "",
        failure_threshold: int = 5,
        base_cooldown: float = 30.0,
        max_cooldown: float = 300.0,
        half_open_successes: int = 1,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold 必须大于 0")
        if base_cooldown <= 0 or max_cooldown <= 0:
            raise ValueError("冷却时间必须大于 0")

        self.name = name or "circuit_breaker"
        self.failure_threshold = int(failure_threshold)
        self.base_cooldown = float(base_cooldown)
        self.max_cooldown = float(max_cooldown)
        self.half_open_successes = max(1, int(half_open_successes))

        self._state = self.CLOSED
        self._failure_count = 0
        self._half_open_success_count = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.RLock()

    @property
    def state(self) -> str:
        """当前熔断器状态。"""
        with self._lock:
            return self._state

    def can_execute(self) -> bool:
        """判断当前是否允许执行请求。

        OPEN 状态下若冷却时间已到，自动转为 HALF_OPEN 并允许一次探测。
        """
        with self._lock:
            if self._state == self.CLOSED:
                return True

            if self._state == self.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                cooldown = self._current_cooldown()
                if elapsed >= cooldown:
                    logger.debug(
                        "CircuitBreaker %s: 冷却完成，进入 HALF_OPEN", self.name
                    )
                    self._state = self.HALF_OPEN
                    self._half_open_success_count = 0
                    return True
                return False

            # HALF_OPEN：允许探测请求
            return True

    def record_success(self) -> None:
        """记录一次成功响应。"""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._half_open_success_count += 1
                if self._half_open_success_count >= self.half_open_successes:
                    logger.debug(
                        "CircuitBreaker %s: HALF_OPEN 探测成功，恢复 CLOSED", self.name
                    )
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._half_open_success_count = 0
            elif self._state == self.CLOSED:
                # 成功时逐渐减少失败计数，给服务恢复的机会
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """记录一次失败响应。"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == self.HALF_OPEN:
                logger.debug(
                    "CircuitBreaker %s: HALF_OPEN 探测失败，重新 OPEN", self.name
                )
                self._state = self.OPEN
            elif self._failure_count >= self.failure_threshold:
                logger.debug(
                    "CircuitBreaker %s: 连续失败 %d 次，进入 OPEN",
                    self.name,
                    self._failure_count,
                )
                self._state = self.OPEN

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """在熔断器保护下调用 func。

        若熔断器 OPEN 且未过冷却时间，直接抛出 CircuitBreakerOpenError。
        调用成功自动 record_success，失败自动 record_failure。
        """
        if not self.can_execute():
            raise CircuitBreakerOpenError(
                f"熔断器 {self.name} 处于 OPEN 状态，请求被拒绝"
            )

        try:
            result = func(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise

        self.record_success()
        return result

    def _current_cooldown(self) -> float:
        """计算当前 OPEN 状态的冷却时间（指数退避）。"""
        # 超过阈值后的额外失败次数，从 0 开始
        excess = max(0, self._failure_count - self.failure_threshold)
        cooldown = self.base_cooldown * (2 ** excess)
        return min(cooldown, self.max_cooldown)

    def get_state_summary(self) -> dict[str, Any]:
        """返回熔断器当前状态摘要，便于观测。"""
        with self._lock:
            summary = {
                "name": self.name,
                "state": self._state,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "current_cooldown": self._current_cooldown()
                if self._state == self.OPEN
                else 0.0,
                "remaining_cooldown": max(
                    0.0,
                    self._current_cooldown()
                    - (time.monotonic() - self._last_failure_time),
                )
                if self._state == self.OPEN
                else 0.0,
            }
        return summary
