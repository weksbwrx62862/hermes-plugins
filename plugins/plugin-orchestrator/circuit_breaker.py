"""
Circuit Breaker — 防止坏插件拖垮整个请求循环

三态熔断器（Hystrix 风格）:
  CLOSED    → 正常执行
  OPEN      → 跳过该钩子（熔断），等待恢复超时后进入 HALF_OPEN
  HALF_OPEN → 尝试一次，成功则恢复 CLOSED，失败则重计熔断

线程安全，每个 (plugin_name, hook_name) 一个 breaker。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ── 熔断器状态 ────────────────────────────────────────────────────────

class BreakerState:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """单个插件的单个钩子的熔断器。"""

    __slots__ = (
        "plugin_name", "hook_name",
        "_state", "_failure_count", "_last_failure_time",
        "_failure_threshold", "_recovery_timeout", "_half_open_max",
        "_half_open_attempts", "_total_failures", "_total_calls",
        "_lock",
    )

    def __init__(
        self,
        plugin_name: str,
        hook_name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        half_open_max: int = 1,
    ):
        self.plugin_name = plugin_name
        self.hook_name = hook_name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max = half_open_max

        self._state = BreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_attempts = 0
        self._total_failures = 0
        self._total_calls = 0
        self._lock = threading.Lock()

    # ── 公共查询 ──

    @property
    def state(self) -> str:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def total_failures(self) -> int:
        return self._total_failures

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def last_failure_time(self) -> float:
        return self._last_failure_time

    # ── 核心逻辑 ──

    def is_open(self) -> bool:
        """当前是否处于熔断状态（跳过）。"""
        with self._lock:
            if self._state == BreakerState.CLOSED:
                return False

            if self._state == BreakerState.OPEN:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self._recovery_timeout:
                    # 尝试 HALF_OPEN
                    self._state = BreakerState.HALF_OPEN
                    self._half_open_attempts = 0
                    logger.info(
                        "Breaker %s.%s: OPEN → HALF_OPEN (%.1fs elapsed)",
                        self.plugin_name, self.hook_name, elapsed,
                    )
                    return False
                return True  # 仍处熔断期

            # HALF_OPEN: 已用完尝试次数则跳过
            if self._half_open_attempts >= self._half_open_max:
                logger.debug(
                    "Breaker %s.%s: HALF_OPEN max attempts reached, skipping",
                    self.plugin_name, self.hook_name,
                )
                return True
            return False

    def on_success(self):
        """调用成功：CLOSED 保持不变，HALF_OPEN 恢复 CLOSED。"""
        with self._lock:
            self._total_calls += 1

            if self._state == BreakerState.HALF_OPEN:
                # 恢复
                old_state = self._state
                self._state = BreakerState.CLOSED
                self._failure_count = 0
                self._half_open_attempts = 0
                logger.info(
                    "Breaker %s.%s: HALF_OPEN → CLOSED (recovered)",
                    self.plugin_name, self.hook_name,
                )
                return old_state, BreakerState.CLOSED

            if self._state == BreakerState.OPEN:
                # 如果有人在不通过 is_open 的情况下调用了 on_success，
                # 也视为恢复（防御性）
                self._state = BreakerState.CLOSED
                self._failure_count = 0
                self._half_open_attempts = 0
                return BreakerState.OPEN, BreakerState.CLOSED

            # CLOSED: 失败计数保持（无法降级到 0，避免偶然成功清除历史）
            return self._state, self._state

    def on_failure(self):
        """调用失败：计数递增，达到阈值后 OPEN。"""
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()
            prev_state = self._state

            if self._state == BreakerState.HALF_OPEN:
                self._half_open_attempts += 1
                # 如果超限，回到 OPEN
                if self._half_open_attempts >= self._half_open_max:
                    self._state = BreakerState.OPEN
                    logger.warning(
                        "Breaker %s.%s: HALF_OPEN → OPEN (attempt %d/%d failed)",
                        self.plugin_name, self.hook_name,
                        self._half_open_attempts, self._half_open_max,
                    )
                    return prev_state, BreakerState.OPEN
                return prev_state, BreakerState.HALF_OPEN

            # CLOSED / OPEN: 检查是否达到阈值
            if self._failure_count >= self._failure_threshold and self._state == BreakerState.CLOSED:
                self._state = BreakerState.OPEN
                logger.warning(
                    "Breaker %s.%s: CLOSED → OPEN (%d failures)",
                    self.plugin_name, self.hook_name,
                    self._failure_count,
                )
                return BreakerState.CLOSED, BreakerState.OPEN

            return prev_state, self._state

    # ── 管理接口 ──

    def reset(self):
        """强制重置为 CLOSED。"""
        with self._lock:
            old = self._state
            self._state = BreakerState.CLOSED
            self._failure_count = 0
            self._half_open_attempts = 0
            return old

    def trip(self):
        """强制熔断（OPEN）。"""
        with self._lock:
            old = self._state
            self._state = BreakerState.OPEN
            self._last_failure_time = time.time()
            return old

    def dump(self) -> dict:
        """导出状态（用于诊断）。"""
        with self._lock:
            return {
                "plugin": self.plugin_name,
                "hook": self.hook_name,
                "state": self._state,
                "failure_count": self._failure_count,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "failure_threshold": self._failure_threshold,
                "recovery_timeout": self._recovery_timeout,
                "half_open_max": self._half_open_max,
                "last_failure_age": (
                    time.time() - self._last_failure_time
                    if self._last_failure_time > 0 else 0
                ),
            }


# ── 全局注册表 ────────────────────────────────────────────────────────

class CircuitBreakerRegistry:
    """全局熔断器注册表。线程安全。"""

    def __init__(self):
        self._breakers: Dict[Tuple[str, str], CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        plugin_name: str,
        hook_name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> CircuitBreaker:
        key = (plugin_name, hook_name)
        with self._lock:
            if key not in self._breakers:
                self._breakers[key] = CircuitBreaker(
                    plugin_name=plugin_name,
                    hook_name=hook_name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                )
            return self._breakers[key]

    def get(self, plugin_name: str, hook_name: str) -> Optional[CircuitBreaker]:
        key = (plugin_name, hook_name)
        with self._lock:
            return self._breakers.get(key)

    def list_all(self) -> list[CircuitBreaker]:
        with self._lock:
            return list(self._breakers.values())

    def list_open(self) -> list[CircuitBreaker]:
        with self._lock:
            return [b for b in self._breakers.values() if b.state == BreakerState.OPEN]

    def reset_all(self):
        with self._lock:
            for b in self._breakers.values():
                b.reset()

    def dump_all(self) -> list[dict]:
        return [b.dump() for b in self.list_all()]


# ── 全局单例 ──────────────────────────────────────────────────────────

_global_registry: Optional[CircuitBreakerRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> CircuitBreakerRegistry:
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = CircuitBreakerRegistry()
    return _global_registry
