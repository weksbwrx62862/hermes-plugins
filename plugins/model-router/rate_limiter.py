"""Deployment 级滑动窗口限流器。

支持两种后端：
  - 内存（默认）：线程安全，适合单进程部署。
  - Redis：通过外部 redis.Redis 客户端共享计数，适合多进程/多副本部署。

限流维度：
  - RPM：每分钟请求数
  - TPM：每分钟 token 数
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RateLimiterQuotaError(Exception):
    """配额配置错误。"""


class RateLimiterBackendError(Exception):
    """后端（Redis）操作失败。"""


class RateLimiter:
    """为每个 deployment 维护滑动窗口 TPM/RPM 计数器。

    参数:
        quotas: deployment 配额，格式
            {"deployment_id": {"rpm": 100, "tpm": 10000}}
        redis_client: 可选 redis.Redis 客户端；为 None 时使用内存后端
        window_seconds: 滑动窗口长度，默认 60 秒
        warning_ratio: 配额使用比例达到该值时标记为警告，默认 0.8
        key_prefix: Redis key 前缀，默认 "model_router:rate_limiter"
    """

    def __init__(
        self,
        quotas: Optional[dict[str, dict[str, int]]] = None,
        redis_client: Optional[Any] = None,
        window_seconds: int = 60,
        warning_ratio: float = 0.8,
        key_prefix: str = "model_router:rate_limiter",
    ) -> None:
        if window_seconds <= 0:
            raise RateLimiterQuotaError("window_seconds 必须大于 0")
        if not 0 < warning_ratio <= 1:
            raise RateLimiterQuotaError("warning_ratio 必须在 (0, 1] 之间")

        self._quotas: dict[str, dict[str, int]] = dict(quotas or {})
        self._redis = redis_client
        self._window = float(window_seconds)
        self._warning_ratio = float(warning_ratio)
        self._key_prefix = key_prefix

        # 内存后端数据结构：deployment_id -> [(timestamp, tokens), ...]
        self._memory: dict[str, list[tuple[float, int]]] = {}
        self._lock = threading.RLock()

    def set_quota(self, deployment_id: str, rpm: int, tpm: int) -> None:
        """动态设置 deployment 配额。"""
        if rpm <= 0 or tpm <= 0:
            raise RateLimiterQuotaError("rpm 与 tpm 必须大于 0")
        self._quotas[deployment_id] = {"rpm": rpm, "tpm": tpm}

    def get_quota(self, deployment_id: str) -> dict[str, int]:
        """返回 deployment 的 rpm/tpm 配额；不存在则返回无限额。"""
        return self._quotas.get(deployment_id, {"rpm": 0, "tpm": 0})

    def record_usage(self, deployment_id: str, tokens: int = 1) -> dict[str, Any]:
        """记录一次请求产生的 token 使用量。

        返回当前窗口内的统计摘要：
            {
                "rpm": int,
                "tpm": int,
                "rpm_ratio": float,
                "tpm_ratio": float,
                "warning": bool,
                "reason": str,
            }
        """
        if tokens < 0:
            tokens = 0

        now = time.monotonic()
        if self._redis is not None:
            self._record_usage_redis(deployment_id, tokens, now)
            return self.get_quota_ratio(deployment_id, now=now)

        with self._lock:
            records = self._memory.setdefault(deployment_id, [])
            records.append((now, tokens))
            self._cleanup_window(records, now)
            return self._compute_ratio(deployment_id, records, now)

    def get_quota_ratio(
        self, deployment_id: str, now: Optional[float] = None
    ) -> dict[str, Any]:
        """查询 deployment 在当前滑动窗口内的配额使用比例与警告状态。"""
        now = now if now is not None else time.monotonic()
        if self._redis is not None:
            return self._get_quota_ratio_redis(deployment_id, now)

        with self._lock:
            records = self._memory.get(deployment_id, [])
            self._cleanup_window(records, now)
            return self._compute_ratio(deployment_id, records, now)

    def _cleanup_window(self, records: list[tuple[float, int]], now: float) -> None:
        """移除滑动窗口外的旧记录。"""
        cutoff = now - self._window
        # 列表按时间递增，从头部删除过期项
        while records and records[0][0] < cutoff:
            records.pop(0)

    def _compute_ratio(
        self,
        deployment_id: str,
        records: list[tuple[float, int]],
        now: float,
    ) -> dict[str, Any]:
        """根据内存记录计算配额比例。"""
        quota = self.get_quota(deployment_id)
        rpm_limit = quota.get("rpm", 0)
        tpm_limit = quota.get("tpm", 0)

        rpm = len(records)
        tpm = sum(r[1] for r in records)

        rpm_ratio = rpm / rpm_limit if rpm_limit > 0 else 0.0
        tpm_ratio = tpm / tpm_limit if tpm_limit > 0 else 0.0

        warning = rpm_ratio >= self._warning_ratio or tpm_ratio >= self._warning_ratio
        reason = ""
        if warning:
            parts = []
            if rpm_ratio >= self._warning_ratio:
                parts.append(f"RPM 使用比例 {rpm_ratio:.0%}")
            if tpm_ratio >= self._warning_ratio:
                parts.append(f"TPM 使用比例 {tpm_ratio:.0%}")
            reason = "、".join(parts) + " 超过警告阈值"

        return {
            "deployment_id": deployment_id,
            "rpm": rpm,
            "tpm": tpm,
            "rpm_ratio": rpm_ratio,
            "tpm_ratio": tpm_ratio,
            "warning": warning,
            "reason": reason,
        }

    # ── Redis 后端实现 ───────────────────────────────────────────────

    def _redis_key(self, deployment_id: str) -> str:
        return f"{self._key_prefix}:{deployment_id}:window"

    def _record_usage_redis(self, deployment_id: str, tokens: int, now: float) -> None:
        """使用 Redis sorted set 记录一次请求。"""
        key = self._redis_key(deployment_id)
        member = f"{now:.6f}:{tokens}:{uuid.uuid4().hex}"
        try:
            pipe = self._redis.pipeline()
            pipe.zadd(key, {member: now})
            # 清理窗口外数据
            pipe.zremrangebyscore(key, 0, now - self._window)
            pipe.expire(key, int(self._window) + 1)
            pipe.execute()
        except Exception as exc:
            logger.warning(
                "RateLimiter Redis 写入失败，降级到内存后端: %s", exc
            )
            self._redis = None
            with self._lock:
                records = self._memory.setdefault(deployment_id, [])
                records.append((now, tokens))
                self._cleanup_window(records, now)

    def _get_quota_ratio_redis(
        self, deployment_id: str, now: float
    ) -> dict[str, Any]:
        """从 Redis sorted set 读取窗口内统计。"""
        key = self._redis_key(deployment_id)
        try:
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(key, 0, now - self._window)
            pipe.zrange(key, 0, -1)
            _, members = pipe.execute()
        except Exception as exc:
            logger.warning(
                "RateLimiter Redis 读取失败，降级到内存后端: %s", exc
            )
            self._redis = None
            return self.get_quota_ratio(deployment_id, now=now)

        rpm = len(members)
        tpm = 0
        for member in members:
            try:
                # member 格式: timestamp:tokens:uuid
                tpm += int(member.split(":")[1])
            except Exception:
                continue

        quota = self.get_quota(deployment_id)
        rpm_limit = quota.get("rpm", 0)
        tpm_limit = quota.get("tpm", 0)

        rpm_ratio = rpm / rpm_limit if rpm_limit > 0 else 0.0
        tpm_ratio = tpm / tpm_limit if tpm_limit > 0 else 0.0

        warning = rpm_ratio >= self._warning_ratio or tpm_ratio >= self._warning_ratio
        reason = ""
        if warning:
            parts = []
            if rpm_ratio >= self._warning_ratio:
                parts.append(f"RPM 使用比例 {rpm_ratio:.0%}")
            if tpm_ratio >= self._warning_ratio:
                parts.append(f"TPM 使用比例 {tpm_ratio:.0%}")
            reason = "、".join(parts) + " 超过警告阈值"

        return {
            "deployment_id": deployment_id,
            "rpm": rpm,
            "tpm": tpm,
            "rpm_ratio": rpm_ratio,
            "tpm_ratio": tpm_ratio,
            "warning": warning,
            "reason": reason,
        }
