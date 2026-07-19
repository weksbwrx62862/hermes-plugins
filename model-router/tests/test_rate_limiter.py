"""RateLimiter 单元测试。

覆盖：
  - 滑动窗口 RPM/TPM 计数正确
  - 配额接近阈值时发出警告
  - Redis 后端可用时行为与内存一致
  - Redis 故障时自动降级到内存
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# 加载被测模块（目录名含连字符，使用 importlib）
_rate_limiter_path = Path(__file__).resolve().parent.parent / "rate_limiter.py"
_spec = importlib.util.spec_from_file_location("rate_limiter", str(_rate_limiter_path))
rate_limiter = importlib.util.module_from_spec(_spec)
sys.modules["rate_limiter"] = rate_limiter
_spec.loader.exec_module(rate_limiter)

RateLimiter = rate_limiter.RateLimiter
RateLimiterQuotaError = rate_limiter.RateLimiterQuotaError


class TestRateLimiterMemory:
    """内存后端测试。"""

    def test_init_default(self):
        """默认构造参数生效。"""
        rl = RateLimiter()
        assert rl._window == 60.0
        assert rl._warning_ratio == 0.8
        assert rl._redis is None

    def test_init_invalid_window(self):
        """非法窗口长度应抛异常。"""
        with pytest.raises(RateLimiterQuotaError):
            RateLimiter(window_seconds=0)

    def test_init_invalid_warning_ratio(self):
        """非法警告比例应抛异常。"""
        with pytest.raises(RateLimiterQuotaError):
            RateLimiter(warning_ratio=1.5)
        with pytest.raises(RateLimiterQuotaError):
            RateLimiter(warning_ratio=0)

    def test_record_usage_counts_rpm_and_tpm(self):
        """多次记录后 RPM/TPM 计数正确。"""
        rl = RateLimiter(
            quotas={"dep-a": {"rpm": 100, "tpm": 1000}},
            window_seconds=60,
        )
        for _ in range(5):
            rl.record_usage("dep-a", tokens=10)

        result = rl.get_quota_ratio("dep-a")
        assert result["rpm"] == 5
        assert result["tpm"] == 50
        assert result["rpm_ratio"] == 0.05
        assert result["tpm_ratio"] == 0.05
        assert result["warning"] is False

    def test_window_slides(self):
        """窗口外的旧记录应被清理。"""
        rl = RateLimiter(
            quotas={"dep-a": {"rpm": 100, "tpm": 1000}},
            window_seconds=0.2,
        )
        rl.record_usage("dep-a", tokens=10)
        time.sleep(0.25)
        rl.record_usage("dep-a", tokens=20)

        result = rl.get_quota_ratio("dep-a")
        assert result["rpm"] == 1
        assert result["tpm"] == 20

    def test_warning_at_80_percent(self):
        """配额使用比例达到 80% 时标记警告。"""
        rl = RateLimiter(
            quotas={"dep-a": {"rpm": 10, "tpm": 100}},
            window_seconds=60,
        )
        # RPM 达到 80%（8/10）
        for _ in range(8):
            rl.record_usage("dep-a", tokens=1)
        result = rl.get_quota_ratio("dep-a")
        assert result["warning"] is True
        assert "RPM" in result["reason"]

    def test_warning_by_tpm(self):
        """TPM 达到阈值时同样触发警告。"""
        rl = RateLimiter(
            quotas={"dep-a": {"rpm": 100, "tpm": 100}},
            window_seconds=60,
        )
        rl.record_usage("dep-a", tokens=80)
        result = rl.get_quota_ratio("dep-a")
        assert result["tpm_ratio"] == 0.8
        assert result["warning"] is True
        assert "TPM" in result["reason"]

    def test_multiple_deployments_isolated(self):
        """不同 deployment 的计数相互隔离。"""
        rl = RateLimiter(
            quotas={
                "dep-a": {"rpm": 10, "tpm": 100},
                "dep-b": {"rpm": 20, "tpm": 200},
            }
        )
        rl.record_usage("dep-a", tokens=10)
        rl.record_usage("dep-b", tokens=20)
        assert rl.get_quota_ratio("dep-a")["rpm"] == 1
        assert rl.get_quota_ratio("dep-b")["rpm"] == 1

    def test_no_quota_returns_zero_ratio(self):
        """未配置配额的 deployment 比例恒为 0。"""
        rl = RateLimiter()
        rl.record_usage("unknown", tokens=999)
        result = rl.get_quota_ratio("unknown")
        assert result["rpm_ratio"] == 0.0
        assert result["tpm_ratio"] == 0.0
        assert result["warning"] is False

    def test_set_quota(self):
        """动态设置配额生效。"""
        rl = RateLimiter()
        rl.set_quota("dep-x", rpm=10, tpm=100)
        for _ in range(8):
            rl.record_usage("dep-x", tokens=1)
        assert rl.get_quota_ratio("dep-x")["warning"] is True


class TestRateLimiterRedis:
    """Redis 后端测试（使用 mock）。"""

    def _make_mock_redis(self):
        """构造一个支持 pipeline/zadd/zremrangebyscore/zrange/expire 的 mock Redis。"""
        store = {}

        class Pipeline:
            def __init__(self):
                self._ops = []

            def zadd(self, key, mapping):
                self._ops.append(("zadd", key, mapping))
                return self

            def zremrangebyscore(self, key, min_score, max_score):
                self._ops.append(("zremrangebyscore", key, min_score, max_score))
                return self

            def zrange(self, key, start, end):
                self._ops.append(("zrange", key, start, end))
                return self

            def expire(self, key, seconds):
                self._ops.append(("expire", key, seconds))
                return self

            def execute(self):
                results = []
                for op in self._ops:
                    name = op[0]
                    key = op[1]
                    if name == "zadd":
                        mapping = op[2]
                        if key not in store:
                            store[key] = {}
                        for member, score in mapping.items():
                            store[key][member] = score
                        results.append(len(mapping))
                    elif name == "zremrangebyscore":
                        min_score, max_score = op[2], op[3]
                        removed = 0
                        if key in store:
                            to_remove = [
                                m
                                for m, s in store[key].items()
                                if min_score <= s <= max_score
                            ]
                            for m in to_remove:
                                del store[key][m]
                            removed = len(to_remove)
                        results.append(removed)
                    elif name == "zrange":
                        start, end = op[2], op[3]
                        members = sorted(
                            store.get(key, {}).items(), key=lambda x: x[1]
                        )
                        if end == -1:
                            end = len(members)
                        else:
                            end = end + 1
                        results.append([m[0] for m in members[start:end]])
                    elif name == "expire":
                        results.append(True)
                self._ops = []
                return results

        class MockRedis:
            def pipeline(self):
                return Pipeline()

        return MockRedis(), store

    def test_redis_backend_counts(self):
        """mock Redis 后端计数正确。"""
        mock_redis, _ = self._make_mock_redis()
        rl = RateLimiter(
            quotas={"dep-r": {"rpm": 100, "tpm": 1000}},
            redis_client=mock_redis,
            window_seconds=60,
        )
        for _ in range(3):
            rl.record_usage("dep-r", tokens=10)

        result = rl.get_quota_ratio("dep-r")
        assert result["rpm"] == 3
        assert result["tpm"] == 30
        assert result["warning"] is False

    def test_redis_failure_fallback_to_memory(self):
        """Redis 异常时自动降级到内存后端。"""
        bad_redis = MagicMock()
        bad_redis.pipeline.side_effect = RuntimeError("Redis 连接失败")

        rl = RateLimiter(
            quotas={"dep-fail": {"rpm": 10, "tpm": 100}},
            redis_client=bad_redis,
        )
        rl.record_usage("dep-fail", tokens=5)
        assert rl._redis is None
        result = rl.get_quota_ratio("dep-fail")
        assert result["rpm"] == 1
        assert result["tpm"] == 5
