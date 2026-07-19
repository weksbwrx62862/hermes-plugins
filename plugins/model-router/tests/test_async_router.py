"""AsyncRouter 单元测试。

覆盖：
  - 异步路由接口与返回格式
  - 缓存命中与写入
  - 事件循环不被同步路由阻塞
  - 10 并发下 P95 延迟显著低于同步直接调用版本
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# 加载 async_router.py（目录名含连字符）
_async_router_path = Path(__file__).resolve().parent.parent / "async_router.py"
_async_router_spec = importlib.util.spec_from_file_location("async_router", str(_async_router_path))
async_router = importlib.util.module_from_spec(_async_router_spec)
sys.modules["async_router"] = async_router
_async_router_spec.loader.exec_module(async_router)

AsyncRouter = async_router.AsyncRouter

# 加载 cache.py
_cache_path = Path(__file__).resolve().parent.parent / "cache.py"
_cache_spec = importlib.util.spec_from_file_location("route_cache", str(_cache_path))
route_cache = importlib.util.module_from_spec(_cache_spec)
sys.modules["route_cache"] = route_cache
_cache_spec.loader.exec_module(route_cache)

RouteCache = route_cache.RouteCache

# 加载 model-router __init__.py（若已由其他测试模块加载则复用，避免多份模块对象导致 sys.modules 不一致）
_model_router_path = Path(__file__).resolve().parent.parent / "__init__.py"
_model_router_spec = importlib.util.spec_from_file_location("plugins.model-router", str(_model_router_path))
model_router = sys.modules.get("plugins.model-router")
if model_router is None:
    model_router = importlib.util.module_from_spec(_model_router_spec)
    sys.modules["plugins.model-router"] = model_router
    _model_router_spec.loader.exec_module(model_router)


@pytest.fixture
def router():
    """返回未启用缓存/分类器的 AsyncRouter 实例。"""
    return AsyncRouter()


@pytest.fixture
def router_with_cache():
    """返回启用 RouteCache 的 AsyncRouter 实例。"""
    return AsyncRouter(route_cache=RouteCache(capacity=8, ttl=60.0))


def _sample_route_result(query: str = "test") -> dict:
    """构造与 _route() 格式对齐的最小路由结果。"""
    return {
        "name": "deepseek-v4-flash",
        "provider": "openai",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "key": "sk-test",
        "key_masked": "sk***est",
        "strategy": "auto",
        "complexity": 3,
        "task_type": "code",
        "pool_size": 4,
        "alternatives": ["mimo-v2.5", "qwen3.5-flash"],
        "score_breakdown": {"deepseek-v4-flash": 95.0, "mimo-v2.5": 80.0},
        "fallback_chain": [
            {"provider": "mimo", "model": "mimo-v2.5", "base_url": "", "api_key_masked": "***"},
        ],
        "time_info": {
            "beijing_time": "2026-07-15 12:00:00",
            "is_off_peak": False,
            "period": "高峰期",
        },
        "selection_reason": "测试路由",
    }


class TestInterface:
    """接口与格式测试。"""

    @pytest.mark.asyncio
    async def test_route_returns_dict(self, router):
        """route() 返回与 _route() 格式一致的字典。"""
        result = await router.route(
            query="写一个 Python 快排",
            strategy="auto",
            messages=[{"role": "user", "content": "写一个 Python 快排"}],
            context={"dev_stage": "build"},
        )
        assert isinstance(result, dict)
        # 核心字段必须存在
        for key in ("name", "provider", "model", "strategy", "complexity", "task_type", "fallback_chain"):
            assert key in result, f"缺少字段 {key}"

    @pytest.mark.asyncio
    async def test_route_cache_flow(self, router_with_cache):
        """缓存命中后应直接返回缓存结果。"""
        router = router_with_cache
        query = "缓存测试查询"

        # 第一次调用写入缓存
        first = await router.route(query, strategy="auto")
        assert first is not None
        assert first.get("from_cache") is False

        # 第二次调用应命中缓存
        second = await router.route(query, strategy="auto")
        assert second.get("from_cache") is True
        assert second["name"] == first["name"]
        assert second["provider"] == first["provider"]


class TestNonBlocking:
    """事件循环非阻塞测试。"""

    @pytest.mark.asyncio
    async def test_route_does_not_block_event_loop(self, router):
        """同步路由逻辑在线程池中执行时，其他协程应能继续运行。"""
        heartbeat_interval = 0.01  # 10ms
        heartbeat_count = 0
        stop_heartbeat = False

        async def heartbeat():
            nonlocal heartbeat_count
            while not stop_heartbeat:
                await asyncio.sleep(heartbeat_interval)
                heartbeat_count += 1

        # 模拟一个耗时 80ms 的同步路由
        def slow_route(*args, **kwargs):
            time.sleep(0.08)
            return _sample_route_result()

        original_route = model_router._route
        model_router._route = slow_route
        try:
            hb_task = asyncio.create_task(heartbeat())
            result = await router.route("阻塞测试", strategy="auto")
            stop_heartbeat = True
            await hb_task

            assert result is not None
            # 80ms 内 10ms 心跳应至少触发 5 次，证明事件循环未被阻塞
            assert heartbeat_count >= 5, f"事件循环被阻塞，心跳仅触发 {heartbeat_count} 次"
        finally:
            model_router._route = original_route


class TestConcurrencyPerformance:
    """并发性能测试。"""

    @pytest.mark.asyncio
    async def test_concurrent_p95_beat_sync_direct(self, router):
        """10 并发下异步路由 P95 延迟低于同步直接调用的 50%。

        使用 time.sleep 模拟耗时 50ms 的同步路由计算：
          - 同步直接调用：10 个请求在事件循环中串行执行，第 i 个请求完成时间约为 i*50ms。
          - 异步路由：通过线程池并发执行，所有请求几乎同时完成。

        这里以“批次启动到单个请求完成”的耗时作为延迟指标，更能体现
        事件循环不被阻塞的优势。
        """
        delay = 0.05  # 50ms

        def slow_route(*args, **kwargs):
            time.sleep(delay)
            return _sample_route_result()

        original_route = model_router._route
        model_router._route = slow_route
        try:
            queries = [f"并发查询 {i}" for i in range(10)]

            def p95(values: list[float]) -> float:
                sorted_vals = sorted(values)
                idx = int(len(sorted_vals) * 0.95) - 1
                return sorted_vals[max(0, idx)]

            # 1) 同步直接调用版本：在协程里直接调用阻塞函数，请求串行执行
            async def run_sync_batch() -> list[float]:
                batch_start = time.monotonic()
                latencies = []
                for q in queries:
                    model_router._route(q, "auto")
                    latencies.append(time.monotonic() - batch_start)
                return latencies

            sync_latencies = await run_sync_batch()
            sync_p95 = p95(sync_latencies)

            # 2) AsyncRouter 异步版本：通过线程池并发执行
            async def run_async_batch() -> list[float]:
                batch_start = time.monotonic()

                async def async_route_coro(q: str) -> float:
                    await router.route(q, strategy="auto")
                    return time.monotonic() - batch_start

                return await asyncio.gather(*(async_route_coro(q) for q in queries))

            async_latencies = await run_async_batch()
            async_p95 = p95(async_latencies)

            print(f"\n同步直接调用 P95={sync_p95*1000:.1f}ms")
            print(f"AsyncRouter P95={async_p95*1000:.1f}ms")

            # 异步 P95 应低于同步 P95 的 50%
            assert async_p95 < sync_p95 * 0.5, (
                f"异步 P95 {async_p95*1000:.1f}ms 未低于同步 P95 {sync_p95*1000:.1f}ms 的 50%"
            )
        finally:
            model_router._route = original_route


class TestThreadPool:
    """自定义线程池测试。"""

    @pytest.mark.asyncio
    async def test_custom_executor(self):
        """AsyncRouter 应支持传入自定义 ThreadPoolExecutor。"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            router = AsyncRouter(executor=executor)
            result = await router.route("自定义线程池测试", strategy="auto")
            assert isinstance(result, dict)
            assert "name" in result
