"""model-router 集成测试。

验证各新模块正确集成到 __init__.py 中：
  - 缓存命中流程
  - Provider 选择器调用
  - RateLimiter / CircuitBreaker 状态更新
  - RouterTracer 记录关键字段
  - handle_pre_llm_call 异步/同步兼容性
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

# 加载 model-router __init__.py（目录名含连字符，无法直接 import；
# 若已由其他测试模块加载则复用，避免多份模块对象导致 sys.modules 不一致）
_model_router_path = Path(__file__).resolve().parent.parent / "__init__.py"
_model_router_spec = importlib.util.spec_from_file_location(
    "plugins.model-router", str(_model_router_path)
)
model_router = sys.modules.get("plugins.model-router")
if model_router is None:
    model_router = importlib.util.module_from_spec(_model_router_spec)
    sys.modules["plugins.model-router"] = model_router
    _model_router_spec.loader.exec_module(model_router)


def _temp_config() -> dict[str, Any]:
    """返回用于集成测试的最小配置。"""
    return {
        "plugins": {
            "model-router": {
                "strategy": "auto",
                "allowed_providers": ["openai", "deepseek", "stepfun"],
                "mimo_enabled": False,
                "use_cache": True,
                "cache_capacity": 8,
                "cache_ttl_seconds": 60.0,
                "cache_semantic": False,
                "use_embed_router": False,
                "use_shared_assessor": False,
                "use_async": False,
                "use_provider_selector": True,
                "provider_selector_weights": {
                    "latency": 1.0,
                    "cost": 1.0,
                    "quality": 1.0,
                    "quota": 1.0,
                },
                "use_rate_limiter": True,
                "rate_limiter_window_seconds": 60,
                "rate_limiter_warning_ratio": 0.8,
                "use_circuit_breaker": True,
                "circuit_breaker_threshold": 5,
                "circuit_breaker_base_cooldown": 30.0,
                "circuit_breaker_max_cooldown": 300.0,
                "use_tracing": True,
                "tracer_name": "model-router-test",
                "use_config_reloader": False,
                "use_rl_router": False,
                "rl_algorithm": "qlearning",
                "provider_region_preference": [],
                "provider_region_bonus": 5.0,
                "rate_limiter_quotas": {
                    "openai:deepseek-v4-flash": {"rpm": 30, "tpm": 20000},
                    "deepseek:deepseek-v4-flash": {"rpm": 30, "tpm": 20000},
                },
                "deployments": {
                    "deepseek-v4-flash": [
                        {
                            "provider": "openai",
                            "model": "deepseek-v4-flash",
                            "latency_ms": 150,
                            "cost": 1,
                            "quality": 5,
                            "speed": 5,
                            "rpm_limit": 30,
                            "tpm_limit": 20000,
                            "region": "cn",
                        },
                        {
                            "provider": "deepseek",
                            "model": "deepseek-v4-flash",
                            "latency_ms": 180,
                            "cost": 1,
                            "quality": 5,
                            "speed": 5,
                            "rpm_limit": 30,
                            "tpm_limit": 20000,
                            "region": "cn",
                        },
                    ],
                },
                "model_attrs": {
                    "deepseek-v4-flash": {
                        "context_window": 1_000_000,
                        "cost": 1,
                        "quality": 5,
                        "speed": 5,
                    },
                },
                "scoring": {},
            },
        },
        "providers": {
            "openai": {
                "api_key": "sk-openai",
                "base_url": "https://api.openai.com/v1",
                "models": ["deepseek-v4-flash"],
            },
            "deepseek": {
                "api_key": "sk-deepseek",
                "base_url": "https://api.deepseek.com/v1",
                "models": ["deepseek-v4-flash"],
            },
        },
    }


@pytest.fixture
def mock_config(monkeypatch):
    """替换配置加载为临时测试配置，并清理相关缓存。"""
    cfg = _temp_config()

    def _load_temp_config():
        return cfg

    monkeypatch.setattr(model_router, "_load_config", _load_temp_config)
    monkeypatch.setattr(model_router, "_CONFIG_CACHE", None)
    monkeypatch.setattr(model_router, "_CONFIG_CACHE_TIME", 0.0)
    monkeypatch.setattr(model_router, "_POOL_CACHE", None)
    monkeypatch.setattr(model_router, "_POOL_CACHE_TIME", 0.0)
    yield cfg
    model_router._CONFIG_CACHE = None
    model_router._CONFIG_CACHE_TIME = 0.0
    model_router._POOL_CACHE = None
    model_router._POOL_CACHE_TIME = 0.0


@pytest.fixture
def reset_instances():
    """重置各子模块实例与熔断器状态。"""
    # 保存原始状态以便恢复
    originals = {
        "_route_cache_instance": model_router._route_cache_instance,
        "_provider_selector_instance": model_router._provider_selector_instance,
        "_rate_limiter_instance": model_router._rate_limiter_instance,
        "_router_tracer_instance": model_router._router_tracer_instance,
        "_rl_router_instance": model_router._rl_router_instance,
        "_async_router_instance": model_router._async_router_instance,
        "_CircuitBreakerClass": model_router._CircuitBreakerClass,
        "_ConfigReloader": model_router._ConfigReloader,
        "_EmbedTaskClassifier": model_router._EmbedTaskClassifier,
        "_ProviderSelector": model_router._ProviderSelector,
        "_RateLimiter": model_router._RateLimiter,
        "_RouterTracer": model_router._RouterTracer,
        "_QLearningRouter": model_router._QLearningRouter,
        "_ContextualBanditRouter": model_router._ContextualBanditRouter,
    }
    breakers_snapshot = dict(model_router.circuit_breakers)

    model_router._route_cache_instance = None
    model_router._provider_selector_instance = None
    model_router._rate_limiter_instance = None
    model_router._router_tracer_instance = None
    model_router._rl_router_instance = None
    model_router._async_router_instance = None
    model_router._CircuitBreakerClass = None
    model_router.circuit_breakers.clear()

    yield

    # 恢复原始状态
    for key, value in originals.items():
        setattr(model_router, key, value)
    model_router.circuit_breakers.clear()
    model_router.circuit_breakers.update(breakers_snapshot)


class TestCacheFlow:
    """路由缓存命中流程测试。"""

    def test_cache_hit_returns_cached_result(self, mock_config, reset_instances, monkeypatch):
        """相同查询第二次调用应命中缓存并直接返回。"""
        model_router.setup(config=mock_config["plugins"]["model-router"])
        query = "集成测试缓存命中"

        first = model_router._route(query, "auto")
        assert first is not None
        assert first.get("from_cache") is not True

        original_route = model_router._route
        call_count = [0]

        def counted_route(*args, **kwargs):
            call_count[0] += 1
            return original_route(*args, **kwargs)

        monkeypatch.setattr(model_router, "_route", counted_route)
        second = model_router._route(query, "auto")

        # 命中缓存时 _route 仍被调用一次（缓存查询在 _route 内部完成），
        # 但应直接返回缓存结果而不再执行评分逻辑。
        assert second.get("from_cache") is True
        assert second["name"] == first["name"]
        assert second["provider"] == first["provider"]
        assert call_count[0] == 1


class TestProviderSelector:
    """Provider 选择器集成测试。"""

    def test_provider_selector_called(self, mock_config, reset_instances, monkeypatch):
        """启用 Provider 选择器时，select 方法应被调用。"""
        model_router.setup(config=mock_config["plugins"]["model-router"])
        selector = model_router._provider_selector_instance
        assert selector is not None

        original_select = selector.select
        call_args = []

        def mock_select(*args, **kwargs):
            call_args.append((args, kwargs))
            return original_select(*args, **kwargs)

        monkeypatch.setattr(selector, "select", mock_select)
        result = model_router._route("测试 Provider 选择器", "auto")

        assert result is not None
        assert len(call_args) > 0
        assert call_args[0][0][0] == "deepseek-v4-flash"


class TestRateLimiterAndCircuitBreaker:
    """RateLimiter 与 CircuitBreaker 更新测试。"""

    def test_post_llm_call_updates_rate_limiter(self, mock_config, reset_instances, monkeypatch):
        """handle_post_llm_call 应调用 RateLimiter.record_usage。"""
        model_router.setup(config=mock_config["plugins"]["model-router"])
        rl = model_router._rate_limiter_instance
        assert rl is not None

        calls = []
        original_record_usage = rl.record_usage

        def mock_record_usage(deployment_id, tokens=1):
            calls.append((deployment_id, tokens))
            return original_record_usage(deployment_id, tokens)

        monkeypatch.setattr(rl, "record_usage", mock_record_usage)
        model_router.handle_post_llm_call(
            model="deepseek-v4-flash",
            provider="openai",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            task_type="code",
            complexity=3,
        )

        assert len(calls) == 1
        assert calls[0][0] == "openai:deepseek-v4-flash"
        assert calls[0][1] == 30

    def test_post_llm_call_updates_circuit_breaker_success(
        self, mock_config, reset_instances
    ):
        """handle_post_llm_call 成功时应调用 CircuitBreaker.record_success。"""
        model_router.setup(config=mock_config["plugins"]["model-router"])
        cb = model_router._get_circuit_breaker("openai", "deepseek-v4-flash")
        assert cb is not None

        # 先制造若干失败，使熔断器处于 HALF_OPEN 或带失败计数状态
        for _ in range(3):
            cb.record_failure()

        model_router.handle_post_llm_call(
            model="deepseek-v4-flash",
            provider="openai",
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )

        # 成功后失败计数应被递减或清零
        assert cb._failure_count < 3

    def test_post_llm_call_updates_circuit_breaker_failure(
        self, mock_config, reset_instances
    ):
        """handle_post_llm_call 传入 error 时应调用 CircuitBreaker.record_failure。"""
        model_router.setup(config=mock_config["plugins"]["model-router"])
        cb = model_router._get_circuit_breaker("deepseek", "deepseek-v4-flash")
        assert cb is not None

        initial_failures = cb._failure_count
        model_router.handle_post_llm_call(
            model="deepseek-v4-flash",
            provider="deepseek",
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            error=True,
        )

        assert cb._failure_count == initial_failures + 1


class TestTracing:
    """路由追踪集成测试。"""

    def test_tracing_records_route_decision(self, mock_config, reset_instances, monkeypatch):
        """启用 tracing 时，_route 应调用 RouterTracer 记录决策字段。"""
        model_router.setup(config=mock_config["plugins"]["model-router"])
        tracer = model_router._router_tracer_instance
        assert tracer is not None

        calls = []
        original_record = tracer.record_route_decision

        def mock_record(**kwargs):
            calls.append(kwargs)
            return original_record(**kwargs)

        monkeypatch.setattr(tracer, "record_route_decision", mock_record)
        result = model_router._route("测试 Tracing 记录", "auto")

        assert result is not None
        assert len(calls) == 1
        recorded = calls[0]
        assert recorded.get("complexity") == result["complexity"]
        assert recorded.get("task_type") == result["task_type"]
        assert recorded.get("selected_model") == result["name"]
        assert recorded.get("selected_provider") == result["provider"]
        assert isinstance(recorded.get("candidate_scores"), list)
        assert recorded.get("cache_hit") is False
        assert recorded.get("elapsed_ms", 0) >= 0


class TestPreLlmCallAsync:
    """handle_pre_llm_call 异步/同步兼容性测试。"""

    @pytest.mark.asyncio
    async def test_async_handle_pre_llm_call(self, mock_config, reset_instances):
        """异步入口 handle_pre_llm_call 可在线程池中完成路由。"""
        cfg = mock_config["plugins"]["model-router"].copy()
        cfg["use_async"] = True
        model_router.setup(config=cfg)

        result = await model_router.handle_pre_llm_call(
            session_id="test-session-async",
            user_message="写一个 Python 快排",
            messages=[{"role": "user", "content": "写一个 Python 快排"}],
        )

        assert result is not None
        assert "context" in result
        assert result["context_merge"]["model_selection"]

    def test_sync_handle_pre_llm_call(self, mock_config, reset_instances):
        """同步入口 _handle_pre_llm_call_sync 不依赖事件循环即可工作。"""
        model_router.setup(config=mock_config["plugins"]["model-router"])

        result = model_router._handle_pre_llm_call_sync(
            session_id="test-session-sync",
            user_message="解释什么是递归",
            messages=[{"role": "user", "content": "解释什么是递归"}],
        )

        assert result is not None
        assert "context" in result
        assert result["context_merge"]["model_selection"]

    def test_sync_wrapper_fallback(self, mock_config, reset_instances):
        """同步包装器在无线程事件循环时可运行异步 handle_pre_llm_call。"""
        cfg = mock_config["plugins"]["model-router"].copy()
        cfg["use_async"] = True
        model_router.setup(config=cfg)

        result = model_router._handle_pre_llm_call_sync_wrapper(
            session_id="test-session-wrapper",
            user_message="总结这段文字",
            messages=[{"role": "user", "content": "总结这段文字"}],
        )

        assert result is not None
        assert "context" in result
