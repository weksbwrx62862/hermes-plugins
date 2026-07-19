"""ModeSelectionEngine 单元测试。"""

import random

import pytest

from adaptive_multi_agent.selector import ModeSelectionEngine
from adaptive_multi_agent.subagent import AgentMode, CircuitBreaker


@pytest.fixture
def engine(monkeypatch):
    """提供不访问真实数据库的 ModeSelectionEngine 实例。"""
    import adaptive_multi_agent.selector as selector_mod

    # 屏蔽持久化读写，避免写入 ~/.hermes
    monkeypatch.setattr(selector_mod, "load_performance", lambda: {})
    monkeypatch.setattr(selector_mod, "save_performance", lambda *args, **kwargs: None)
    return ModeSelectionEngine()


class TestApplyRules:
    """测试 _apply_rules 对 6 种模式的规则映射。"""

    @pytest.mark.parametrize(
        "assessment,expected_mode",
        [
            (
                {"complexity_score": 2.0, "task_type": "default", "features": {}},
                AgentMode.GENERATOR_VERIFIER,
            ),
            (
                {
                    "complexity_score": 4.0,
                    "task_type": "default",
                    "features": {"has_explicit_verification": True},
                },
                AgentMode.GENERATOR_VERIFIER,
            ),
            (
                {
                    "complexity_score": 5.0,
                    "task_type": "default",
                    "features": {"is_event_driven": True},
                },
                AgentMode.MESSAGE_BUS,
            ),
            (
                {
                    "complexity_score": 5.0,
                    "task_type": "default",
                    "features": {"requires_shared_knowledge": True},
                },
                AgentMode.SHARED_STATE,
            ),
            (
                {
                    "complexity_score": 5.0,
                    "task_type": "analysis",
                    "features": {},
                },
                AgentMode.PARALLEL_FUSION,
            ),
            (
                {
                    "complexity_score": 6.0,
                    "task_type": "default",
                    "features": {"has_roles": True},
                },
                AgentMode.AGENT_TEAMS,
            ),
            (
                {
                    "complexity_score": 5.5,
                    "task_type": "default",
                    "features": {},
                },
                AgentMode.ORCHESTRATOR_SUBAGENT,
            ),
        ],
    )
    def test_rule_mapping(self, engine, assessment, expected_mode):
        candidates = engine._apply_rules(
            assessment["complexity_score"],
            assessment["task_type"],
            assessment["features"],
        )
        assert expected_mode in candidates
        assert all(isinstance(m, AgentMode) for m in candidates)


class TestThompsonSampling:
    """测试 Thompson Sampling 采样不会崩溃且返回合法 AgentMode。"""

    def test_ts_sample_returns_probability(self, engine):
        value = engine._ts_sample("default", AgentMode.GENERATOR_VERIFIER)
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_ts_select_returns_valid_mode(self, engine):
        random.seed(42)
        candidates = [
            AgentMode.SHARED_STATE,
            AgentMode.PARALLEL_FUSION,
            AgentMode.AGENT_TEAMS,
            AgentMode.MESSAGE_BUS,
        ]
        selected, samples = engine._ts_select(candidates, "default")
        assert isinstance(selected, AgentMode)
        assert selected in candidates
        assert len(samples) == len(candidates)

    def test_select_mode_returns_valid_mode(self, engine):
        random.seed(42)
        assessment = {
            "complexity_score": 9.0,
            "task_type": "complex",
            "features": {"multi_perspective": True},
        }
        selected = engine.select_mode(assessment)
        assert isinstance(selected, AgentMode)


class TestCircuitBreaker:
    """测试熔断器过滤：某个模式被熔断后不会出现在候选中。"""

    def test_open_breaker_excludes_mode(self, engine):
        broken_cb = CircuitBreaker(failure_threshold=1)
        broken_cb.record_failure()
        assert broken_cb.is_available() is False

        engine_with_cb = ModeSelectionEngine(
            circuit_breakers={AgentMode.SHARED_STATE: broken_cb}
        )
        assessment = {
            "complexity_score": 9.0,
            "task_type": "complex",
            "features": {},
        }
        # 重复调用多次，确保不会因为随机采样选中已熔断模式
        random.seed(7)
        for _ in range(20):
            selected = engine_with_cb.select_mode(assessment)
            assert selected is not AgentMode.SHARED_STATE


class TestColdStart:
    """测试冷启动先验：无历史数据时仍能返回候选。"""

    def test_cold_start_prior(self, engine):
        prior = engine._get_cold_start_prior("default", "generator_verifier")
        assert isinstance(prior, tuple)
        assert len(prior) == 2
        assert prior[0] > 0
        assert prior[1] > 0

    def test_select_mode_without_history(self, engine):
        assessment = {
            "complexity_score": 9.0,
            "task_type": "default",
            "features": {},
        }
        selected = engine.select_mode(assessment)
        assert isinstance(selected, AgentMode)
