"""subagent 模块单元测试（状态机、熔断器、重试策略）。"""

import pytest

from adaptive_multi_agent.subagent import (
    CircuitBreaker,
    RetryPolicy,
    TaskState,
    TaskStateMachine,
)


class TestTaskStateMachine:
    """测试 TaskStateMachine 合法/非法状态转换。"""

    @pytest.fixture
    def fsm(self):
        return TaskStateMachine(task_id="t-1")

    def test_initial_state(self, fsm):
        assert fsm.state == TaskState.INIT

    def test_valid_transition(self, fsm):
        assert fsm.transition(TaskState.PLANNING) is True
        assert fsm.state == TaskState.PLANNING
        assert fsm.transition(TaskState.EXECUTING) is True
        assert fsm.transition(TaskState.REVIEWING) is True
        assert fsm.transition(TaskState.DONE) is True

    def test_invalid_transition_blocked(self, fsm):
        assert fsm.transition(TaskState.DONE) is False
        assert fsm.state == TaskState.INIT

    def test_failed_is_terminal(self, fsm):
        fsm.transition(TaskState.PLANNING)
        fsm.transition(TaskState.FAILED)
        assert fsm.state == TaskState.FAILED
        assert fsm.transition(TaskState.DONE) is False

    def test_transition_callback(self):
        called = []

        def cb(from_state, to_state, task_id):
            called.append((from_state, to_state, task_id))

        fsm = TaskStateMachine(task_id="t-2", on_transition=cb)
        fsm.transition(TaskState.PLANNING)
        assert len(called) == 1
        assert called[0][2] == "t-2"


class TestCircuitBreaker:
    """测试 CircuitBreaker 达到阈值后进入不可用状态，冷却后可恢复。"""

    def test_closed_by_default(self):
        cb = CircuitBreaker()
        assert cb.is_available() is True
        assert cb.get_state() == CircuitBreaker.CLOSED

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_available() is False
        assert cb.get_state() == CircuitBreaker.OPEN

    def test_recovers_after_cooldown(self, monkeypatch):
        current_time = [1000.0]
        monkeypatch.setattr(
            "adaptive_multi_agent.subagent.time.time", lambda: current_time[0]
        )

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_available() is False

        # 时间推进超过恢复窗口
        current_time[0] += 15.0
        assert cb.is_available() is True
        assert cb.get_state() == CircuitBreaker.HALF_OPEN

    def test_success_closes_breaker(self, monkeypatch):
        current_time = [1000.0]
        monkeypatch.setattr(
            "adaptive_multi_agent.subagent.time.time", lambda: current_time[0]
        )

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5.0)
        cb.record_failure()
        current_time[0] += 10.0
        cb.is_available()  # 进入 HALF_OPEN
        cb.record_success()
        assert cb.get_state() == CircuitBreaker.CLOSED
        assert cb.is_available() is True


class TestRetryPolicy:
    """测试 RetryPolicy 基本行为。"""

    @pytest.fixture
    def policy(self):
        return RetryPolicy()

    def test_should_retry_rate_limit(self, policy):
        assert policy.should_retry("rate_limit", current_retry=0) is True
        assert policy.should_retry("rate_limit", current_retry=2) is True
        assert policy.should_retry("rate_limit", current_retry=3) is False

    def test_non_retryable_error(self, policy):
        assert policy.should_retry("validation", current_retry=0) is False

    def test_wait_time_increases_and_capped(self, policy):
        w0 = policy.get_wait_time(0)
        w1 = policy.get_wait_time(1)
        w2 = policy.get_wait_time(10)
        assert w1 > w0
        assert w2 <= policy.max_delay + policy.jitter_range * policy.max_delay

    def test_classify_rate_limit(self, policy):
        err = Exception("Rate limit exceeded: 429")
        assert policy.classify_error(err) == "rate_limit"

    def test_classify_internal_error(self, policy):
        err = ValueError("invalid something")
        assert policy.classify_error(err) == "internal_error"

    def test_classify_timeout(self, policy):
        err = Exception("execution expired after deadline")
        assert policy.classify_error(err) == "timeout"
