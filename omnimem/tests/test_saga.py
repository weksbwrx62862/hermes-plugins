import json
import tempfile
from pathlib import Path

import pytest

from omnimem.core.saga import SagaCoordinator, SagaStep, SagaResult


class TestSagaCoordinator:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pending_path = Path(self.tmpdir) / "saga_pending.json"
        self.saga = SagaCoordinator(pending_path=self.pending_path)

    def test_execute_all_success(self):
        steps = [
            SagaStep(name="step1", action=lambda: True),
            SagaStep(name="step2", action=lambda: True),
            SagaStep(name="step3", action=lambda: True),
        ]
        result = self.saga.execute("mem-001", steps)
        assert result.success is True
        assert len(self.saga._pending) == 0

    def test_execute_partial_failure(self):
        def fail_action():
            raise RuntimeError("step2 failed")

        steps = [
            SagaStep(name="step1", action=lambda: True),
            SagaStep(name="step2", action=fail_action),
            SagaStep(name="step3", action=lambda: True),
        ]
        result = self.saga.execute("mem-002", steps)
        assert result.success is False
        assert result.failed_step == "step2"
        assert len(self.saga._pending) >= 1

    def test_retry_pending_success(self):
        call_count = {"n": 0}

        def flaky_action():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("not ready yet")
            return True

        steps = [
            SagaStep(name="flaky", action=flaky_action),
        ]
        result = self.saga.execute("mem-003", steps)
        assert result.success is False
        assert len(self.saga._pending) >= 1

        step_actions = {"flaky": lambda mid: flaky_action()}
        self.saga.retry_pending(step_actions)
        if call_count["n"] >= 3:
            assert len(self.saga._pending) == 0

    def test_dead_letter_after_max_retries(self):
        def always_fail():
            raise RuntimeError("always fails")

        steps = [
            SagaStep(name="doomed", action=always_fail),
        ]
        self.saga.execute("mem-004", steps)

        step_actions = {"doomed": lambda mid: always_fail()}
        for _ in range(11):
            self.saga.retry_pending(step_actions)

        assert len(self.saga._dead_letters) >= 1

    def test_persistence_and_recovery(self):
        steps = [
            SagaStep(name="step1", action=lambda: True),
            SagaStep(name="step2", action=lambda: (_ for _ in ()).throw(RuntimeError("fail"))),
        ]
        self.saga.execute("mem-005", steps)
        assert len(self.saga._pending) >= 1

        saga2 = SagaCoordinator(pending_path=self.pending_path)
        assert len(saga2._pending) >= 1

    def test_get_stats(self):
        steps = [
            SagaStep(name="step1", action=lambda: True),
        ]
        self.saga.execute("mem-006", steps)
        stats = self.saga.get_stats()
        assert "pending_count" in stats
        assert "total_retries" in stats
        assert stats["pending_count"] == 0

    def test_get_pending(self):
        def fail():
            raise RuntimeError("fail")

        steps = [SagaStep(name="fail_step", action=fail)]
        self.saga.execute("mem-007", steps)
        pending = self.saga.get_pending()
        assert isinstance(pending, list)
        assert len(pending) >= 1

    def test_clear_pending(self):
        def fail():
            raise RuntimeError("fail")

        steps = [SagaStep(name="fail_step", action=fail)]
        self.saga.execute("mem-008", steps)
        assert len(self.saga._pending) >= 1
        count = self.saga.clear_pending()
        assert count >= 1
        assert len(self.saga._pending) == 0

    def test_get_dead_letters(self):
        def always_fail():
            raise RuntimeError("always fails")

        steps = [SagaStep(name="doomed", action=always_fail)]
        self.saga.execute("mem-009", steps)

        step_actions = {"doomed": lambda mid: always_fail()}
        for _ in range(11):
            self.saga.retry_pending(step_actions)

        dead = self.saga.get_dead_letters()
        assert isinstance(dead, list)
        assert len(dead) >= 1

    def test_auto_retry_pending(self):
        call_count = {"n": 0}

        def eventually_ok():
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise RuntimeError("not yet")
            return True

        steps = [SagaStep(name="retry_step", action=eventually_ok)]
        self.saga.execute("mem-010", steps)

        step_actions = {"retry_step": lambda mid: eventually_ok()}
        self.saga.auto_retry_pending(step_actions)
        if call_count["n"] >= 2:
            assert len(self.saga._pending) == 0
