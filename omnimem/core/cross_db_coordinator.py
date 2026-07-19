from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class DbWriteStep:
    name: str
    action: Callable[[], Any]
    compensate: Callable[[], Any] | None = None
    max_retries: int = 3

@dataclass
class CrossDbResult:
    success: bool
    completed: list[str] = field(default_factory=list)
    failed: str | None = None
    compensated: list[str] = field(default_factory=list)

class CrossDbCoordinator:
    def __init__(self) -> None:
        self._pending: list[tuple[DbWriteStep, int]] = []

    def execute(self, steps: list[DbWriteStep]) -> CrossDbResult:
        completed: list[str] = []
        for step in steps:
            try:
                step.action()
                completed.append(step.name)
                logger.debug("CrossDb step '%s' succeeded", step.name)
            except Exception as e:
                logger.warning("CrossDb step '%s' failed: %s", step.name, e)
                self._compensate(completed, steps)
                self._pending.append((step, 0))
                return CrossDbResult(
                    success=False,
                    completed=completed,
                    failed=step.name,
                    compensated=list(reversed(completed)),
                )
        return CrossDbResult(success=True, completed=completed)

    def _compensate(self, completed: list[str], steps: list[DbWriteStep]) -> None:
        step_map = {s.name: s for s in steps}
        for name in reversed(completed):
            step = step_map.get(name)
            if step and step.compensate:
                try:
                    step.compensate()
                    logger.debug("CrossDb compensate '%s' succeeded", name)
                except Exception as e:
                    logger.warning("CrossDb compensate '%s' failed: %s", name, e)

    def retry_pending(self, backoff_enabled: bool = True) -> int:
        if not self._pending:
            return 0
        succeeded = 0
        remaining: list[tuple[DbWriteStep, int]] = []
        for step, attempt in self._pending:
            try:
                if backoff_enabled and attempt > 0:
                    time.sleep(min(0.1 * (2 ** attempt), 5.0))
                step.action()
                succeeded += 1
                logger.info("CrossDb retry step '%s' succeeded on attempt %d", step.name, attempt + 1)
            except Exception as e:
                next_attempt = attempt + 1
                if next_attempt >= step.max_retries:
                    logger.error(
                        "CrossDb step '%s' exceeded max_retries (%d): %s",
                        step.name, step.max_retries, e,
                    )
                else:
                    remaining.append((step, next_attempt))
                    logger.warning("CrossDb retry step '%s' failed (attempt %d): %s", step.name, next_attempt, e)
        self._pending = remaining
        return succeeded

    @property
    def has_pending(self) -> bool:
        return len(self._pending) > 0
