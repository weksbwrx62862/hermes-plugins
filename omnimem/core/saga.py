"""SagaCoordinator — Saga 事务协调器。

解决 OmniMem 四数据源（Store/Index/Vector/BM25/KG）写入不一致问题。
设计原则：
  1. 主存储（Store）作为唯一事实来源，必须先成功
  2. 索引/检索/图谱作为派生数据，允许最终一致
  3. 失败时记录到 pending queue，由后台任务重试补偿
  4. 超过最大重试次数后丢弃，避免无限重试

不实现跨服务分布式事务，而是本地 Saga 模式：
  - 正向操作：按序执行各步骤
  - 反向补偿：目前阶段只记录失败，后台重试（而非回滚主存储）
    原因：记忆写入是追加操作，回滚意义不大，重试补索引更有价值。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnimem.utils.metrics import (
    get_alert_manager,
    record_saga_dead_letter,
    set_saga_pending_count,
)

logger = logging.getLogger(__name__)

# 重试配置
_MAX_RETRY_COUNT = 10
_BASE_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 300.0


@dataclass
class SagaStep:
    """Saga 单步定义。"""

    name: str
    action: Callable[[], Any]
    compensate: Callable[[], Any] | None = None


@dataclass
class SagaResult:
    """Saga 执行结果。"""

    success: bool
    memory_id: str
    completed_steps: list[str] = field(default_factory=list)
    failed_step: str = ""
    error: str = ""
    step_results: dict[str, Any] = field(default_factory=dict)


class SagaCoordinator:
    """Saga 事务协调器。

    职责：
      1. 编排多步骤写入（store → index → retriever → kg）
      2. 记录失败到 pending queue，含重试计数
      3. 提供 retry_pending() 供后台任务批量补偿
      4. 指数退避 + 最大重试次数，超限后丢弃并告警
    """

    def __init__(self, pending_path: Path | None = None,
                 max_retry: int = _MAX_RETRY_COUNT,
                 base_backoff: float = _BASE_BACKOFF_SECONDS,
                 max_backoff: float = _MAX_BACKOFF_SECONDS,
                 circuit_breaker_threshold: int = 5):
        """初始化 Saga 协调器。

        Args:
            pending_path: pending 队列持久化文件路径。
            max_retry: 最大重试次数，超限后丢弃。
            base_backoff: 指数退避基础间隔（秒）。
            max_backoff: 最大退避间隔（秒）。
            circuit_breaker_threshold: 连续失败达到此阈值时触发熔断，暂停重试。
        """
        self._pending: list[dict[str, Any]] = []
        self._pending_path = pending_path
        self._max_retry = max_retry
        self._base_backoff = base_backoff
        self._max_backoff = max_backoff
        self._total_retries = 0
        self._dead_letters: list[dict[str, Any]] = []
        # 熔断器状态
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._consecutive_failures = 0
        self._circuit_open = False
        if pending_path and pending_path.exists():
            self._load_pending()

    def execute(self, memory_id: str, steps: list[SagaStep]) -> SagaResult:
        """执行 Saga 事务。

        按顺序执行 steps，任一失败即停止，执行已完成步骤的补偿回调。
        成功执行的步骤返回值会被收集到 SagaResult.step_results 中。

        Args:
            memory_id: 记忆 ID，用于追踪和重试
            steps: Saga 步骤列表

        Returns:
            SagaResult，包含成功/失败状态、步骤详情和各步骤返回值
        """
        # 熔断器检查：连续失败过多时暂停执行
        if self._circuit_open:
            logger.warning(
                "Saga circuit breaker OPEN for %s (consecutive failures=%d) — skipping",
                memory_id, self._consecutive_failures,
            )
            return SagaResult(
                success=False,
                memory_id=memory_id,
                failed_step="__circuit_breaker__",
                error=f"circuit breaker open after {self._consecutive_failures} consecutive failures",
            )

        completed: list[str] = []
        completed_steps: list[SagaStep] = []
        step_results: dict[str, Any] = {}
        for step in steps:
            try:
                result = step.action()
                completed.append(step.name)
                completed_steps.append(step)
                step_results[step.name] = result
                logger.warning("Saga step '%s' OK for %s", step.name, memory_id)
            except Exception as e:
                logger.warning(
                    "Saga step '%s' failed for %s: %s",
                    step.name,
                    memory_id,
                    e,
                )
                # 执行已完成步骤的补偿回调（逆序）
                self._run_compensations(memory_id, completed_steps)
                # 更新熔断器计数
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._circuit_breaker_threshold:
                    self._circuit_open = True
                    logger.error(
                        "Saga circuit breaker TRIPPED after %d consecutive failures",
                        self._consecutive_failures,
                    )
                    # 触发熔断告警
                    get_alert_manager().fire(
                        name="saga_circuit_breaker_tripped",
                        severity="critical",
                        message=f"Saga 熔断器已触发（连续失败 {self._consecutive_failures} 次）",
                        consecutive_failures=self._consecutive_failures,
                        threshold=self._circuit_breaker_threshold,
                        memory_id=memory_id,
                        failed_step=step.name,
                    )
                record = {
                    "memory_id": memory_id,
                    "failed_step": step.name,
                    "completed_steps": completed,
                    "error": str(e),
                    "retry_count": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._pending.append(record)
                self._persist_pending()
                set_saga_pending_count(len(self._pending))
                return SagaResult(
                    success=False,
                    memory_id=memory_id,
                    completed_steps=completed,
                    failed_step=step.name,
                    error=str(e),
                    step_results=step_results,
                )
        # 成功：重置熔断器计数
        self._consecutive_failures = 0
        self._circuit_open = False
        return SagaResult(
            success=True,
            memory_id=memory_id,
            completed_steps=completed,
            step_results=step_results,
        )

    async def async_execute(self, steps: list[SagaStep], context: dict[str, Any]) -> dict[str, Any]:
        """异步执行 Saga 事务。

        与同步 execute 逻辑一致，但：
          - 异步 action/compensate 直接 await
          - 同步 action/compensate 使用 asyncio.to_thread 包装，避免阻塞事件循环
          - 保持相同的熔断器和重试逻辑

        Args:
            steps: Saga 步骤列表（SagaStep，action/compensate 可为同步或异步函数）
            context: 上下文字典，需包含 "memory_id" 键

        Returns:
            SagaResult 的字典表示，包含 success/memory_id/completed_steps/
            failed_step/error/step_results 字段
        """
        memory_id = context.get("memory_id", "")

        # 熔断器检查：连续失败过多时暂停执行
        if self._circuit_open:
            logger.warning(
                "Saga async circuit breaker OPEN for %s (consecutive failures=%d) — skipping",
                memory_id, self._consecutive_failures,
            )
            return {
                "success": False,
                "memory_id": memory_id,
                "completed_steps": [],
                "failed_step": "__circuit_breaker__",
                "error": f"circuit breaker open after {self._consecutive_failures} consecutive failures",
                "step_results": {},
            }

        completed: list[str] = []
        completed_steps: list[SagaStep] = []
        step_results: dict[str, Any] = {}

        for step in steps:
            try:
                # ★ 判断 action 是同步还是异步，分别处理
                if asyncio.iscoroutinefunction(step.action):
                    result = await step.action()
                else:
                    result = await asyncio.to_thread(step.action)
                completed.append(step.name)
                completed_steps.append(step)
                step_results[step.name] = result
                logger.warning("Saga async step '%s' OK for %s", step.name, memory_id)
            except Exception as e:
                logger.warning(
                    "Saga async step '%s' failed for %s: %s",
                    step.name, memory_id, e,
                )
                # 异步执行已完成步骤的补偿回调（逆序）
                await self._async_run_compensations(memory_id, completed_steps)
                # 更新熔断器计数
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._circuit_breaker_threshold:
                    self._circuit_open = True
                    logger.error(
                        "Saga async circuit breaker TRIPPED after %d consecutive failures",
                        self._consecutive_failures,
                    )
                    get_alert_manager().fire(
                        name="saga_circuit_breaker_tripped",
                        severity="critical",
                        message=f"Saga 异步熔断器已触发（连续失败 {self._consecutive_failures} 次）",
                        consecutive_failures=self._consecutive_failures,
                        threshold=self._circuit_breaker_threshold,
                        memory_id=memory_id,
                        failed_step=step.name,
                        phase="async",
                    )
                record = {
                    "memory_id": memory_id,
                    "failed_step": step.name,
                    "completed_steps": completed,
                    "error": str(e),
                    "retry_count": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._pending.append(record)
                self._persist_pending()
                set_saga_pending_count(len(self._pending))
                return {
                    "success": False,
                    "memory_id": memory_id,
                    "completed_steps": completed,
                    "failed_step": step.name,
                    "error": str(e),
                    "step_results": step_results,
                }

        # 成功：重置熔断器计数
        self._consecutive_failures = 0
        self._circuit_open = False
        return {
            "success": True,
            "memory_id": memory_id,
            "completed_steps": completed,
            "step_results": step_results,
        }

    async def _async_run_compensations(self, memory_id: str, completed_steps: list[SagaStep]) -> None:
        """异步逆序执行已完成步骤的补偿回调。

        补偿函数可为同步或异步：
          - 异步函数直接 await
          - 同步函数使用 asyncio.to_thread 包装
        """
        for step in reversed(completed_steps):
            if step.compensate is not None:
                try:
                    if asyncio.iscoroutinefunction(step.compensate):
                        await step.compensate()
                    else:
                        await asyncio.to_thread(step.compensate)
                    logger.info(
                        "Saga async compensate '%s' OK for %s", step.name, memory_id,
                    )
                except Exception as ce:
                    logger.warning(
                        "Saga async compensate '%s' failed for %s: %s",
                        step.name, memory_id, ce,
                    )

    def _run_compensations(self, memory_id: str, completed_steps: list[SagaStep]) -> None:
        """逆序执行已完成步骤的补偿回调。"""
        for step in reversed(completed_steps):
            if step.compensate is not None:
                try:
                    step.compensate()
                    logger.info(
                        "Saga compensate '%s' OK for %s", step.name, memory_id,
                    )
                except Exception as ce:
                    logger.warning(
                        "Saga compensate '%s' failed for %s: %s",
                        step.name, memory_id, ce,
                    )

    def reset_circuit_breaker(self) -> None:
        """手动重置熔断器状态。"""
        self._consecutive_failures = 0
        self._circuit_open = False

    def get_pending(self) -> list[dict[str, Any]]:
        """获取所有待重试的 pending 记录。"""
        return list(self._pending)

    def get_stats(self) -> dict[str, Any]:
        """获取 Saga 统计信息。"""
        return {
            "pending_count": len(self._pending),
            "total_retries": self._total_retries,
            "max_retry": self._max_retry,
        }

    def get_dead_letters(self) -> list[dict[str, Any]]:
        """获取所有 dead_letter 记录（超限未重试成功的记录）。"""
        return list(self._dead_letters)

    def _retry_pending_impl(
        self,
        step_actions: dict[str, Callable[[str], Any]],
        backoff_enabled: bool = True,
    ) -> int:
        """批量重试 pending 任务的内部实现。

        Args:
            step_actions: 步骤名 → (memory_id) -> Any 的映射。
            backoff_enabled: 是否启用指数退避等待。

        Returns:
            成功修复的条目数
        """
        if not self._pending:
            return 0

        # 熔断器检查
        if self._circuit_open:
            logger.warning(
                "Saga retry skipped — circuit breaker open (consecutive failures=%d)",
                self._consecutive_failures,
            )
            return 0

        label = "retry" if backoff_enabled else "auto-retry"
        fixed = 0
        still_pending: list[dict[str, Any]] = []

        for record in self._pending:
            memory_id = record.get("memory_id", "")
            failed_step = record.get("failed_step", "")
            retry_count = record.get("retry_count", 0)
            action = step_actions.get(failed_step)

            if not action:
                logger.warning("No retry action for step '%s', keeping pending", failed_step)
                still_pending.append(record)
                continue

            if retry_count >= self._max_retry:
                record["dead_letter_reason"] = "exceeded_max_retry"
                self._dead_letters.append(record)
                logger.warning(
                    "Saga record %s step '%s' exceeded max retry (%d) — moved to dead_letter. Error: %s",
                    memory_id, failed_step, self._max_retry, record.get("error", "unknown"),
                )
                # 记录 dead_letter 指标并触发告警
                record_saga_dead_letter()
                get_alert_manager().fire(
                    name="saga_dead_letter_accumulation",
                    severity="warning",
                    message=f"Saga 记录 {memory_id} 步骤 '{failed_step}' 超过最大重试次数 ({self._max_retry})，已转入 dead_letter",
                    memory_id=memory_id,
                    failed_step=failed_step,
                    retry_count=retry_count,
                    max_retry=self._max_retry,
                    error=record.get("error", "unknown"),
                    dead_letter_count=len(self._dead_letters),
                )
                continue

            if backoff_enabled:
                backoff = min(self._base_backoff * (2 ** retry_count), self._max_backoff)
                time.sleep(backoff)

            try:
                action(memory_id)
                logger.info("Saga %s OK: %s step '%s' (attempt %d)", label, memory_id, failed_step, retry_count + 1)
                self._total_retries += 1
                self._consecutive_failures = 0  # 成功时重置
                fixed += 1
            except Exception as e:
                record["retry_count"] = retry_count + 1
                record["last_error"] = str(e)
                logger.warning(
                    "Saga %s failed: %s step '%s' (attempt %d/%d): %s",
                    label, memory_id, failed_step, retry_count + 1, self._max_retry, e,
                )
                self._total_retries += 1
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._circuit_breaker_threshold:
                    self._circuit_open = True
                    logger.error(
                        "Saga circuit breaker TRIPPED during retry after %d consecutive failures",
                        self._consecutive_failures,
                    )
                    # 触发熔断告警（重试期间）
                    get_alert_manager().fire(
                        name="saga_circuit_breaker_tripped",
                        severity="critical",
                        message=f"Saga 熔断器在重试期间触发（连续失败 {self._consecutive_failures} 次）",
                        consecutive_failures=self._consecutive_failures,
                        threshold=self._circuit_breaker_threshold,
                        memory_id=memory_id,
                        failed_step=failed_step,
                        phase="retry",
                    )
                    # 熔断：剩余 pending 不再重试
                    still_pending.append(record)
                    still_pending.extend(self._pending[self._pending.index(record) + 1:])
                    break
                still_pending.append(record)

        self._pending = still_pending
        self._persist_pending()
        set_saga_pending_count(len(self._pending))
        return fixed

    def retry_pending(
        self,
        step_actions: dict[str, Callable[[str], Any]],
    ) -> int:
        """批量重试 pending 任务，含指数退避和最大重试次数。

        Args:
            step_actions: 步骤名 → (memory_id) -> Any 的映射。

        Returns:
            成功修复的条目数
        """
        return self._retry_pending_impl(step_actions, backoff_enabled=True)

    def auto_retry_pending(
        self,
        step_actions: dict[str, Callable[[str], Any]],
    ) -> int:
        """启动时自动重试 pending 任务，无指数退避。

        适合在 provider.initialize() 末尾调用，不需要退避等待。

        Args:
            step_actions: 步骤名 → (memory_id) -> Any 的映射。

        Returns:
            成功修复的条目数
        """
        return self._retry_pending_impl(step_actions, backoff_enabled=False)

    def clear_pending(self) -> int:
        """清空 pending 队列（谨慎使用）。返回清空的条目数。"""
        count = len(self._pending)
        self._pending.clear()
        self._persist_pending()
        set_saga_pending_count(0)
        return count

    # ─── 持久化 ─────────────────────────────────────────────

    def _persist_pending(self) -> None:
        """将 pending 队列持久化到磁盘。"""
        if not self._pending_path:
            return
        try:
            self._pending_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._pending_path, "w", encoding="utf-8") as f:
                json.dump(self._pending, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Saga pending persist failed: %s", e)

    def _load_pending(self) -> None:
        """从磁盘加载 pending 队列。"""
        if not self._pending_path or not self._pending_path.exists():
            return
        try:
            with open(self._pending_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._pending = data
                logger.info("Loaded %d pending saga records", len(self._pending))
        except Exception as e:
            logger.warning("Saga pending load failed: %s", e)
