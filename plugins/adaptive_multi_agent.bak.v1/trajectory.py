"""Agent 轨迹记录器 — 记录每次执行的完整轨迹

参考 Anthropic Agent Eval 设计：
- 记录任务输入、执行步骤、工具调用、输出结果
- 支持轨迹回放和错误分析
- 为 LLM-as-Judge 评估提供数据基础
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .persistence import AMAPersistence, get_persistence

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """单次工具调用记录"""
    tool_name: str
    args: Dict[str, Any]
    result: Any = None
    success: bool = True
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class StepRecord:
    """单步执行记录"""
    step_id: str
    step_type: str  # "plan" | "execute" | "verify" | "retry" | "tool_call" | "llm_call"
    description: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_data: Any = None
    output_data: Any = None
    success: bool = True
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Trajectory:
    """完整执行轨迹"""
    trajectory_id: str
    task: str
    context: Optional[str]
    mode: str  # 使用的执行模式
    complexity_score: float
    task_type: str
    steps: List[StepRecord] = field(default_factory=list)
    final_result: Any = None
    success: bool = False
    error: Optional[str] = None
    total_duration_ms: float = 0.0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """转换为可序列化的字典"""
        return {
            "trajectory_id": self.trajectory_id,
            "task": self.task,
            "context": self.context,
            "mode": self.mode,
            "complexity_score": self.complexity_score,
            "task_type": self.task_type,
            "steps": [asdict(s) for s in self.steps],
            "final_result": str(self.final_result)[:2000] if self.final_result else None,
            "success": self.success,
            "error": self.error,
            "total_duration_ms": self.total_duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


class TrajectoryRecorder:
    """轨迹记录器 — 记录 Agent 执行的完整轨迹

    核心功能：
    1. 记录每个执行步骤
    2. 记录工具调用及其结果
    3. 持久化到 SQLite
    4. 支持轨迹查询和回放
    """

    def __init__(self, data_dir: Optional[Path] = None):
        # data_dir 仅保留用于向后兼容； trajectory 数据现在统一存入 ama_state.db
        self._data_dir = data_dir
        self._persistence: AMAPersistence = get_persistence()
        self._lock = threading.Lock()
        self._current: Optional[Trajectory] = None

    def start(
        self,
        task: str,
        context: Optional[str] = None,
        mode: str = "",
        complexity_score: float = 0.0,
        task_type: str = "",
        metadata: Optional[Dict] = None,
    ) -> str:
        """开始记录新轨迹"""
        trajectory_id = str(uuid.uuid4())[:12]

        self._current = Trajectory(
            trajectory_id=trajectory_id,
            task=task,
            context=context,
            mode=mode,
            complexity_score=complexity_score,
            task_type=task_type,
            metadata=metadata or {},
        )

        logger.info("[Trajectory] 开始记录: %s | task=%s", trajectory_id, task[:100])
        return trajectory_id

    def add_step(
        self,
        step_type: str,
        description: str,
        input_data: Any = None,
        output_data: Any = None,
        success: bool = True,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> str:
        """添加执行步骤"""
        if not self._current:
            return ""

        step_id = str(uuid.uuid4())[:8]
        step = StepRecord(
            step_id=step_id,
            step_type=step_type,
            description=description,
            input_data=input_data,
            output_data=output_data,
            success=success,
            error=error,
            duration_ms=duration_ms,
        )
        self._current.steps.append(step)
        return step_id

    def add_tool_call(
        self,
        step_id: str,
        tool_name: str,
        args: Dict[str, Any],
        result: Any = None,
        success: bool = True,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
    ):
        """添加工具调用记录"""
        if not self._current:
            return

        # 找到对应的步骤
        for step in self._current.steps:
            if step.step_id == step_id:
                tool_call = ToolCall(
                    tool_name=tool_name,
                    args=args,
                    result=str(result)[:1000] if result else None,
                    success=success,
                    error=error,
                    duration_ms=duration_ms,
                )
                step.tool_calls.append(tool_call)
                break

    def finish(
        self,
        final_result: Any = None,
        success: bool = False,
        error: Optional[str] = None,
    ) -> Optional[Trajectory]:
        """完成轨迹记录并持久化"""
        if not self._current:
            return None

        self._current.final_result = final_result
        self._current.success = success
        self._current.error = error
        self._current.completed_at = time.time()
        self._current.total_duration_ms = (
            (self._current.completed_at - self._current.started_at) * 1000
        )

        # 持久化
        self._persist(self._current)

        trajectory = self._current
        self._current = None

        logger.info(
            "[Trajectory] 完成记录: %s | success=%s | steps=%d | duration=%.1fms",
            trajectory.trajectory_id,
            trajectory.success,
            len(trajectory.steps),
            trajectory.total_duration_ms,
        )
        return trajectory

    def _persist(self, trajectory: Trajectory):
        """持久化轨迹到统一数据库"""
        try:
            with self._lock:
                self._persistence.save_trajectory(trajectory)
        except Exception as e:
            logger.error("[Trajectory] 持久化失败: %s", e)

    def query(
        self,
        limit: int = 20,
        success_only: Optional[bool] = None,
        mode: Optional[str] = None,
    ) -> List[Dict]:
        """查询轨迹记录"""
        try:
            return self._persistence.query_trajectories(
                limit=limit, success_only=success_only, mode=mode
            )
        except Exception as e:
            logger.error("[Trajectory] 查询失败: %s", e)
            return []

    def get_trajectory(self, trajectory_id: str) -> Optional[Dict]:
        """获取完整轨迹详情"""
        try:
            return self._persistence.get_trajectory(trajectory_id)
        except Exception as e:
            logger.error("[Trajectory] 获取详情失败: %s", e)
            return None

    def update_grade(self, trajectory_id: str, score: float, feedback: str):
        """更新轨迹评分"""
        try:
            with self._lock:
                self._persistence.update_grade(trajectory_id, score, feedback)
        except Exception as e:
            logger.error("[Trajectory] 更新评分失败: %s", e)

    def get_stats(self) -> Dict:
        """获取轨迹统计"""
        try:
            return self._persistence.get_trajectory_stats()
        except Exception as e:
            logger.error("[Trajectory] 统计失败: %s", e)
            return {}


# 全局实例
_recorder: Optional[TrajectoryRecorder] = None
_recorder_lock = threading.Lock()


def get_recorder() -> TrajectoryRecorder:
    """获取全局轨迹记录器"""
    global _recorder
    if _recorder is None:
        with _recorder_lock:
            if _recorder is None:
                _recorder = TrajectoryRecorder()
    return _recorder
