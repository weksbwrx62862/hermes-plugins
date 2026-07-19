"""AMACheckpoint — LangGraph 风格的执行断点恢复。

基于现有的 ama_state_snapshots 表，提供:
1. 执行中自动落 checkpoint（每步保存状态）
2. 中断恢复：从最近的 checkpoint 继续执行
3. ama_resume 工具：手动恢复中断的任务
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .persistence import (
    _get_conn,
    load_snapshot,
    save_snapshot,
)

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """一个检查点的执行状态快照。"""
    trace_id: str
    round_num: int
    task: str
    task_type: str = ""
    complexity_score: float = 0.0
    mode: str = ""
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    results_so_far: dict[str, Any] = field(default_factory=dict)
    status: str = "checkpoint"  # checkpoint | completed | failed
    created_at: str = ""


class AMACheckpoint:
    """AMA 检查点管理器。

    用法：
        cp = AMACheckpoint()
        
        # 保存检查点
        cp.save(trace_id, round_num, state_dict)
        
        # 加载最近的检查点
        state = cp.load_latest(trace_id)
        
        # 查询可恢复的 trace
        resumeable = cp.list_resumeable()
    """

    _TRACE_LIFETIME_HOURS = 24  # trace 有效期为 24 小时

    @staticmethod
    def save(
        trace_id: str,
        round_num: int,
        task: str,
        mode: str = "",
        task_type: str = "",
        complexity_score: float = 0.0,
        completed_steps: list[str] | None = None,
        pending_steps: list[str] | None = None,
        results_so_far: dict[str, Any] | None = None,
    ) -> None:
        """保存一个检查点。"""
        state = CheckpointState(
            trace_id=trace_id,
            round_num=round_num,
            task=task,
            task_type=task_type,
            complexity_score=complexity_score,
            mode=mode,
            completed_steps=completed_steps or [],
            pending_steps=pending_steps or [],
            results_so_far=results_so_far or {},
            status="checkpoint",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            save_snapshot(trace_id, round_num, _to_dict(state))
        except Exception as e:
            logger.warning("AMACheckpoint save failed: %s", e)

    @staticmethod
    def load_latest(trace_id: str) -> Optional[CheckpointState]:
        """加载最近的检查点。"""
        try:
            conn = _get_conn()
            rows = conn.execute(
                """SELECT state_json, round FROM ama_state_snapshots 
                WHERE trace_id = ? 
                ORDER BY round DESC LIMIT 1""",
                (trace_id,),
            ).fetchall()
            if rows:
                state_json = rows[0][0]
                if isinstance(state_json, str):
                    return _from_dict(json.loads(state_json))
            return None
        except Exception as e:
            logger.warning("AMACheckpoint load_latest failed: %s", e)
            return None

    @staticmethod
    def mark_completed(trace_id: str) -> None:
        """标记 trace 为已完成。"""
        try:
            conn = _get_conn()
            conn.execute(
                "UPDATE ama_executions SET status='completed', updated_at=CURRENT_TIMESTAMP WHERE trace_id=?",
                (trace_id,),
            )
            conn.commit()
        except Exception as e:
            logger.warning("AMACheckpoint mark_completed failed: %s", e)

    @staticmethod
    def mark_interrupted(trace_id: str, reason: str = "") -> None:
        """标记 trace 为中断。"""
        try:
            conn = _get_conn()
            conn.execute(
                """UPDATE ama_executions SET status='interrupted', 
                error_category=?, updated_at=CURRENT_TIMESTAMP WHERE trace_id=?""",
                (f"interrupt:{reason}", trace_id),
            )
            conn.commit()
        except Exception as e:
            logger.warning("AMACheckpoint mark_interrupted failed: %s", e)

    @staticmethod
    def list_resumeable() -> list[dict[str, Any]]:
        """列出所有可恢复的 trace（24 小时内中断的）。"""
        try:
            conn = _get_conn()
            rows = conn.execute("""
                SELECT e.trace_id, e.task, e.task_type, e.mode_used, e.complexity_score,
                       e.created_at, e.status, e.error_category,
                       (SELECT MAX(s.round) FROM ama_state_snapshots s WHERE s.trace_id=e.trace_id) as max_round
                FROM ama_executions e
                WHERE e.status = 'interrupted'
                  AND e.created_at > datetime('now', '-24 hours')
                ORDER BY e.created_at DESC
                LIMIT 10
            """).fetchall()
            return [
                {
                    "trace_id": r[0],
                    "task": r[1],
                    "task_type": r[2],
                    "mode": r[3],
                    "complexity_score": r[4],
                    "created_at": r[5],
                    "status": r[6],
                    "error_category": r[7],
                    "last_checkpoint_round": r[8],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning("AMACheckpoint list_resumeable failed: %s", e)
            return []

    @staticmethod
    def generate_trace_id() -> str:
        """生成新的 trace_id。"""
        return str(uuid.uuid4())[:12]


def _to_dict(state: CheckpointState) -> dict:
    return {
        "trace_id": state.trace_id,
        "round_num": state.round_num,
        "task": state.task,
        "task_type": state.task_type,
        "complexity_score": state.complexity_score,
        "mode": state.mode,
        "completed_steps": state.completed_steps,
        "pending_steps": state.pending_steps,
        "results_so_far": state.results_so_far,
        "status": state.status,
        "created_at": state.created_at,
    }


def _from_dict(d: dict) -> CheckpointState:
    return CheckpointState(
        trace_id=d.get("trace_id", ""),
        round_num=d.get("round_num", 0),
        task=d.get("task", ""),
        task_type=d.get("task_type", ""),
        complexity_score=d.get("complexity_score", 0.0),
        mode=d.get("mode", ""),
        completed_steps=d.get("completed_steps", []),
        pending_steps=d.get("pending_steps", []),
        results_so_far=d.get("results_so_far", {}),
        status=d.get("status", "checkpoint"),
        created_at=d.get("created_at", ""),
    )
