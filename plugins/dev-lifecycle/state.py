from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import threading

DB_DIR = Path.home() / ".hermes" / "plugins" / "dev-lifecycle"
DB_PATH = DB_DIR / "workflow.db"

STAGE_ORDER = ["ideate", "build", "deliver"]

_lock = threading.Lock()


@dataclass
class WorkflowState:
    project_path: str
    current_stage: str
    skills_status: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


def _ensure_db() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS workflows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_path TEXT NOT NULL,
            current_stage TEXT NOT NULL,
            skills_status TEXT NOT NULL DEFAULT '{}',
            stage_skills_map TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS skill_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow_id INTEGER NOT NULL,
            skill_name TEXT NOT NULL,
            event_type TEXT NOT NULL,
            timestamp REAL NOT NULL,
            FOREIGN KEY (workflow_id) REFERENCES workflows(id)
        );
        """
    )
    conn.commit()
    conn.close()


def _row_to_state(row: tuple) -> WorkflowState:
    _, project_path, current_stage, skills_status_json, _, created_at, updated_at, _ = row
    return WorkflowState(
        project_path=project_path,
        current_stage=current_stage,
        skills_status=json.loads(skills_status_json),
        created_at=created_at,
        updated_at=updated_at,
    )


def _row_to_stage_map(row: tuple) -> Dict[str, list]:
    _, _, _, _, stage_skills_map_json, _, _, _ = row
    return json.loads(stage_skills_map_json)


class WorkflowManager:
    def __init__(self) -> None:
        _ensure_db()

    def start(self, project_path: str, lifecycle_config: dict) -> WorkflowState:
        skills_status: Dict[str, str] = {}
        stage_skills_map: Dict[str, list] = {}
        stages = lifecycle_config.get("stages", {})

        for stage_name, stage_cfg in stages.items():
            stage_skills = stage_cfg.get("skills", [])
            stage_skills_map[stage_name] = stage_skills
            for skill in stage_skills:
                skills_status[skill] = "pending"

        now = time.time()
        state = WorkflowState(
            project_path=project_path,
            current_stage="ideate",
            skills_status=skills_status,
            created_at=now,
            updated_at=now,
        )

        with _lock:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE workflows SET is_active = 0 WHERE project_path = ? AND is_active = 1",
                (project_path,),
            )
            cursor.execute(
                "INSERT INTO workflows (project_path, current_stage, skills_status, stage_skills_map, created_at, updated_at, is_active) "
                "VALUES (?, ?, ?, ?, ?, ?, 1)",
                (
                    state.project_path,
                    state.current_stage,
                    json.dumps(state.skills_status, ensure_ascii=False),
                    json.dumps(stage_skills_map, ensure_ascii=False),
                    state.created_at,
                    state.updated_at,
                ),
            )
            workflow_id = cursor.lastrowid

            ideate_skills = stage_skills_map.get("ideate", [])
            for skill in ideate_skills:
                cursor.execute(
                    "INSERT INTO skill_events (workflow_id, skill_name, event_type, timestamp) VALUES (?, ?, 'started', ?)",
                    (workflow_id, skill, now),
                )
            conn.commit()
            conn.close()

        return state

    def advance(self, workflow_id: int, skill_name: str) -> dict:
        with _lock:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, project_path, current_stage, skills_status, stage_skills_map, created_at, updated_at, is_active "
                "FROM workflows WHERE id = ? AND is_active = 1",
                (workflow_id,),
            )
            row = cursor.fetchone()
            if row is None:
                conn.close()
                raise ValueError(f"未找到活跃工作流: {workflow_id}")

            state = _row_to_state(row)
            stage_skills_map = _row_to_stage_map(row)

            if skill_name not in state.skills_status:
                conn.close()
                raise ValueError(f"技能 {skill_name} 不属于当前工作流")

            state.skills_status[skill_name] = "completed"
            now = time.time()
            state.updated_at = now

            cursor.execute(
                "INSERT INTO skill_events (workflow_id, skill_name, event_type, timestamp) VALUES (?, ?, 'completed', ?)",
                (workflow_id, skill_name, now),
            )

            current_stage_skills = stage_skills_map.get(state.current_stage, [])
            all_stage_done = all(
                state.skills_status.get(s) in ("completed", "skipped")
                for s in current_stage_skills
            )

            can_advance = False
            next_stage = None
            next_skill = None

            if all_stage_done:
                stage_idx = STAGE_ORDER.index(state.current_stage) if state.current_stage in STAGE_ORDER else -1
                if stage_idx < len(STAGE_ORDER) - 1:
                    next_stage = STAGE_ORDER[stage_idx + 1]
                    can_advance = True
                    next_stage_skills = stage_skills_map.get(next_stage, [])
                    next_skill = next_stage_skills[0] if next_stage_skills else None
                    state.current_stage = next_stage
                    state.updated_at = now
                elif state.current_stage == STAGE_ORDER[-1]:
                    # deliver 阶段所有技能完成，自动转为 complete 状态
                    state.current_stage = "complete"
                    state.updated_at = now

            cursor.execute(
                "UPDATE workflows SET current_stage = ?, skills_status = ?, updated_at = ? WHERE id = ?",
                (
                    state.current_stage,
                    json.dumps(state.skills_status, ensure_ascii=False),
                    state.updated_at,
                    workflow_id,
                ),
            )

            if can_advance and next_skill:
                cursor.execute(
                    "INSERT INTO skill_events (workflow_id, skill_name, event_type, timestamp) VALUES (?, ?, 'started', ?)",
                    (workflow_id, next_skill, now),
                )

            if state.current_stage == "complete":
                cursor.execute(
                    "UPDATE workflows SET is_active = 0 WHERE id = ?",
                    (workflow_id,),
                )

            conn.commit()
            conn.close()

        return {
            "next_skill": next_skill,
            "can_advance": can_advance,
            "current_stage": state.current_stage,
        }

    def rollback(self, workflow_id: int, to_stage: str) -> WorkflowState:
        if to_stage not in STAGE_ORDER:
            raise ValueError(f"无效阶段: {to_stage}")

        with _lock:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, project_path, current_stage, skills_status, stage_skills_map, created_at, updated_at, is_active "
                "FROM workflows WHERE id = ?",
                (workflow_id,),
            )
            row = cursor.fetchone()
            if row is None:
                conn.close()
                raise ValueError(f"未找到工作流: {workflow_id}")

            state = _row_to_state(row)
            stage_skills_map = _row_to_stage_map(row)
            now = time.time()

            current_idx = STAGE_ORDER.index(state.current_stage) if state.current_stage in STAGE_ORDER else len(STAGE_ORDER)
            target_idx = STAGE_ORDER.index(to_stage)

            if target_idx >= current_idx:
                conn.close()
                raise ValueError(f"只能回退到更早的阶段，当前: {state.current_stage}，目标: {to_stage}")

            for stage in STAGE_ORDER[target_idx + 1 : current_idx + 1]:
                stage_skills = stage_skills_map.get(stage, [])
                for skill in stage_skills:
                    if skill in state.skills_status and state.skills_status[skill] == "completed":
                        state.skills_status[skill] = "skipped"

            state.current_stage = to_stage
            state.updated_at = now

            cursor.execute(
                "UPDATE workflows SET current_stage = ?, skills_status = ?, updated_at = ?, is_active = 1 WHERE id = ?",
                (
                    state.current_stage,
                    json.dumps(state.skills_status, ensure_ascii=False),
                    state.updated_at,
                    workflow_id,
                ),
            )

            cursor.execute(
                "INSERT INTO skill_events (workflow_id, skill_name, event_type, timestamp) VALUES (?, ?, 'started', ?)",
                (workflow_id, f"rollback:{to_stage}", now),
            )

            conn.commit()
            conn.close()

        return state

    def get_status(self, workflow_id: int) -> WorkflowState:
        with _lock:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, project_path, current_stage, skills_status, stage_skills_map, created_at, updated_at, is_active "
                "FROM workflows WHERE id = ?",
                (workflow_id,),
            )
            row = cursor.fetchone()
            conn.close()

        if row is None:
            raise ValueError(f"未找到工作流: {workflow_id}")

        return _row_to_state(row)

    def list_active(self, project_path: Optional[str] = None) -> list:
        with _lock:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            if project_path:
                cursor.execute(
                    "SELECT id, project_path, current_stage, skills_status, stage_skills_map, created_at, updated_at, is_active "
                    "FROM workflows WHERE is_active = 1 AND project_path = ?",
                    (project_path,),
                )
            else:
                cursor.execute(
                    "SELECT id, project_path, current_stage, skills_status, stage_skills_map, created_at, updated_at, is_active "
                    "FROM workflows WHERE is_active = 1"
                )
            rows = cursor.fetchall()
            conn.close()

        return [_row_to_state(row) for row in rows]

    def resume(self, project_path: str) -> Optional[WorkflowState]:
        with _lock:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, project_path, current_stage, skills_status, stage_skills_map, created_at, updated_at, is_active "
                "FROM workflows WHERE project_path = ? AND is_active = 1 "
                "ORDER BY updated_at DESC LIMIT 1",
                (project_path,),
            )
            row = cursor.fetchone()
            conn.close()

        if row is None:
            return None

        return _row_to_state(row)
