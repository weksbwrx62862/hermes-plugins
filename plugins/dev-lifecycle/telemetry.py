"""dev-lifecycle 插件 — 遥测模块。

记录技能使用事件到 SQLite，生成使用报告和技能统计。
与 state.py 共享 workflow.db 数据库。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("plugins.dev-lifecycle.telemetry")

DEFAULT_DB_PATH = Path(
    "~/.hermes/plugins/dev-lifecycle/workflow.db"
).expanduser()


@dataclass
class TelemetryEvent:
    """遥测事件数据类。"""

    skill_name: str
    stage: str
    project_type: str
    timestamp: float = field(default_factory=time.time)
    duration: Optional[float] = None


class TelemetryRecorder:
    """遥测记录器，线程安全地记录和查询遥测事件。"""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    duration REAL,
                    project_type TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            self._conn.commit()

    def record(self, event: TelemetryEvent) -> None:
        """记录遥测事件。"""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO telemetry (skill_name, stage, duration, project_type, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.skill_name,
                    event.stage,
                    event.duration,
                    event.project_type,
                    event.timestamp,
                ),
            )
            self._conn.commit()
        logger.debug("遥测事件已记录: %s/%s", event.skill_name, event.stage)

    def report(self) -> Dict:
        """生成使用报告。"""
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
            if total == 0:
                return {
                    "skill_usage": {},
                    "stage_durations": {},
                    "skip_rate": {},
                    "total_events": 0,
                    "top_skills": [],
                }

            # 技能使用次数
            usage_rows = self._conn.execute(
                "SELECT skill_name, COUNT(*) AS cnt FROM telemetry GROUP BY skill_name"
            ).fetchall()
            skill_usage: Dict[str, int] = {
                r["skill_name"]: r["cnt"] for r in usage_rows
            }

            # 各阶段平均停留时间
            dur_rows = self._conn.execute(
                """
                SELECT stage, AVG(duration) AS avg_dur
                FROM telemetry
                WHERE duration IS NOT NULL
                GROUP BY stage
                """
            ).fetchall()
            stage_durations: Dict[str, float] = {
                r["stage"]: round(r["avg_dur"], 3) for r in dur_rows
            }

            # 各技能跳过率（duration 为 NULL 视为跳过）
            skip_rows = self._conn.execute(
                """
                SELECT skill_name,
                       SUM(CASE WHEN duration IS NULL THEN 1 ELSE 0 END) AS skipped,
                       COUNT(*) AS total
                FROM telemetry
                GROUP BY skill_name
                """
            ).fetchall()
            skip_rate: Dict[str, float] = {
                r["skill_name"]: round(r["skipped"] / r["total"], 4)
                for r in skip_rows
            }

            # 使用最多的前5个技能
            top_rows = self._conn.execute(
                """
                SELECT skill_name, COUNT(*) AS cnt
                FROM telemetry
                GROUP BY skill_name
                ORDER BY cnt DESC
                LIMIT 5
                """
            ).fetchall()
            top_skills: List[Tuple[str, int]] = [
                (r["skill_name"], r["cnt"]) for r in top_rows
            ]

        return {
            "skill_usage": skill_usage,
            "stage_durations": stage_durations,
            "skip_rate": skip_rate,
            "total_events": total,
            "top_skills": top_skills,
        }

    def get_skill_stats(self, skill_name: str) -> Dict:
        """获取单个技能的详细统计。"""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS total,
                       AVG(duration) AS avg_duration,
                       MIN(duration) AS min_duration,
                       MAX(duration) AS max_duration,
                       SUM(CASE WHEN duration IS NULL THEN 1 ELSE 0 END) AS skipped
                FROM telemetry
                WHERE skill_name = ?
                """,
                (skill_name,),
            ).fetchone()

            if row is None or row["total"] == 0:
                return {"skill_name": skill_name, "found": False}

            stage_rows = self._conn.execute(
                """
                SELECT stage, COUNT(*) AS cnt
                FROM telemetry
                WHERE skill_name = ?
                GROUP BY stage
                """,
                (skill_name,),
            ).fetchall()

            project_rows = self._conn.execute(
                """
                SELECT project_type, COUNT(*) AS cnt
                FROM telemetry
                WHERE skill_name = ?
                GROUP BY project_type
                """,
                (skill_name,),
            ).fetchall()

        total = row["total"]
        return {
            "skill_name": skill_name,
            "found": True,
            "total_uses": total,
            "avg_duration": round(row["avg_duration"], 3) if row["avg_duration"] else None,
            "min_duration": row["min_duration"],
            "max_duration": row["max_duration"],
            "skip_count": row["skipped"],
            "skip_rate": round(row["skipped"] / total, 4),
            "stages": {r["stage"]: r["cnt"] for r in stage_rows},
            "project_types": {r["project_type"]: r["cnt"] for r in project_rows},
        }

    def close(self) -> None:
        """关闭数据库连接。"""
        with self._lock:
            self._conn.close()
