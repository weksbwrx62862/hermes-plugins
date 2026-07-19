from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path.home() / ".hermes" / "ama_state.db"

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS ama_performance (
    task_type TEXT NOT NULL,
    mode TEXT NOT NULL,
    trials INTEGER NOT NULL DEFAULT 0,
    successes INTEGER NOT NULL DEFAULT 0,
    avg_tokens REAL NOT NULL DEFAULT 0,
    avg_time REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (task_type, mode)
);

CREATE TABLE IF NOT EXISTS ama_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    task TEXT,
    task_type TEXT,
    complexity_score REAL,
    mode_used TEXT,
    original_mode TEXT,
    success INTEGER,
    token_usage INTEGER,
    time_taken REAL,
    switched_modes INTEGER,
    switch_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ama_state_snapshots (
    trace_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trace_id, round)
);

CREATE TABLE IF NOT EXISTS ama_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_hash TEXT NOT NULL,
    task TEXT NOT NULL,
    result TEXT NOT NULL,
    task_type TEXT,
    mode TEXT,
    success INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ama_memory_task_hash ON ama_memory(task_hash);

CREATE TABLE IF NOT EXISTS ama_trajectories (
    trajectory_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    context TEXT,
    mode TEXT,
    complexity_score REAL,
    task_type TEXT,
    steps_json TEXT,
    final_result TEXT,
    success INTEGER,
    error TEXT,
    total_duration_ms REAL,
    started_at REAL,
    completed_at REAL,
    metadata_json TEXT,
    grade_score REAL,
    grade_feedback TEXT
);
CREATE INDEX IF NOT EXISTS idx_trajectories_started_at ON ama_trajectories(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_trajectories_success ON ama_trajectories(success);

-- Trellis-inspired: session journal for cross-session memory
CREATE TABLE IF NOT EXISTS ama_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_hash TEXT NOT NULL,
    task TEXT NOT NULL,
    task_type TEXT,
    status TEXT NOT NULL DEFAULT 'in_progress',
    journal_text TEXT,
    spec_ref TEXT,
    timeline_json TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ama_journal_task_hash ON ama_journal(task_hash);
CREATE INDEX IF NOT EXISTS idx_ama_journal_status ON ama_journal(status);

-- Trellis-inspired: task-level spec definitions
CREATE TABLE IF NOT EXISTS ama_spec (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_key TEXT NOT NULL UNIQUE,
    spec_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    content TEXT NOT NULL,
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ama_spec_task_type ON ama_spec(task_type);
"""

_MIGRATE_SQL = [
    "ALTER TABLE ama_executions ADD COLUMN trace_id TEXT",
    "ALTER TABLE ama_executions ADD COLUMN status TEXT",
    "ALTER TABLE ama_executions ADD COLUMN error_category TEXT",
    "ALTER TABLE ama_executions ADD COLUMN retries_attempted INTEGER DEFAULT 0",
    "ALTER TABLE ama_executions ADD COLUMN timeout_seconds INTEGER",
    "ALTER TABLE ama_executions ADD COLUMN updated_at TIMESTAMP",
]


class AMAPersistence:
    """AMA 统一持久化层。

    封装 SQLite 连接、事务和迁移，管理单一数据库 ``~/.hermes/ama_state.db``，
    集中维护 performance、executions、snapshots、memory 和 trajectories 数据。
    """

    _instance: Optional["AMAPersistence"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, db_path: Optional[Path] = None) -> "AMAPersistence":
        # 默认路径使用单例，避免重复建立连接池
        if db_path is None:
            if cls._instance is None:
                with cls._instance_lock:
                    if cls._instance is None:
                        cls._instance = super().__new__(cls)
                        cls._instance._initialized = False
            return cls._instance
        return super().__new__(cls)

    def __init__(self, db_path: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return
        self.db_path = Path(db_path) if db_path else _DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        """建立新的 SQLite 连接。"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        """初始化表结构并执行迁移。"""
        conn = self._connect()
        try:
            conn.executescript(_CREATE_TABLES)
            # 兼容迁移：添加新列（忽略已存在错误）
            for sql in _MIGRATE_SQL:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """提供事务上下文管理器，确保提交或回滚。"""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── performance ──

    def load_performance(self) -> Dict[str, Dict[str, Dict]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT task_type, mode, trials, successes, avg_tokens, avg_time FROM ama_performance"
            ).fetchall()
            result: Dict[str, Dict[str, Dict]] = {}
            for row in rows:
                result.setdefault(row["task_type"], {})[row["mode"]] = {
                    "trials": row["trials"],
                    "successes": row["successes"],
                    "avg_tokens": row["avg_tokens"],
                    "avg_time": row["avg_time"],
                }
            return result
        finally:
            conn.close()

    def save_performance(self, task_type: str, mode: str, stats: Dict) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO ama_performance (task_type, mode, trials, successes, avg_tokens, avg_time)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(task_type, mode) DO UPDATE SET
                     trials=excluded.trials, successes=excluded.successes,
                     avg_tokens=excluded.avg_tokens, avg_time=excluded.avg_time""",
                (
                    task_type,
                    mode,
                    stats["trials"],
                    stats["successes"],
                    stats["avg_tokens"],
                    stats["avg_time"],
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ── executions ──

    def record_execution(
        self,
        session_id: Optional[str],
        task: str,
        task_type: str,
        complexity_score: float,
        mode_used: str,
        original_mode: Optional[str],
        success: bool,
        token_usage: int,
        time_taken: float,
        switched_modes: bool,
        switch_reason: Optional[str] = None,
        trace_id: Optional[str] = None,
        status: Optional[str] = None,
        error_category: Optional[str] = None,
        retries_attempted: int = 0,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO ama_executions
                   (session_id, task, task_type, complexity_score, mode_used, original_mode,
                    success, token_usage, time_taken, switched_modes, switch_reason,
                    trace_id, status, error_category, retries_attempted, timeout_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    task,
                    task_type,
                    complexity_score,
                    mode_used,
                    original_mode,
                    int(success),
                    token_usage,
                    time_taken,
                    int(switched_modes),
                    switch_reason,
                    trace_id,
                    status,
                    error_category,
                    retries_attempted,
                    timeout_seconds,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def save_execution_transaction(
        self,
        task_type: str,
        mode: str,
        stats: Dict,
        session_id,
        task: str,
        complexity_score: float,
        mode_used: str,
        original_mode,
        success: bool,
        token_usage: int,
        time_taken: float,
        switched_modes: bool,
        switch_reason=None,
        trace_id=None,
        status=None,
        error_category=None,
        retries_attempted: int = 0,
        timeout_seconds=None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO ama_performance (task_type, mode, trials, successes, avg_tokens, avg_time)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(task_type, mode) DO UPDATE SET
                     trials=excluded.trials, successes=excluded.successes,
                     avg_tokens=excluded.avg_tokens, avg_time=excluded.avg_time""",
                (task_type, mode, stats["trials"], stats["successes"], stats["avg_tokens"], stats["avg_time"]),
            )
            conn.execute(
                """INSERT INTO ama_executions
                   (session_id, task, task_type, complexity_score, mode_used, original_mode,
                    success, token_usage, time_taken, switched_modes, switch_reason,
                    trace_id, status, error_category, retries_attempted, timeout_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, task, task_type, complexity_score, mode_used, original_mode,
                 int(success), token_usage, time_taken, int(switched_modes), switch_reason,
                 trace_id, status, error_category, retries_attempted, timeout_seconds),
            )

    def get_execution_by_trace_id(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """按 trace_id 查询执行记录"""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM ama_executions WHERE trace_id = ? ORDER BY created_at DESC LIMIT 1",
                (trace_id,),
            ).fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def get_stats(self, detail: bool = False, period: str = "all") -> Dict[str, Any]:
        conn = self._connect()
        try:
            # 时间过滤
            time_filter = ""
            time_args: list = []
            if period == "day":
                time_filter = "WHERE created_at >= datetime('now', ?)"
                time_args = ["-1 day"]
            elif period == "week":
                time_filter = "WHERE created_at >= datetime('now', ?)"
                time_args = ["-7 days"]
            elif period == "month":
                time_filter = "WHERE created_at >= datetime('now', ?)"
                time_args = ["-30 days"]

            total = conn.execute(f"SELECT COUNT(*) as c FROM ama_executions {time_filter}", time_args).fetchone()["c"]
            if total == 0:
                return {"total_executions": 0, "period": period}

            mode_usage = {}
            for row in conn.execute(
                f"SELECT mode_used, COUNT(*) as cnt FROM ama_executions {time_filter} GROUP BY mode_used", time_args
            ).fetchall():
                mode_usage[row["mode_used"]] = row["cnt"]

            success_rates = {}
            for row in conn.execute(
                f"SELECT mode_used, SUM(success) as s, COUNT(*) as t FROM ama_executions {time_filter} GROUP BY mode_used", time_args
            ).fetchall():
                success_rates[row["mode_used"]] = {
                    "rate": row["s"] / row["t"],
                    "count": row["t"],
                }

            # 趋势数据：按天聚合，用于可视化
            trend_sql = f"""
                SELECT date(created_at) as day,
                       mode_used,
                       COUNT(*) as cnt,
                       SUM(success) as successes
                FROM ama_executions {time_filter}
                GROUP BY date(created_at), mode_used
                ORDER BY day DESC
            """
            trends = {}
            for row in conn.execute(trend_sql, time_args).fetchall():
                d = row["day"]
                m = row["mode_used"]
                trends.setdefault(d, {})[m] = {
                    "count": row["cnt"],
                    "successes": row["successes"],
                }

            # 模式切换统计
            switch_where = f"{time_filter} AND" if time_filter else "WHERE"
            switch_args = time_args if time_args else []
            switch_total = conn.execute(
                f"SELECT COUNT(*) as c FROM ama_executions {switch_where} switched_modes=1", switch_args
            ).fetchone()["c"]
            switch_rate = switch_total / total if total > 0 else 0

            result = {
                "total_executions": total,
                "period": period,
                "mode_usage": mode_usage,
                "success_rates": success_rates,
                "switch_events": switch_total,
                "switch_rate": switch_rate,
                "daily_trends": trends,
            }

            if detail:
                perf = {}
                for row in conn.execute("SELECT * FROM ama_performance").fetchall():
                    perf.setdefault(row["task_type"], {})[row["mode"]] = {
                        "trials": row["trials"],
                        "successes": row["successes"],
                        "avg_tokens": row["avg_tokens"],
                        "avg_time": row["avg_time"],
                    }
                result["historical_performance"] = perf

            recent_rows = conn.execute(
                f"SELECT * FROM ama_executions {time_filter} ORDER BY created_at DESC LIMIT 5",
                time_args,
            ).fetchall()
            result["recent_executions"] = [dict(r) for r in recent_rows]

            return result
        finally:
            conn.close()

    # ── snapshots ──

    def save_snapshot(self, trace_id: str, round_num: int, state: Dict) -> None:
        """保存 shared_state 快照"""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO ama_state_snapshots (trace_id, round, state_json)
                   VALUES (?, ?, ?)
                   ON CONFLICT(trace_id, round) DO UPDATE SET state_json=excluded.state_json""",
                (trace_id, round_num, json.dumps(state, ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()

    def load_snapshot(self, trace_id: str, round_num: int) -> Optional[Dict]:
        """加载 shared_state 快照"""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT state_json FROM ama_state_snapshots WHERE trace_id = ? AND round = ?",
                (trace_id, round_num),
            ).fetchone()
            if row:
                return json.loads(row["state_json"])
            return None
        finally:
            conn.close()

    # ── memory ──

    def save_memory(self, task: str, result: str, task_type: str = "", mode: str = "", success: bool = True) -> None:
        """保存任务-结果对到记忆层"""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO ama_memory (task_hash, task, result, task_type, mode, success)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self._task_hash(task), task, result, task_type, mode, int(success)),
            )
            conn.commit()
        finally:
            conn.close()

    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """基于关键词匹配检索历史任务结果（简化版，无需 embedding）"""
        conn = self._connect()
        try:
            keywords = query.lower().split()[:5]
            if not keywords:
                return []
            conditions = []
            args = []
            for kw in keywords:
                conditions.append("(task LIKE ? OR result LIKE ?)")
                args.extend([f"%{kw}%", f"%{kw}%"])
            where_clause = " AND ".join(conditions)
            rows = conn.execute(
                f"SELECT task, result, task_type, mode, success, created_at FROM ama_memory WHERE {where_clause} ORDER BY created_at DESC LIMIT ?",
                args + [limit],
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    @staticmethod
    def _task_hash(task: str) -> str:
        """计算任务描述的哈希值，用于快速查找"""
        return hashlib.md5(task.encode()).hexdigest()

    # ── trajectories ──

    def save_trajectory(self, trajectory: Any) -> None:
        """保存或更新一条完整轨迹。

        Args:
            trajectory: ``Trajectory`` 对象或其 ``to_dict`` 结果字典。
        """
        if hasattr(trajectory, "to_dict"):
            data = trajectory.to_dict()
        else:
            data = trajectory

        steps_json = json.dumps(data.get("steps", []), ensure_ascii=False)
        metadata_json = json.dumps(data.get("metadata", {}), ensure_ascii=False)
        final_result = data.get("final_result")
        if final_result is not None:
            final_result = str(final_result)[:5000]

        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO ama_trajectories
                   (trajectory_id, task, context, mode, complexity_score, task_type,
                    steps_json, final_result, success, error, total_duration_ms,
                    started_at, completed_at, metadata_json, grade_score, grade_feedback)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["trajectory_id"],
                    data["task"],
                    data.get("context"),
                    data.get("mode"),
                    data.get("complexity_score"),
                    data.get("task_type"),
                    steps_json,
                    final_result,
                    1 if data.get("success") else 0,
                    data.get("error"),
                    data.get("total_duration_ms"),
                    data.get("started_at"),
                    data.get("completed_at"),
                    metadata_json,
                    data.get("grade_score"),
                    data.get("grade_feedback"),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_trajectory(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """获取完整轨迹详情。"""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM ama_trajectories WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_trajectory(dict(row))
        finally:
            conn.close()

    def query_trajectories(
        self,
        limit: int = 20,
        success_only: Optional[bool] = None,
        mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """查询轨迹记录列表，返回简化字段。"""
        conditions = []
        params: List[Any] = []

        if success_only is not None:
            conditions.append("success = ?")
            params.append(1 if success_only else 0)

        if mode:
            conditions.append("mode = ?")
            params.append(mode)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        conn = self._connect()
        try:
            rows = conn.execute(
                f"""SELECT trajectory_id, task, mode, complexity_score, success,
                           total_duration_ms, started_at, error, grade_score
                    FROM ama_trajectories {where}
                    ORDER BY started_at DESC LIMIT ?""",
                params + [limit],
            ).fetchall()

            return [
                {
                    "trajectory_id": r["trajectory_id"],
                    "task": r["task"][:100] if r["task"] else "",
                    "mode": r["mode"],
                    "complexity_score": r["complexity_score"],
                    "success": bool(r["success"]),
                    "duration_ms": r["total_duration_ms"],
                    "started_at": datetime.fromtimestamp(r["started_at"]).isoformat() if r["started_at"] else None,
                    "error": r["error"],
                    "grade_score": r["grade_score"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def update_grade(self, trajectory_id: str, score: float, feedback: str) -> None:
        """更新轨迹评分。"""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE ama_trajectories SET grade_score = ?, grade_feedback = ? WHERE trajectory_id = ?",
                (score, feedback, trajectory_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_trajectory_stats(self) -> Dict[str, Any]:
        """获取轨迹统计。"""
        conn = self._connect()
        try:
            total = conn.execute("SELECT COUNT(*) as c FROM ama_trajectories").fetchone()["c"]
            success = conn.execute("SELECT COUNT(*) as c FROM ama_trajectories WHERE success = 1").fetchone()["c"]
            failed = total - success

            avg_duration = conn.execute(
                "SELECT AVG(total_duration_ms) as avg FROM ama_trajectories WHERE success = 1"
            ).fetchone()["avg"] or 0

            avg_grade = conn.execute(
                "SELECT AVG(grade_score) as avg FROM ama_trajectories WHERE grade_score IS NOT NULL"
            ).fetchone()["avg"] or 0

            return {
                "total": total,
                "success": success,
                "failed": failed,
                "success_rate": success / total if total > 0 else 0,
                "avg_duration_ms": round(avg_duration, 1),
                "avg_grade_score": round(avg_grade, 2),
                "db_path": str(self.db_path),
            }
        finally:
            conn.close()

    @staticmethod
    def _row_to_trajectory(row: Dict[str, Any]) -> Dict[str, Any]:
        """将数据库行转换为轨迹字典。"""
        return {
            "trajectory_id": row["trajectory_id"],
            "task": row["task"],
            "context": row["context"],
            "mode": row["mode"],
            "complexity_score": row["complexity_score"],
            "task_type": row["task_type"],
            "steps": json.loads(row["steps_json"]) if row["steps_json"] else [],
            "final_result": row["final_result"],
            "success": bool(row["success"]),
            "error": row["error"],
            "total_duration_ms": row["total_duration_ms"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            "grade_score": row["grade_score"],
            "grade_feedback": row["grade_feedback"],
        }

    # ── Trellis-inspired: journal & spec ──────────────────────

    def upsert_journal(
        self, task: str, task_type: str = "", journal_text: str = "",
        spec_ref: str = "", status: str = "in_progress",
        timeline_entry: Optional[Dict] = None,
    ) -> int:
        """创建或更新会话日志（跨会话记忆）。

        若同 task_hash 已有记录，追加 journal_text 并合并 timeline；
        否则新建。
        """
        task_hash = self._task_hash(task)
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT id, journal_text, timeline_json, status, spec_ref FROM ama_journal WHERE task_hash=?",
                (task_hash,),
            ).fetchone()
            if existing:
                merged_text = (existing["journal_text"] or "") + "\n" + journal_text
                timeline = json.loads(existing["timeline_json"] or "[]")
                if timeline_entry:
                    timeline_entry["_ts"] = datetime.now().isoformat()
                    timeline.append(timeline_entry)
                conn.execute(
                    """UPDATE ama_journal
                       SET journal_text=?, timeline_json=?, updated_at=CURRENT_TIMESTAMP,
                           status=?, spec_ref=?
                       WHERE id=?""",
                    (merged_text.strip(), json.dumps(timeline, ensure_ascii=False),
                     status, spec_ref if spec_ref else (existing["spec_ref"] or ""), existing["id"]),
                )
            else:
                timeline = [timeline_entry] if timeline_entry else []
                if timeline_entry:
                    timeline_entry["_ts"] = datetime.now().isoformat()
                conn.execute(
                    """INSERT INTO ama_journal (task_hash, task, task_type, status, journal_text, spec_ref, timeline_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (task_hash, task, task_type, status, journal_text.strip(),
                     spec_ref, json.dumps(timeline, ensure_ascii=False)),
                )
            conn.commit()
            return existing["id"] if existing else conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        finally:
            conn.close()

    def get_journal(self, task_hash: str) -> Optional[Dict[str, Any]]:
        """按 task_hash 读取会话日志。"""
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM ama_journal WHERE task_hash=?", (task_hash,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def search_journals(self, task_type: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        """按 task_type 搜索相关历史日志。"""
        conn = self._connect()
        try:
            if task_type:
                rows = conn.execute(
                    "SELECT * FROM ama_journal WHERE task_type=? ORDER BY updated_at DESC LIMIT ?",
                    (task_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ama_journal ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def upsert_spec(self, spec_key: str, spec_name: str, task_type: str, content: str, priority: int = 0) -> None:
        """创建或更新任务规范。"""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO ama_spec (spec_key, spec_name, task_type, content, priority)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(spec_key) DO UPDATE SET
                   spec_name=excluded.spec_name, task_type=excluded.task_type,
                   content=excluded.content, priority=excluded.priority,
                   updated_at=CURRENT_TIMESTAMP""",
                (spec_key, spec_name, task_type, content, priority),
            )
            conn.commit()
        finally:
            conn.close()

    def get_spec(self, spec_key: str) -> Optional[Dict[str, Any]]:
        """按 spec_key 读取规范。"""
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM ama_spec WHERE spec_key=?", (spec_key,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_specs(self, task_type: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        """列出任务相关的所有规范（按优先级降序）。"""
        conn = self._connect()
        try:
            if task_type:
                rows = conn.execute(
                    "SELECT * FROM ama_spec WHERE task_type=? ORDER BY priority DESC, updated_at DESC LIMIT ?",
                    (task_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ama_spec ORDER BY priority DESC, updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def build_spec_context(self, task_type: str = "", spec_keys: Optional[List[str]] = None) -> str:
        """组合生成注入子代理的规范上下文字符串。

        Args:
            task_type: 按任务类型加载相关 spec
            spec_keys: 额外指定 spec key 列表
        """
        parts = []
        if spec_keys:
            for key in spec_keys:
                spec = self.get_spec(key)
                if spec:
                    parts.append(f"## 规范: {spec['spec_name']}\n{spec['content']}")
        if task_type:
            specs = self.list_specs(task_type)
            for spec in specs:
                if spec["spec_key"] not in (spec_keys or []):
                    parts.append(f"## 规范: {spec['spec_name']}\n{spec['content']}")
        return "\n\n---\n\n".join(parts) if parts else ""


def _get_conn() -> sqlite3.Connection:
    """获取默认持久化实例的 SQLite 连接（供 checkpoint 等模块直接使用）。"""
    return get_persistence()._connect()


# 模块级默认持久化实例（单例）
_persistence: Optional[AMAPersistence] = None
_persistence_lock = threading.Lock()


def get_persistence(db_path: Optional[Path] = None) -> AMAPersistence:
    """获取默认 AMAPersistence 实例（单例）。"""
    global _persistence
    if _persistence is None:
        with _persistence_lock:
            if _persistence is None:
                _persistence = AMAPersistence(db_path=db_path)
    return _persistence


_LEGACY_TRAJECTORIES_DIR = Path.home() / ".hermes" / "ama_trajectories"
_LEGACY_TRAJECTORIES_DB = _LEGACY_TRAJECTORIES_DIR / "trajectories.db"
_LEGACY_MIGRATED_MARKER = _LEGACY_TRAJECTORIES_DIR / ".migrated"


def migrate_legacy_trajectories(
    legacy_db_path: Optional[Path] = None,
    migrated_marker: Optional[Path] = None,
    persistence: Optional[AMAPersistence] = None,
) -> int:
    """将旧版独立 trajectories.db 迁移到统一的 ama_state.db。

    迁移逻辑：
    1. 检测旧数据库文件是否存在；不存在则直接返回 0。
    2. 检测 .migrated 标记文件；已存在则跳过，避免重复迁移。
    3. 读取旧表 ``trajectories`` 的每条记录，解析 JSON 字段后构造轨迹字典。
    4. 调用 ``AMAPersistence.save_trajectory`` 写入统一数据库。
    5. 写入 .migrated 标记文件，记录迁移时间戳与数量。
    6. 保留旧数据库文件，不删除，作为备份。

    Args:
        legacy_db_path: 旧数据库路径，默认 ``~/.hermes/ama_trajectories/trajectories.db``。
        migrated_marker: 迁移标记文件路径，默认 ``~/.hermes/ama_trajectories/.migrated``。
        persistence: 指定的 AMAPersistence 实例，默认使用全局单例。

    Returns:
        成功迁移的轨迹数量。
    """
    legacy_db = Path(legacy_db_path) if legacy_db_path else _LEGACY_TRAJECTORIES_DB
    marker = Path(migrated_marker) if migrated_marker else _LEGACY_MIGRATED_MARKER

    if not legacy_db.exists():
        logger.info("旧版 trajectories.db 不存在，跳过迁移: %s", legacy_db)
        return 0

    if marker.exists():
        logger.info("已存在迁移标记 %s，跳过重复迁移", marker)
        return 0

    pers = persistence or get_persistence()
    migrated_count = 0

    conn = sqlite3.connect(str(legacy_db))
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM trajectories").fetchall()

        for row in rows:
            row_dict = dict(row)
            steps_json = row_dict.get("steps_json")
            metadata_json = row_dict.get("metadata_json")

            try:
                steps = json.loads(steps_json) if steps_json else []
            except (json.JSONDecodeError, TypeError):
                steps = []

            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            trajectory_data = {
                "trajectory_id": row_dict.get("trajectory_id"),
                "task": row_dict.get("task", ""),
                "context": row_dict.get("context"),
                "mode": row_dict.get("mode"),
                "complexity_score": row_dict.get("complexity_score"),
                "task_type": row_dict.get("task_type"),
                "steps": steps,
                "final_result": row_dict.get("final_result"),
                "success": bool(row_dict.get("success")),
                "error": row_dict.get("error"),
                "total_duration_ms": row_dict.get("total_duration_ms"),
                "started_at": row_dict.get("started_at"),
                "completed_at": row_dict.get("completed_at"),
                "metadata": metadata,
                "grade_score": row_dict.get("grade_score"),
                "grade_feedback": row_dict.get("grade_feedback"),
            }
            pers.save_trajectory(trajectory_data)
            migrated_count += 1

        logger.info(
            "已从 %s 迁移 %d 条轨迹记录到 %s",
            legacy_db,
            migrated_count,
            pers.db_path,
        )
    finally:
        conn.close()

    # 写入迁移标记，防止重复迁移
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        f"migrated_at={datetime.now().isoformat()}\ncount={migrated_count}\n",
        encoding="utf-8",
    )
    return migrated_count


# ── Facade 函数：保留原有接口，内部委托给 AMAPersistence ──

def load_performance() -> Dict[str, Dict[str, Dict]]:
    return get_persistence().load_performance()


def save_performance(task_type: str, mode: str, stats: Dict) -> None:
    return get_persistence().save_performance(task_type, mode, stats)


def record_execution(
    session_id: Optional[str],
    task: str,
    task_type: str,
    complexity_score: float,
    mode_used: str,
    original_mode: Optional[str],
    success: bool,
    token_usage: int,
    time_taken: float,
    switched_modes: bool,
    switch_reason: Optional[str] = None,
    trace_id: Optional[str] = None,
    status: Optional[str] = None,
    error_category: Optional[str] = None,
    retries_attempted: int = 0,
    timeout_seconds: Optional[int] = None,
) -> None:
    return get_persistence().record_execution(
        session_id, task, task_type, complexity_score, mode_used, original_mode,
        success, token_usage, time_taken, switched_modes, switch_reason,
        trace_id, status, error_category, retries_attempted, timeout_seconds,
    )


def save_execution_transaction(
    task_type: str,
    mode: str,
    stats: Dict,
    session_id,
    task: str,
    complexity_score: float,
    mode_used: str,
    original_mode,
    success: bool,
    token_usage: int,
    time_taken: float,
    switched_modes: bool,
    switch_reason=None,
    trace_id=None,
    status=None,
    error_category=None,
    retries_attempted: int = 0,
    timeout_seconds=None,
) -> None:
    return get_persistence().save_execution_transaction(
        task_type, mode, stats, session_id, task, complexity_score, mode_used,
        original_mode, success, token_usage, time_taken, switched_modes, switch_reason,
        trace_id, status, error_category, retries_attempted, timeout_seconds,
    )


def get_execution_by_trace_id(trace_id: str) -> Optional[Dict[str, Any]]:
    """按 trace_id 查询执行记录"""
    return get_persistence().get_execution_by_trace_id(trace_id)


def get_stats(detail: bool = False, period: str = "all") -> Dict[str, Any]:
    return get_persistence().get_stats(detail=detail, period=period)


def save_snapshot(trace_id: str, round_num: int, state: Dict) -> None:
    """保存 shared_state 快照"""
    return get_persistence().save_snapshot(trace_id, round_num, state)


def load_snapshot(trace_id: str, round_num: int) -> Optional[Dict]:
    """加载 shared_state 快照"""
    return get_persistence().load_snapshot(trace_id, round_num)


def save_memory(task: str, result: str, task_type: str = "", mode: str = "", success: bool = True) -> None:
    """保存任务-结果对到记忆层"""
    return get_persistence().save_memory(task, result, task_type, mode, success)


def search_memory(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """基于关键词匹配检索历史任务结果（简化版，无需 embedding）"""
    return get_persistence().search_memory(query, limit)


# 便捷 facade，供 trajectory.py 直接使用
def save_trajectory(trajectory: Any) -> None:
    """保存轨迹记录"""
    return get_persistence().save_trajectory(trajectory)


def get_trajectory(trajectory_id: str) -> Optional[Dict[str, Any]]:
    """获取完整轨迹详情"""
    return get_persistence().get_trajectory(trajectory_id)


def query_trajectories(
    limit: int = 20,
    success_only: Optional[bool] = None,
    mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """查询轨迹记录"""
    return get_persistence().query_trajectories(limit, success_only, mode)


def update_grade(trajectory_id: str, score: float, feedback: str) -> None:
    """更新轨迹评分"""
    return get_persistence().update_grade(trajectory_id, score, feedback)


def get_trajectory_stats() -> Dict[str, Any]:
    """获取轨迹统计"""
    return get_persistence().get_trajectory_stats()


# ── Journal & Spec facade ────────────────────────────

def upsert_journal(
    task: str, task_type: str = "", journal_text: str = "",
    spec_ref: str = "", status: str = "in_progress",
    timeline_entry: Optional[Dict] = None,
) -> int:
    return get_persistence().upsert_journal(task, task_type, journal_text, spec_ref, status, timeline_entry)


def get_journal(task_hash: str) -> Optional[Dict[str, Any]]:
    return get_persistence().get_journal(task_hash)


def search_journals(task_type: str = "", limit: int = 10) -> List[Dict[str, Any]]:
    return get_persistence().search_journals(task_type, limit)


def upsert_spec(spec_key: str, spec_name: str, task_type: str, content: str, priority: int = 0) -> None:
    return get_persistence().upsert_spec(spec_key, spec_name, task_type, content, priority)


def get_spec(spec_key: str) -> Optional[Dict[str, Any]]:
    return get_persistence().get_spec(spec_key)


def list_specs(task_type: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    return get_persistence().list_specs(task_type, limit)


def build_spec_context(task_type: str = "", spec_keys: Optional[List[str]] = None) -> str:
    return get_persistence().build_spec_context(task_type, spec_keys)
