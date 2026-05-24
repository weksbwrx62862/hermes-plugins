from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

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
"""

_MIGRATE_SQL = [
    "ALTER TABLE ama_executions ADD COLUMN trace_id TEXT",
    "ALTER TABLE ama_executions ADD COLUMN status TEXT",
    "ALTER TABLE ama_executions ADD COLUMN error_category TEXT",
    "ALTER TABLE ama_executions ADD COLUMN retries_attempted INTEGER DEFAULT 0",
    "ALTER TABLE ama_executions ADD COLUMN timeout_seconds INTEGER",
]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_CREATE_TABLES)
    # 兼容迁移：添加新列（忽略已存在错误）
    for sql in _MIGRATE_SQL:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass
    return conn


def load_performance() -> Dict[str, Dict[str, Dict]]:
    conn = _get_conn()
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


def save_performance(task_type: str, mode: str, stats: Dict) -> None:
    conn = _get_conn()
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
    conn = _get_conn()
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
    conn = _get_conn()
    try:
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
        conn.commit()
    finally:
        conn.close()


def get_execution_by_trace_id(trace_id: str) -> Optional[Dict[str, Any]]:
    """按 trace_id 查询执行记录"""
    conn = _get_conn()
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


def get_stats(detail: bool = False, period: str = "all") -> Dict[str, Any]:
    conn = _get_conn()
    try:
        # ── 时间过滤 ──
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

        # ── 趋势数据：按天聚合，用于可视化 ──
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

        # ── 模式切换统计 ──
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


def save_snapshot(trace_id: str, round_num: int, state: Dict) -> None:
    """保存 shared_state 快照"""
    conn = _get_conn()
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


def load_snapshot(trace_id: str, round_num: int) -> Optional[Dict]:
    """加载 shared_state 快照"""
    conn = _get_conn()
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


def _task_hash(task: str) -> str:
    """计算任务描述的哈希值，用于快速查找"""
    return hashlib.md5(task.encode()).hexdigest()


def save_memory(task: str, result: str, task_type: str = "", mode: str = "", success: bool = True) -> None:
    """保存任务-结果对到记忆层"""
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO ama_memory (task_hash, task, result, task_type, mode, success)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (_task_hash(task), task, result, task_type, mode, int(success)),
        )
        conn.commit()
    finally:
        conn.close()


def search_memory(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """基于关键词匹配检索历史任务结果（简化版，无需 embedding）"""
    conn = _get_conn()
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
