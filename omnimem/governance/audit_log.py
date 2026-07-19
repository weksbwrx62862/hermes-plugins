import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)

class AuditLogger:
    def __init__(self, governance_dir: Path, max_rows: int = 100000):
        self._db_path = governance_dir / "audit_log.db"
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._max_rows = max_rows
        self._ensure_table()

    def _ensure_table(self) -> None:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        migrator = SchemaMigrator(conn)
        migrator.migrate(
            table_name="audit_log",
            create_sql="""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    operation TEXT NOT NULL,
                    memory_id TEXT,
                    details TEXT,
                    result TEXT,
                    instance_id TEXT
                )
            """,
            migrations=[],
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_operation ON audit_log(operation)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_memory_id ON audit_log(memory_id)")
        conn.commit()
        conn.close()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")

    def log(self, operation: str, memory_id: str | None = None, details: dict | None = None, result: str = "success", instance_id: str | None = None) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO audit_log (timestamp, operation, memory_id, details, result, instance_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (time.time(), operation, memory_id, json.dumps(details, ensure_ascii=False) if details else None, result, instance_id),
                )
                self._conn.commit()
                self._rotate_if_needed()
            except Exception as e:
                logger.warning("Audit log write failed: %s", e)

    def _rotate_if_needed(self):
        try:
            count = self._conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            if count > self._max_rows:
                self._conn.execute("DELETE FROM audit_log WHERE rowid IN (SELECT rowid FROM audit_log ORDER BY rowid ASC LIMIT ?)", (count - self._max_rows,))
                self._conn.commit()
        except Exception as e:
            logger.warning("AuditLog _rotate_if_needed failed: %s", e)

    def query(self, operation: str | None = None, memory_id: str | None = None, from_time: float | None = None, to_time: float | None = None, limit: int = 100) -> list[dict]:
        conditions = []
        params: list[Any] = []
        if operation:
            conditions.append("operation = ?")
            params.append(operation)
        if memory_id:
            conditions.append("memory_id = ?")
            params.append(memory_id)
        if from_time:
            conditions.append("timestamp >= ?")
            params.append(from_time)
        if to_time:
            conditions.append("timestamp <= ?")
            params.append(to_time)
        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        with self._lock:
            cursor = self._conn.execute(
                f"SELECT id, timestamp, operation, memory_id, details, result, instance_id FROM audit_log WHERE {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            )
            rows = cursor.fetchall()
        return [
            {"id": r[0], "timestamp": r[1], "operation": r[2], "memory_id": r[3], "details": json.loads(r[4]) if r[4] else None, "result": r[5], "instance_id": r[6]}
            for r in rows
        ]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
