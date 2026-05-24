"""ProvenanceTracker — 溯源追踪（SQLite 持久化）。

记录每条记忆的来源信息：
  - source: 来源（session_id / tool_call / auto_detect 等）
  - method: 写入方式（手动/自动/工具）
  - timestamp: 写入时间
  - chain: 记忆链（记忆的演化路径）

持久化到 SQLite，进程重启后可恢复。
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)


class ProvenanceTracker:
    """溯源追踪引擎，SQLite 持久化。"""

    def __init__(self, data_dir: Path | None = None):
        self._provenance: dict[str, dict[str, Any]] = {}
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

        if data_dir:
            self._init_db(data_dir)

    def _init_db(self, data_dir: Path) -> None:
        """初始化溯源数据库。"""
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "provenance.db"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="provenance",
            create_sql="""
                CREATE TABLE IF NOT EXISTS provenance (
                    memory_id TEXT PRIMARY KEY,
                    source TEXT,
                    method TEXT,
                    timestamp TEXT,
                    content_hash TEXT,
                    parent_id TEXT,
                    metadata TEXT
                )
            """,
            migrations=[],
        )
        self._conn.commit()
        # 从数据库恢复内存索引
        self._restore()

    def _restore(self) -> None:
        """从数据库恢复溯源数据到内存。"""
        if not self._conn:
            return
        try:
            rows = self._conn.execute(
                "SELECT memory_id, source, method, timestamp, content_hash, parent_id, metadata FROM provenance"
            ).fetchall()
            for row in rows:
                memory_id, source, method, timestamp, content_hash, parent_id, metadata_str = row
                prov = {
                    "source": source or "",
                    "method": method or "",
                    "timestamp": timestamp or "",
                    "content_hash": content_hash or "",
                }
                if parent_id:
                    prov["parent_id"] = parent_id
                if metadata_str:
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        prov["metadata"] = json.loads(metadata_str)
                self._provenance[memory_id] = prov
            logger.info("ProvenanceTracker: restored %d entries from disk", len(self._provenance))
        except Exception as e:
            logger.warning("Provenance restore failed: %s", e)

    def track(
        self,
        content: str,
        source: str = "",
        method: str = "auto_detect",
    ) -> dict[str, Any]:
        """记录记忆的溯源信息。

        Args:
            content: 记忆内容
            source: 来源标识
            method: 写入方式

        Returns:
            溯源信息字典
        """
        now = datetime.now(timezone.utc).isoformat()
        provenance = {
            "source": source,
            "method": method,
            "timestamp": now,
            "content_hash": self._hash(content),
        }
        return provenance

    def lookup(self, memory_id: str) -> dict[str, Any]:
        """查询记忆的溯源信息。"""
        return self._provenance.get(
            memory_id,
            {
                "status": "not_found",
                "memory_id": memory_id,
            },
        )

    def record(self, memory_id: str, provenance: dict[str, Any]) -> None:
        """记录溯源信息到索引，并持久化。"""
        self._provenance[memory_id] = provenance
        self._persist(memory_id, provenance)

    def get_chain(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆的演化链。"""
        chain = []
        current_id = memory_id
        visited = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            prov = self._provenance.get(current_id)
            if prov:
                chain.append(prov)
                current_id = prov.get("parent_id") or ""
            else:
                break

        return chain

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ─── 持久化 ───────────────────────────────────────────────

    def _persist(self, memory_id: str, provenance: dict[str, Any]) -> None:
        """持久化溯源记录到 SQLite。"""
        if not self._conn:
            return
        with self._lock:
            try:
                metadata = provenance.get("metadata")
                metadata_str = json.dumps(metadata, ensure_ascii=False) if metadata else None
                self._conn.execute(
                    """INSERT OR REPLACE INTO provenance
                       (memory_id, source, method, timestamp, content_hash, parent_id, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        memory_id,
                        provenance.get("source", ""),
                        provenance.get("method", ""),
                        provenance.get("timestamp", ""),
                        provenance.get("content_hash", ""),
                        provenance.get("parent_id"),
                        metadata_str,
                    ),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning("Provenance persist failed: %s", e)

    @staticmethod
    def _hash(content: str) -> str:
        """计算内容哈希。"""
        import hashlib

        return hashlib.sha256(content.encode()).hexdigest()[:16]
