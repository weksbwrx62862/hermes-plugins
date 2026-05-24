"""SQLite 数据库迁移框架。

提供声明式的表结构迁移能力：
  - 自动维护 _schema_versions 版本追踪表
  - 幂等迁移：重复调用不会重复执行已应用的迁移
  - 事务保护：整个迁移过程在单事务中完成
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SchemaMigrator:
    """SQLite 表结构迁移器。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._ensure_versions_table()

    def _ensure_versions_table(self) -> None:
        """创建 _schema_versions 版本追踪表。"""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _schema_versions (
                table_name  TEXT PRIMARY KEY,
                version     INTEGER DEFAULT 0,
                migrated_at TEXT
            )
            """
        )
        self._conn.commit()

    def get_version(self, table_name: str) -> int:
        """查询指定表的当前迁移版本号，不存在则返回 0。"""
        row = self._conn.execute(
            "SELECT version FROM _schema_versions WHERE table_name = ?",
            (table_name,),
        ).fetchone()
        return row[0] if row else 0

    def set_version(self, table_name: str, version: int) -> None:
        """更新指定表的迁移版本号。"""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO _schema_versions (table_name, version, migrated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(table_name) DO UPDATE SET
                version = excluded.version,
                migrated_at = excluded.migrated_at
            """,
            (table_name, version, now),
        )
        self._conn.commit()

    def migrate(
        self,
        table_name: str,
        create_sql: str,
        migrations: list[tuple[int, str]],
    ) -> None:
        """执行表结构迁移。

        Args:
            table_name: 目标表名，用于版本追踪。
            create_sql: 建表语句（应包含 CREATE TABLE IF NOT EXISTS）。
            migrations: 迁移列表，每项为 (版本号, ALTER SQL)，
                        仅执行版本号大于当前版本的迁移。
        """
        current_version = self.get_version(table_name)

        pending = [(v, sql) for v, sql in migrations if v > current_version]
        pending.sort(key=lambda item: item[0])

        try:
            with self._conn:
                self._conn.execute(create_sql)
                for version, sql in pending:
                    self._conn.execute(sql)
                    now = datetime.now(timezone.utc).isoformat()
                    self._conn.execute(
                        """
                        INSERT INTO _schema_versions (table_name, version, migrated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(table_name) DO UPDATE SET
                            version = excluded.version,
                            migrated_at = excluded.migrated_at
                        """,
                        (table_name, version, now),
                    )
        except sqlite3.Error:
            logger.exception("迁移失败: table=%s", table_name)
            raise

        if pending:
            logger.info(
                "迁移完成: table=%s, %d→%d",
                table_name,
                current_version,
                pending[-1][0],
            )
