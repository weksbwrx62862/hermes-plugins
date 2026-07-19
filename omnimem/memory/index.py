"""ThreeLevelIndex — 三层索引 L0/L1/L2。

参考 OpenViking 的三层索引设计：
  - L0 (目录索引): Wing/Hall/Room 结构索引，最小化加载
  - L1 (摘要索引): Closet 摘要，中等粒度
  - L2 (全文索引): Drawer 原文，最大精度

索引存储在 SQLite 中，支持快速查找和范围查询。
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)

_DB_RETRY_COUNT = 3
_DB_RETRY_DELAY = 0.1


def _retry_db_op(fn, *args, **kwargs):
    """★ P2修复Minor-4：SQLite 操作重试，解决并发锁超时问题。"""
    for attempt in range(_DB_RETRY_COUNT):
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < _DB_RETRY_COUNT - 1:
                import time
                time.sleep(_DB_RETRY_DELAY * (attempt + 1))
                continue
            raise


class ThreeLevelIndex:
    """三层索引 L0/L1/L2。

    批量提交优化：add() 不立即 commit，攒到阈值或显式 flush() 时统一提交，
    减少磁盘 fsync 次数。
    """

    _BATCH_THRESHOLD = 5  # 每 5 次写入 commit 一次

    def __init__(self, index_dir: Path):
        self._index_dir = index_dir
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._index_dir / "index.db"
        self._conn: sqlite3.Connection | None = None
        self._pending_writes = 0
        self._init_db()

    @property
    def db_path(self) -> Path:
        """公开访问数据库文件路径。"""
        return self._db_path

    def _init_db(self) -> None:
        """初始化 SQLite 数据库。"""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="memory_index",
            create_sql="""
                CREATE TABLE IF NOT EXISTS memory_index (
                    memory_id TEXT PRIMARY KEY,
                    wing TEXT NOT NULL,
                    hall TEXT NOT NULL,
                    room TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT,
                    type TEXT NOT NULL,
                    confidence INTEGER DEFAULT 3,
                    privacy TEXT DEFAULT 'personal',
                    scope TEXT DEFAULT 'personal',
                    stored_at TEXT,
                    provenance TEXT,
                    metadata TEXT
                )
            """,
            migrations=[],
        )

        # ★ 旧 schema 迁移：补全缺失的列
        _migrate_columns = [
            ("wing", "TEXT NOT NULL DEFAULT ''"),
            ("hall", "TEXT NOT NULL DEFAULT ''"),
            ("room", "TEXT NOT NULL DEFAULT ''"),
            ("summary", "TEXT"),
            ("confidence", "INTEGER DEFAULT 3"),
            ("privacy", "TEXT DEFAULT 'personal'"),
            ("scope", "TEXT DEFAULT 'personal'"),
            ("stored_at", "TEXT"),
            ("provenance", "TEXT"),
            ("metadata", "TEXT"),
            ("conflicting_with", "TEXT"),
            ("conflict_type", "TEXT"),
            ("is_updated", "INTEGER DEFAULT 0"),
            ("is_superseded", "INTEGER DEFAULT 0"),
        ]
        for col_name, col_def in _migrate_columns:
            try:
                self._conn.execute(f"SELECT {col_name} FROM memory_index LIMIT 1")
            except sqlite3.OperationalError:
                # 列名来自硬编码常量，非用户输入，安全使用 f-string
                self._conn.execute(f"ALTER TABLE memory_index ADD COLUMN {col_name} {col_def}")
                logger.info("Index migrated: added %s column", col_name)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_wing ON memory_index(wing)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_type ON memory_index(type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_stored_at ON memory_index(stored_at)
        """)
        # FTS5 fulltext index (replaces LIKE '%keyword%' full-scan in search_l2)
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_index_fts USING fts5(content)
        """)
        # ★ 修复根因 A:改用 DELETE FROM 替代 FTS5 'delete' 特殊命令
        #   'delete' 命令在 contentful FTS5 表上不可用(仅 external content/contentless 表可用)
        #   用标准 DELETE FROM 替代,且先 DROP 旧触发器(IF NOT EXISTS 不会替换已存在的)
        self._conn.execute("DROP TRIGGER IF EXISTS memory_index_fts_ai")
        self._conn.execute("DROP TRIGGER IF EXISTS memory_index_fts_ad")
        self._conn.execute("DROP TRIGGER IF EXISTS memory_index_fts_au")
        self._conn.execute("""
            CREATE TRIGGER memory_index_fts_ai AFTER INSERT ON memory_index BEGIN
                INSERT INTO memory_index_fts(rowid, content) VALUES (new.rowid, new.content);
            END
        """)
        self._conn.execute("""
            CREATE TRIGGER memory_index_fts_ad AFTER DELETE ON memory_index BEGIN
                DELETE FROM memory_index_fts WHERE rowid = old.rowid;
            END
        """)
        self._conn.execute("""
            CREATE TRIGGER memory_index_fts_au AFTER UPDATE ON memory_index BEGIN
                DELETE FROM memory_index_fts WHERE rowid = old.rowid;
                INSERT INTO memory_index_fts(rowid, content) VALUES (new.rowid, new.content);
            END
        """)

        self._conn.commit()

    def _maybe_commit(self) -> None:
        """检查待写入数是否达到阈值，达到则提交事务。"""
        assert self._conn is not None
        self._pending_writes += 1
        if self._pending_writes >= self._BATCH_THRESHOLD:
            _retry_db_op(self._conn.commit)
            self._pending_writes = 0

    def add(
        self,
        memory_id: str,
        wing: str,
        hall: str,
        room: str,
        content: str,
        summary: str = "",
        type: str = "fact",
        confidence: int = 3,
        privacy: str = "personal",
        scope: str = "personal",
        stored_at: str = "",
        provenance: str = "",
        metadata: str = "",
    ) -> None:
        """添加一条索引记录。"""
        assert self._conn is not None
        if not stored_at:
            stored_at = datetime.now().isoformat()
        try:
            _retry_db_op(
                self._conn.execute,
                """INSERT OR REPLACE INTO memory_index
                   (memory_id, wing, hall, room, content, summary, type,
                    confidence, privacy, scope, stored_at, provenance, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory_id,
                    wing,
                    hall,
                    room,
                    content,
                    summary,
                    type,
                    confidence,
                    privacy,
                    scope,
                    stored_at,
                    provenance,
                    metadata,
                ),
            )
            self._maybe_commit()
        except Exception as e:
            logger.warning("Index add failed for %s: %s", memory_id, e)

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """根据 ID 获取索引记录。"""
        assert self._conn is not None
        try:
            row = self._conn.execute(
                "SELECT * FROM memory_index WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row:
                return self._row_to_dict(row)
        except Exception as e:
            logger.warning("Index get failed for %s: %s", memory_id, e)
        return None

    def delete(self, memory_id: str) -> bool:
        """从索引中删除记录。"""
        assert self._conn is not None
        try:
            self._conn.execute(
                "DELETE FROM memory_index WHERE memory_id = ?",
                (memory_id,),
            )
            self._maybe_commit()
            return True
        except Exception as e:
            logger.warning("Index delete failed for %s: %s", memory_id, e)
            return False

    def search_l0(self, wing: str = "", hall: str = "") -> list[str]:
        """L0 目录索引：返回匹配的 Room 列表。"""
        assert self._conn is not None
        query = "SELECT DISTINCT room FROM memory_index WHERE 1=1"
        params = []
        if wing:
            query += " AND wing = ?"
            params.append(wing)
        if hall:
            query += " AND hall = ?"
            params.append(hall)
        try:
            rows = self._conn.execute(query, params).fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            logger.warning("L0 search failed: %s", e)
            raise

    def search_by_directory(
        self,
        wing: str = "",
        hall: str = "",
        room: str = "",
    ) -> list[dict[str, Any]]:
        """按目录结构查询索引条目。

        内化 OpenViking 的目录定位能力：
        通过 Wing/Hall/Room 三级目录缩小搜索空间，
        返回目录内所有条目的 memory_id 和摘要。

        Args:
            wing: Wing 名称（personal/team/public）
            hall: Hall 名称（facts/preferences/...）
            room: Room 名称（话题）

        Returns:
            匹配的索引条目列表
        """
        assert self._conn is not None
        query = "SELECT memory_id, wing, hall, room, summary, type, confidence FROM memory_index WHERE 1=1"
        params = []
        if wing:
            query += " AND wing = ?"
            params.append(wing)
        if hall:
            query += " AND hall = ?"
            params.append(hall)
        if room:
            query += " AND room = ?"
            params.append(room)
        try:
            rows = self._conn.execute(query, params).fetchall()
            return [
                {
                    "memory_id": r[0],
                    "wing": r[1],
                    "hall": r[2],
                    "room": r[3],
                    "summary": r[4],
                    "type": r[5],
                    "confidence": r[6],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning("Directory search failed: %s", e)
            raise

    def search_l1(self, wing: str = "", type: str = "", limit: int = 50) -> list[dict[str, Any]]:
        """L1 摘要索引：返回摘要记录（含 content 用于 warm_up）。"""
        assert self._conn is not None
        query = "SELECT memory_id, wing, hall, room, summary, type, confidence, privacy, stored_at, content, conflicting_with, conflict_type FROM memory_index WHERE 1=1"
        params = []
        if wing:
            query += " AND wing = ?"
            params.append(wing)
        if type:
            query += " AND type = ?"
            params.append(type)
        query += " ORDER BY stored_at DESC LIMIT ?"
        params.append(str(limit))
        try:
            rows = self._conn.execute(query, params).fetchall()
            return [
                {
                    "memory_id": r[0],
                    "wing": r[1],
                    "hall": r[2],
                    "room": r[3],
                    "summary": r[4],
                    "type": r[5],
                    "confidence": r[6],
                    "privacy": r[7],
                    "stored_at": r[8],
                    "content": r[9] if len(r) > 9 else "",
                    "conflicting_with": r[10] if len(r) > 10 else "",
                    "conflict_type": r[11] if len(r) > 11 else "",
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning("L1 search failed: %s", e)
            raise

    def search_l2(
        self,
        keyword: str = "",
        wing: str = "",
        type: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """L2 全文索引：返回完整记录（FTS5 优先，LIKE 回退）。"""
        assert self._conn is not None
        use_fts5 = False
        fts_keyword = ""
        if keyword:
            # ★ 修复根因 B:FTS5 默认 unicode61 分词器把中文当单个 token,
            #   双引号短语查询无法匹配短关键字,中文强制走 LIKE 回退
            has_non_ascii = any(ord(c) > 127 for c in keyword) if keyword else False
            if has_non_ascii:
                use_fts5 = False
            else:
                # 检测 FTS5 表是否可用
                try:
                    self._conn.execute(
                        "SELECT 1 FROM memory_index_fts LIMIT 0"
                    )
                    use_fts5 = True
                except sqlite3.OperationalError:
                    pass

            if use_fts5:
                # FTS5: 转义特殊字符并构造 MATCH 查询
                fts_keyword = self._escape_fts5(keyword)
                base_query = """
                    SELECT memory_index.* FROM memory_index
                    JOIN memory_index_fts ON memory_index.rowid = memory_index_fts.rowid
                    WHERE memory_index_fts MATCH ?
                """
                params: list[Any] = [fts_keyword]
                if wing:
                    base_query += " AND memory_index.wing = ?"
                    params.append(wing)
                if type:
                    base_query += " AND memory_index.type = ?"
                    params.append(type)
                base_query += " ORDER BY memory_index.stored_at DESC LIMIT ?"
                params.append(limit)
            else:
                # LIKE 回退（旧 schema 或 FTS5 不可用）
                escaped = keyword.replace("%", "\\%").replace("_", "\\_")
                base_query = (
                    "SELECT * FROM memory_index WHERE content LIKE ? ESCAPE '\\'"
                )
                params = [f"%{escaped}%"]
                if wing:
                    base_query += " AND wing = ?"
                    params.append(wing)
                if type:
                    base_query += " AND type = ?"
                    params.append(type)
                base_query += " ORDER BY stored_at DESC LIMIT ?"
                params.append(limit)
        else:
            base_query = "SELECT * FROM memory_index WHERE 1=1"
            params = []
            if wing:
                base_query += " AND wing = ?"
                params.append(wing)
            if type:
                base_query += " AND type = ?"
                params.append(type)
            base_query += " ORDER BY stored_at DESC LIMIT ?"
            params.append(limit)

        try:
            rows = self._conn.execute(base_query, params).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.warning("L2 search failed: %s", e)
            raise

    @staticmethod
    def _escape_fts5(keyword: str) -> str:
        """转义 FTS5 特殊字符，构造安全的 MATCH 查询。

        FTS5 中需要转义的特殊字符：*  \"  -  (  ) 以及前后缀引号。
        简单关键字直接包裹在双引号中作为短语查询。
        """
        # 简单处理：用双引号包裹作为短语匹配
        # 先移除已有的双引号，再包裹
        clean = keyword.replace('"', '')
        return f'"{clean}"'

    def search_all_for_retrieval(self, limit: int = 1000) -> list[dict[str, Any]]:
        """获取所有记录（用于检索引擎全量索引）。"""
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                "SELECT * FROM memory_index ORDER BY stored_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.warning("Full index scan failed: %s", e)
            raise

    def update_privacy(self, memory_id: str, privacy: str) -> bool:
        """更新隐私级别。"""
        assert self._conn is not None
        try:
            self._conn.execute(
                "UPDATE memory_index SET privacy = ? WHERE memory_id = ?",
                (privacy, memory_id),
            )
            self._maybe_commit()
            return True
        except Exception as e:
            logger.warning("Privacy update failed: %s", e)
            return False

    def update_field(self, memory_id: str, immediate: bool = False, **fields: Any) -> bool:
        """更新索引中的指定字段。

        Args:
            memory_id: 记忆 ID
            immediate: 为 True 时直接 commit 而非走 _maybe_commit 批处理，
                       适用于 governance 等需要跨组件一致性的场景
        """
        assert self._conn is not None
        if not fields:
            return False
        try:
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            values = list(fields.values()) + [memory_id]
            _retry_db_op(
                self._conn.execute,
                f"UPDATE memory_index SET {set_clause} WHERE memory_id = ?",
                values,
            )
            if immediate:
                _retry_db_op(self._conn.commit)
                self._pending_writes = 0
            else:
                self._maybe_commit()
            return True
        except Exception as e:
            logger.warning("Field update failed: %s", e)
            return False

    def remove(self, memory_id: str) -> bool:
        """删除索引记录。"""
        assert self._conn is not None
        try:
            self._conn.execute(
                "DELETE FROM memory_index WHERE memory_id = ?",
                (memory_id,),
            )
            self._maybe_commit()
            return True
        except Exception as e:
            logger.warning("Index remove failed: %s", e)
            return False

    def close(self) -> None:
        """关闭数据库连接。"""
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None

    def flush(self) -> None:
        """显式提交所有待写入。"""
        if self._conn and self._pending_writes > 0:
            try:
                _retry_db_op(self._conn.commit)
                self._pending_writes = 0
            except Exception as e:
                logger.warning("Index flush failed: %s", e)

    def _row_to_dict(self, row: tuple[Any, ...]) -> dict[str, Any]:
        """将数据库行转为字典。"""
        keys = [
            "memory_id",
            "wing",
            "hall",
            "room",
            "content",
            "summary",
            "type",
            "confidence",
            "privacy",
            "scope",
            "stored_at",
            "provenance",
            "metadata",
            "conflicting_with",
            "conflict_type",
        ]
        result = {}
        for i, key in enumerate(keys):
            if i < len(row):
                val = row[i]
                if key == "provenance" and val:
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        val = json.loads(val)
                if key == "metadata" and val:
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        val = json.loads(val)
                result[key] = val
        return result
