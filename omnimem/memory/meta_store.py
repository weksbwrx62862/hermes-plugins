"""MetaStore — SQLite 元数据存储。

P0方案一：将 DrawerClosetStore 的元数据管理从文件系统迁移到 SQLite，
保留 Drawer 文件作为原始内容冷备份。

核心设计：
  1. 元数据（wing/room/type/summary 等）存 SQLite，利用 B-tree 索引加速查询
  2. 完整原文仍存 Drawer 文件，get() 需要时按需读取
  3. 提供 FTS5 全文搜索（若可用），回退到 LIKE
  4. 与 DrawerClosetStore 接口兼容，便于渐进式切换
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from omnimem.utils.migration import SchemaMigrator

# 异步 SQLite 支持（可选降级）
try:
    import aiosqlite
except Exception:  # pragma: no cover
    aiosqlite = None  # type: ignore[assignment]

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


class MetaStore:
    """SQLite 元数据存储引擎。

    表结构：
      memories      — 核心元数据表
      memories_fts  — 可选 FTS5 虚拟表（全文搜索）
    """

    _VALID_COLUMNS = {
        "memory_id", "wing", "hall", "room", "type", "confidence",
        "privacy", "stored_at", "summary", "content_preview",
        "drawer_path", "vc", "created_at", "conflicting_with", "conflict_type",
    }

    def __init__(self, db_dir: Path):
        self._db_dir = db_dir
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._db_dir / "meta_store.db"
        self._conn: sqlite3.Connection | None = None
        self._fts_enabled = False
        self._lock = threading.RLock()
        self._pending_writes = 0
        self._batch_size = 20
        self._init_db()

    @property
    def db_path(self) -> Path:
        """公开访问数据库文件路径。"""
        return self._db_path

    def _init_db(self) -> None:
        """初始化数据库表结构和索引。"""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        # ★ 修复3: 设置 row_factory 以支持 row.keys() 动态列名
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")

        # 核心元数据表
        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="memories",
            create_sql="""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    wing TEXT,
                    hall TEXT,
                    room TEXT,
                    type TEXT,
                    confidence INTEGER DEFAULT 3,
                    privacy TEXT DEFAULT 'personal',
                    stored_at TEXT,
                    summary TEXT,
                    content_preview TEXT,
                    drawer_path TEXT,
                    vc TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """,
            migrations=[],
        )

        # 向后兼容：旧表无 vc 列时自动迁移
        try:
            self._conn.execute("SELECT vc FROM memories LIMIT 1")
        except sqlite3.OperationalError:
            self._conn.execute("ALTER TABLE memories ADD COLUMN vc TEXT")
            logger.info("MetaStore migrated: added vc column")

        # ★ R28v2修复BUG-3：添加冲突信息列迁移
        for col in ("conflicting_with", "conflict_type"):
            try:
                self._conn.execute(f"SELECT {col} FROM memories LIMIT 1")
            except sqlite3.OperationalError:
                # 列名来自硬编码常量，非用户输入，安全使用 f-string
                self._conn.execute(f"ALTER TABLE memories ADD COLUMN {col} TEXT")
                logger.info("MetaStore migrated: added %s column", col)

        # 单列索引
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_wing ON memories(wing)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_privacy ON memories(privacy)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_stored_at ON memories(stored_at)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_room ON memories(room)")

        # 尝试创建 FTS5 虚拟表（全文搜索）
        try:
            self._conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id UNINDEXED,
                    summary,
                    content_preview,
                    content='memories',
                    content_rowid='rowid'
                )
            """)
            # 创建触发器保持 FTS 表同步
            self._conn.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, summary, content_preview)
                    VALUES (new.rowid, new.summary, new.content_preview);
                END
            """)
            self._conn.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, summary, content_preview)
                    VALUES ('delete', old.rowid, old.summary, old.content_preview);
                END
            """)
            self._conn.execute("""
                CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, summary, content_preview)
                    VALUES ('delete', old.rowid, old.summary, old.content_preview);
                    INSERT INTO memories_fts(rowid, summary, content_preview)
                    VALUES (new.rowid, new.summary, new.content_preview);
                END
            """)
            self._fts_enabled = True
            logger.debug(
                "MetaStore: FTS5 全文检索已启用 (日志级别已从 warning 降为 debug), db=%s",
                getattr(self, '_db_path', 'unknown'),
            )
        except Exception:
            logger.debug(
                "MetaStore: FTS5 不可用, 降级为 LIKE 搜索 (日志级别已从 warning 降为 debug)",
            )
            self._fts_enabled = False

        self._conn.commit()

    # ─── CRUD ─────────────────────────────────────────────────

    def add(self, memory_id: str, **fields: Any) -> None:
        """添加或替换一条元数据记录。

        采用批量提交策略：累积到 _batch_size 后再 commit，减少高频写入时的 fsync 开销。
        未提交的数据在 WAL 模式下对同一连接可见，不影响查询正确性。
        """
        if not self._conn:
            return
        with self._lock:
            cols = ["memory_id"] + [k for k in fields if k != "memory_id" and k in self._VALID_COLUMNS]
            vals = [memory_id] + [fields.get(k, "") for k in cols[1:]]
            placeholders = ",".join("?" * len(cols))
            try:
                _retry_db_op(
                    self._conn.execute,
                    f"INSERT OR REPLACE INTO memories ({','.join(cols)}) VALUES ({placeholders})",
                    vals,
                )
                self._pending_writes += 1
                if self._pending_writes >= self._batch_size:
                    _retry_db_op(self._conn.commit)
                    self._pending_writes = 0
            except Exception as e:
                logger.warning("MetaStore add failed for %s: %s", memory_id, e)

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """根据 ID 获取元数据。"""
        if not self._conn:
            return None
        try:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if row:
                return self._row_to_dict(row)
        except Exception as e:
            logger.warning("MetaStore get failed for %s: %s", memory_id, e)
        return None

    def update_privacy(self, memory_id: str, privacy: str, new_wing: str = "") -> bool:
        """更新隐私级别和可选 wing。"""
        if not self._conn:
            return False
        with self._lock:
            try:
                if new_wing:
                    _retry_db_op(
                        self._conn.execute,
                        "UPDATE memories SET privacy = ?, wing = ? WHERE memory_id = ?",
                        (privacy, new_wing, memory_id),
                    )
                else:
                    _retry_db_op(
                        self._conn.execute,
                        "UPDATE memories SET privacy = ? WHERE memory_id = ?",
                        (privacy, memory_id),
                    )
                _retry_db_op(self._conn.commit)
                return True
            except Exception as e:
                logger.warning("MetaStore update_privacy failed for %s: %s", memory_id, e)
                return False

    def update_field(self, memory_id: str, **fields: Any) -> bool:
        """更新指定字段。"""
        if not self._conn or not fields:
            return False
        with self._lock:
            try:
                safe_fields = {k: v for k, v in fields.items() if k in self._VALID_COLUMNS}
                if not safe_fields:
                    return False
                set_clause = ", ".join(f"{k} = ?" for k in safe_fields)
                values = list(safe_fields.values()) + [memory_id]
                _retry_db_op(
                    self._conn.execute,
                    f"UPDATE memories SET {set_clause} WHERE memory_id = ?",
                    values,
                )
                _retry_db_op(self._conn.commit)
                return True
            except Exception as e:
                logger.warning("MetaStore update_field failed for %s: %s", memory_id, e)
                return False

    def delete(self, memory_id: str) -> bool:
        """删除元数据记录。"""
        if not self._conn:
            return False
        with self._lock:
            try:
                self._conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
                self._conn.commit()
                return True
            except Exception as e:
                logger.warning("MetaStore delete failed for %s: %s", memory_id, e)
                return False

    # ─── 搜索 ─────────────────────────────────────────────────

    def search(
        self,
        wing: str = "",
        room: str = "",
        memory_type: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """按条件搜索元数据。利用 SQLite 索引加速。"""
        if not self._conn:
            return []
        conditions: list[str] = []
        params: list[Any] = []
        if memory_type:
            conditions.append("type = ?")
            params.append(memory_type)
        if wing:
            conditions.append("wing = ?")
            params.append(wing)
        if room:
            conditions.append("room = ?")
            params.append(room)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        try:
            rows = self._conn.execute(
                f"SELECT * FROM memories {where} ORDER BY stored_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.warning("MetaStore search failed: %s", e)
            raise

    def search_by_content(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """按内容关键词搜索。

        优先使用 FTS5（若可用），否则回退到 LIKE。
        """
        if not self._conn or not query:
            return []
        try:
            if self._fts_enabled:
                # FTS5 查询：转义双引号（用 "" 表示字面量双引号），然后用双引号包裹
                escaped = query.replace('"', '""')
                safe_query = f'"{escaped}"'
                try:
                    rows = self._conn.execute(
                        """SELECT m.* FROM memories_fts f
                           JOIN memories m ON m.rowid = f.rowid
                           WHERE memories_fts MATCH ?
                           ORDER BY rank
                           LIMIT ?""",
                        (safe_query, limit),
                    ).fetchall()
                    if rows:
                        return [self._row_to_dict(r) for r in rows]
                except Exception:
                    # FTS5 查询失败（特殊字符等），降级到 LIKE
                    logger.debug("FTS5 match failed, falling back to LIKE for query: %s", query[:50])
            # LIKE 查询（FTS5 不可用、失败、或返回空结果时的降级路径）
            escaped_query = query.replace("%", "\\%").replace("_", "\\_")
            q = f"%{escaped_query}%"
            rows = self._conn.execute(
                """SELECT * FROM memories
                   WHERE summary LIKE ? ESCAPE '\\' OR content_preview LIKE ? ESCAPE '\\'
                   ORDER BY stored_at DESC
                   LIMIT ?""",
                (q, q, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.warning("MetaStore search_by_content failed: %s", e)
            raise

    def get_all(self, limit: int = 5000) -> list[dict[str, Any]]:
        """获取所有元数据记录。"""
        if not self._conn:
            return []
        try:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY stored_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.error("MetaStore get_all failed: %s", e)
            raise

    def count(self) -> int:
        """返回记录总数，-1 表示数据库故障。"""
        if not self._conn:
            return -1
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            return row[0] if row else -1
        except Exception as e:
            logger.error("MetaStore count failed: %s", e)
            return -1

    def warm_up(self, entries: list[dict[str, Any]]) -> int:
        """批量预热：将现有条目导入 SQLite。

        使用 INSERT OR IGNORE 策略：已存在的记录不会被覆盖（保留 drawer_path 等字段），
        仅插入新记录。
        """
        if not self._conn:
            return 0
        added = 0
        with self._lock:
            try:
                for entry in entries:
                    mid = entry.get("memory_id", "")
                    if not mid:
                        continue
                    # 直接执行 SQL，避免调用 self.add() 导致重入死锁
                    fields = {
                        "wing": entry.get("wing", ""),
                        "hall": entry.get("hall", entry.get("type", "fact")),
                        "room": entry.get("room", ""),
                        "type": entry.get("type", "fact"),
                        "confidence": entry.get("confidence", 3),
                        "privacy": entry.get("privacy", "personal"),
                        "stored_at": entry.get("stored_at", ""),
                        "summary": entry.get("summary", ""),
                        "content_preview": entry.get("content", "")[:500],
                        "vc": entry.get("vc", ""),
                    }
                    # ★ BUGFIX: 包含 drawer_path，从 entry 或标准目录结构计算
                    drawer_path = entry.get("drawer_path", "")
                    if not drawer_path:
                        wing = entry.get("wing", "")
                        hall = entry.get("hall", entry.get("type", "fact"))
                        room = entry.get("room", "")
                        if wing and hall and room and mid:
                            drawer_path = str(
                                self._db_dir.parent / wing / hall / room / "drawer" / f"{mid}.md"
                            )
                    if drawer_path:
                        fields["drawer_path"] = drawer_path
                    cols = ["memory_id"] + [k for k in fields if k in self._VALID_COLUMNS]
                    vals = [mid] + [fields.get(k, "") for k in cols[1:]]
                    placeholders = ",".join("?" * len(cols))
                    # ★ BUGFIX: 使用 INSERT OR IGNORE 避免覆盖已有记录的 drawer_path
                    _retry_db_op(
                        self._conn.execute,
                        f"INSERT OR IGNORE INTO memories ({','.join(cols)}) VALUES ({placeholders})",
                        vals,
                    )
                    added += 1
                self._conn.commit()
                logger.info("MetaStore warmed up %d entries", added)
            except Exception as e:
                logger.warning("MetaStore warm_up failed: %s", e)
        return added

    def flush(self) -> None:
        """显式提交所有待写入的数据。"""
        if self._conn and self._pending_writes > 0:
            with self._lock:
                if self._pending_writes > 0:
                    self._conn.commit()
                    self._pending_writes = 0

    # ─── 内部方法 ─────────────────────────────────────────────

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """将 SQLite 行转为字典。★ 修复3: 使用动态列名，避免与表结构不同步。"""
        return dict(zip(row.keys(), tuple(row)))

    def sync_from_index(self, index_db_path: Path) -> tuple[int, int]:
        """从 index.db 同步到 meta_store.db，清理失联条目并补充缺失条目。

        Args:
            index_db_path: index.db 的文件路径

        Returns:
            (stale_count, missing_count) — 清理的失联条目数和补充的缺失条目数
        """
        import sqlite3 as _sql

        if not self._conn or not index_db_path.exists():
            return (0, 0)
        with self._lock:
            try:
                idx_db = _sql.connect(str(index_db_path), check_same_thread=False)
                # 获取两组 ID
                meta_rows = self._conn.execute(
                    "SELECT memory_id, wing, type, room, summary, confidence, privacy, stored_at, content_preview FROM memories"
                ).fetchall()
                idx_rows = idx_db.execute("SELECT memory_id FROM memory_index").fetchall()
                meta_ids = {r[0] for r in meta_rows}
                idx_ids = {r[0] for r in idx_rows}
                # 清理 index.db 中失联条目
                stale = idx_ids - meta_ids
                for mid in stale:
                    idx_db.execute("DELETE FROM memory_index WHERE memory_id = ?", (mid,))
                # 补充 index.db 中缺失条目
                missing = meta_ids - idx_ids
                if missing:
                    meta_map = {}
                    for r in meta_rows:
                        mid = r[0]
                        if mid in missing:
                            meta_map[mid] = {
                                'wing': r[1] or 'personal',
                                'hall': r[2] or 'facts',
                                'room': r[3] or 'default',
                                'summary': r[4] or '',
                                'type': r[2] or 'fact',
                                'confidence': r[5] or 3,
                                'privacy': r[6] or 'personal',
                                'stored_at': r[7] or '',
                                'content': r[8] or r[4] or '',
                            }
                    for mid, m in meta_map.items():
                        idx_db.execute(
                            "INSERT INTO memory_index (memory_id, wing, hall, room, summary, content, type, confidence, privacy, scope, stored_at, provenance) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (mid, m['wing'], m['hall'], m['room'], m['summary'], m['content'],
                             m['type'], m['confidence'], m['privacy'], m['wing'], m['stored_at'], '{}'))
                idx_db.commit()
                idx_db.close()
                return (len(stale), len(missing))
            except Exception as e:
                logger.warning("MetaStore sync_from_index failed: %s", e)
                return (0, 0)

    def close(self) -> None:
        """关闭数据库连接。"""
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None


class AsyncMetaStore:
    """基于 aiosqlite 的异步元数据存储，与 MetaStore 共享表结构。

    所有写操作默认批量提交（_batch_size=20），并通过 WAL 模式提升并发性能。
    """

    _VALID_COLUMNS = MetaStore._VALID_COLUMNS

    def __init__(self, db_dir: Path, db_name: str = "meta_store.db"):
        self._db_dir = db_dir
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._db_dir / db_name
        self._conn: Any = None
        self._fts_enabled = False
        self._pending_writes = 0
        self._batch_size = 20
        self._lock = asyncio.Lock()

    @property
    def db_path(self) -> Path:
        """公开访问数据库文件路径。"""
        return self._db_path

    async def _ensure_initialized(self) -> None:
        """异步初始化数据库连接、表结构、索引与 FTS5。"""
        if self._conn is not None:
            return
        if aiosqlite is None:
            raise RuntimeError("aiosqlite not installed")
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")

        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                wing TEXT,
                hall TEXT,
                room TEXT,
                type TEXT,
                confidence INTEGER DEFAULT 3,
                privacy TEXT DEFAULT 'personal',
                stored_at TEXT,
                summary TEXT,
                content_preview TEXT,
                drawer_path TEXT,
                vc TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        for col in ("vc", "conflicting_with", "conflict_type"):
            try:
                await self._conn.execute(f"SELECT {col} FROM memories LIMIT 1")
            except Exception:
                await self._conn.execute(f"ALTER TABLE memories ADD COLUMN {col} TEXT")
                logger.info("AsyncMetaStore migrated: added %s column", col)

        for col in ("type", "wing", "privacy", "stored_at", "room"):
            await self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_mem_{col} ON memories({col})"
            )

        try:
            await self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id UNINDEXED,
                    summary,
                    content_preview,
                    content='memories',
                    content_rowid='rowid'
                )
                """
            )
            await self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, summary, content_preview)
                    VALUES (new.rowid, new.summary, new.content_preview);
                END
                """
            )
            await self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, summary, content_preview)
                    VALUES ('delete', old.rowid, old.summary, old.content_preview);
                END
                """
            )
            await self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, summary, content_preview)
                    VALUES ('delete', old.rowid, old.summary, old.content_preview);
                    INSERT INTO memories_fts(rowid, summary, content_preview)
                    VALUES (new.rowid, new.summary, new.content_preview);
                END
                """
            )
            self._fts_enabled = True
            logger.debug(
                "AsyncMetaStore: FTS5 全文检索已启用 (日志级别已从 warning 降为 debug), db=%s",
                getattr(self, '_db_path', 'unknown'),
            )
        except Exception:
            logger.debug(
                "AsyncMetaStore: FTS5 不可用, 降级为 LIKE 搜索 (日志级别已从 warning 降为 debug)",
            )
            self._fts_enabled = False

        await self._conn.commit()

    async def add(self, memory_id: str, **fields: Any) -> None:
        """异步添加或替换一条元数据记录（批量提交）。"""
        await self._ensure_initialized()
        if self._conn is None:
            return
        async with self._lock:
            cols = ["memory_id"] + [
                k for k in fields if k != "memory_id" and k in self._VALID_COLUMNS
            ]
            vals = [memory_id] + [fields.get(k, "") for k in cols[1:]]
            placeholders = ",".join("?" * len(cols))
            try:
                await self._conn.execute(
                    f"INSERT OR REPLACE INTO memories ({','.join(cols)}) VALUES ({placeholders})",
                    vals,
                )
                self._pending_writes += 1
                if self._pending_writes >= self._batch_size:
                    await self._conn.commit()
                    self._pending_writes = 0
            except Exception as e:
                logger.warning("AsyncMetaStore add failed for %s: %s", memory_id, e)

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """异步根据 ID 获取元数据。"""
        await self._ensure_initialized()
        if self._conn is None:
            return None
        try:
            async with self._lock:
                cursor = await self._conn.execute(
                    "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
                )
                row = await cursor.fetchone()
            if row:
                return self._row_to_dict(row)
        except Exception as e:
            logger.warning("AsyncMetaStore get failed for %s: %s", memory_id, e)
        return None

    async def update_privacy(self, memory_id: str, privacy: str, new_wing: str = "") -> bool:
        """异步更新隐私级别和可选 wing。"""
        await self._ensure_initialized()
        if self._conn is None:
            return False
        async with self._lock:
            try:
                if new_wing:
                    await self._conn.execute(
                        "UPDATE memories SET privacy = ?, wing = ? WHERE memory_id = ?",
                        (privacy, new_wing, memory_id),
                    )
                else:
                    await self._conn.execute(
                        "UPDATE memories SET privacy = ? WHERE memory_id = ?",
                        (privacy, memory_id),
                    )
                await self._conn.commit()
                return True
            except Exception as e:
                logger.warning("AsyncMetaStore update_privacy failed for %s: %s", memory_id, e)
                return False

    async def update_field(self, memory_id: str, **fields: Any) -> bool:
        """异步更新指定字段。"""
        await self._ensure_initialized()
        if self._conn is None or not fields:
            return False
        async with self._lock:
            try:
                safe_fields = {k: v for k, v in fields.items() if k in self._VALID_COLUMNS}
                if not safe_fields:
                    return False
                set_clause = ", ".join(f"{k} = ?" for k in safe_fields)
                values = list(safe_fields.values()) + [memory_id]
                await self._conn.execute(
                    f"UPDATE memories SET {set_clause} WHERE memory_id = ?",
                    values,
                )
                await self._conn.commit()
                return True
            except Exception as e:
                logger.warning("AsyncMetaStore update_field failed for %s: %s", memory_id, e)
                return False

    async def delete(self, memory_id: str) -> bool:
        """异步删除元数据记录。"""
        await self._ensure_initialized()
        if self._conn is None:
            return False
        async with self._lock:
            try:
                await self._conn.execute(
                    "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
                )
                await self._conn.commit()
                return True
            except Exception as e:
                logger.warning("AsyncMetaStore delete failed for %s: %s", memory_id, e)
                return False

    async def search(
        self,
        wing: str = "",
        room: str = "",
        memory_type: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """异步按条件搜索元数据。"""
        await self._ensure_initialized()
        if self._conn is None:
            return []
        conditions: list[str] = []
        params: list[Any] = []
        if memory_type:
            conditions.append("type = ?")
            params.append(memory_type)
        if wing:
            conditions.append("wing = ?")
            params.append(wing)
        if room:
            conditions.append("room = ?")
            params.append(room)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        try:
            async with self._lock:
                cursor = await self._conn.execute(
                    f"SELECT * FROM memories {where} ORDER BY stored_at DESC LIMIT ?",
                    params + [limit],
                )
                rows = await cursor.fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.warning("AsyncMetaStore search failed: %s", e)
            return []

    async def search_by_content(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """异步按内容关键词搜索（FTS5 / LIKE 降级）。"""
        await self._ensure_initialized()
        if self._conn is None or not query:
            return []
        try:
            if self._fts_enabled:
                escaped = query.replace('"', '""')
                safe_query = f'"{escaped}"'
                try:
                    async with self._lock:
                        cursor = await self._conn.execute(
                            """SELECT m.* FROM memories_fts f
                               JOIN memories m ON m.rowid = f.rowid
                               WHERE memories_fts MATCH ?
                               ORDER BY rank
                               LIMIT ?""",
                            (safe_query, limit),
                        )
                        rows = await cursor.fetchall()
                    if rows:
                        return [self._row_to_dict(r) for r in rows]
                except Exception:
                    logger.debug("FTS5 match failed, falling back to LIKE for query: %s", query[:50])
            escaped_query = query.replace("%", "\\%").replace("_", "\\_")
            q = f"%{escaped_query}%"
            async with self._lock:
                cursor = await self._conn.execute(
                    """SELECT * FROM memories
                       WHERE summary LIKE ? ESCAPE '\\' OR content_preview LIKE ? ESCAPE '\\'
                       ORDER BY stored_at DESC
                       LIMIT ?""",
                    (q, q, limit),
                )
                rows = await cursor.fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.warning("AsyncMetaStore search_by_content failed: %s", e)
            return []

    async def get_all(self, limit: int = 5000) -> list[dict[str, Any]]:
        """异步获取所有元数据记录。"""
        await self._ensure_initialized()
        if self._conn is None:
            return []
        try:
            async with self._lock:
                cursor = await self._conn.execute(
                    "SELECT * FROM memories ORDER BY stored_at DESC LIMIT ?", (limit,)
                )
                rows = await cursor.fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.error("AsyncMetaStore get_all failed: %s", e)
            return []

    async def count(self) -> int:
        """异步返回记录总数。"""
        await self._ensure_initialized()
        if self._conn is None:
            return -1
        try:
            async with self._lock:
                cursor = await self._conn.execute("SELECT COUNT(*) FROM memories")
                row = await cursor.fetchone()
            return row[0] if row else -1
        except Exception as e:
            logger.error("AsyncMetaStore count failed: %s", e)
            return -1

    async def warm_up(self, entries: list[dict[str, Any]]) -> int:
        """异步批量预热：将现有条目导入 SQLite。"""
        await self._ensure_initialized()
        if self._conn is None:
            return 0
        added = 0
        async with self._lock:
            try:
                for entry in entries:
                    mid = entry.get("memory_id", "")
                    if not mid:
                        continue
                    fields = {
                        "wing": entry.get("wing", ""),
                        "hall": entry.get("hall", entry.get("type", "fact")),
                        "room": entry.get("room", ""),
                        "type": entry.get("type", "fact"),
                        "confidence": entry.get("confidence", 3),
                        "privacy": entry.get("privacy", "personal"),
                        "stored_at": entry.get("stored_at", ""),
                        "summary": entry.get("summary", ""),
                        "content_preview": entry.get("content", "")[:500],
                        "vc": entry.get("vc", ""),
                    }
                    drawer_path = entry.get("drawer_path", "")
                    if not drawer_path:
                        wing = entry.get("wing", "")
                        hall = entry.get("hall", entry.get("type", "fact"))
                        room = entry.get("room", "")
                        if wing and hall and room and mid:
                            drawer_path = str(
                                self._db_dir.parent / wing / hall / room / "drawer" / f"{mid}.md"
                            )
                    if drawer_path:
                        fields["drawer_path"] = drawer_path
                    cols = ["memory_id"] + [k for k in fields if k in self._VALID_COLUMNS]
                    vals = [mid] + [fields.get(k, "") for k in cols[1:]]
                    placeholders = ",".join("?" * len(cols))
                    await self._conn.execute(
                        f"INSERT OR IGNORE INTO memories ({','.join(cols)}) VALUES ({placeholders})",
                        vals,
                    )
                    added += 1
                await self._conn.commit()
                logger.info("AsyncMetaStore warmed up %d entries", added)
            except Exception as e:
                logger.warning("AsyncMetaStore warm_up failed: %s", e)
        return added

    async def flush(self) -> None:
        """异步显式提交所有待写入的数据。"""
        if self._conn is None or self._pending_writes == 0:
            return
        async with self._lock:
            if self._pending_writes > 0:
                await self._conn.commit()
                self._pending_writes = 0

    async def close(self) -> None:
        """异步关闭数据库连接。"""
        await self.flush()
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        """将 aiosqlite 行转为字典。"""
        return dict(zip(row.keys(), tuple(row)))
