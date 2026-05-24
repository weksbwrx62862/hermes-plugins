"""TemporalKnowledgeGraph — 时序知识图谱。

对标 Zep/Graphiti 的核心时序推理能力：
  - TemporalTriple: 扩展三元组，增加 valid_at/invalid_at/superseded_by 时序字段
  - 时点查询: query_at_time() 支持历史快照
  - 矛盾检测: detect_contradiction() 自动发现语义冲突
  - 实体时间线: get_timeline() 追踪实体关系演变

与 deep/knowledge_graph.py 的关系:
  - 独立存储（temporal_kg.db），不修改原有 triples 表
  - 通过 add_triple_from_kg() 方法从 KnowledgeGraph 同步三元组
  - recall 中作为补充检索通道，不替代原有图谱检索
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)


@dataclass
class TemporalTriple:
    """时序三元组数据类。

    扩展标准 (subject, predicate, object) 三元组，增加时序有效性字段，
    对标 Zep 的 Episode 和 Graphiti 的 TemporalEdge。
    """

    id: str = ""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    valid_at: str = ""
    invalid_at: str | None = None
    superseded_by: str | None = None
    source_memory_id: str = ""
    confidence: int = 3
    created_at: str = ""

    def is_valid_at(self, at_time: str) -> bool:
        """判断三元组在指定时间点是否有效。"""
        if not self.valid_at or self.valid_at > at_time:
            return False
        if self.invalid_at and self.invalid_at <= at_time:
            return False
        return True

    def is_current(self) -> bool:
        """判断三元组当前是否有效（invalid_at 为空）。"""
        return self.invalid_at is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "superseded_by": self.superseded_by,
            "source_memory_id": self.source_memory_id,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }


class TemporalKnowledgeGraph:
    """时序知识图谱 — SQLite 存储，支持时点查询和矛盾检测。

    参考 governance/forgetting.py 的 SQLite 模式：
      - WAL 日志模式
      - SchemaMigrator 迁移框架
      - 批量提交优化
    """

    _BATCH_THRESHOLD = 5

    def __init__(self, data_dir: Path, config: Any = None):
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "temporal_kg.db"
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._pending_writes = 0
        self._init_db()

    def _init_db(self) -> None:
        """初始化时序知识图谱数据库。"""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")

        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="temporal_triples",
            create_sql="""
                CREATE TABLE IF NOT EXISTS temporal_triples (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    valid_at TEXT NOT NULL,
                    invalid_at TEXT,
                    superseded_by TEXT,
                    source_memory_id TEXT,
                    confidence INTEGER DEFAULT 3,
                    created_at TEXT NOT NULL
                )
            """,
            migrations=[],
        )

        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tt_subject ON temporal_triples(subject)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tt_predicate ON temporal_triples(predicate)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tt_valid ON temporal_triples(valid_at, invalid_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tt_object ON temporal_triples(object)"
        )

        self._conn.commit()

    # ─── 三元组写入 ─────────────────────────────────────────────

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_at: str,
        source_memory_id: str = "",
        confidence: int = 3,
    ) -> str:
        """添加带时间戳的三元组。

        如果检测到矛盾（同一 subject+predicate 已有当前有效的不同 object），
        自动将旧三元组标记为过时。

        Args:
            subject: 主语实体
            predicate: 关系谓词
            obj: 宾语实体
            valid_at: 有效起始时间（ISO 8601）
            source_memory_id: 来源记忆 ID
            confidence: 置信度（1-5）

        Returns:
            新三元组的 ID
        """
        with self._lock:
            assert self._conn is not None
            triple_id = _generate_id()
            now = datetime.now(timezone.utc).isoformat()

            try:
                # 矛盾检测：同一 subject+predicate 的当前有效三元组
                contradiction = self._detect_contradiction_locked(
                    subject, predicate, obj
                )
                if contradiction:
                    self._invalidate_triple_locked(
                        contradiction.id,
                        invalid_at=valid_at,
                        superseded_by=triple_id,
                    )

                self._conn.execute(
                    """INSERT INTO temporal_triples
                       (id, subject, predicate, object, valid_at, invalid_at,
                        superseded_by, source_memory_id, confidence, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        triple_id,
                        subject,
                        predicate,
                        obj,
                        valid_at,
                        None,
                        None,
                        source_memory_id,
                        confidence,
                        now,
                    ),
                )
                self._pending_writes += 1
                self._maybe_commit()
                return triple_id
            except Exception as e:
                logger.warning("add_triple 失败: %s", e)
                return ""

    def invalidate_triple(
        self,
        triple_id: str,
        invalid_at: str,
        superseded_by: str | None = None,
    ) -> None:
        """标记三元组过时。

        Args:
            triple_id: 要失效的三元组 ID
            invalid_at: 失效时间（ISO 8601）
            superseded_by: 取代该三元组的新三元组 ID
        """
        with self._lock:
            self._invalidate_triple_locked(triple_id, invalid_at, superseded_by)

    def _invalidate_triple_locked(
        self,
        triple_id: str,
        invalid_at: str,
        superseded_by: str | None = None,
    ) -> None:
        """标记三元组过时（内部已持有锁时调用）。"""
        assert self._conn is not None
        try:
            self._conn.execute(
                "UPDATE temporal_triples SET invalid_at = ?, superseded_by = ? WHERE id = ?",
                (invalid_at, superseded_by, triple_id),
            )
            self._pending_writes += 1
            self._maybe_commit()
        except Exception as e:
            logger.warning("invalidate_triple 失败: %s", e)

    # ─── 时序查询 ───────────────────────────────────────────────

    def query_current(
        self, subject: str, predicate: str
    ) -> list[TemporalTriple]:
        """查询当前有效的三元组（invalid_at IS NULL）。

        Args:
            subject: 主语实体（空字符串表示通配）
            predicate: 关系谓词（空字符串表示通配）

        Returns:
            当前有效的 TemporalTriple 列表
        """
        assert self._conn is not None
        try:
            conditions = ["invalid_at IS NULL"]
            params: list[str] = []
            if subject:
                conditions.append("subject = ?")
                params.append(subject)
            if predicate:
                conditions.append("predicate = ?")
                params.append(predicate)

            where = " AND ".join(conditions)
            rows = self._conn.execute(
                f"SELECT * FROM temporal_triples WHERE {where} ORDER BY valid_at DESC",
                params,
            ).fetchall()
            return self._rows_to_triples(rows)
        except Exception as e:
            logger.warning("query_current 失败: %s", e)
            return []

    def query_at_time(
        self, subject: str, predicate: str, at_time: str
    ) -> list[TemporalTriple]:
        """查询指定时间有效的三元组。

        条件: valid_at <= at_time AND (invalid_at IS NULL OR invalid_at > at_time)

        Args:
            subject: 主语实体（空字符串表示通配）
            predicate: 关系谓词（空字符串表示通配）
            at_time: 查询时间点（ISO 8601）

        Returns:
            在 at_time 有效的 TemporalTriple 列表
        """
        assert self._conn is not None
        try:
            conditions = [
                "valid_at <= ?",
                "(invalid_at IS NULL OR invalid_at > ?)",
            ]
            params: list[str] = [at_time, at_time]
            if subject:
                conditions.append("subject = ?")
                params.append(subject)
            if predicate:
                conditions.append("predicate = ?")
                params.append(predicate)

            where = " AND ".join(conditions)
            rows = self._conn.execute(
                f"SELECT * FROM temporal_triples WHERE {where} ORDER BY valid_at DESC",
                params,
            ).fetchall()
            return self._rows_to_triples(rows)
        except Exception as e:
            logger.warning("query_at_time 失败: %s", e)
            return []

    # ─── 矛盾检测 ───────────────────────────────────────────────

    def detect_contradiction(
        self, subject: str, predicate: str, obj: str
    ) -> TemporalTriple | None:
        """检测与已有事实的矛盾。

        查找同一 subject+predicate 下当前有效但 object 不同的三元组。

        Args:
            subject: 主语实体
            predicate: 关系谓词
            obj: 新的宾语实体

        Returns:
            矛盾的 TemporalTriple，无矛盾返回 None
        """
        with self._lock:
            return self._detect_contradiction_locked(subject, predicate, obj)

    def _detect_contradiction_locked(
        self, subject: str, predicate: str, obj: str
    ) -> TemporalTriple | None:
        """矛盾检测（内部已持有锁时调用）。"""
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                """SELECT * FROM temporal_triples
                   WHERE subject = ? AND predicate = ? AND object != ? AND invalid_at IS NULL
                   LIMIT 1""",
                (subject, predicate, obj),
            ).fetchall()
            if rows:
                return self._row_to_triple(rows[0])
            return None
        except Exception as e:
            logger.warning("detect_contradiction 失败: %s", e)
            return None

    # ─── 实体时间线 ─────────────────────────────────────────────

    def get_timeline(self, subject: str, limit: int = 50) -> list[TemporalTriple]:
        """获取实体的完整时间线。

        返回与该实体相关的所有三元组（作为主语或宾语），按 valid_at 升序排列，
        形成实体关系演变时间线。对标 Zep/Graphiti 的 Entity Timeline。

        Args:
            subject: 实体名称
            limit: 最大返回数量

        Returns:
            按 valid_at 升序的 TemporalTriple 列表
        """
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                """SELECT * FROM temporal_triples
                   WHERE subject = ? OR object = ?
                   ORDER BY valid_at ASC LIMIT ?""",
                (subject, subject, limit),
            ).fetchall()
            return self._rows_to_triples(rows)
        except Exception as e:
            logger.warning("get_timeline 失败: %s", e)
            return []

    def get_timeline_text(self, subject: str, limit: int = 20) -> str:
        """生成实体时间线的可读文本，适合注入 LLM 上下文。

        Returns:
            格式化的时间线文本，无结果返回空字符串
        """
        timeline = self.get_timeline(subject, limit=limit)
        if not timeline:
            return ""

        relation_labels = {
            "uses": "开始使用",
            "belongs_to": "归属",
            "causes": "引起",
            "replaces": "取代",
            "connects_to": "关联到",
            "contains": "包含",
            "located_in": "位于",
            "better_than": "优于",
            "not_uses": "不再使用",
            "differs_from": "不同于",
        }

        lines = [f"[{subject} 时序时间线]"]
        for t in timeline:
            valid_date = t.valid_at[:10] if t.valid_at else "?"
            label = relation_labels.get(t.predicate, t.predicate)
            status = "✓" if t.is_current() else "✗"
            line = f"  {valid_date} {status} {t.subject} {label} {t.object}"
            if t.invalid_at:
                line += f" (至 {t.invalid_at[:10]})"
            if t.superseded_by:
                line += f" → 被 {t.superseded_by[:8]} 取代"
            lines.append(line)

        return "\n".join(lines)

    # ─── 从 KnowledgeGraph 同步 ─────────────────────────────────

    def add_triple_from_kg(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_at: str,
        source_memory_id: str = "",
        confidence: int = 3,
    ) -> str:
        """从 KnowledgeGraph 同步三元组到时序图谱。

        与 add_triple 相同逻辑，但语义上标记来源为 KG 同步，
        便于后续溯源。

        Returns:
            新三元组的 ID
        """
        return self.add_triple(
            subject=subject,
            predicate=predicate,
            obj=obj,
            valid_at=valid_at,
            source_memory_id=source_memory_id or "kg_sync",
            confidence=confidence,
        )

    # ─── 时序图谱检索 ───────────────────────────────────────────

    def temporal_search(
        self, query_entities: list[str], at_time: str | None = None, limit: int = 20
    ) -> list[TemporalTriple]:
        """时序图谱检索：对实体列表查询当前或历史状态。

        Args:
            query_entities: 查询实体列表
            at_time: 查询时间点，None 表示查询当前
            limit: 最大返回数量

        Returns:
            匹配的 TemporalTriple 列表
        """
        if not query_entities:
            return []

        all_triples: list[TemporalTriple] = []
        seen_ids: set[str] = set()

        for entity in query_entities[:3]:
            if at_time:
                triples = self.query_at_time(entity, "", at_time)
            else:
                triples = self.query_current(entity, "")
            for t in triples:
                if t.id not in seen_ids:
                    seen_ids.add(t.id)
                    all_triples.append(t)

        return all_triples[:limit]

    def temporal_rag_context(
        self, query_entities: list[str], at_time: str | None = None
    ) -> str:
        """生成时序图谱的 RAG 上下文文本。

        Args:
            query_entities: 查询实体列表
            at_time: 查询时间点，None 表示当前

        Returns:
            格式化的时序上下文文本
        """
        triples = self.temporal_search(query_entities, at_time=at_time)
        if not triples:
            return ""

        time_label = at_time[:10] if at_time else "当前"
        lines = [f"[时序知识图谱 @ {time_label}]"]

        for t in triples:
            status = "✓" if t.is_current() else "✗"
            line = f"  {status} {t.subject} {t.predicate} {t.object}"
            if t.valid_at:
                line += f" (自 {t.valid_at[:10]})"
            if t.invalid_at:
                line += f" (至 {t.invalid_at[:10]})"
            lines.append(line)

        return "\n".join(lines)

    # ─── 统计 ───────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """获取时序图谱统计。"""
        stats: dict[str, Any] = {
            "total_triples": 0,
            "current_triples": 0,
            "superseded_triples": 0,
        }
        if not self._conn:
            return stats
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM temporal_triples"
            ).fetchone()
            stats["total_triples"] = row[0] if row else 0

            row = self._conn.execute(
                "SELECT COUNT(*) FROM temporal_triples WHERE invalid_at IS NULL"
            ).fetchone()
            stats["current_triples"] = row[0] if row else 0

            row = self._conn.execute(
                "SELECT COUNT(*) FROM temporal_triples WHERE superseded_by IS NOT NULL"
            ).fetchone()
            stats["superseded_triples"] = row[0] if row else 0
        except Exception as e:
            logger.warning("get_stats 失败: %s", e)
        return stats

    # ─── 生命周期 ───────────────────────────────────────────────

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
                self._conn.commit()
                self._pending_writes = 0
            except Exception as e:
                logger.warning("flush 失败: %s", e)

    # ─── 内部方法 ───────────────────────────────────────────────

    def _maybe_commit(self) -> None:
        """到达阈值时提交。"""
        if self._pending_writes >= self._BATCH_THRESHOLD:
            assert self._conn is not None
            self._conn.commit()
            self._pending_writes = 0

    def _rows_to_triples(self, rows: list[Any]) -> list[TemporalTriple]:
        """将行转为 TemporalTriple 列表。"""
        return [self._row_to_triple(row) for row in rows]

    @staticmethod
    def _row_to_triple(row: Any) -> TemporalTriple:
        """将单行转为 TemporalTriple。"""
        return TemporalTriple(
            id=row[0] if len(row) > 0 else "",
            subject=row[1] if len(row) > 1 else "",
            predicate=row[2] if len(row) > 2 else "",
            object=row[3] if len(row) > 3 else "",
            valid_at=row[4] if len(row) > 4 else "",
            invalid_at=row[5] if len(row) > 5 else None,
            superseded_by=row[6] if len(row) > 6 else None,
            source_memory_id=row[7] if len(row) > 7 else "",
            confidence=row[8] if len(row) > 8 else 3,
            created_at=row[9] if len(row) > 9 else "",
        )


def _generate_id() -> str:
    """生成三元组唯一 ID。"""
    return f"tt-{uuid.uuid4().hex[:12]}"
