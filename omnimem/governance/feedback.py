"""FeedbackCollector — 检索反馈收集器。

Cross-Encoder 在线学习的基础层：
  1. 记录 Agent 实际使用了哪些召回结果（通过 omni_detail 调用追踪）
  2. 记录 recall 返回的候选列表与最终 Agent 的选择
  3. 基于统计生成来源权重（vector/bm25/graph/store_supplement），
     供 HybridRetriever 的 RRF 融合动态调整

设计为轻量级实现：不训练神经网络，而是用统计反馈做来源加权。
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """检索反馈收集器。

    表结构：
      feedback_clicks   — 用户/Agent 实际点击/使用的记忆
      feedback_shown    — recall/prefetch 返回但未使用的候选
    """

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "feedback.db"
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """初始化反馈数据库。"""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")

        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="feedback_clicks",
            create_sql="""
                CREATE TABLE IF NOT EXISTS feedback_clicks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    memory_id TEXT,
                    source_type TEXT,
                    rank INTEGER,
                    action TEXT DEFAULT 'click',
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """,
            migrations=[],
        )
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fb_query ON feedback_clicks(query)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fb_mid ON feedback_clicks(memory_id)
        """)

        migrator.migrate(
            table_name="feedback_shown",
            create_sql="""
                CREATE TABLE IF NOT EXISTS feedback_shown (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    memory_id TEXT,
                    source_type TEXT,
                    rank INTEGER,
                    was_clicked INTEGER DEFAULT 0,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """,
            migrations=[],
        )
        self._conn.commit()

    def record_click(
        self,
        query: str,
        memory_id: str,
        source_type: str = "",
        rank: int = 0,
    ) -> None:
        """记录 Agent 实际使用了一条记忆（如通过 omni_detail 拉取）。"""
        if not self._conn:
            return
        try:
            self._conn.execute(
                """INSERT INTO feedback_clicks (query, memory_id, source_type, rank)
                   VALUES (?, ?, ?, ?)""",
                (query, memory_id, source_type, rank),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning("Feedback click record failed: %s", e)

    def record_shown(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> None:
        """记录 recall/prefetch 返回的候选列表。

        Args:
            query: 查询文本
            candidates: 候选结果列表，每项需含 memory_id 和 _source/type
        """
        if not self._conn or not candidates:
            return
        try:
            for rank, c in enumerate(candidates[:20], start=1):  # 只记录前 20
                self._conn.execute(
                    """INSERT INTO feedback_shown (query, memory_id, source_type, rank)
                       VALUES (?, ?, ?, ?)""",
                    (
                        query,
                        c.get("memory_id", ""),
                        c.get("_source", "") or c.get("type", "unknown"),
                        rank,
                    ),
                )
            self._conn.commit()
        except Exception as e:
            logger.warning("Feedback shown record failed: %s", e)

    def get_source_weights(self, window: int = 100) -> dict[str, float]:
        """基于最近反馈计算各来源的权重。

        逻辑：
          weight(source) = 1.0 + 0.5 * (点击率 - 基准点击率)
          点击率 = clicks / shown

        Args:
            window: 最近多少条 click 记录参与计算

        Returns:
            source → weight 映射，默认 1.0
        """
        if not self._conn:
            return {}
        try:
            # 获取最近 window 次点击涉及的查询
            rows = self._conn.execute(
                """SELECT DISTINCT query FROM feedback_clicks
                   ORDER BY timestamp DESC LIMIT ?""",
                (window,),
            ).fetchall()
            recent_queries = [r[0] for r in rows if r[0]]
            if not recent_queries:
                return {}

            placeholders = ",".join("?" * len(recent_queries))
            # 各来源被点击次数
            click_rows = self._conn.execute(
                f"""SELECT source_type, COUNT(*) FROM feedback_clicks
                    WHERE query IN ({placeholders}) AND source_type != ''
                    GROUP BY source_type""",
                recent_queries,
            ).fetchall()
            # 各来源被展示次数
            shown_rows = self._conn.execute(
                f"""SELECT source_type, COUNT(*) FROM feedback_shown
                    WHERE query IN ({placeholders}) AND source_type != ''
                    GROUP BY source_type""",
                recent_queries,
            ).fetchall()

            clicks = {r[0]: r[1] for r in click_rows}
            shown = {r[0]: r[1] for r in shown_rows}

            weights: dict[str, float] = {}
            for src, s_count in shown.items():
                c_count = clicks.get(src, 0)
                ctr = c_count / s_count if s_count > 0 else 0
                # 基准点击率假设 0.3，高于则加权，低于则降权
                weights[src] = round(1.0 + 0.8 * (ctr - 0.3), 2)
            return weights
        except Exception as e:
            logger.warning("Source weights calculation failed: %s", e)
            return {}

    def get_training_triplets(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """获取用于 Cross-Encoder 微调的三元组样本。

        Returns:
            [{"query": str, "positive": str, "negative": str}, ...]
        """
        if not self._conn:
            return []
        try:
            rows = self._conn.execute(
                """SELECT query, memory_id FROM feedback_clicks
                   ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            triplets = []
            for query, mid in rows:
                # 找到同一查询下被展示但未点击的作为负例
                neg = self._conn.execute(
                    """SELECT memory_id FROM feedback_shown
                       WHERE query = ? AND memory_id != ?
                       ORDER BY rank DESC LIMIT 1""",
                    (query, mid),
                ).fetchone()
                if neg:
                    triplets.append(
                        {
                            "query": query,
                            "positive": mid,
                            "negative": neg[0],
                        }
                    )
            return triplets
        except Exception as e:
            logger.warning("Training triplets fetch failed: %s", e)
            return []

    def get_stats(self) -> dict[str, Any]:
        """返回反馈统计。"""
        stats = {"total_clicks": 0, "total_shown": 0, "sources": {}}
        if not self._conn:
            return stats
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM feedback_clicks").fetchone()
            stats["total_clicks"] = row[0] if row else 0
            row = self._conn.execute("SELECT COUNT(*) FROM feedback_shown").fetchone()
            stats["total_shown"] = row[0] if row else 0
            rows = self._conn.execute(
                "SELECT source_type, COUNT(*) FROM feedback_clicks GROUP BY source_type"
            ).fetchall()
            stats["sources"] = {r[0]: r[1] for r in rows}
        except Exception as e:
            logger.warning("Feedback stats fetch failed: %s", e)
        return stats

    def get_memory_trust(self, memory_id: str) -> float:
        """基于反馈数据计算记忆信任评分 (0.0-1.0)。

        对标 Holographic 的信任评分：每次点击 +0.05，展示未点击不扣分，
        基于点击频次衰减归一。
        """
        if not self._conn:
            return 0.5  # default trust
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM feedback_clicks WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            click_count = row[0] if row else 0

            # 归一化：最多 20 次点击得满分
            base_trust = min(click_count / 20.0, 1.0)

            # 最近一次点击越近，信任越高
            last_click = self._conn.execute(
                "SELECT timestamp FROM feedback_clicks WHERE memory_id = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (memory_id,),
            ).fetchone()

            if last_click and last_click[0]:
                try:
                    from datetime import datetime, timezone
                    last_dt = datetime.fromisoformat(last_click[0].replace("Z", "+00:00"))
                    days_ago = (datetime.now(timezone.utc) - last_dt.replace(tzinfo=timezone.utc)).days
                    recency_bonus = max(0, 0.1 - days_ago * 0.01)  # 10天内有效
                    base_trust = min(base_trust + recency_bonus, 1.0)
                except (ValueError, TypeError):
                    logger.debug("Feedback: parse last_click datetime failed", exc_info=True)

            return round(base_trust, 2)
        except Exception as e:
            logger.warning("Trust score calculation failed: %s", e)
            return 0.5

    def close(self):
        if self._conn:
            try:
                self._conn.close()
                self._conn = None
            except Exception:
                logger.debug("Feedback: DB connection close failed", exc_info=True)
