"""RetrievalQualityEvaluator — 检索质量评估与自动调优。

核心指标:
  - precision: 精确率 = |relevant ∩ returned| / |returned|
  - recall: 召回率 = |relevant ∩ returned| / |relevant|
  - MRR: Mean Reciprocal Rank = 1/rank_of_first_relevant
  - nDCG: Normalized Discounted Cumulative Gain = DCG / IDCG

自动调优策略:
  - 精确率低 → 提高 min_rrf 阈值（过滤更多低质量结果）
  - 召回率低 → 降低 min_rrf 阈值、增加 top_k
  - MRR 低 → 调整 RRF 权重（向量权重↑、BM25权重↓）
  - 延迟高 → 减少 top_k、启用缓存
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    precision: float
    recall: float
    mrr: float
    ndcg: float
    latency_ms: float
    result_count: int
    query: str
    timestamp: str


class RetrievalQualityEvaluator:
    """检索质量评估器：计算指标 + 持久化记录 + 趋势分析 + 自动调优建议。"""

    def __init__(self, data_dir: Path, config: Any = None) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "quality_eval.db"
        self._conn: sqlite3.Connection | None = None
        self._config = config
        self._init_db()

    def _init_db(self) -> None:
        """初始化质量评估数据库。"""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS quality_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                precision REAL,
                recall REAL,
                mrr REAL,
                ndcg REAL,
                latency_ms REAL,
                result_count INTEGER,
                timestamp TEXT NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_quality_timestamp ON quality_evaluations(timestamp)"
        )
        self._conn.commit()

    def evaluate(
        self,
        query: str,
        results: list[dict[str, Any]],
        relevant_ids: set[str],
        latency_ms: float,
    ) -> QualityMetrics:
        """计算检索质量指标。

        Args:
            query: 查询文本
            results: 检索结果列表，每项需含 memory_id 和 score
            relevant_ids: 相关文档 ID 集合（ground truth 或启发式推断）
            latency_ms: 检索延迟（毫秒）

        Returns:
            QualityMetrics 数据类实例
        """
        returned_ids = set()
        for r in results:
            mid = r.get("memory_id", "")
            if mid:
                returned_ids.add(mid)

        returned_count = len(returned_ids)
        relevant_count = len(relevant_ids)

        if returned_count == 0:
            precision = 0.0
        else:
            precision = len(relevant_ids & returned_ids) / returned_count

        if relevant_count == 0:
            recall = 1.0 if returned_count == 0 else 0.0
        else:
            recall = len(relevant_ids & returned_ids) / relevant_count

        mrr = self._compute_mrr(results, relevant_ids)
        ndcg = self._compute_ndcg(results, relevant_ids)

        now = datetime.now(timezone.utc).isoformat()

        return QualityMetrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            mrr=round(mrr, 4),
            ndcg=round(ndcg, 4),
            latency_ms=round(latency_ms, 2),
            result_count=returned_count,
            query=query,
            timestamp=now,
        )

    @staticmethod
    def _compute_mrr(results: list[dict[str, Any]], relevant_ids: set[str]) -> float:
        """计算 MRR（Mean Reciprocal Rank）。

        MRR = 1 / rank_of_first_relevant
        """
        for rank, r in enumerate(results, start=1):
            mid = r.get("memory_id", "")
            if mid and mid in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def _compute_ndcg(results: list[dict[str, Any]], relevant_ids: set[str]) -> float:
        """计算 nDCG（Normalized Discounted Cumulative Gain）。

        DCG = sum(relevance_i / log2(rank_i + 1))
        IDCG = 理想排序下的 DCG
        nDCG = DCG / IDCG
        """
        if not relevant_ids:
            return 1.0

        relevance_scores = []
        for r in results:
            mid = r.get("memory_id", "")
            if mid and mid in relevant_ids:
                relevance_scores.append(1.0)
            else:
                relevance_scores.append(0.0)

        dcg = 0.0
        for rank, rel in enumerate(relevance_scores, start=1):
            if rel > 0:
                dcg += rel / math.log2(rank + 1)

        ideal_relevance = sorted(relevance_scores, reverse=True)
        ideal_count = min(len(relevant_ids), len(results))
        ideal_relevance = [1.0] * ideal_count + [0.0] * (len(results) - ideal_count)

        idcg = 0.0
        for rank, rel in enumerate(ideal_relevance, start=1):
            if rel > 0:
                idcg += rel / math.log2(rank + 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def record_evaluation(self, metrics: QualityMetrics) -> None:
        """将评估结果持久化到 SQLite。

        Args:
            metrics: QualityMetrics 实例
        """
        assert self._conn is not None
        try:
            self._conn.execute(
                """INSERT INTO quality_evaluations
                   (query, precision, recall, mrr, ndcg, latency_ms, result_count, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    metrics.query,
                    metrics.precision,
                    metrics.recall,
                    metrics.mrr,
                    metrics.ndcg,
                    metrics.latency_ms,
                    metrics.result_count,
                    metrics.timestamp,
                ),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning("质量评估记录写入失败: %s", e)

    def get_trend(self, days: int = 7) -> dict[str, float]:
        """获取最近 N 天的质量趋势（各指标均值）。

        Args:
            days: 统计窗口天数

        Returns:
            各指标的均值字典
        """
        assert self._conn is not None
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        try:
            row = self._conn.execute(
                """SELECT
                     AVG(precision) as avg_precision,
                     AVG(recall) as avg_recall,
                     AVG(mrr) as avg_mrr,
                     AVG(ndcg) as avg_ndcg,
                     AVG(latency_ms) as avg_latency_ms,
                     AVG(result_count) as avg_result_count,
                     COUNT(*) as sample_count
                   FROM quality_evaluations
                   WHERE timestamp >= ?""",
                (cutoff,),
            ).fetchone()

            if row and row[6] and row[6] > 0:
                return {
                    "precision": round(row[0] or 0, 4),
                    "recall": round(row[1] or 0, 4),
                    "mrr": round(row[2] or 0, 4),
                    "ndcg": round(row[3] or 0, 4),
                    "latency_ms": round(row[4] or 0, 2),
                    "result_count": round(row[5] or 0, 1),
                    "sample_count": int(row[6]),
                }
        except Exception as e:
            logger.warning("获取质量趋势失败: %s", e)

        return {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "latency_ms": 0.0,
            "result_count": 0.0,
            "sample_count": 0,
        }

    def get_auto_tune_suggestions(self) -> dict[str, Any]:
        """根据质量趋势给出参数调优建议。

        策略:
          - 精确率低 → 建议提高 min_rrf 阈值
          - 召回率低 → 建议降低 min_rrf 阈值、增加 top_k
          - MRR 低 → 建议调整 RRF 权重（向量权重↑、BM25权重↓）
          - 延迟高 → 建议减少 top_k、启用缓存

        Returns:
            调优建议字典
        """
        trend = self.get_trend(days=7)
        sample_count = trend.get("sample_count", 0)

        if sample_count < 3:
            return {
                "suggestions": [],
                "reason": f"样本数不足（{sample_count} < 3），暂无调优建议",
                "trend": trend,
            }

        suggestions: list[dict[str, str]] = []
        precision = trend["precision"]
        recall = trend["recall"]
        mrr = trend["mrr"]
        latency = trend["latency_ms"]

        if precision < 0.3:
            suggestions.append({
                "parameter": "min_rrf",
                "action": "increase",
                "description": f"精确率偏低（{precision:.2f}），建议提高 min_rrf 阈值以过滤低质量结果",
                "suggested_value": "0.045",
            })

        if recall < 0.3:
            suggestions.append({
                "parameter": "min_rrf",
                "action": "decrease",
                "description": f"召回率偏低（{recall:.2f}），建议降低 min_rrf 阈值以召回更多结果",
                "suggested_value": "0.025",
            })
            suggestions.append({
                "parameter": "top_k",
                "action": "increase",
                "description": f"召回率偏低（{recall:.2f}），建议增加 top_k 以扩大检索范围",
                "suggested_value": "15",
            })

        if mrr < 0.3:
            suggestions.append({
                "parameter": "rrf_vector_weight",
                "action": "increase",
                "description": f"MRR 偏低（{mrr:.2f}），建议提高向量检索权重（语义匹配优先）",
                "suggested_value": "4.0",
            })
            suggestions.append({
                "parameter": "rrf_bm25_weight",
                "action": "decrease",
                "description": f"MRR 偏低（{mrr:.2f}），建议降低 BM25 权重（减少关键词噪音）",
                "suggested_value": "0.8",
            })

        if latency > 2000:
            suggestions.append({
                "parameter": "top_k",
                "action": "decrease",
                "description": f"检索延迟偏高（{latency:.0f}ms），建议减少 top_k 以降低延迟",
                "suggested_value": "5",
            })
            suggestions.append({
                "parameter": "query_cache_ttl",
                "action": "enable",
                "description": f"检索延迟偏高（{latency:.0f}ms），建议启用查询缓存以减少重复计算",
                "suggested_value": "120",
            })

        return {
            "suggestions": suggestions,
            "reason": f"基于 {sample_count} 个样本的7天趋势分析",
            "trend": trend,
        }

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            try:
                self._conn.close()
            except Exception as e:
                logger.warning("质量评估数据库关闭失败: %s", e)
            self._conn = None

    @staticmethod
    def infer_relevant_ids(results: list[dict[str, Any]]) -> set[str]:
        """启发式推断相关文档 ID（无 ground truth 时的替代方案）。

        规则:
          - score >= 0.05 的条目视为相关
          - 含 type_boost 标记的条目视为相关

        Args:
            results: 检索结果列表

        Returns:
            推断的相关 ID 集合
        """
        relevant = set()
        for r in results:
            mid = r.get("memory_id", "")
            if not mid:
                continue
            score = r.get("score", 0)
            has_type_boost = "type_boost" in r
            if score >= 0.05 or has_type_boost:
                relevant.add(mid)
        return relevant
