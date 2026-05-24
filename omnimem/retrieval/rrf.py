"""RRFFusion — Reciprocal Rank Fusion 融合算法。

将多路检索结果（向量、BM25、实体等）通过 RRF 算法融合为统一排名。
公式: RRF_score(d) = sum(1 / (k + rank_i(d))) for each ranking i
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RRFFusion:
    """Reciprocal Rank Fusion 融合算法。"""

    def __init__(self, k: int = 60, min_rrf: float = 0.035):
        """初始化 RRF。

        Args:
            k: 平滑常数，默认60（标准值）
            min_rrf: 最低RRF分数阈值，低于此值的结果被过滤。
                     默认0.035（双路融合需至少一个通道排名前5才能通过）。
                     原值0.015导致无关查询返回过多噪声结果（QUAL-1）。
        """
        self._k = k
        self._min_rrf = min_rrf

    def merge(
        self,
        result_lists: list[list[dict[str, Any]]],
        id_key: str = "memory_id",
        weights: Optional[list[float]] = None,
        min_rrf: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """融合多路检索结果。

        Args:
            result_lists: 多路检索结果列表，每路是一个 List[dict]
            id_key: 用于标识文档的 key
            weights: 各路权重，默认[1.5, 1.0]（向量1.5x, BM25 1.0x）
                     向量检索能跨越语义鸿沟，给予更高权重
            min_rrf: 覆盖实例默认的最低RRF分数阈值（QUAL-1修复）

        Returns:
            融合后的排序结果
        """
        # ★ 默认权重：第一路（向量）3.0x，第二路（BM25）1.0x
        # ★ QUAL-3修复：从4.0调整至3.0，优化语义检索与关键词平衡
        #   向量检索能跨越"宠物"↔"橘猫"等语义鸿沟，需给予更高权重
        #   权重比 3:1 ≈ 75:25，确保语义相关结果优先于关键词匹配
        if weights is None:
            weights = [3.0, 1.0] + [1.0] * max(0, len(result_lists) - 2)

        # 为每个文档累计 RRF 分数
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        for list_idx, result_list in enumerate(result_lists):
            weight = weights[list_idx] if list_idx < len(weights) else 1.0
            for rank, doc in enumerate(result_list, start=1):
                doc_id = doc.get(id_key, "")
                if not doc_id:
                    content = doc.get("content", "")
                    doc_id = f"hash-{hash(content)}"
                # ★ 加权 RRF 分数：向量检索通道权重更高
                rrf_score = weight / (self._k + rank)
                # ★ 向量通道(cosine similarity)额外加权：
                # 如果原始 score > 0.5，说明语义高度匹配，额外提升
                if list_idx == 0 and doc.get("score", 0) > 0.5:
                    rrf_score *= 1.5
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # 按 RRF 分数降序排列
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True,
        )

        results = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id]
            entry = dict(doc)
            entry["rrf_score"] = rrf_scores[doc_id]
            entry["score"] = rrf_scores[doc_id]  # ★ 统一用 rrf_score 作为最终 score
            results.append(entry)

        # ★ 最低 RRF 分数过滤（QUAL-1修复：从0.015提升至0.035）
        # 向量2x权重下: 排名1=2.0/61=0.0328, 排名5=2.0/65=0.0308
        # BM25 1x权重下: 排名1=1.0/61=0.0164, 排名5=1.0/65=0.0154
        # 双路融合排1: 0.0328+0.0164=0.0492
        # 单路前5: ~0.031（向量）或 ~0.016（BM25）
        # 阈值0.035 → 只保留至少在向量通道排名前5 或 双通道均有排名的结果
        threshold = min_rrf if min_rrf is not None else self._min_rrf
        results = [r for r in results if r["rrf_score"] >= threshold]

        return results
