"""混合检索与分数校准模块

融合向量检索与 BM25 关键词检索，支持 minmax / sigmoid / zscore 三种分数校准策略，
并提供可选的交叉编码器重排序能力。
"""

import logging
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("skill_router_init.hybrid_searcher")


class HybridSearcher:
    """混合检索器：融合向量检索与 BM25 关键词检索

    对向量分数和 BM25 分数按配置策略校准后，再按配置权重加权融合。
    支持可选的 Reranker 重排序（交叉编码器）。
    """

    # Reranker 缓存（类级别，线程安全）
    _reranker_model: Optional[Any] = None
    _reranker_lock = threading.Lock()

    @classmethod
    def _load_reranker(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", backend: str = "lightweight") -> Optional[Any]:
        """加载交叉编码器 Reranker（延迟加载，线程安全）

        当 backend 为 "skillrouter" 时跳过加载，由 SkillRouterBackend 处理重排序。
        """
        # SkillRouter backend handles its own reranking
        if backend == "skillrouter":
            logger.debug("SkillRouter backend active, skipping CrossEncoder reranker load")
            return None

        if cls._reranker_model is not None:
            return cls._reranker_model

        with cls._reranker_lock:
            if cls._reranker_model is not None:
                return cls._reranker_model

            try:
                from sentence_transformers import CrossEncoder

                # 展开 ~ 路径
                resolved_path = os.path.expanduser(model_name)
                logger.info("加载 Reranker 模型: %s", resolved_path)
                cls._reranker_model = CrossEncoder(resolved_path, device="cpu")
                logger.info("Reranker 模型加载完成")
                return cls._reranker_model
            except Exception as e:
                logger.warning("Reranker 模型加载失败: %s，将跳过重排序", e)
                return None

    @staticmethod
    def _rerank_results(
        query: str,
        candidates: List[Dict[str, Any]],
        skills: Dict[str, Dict[str, Any]],
        top_k: int,
        reranker_model: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """使用交叉编码器对候选结果重排序

        参数:
            query: 查询文本
            candidates: 候选结果列表（包含 name 和 score）
            skills: 技能索引字典
            top_k: 返回数量
            reranker_model: 预加载的 Reranker 模型

        返回:
            重排序后的结果列表
        """
        if not candidates or len(candidates) <= 1:
            return candidates[:top_k]

        # 加载 Reranker
        if reranker_model is None:
            reranker_model = HybridSearcher._load_reranker()

        if reranker_model is None:
            logger.debug("Reranker 不可用，返回原始结果")
            return candidates[:top_k]

        try:
            # 构造 (query, doc) 对
            pairs: List[Tuple[str, str]] = []
            for item in candidates:
                name = item["name"]
                skill = skills.get(name, {})
                # 使用技能名称 + 描述作为文档
                doc = f"{name}: {skill.get('description', '')}"
                pairs.append((query, doc))

            # 交叉编码器打分
            scores = reranker_model.predict(pairs, show_progress_bar=False)

            # 重新排序
            scored_candidates: List[Dict[str, Any]] = []
            for i, item in enumerate(candidates):
                scored_candidates.append({
                    **item,
                    "rerank_score": float(scores[i]),
                    "original_score": item["score"],
                    "score": float(scores[i]),  # 使用 Reranker 分数作为最终分数
                })

            scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

            logger.debug("Reranker 重排序完成: %d 个候选，返回 top %d", len(candidates), top_k)
            return scored_candidates[:top_k]

        except Exception as e:
            logger.error("Reranker 重排序失败: %s，返回原始结果", e)
            return candidates[:top_k]

    def __init__(self, config: Dict[str, Any]):
        self._vector_weight = float(config.get("vector_weight", 0.7))
        self._bm25_weight = float(config.get("bm25_weight", 0.3))
        self._use_reranker = bool(config.get("use_reranker", True))
        self._calibration = config.get("hybrid_calibration", "minmax")
        self._reranker_model: Optional[Any] = None

    @staticmethod
    def _minmax_normalize(score_map: Dict[str, float]) -> Dict[str, float]:
        """将分数归一化到 [0, 1] 范围（min-max 归一化）"""
        if not score_map:
            return {}
        values = list(score_map.values())
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return {k: 1.0 for k in score_map}
        return {k: (v - min_v) / (max_v - min_v) for k, v in score_map.items()}

    @staticmethod
    def _sigmoid_normalize(score_map: Dict[str, float]) -> Dict[str, float]:
        """使用 sigmoid 函数压缩分数到 (0, 1)"""
        if not score_map:
            return {}
        return {k: 1.0 / (1.0 + math.exp(-v)) for k, v in score_map.items()}

    @staticmethod
    def _zscore_normalize(score_map: Dict[str, float]) -> Dict[str, float]:
        """使用本次候选的均值与标准差做标准化"""
        if not score_map:
            return {}
        values = list(score_map.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        if std == 0.0:
            return {k: 0.0 for k in score_map}
        return {k: (v - mean) / std for k, v in score_map.items()}

    def _calibrate_scores(self, score_map: Dict[str, float]) -> Dict[str, float]:
        """根据配置策略对分数进行校准"""
        strategy = self._calibration
        if strategy == "sigmoid":
            return self._sigmoid_normalize(score_map)
        if strategy == "zscore":
            return self._zscore_normalize(score_map)
        # 默认 minmax
        return self._minmax_normalize(score_map)

    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_results: Optional[List[Dict[str, Any]]] = None,
        bm25_results: Optional[List[Tuple[str, float]]] = None,
        skill_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """融合向量与 BM25 检索结果

        参数:
            query: 查询文本（保留接口一致性，实际不参与计算）
            top_k: 返回结果数量
            vector_results: 向量检索结果列表，每项含 name 和 score
            bm25_results: BM25 检索结果列表，每项为 (name, score)
            skill_names: 全量技能名列表（用于兜底遍历）

        返回:
            融合后的结果列表，按加权分数降序排列
        """
        vector_results = vector_results or []
        bm25_results = bm25_results or []

        vector_score_map: Dict[str, float] = {r["name"]: r["score"] for r in vector_results}
        bm25_score_map: Dict[str, float] = dict(bm25_results)

        norm_vector = self._calibrate_scores(vector_score_map)
        norm_bm25 = self._calibrate_scores(bm25_score_map)

        all_names: set[str] = set()
        if skill_names:
            all_names.update(skill_names)
        all_names.update(norm_vector.keys())
        all_names.update(norm_bm25.keys())

        fused: List[Dict[str, Any]] = []
        for name in all_names:
            v_score = norm_vector.get(name, 0.0)
            b_score = norm_bm25.get(name, 0.0)
            weighted = self._vector_weight * v_score + self._bm25_weight * b_score
            fused.append({"name": name, "score": weighted})

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]
