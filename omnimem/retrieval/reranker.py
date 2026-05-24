"""CrossEncoderReranker — Cross-Encoder 重排。

使用 Cross-Encoder 模型对检索结果进行精排，提升精度。
可选功能，需要 sentence-transformers 库。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-Encoder 重排。"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", model_path: str = ""):
        self._model_name = model_name
        self._model_path = model_path
        self._model: Any = None

    def _ensure_model(self) -> bool:
        """延迟加载 Cross-Encoder 模型。"""
        if self._model is not None:
            return True
        try:
            # ROCm PyTorch 兼容性
            import torch.distributed as dist
            if not hasattr(dist, 'is_initialized'):
                dist.is_initialized = lambda: False
            from sentence_transformers import CrossEncoder

            model_path = self._model_path or self._model_name
            self._model = CrossEncoder(model_path)
            return True
        except ImportError:
            logger.warning("sentence_transformers not installed — reranking disabled")
            return False
        except Exception as e:
            logger.warning("Cross-Encoder model load failed: %s", e)
            return False

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """对检索结果进行 Cross-Encoder 重排。

        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回前 K 个结果

        Returns:
            重排后的结果列表
        """
        if not results or not self._ensure_model():
            return results[:top_k]

        try:
            pairs = [(query, r.get("content", "")) for r in results]
            if self._model is None:
                return results[:top_k]
            scores = self._model.predict(pairs)

            # 按 score 降序排列
            scored = list(zip(results, scores, strict=False))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for doc, score in scored[:top_k]:
                entry = dict(doc)
                entry["rerank_score"] = float(score)
                reranked.append(entry)
            return reranked
        except Exception as e:
            logger.warning("Reranking failed: %s", e)
            return results[:top_k]
