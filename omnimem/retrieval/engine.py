"""HybridRetriever — 混合检索编排 Facade。

6通道并行检索 + RRF 融合 + 可选 Cross-Encoder Rerank：
  1. 向量检索 (ChromaDB)
  2. BM25 关键词检索
  3. 目录检索 (Wing/Hall/Room 结构过滤)
  4. 实体提升 (Phase 3)
  5. 时间检索 (Phase 3)
  6. 图谱检索 (Phase 3)

Phase 1-2 实现: 向量 + BM25 + RRF 融合

读写锁优化：search() 用读锁（可并行），add() 用写锁（独占），
避免后台 queue_prefetch 写入阻塞主线程 prefetch 搜索。

具体检索逻辑已拆分到以下模块：
  - circuit_breaker.py
  - rw_lock.py
  - query_quality.py
  - synonym_expander.py
  - hybrid_orchestrator.py
"""

from __future__ import annotations

import asyncio as _asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from omnimem.embedding.base import EmbeddingProvider
from omnimem.retrieval.bm25 import BM25Retriever
from omnimem.retrieval.circuit_breaker import CircuitBreaker
from omnimem.retrieval.hybrid_orchestrator import HybridOrchestrator
from omnimem.retrieval.query_quality import is_garbage_query, trim_to_budget
from omnimem.retrieval.registry import DEFAULT_REGISTRY, RetrieverRegistry
from omnimem.retrieval.reranker import CrossEncoderReranker
from omnimem.retrieval.rrf import RRFFusion
from omnimem.retrieval.rw_lock import FairReadWriteLock
from omnimem.retrieval.vector import VectorRetriever
from omnimem.retrieval.vector_store import _emit
from omnimem.utils.cache import MultiLevelCache

logger = logging.getLogger(__name__)

# 向后兼容：保留原模块级名称
_is_garbage_query = is_garbage_query
_trim_to_budget = trim_to_budget


class HybridRetriever:
    """混合检索编排：向量 + BM25 + RRF 融合。

    6通道并行检索 + RRF 融合 + 可选 Cross-Encoder Rerank。
    读写锁优化: search() 用读锁（可并行），add() 用写锁（独占）。
    """

    _QUERY_CACHE_TTL = 60.0

    # 垃圾查询白名单（保留类属性以兼容旧测试）
    _GARBAGE_COMMON_WORDS = frozenset(
        {
            "what", "how", "why", "when", "where", "this", "that", "with", "from",
            "have", "will", "would", "could", "should", "about", "just", "like",
            "only", "some", "them", "than", "into", "over", "also", "back", "after",
            "used", "first", "well", "way", "even", "want", "because", "any", "these",
            "most", "make", "know", "time", "year", "good", "work", "qual", "user",
            "http", "html", "json", "api", "url", "app", "log",
        }
    )

    def __init__(
        self,
        vector_backend: str = "chromadb",
        data_dir: Path | None = None,
        enable_reranker: bool = True,
        embedding_model_path: str = "",
        reranker_model_path: str = "",
        recall_timeout_ms: int = 5000,
        recall_strategy: str = "hybrid",
        enable_catalog: bool = True,
        index: Any = None,
        wing_room: Any = None,
        query_cache_ttl: float = 60.0,
        config: Any | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: Any | None = None,
        registry: RetrieverRegistry | None = None,
    ):
        """初始化混合检索引擎。"""
        self._data_dir = data_dir or Path("/tmp/omnimem/retrieval")
        self._config = config
        self._registry = registry or DEFAULT_REGISTRY

        def _cfg(key: str, default: Any) -> Any:
            if config is None:
                return default
            return config.get(key, default)

        self._rrf_k = _cfg("rrf_k", 35)
        self._rrf_min_score = _cfg("rrf_min_score", 0.04)
        self._circuit_breaker_threshold = _cfg("circuit_breaker_threshold", 3)
        self._circuit_breaker_cooldown_seconds = _cfg("circuit_breaker_cooldown_seconds", 60.0)
        self._max_sync_turn_entries = _cfg("max_sync_turn_entries", 1000)
        # 时序重排序配置
        self._temporal_rerank_alpha = _cfg("temporal_rerank_alpha", 0.5)
        self._temporal_decay_lambda = _cfg("temporal_decay_lambda", 0.1)

        # 同义词映射改为实例属性，避免多实例间相互污染
        from omnimem.retrieval.synonym_expander import SynonymExpander
        self._synonym_map: dict[str, list[str]] = SynonymExpander.load_synonyms()

        # 通道注册表：通道名 → (retriever, weight)
        self._channels: dict[str, tuple[Any, float]] = {}
        vec = VectorRetriever(
            backend=vector_backend,
            data_dir=self._data_dir,
            embedding_model_path=embedding_model_path,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
        bm25 = BM25Retriever(data_dir=self._data_dir)
        self.register_channel("vector", vec, weight=_cfg("vector_weight", 3.0))
        self.register_channel("bm25", bm25, weight=_cfg("bm25_weight", 1.0))
        self._vector = vec
        self._bm25 = bm25
        self._rrf = RRFFusion(k=self._rrf_k, min_rrf=self._rrf_min_score)
        self._reranker = CrossEncoderReranker(model_path=reranker_model_path) if enable_reranker else None
        self._recall_timeout_ms = recall_timeout_ms
        self._recall_strategy = recall_strategy
        self._query_cache_ttl = query_cache_ttl
        self._rw_lock = FairReadWriteLock()
        self._ml_cache: MultiLevelCache | None = None
        self._query_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
        try:
            l3_cache_dir = self._data_dir / "cache"
            self._ml_cache = MultiLevelCache(
                l1_size=1000,
                l1_ttl=query_cache_ttl,
                redis_url=None,
                cache_dir=str(l3_cache_dir),
            )
            logger.info("MultiLevelCache initialized (L1+L3, ttl=%.1fs)", query_cache_ttl)
        except Exception as e:
            logger.warning("MultiLevelCache init failed, falling back to dict cache: %s", e)
            self._ml_cache = None
        self._source_weights: dict[str, float] = {}
        self._catalog: Any = None
        self._vector_breaker = CircuitBreaker(
            threshold=self._circuit_breaker_threshold,
            cooldown=float(self._circuit_breaker_cooldown_seconds),
        )
        self._sync_turn_ids: deque[str] = deque()
        if enable_catalog and index and wing_room:
            try:
                from omnimem.retrieval.catalog import CatalogRetriever
                self._catalog = CatalogRetriever(
                    index=index,
                    wing_room=wing_room,
                    vector_retriever=self._vector,
                    bm25_retriever=self._bm25,
                )
            except Exception as e:
                logger.warning("CatalogRetriever init failed (non-fatal): %s", e)

        # 从注册表加载额外检索通道（新增通道无需修改 engine.py）
        self._load_extra_channels_from_registry()

        # 初始化编排器
        self._orchestrator = HybridOrchestrator(self)

    @staticmethod
    def _load_synonyms() -> dict[str, list[str]]:
        """向后兼容：从 SynonymExpander 加载同义词映射。"""
        from omnimem.retrieval.synonym_expander import SynonymExpander
        return SynonymExpander.load_synonyms()

    def register_channel(self, name: str, retriever: Any, weight: float = 1.0) -> None:
        """注册检索通道。"""
        self._channels[name] = (retriever, weight)

    def unregister_channel(self, name: str) -> None:
        """注销检索通道。"""
        self._channels.pop(name, None)

    def _load_extra_channels_from_registry(self) -> None:
        """从注册表自动加载 vector/bm25 以外的检索通道。"""
        for name in self._registry.list_channels():
            if name in self._channels:
                continue
            cls = self._registry.get(name)
            if cls is None:
                continue
            try:
                retriever = cls(data_dir=self._data_dir, config=self._config)
                self.register_channel(name, retriever, weight=1.0)
                logger.info("Loaded retriever channel from registry: %s", name)
            except Exception as e:
                logger.warning("Failed to load retriever channel %s from registry: %s", name, e)

    def embed_text(self, text: str) -> list[float]:
        """Embed text using the vector retriever."""
        return self._vector.embed_text(text)

    def _check_vector_health(self) -> dict[str, Any]:
        try:
            vec_count = self._vector.count()
        except Exception:
            vec_count = -1
        return {"vector_count": vec_count, "breaker_state": self._vector_breaker.state}

    def warmup(self) -> None:
        """预热：启动时预加载模型、ChromaDB 和 BM25 索引。"""
        logger.info("HybridRetriever warmup: starting...")
        t0 = time.time()
        try:
            self._vector.warmup()
            self._bm25.warmup() if hasattr(self._bm25, "warmup") else None
            elapsed = time.time() - t0
            logger.info("HybridRetriever warmup complete in %.1fs", elapsed)
            try:
                vec_count = self._vector.count()
                if vec_count == 0:
                    logger.warning("HybridRetriever: vector index is empty after warmup (count=0)")
            except Exception:
                logger.debug("HybridRetriever warmup: vector count check failed", exc_info=True)
            _emit(f"[OmniMem] 混合检索引擎就绪 ({elapsed:.1f}s)")
        except Exception as e:
            logger.warning("HybridRetriever warmup failed (non-fatal): %s", e)
            _emit(f"[OmniMem] ⚠ 混合检索引擎预热失败: {e}")

    def vector_count(self) -> int:
        """返回向量检索通道中的条目数。"""
        try:
            return self._vector.count()
        except Exception as e:
            logger.warning("HybridRetriever.vector_count failed: %s", e)
            return -1

    def vector_search(self, content: str, top_k: int = 10) -> list[dict[str, Any]]:
        """直接向量检索。"""
        return self._vector.search(content, top_k=top_k)

    def persist_embedding_cache(self) -> None:
        """持久化嵌入缓存到磁盘。"""
        try:
            if hasattr(self._vector, '_embedding_fn') and self._vector._embedding_fn:
                emb_fn = self._vector._embedding_fn
                if hasattr(emb_fn, 'persist'):
                    emb_fn.persist()
        except Exception as e:
            logger.debug("Embedding cache persist skipped: %s", e)

    def delete(self, memory_id: str) -> None:
        """从所有检索通道中移除指定记忆。"""
        self._rw_lock.acquire_write()
        try:
            self._orchestrator.clear_all_cache()
            try:
                self._vector.delete(memory_id)
            except Exception as e:
                logger.warning("Vector delete failed for %s: %s", memory_id, e)
            try:
                self._bm25.delete(memory_id)
            except Exception as e:
                logger.warning("BM25 delete failed for %s: %s", memory_id, e)
        finally:
            self._rw_lock.release_write()

    def add(self, content: str, memory_id: str, metadata: dict[str, Any]) -> None:
        """添加文档到所有检索通道。"""
        self._rw_lock.acquire_write()
        try:
            self._orchestrator.invalidate_cache_by_memory(memory_id)
            self._vector.add(content, memory_id, metadata)
            self._bm25.add(content, memory_id, metadata)
            self._vector.flush()
        finally:
            self._rw_lock.release_write()

    async def async_add(self, content: str, memory_id: str, metadata: dict[str, Any]) -> None:
        """异步添加文档到所有检索通道。"""
        def _do_add() -> None:
            self._rw_lock.acquire_write()
            try:
                self._orchestrator.invalidate_cache_by_memory(memory_id)
                self._vector.add(content, memory_id, metadata)
                self._bm25.add(content, memory_id, metadata)
                self._vector.flush()
            finally:
                self._rw_lock.release_write()

        await _asyncio.to_thread(_do_add)

    def add_batch(self, documents: list[dict[str, Any]]) -> None:
        """批量添加文档到所有检索通道。"""
        self._rw_lock.acquire_write()
        try:
            self._orchestrator.clear_all_cache()
            self._vector.add_batch(documents)
            self._bm25.add_batch(documents)
            self._vector.flush()
            # ★ KMAG优化：数字实体提取 — 为每篇文档生成结构化数值摘要注入 BM25 索引
            _numeric_docs = self._extract_numeric_entities(documents)
            if _numeric_docs:
                self._bm25.add_batch(_numeric_docs)
        finally:
            self._rw_lock.release_write()

    @staticmethod
    def _extract_numeric_entities(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从文档中提取数值型实体（数量、价格、时间等）生成结构化摘要注入检索。"""
        import re as _re

        _PATTERNS = [
            # 数量: "3 tanks", "five projects", "0.5 hours"
            (r'\b(\d+[\d,.]*)\s*(tank|project|item|shirt|boot|pair|kit|baby|child|friend|magazine|subscription|movie|book|episode|day|hour|minute|mile|dollar|percent|pound|ounce|cup|tsp|tbsp|fish|tank|plant|flower|room|bedroom|bathroom|car|bike|computer|phone|tv|monitor|screen|keyboard|mouse|chair|desk|lamp|light|bulb|door|window|painting|photo|video|song|playlist|album|game|level|character|weapon|armor|potion|spell|skill|achievement|trophy|badge)\w*\b', 1),
            # 金额: "$185", "185 dollars", "$720"
            (r'\$\s*(\d+[\d,.]*)|(\d+[\d,.]*)\s*(?:dollar|USD|buck)\w*', 1),
            # 百分比: "50%", "50 percent"
            (r'(\d+[\d,.]*)\s*%\s*|(\d+[\d,.]*)\s*(?:percent|percentage)\w*', 1),
            # 时间/距离: "45 minutes", "3 hours", "10 miles"
            (r'(\d+[\d,.]*)\s*(minute|hour|day|week|month|year|mile|km|kilometer|feet|inch|metre|meter)\w*\b', 1),
            # 价格短语: "worth triple what I paid", "cost $200"
            (r'(?:worth|cost|price|paid|spent|earned|saved|bought|sold|charge)\w*\s*(?:\$?\s*(\d+[\d,.]*|triple|double|half)\b)', 1),
        ]

        _numeric_docs = []
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue
            meta = dict(doc.get("metadata", {}))
            base_id = doc.get("memory_id", "")
            facts = []

            for pattern, grp in _PATTERNS:
                for m in _re.finditer(pattern, content.lower()):
                    val = m.group(1) or m.group(2) or m.group(3) or m.group(4)
                    if val:
                        # Extract the surrounding words for context
                        start = max(0, m.start() - 30)
                        end = min(len(content), m.end() + 30)
                        context = content[start:end].strip()
                        facts.append(f"[NUMERIC] num_val={val} context: {context}")

            if facts:
                _numeric_docs.append({
                    "content": " | ".join(facts),
                    "memory_id": f"{base_id}_numeric",
                    "metadata": {**meta, "_source": "numeric_extract"},
                })

        return _numeric_docs

    def update_metadata(self, memory_id: str, metadata: dict[str, Any]) -> None:
        """更新检索索引中指定条目的 metadata。"""
        self._rw_lock.acquire_write()
        try:
            self._orchestrator.clear_all_cache()
            self._vector.delete(memory_id)
            self._bm25.delete(memory_id)
            content = metadata.pop("_content", "")
            if content:
                self._vector.add(content, memory_id, metadata)
                self._bm25.add(content, memory_id, metadata)
                self._vector.flush()
        finally:
            self._rw_lock.release_write()

    def search(
        self,
        query: str,
        max_tokens: int = 1500,
        mode: str = "rag",
        top_k: int = 40,
        store: Any = None,  # noqa: ARG002
        enable_trace: bool = False,
        channels_only: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """混合检索：向量 + BM25 + RRF 融合。"""
        allowed_channels = set(channels_only) if channels_only else None
        self._rw_lock.acquire_read()
        try:
            return self._orchestrator.search(
                query, max_tokens, mode, top_k,
                allowed_channels=allowed_channels, enable_trace=enable_trace,
            )
        finally:
            self._rw_lock.release_read()

    async def async_search(
        self,
        query: str,
        max_tokens: int = 1500,
        mode: str = "rag",
        top_k: int = 10,
        store: Any = None,  # noqa: ARG002
        enable_trace: bool = False,
        channels_only: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """异步混合检索。"""
        allowed_channels = set(channels_only) if channels_only else None
        self._rw_lock.acquire_read()
        try:
            return await self._orchestrator.async_search(
                query, max_tokens, mode, top_k,
                allowed_channels=allowed_channels, enable_trace=enable_trace,
            )
        finally:
            self._rw_lock.release_read()

    def index_update(self, user_content: str, assistant_content: str) -> None:  # noqa: ARG002
        """后台异步索引更新（从 sync_turn 调用）。"""
        self._rw_lock.acquire_write()
        try:
            self._orchestrator.index_update(user_content, assistant_content)
        finally:
            self._rw_lock.release_write()

    def flush(self) -> None:
        """刷新所有索引。"""
        self._vector.flush()
        self._bm25.flush()

    def _cleanup_sync_turn_entries(self) -> None:
        """清理旧的 sync_turn 条目，防止索引膨胀。"""
        self._orchestrator.cleanup_sync_turn_entries()

    def set_source_weights(self, weights: dict[str, float]) -> None:
        """设置动态来源权重。"""
        self._source_weights = dict(weights)

    def invalidate_cache(self) -> None:
        """清除查询结果缓存。"""
        self._orchestrator.clear_all_cache()

    def shutdown(self) -> None:
        """关闭检索器并释放线程池资源。"""
        self._orchestrator.shutdown()

    @property
    def bm25_document_count(self) -> int:
        """BM25 已索引文档数。"""
        return self._bm25.document_count

    def rebuild_bm25_from_entries(self, entries: list[dict[str, Any]]) -> int:
        """从索引条目重建 BM25 检索通道。"""
        return self._orchestrator.rebuild_bm25_from_entries(entries)

    def rebuild_all_from_entries(self, entries: list[dict[str, Any]]) -> dict[str, int]:
        """全量重建向量+BM25检索索引。"""
        self._rw_lock.acquire_write()
        try:
            return self._orchestrator.rebuild_all_from_entries(entries)
        finally:
            self._rw_lock.release_write()


__all__ = [
    "HybridRetriever",
    "_is_garbage_query",
    "_trim_to_budget",
]
