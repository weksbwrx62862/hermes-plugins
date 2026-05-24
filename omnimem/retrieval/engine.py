"""HybridRetriever — 混合检索编排。

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
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from omnimem.retrieval.bm25 import BM25Retriever
from omnimem.retrieval.reranker import CrossEncoderReranker
from omnimem.retrieval.rrf import RRFFusion
from omnimem.retrieval.vector import VectorRetriever
from omnimem.protocols import RetrieverProtocol
from omnimem.retrieval.vector_store import _emit

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """三态熔断器 — 向量检索连续故障时自动降级到纯 BM25。

    状态机：
      CLOSED → (阈值次故障) → OPEN → (冷却后) → HALF_OPEN → (成功) → CLOSED
                                                    HALF_OPEN → (失败) → OPEN

    使用：
        breaker = CircuitBreaker(threshold=3, cooldown=60)
        result = breaker.call(lambda: risky_op(), fallback=lambda: safe_op())
    """

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, threshold: int = 3, cooldown: float = 60.0, on_recover=None):
        self._state = self.CLOSED
        self._failures = 0
        self._threshold = threshold
        self._cooldown = cooldown
        self._last_failure_time = 0.0
        self._on_recover = on_recover

    @property
    def state(self) -> str:
        return self._state

    def call(self, fn, fallback):
        """执行 fn()，故障时返回 fallback()。"""
        now = time.time()
        if self._state == self.OPEN:
            if now - self._last_failure_time > self._cooldown:
                self._state = self.HALF_OPEN
                logger.info("CircuitBreaker: OPEN→HALF_OPEN (cooldown elapsed)")
            else:
                logger.warning("CircuitBreaker: OPEN, circuit open (%.1fs remaining)",
                               self._cooldown - (now - self._last_failure_time))
                return fallback()
        try:
            result = fn()
            # 成功 — 恢复
            if self._state == self.HALF_OPEN:
                self._state = self.CLOSED
                self._failures = 0
                logger.info("CircuitBreaker: HALF_OPEN→CLOSED (recovered)")
                if self._on_recover:
                    self._on_recover()
            elif self._failures > 0:
                self._failures = 0
            return result
        except Exception:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self._threshold:
                self._state = self.OPEN
                logger.error("CircuitBreaker: CLOSED→OPEN (%d consecutive failures)", self._failures)
            return fallback()

    def reset(self) -> None:
        """手动重置熔断器。"""
        self._state = self.CLOSED
        self._failures = 0

    def record_failure(self) -> None:
        """记录一次故障，达到阈值时自动 OPEN。"""
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self._threshold:
            self._state = self.OPEN
            logger.error("CircuitBreaker: CLOSED→OPEN (%d consecutive failures)", self._failures)

    def record_success(self) -> None:
        """记录一次成功，HALF_OPEN→CLOSED 或清零计数器。"""
        if self._state == self.HALF_OPEN:
            self._state = self.CLOSED
            self._failures = 0
            logger.info("CircuitBreaker: HALF_OPEN→CLOSED (recovered)")
            if self._on_recover:
                self._on_recover()
        elif self._failures > 0:
            self._failures = 0

    def should_skip(self) -> bool:
        """OPEN 且未冷却时返回 True，调用方应跳过向量检索。"""
        if self._state != self.OPEN:
            return False
        if time.time() - self._last_failure_time > self._cooldown:
            self._state = self.HALF_OPEN
            logger.info("CircuitBreaker: OPEN→HALF_OPEN (cooldown elapsed)")
            return False
        return True


class _ReadWriteLock:
    """简单的读写锁实现。

    多个读者可并行持有读锁；写者必须独占。
    写者优先策略：有写者等待时，新读者排队，防止写者饥饿。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._readers = 0
        self._writers = 0
        self._writer_waiting = 0

    def acquire_read(self) -> None:
        with self._cond:
            while self._writers > 0 or self._writer_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writer_waiting += 1
            while self._readers > 0 or self._writers > 0:
                self._cond.wait()
            self._writer_waiting -= 1
            self._writers += 1

    def release_write(self) -> None:
        with self._cond:
            self._writers -= 1
            self._cond.notify_all()

    def __enter__(self) -> "_ReadWriteLock":
        self.acquire_write()
        return self

    def __exit__(self, *args: object) -> None:
        self.release_write()


class HybridRetriever:
    """混合检索编排：向量 + BM25 + RRF 融合。

    6通道并行检索 + RRF 融合 + 可选 Cross-Encoder Rerank:
      1. 向量检索 (ChromaDB) — 语义相似度
      2. BM25 关键词检索 — 词袋匹配 + 同义词扩展
      3. 目录检索 (Wing/Hall/Room 结构过滤) — Phase 3
      4. 实体提升 — Phase 3
      5. 时间检索 — Phase 3
      6. 图谱检索 — Phase 3

    读写锁优化: search() 用读锁（可并行），add() 用写锁（独占），
    避免后台 queue_prefetch 写入阻塞主线程 prefetch 搜索。

    查询缓存: 相同查询 60s 内复用结果，写入时清除缓存。
    """

    _QUERY_CACHE_TTL = 60.0

    _SYNONYM_MAP: dict[str, list[str]] = {}

    # ★ 类级别垃圾查询白名单：避免每次 _is_garbage_query 调用时重建集合
    _GARBAGE_COMMON_WORDS = frozenset(
        {
            "test",
            "what",
            "how",
            "why",
            "when",
            "where",
            "this",
            "that",
            "with",
            "from",
            "have",
            "will",
            "would",
            "could",
            "should",
            "about",
            "just",
            "like",
            "only",
            "some",
            "them",
            "than",
            "into",
            "over",
            "also",
            "back",
            "after",
            "used",
            "first",
            "well",
            "way",
            "even",
            "want",
            "because",
            "any",
            "these",
            "most",
            "make",
            "know",
            "time",
            "year",
            "good",
            "work",
            "qual",
            "data",
            "info",
            "user",
            "name",
            "code",
            "file",
            "http",
            "html",
            "json",
            "api",
            "url",
            "app",
            "log",
        }
    )

    def __init__(
        self,
        vector_backend: str = "chromadb",
        data_dir: Path | None = None,
        enable_reranker: bool = False,
        embedding_model_path: str = "",
        reranker_model_path: str = "",
        recall_timeout_ms: int = 5000,
        recall_strategy: str = "hybrid",
        enable_catalog: bool = True,
        index: Any = None,
        wing_room: Any = None,
        query_cache_ttl: float = 60.0,
    ):
        """初始化混合检索引擎。

        Args:
            vector_backend: 向量存储后端 (chromadb/qdrant/pgvector)
            data_dir: 检索数据存储目录
            enable_reranker: 是否启用 Cross-Encoder 重排序
            embedding_model_path: 嵌入模型本地路径
            reranker_model_path: 重排序模型本地路径
            recall_timeout_ms: 召回整体超时时间（毫秒），超时后自动降级
            recall_strategy: 召回策略 - hybrid(默认)/keyword(纯BM25)/embedding(纯向量)
            enable_catalog: 是否启用目录递归检索通道
            index: ThreeLevelIndex 实例
            wing_room: WingRoomManager 实例
        """
        self._data_dir = data_dir or Path("/tmp/omnimem/retrieval")
        synonym_map = self._load_synonyms()
        if synonym_map:
            self._SYNONYM_MAP = synonym_map
        # ★ 通道注册表：通道名 → (retriever, weight)
        self._channels: dict[str, tuple[RetrieverProtocol, float]] = {}
        vec = VectorRetriever(backend=vector_backend, data_dir=self._data_dir, embedding_model_path=embedding_model_path)
        bm25 = BM25Retriever(data_dir=self._data_dir)
        self.register_channel("vector", vec, weight=3.0)
        self.register_channel("bm25", bm25, weight=1.0)
        self._vector = vec  # 向后兼容属性引用
        self._bm25 = bm25   # 向后兼容属性引用
        self._rrf = RRFFusion(k=60, min_rrf=0.035)
        self._reranker = CrossEncoderReranker(model_path=reranker_model_path) if enable_reranker else None
        self._recall_timeout_ms = recall_timeout_ms
        self._recall_strategy = recall_strategy
        self._query_cache_ttl = query_cache_ttl
        # ★ 读写锁替代全局互斥锁
        self._rw_lock = _ReadWriteLock()
        # ★ 查询结果缓存：key → (results, timestamp)
        self._query_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
        # ★ P1方案四：动态来源权重（由 FeedbackCollector 驱动）
        self._source_weights: dict[str, float] = {}
        # ★ OPT: 目录递归检索通道（内化 OpenViking find()）
        self._catalog: Any = None
        # ★ OPT: 向量检索熔断器 — 连续故障时自动降级纯BM25
        self._vector_breaker = CircuitBreaker(
            threshold=3,
            cooldown=60.0,
        )
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

    @staticmethod
    def _load_synonyms() -> dict[str, list[str]]:
        try:
            import json
            from pathlib import Path
            synonyms_path = Path(__file__).parent.parent / "config" / "synonyms.json"
            if synonyms_path.exists():
                with open(synonyms_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    result = {}
                    for k, v in data.items():
                        if isinstance(v, list):
                            result[k] = v
                        elif isinstance(v, str):
                            result[k] = [v]
                    if result:
                        logger.info("Loaded %d synonym entries from %s", len(result), synonyms_path)
                        return result
            logger.warning("synonyms.json not found or empty at %s, synonym expansion disabled", synonyms_path)
        except Exception as e:
            logger.warning("Failed to load synonyms.json: %s, synonym expansion disabled", e)
        return {}

    def register_channel(self, name: str, retriever: RetrieverProtocol, weight: float = 1.0) -> None:
        """注册检索通道。

        Args:
            name: 通道名称（如 "vector"、"bm25"）
            retriever: 符合 RetrieverProtocol 的检索器实例
            weight: RRF 融合权重，默认 1.0
        """
        self._channels[name] = (retriever, weight)

    def unregister_channel(self, name: str) -> None:
        """注销检索通道。

        Args:
            name: 要注销的通道名称
        """
        self._channels.pop(name, None)

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
                pass
            _emit(f"[OmniMem] 混合检索引擎就绪 ({elapsed:.1f}s)")
        except Exception as e:
            logger.warning("HybridRetriever warmup failed (non-fatal): %s", e)
            _emit(f"[OmniMem] ⚠ 混合检索引擎预热失败: {e}")

    def delete(self, memory_id: str) -> None:
        """从所有检索通道中移除指定记忆（软故障，用于归档清理）。"""
        self._rw_lock.acquire_write()
        try:
            self._query_cache.clear()
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
            self._query_cache.clear()
            self._vector.add(content, memory_id, metadata)
            self._bm25.add(content, memory_id, metadata)
            # ★ R25修复Minor-3：写入后立即 flush 确保向量索引可搜
            self._vector.flush()
        finally:
            self._rw_lock.release_write()

    def add_batch(self, documents: list[dict[str, Any]]) -> None:
        """批量添加文档到所有检索通道。

        Args:
            documents: 文档列表，每项需包含 content 和 memory_id 字段
        """
        self._rw_lock.acquire_write()
        try:
            self._query_cache.clear()
            self._vector.add_batch(documents)
            self._bm25.add_batch(documents)
            self._vector.flush()
        finally:
            self._rw_lock.release_write()

    def update_metadata(self, memory_id: str, metadata: dict[str, Any]) -> None:
        """更新检索索引中指定条目的 metadata（如 wing/privacy）。

        通过 delete + re-add 实现，因为 ChromaDB/BM25 不支持原地更新 metadata。
        需要同时传入 content 以便重新索引。

        Args:
            memory_id: 记忆 ID
            metadata: 新的 metadata 字典（必须包含 content 或调用方需先获取 content）
        """
        self._rw_lock.acquire_write()
        try:
            self._query_cache.clear()
            # 从向量索引删除旧条目
            self._vector.delete(memory_id)
            # 从 BM25 索引删除旧条目
            self._bm25.delete(memory_id)
            # 用新 metadata 重新添加
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
        top_k: int = 10,
        store: Any = None,  # noqa: ARG002
        enable_trace: bool = False,  # ★ 新增：是否记录检索轨迹
        channels_only: list[str] | None = None,  # ★ 新增：仅使用指定通道检索
    ) -> list[dict[str, Any]]:
        """混合检索：向量 + BM25 + RRF 融合。

        RRF 融合检索流程:
          1. 查询缓存检查（60s TTL）
          2. 垃圾查询检测 → 限制 top_k 和 max_tokens
          3. 向量检索通道: ChromaDB 语义搜索
          4. BM25 检索通道: 关键词搜索 + 同义词扩展
          5. 目录递归检索通道: Wing/Hall/Room 定向搜索
          6. RRF 融合: 合并三路结果，数据量自适应 min_rrf 阈值
          7. 垃圾查询二次验证: 低分结果过滤
          8. 可选 Cross-Encoder Rerank
          9. Token 预算裁剪

        mode:
          rag: 快速向量+BM25混合检索（毫秒级）
          llm: 深度检索 — 更多结果 + store 内容搜索补充通道

        Args:
            query: 检索查询文本
            max_tokens: 返回结果的最大 token 预算
            mode: 检索模式 (rag/llm)
            top_k: 返回结果数
            store: 存储实例（用于 llm 模式）
            enable_trace: 是否记录检索轨迹（默认 False）
            channels_only: 仅使用指定名称的通道检索，如 ["vector", "bm25"]；未指定的通道跳过（不报错）

        Returns:
            检索结果列表，每项包含 content/memory_id/score/metadata 等字段
        """
        # ★ 新增：创建检索轨迹记录器
        from omnimem.retrieval.trace import SearchTrace
        trace = SearchTrace(query) if enable_trace else None

        # ★ 单通道测试：channels_only 指定时仅使用指定通道检索
        _allowed_channels = set(channels_only) if channels_only else None

        is_garbage = _is_garbage_query(query)

        try:
            doc_count = self._vector.count()
        except Exception:
            doc_count = 0

        if is_garbage:
            top_k = min(top_k, 2)
            max_tokens = min(max_tokens, 200)
            if doc_count >= 100:
                top_k = 1
            elif doc_count >= 30:
                top_k = min(top_k, 1)

        if mode == "llm" and not is_garbage:
            top_k = max(top_k, 20)
            max_tokens = max(max_tokens, 3000)

        self._rw_lock.acquire_read()
        try:
            # ★ 查询缓存检查（在读锁内，避免与 add 清缓存竞态）
            cache_key = f"{query}|{max_tokens}|{mode}|{top_k}"
            now = time.time()
            if cache_key in self._query_cache:
                cached_results, cached_time = self._query_cache[cache_key]
                if now - cached_time < self._query_cache_ttl:
                    logger.debug("HybridRetriever query cache hit: %s", query[:50])
                    return cached_results

            # ★ 动态通道并行检索，降低搜索延迟
            # ★ OPT: 按 recall_strategy 分流 + 超时降级
            channel_results: dict[str, list[dict[str, Any]]] = {}

            if self._recall_strategy == "keyword":
                # 纯关键词模式：仅运行 BM25 通道
                if "bm25" in self._channels and (not _allowed_channels or "bm25" in _allowed_channels):
                    channel_results["bm25"] = self._bm25_search(query, top_k)
            elif self._recall_strategy == "embedding":
                # 纯向量模式：仅运行向量通道，熔断器保护
                if "vector" in self._channels and (not _allowed_channels or "vector" in _allowed_channels):
                    if self._vector_breaker.should_skip():
                        logger.warning("CircuitBreaker OPEN: skipping vector search, no results")
                        _emit("[OmniMem] ⚠ 向量检索不可用，已降级到关键词模式")
                    else:
                        try:
                            channel_results["vector"] = self._vector_search(query, top_k)
                            self._vector_breaker.record_success()
                        except Exception:
                            self._vector_breaker.record_failure()
            else:
                # hybrid 模式：动态通道并行检索 + 超时降级
                timeout_sec = self._recall_timeout_ms / 1000.0
                futures: dict[str, Any] = {}
                n_workers = len(self._channels) + (1 if self._catalog else 0)

                with ThreadPoolExecutor(max_workers=max(n_workers, 1)) as executor:
                    # ★ 遍历注册通道，并行提交检索任务
                    for name, (retriever, _weight) in self._channels.items():
                        # channels_only 过滤
                        if _allowed_channels and name not in _allowed_channels:
                            continue
                        # 向量通道：熔断器保护
                        if name == "vector" and self._vector_breaker.should_skip():
                            logger.warning("CircuitBreaker OPEN: skipping vector search, degrading to BM25+Catalog only")
                            _emit("[OmniMem] ⚠ 向量检索不可用，已降级到关键词模式")
                            continue
                        # BM25 通道：使用含同义词扩展的搜索方法
                        if name == "bm25":
                            futures[name] = executor.submit(self._bm25_search, query, top_k)
                        else:
                            futures[name] = executor.submit(retriever.search, query, top_k=top_k)

                    # 目录检索通道（未纳入通道注册表，保持独立）
                    if self._catalog and (not _allowed_channels or "catalog" in _allowed_channels):
                        futures["catalog"] = executor.submit(self._catalog_search, query, top_k)

                    # ★ 收集各通道结果
                    for name, future in futures.items():
                        try:
                            channel_results[name] = future.result(timeout=timeout_sec)
                            if name == "vector":
                                self._vector_breaker.record_success()
                        except (TimeoutError, Exception) as e:
                            channel_results[name] = []
                            future.cancel()
                            if name == "vector":
                                self._vector_breaker.record_failure()
                            logger.warning("%s search timeout/degraded (%dms): %s",
                                           name, self._recall_timeout_ms, e)

                    # 所有通道超时时关闭线程池
                    if not any(channel_results.values()):
                        executor.shutdown(wait=False, cancel_futures=True)

                # ★ 记录各通道检索轨迹
                if trace:
                    for ch_name, ch_results in channel_results.items():
                        trace.add_step("channel_search", channel=ch_name,
                                       output_count=len(ch_results))

                # 降级日志
                active = [n for n, r in channel_results.items() if r]
                if len(active) == 1 and active[0] != "vector":
                    logger.info("Recall degraded: only %s channel has results", active[0])
                    _emit("[OmniMem] ⚠ 向量检索不可用，已降级到关键词模式")

            results = self._rrf_fuse(
                query,
                channel_results,
                is_garbage=is_garbage,
                doc_count=doc_count,
                top_k=top_k,
                max_tokens=max_tokens,
            )

            # ★ 记录 RRF 融合轨迹
            if trace:
                trace.add_step("rrf_fuse",
                               input_count=sum(len(r) for r in channel_results.values()),
                               output_count=len(results))
            
            # ★ 过滤 sync_turn 条目（对话片段不应污染记忆检索结果）
            results = [r for r in results if r.get("source") != "sync_turn"]

            # ★ 低召回类型补充：reasoning/action 关键词密度低，额外扩展查询
            results = self._supplement_low_recall_types(query, results, top_k)

            # ★ 类型权重：reasoning/action 提高 1.3x
            results = self._apply_type_boost(results)

            # ★ 缓存搜索结果
            self._query_cache[cache_key] = (results, now)
            
            # ★ 新增：将轨迹附加到最后一个结果
            if trace and results:
                results[-1]["_trace"] = trace.to_dict()
            
            return results
        finally:
            self._rw_lock.release_read()

    async def async_search(
        self,
        query: str,
        max_tokens: int = 1500,
        mode: str = "rag",
        top_k: int = 10,
        store: Any = None,
        enable_trace: bool = False,
        channels_only: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """异步混合检索：使用 asyncio 并行执行各通道检索。

        与同步 search() 逻辑一致，但使用 asyncio.gather() 并行执行
        向量检索和 BM25 检索，各通道用 asyncio.to_thread() 包装为异步。
        适用于 asyncio 事件循环中非阻塞调用。

        Args:
            query: 检索查询文本
            max_tokens: 返回结果的最大 token 预算
            mode: 检索模式 (rag/llm)
            top_k: 返回结果数
            store: 存储实例（用于 llm 模式）
            enable_trace: 是否记录检索轨迹
            channels_only: 仅使用指定名称的通道检索

        Returns:
            检索结果列表
        """
        import asyncio as _asyncio

        from omnimem.retrieval.trace import SearchTrace

        trace = SearchTrace(query) if enable_trace else None
        _allowed_channels = set(channels_only) if channels_only else None

        is_garbage = _is_garbage_query(query)

        try:
            doc_count = self._vector.count()
        except Exception:
            doc_count = 0

        if is_garbage:
            top_k = min(top_k, 2)
            max_tokens = min(max_tokens, 200)
            if doc_count >= 100:
                top_k = 1
            elif doc_count >= 30:
                top_k = min(top_k, 1)

        if mode == "llm" and not is_garbage:
            top_k = max(top_k, 20)
            max_tokens = max(max_tokens, 3000)

        self._rw_lock.acquire_read()
        try:
            cache_key = f"{query}|{max_tokens}|{mode}|{top_k}"
            now = time.time()
            if cache_key in self._query_cache:
                cached_results, cached_time = self._query_cache[cache_key]
                if now - cached_time < self._query_cache_ttl:
                    logger.debug("HybridRetriever async query cache hit: %s", query[:50])
                    return cached_results

            channel_results: dict[str, list[dict[str, Any]]] = {}

            if self._recall_strategy == "keyword":
                if "bm25" in self._channels and (not _allowed_channels or "bm25" in _allowed_channels):
                    channel_results["bm25"] = await _asyncio.to_thread(self._bm25_search, query, top_k)
            elif self._recall_strategy == "embedding":
                if "vector" in self._channels and (not _allowed_channels or "vector" in _allowed_channels):
                    if self._vector_breaker.should_skip():
                        logger.warning("CircuitBreaker OPEN: skipping async vector search")
                        _emit("[OmniMem] ⚠ 向量检索不可用，已降级到关键词模式")
                    else:
                        try:
                            channel_results["vector"] = await _asyncio.to_thread(
                                self._vector_search, query, top_k
                            )
                            self._vector_breaker.record_success()
                        except Exception:
                            self._vector_breaker.record_failure()
            else:
                # hybrid 模式：asyncio.gather 并行检索
                async_tasks: dict[str, Any] = {}

                for name, (retriever, _weight) in self._channels.items():
                    if _allowed_channels and name not in _allowed_channels:
                        continue
                    if name == "vector" and self._vector_breaker.should_skip():
                        logger.warning("CircuitBreaker OPEN: skipping async vector search, degrading to BM25+Catalog only")
                        _emit("[OmniMem] ⚠ 向量检索不可用，已降级到关键词模式")
                        continue
                    if name == "bm25":
                        async_tasks[name] = _asyncio.to_thread(self._bm25_search, query, top_k)
                    else:
                        async_tasks[name] = _asyncio.to_thread(retriever.search, query, top_k=top_k)

                if self._catalog and (not _allowed_channels or "catalog" in _allowed_channels):
                    async_tasks["catalog"] = _asyncio.to_thread(self._catalog_search, query, top_k)

                # 并行执行所有通道检索
                if async_tasks:
                    task_names = list(async_tasks.keys())
                    task_coros = list(async_tasks.values())
                    results_list = await _asyncio.gather(*task_coros, return_exceptions=True)
                    for name, result in zip(task_names, results_list):
                        if isinstance(result, Exception):
                            channel_results[name] = []
                            if name == "vector":
                                self._vector_breaker.record_failure()
                            logger.warning("async %s search failed: %s", name, result)
                        else:
                            channel_results[name] = result
                            if name == "vector":
                                self._vector_breaker.record_success()

                if trace:
                    for ch_name, ch_results in channel_results.items():
                        trace.add_step("channel_search", channel=ch_name,
                                       output_count=len(ch_results))

                active = [n for n, r in channel_results.items() if r]
                if len(active) == 1 and active[0] != "vector":
                    logger.info("Async recall degraded: only %s channel has results", active[0])
                    _emit("[OmniMem] ⚠ 向量检索不可用，已降级到关键词模式")

            results = self._rrf_fuse(
                query,
                channel_results,
                is_garbage=is_garbage,
                doc_count=doc_count,
                top_k=top_k,
                max_tokens=max_tokens,
            )

            if trace:
                trace.add_step("rrf_fuse",
                               input_count=sum(len(r) for r in channel_results.values()),
                               output_count=len(results))

            results = [r for r in results if r.get("source") != "sync_turn"]
            results = self._supplement_low_recall_types(query, results, top_k)
            results = self._apply_type_boost(results)

            self._query_cache[cache_key] = (results, now)

            if trace and results:
                results[-1]["_trace"] = trace.to_dict()

            return results
        finally:
            self._rw_lock.release_read()

    def _vector_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """向量检索通道。"""
        return self._vector.search(query, top_k=top_k)

    def _bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """BM25 检索通道（含同义词扩展）。

        ★ 同义词扩展 BM25 查询：弥补词袋模型的语义鸿沟（QUAL-3修复）
        注意：单字会被 _tokenize 丢弃，所以用2+字词
        ★ 扩展策略：上位词↔下位词 双向扩展 + 品种级细粒度覆盖
        """
        bm25_results = self._bm25.search(query, top_k=top_k)
        for key, synonyms in self._SYNONYM_MAP.items():
            if key in query:
                for syn in synonyms:
                    expanded = query.replace(key, syn)
                    expanded_results = self._bm25.search(expanded, top_k=top_k)
                    existing_ids = {r.get("memory_id", "") for r in bm25_results}
                    for r in expanded_results:
                        if r.get("memory_id", "") not in existing_ids:
                            bm25_results.append(r)
                            existing_ids.add(r.get("memory_id", ""))
        return bm25_results

    def _catalog_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """目录递归检索通道（内化 OpenViking find()）。"""
        if not self._catalog:
            return []
        try:
            return self._catalog.search(query, top_k=top_k)
        except Exception as e:
            logger.warning("Catalog search failed (non-fatal): %s", e)
            return []

    def _rrf_fuse(
        self,
        query: str,
        channel_results: dict[str, list[dict[str, Any]]],
        *,
        is_garbage: bool,
        doc_count: int,
        top_k: int,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """RRF 融合 + 数据量自适应阈值 + 垃圾查询二次验证 + Rerank + Token 裁剪。"""
        # 数据量自适应 min_rrf
        # ★ R22修复：RRF 分数数学上限 = weight/(k+rank)
        # 向量 rank1: 3.0/61=0.0492, +bonus(x1.5)=0.0738
        # BM25 rank1: 1.0/61=0.0164
        # 双路最优: 0.0738+0.0164=0.0902
        # 阈值不能超过数学上限，否则所有结果被过滤
        adaptive_min_rrf = 0.035
        if doc_count >= 200 or doc_count >= 100:
            adaptive_min_rrf = 0.04
        elif doc_count >= 50 or doc_count >= 20:
            adaptive_min_rrf = 0.035
        # ★ R25优化：语料过少时（<10条）降低阈值
        # 小语料下 BM25 IDF 不可靠，单通道匹配的 RRF 分数低
        # 如仅 BM25 rank1: 1.0/61=0.0164 < 0.035 会被误过滤
        elif doc_count < 10:
            adaptive_min_rrf = 0.01

        # ★ 从通道注册表动态获取权重，构建结果列表
        base_weights: list[float] = []
        result_lists: list[list[dict[str, Any]]] = []
        active_names: list[str] = []
        for name, results in channel_results.items():
            if not results:
                continue
            if name in self._channels:
                weight = self._channels[name][1]
            elif name == "catalog":
                weight = 2.0
            else:
                weight = 1.0
            base_weights.append(weight)
            result_lists.append(results)
            active_names.append(name)

        # ★ 多通道降级：仅一个通道有结果时降低阈值
        active_channels = len(result_lists)

        if active_channels <= 1:
            adaptive_min_rrf = min(adaptive_min_rrf, 0.01)
            if active_channels == 1:
                logger.warning("RRF degraded: only %s channel has results, weight=%.1f",
                             active_names[0], base_weights[0] if base_weights else 0)

        # ★ P1方案四：应用动态来源权重（基于 FeedbackCollector 的 CTR 统计）
        if self._source_weights:
            for i, name in enumerate(active_names):
                base_weights[i] *= self._source_weights.get(name, 1.0)

        if not result_lists:
            return []

        fused = self._rrf.merge(
            result_lists,
            min_rrf=adaptive_min_rrf,
            weights=base_weights,
        )

        # ★ QUAL-1 R14/R24修复：垃圾查询结果二次验证
        # 垃圾查询（如 zzzzzxyz123）不应返回任何结果
        if is_garbage and fused:
            # ★ R24修复：垃圾查询直接清空结果
            # 之前的阈值 0.02 过于宽松，向量搜索总返回 top_k 个最近邻，
            # 即使查询无意义，score 仍可能达到 0.05+
            fused = []

        # 可选 Rerank
        if self._reranker and len(fused) > 3:
            fused = self._reranker.rerank(query, fused, top_k=top_k)

        # Token 预算裁剪
        return _trim_to_budget(fused, max_tokens)

    def index_update(self, user_content: str, assistant_content: str) -> None:  # noqa: ARG002
        """后台异步索引更新（从 sync_turn 调用）。

        ★ 只存提炼后的用户消息核心事实，不存 "User:...Assistant:..." 对话原文。
        对话原文被存入 BM25 会导致 recall 返回大段对话片段而非事实。
        """
        import re
        import uuid

        clean_user = re.sub(
            r"### Relevant Memories(?:\s*\(prefetched\))?\s*\n.*"
            r"(?=\n(?!- )|\Z)",
            "",
            user_content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        clean_user = re.sub(r"^- \[cached\].*$", "", clean_user, flags=re.MULTILINE).strip()

        # ★ 只存用户消息的提炼摘要，不存对话原文
        # 截取用户消息的核心部分（前200字），不拼 Assistant 回复
        content = clean_user[:200].strip() if clean_user else ""
        if not content or len(content) < 5:
            return  # 空或极短内容不索引

        idx_id = f"sync-{uuid.uuid4().hex[:8]}"
        self._rw_lock.acquire_write()
        try:
            self._query_cache.clear()
            self._vector.add(content, memory_id=idx_id, metadata={"source": "sync_turn"})
            self._bm25.add(content, memory_id=idx_id, metadata={"source": "sync_turn"})
        finally:
            self._rw_lock.release_write()

    def flush(self) -> None:
        """刷新所有索引。"""
        self._vector.flush()
        self._bm25.flush()

    def set_source_weights(self, weights: dict[str, float]) -> None:
        """设置动态来源权重（由 FeedbackCollector 驱动）。

        Args:
            weights: 来源类型 → 权重映射，如 {"vector": 3.5, "bm25": 0.8}
        """
        self._source_weights = dict(weights)

    # ── 类型加权 ──

    # reasoning/action 类型的记忆包含高价值信息但关键词密度低，
    # 需要提高权重避免被 fact/preference 等高频词类型淹没。
    _TYPE_BOOST: dict[str, float] = {
        "reasoning": 1.3,
        "action": 1.3,
        "correction": 1.1,
    }

    @classmethod
    def _apply_type_boost(cls, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """对 reasoning/action/correction 类型应用分数加权。"""
        for r in results:
            mem_type = r.get("type", "")
            boost = cls._TYPE_BOOST.get(mem_type, 1.0)
            if boost > 1.0:
                current_score = r.get("score", r.get("rrf_score", 0))
                r["score"] = round(current_score * boost, 5)
                r["type_boost"] = boost
        # 按加权后分数重新排序
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    def _supplement_low_recall_types(
        self, query: str, results: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """对 reasoning/action 类型做扩展查询，弥补关键词密度低导致的召回不足。

        reasoning/action 类型记忆通常包含长句叙述而非密集关键词，
        在 BM25 中容易被 fact/preference 等类型淹没。
        当主检索结果中这些类型<2条时，用类型前缀做补充搜索。
        """
        existing_ids = {r.get("memory_id", "") for r in results}
        type_counts = {}
        for r in results:
            t = r.get("type", "")
            type_counts[t] = type_counts.get(t, 0) + 1

        need_reasoning = type_counts.get("reasoning", 0) < 2
        need_action = type_counts.get("action", 0) < 2

        if not need_reasoning and not need_action:
            return results

        # 用 enriched 前缀做 BM25 扩展搜索
        extra_queries = []
        if need_reasoning:
            extra_queries.append(f"[教训/经验/踩坑] {query}")
        if need_action:
            extra_queries.append(f"[Agent行为/工具调用] {query}")

        for eq in extra_queries:
            extra_results = self._bm25.search(eq, top_k=5)
            for r in extra_results:
                mid = r.get("memory_id", "")
                if mid in existing_ids:
                    continue
                mem_type = r.get("type", "")
                if mem_type not in ("reasoning", "action"):
                    continue
                r["_source"] = "type_supplement"
                r["score"] = r.get("score", 0) * 0.8  # 略低于主检索
                results.append(r)
                existing_ids.add(mid)

        return results

    def invalidate_cache(self) -> None:
        """清除查询结果缓存（写入时调用）。"""
        self._query_cache.clear()

    @property
    def bm25_document_count(self) -> int:
        """BM25 已索引文档数（用于判断是否需要跨会话重建）。"""
        return self._bm25.document_count

    def rebuild_bm25_from_entries(self, entries: list[dict[str, Any]]) -> int:
        """从索引条目重建 BM25 检索通道（跨会话持久化恢复）。

        若 BM25 已从磁盘缓存恢复且有数据，做增量更新：找出缺失的
        memory_id 并追加，而不是跳过重建。这解决磁盘缓存落后于
        SQLite 索引导致的 BM25 召回缺失问题。

        Args:
            entries: 索引条目列表，需含 content/summary, memory_id, type, scope 等字段

        Returns:
            新增的条目数
        """
        if self._bm25.cache_loaded and self._bm25.document_count > 0:
            existing_ids = {doc.get("memory_id", "") for doc in self._bm25._documents}
            new_entries = [e for e in entries if e.get("memory_id", "") not in existing_ids]
            if not new_entries:
                logger.warning(
                    "BM25 already has all %d entries from disk cache, skipping",
                    self._bm25.document_count,
                )
                return 0
            logger.info(
                "BM25 has %d from cache, adding %d new entries",
                self._bm25.document_count,
                len(new_entries),
            )
            count = 0
            for entry in new_entries:
                content = entry.get("content", "") or entry.get("summary", "")
                memory_id = entry.get("memory_id", "")
                if content and memory_id:
                    mem_type = entry.get("type", "fact")
                    room = entry.get("room", "")
                    enriched = _enrich_for_rebuild(content, mem_type, room)
                    self._bm25.add_document(memory_id, enriched)
                    count += 1
            self._bm25.flush()
            return count
        return self._bm25.rebuild_from_entries(entries)

    def rebuild_all_from_entries(self, entries: list[dict[str, Any]]) -> dict[str, int]:
        """全量重建向量+BM25检索索引（解决历史向量退化问题）。

        ★ P1修复QUAL-2：历史向量索引随时间衰减（ChromaDB 内部优化/碎片化），
        定期全量重建可恢复召回率。

        Args:
            entries: 索引条目列表，需含 content/memory_id/type/wing/privacy 等字段

        Returns:
            重建统计 {"vector": count, "bm25": count}
        """
        self._rw_lock.acquire_write()
        try:
            self._query_cache.clear()
            vec_count = 0
            bm25_count = 0
            for entry in entries:
                mid = entry.get("memory_id", "")
                content = entry.get("content", "")
                if not mid or not content:
                    continue
                metadata = {
                    "memory_id": mid,
                    "type": entry.get("type", "fact"),
                    "wing": entry.get("wing", ""),
                    "privacy": entry.get("privacy", "personal"),
                    "confidence": entry.get("confidence", 3),
                }
                mem_type = entry.get("type", "fact")
                room = entry.get("room", "")
                # ★ R34修复Minor-3：重建时也为 secret/skill/procedural 附加可搜索描述
                enriched = _enrich_for_rebuild(content, mem_type, room)
                try:
                    self._vector.delete(mid)
                except Exception as e:
                    logger.warning("Vector delete failed in rebuild for %s: %s", mid, e)
                try:
                    self._bm25.delete(mid)
                except Exception as e:
                    logger.warning("BM25 delete failed in rebuild for %s: %s", mid, e)
                try:
                    self._vector.add(enriched, mid, metadata)
                    vec_count += 1
                except Exception as e:
                    logger.warning("rebuild vector add failed for %s: %s", mid, e)
                try:
                    self._bm25.add(enriched, mid, metadata)
                    bm25_count += 1
                except Exception as e:
                    logger.warning("rebuild bm25 add failed for %s: %s", mid, e)
            self._vector.flush()
            logger.info(
                "HybridRetriever rebuild: vector=%d, bm25=%d from %d entries",
                vec_count, bm25_count, len(entries),
            )
            return {"vector": vec_count, "bm25": bm25_count}
        finally:
            self._rw_lock.release_write()


def _enrich_for_rebuild(content: str, mem_type: str, room: str = "") -> str:
    """★ R34修复Minor-3：重建索引时为各类型附加可搜索描述。"""
    type_prefixes = {
        "secret": "[加密信息/密钥/凭证]",
        "skill": "[技能/步骤/教程]",
        "procedural": "[流程/操作/指南]",
        "reasoning": "[教训/经验/踩坑]",
        "action": "[Agent行为/工具调用]",
    }
    prefix = type_prefixes.get(mem_type, "")
    if prefix:
        return f"{prefix} {room} {content}"
    return content


def _is_garbage_query(query: str) -> bool:
    """检测查询是否为无意义/垃圾输入（QUAL-1修复）。

    以下情况判定为垃圾查询：
    1. 纯随机字符串（连续5+非词典字符且无中文/常见英文单词）
    2. 极短查询（<2字符）且无中文
    3. 纯数字/纯符号串
    4. 包含少量常见词但主体为随机字符（如 zzzzzxyz123test）

    Returns:
        True 表示应限制返回结果数量
    """
    import re

    q = query.strip()
    if not q or len(q) < 2:
        return True

    if re.search(r"[\u4e00-\u9fff]", q):
        return False

    words = re.findall(r"[a-zA-Z]{3,}", q.lower())
    word_set = set(words)
    matched_common = word_set & _GARBAGE_COMMON_WORDS

    if len(matched_common) >= 2:
        common_char_len = sum(len(w) for w in words if w in matched_common)
        if common_char_len / len(q) > 0.4 and len(word_set) <= len(matched_common) + 2:
            return False

    if matched_common and len(q) > 8:
        non_word_chars = re.sub(r"[a-zA-Z]{3,}", "", q)
        noise_ratio = len(non_word_chars) / len(q)
        if noise_ratio > 0.5:
            return True

    random_chars = re.sub(r"[a-zA-Z0-9\s]", "", q)
    if len(random_chars) > len(q) * 0.6:
        return True

    if re.match(r"^[\d\s]+$", q):
        return True

    alpha_seq = re.findall(r"[a-zA-Z]{5,}", q)
    for seq in alpha_seq:
        seq_lower = seq.lower()
        if seq_lower not in _GARBAGE_COMMON_WORDS:
            vowel_count = sum(1 for c in seq_lower if c in "aeiou")
            unique_chars = len(set(seq_lower))
            if vowel_count == 0 or unique_chars <= 2:
                return True

    if len(matched_common) == 1 and len(q) < 6 and len(word_set) <= 1:
        return True

    return not matched_common and re.match(r"^[a-zA-Z0-9]+$", q) is not None and len(q) > 8


_GARBAGE_COMMON_WORDS = HybridRetriever._GARBAGE_COMMON_WORDS


def _trim_to_budget(results: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    """裁剪结果到 Token 预算内。"""
    budget = max_tokens
    chars_per_token = 4
    trimmed = []
    used = 0
    for r in results:
        content = r.get("content", "")
        est_tokens = max(1, len(content) // chars_per_token)
        if used + est_tokens <= budget:
            trimmed.append(r)
            used += est_tokens
        else:
            continue
    return trimmed
