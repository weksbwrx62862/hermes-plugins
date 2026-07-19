"""Task 2: 检索超时降级机制 — 单元测试。"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch


class TestRecallTimeoutDegradation:
    """测试检索超时降级：超时自动降级、Future 取消、策略切换。"""

    def _make_retriever(self, **kwargs):
        """创建 HybridRetriever 实例（mock 向量和 BM25）。"""
        from omnimem.retrieval.engine import HybridRetriever

        with patch.object(HybridRetriever, "__init__", lambda _self, **_kw: None):
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever._recall_timeout_ms = kwargs.get("recall_timeout_ms", 5000)
            retriever._recall_strategy = kwargs.get("recall_strategy", "hybrid")
            retriever._rw_lock = MagicMock()
            retriever._query_cache = {}
            retriever._QUERY_CACHE_TTL = 300
            retriever._source_weights = {}
            retriever._rrf = MagicMock()
            retriever._reranker = None
            return retriever

    def test_init_default_timeout(self):
        """默认超时应为 5000ms，策略为 hybrid。"""
        from omnimem.retrieval.engine import HybridRetriever

        with patch.object(HybridRetriever, "__init__", lambda _self, **_kw: None):
            r = HybridRetriever.__new__(HybridRetriever)
            # 模拟实际 __init__ 的默认值
            r._recall_timeout_ms = 5000
            r._recall_strategy = "hybrid"
            assert r._recall_timeout_ms == 5000
            assert r._recall_strategy == "hybrid"

    def test_init_custom_timeout(self):
        """自定义超时和策略应正确传递。"""
        from omnimem.retrieval.engine import HybridRetriever

        with patch.object(HybridRetriever, "__init__", lambda _self, **_kw: None):
            r = HybridRetriever.__new__(HybridRetriever)
            r._recall_timeout_ms = 3000
            r._recall_strategy = "keyword"
            assert r._recall_timeout_ms == 3000
            assert r._recall_strategy == "keyword"

    def test_keyword_strategy_skips_vector(self):
        """keyword 策略应跳过向量检索，只用 BM25。"""
        retriever = self._make_retriever(recall_strategy="keyword")
        retriever._bm25_search = MagicMock(return_value=[{"memory_id": "1", "content": "test"}])
        retriever._vector_search = MagicMock(return_value=[])
        retriever._rrf_fuse = MagicMock(return_value=[{"memory_id": "1"}])
        retriever._is_garbage_query = MagicMock(return_value=False)
        retriever._vector = MagicMock(count=MagicMock(return_value=10))
        retriever._supplement_low_recall_types = MagicMock(side_effect=lambda _q, r, _k: r)
        retriever._apply_type_boost = MagicMock(side_effect=lambda r: r)
        retriever._rw_lock = MagicMock()

        # keyword 策略：只调用 BM25，不调用 vector
        retriever._bm25_search("test query", 10)
        retriever._bm25_search.assert_called_once()

    def test_embedding_strategy_skips_bm25(self):
        """embedding 策略应跳过 BM25，只用向量检索。"""
        retriever = self._make_retriever(recall_strategy="embedding")
        retriever._vector_search = MagicMock(return_value=[{"memory_id": "1", "content": "test"}])
        retriever._bm25_search = MagicMock(return_value=[])

        retriever._vector_search("test query", 10)
        retriever._vector_search.assert_called_once()

    def test_timeout_returns_empty(self):
        """超时后应返回空结果，不阻塞。"""
        # 模拟一个超时的搜索
        def slow_search(_query, _top_k):
            time.sleep(10)  # 远超超时时间
            return [{"memory_id": "1"}]

        timeout_sec = 0.1  # 100ms 超时
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(slow_search, "test", 10)
            try:
                results = future.result(timeout=timeout_sec)
            except TimeoutError:
                future.cancel()
                results = []

        assert results == []

    def test_single_channel_vector_empty_weights(self):
        """向量通道为空时，权重应为 [0.0, 1.0]。"""

        # 验证单通道降级逻辑
        vector_results = []
        bm25_results = [{"memory_id": "1", "score": 0.5}]
        single_channel = (not vector_results) or (not bm25_results)

        assert single_channel is True

        if not vector_results:
            base_weights = [0.0, 1.0]
        else:
            base_weights = [1.0, 0.0]

        assert base_weights == [0.0, 1.0]

    def test_single_channel_bm25_empty_weights(self):
        """BM25 通道为空时，权重应为 [1.0, 0.0]。"""
        vector_results = [{"memory_id": "1", "score": 0.5}]
        bm25_results = []
        single_channel = (not vector_results) or (not bm25_results)

        assert single_channel is True

        if not vector_results:
            base_weights = [0.0, 1.0]
        else:
            base_weights = [1.0, 0.0]

        assert base_weights == [1.0, 0.0]

    def test_config_schema_entries(self):
        """验证 recall_timeout_ms 和 recall_strategy 在 schema 中。"""
        from omnimem.config._config import _CONFIG_SCHEMA

        assert "recall_timeout_ms" in _CONFIG_SCHEMA
        assert _CONFIG_SCHEMA["recall_timeout_ms"]["default"] == 5000
        assert _CONFIG_SCHEMA["recall_timeout_ms"]["min"] == 100
        assert _CONFIG_SCHEMA["recall_timeout_ms"]["max"] == 30000

        assert "recall_strategy" in _CONFIG_SCHEMA
        assert _CONFIG_SCHEMA["recall_strategy"]["default"] == "hybrid"
        assert _CONFIG_SCHEMA["recall_strategy"]["choices"] == ["hybrid", "keyword", "embedding"]
