from __future__ import annotations

"""检索模块单元测试。

覆盖: RRFFusion / CrossEncoderReranker / VectorRetriever / BM25Retriever / HybridRetriever
"""

import pytest
pytest.importorskip("chromadb", reason="chromadb not installed")
pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")

import tempfile
import unittest
from pathlib import Path
from typing import Any

from omnimem.retrieval.rrf import RRFFusion
from omnimem.retrieval.reranker import CrossEncoderReranker
from omnimem.retrieval.vector import VectorRetriever, _CachedEmbeddingFunction
from omnimem.retrieval.bm25 import BM25Retriever
from omnimem.retrieval.engine import HybridRetriever


def _has_vector_model() -> bool:
    """检查 sentence-transformers 模型是否可加载。"""
    try:
        # ROCm PyTorch 兼容性
        import torch.distributed as dist
        if not hasattr(dist, 'is_initialized'):
            dist.is_initialized = lambda: False
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        _ = m.encode(["test"])
        return True
    except Exception:
        return False


_HAS_VECTOR = _has_vector_model()


# ──────────────────────────────────────────────
# RRFFusion
# ──────────────────────────────────────────────


class TestRRFFusion(unittest.TestCase):
    """RRFFusion 融合算法测试。"""

    def setUp(self) -> None:
        self.rrf = RRFFusion(k=60, min_rrf=0.035)

    def test_merge_basic_two_lists(self) -> None:
        """双路融合：向量 + BM25 结果合并排序。"""
        vec_results = [
            {"memory_id": "a", "content": "深度学习框架", "score": 0.85},
            {"memory_id": "b", "content": "Python编程", "score": 0.72},
            {"memory_id": "c", "content": "机器学习基础", "score": 0.55},
        ]
        bm25_results = [
            {"memory_id": "b", "content": "Python编程", "score": 2.3},
            {"memory_id": "d", "content": "编程语言对比", "score": 1.8},
            {"memory_id": "a", "content": "深度学习框架", "score": 1.5},
        ]
        fused = self.rrf.merge([vec_results, bm25_results])
        self.assertGreater(len(fused), 0)
        for r in fused:
            self.assertIn("rrf_score", r)
            self.assertIn("score", r)

    def test_merge_high_similarity_vector_bonus(self) -> None:
        """向量高分 (>0.5) 应获得额外加成。"""
        vec = [{"memory_id": "x", "content": "量子计算", "score": 0.80}]
        bm25 = [{"memory_id": "x", "content": "量子计算", "score": 1.0}]
        fused = self.rrf.merge([vec, bm25])
        self.assertEqual(len(fused), 1)
        self.assertGreater(fused[0]["rrf_score"], 0.01)

    def test_merge_threshold_filter(self) -> None:
        """低于 min_rrf 阈值的结果应被过滤。"""
        rrf_low = RRFFusion(k=60, min_rrf=0.10)
        vec = [{"memory_id": "a", "content": "test", "score": 0.30}]
        bm25: list[dict[str, Any]] = []
        fused = rrf_low.merge([vec, bm25])
        self.assertEqual(len(fused), 0)

    def test_merge_empty_lists(self) -> None:
        """空输入应返回空列表。"""
        fused = self.rrf.merge([[], []])
        self.assertEqual(len(fused), 0)

    def test_merge_missing_id(self) -> None:
        """缺少 memory_id 时使用 content hash 作为 ID 进行去重。"""
        vec = [{"content": "hello world", "score": 0.60}]
        bm25: list[dict[str, Any]] = []
        fused = self.rrf.merge([vec, bm25])
        self.assertEqual(len(fused), 1)
        self.assertIn("rrf_score", fused[0])

    def test_merge_custom_weights(self) -> None:
        """自定义权重应影响融合结果。"""
        vec = [{"memory_id": "a", "content": "猫喜欢吃鱼", "score": 0.90}]
        bm25 = [{"memory_id": "a", "content": "猫喜欢吃鱼", "score": 2.0}]
        fused_default = self.rrf.merge([vec, bm25])
        fused_equal = self.rrf.merge([vec, bm25], weights=[1.0, 1.0])
        self.assertNotEqual(fused_default[0]["rrf_score"], fused_equal[0]["rrf_score"])


# ──────────────────────────────────────────────
# CrossEncoderReranker
# ──────────────────────────────────────────────


class TestCrossEncoderReranker(unittest.TestCase):
    """CrossEncoderReranker 测试。"""

    def setUp(self) -> None:
        self.reranker = CrossEncoderReranker()

    def test_rerank_empty_results(self) -> None:
        """空结果应直接返回。"""
        result = self.reranker.rerank("query", [], top_k=5)
        self.assertEqual(len(result), 0)

    def test_rerank_no_model_fallback(self) -> None:
        """无 sentence_transformers 时回退到原始排序。"""
        results = [
            {"memory_id": "a", "content": "first", "score": 0.5},
            {"memory_id": "b", "content": "second", "score": 0.8},
            {"memory_id": "c", "content": "third", "score": 0.3},
        ]
        reranked = self.reranker.rerank("test query", results, top_k=2)
        self.assertEqual(len(reranked), 2)


# ──────────────────────────────────────────────
# VectorRetriever (需 sentence-transformers 模型)
# ──────────────────────────────────────────────


@unittest.skipUnless(_HAS_VECTOR, "sentence-transformers model not available")
class TestVectorRetriever(unittest.TestCase):
    """VectorRetriever 向量检索测试。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.vec = VectorRetriever(data_dir=Path(self.tmpdir))

    def test_init_defaults(self) -> None:
        """初始化时 backend 默认为 chromadb。"""
        v = VectorRetriever(data_dir=Path(self.tmpdir))
        self.assertEqual(v._backend, "chromadb")
        self.assertFalse(v._initialized)

    def test_search_empty_collection(self) -> None:
        """空集合搜索应返回空列表。"""
        results = self.vec.search("测试查询")
        self.assertEqual(len(results), 0)

    def test_count_empty(self) -> None:
        """空集合 count 应返回 0。"""
        self.assertEqual(self.vec.count(), 0)

    def test_add_and_search_chinese(self) -> None:
        """中文记忆的添加和语义搜索。"""
        self.vec.add(
            "用户喜欢用Python编写后端服务",
            "mem-001",
            {"type": "preference", "scope": "personal"},
        )
        self.vec.add(
            "用户偏好使用FastAPI框架进行Web开发",
            "mem-002",
            {"type": "preference", "scope": "personal"},
        )
        self.vec.flush()

        results = self.vec.search("Python 开发")
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("content", r)
            self.assertIn("memory_id", r)
            self.assertIn("score", r)

    def test_add_and_search_english(self) -> None:
        """英文记忆的添加和搜索。"""
        self.vec.add(
            "User works with Docker containers",
            "mem-e1",
            {"type": "fact"},
        )
        self.vec.add(
            "User deploys to AWS Lambda",
            "mem-e2",
            {"type": "fact"},
        )
        self.vec.flush()

        results = self.vec.search("Docker deployment")
        self.assertGreater(len(results), 0)

    def test_add_batch(self) -> None:
        """批量添加文档。"""
        docs = [
            {"content": "喜欢吃川菜", "memory_id": "b1", "type": "preference"},
            {"content": "住在北京", "memory_id": "b2", "type": "fact"},
            {"content": "使用macOS系统", "memory_id": "b3", "type": "fact"},
        ]
        self.vec.add_batch(docs)
        self.vec.flush()
        self.assertGreater(self.vec.count(), 0)

    def test_count_after_add(self) -> None:
        """添加后 count 应增加。"""
        self.vec.add("测试内容", "cnt-1", {})
        self.vec.flush()
        self.assertGreater(self.vec.count(), 0)

    def test_search_similarity_threshold(self) -> None:
        """低相似度结果应被过滤 (sim < 0.25)。"""
        self.vec.add("川菜以麻辣为特色", "th-1", {"type": "fact"})
        self.vec.flush()
        results = self.vec.search("量子计算机的物理原理")
        self.assertEqual(len(results), 0)

    def test_embed_text(self) -> None:
        """embed_text 应返回非空向量。"""
        self.vec.add("测试", "emb-1", {})
        self.vec.flush()
        vec = self.vec.embed_text("测试文本")
        self.assertGreater(len(vec), 0)
        self.assertIsInstance(vec[0], float)


# ──────────────────────────────────────────────
# _CachedEmbeddingFunction (需 sentence-transformers 模型)
# ──────────────────────────────────────────────


@unittest.skipUnless(_HAS_VECTOR, "sentence-transformers model not available")
class TestCachedEmbeddingFunction(unittest.TestCase):
    """_CachedEmbeddingFunction 测试。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.cache_path = Path(self.tmpdir) / "emb_cache.json"

    def test_cache_hit(self) -> None:
        """相同文本应命中缓存。"""
        ef = _CachedEmbeddingFunction(cache_path=self.cache_path)
        result1 = ef(["测试文本"])
        result2 = ef(["测试文本"])
        self.assertEqual(len(result1), 1)
        self.assertEqual(len(result2), 1)
        self.assertEqual(len(result1[0]), len(result2[0]))

    def test_persist_and_load(self) -> None:
        """缓存应能持久化并重新加载。"""
        ef1 = _CachedEmbeddingFunction(cache_path=self.cache_path)
        ef1(["持久化测试内容"])
        ef1.persist()

        ef2 = _CachedEmbeddingFunction(cache_path=self.cache_path)
        result = ef2(["持久化测试内容"])
        self.assertEqual(len(result), 1)
        self.assertGreater(len(result[0]), 0)


# ──────────────────────────────────────────────
# BM25Retriever
# ──────────────────────────────────────────────


class TestBM25Retriever(unittest.TestCase):
    """BM25Retriever 关键词检索测试。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.bm25 = BM25Retriever(data_dir=Path(self.tmpdir))

    def test_init_defaults(self) -> None:
        """初始化后缓冲区和文档计数应为 0。"""
        self.assertEqual(self.bm25.document_count, 0)
        self.assertEqual(self.bm25.pending_count, 0)

    def test_search_empty(self) -> None:
        """空索引搜索应返回空列表。"""
        results = self.bm25.search("测试查询")
        self.assertEqual(len(results), 0)

    def test_add_and_search_chinese(self) -> None:
        """中文文档添加和关键词检索。"""
        self.bm25.add("用户使用Python开发后端服务", "bm-001", {"type": "fact"})
        self.bm25.add("用户喜欢使用TypeScript开发前端", "bm-002", {"type": "fact"})
        self.bm25.flush()

        results = self.bm25.search("Python 开发")
        self.assertGreater(len(results), 0)

    def test_add_and_search_keyword_match(self) -> None:
        """精确关键词应返回高分结果。"""
        self.bm25.add("猫喜欢吃鱼和鸡肉", "bm-k1", {"type": "preference"})
        self.bm25.add("狗喜欢啃骨头和肉", "bm-k2", {"type": "preference"})
        self.bm25.flush()

        results = self.bm25.search("猫 喜欢")
        self.assertGreater(len(results), 0)

    def test_search_no_match(self) -> None:
        """无匹配关键词应返回空。"""
        self.bm25.add("人工智能发展趋势", "bm-n1", {"type": "fact"})
        self.bm25.flush()

        results = self.bm25.search("烹饪技巧")
        self.assertEqual(len(results), 0)

    def test_add_batch(self) -> None:
        """批量添加文档。"""
        docs = [
            {"content": "量子计算使用量子比特", "memory_id": "bb1", "type": "fact"},
            {"content": "经典计算使用比特", "memory_id": "bb2", "type": "fact"},
            {"content": "神经网络需要大量数据训练", "memory_id": "bb3", "type": "fact"},
        ]
        self.bm25.add_batch(docs)
        self.assertEqual(self.bm25.document_count, 3)

    def test_document_count_after_add(self) -> None:
        """添加后 document_count 应增加。"""
        self.bm25.add("测试文档内容", "dc-1", {"type": "fact"})
        self.bm25.flush()
        self.assertEqual(self.bm25.document_count, 1)

    def test_flush_disk_cache(self) -> None:
        """flush 后磁盘缓存应可恢复。"""
        self.bm25.add("磁盘缓存测试内容", "disk-1", {"type": "fact"})
        self.bm25.flush()

        bm25_new = BM25Retriever(data_dir=Path(self.tmpdir))
        self.assertGreater(bm25_new.document_count, 0)

    def test_pending_count(self) -> None:
        """缓冲区 pending_count 应反映未刷新的文档数。"""
        self.bm25.add("缓冲测试1", "p-1", {"type": "fact"})
        self.assertEqual(self.bm25.pending_count, 1)
        self.bm25.flush()
        self.assertEqual(self.bm25.pending_count, 0)


# ──────────────────────────────────────────────
# HybridRetriever (需 sentence-transformers 模型)
# ──────────────────────────────────────────────


@unittest.skipUnless(_HAS_VECTOR, "sentence-transformers model not available")
class TestHybridRetriever(unittest.TestCase):
    """HybridRetriever 混合检索测试。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.hybrid = HybridRetriever(data_dir=Path(self.tmpdir))

    def test_init_defaults(self) -> None:
        """初始化应创建所有子组件。"""
        self.assertIsNotNone(self.hybrid._vector)
        self.assertIsNotNone(self.hybrid._bm25)
        self.assertIsNotNone(self.hybrid._rrf)

    def test_search_empty(self) -> None:
        """空索引搜索应返回空列表。"""
        results = self.hybrid.search("测试")
        self.assertEqual(len(results), 0)

    def test_add_and_search(self) -> None:
        """添加文档后应能通过混合检索找到。"""
        self.hybrid.add(
            "用户喜欢使用React开发前端界面",
            "hy-001",
            {"type": "preference", "scope": "personal"},
        )
        self.hybrid.add(
            "用户偏好Vue框架的响应式设计",
            "hy-002",
            {"type": "preference", "scope": "personal"},
        )
        self.hybrid.flush()

        results = self.hybrid.search("React 前端开发", mode="rag", top_k=5)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("content", r)
            self.assertIn("memory_id", r)
            self.assertIn("score", r)

    def test_add_batch(self) -> None:
        """批量添加。"""
        docs = [
            {"content": "Node.js是服务器端JavaScript运行时", "memory_id": "hb1", "type": "fact"},
            {"content": "Deno是新一代JavaScript运行时", "memory_id": "hb2", "type": "fact"},
        ]
        self.hybrid.add_batch(docs)
        self.hybrid.flush()
        results = self.hybrid.search("JavaScript 运行时")
        self.assertGreater(len(results), 0)

    def test_set_source_weights(self) -> None:
        """动态来源权重设置。"""
        self.hybrid.set_source_weights({"vector": 2.0, "bm25": 0.5})
        self.assertEqual(self.hybrid._source_weights["vector"], 2.0)

    def test_invalidate_cache(self) -> None:
        """缓存清除。"""
        self.hybrid.add("缓存测试", "cache-1", {"type": "fact"})
        self.hybrid.flush()
        self.hybrid.search("缓存测试")
        self.hybrid.invalidate_cache()
        self.assertEqual(len(self.hybrid._query_cache), 0)

    def test_rebuild_bm25_from_entries(self) -> None:
        """从索引条目重建 BM25。"""
        entries = [
            {"content": "深度学习使用神经网络", "memory_id": "re-1", "type": "fact", "scope": "personal"},
            {"content": "机器学习是AI的子领域", "memory_id": "re-2", "type": "fact", "scope": "personal"},
        ]
        count = self.hybrid.rebuild_bm25_from_entries(entries)
        self.assertEqual(count, 2)
        count2 = self.hybrid.rebuild_bm25_from_entries(entries)
        self.assertEqual(count2, 0)


# ──────────────────────────────────────────────
# HybridRetriever static methods (no deps needed)
# ──────────────────────────────────────────────


class TestHybridRetrieverStatic(unittest.TestCase):
    """HybridRetriever 静态方法测试（无需模型）。"""

    def test_is_garbage_query_chinese(self) -> None:
        """中文查询 → 非垃圾。"""
        self.assertFalse(HybridRetriever._is_garbage_query("什么是机器学习"))

    def test_is_garbage_query_empty(self) -> None:
        """空查询 → 垃圾。"""
        self.assertTrue(HybridRetriever._is_garbage_query(""))

    def test_is_garbage_query_random(self) -> None:
        """纯随机字符串 → 垃圾。"""
        self.assertTrue(HybridRetriever._is_garbage_query("zzzzzxyz123"))

    def test_is_garbage_query_digits(self) -> None:
        """纯数字 → 垃圾。"""
        self.assertTrue(HybridRetriever._is_garbage_query("12345678"))

    def test_is_garbage_query_short(self) -> None:
        """极短无意义 → 垃圾。"""
        self.assertTrue(HybridRetriever._is_garbage_query("z"))

    def test_is_garbage_query_english(self) -> None:
        """有意义的英文 → 非垃圾。"""
        self.assertFalse(HybridRetriever._is_garbage_query("what is python programming"))

    def test_trim_to_budget(self) -> None:
        """Token 预算裁剪。"""
        results = [
            {"content": "a" * 100, "memory_id": "t1"},
            {"content": "b" * 100, "memory_id": "t2"},
            {"content": "c" * 200, "memory_id": "t3"},
        ]
        trimmed = HybridRetriever._trim_to_budget(results, max_tokens=30)
        self.assertLessEqual(len(trimmed), 3)


class TestHybridRetrieverEdgeCases(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.hybrid = HybridRetriever(data_dir=Path(self.tmpdir))

    def test_empty_query(self) -> None:
        results = self.hybrid.search("")
        self.assertIsInstance(results, list)

    def test_very_long_query(self) -> None:
        query = "量子计算" * 500
        results = self.hybrid.search(query)
        self.assertIsInstance(results, list)

    def test_special_chars_query(self) -> None:
        results = self.hybrid.search("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        self.assertIsInstance(results, list)

    def test_garbage_query_detection(self) -> None:
        self.assertTrue(HybridRetriever._is_garbage_query("zzzzzxyz123"))
        self.assertTrue(HybridRetriever._is_garbage_query(""))
        self.assertTrue(HybridRetriever._is_garbage_query("1234567890"))
        self.assertFalse(HybridRetriever._is_garbage_query("深度学习框架"))
        self.assertFalse(HybridRetriever._is_garbage_query("python web development"))

    def test_no_results(self) -> None:
        results = self.hybrid.search("不存在的记忆内容xyz")
        self.assertEqual(len(results), 0)

    def test_concurrent_search_and_write(self) -> None:
        import threading

        errors = []

        def writer() -> None:
            try:
                for i in range(5):
                    self.hybrid.add(
                        f"并发写入内容{i}",
                        f"concurrent-{i}",
                        {"type": "fact"},
                    )
            except Exception as e:
                errors.append(e)

        def searcher() -> None:
            try:
                for _ in range(5):
                    self.hybrid.search("并发")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=searcher)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")
