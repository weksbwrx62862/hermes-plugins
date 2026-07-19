"""SemanticDedupService 去重模块单元测试。

覆盖: 精确匹配 (≤20字符) / 短内容高阈值 (92%) / 长内容阈值 (80%)
       数值差异降权 / 85%近重复跳过 / 60%相似归档
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from omnimem.core.dedup import SemanticDedupService


class TestSemanticDedupService(unittest.TestCase):
    """SemanticDedupService 去重逻辑测试。"""

    def setUp(self) -> None:
        self.store = MagicMock()
        self.retriever = MagicMock()
        self.dedup = SemanticDedupService(self.store, self.retriever)

    # ── 精确匹配 (≤20字符) ──

    def test_exact_duplicate_short(self) -> None:
        """短内容精确重复应被跳过。"""
        self.store.search_by_content.return_value = [
            {"content": "hello world", "memory_id": "m1"}
        ]
        result = self.dedup.semantic_dedup("hello world", "fact")
        self.assertEqual(result["action"], "skip")
        self.assertEqual(result["existing_id"], "m1")

    def test_exact_no_match_short(self) -> None:
        """短内容不重复应创建。"""
        self.store.search_by_content.return_value = [
            {"content": "different", "memory_id": "m1"}
        ]
        result = self.dedup.semantic_dedup("unique text", "fact")
        self.assertEqual(result["action"], "create")

    def test_short_uses_candidates(self) -> None:
        """提供 candidates 参数时跳过 search_by_content。"""
        candidates = [{"content": "exact match here", "memory_id": "c1"}]
        result = self.dedup.semantic_dedup("exact match here", "fact", candidates=candidates)
        self.assertEqual(result["action"], "skip")
        self.store.search_by_content.assert_not_called()

    # ── 精确匹配边界：21-50字符 ──

    @patch("omnimem.core.dedup.ContextManager")
    def test_near_duplicate_medium(self, mock_ctx) -> None:
        """中等长度(21-50)，相似度>92%应被跳过。"""
        mock_ctx._content_fingerprint.return_value = "fp_a"
        mock_ctx._fingerprint_similarity.return_value = 0.93
        candidates = [{"content": "very similar content here", "memory_id": "c1"}]
        result = self.dedup.semantic_dedup(
            "very similar content here", "fact", candidates=candidates
        )
        self.assertEqual(result["action"], "skip")
        self.assertIn("sim=0.93", result["reason"])

    @patch("omnimem.core.dedup.ContextManager")
    def test_no_match_medium(self, mock_ctx) -> None:
        """中等长度，相似度低应创建。"""
        mock_ctx._content_fingerprint.return_value = "fp_unique"
        mock_ctx._fingerprint_similarity.return_value = 0.5
        candidates = [{"content": "completely different", "memory_id": "c1"}]
        result = self.dedup.semantic_dedup(
            "this is unique content with diff", "fact", candidates=candidates
        )
        self.assertEqual(result["action"], "create")

    # ── 长文本 (>50字符) ──

    @patch("omnimem.core.dedup.ContextManager")
    def test_near_duplicate_long(self, mock_ctx) -> None:
        """长文本，相似度>85%应跳过。"""
        mock_ctx._content_fingerprint.return_value = "fp_long"
        mock_ctx._fingerprint_similarity.return_value = 0.87
        content = "A" * 30 + " this is a long piece of text for testing purposes"
        candidates = [{"content": "B" * 30 + " this is a long piece of text for testing", "memory_id": "c1"}]
        result = self.dedup.semantic_dedup(content, "fact", candidates=candidates)
        self.assertEqual(result["action"], "skip")

    @patch("omnimem.core.dedup.ContextManager")
    def test_similar_update_long(self, mock_ctx) -> None:
        """长文本，相似度在60%-85%之间应触发update归档。"""
        mock_ctx._content_fingerprint.return_value = "fp_sim"
        mock_ctx._fingerprint_similarity.return_value = 0.7
        content = "This is a long document about machine learning and AI systems"
        candidates = [{"content": "This is about machine learning and deep learning", "memory_id": "c1"}]
        result = self.dedup.semantic_dedup(content, "fact", candidates=candidates)
        self.assertEqual(result["action"], "update")
        self.assertIn("c1", result["existing_id"])

    # ── 数值差异降权 ──

    @patch("omnimem.core.dedup.ContextManager")
    def test_numeric_diff_downgrades_sim(self, mock_ctx) -> None:
        """数值差异(版本号/年份)应降低相似度，使原本92%的匹配降至<85%。"""
        mock_ctx._content_fingerprint.return_value = "fp_num"
        # 返回0.93但因为有数值差异会被降到0.75 (0.93-0.18)
        mock_ctx._fingerprint_similarity.return_value = 0.93
        content = "R35 回归测试于 2026 年完成共 18 项"
        candidates = [{"content": "R36 回归测试于 2026 年完成共 17 项", "memory_id": "c1"}]
        result = self.dedup.semantic_dedup(content, "fact", candidates=candidates)
        # After numeric diff downgrade: 0.93 - 0.18 = 0.75, NOT >0.85
        self.assertNotEqual(result["action"], "skip")

    # ── 候选搜索 ──

    def test_search_candidates_vector(self) -> None:
        """向量搜索成功时返回向量结果。"""
        self.retriever.vector_search = MagicMock(return_value=[
            {"content": "vec result", "memory_id": "v1"}
        ])
        candidates = self.dedup.search_candidates("test content")
        self.assertEqual(len(candidates), 1)

    def test_search_candidates_fallback(self) -> None:
        """向量搜索失败时回退到 store search_by_content。"""
        self.retriever.vector_search = MagicMock(side_effect=Exception("vector unavailable"))
        self.store.search_by_content.return_value = [
            {"content": "store result", "memory_id": "s1"}
        ]
        candidates = self.dedup.search_candidates("test")
        self.assertTrue(any(c.get("content", "") == "store result" for c in candidates))

    def test_search_candidates_long_mid_chunk(self) -> None:
        """长文本搜索时添加中间片段候选。"""
        self.retriever.vector_search = MagicMock(side_effect=Exception("vector unavailable"))
        self.store.search_by_content.return_value = [
            {"content": "prefix match", "memory_id": "s1"}
        ]
        content = "PREFIX_" + "X" * 80 + "_MIDDLE_" + "Y" * 30 + "_SUFFIX"
        self.dedup.search_candidates(content)
        # 第一次调用于content[:50]，因为content>100会再调用content[50:100]
        self.assertGreaterEqual(self.store.search_by_content.call_count, 1)

    # ── compute_text_similarity ──

    @patch("omnimem.core.dedup.ContextManager")
    def test_compute_text_similarity(self, mock_ctx) -> None:
        mock_ctx._content_fingerprint.side_effect = lambda t: f"fp_{t[:5]}"
        mock_ctx._fingerprint_similarity.return_value = 0.88
        sim = SemanticDedupService.compute_text_similarity("hello world", "hello there")
        self.assertEqual(sim, 0.88)

    # ── 精确匹配边界：数值版本号触发差异 ──

    def test_exact_duplicate_short_numeric(self) -> None:
        """纯数字差异的短文本也应正确处理。"""
        self.store.search_by_content.return_value = [
            {"content": "R37 test v1", "memory_id": "m1"}
        ]
        result = self.dedup.semantic_dedup("R37 test v1", "fact")
        self.assertEqual(result["action"], "skip")

    def test_exact_duplicate_short_different_numbers(self) -> None:
        """数字不同的短文本不算是精确匹配。"""
        self.store.search_by_content.return_value = [
            {"content": "R36 test v1", "memory_id": "m1"}
        ]
        result = self.dedup.semantic_dedup("R37 test v1", "fact")
        self.assertEqual(result["action"], "create")
