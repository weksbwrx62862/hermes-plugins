from __future__ import annotations

import logging
import re
from typing import Any

from omnimem.context.manager import ContextManager

logger = logging.getLogger(__name__)


class SemanticDedupService:

    def __init__(self, store, retriever):
        self._store = store
        self._retriever = retriever

    def semantic_dedup(
        self, content: str, memory_type: str, candidates: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        if len(content) <= 20:
            exact = candidates or self._store.search_by_content(content, limit=5)
            for m in exact:
                if m.get("content", "").strip() == content.strip():
                    return {
                        "action": "skip",
                        "existing_id": m.get("memory_id", ""),
                        "reason": "Exact duplicate",
                    }
            return {"action": "create"}

        short_skip_threshold = 0.92 if len(content) <= 50 else 0.8

        similar = candidates
        if similar is None:
            similar = self.search_candidates(content)

        for m in similar:
            existing_content = m.get("content", "")
            sim = self.compute_text_similarity(content, existing_content)

            if sim > short_skip_threshold:
                nums_a = set(re.findall(r"\d+", content))
                nums_b = set(re.findall(r"\d+", existing_content))
                has_numeric_diff = bool(nums_a ^ nums_b) or (len(nums_a) >= 2 and len(nums_b) >= 2)
                if has_numeric_diff:
                    sim = max(sim - 0.18, 0.5)

            if sim > 0.85:
                return {
                    "action": "skip",
                    "existing_id": m.get("memory_id", ""),
                    "reason": f"Near-duplicate (sim={sim:.2f})",
                }
            if sim > 0.6:
                return {
                    "action": "update",
                    "existing_id": m.get("memory_id", ""),
                    "reason": f"Similar (sim={sim:.2f}), archiving old",
                }

        return {"action": "create"}

    def search_candidates(self, content: str) -> list[dict[str, Any]]:
        similar = []
        try:
            if self._retriever and hasattr(self._retriever, "_vector") and self._retriever._vector:
                vector_results = self._retriever._vector.search(content, top_k=10)
                if vector_results:
                    similar = vector_results
        except Exception as e:
            logger.warning("Vector search for semantic dedup failed: %s", e)
        if not similar:
            similar = self._store.search_by_content(content[:50], limit=10)
        if len(content) > 100:
            mid_results = self._store.search_by_content(content[50:100], limit=5)
            existing_ids = {m.get("memory_id", "") for m in similar}
            for m in mid_results:
                if m.get("memory_id", "") not in existing_ids:
                    similar.append(m)
        return similar

    def unified_candidate_search(self, content: str) -> list[dict[str, Any]]:
        return self.search_candidates(content)

    @staticmethod
    def compute_text_similarity(text_a: str, text_b: str) -> float:
        fp_a = ContextManager._content_fingerprint(text_a)
        fp_b = ContextManager._content_fingerprint(text_b)
        return ContextManager._fingerprint_similarity(fp_a, fp_b)
