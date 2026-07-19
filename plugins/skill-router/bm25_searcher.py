"""BM25 关键词检索模块

提供基于倒排索引的 BM25 检索实现，支持 jieba 分词与字符级降级。
"""

import logging
import math
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("skill_router_init.bm25_searcher")


class BM25Searcher:
    """基于 BM25 算法的关键词检索器

    使用 jieba 分词（不可用时降级为字符级分词），在初始化时预计算
    倒排索引和 IDF 值，search 时仅遍历查询词项对应的 posting 列表。
    """

    _K1 = 1.5
    _B = 0.75

    def __init__(self, skills: Dict[str, Dict[str, Any]]):
        self._use_jieba = False
        try:
            import jieba

            jieba.setLogLevel(logging.WARNING)
            self._use_jieba = True
            logger.debug("BM25Searcher: 使用 jieba 分词")
        except ImportError:
            logger.debug("BM25Searcher: jieba 不可用，降级为字符级分词")

        self._idf: Dict[str, float] = {}
        self._avg_dl: float = 0.0
        self._doc_len: Dict[str, int] = {}
        self._inverted_index: Dict[str, List[Tuple[str, int]]] = {}

        if not skills:
            return

        doc_freq: Dict[str, int] = {}
        total_dl = 0
        doc_term_freqs: Dict[str, Dict[str, int]] = {}

        for name, info in skills.items():
            text = f"{name} {info.get('description', '')} {info.get('body_text', '')}"
            tokens = self._tokenize(text)
            dl = len(tokens)
            self._doc_len[name] = dl
            total_dl += dl

            tf_map: Dict[str, int] = {}
            seen: set[str] = set()
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
                if t not in seen:
                    doc_freq[t] = doc_freq.get(t, 0) + 1
                    seen.add(t)
            doc_term_freqs[name] = tf_map

        n_docs = len(skills)
        self._avg_dl = total_dl / n_docs if n_docs > 0 else 1.0

        for term, df in doc_freq.items():
            self._idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        # 构建倒排索引：词项 -> [(文档名, 词频), ...]
        for name, tf_map in doc_term_freqs.items():
            for term, tf in tf_map.items():
                self._inverted_index.setdefault(term, []).append((name, tf))

    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词"""
        if self._use_jieba:
            import jieba

            return [w for w in jieba.cut(text) if w.strip()]
        return [c for c in text if c.strip()]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """检索与查询最相关的技能，返回 (skill_name, score) 列表"""
        if not self._inverted_index:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = {}

        for qt in query_tokens:
            idf = self._idf.get(qt, 0.0)
            if idf <= 0.0:
                continue
            postings = self._inverted_index.get(qt)
            if not postings:
                continue
            for name, tf in postings:
                dl = self._doc_len[name]
                numerator = tf * (self._K1 + 1)
                denominator = tf + self._K1 * (1 - self._B + self._B * dl / self._avg_dl)
                scores[name] = scores.get(name, 0.0) + idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
