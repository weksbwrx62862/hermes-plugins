"""BM25Searcher 单元测试

验证倒排索引版 BM25 与暴力版在分数与排序上一致，并覆盖边界情况。
"""

import importlib.util
import math
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INIT_PATH = os.path.join(ROOT, "__init__.py")


def _load_init():
    """通过 importlib 加载 skill-router 的 __init__.py。

    目录名含连字符，无法作为普通包导入，因此使用文件路径加载。
    """
    spec = importlib.util.spec_from_file_location("skill_router_init", _INIT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["skill_router_init"] = module
    spec.loader.exec_module(module)
    return module


def _tokenize(text: str, use_jieba: bool) -> List[str]:
    """与 BM25Searcher 一致的分词逻辑（测试用）"""
    if use_jieba:
        import jieba
        return [w for w in jieba.cut(text) if w.strip()]
    return [c for c in text if c.strip()]


def _brute_force_bm25(
    skills: Dict[str, Dict[str, Any]], query: str, k1: float = 1.5, b: float = 0.75
) -> List[Tuple[str, float]]:
    """暴力实现 BM25，用于与 BM25Searcher 对比。"""
    try:
        import jieba
        jieba.setLogLevel(__import__("logging").WARNING)
        use_jieba = True
    except ImportError:
        use_jieba = False

    doc_tokens: Dict[str, List[str]] = {}
    doc_len: Dict[str, int] = {}
    doc_freq: Dict[str, int] = {}
    total_dl = 0

    for name, info in skills.items():
        text = f"{name} {info.get('description', '')} {info.get('body_text', '')}"
        tokens = _tokenize(text, use_jieba)
        doc_tokens[name] = tokens
        doc_len[name] = len(tokens)
        total_dl += len(tokens)

        seen = set()
        for t in tokens:
            if t not in seen:
                doc_freq[t] = doc_freq.get(t, 0) + 1
                seen.add(t)

    n_docs = len(skills)
    avg_dl = total_dl / n_docs if n_docs > 0 else 1.0
    idf = {
        term: math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        for term, df in doc_freq.items()
    }

    query_tokens = _tokenize(query, use_jieba)
    scores: Dict[str, float] = {}

    for name, tokens in doc_tokens.items():
        tf_map: Dict[str, int] = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        dl = doc_len[name]
        for qt in query_tokens:
            if qt not in tf_map:
                continue
            tf = tf_map[qt]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avg_dl)
            score += idf.get(qt, 0.0) * numerator / denominator

        if score > 0:
            scores[name] = score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _sample_skills() -> Dict[str, Dict[str, Any]]:
    return {
        "skill-a": {"description": "hello world", "body_text": "foo bar baz"},
        "skill-b": {"description": "hello test", "body_text": "baz qux"},
        "skill-c": {"description": "another topic", "body_text": "lorem ipsum"},
    }


def test_search_returns_sorted_tuples():
    module = _load_init()
    searcher = module.BM25Searcher(_sample_skills())
    results = searcher.search("hello", top_k=10)

    assert isinstance(results, list)
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_inverted_index_matches_brute_force():
    module = _load_init()
    skills = _sample_skills()
    searcher = module.BM25Searcher(skills)

    expected = _brute_force_bm25(skills, "hello world")
    actual = searcher.search("hello world", top_k=100)

    assert [name for name, _ in actual] == [name for name, _ in expected]
    for (_, score_a), (_, score_b) in zip(actual, expected):
        assert abs(score_a - score_b) < 1e-9


def test_empty_skills_returns_empty():
    module = _load_init()
    searcher = module.BM25Searcher({})
    assert searcher.search("hello") == []


def test_empty_query_returns_empty():
    module = _load_init()
    searcher = module.BM25Searcher(_sample_skills())
    assert searcher.search("   ") == []
    assert searcher.search("") == []


def test_no_match_returns_empty():
    module = _load_init()
    searcher = module.BM25Searcher(_sample_skills())
    assert searcher.search("zzzzzzzzz") == []


def test_top_k_truncation():
    module = _load_init()
    skills = {f"skill-{i}": {"description": f"common token{i}", "body_text": ""} for i in range(20)}
    searcher = module.BM25Searcher(skills)
    results = searcher.search("common", top_k=5)
    assert len(results) == 5
