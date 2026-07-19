"""HybridSearcher 单元测试

验证三种分数校准策略、纯向量/纯 BM25/混合融合、空候选与单候选等边界行为。
所有测试使用 mock 结果，不加载真实模型。
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HYBRID_PATH = os.path.join(ROOT, "hybrid_searcher.py")


def _load_hybrid_searcher():
    """动态加载 hybrid_searcher 模块"""
    spec = importlib.util.spec_from_file_location("hybrid_searcher", _HYBRID_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hybrid_searcher"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_hybrid_searcher()


def _search(module, calibration: str, vector_results, bm25_results, skill_names=None):
    """构造 HybridSearcher 并执行搜索的辅助函数"""
    searcher = module.HybridSearcher(config={
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
        "use_reranker": False,
        "hybrid_calibration": calibration,
    })
    return searcher.search(
        query="测试查询",
        top_k=10,
        vector_results=vector_results,
        bm25_results=bm25_results,
        skill_names=skill_names,
    )


def test_calibration_strategies_produce_different_distributions(module):
    """minmax / sigmoid / zscore 三种校准策略应产生不同分数分布"""
    vector_results = [
        {"name": "skill-a", "score": 0.2},
        {"name": "skill-b", "score": 0.8},
    ]
    bm25_results = [("skill-b", 2.0), ("skill-c", 8.0)]

    minmax_scores = {r["name"]: r["score"] for r in _search(module, "minmax", vector_results, bm25_results)}
    sigmoid_scores = {r["name"]: r["score"] for r in _search(module, "sigmoid", vector_results, bm25_results)}
    zscore_scores = {r["name"]: r["score"] for r in _search(module, "zscore", vector_results, bm25_results)}

    # minmax 会分别将向量与 BM25 的最大值拉到 1.0
    assert minmax_scores["skill-b"] == pytest.approx(0.7, abs=1e-9)
    assert minmax_scores["skill-c"] == pytest.approx(0.3, abs=1e-9)

    # sigmoid 的分数应在 (0,1) 区间且不与 minmax 相同
    assert 0.0 < sigmoid_scores["skill-b"] < 1.0
    assert sigmoid_scores != minmax_scores

    # zscore 标准化后分数可为负且分布明显不同
    assert zscore_scores != minmax_scores
    assert zscore_scores != sigmoid_scores


def test_pure_vector_search(module):
    """纯向量输入时，BM25 分支贡献为 0"""
    vector_results = [
        {"name": "skill-a", "score": 0.1},
        {"name": "skill-b", "score": 0.9},
    ]
    results = _search(module, "minmax", vector_results, [])
    scores = {r["name"]: r["score"] for r in results}
    assert scores["skill-a"] == pytest.approx(0.0, abs=1e-9)
    assert scores["skill-b"] == pytest.approx(0.7, abs=1e-9)


def test_pure_bm25_search(module):
    """纯 BM25 输入时，向量分支贡献为 0"""
    bm25_results = [("skill-a", 1.0), ("skill-b", 3.0)]
    results = _search(module, "minmax", [], bm25_results)
    scores = {r["name"]: r["score"] for r in results}
    assert scores["skill-a"] == pytest.approx(0.0, abs=1e-9)
    assert scores["skill-b"] == pytest.approx(0.3, abs=1e-9)


def test_hybrid_fusion_weighted_sum(module):
    """混合输入时，结果应为加权求和"""
    vector_results = [{"name": "skill-x", "score": 0.5}]
    bm25_results = [("skill-x", 5.0)]
    results = _search(module, "minmax", vector_results, bm25_results)
    assert len(results) == 1
    # minmax 后向量=1，BM25=1，加权和=0.7*1+0.3*1=1
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-9)


def test_empty_candidates_returns_empty(module):
    """空候选集应返回空列表"""
    assert _search(module, "minmax", [], []) == []
    assert _search(module, "sigmoid", [], []) == []
    assert _search(module, "zscore", [], []) == []


def test_single_candidate(module):
    """单候选集在不同校准策略下行为正确"""
    vector_results = [{"name": "only", "score": 0.5}]

    minmax_result = _search(module, "minmax", vector_results, [])[0]
    sigmoid_result = _search(module, "sigmoid", vector_results, [])[0]
    zscore_result = _search(module, "zscore", vector_results, [])[0]

    assert minmax_result["score"] == pytest.approx(0.7, abs=1e-9)
    assert 0.0 < sigmoid_result["score"] < 0.7
    assert zscore_result["score"] == pytest.approx(0.0, abs=1e-9)


def test_top_k_ordering_and_truncation(module):
    """结果应按分数降序排列并截断到 top_k"""
    vector_results = [
        {"name": f"skill-{i}", "score": float(i)}
        for i in range(10)
    ]
    results = module.HybridSearcher(config={
        "vector_weight": 1.0,
        "bm25_weight": 0.0,
        "hybrid_calibration": "minmax",
    }).search("q", top_k=3, vector_results=vector_results)

    assert len(results) == 3
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["name"] == "skill-9"
