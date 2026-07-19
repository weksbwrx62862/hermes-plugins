"""evaluate.py 与线上逻辑一致性测试

验证 evaluate.py 加载技能时不截断 body，且其 BM25 实现与 __init__.py 中的
BM25Searcher 在相同输入下结果一致。
"""

import importlib.util
import os
import sqlite3
import sys
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INIT_PATH = os.path.join(ROOT, "__init__.py")
_EVALUATE_PATH = os.path.join(ROOT, "scripts", "evaluate.py")


def _load_init():
    """加载 skill-router 的 __init__.py"""
    spec = importlib.util.spec_from_file_location("skill_router_init", _INIT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["skill_router_init"] = module
    spec.loader.exec_module(module)
    return module


def _load_evaluate():
    """加载 scripts/evaluate.py"""
    spec = importlib.util.spec_from_file_location("evaluate", _EVALUATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = module
    spec.loader.exec_module(module)
    return module


def _sample_skills() -> Dict[str, Dict[str, Any]]:
    """BM25Searcher 使用的技能格式（body_text）"""
    return {
        "skill-a": {"description": "hello world", "body_text": "foo bar baz"},
        "skill-b": {"description": "hello test", "body_text": "baz qux"},
        "skill-c": {"description": "another topic", "body_text": "lorem ipsum"},
    }


def test_load_skills_does_not_truncate():
    """验证 evaluate.load_skills 返回完整 description 与 body_text"""
    import tempfile

    evaluate = _load_evaluate()

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        evaluate.DB_PATH = db_path

        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE skills (name TEXT, description TEXT, body TEXT)")
        long_body = "x" * 10000
        long_desc = "d" * 10000
        conn.execute(
            "INSERT INTO skills VALUES (?, ?, ?)",
            ("skill-long", long_desc, long_body),
        )
        conn.commit()
        conn.close()

        skills = evaluate.load_skills()
        assert skills["skill-long"]["body_text"] == long_body
        assert skills["skill-long"]["description"] == long_desc
    finally:
        os.unlink(db_path)


def test_evaluate_bm25_matches_online_searcher():
    """验证 evaluate.py 复用的 BM25Searcher 与线上 BM25Searcher 分数排序一致"""
    init = _load_init()
    evaluate = _load_evaluate()

    skills = _sample_skills()

    searcher = init.BM25Searcher(skills)
    evaluator = evaluate.BM25Searcher(skills)

    query = "hello world"
    r_searcher = searcher.search(query, top_k=10)
    r_evaluator = evaluator.search(query, top_k=10)

    assert [name for name, _ in r_searcher] == [name for name, _ in r_evaluator]
    for (_, score_a), (_, score_b) in zip(r_searcher, r_evaluator):
        assert abs(score_a - score_b) < 1e-9
