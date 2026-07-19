"""FeedbackStore 单元测试

验证反馈记录读取、成功/跳过评分调整、衰减函数与边界值限制。
所有测试不依赖真实文件系统持久化（通过临时替换 FEEDBACK_FILE 或仅测内存逻辑）。
"""

import importlib.util
import os
import sys
import tempfile
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INIT_PATH = os.path.join(ROOT, "__init__.py")


def _load_init():
    """动态加载 skill-router 入口模块"""
    spec = importlib.util.spec_from_file_location("skill_router_init", _INIT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["skill_router_init"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_init()


def test_record_success_and_skip_updates_adjustment(module):
    """成功反馈加分，跳过反馈减分"""
    store = module.FeedbackStore()
    store._records = []  # 清空历史，避免跨测试污染

    store.record("skill-a", "query-1", "success")
    store.record("skill-a", "query-2", "success")
    store.record("skill-a", "query-3", "skip")

    adjustment = store.get_adjustments("skill-a")
    expected = 2 * store.SUCCESS_DELTA + 1 * store.SKIP_DELTA
    assert adjustment == pytest.approx(expected, abs=1e-9)


def test_get_adjustments_per_skill_isolated(module):
    """不同技能的反馈调整值相互隔离"""
    store = module.FeedbackStore()
    store._records = []

    store.record("skill-a", "q", "success")
    store.record("skill-b", "q", "skip")

    assert store.get_adjustments("skill-a") > 0
    assert store.get_adjustments("skill-b") < 0
    assert store.get_adjustments("skill-c") == 0


def test_invalid_feedback_type_is_ignored(module):
    """非法反馈类型不应影响调整值"""
    store = module.FeedbackStore()
    store._records = []

    store.record("skill-a", "q", "invalid")
    assert store.get_adjustments("skill-a") == 0


def test_adjustment_is_normalized(module):
    """反馈调整值应被限制在 [-MAX_ADJUSTMENT, MAX_ADJUSTMENT]"""
    store = module.FeedbackStore()
    store._records = []

    # 大量成功反馈，应被上限截断
    for i in range(100):
        store.record("skill-a", f"q-{i}", "success")
    assert store.get_adjustments("skill-a") == pytest.approx(store.MAX_ADJUSTMENT, abs=1e-9)

    store._records = []
    # 大量跳过反馈，应被下限截断
    for i in range(100):
        store.record("skill-b", f"q-{i}", "skip")
    assert store.get_adjustments("skill-b") == pytest.approx(-store.MAX_ADJUSTMENT, abs=1e-9)


def test_decay_weight_within_threshold(module):
    """24 小时内反馈权重为 1.0"""
    store = module.FeedbackStore()
    assert store._decay_weight(0.0) == pytest.approx(1.0, abs=1e-9)
    assert store._decay_weight(23.9) == pytest.approx(1.0, abs=1e-9)
    assert store._decay_weight(24.0) == pytest.approx(1.0, abs=1e-9)


def test_decay_weight_after_threshold(module):
    """超过 24 小时后按半衰期 12 小时指数衰减"""
    store = module.FeedbackStore()
    # 24 + 12 = 36 小时，权重应为 0.5
    assert store._decay_weight(36.0) == pytest.approx(0.5, abs=1e-9)
    # 24 + 24 = 48 小时，权重应为 0.25
    assert store._decay_weight(48.0) == pytest.approx(0.25, abs=1e-9)


def test_old_feedback_is_decayed(module):
    """过期反馈的权重应衰减（36 小时刚好半衰期，权重 0.5）"""
    store = module.FeedbackStore()
    store._records = []

    now = time.time()
    old_timestamp = now - 36 * 3600  # 36 小时前（半衰期 12 小时，权重 0.5）
    store._records.append({
        "skill_name": "skill-a",
        "query": "old-query",
        "feedback_type": "success",
        "timestamp": old_timestamp,
    })

    adjustment = store.get_adjustments("skill-a")
    expected = store.SUCCESS_DELTA * 0.5
    assert adjustment == pytest.approx(expected, rel=1e-2)
    assert 0 < adjustment < store.SUCCESS_DELTA


def test_persistence_to_jsonl(module):
    """反馈记录应持久化到 JSONL 文件"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        feedback_path = tmp.name

    try:
        store = module.FeedbackStore()
        store.FEEDBACK_FILE = __import__("pathlib").Path(feedback_path)
        store._records = []

        store.record("skill-a", "query-1", "success")
        store.record("skill-b", "query-2", "skip")

        # 重新加载应读到之前写入的记录
        store2 = module.FeedbackStore()
        store2.FEEDBACK_FILE = __import__("pathlib").Path(feedback_path)

        assert any(r["skill_name"] == "skill-a" and r["feedback_type"] == "success" for r in store2._records)
        assert any(r["skill_name"] == "skill-b" and r["feedback_type"] == "skip" for r in store2._records)
    finally:
        os.unlink(feedback_path)
