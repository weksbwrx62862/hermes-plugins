"""EmbedTaskClassifier 单元测试与准确率评估。

覆盖：
  - 基本分类接口与返回值格式
  - 模型加载成功后的语义分类准确率
  - 与关键词启发式的准确率对比
  - 复杂度桶映射与置信度范围
  - 兜底降级路径
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERMES_HOME = Path.home() / ".hermes"
if str(HERMES_HOME) not in sys.path:
    sys.path.insert(0, str(HERMES_HOME))

# 加载 embed_router.py（目录名含连字符）
_embed_path = Path(__file__).resolve().parent.parent / "embed_router.py"
_embed_spec = importlib.util.spec_from_file_location("embed_router", str(_embed_path))
embed_router = importlib.util.module_from_spec(_embed_spec)
sys.modules["embed_router"] = embed_router
_embed_spec.loader.exec_module(embed_router)

EmbedTaskClassifier = embed_router.EmbedTaskClassifier
_KeywordFallback = embed_router._KeywordFallback

# 加载测试语料
_script_dir = HERMES_HOME / "scripts"
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from router_test_corpus import CORPUS, TASK_TYPES  # noqa: E402

# 加载 model-router 用于关键词启发式基线
import importlib as _importlib  # noqa: E402

model_router = _importlib.import_module("plugins.model-router")


@pytest.fixture(scope="module")
def classifier():
    """模块级 fixture：初始化并等待嵌入模型加载完成。"""
    clf = EmbedTaskClassifier(lazy=True)
    ready = clf.wait_ready(timeout=90.0)
    if not ready:
        pytest.fail("嵌入模型在 90 秒内未加载完成")
    if not clf.is_ready():
        pytest.fail("嵌入模型加载失败，无法继续语义分类测试")
    return clf


class TestEmbedRouterInterface:
    """接口正确性测试。"""

    def test_classify_returns_tuple(self, classifier):
        """classify 返回三元组。"""
        result = classifier.classify("写一个 Python 函数计算斐波那契数列")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_task_type_in_valid_set(self, classifier):
        """预测任务类型在合法集合内。"""
        task_type, _, _ = classifier.classify("总结这篇论文")
        assert task_type in TASK_TYPES

    def test_complexity_bucket_range(self, classifier):
        """复杂度桶为 1-5 的整数。"""
        _, complexity, _ = classifier.classify("设计一个高并发系统")
        assert isinstance(complexity, int)
        assert 1 <= complexity <= 5

    def test_confidence_range(self, classifier):
        """置信度在 [0, 1] 之间。"""
        _, _, confidence = classifier.classify("今天天气怎么样")
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_empty_query(self, classifier):
        """空查询返回兜底结果。"""
        task_type, complexity, confidence = classifier.classify("")
        assert task_type in TASK_TYPES
        assert 1 <= complexity <= 5
        assert 0.0 <= confidence <= 1.0

    def test_reload_examples(self, classifier):
        """热更新示例库成功。"""
        assert classifier.reload_examples() is True


class TestKeywordFallback:
    """关键词启发式兜底测试。"""

    def test_fallback_task_type(self):
        """兜底分类器返回合法任务类型。"""
        assert _KeywordFallback.detect_task_type("今天北京天气怎么样？") == "simple_qa"
        assert _KeywordFallback.detect_task_type("把这段评论分类为正面或负面") == "classify"
        assert _KeywordFallback.detect_task_type("从合同中提取签约日期") == "extract"

    def test_fallback_complexity_range(self):
        """兜底复杂度在 1-5 之间。"""
        complexity = _KeywordFallback.estimate_complexity("设计一个分布式系统")
        assert 1 <= complexity <= 5


class TestAccuracy:
    """准确率对比测试。"""

    def test_semantic_accuracy_beat_keyword_baseline(self, classifier):
        """语义分类准确率应比关键词启发式提升 ≥10 个百分点，且 ≥86%。"""
        keyword_correct = 0
        semantic_correct = 0
        complexity_errors_keyword = []
        complexity_errors_semantic = []

        for item in CORPUS:
            text = item["text"]
            expected_type = item["expected_type"]
            expected_complexity = item["expected_complexity"]

            pred_type_keyword = model_router._detect_task_type(text)
            pred_complexity_keyword = model_router._estimate_complexity(text)

            pred_type_semantic, pred_complexity_semantic, _ = classifier.classify(text)

            if pred_type_keyword == expected_type:
                keyword_correct += 1
            if pred_type_semantic == expected_type:
                semantic_correct += 1

            complexity_errors_keyword.append(abs(pred_complexity_keyword - expected_complexity))
            complexity_errors_semantic.append(abs(pred_complexity_semantic - expected_complexity))

        total = len(CORPUS)
        keyword_acc = keyword_correct / total
        semantic_acc = semantic_correct / total

        keyword_mae = sum(complexity_errors_keyword) / total
        semantic_mae = sum(complexity_errors_semantic) / total

        print(f"\n关键词启发式：准确率={keyword_acc*100:.2f}%, 复杂度 MAE={keyword_mae:.2f}")
        print(f"语义分类：准确率={semantic_acc*100:.2f}%, 复杂度 MAE={semantic_mae:.2f}")
        print(f"提升：{(semantic_acc - keyword_acc)*100:.2f} 个百分点")

        assert semantic_acc >= 0.86, f"语义分类准确率 {semantic_acc*100:.2f}% 低于 86%"
        assert semantic_acc - keyword_acc >= 0.10, (
            f"语义分类准确率 {semantic_acc*100:.2f}% 未比关键词启发式 {keyword_acc*100:.2f}% 提升 10 个百分点"
        )
