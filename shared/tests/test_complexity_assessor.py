"""ComplexityAssessor 单元测试。"""

import math

import pytest

from shared.complexity_assessor import ComplexityAssessor


@pytest.fixture
def assessor():
    return ComplexityAssessor()


class TestScoreMapping:
    """验证 score_10 与 score_5 的映射关系。"""

    def test_score_10_to_score_5_formula(self):
        cases = [
            (1.0, 1), (1.5, 1), (2.0, 1),
            (2.01, 2), (3.0, 2), (4.0, 2),
            (4.01, 3), (5.0, 3), (6.0, 3),
            (6.01, 4), (7.0, 4), (8.0, 4),
            (8.01, 5), (9.0, 5), (10.0, 5),
        ]
        for score_10, expected_score_5 in cases:
            assert math.ceil(score_10 / 2) == expected_score_5

    def test_assess_returns_required_fields_and_ranges(self, assessor):
        result = assessor.assess("分析并优化这段代码的性能")
        assert "score_10" in result
        assert "score_5" in result
        assert "task_type" in result
        assert "confidence" in result
        assert "features" in result

        assert 1.0 <= result["score_10"] <= 10.0
        assert isinstance(result["score_5"], int)
        assert 1 <= result["score_5"] <= 5
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["score_5"] == math.ceil(result["score_10"] / 2)

    def test_score_5_boundary_values(self, assessor):
        # 极简单查询应落在最低档
        simple = assessor.assess("你好")
        assert simple["score_5"] in (1, 2)

        # 极复杂查询应落在最高档
        complex_query = (
            "设计一个完整的微服务电商平台，包含用户认证、订单、支付、库存服务，"
            "需要多角色团队协作，并行开发多个模块，并编写详细的架构设计文档"
        )
        complex_result = assessor.assess(complex_query)
        assert complex_result["score_5"] == 5


class TestModelRouterCompatibility:
    """与 model-router _estimate_complexity 的偏差测试（±1 级）。"""

    QUERIES = [
        "你好",
        "请协作完成一个数据分析报告",
        "请并行处理以下多个任务，并整合结果",
        "设计并实现一个带单元测试的 REST API 服务，包含用户认证、数据验证和错误处理，需要团队协作完成",
        "设计一个完整的微服务电商平台，包含用户认证、订单、支付、库存服务，需要多角色团队协作，并行开发多个模块，并编写详细的架构设计文档",
    ]

    def test_score_5_within_tolerance(self, assessor, model_router_module):
        estimate_complexity = model_router_module._estimate_complexity
        for query in self.QUERIES:
            unified = assessor.assess(query)
            router_score = estimate_complexity(query)
            assert abs(unified["score_5"] - router_score) <= 1, (
                f"查询 '{query[:30]}...': unified score_5={unified['score_5']} "
                f"vs model-router={router_score}"
            )


class TestAMACompatibility:
    """与 AMA TaskComplexityAssessor 的偏差测试（±1.5 级）。"""

    QUERIES = [
        "你好",
        "请协作完成一个数据分析报告",
        "请并行处理以下多个任务，并整合结果",
        "设计并实现一个带单元测试的 REST API 服务，包含用户认证、数据验证和错误处理，需要团队协作完成",
        "设计一个完整的微服务电商平台，包含用户认证、订单、支付、库存服务，需要多角色团队协作，并行开发多个模块，并编写详细的架构设计文档",
    ]

    def test_score_10_within_tolerance(self, assessor, ama_assessor):
        for query in self.QUERIES:
            unified = assessor.assess(query)
            ama_score = ama_assessor.assess(query)["complexity_score"]
            assert abs(unified["score_10"] - ama_score) <= 1.5, (
                f"查询 '{query[:30]}...': unified score_10={unified['score_10']} "
                f"vs AMA={ama_score}"
            )


class TestFeatureExtraction:
    """验证特征抽取行为。"""

    def test_negation_filter(self, assessor):
        result = assessor.assess("这里不需要验证，直接输出即可")
        assert result["features"]["has_explicit_verification"] is False

    def test_parallelism_feature(self, assessor):
        result = assessor.assess("请并行处理以下任务")
        assert result["features"]["needs_parallelism"] is True

    def test_output_format_features(self, assessor):
        result = assessor.assess("请输出一份对比表格")
        assert result["features"]["requires_table"] is True
        assert result["features"]["requires_comparison"] is True

    def test_task_length_feature(self, assessor):
        result = assessor.assess("短")
        assert result["features"]["task_length"] == 1


class TestLLMRefinementInterface:
    """验证 LLM 二次精修接口默认不调用、可注入。"""

    def test_default_no_llm_call(self, assessor):
        result = assessor.assess("分析并优化这段代码")
        assert result.get("llm_refined") is None or result.get("llm_refined") is False

    def test_llm_not_triggered_when_confidence_high(self, assessor):
        # 高置信度时不应进入精修分支
        result = assessor.assess(
            "设计一个完整的微服务电商平台",
            enable_llm=True,
        )
        assert result.get("llm_refined") is None or result.get("llm_refined") is False

    def test_custom_llm_refine_fn(self, assessor):
        def refine_fn(result, query, context):
            return {**result, "score_10": 8.0}

        assessor.llm_refine_fn = refine_fn
        assessor.enable_llm = True
        assessor.llm_refinement_threshold = 1.0  # 强制触发

        result = assessor.assess("简单问题")
        assert result["llm_refined"] is True
        assert result["score_10"] == 8.0
        assert result["score_5"] == 4

    def test_llm_refine_fn_invalid_result_fallback(self, assessor):
        def bad_refine_fn(result, query, context):
            return {"invalid": True}

        assessor.llm_refine_fn = bad_refine_fn
        assessor.enable_llm = True
        assessor.llm_refinement_threshold = 1.0

        result = assessor.assess("简单问题")
        assert result["llm_refined"] is False
        assert "score_10" in result
