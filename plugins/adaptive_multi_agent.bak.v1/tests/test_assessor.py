"""TaskComplexityAssessor 单元测试。"""

import pytest

from adaptive_multi_agent.assessor import TaskComplexityAssessor
from adaptive_multi_agent.subagent import AgentMode


@pytest.fixture
def assessor():
    return TaskComplexityAssessor()


class TestExtractFeatures:
    """测试特征抽取对关键词的识别。"""

    def test_explicit_verification_keyword(self, assessor):
        features = assessor._extract_features("完成任务后需要验证结果", None)
        assert features["has_explicit_verification"] is True

    def test_parallelism_keyword(self, assessor):
        features = assessor._extract_features("请并行处理以下多个请求", None)
        assert features["needs_parallelism"] is True

    def test_roles_keyword(self, assessor):
        features = assessor._extract_features("需要多个角色分工协作", None)
        assert features["has_roles"] is True

    def test_event_driven_keyword(self, assessor):
        features = assessor._extract_features("当事件发生时触发处理流程", None)
        assert features["is_event_driven"] is True

    def test_collaboration_keyword(self, assessor):
        features = assessor._extract_features("请与团队共同协作完成", None)
        assert features["needs_collaboration"] is True

    def test_iterative_potential_keyword(self, assessor):
        features = assessor._extract_features("请迭代改进当前方案", None)
        assert features["iterative_potential"] is True

    def test_shared_knowledge_keyword(self, assessor):
        features = assessor._extract_features("需要整合知识库中的信息", None)
        assert features["requires_shared_knowledge"] is True

    def test_reasoning_depth_signal(self, assessor):
        features = assessor._extract_features("请推导该结论的因果逻辑", None)
        assert features["reasoning_depth"] is True

    def test_cross_reference_signal(self, assessor):
        features = assessor._extract_features("请结合多个领域的资料进行关联", None)
        assert features["cross_reference"] is True


class TestNegationFilter:
    """测试否定前缀过滤。"""

    def test_not_need_verification(self, assessor):
        # "不需要验证" 不应触发 has_explicit_verification
        features = assessor._extract_features("这里不需要验证，直接输出即可", None)
        assert features["has_explicit_verification"] is False

    def test_no_check(self, assessor):
        features = assessor._extract_features("无需检查细节", None)
        assert features["has_explicit_verification"] is False

    def test_positive_verification(self, assessor):
        features = assessor._extract_features("需要验证正确性", None)
        assert features["has_explicit_verification"] is True

    def test_direct_negation(self, assessor):
        features = assessor._extract_features("不验证", None)
        assert features["has_explicit_verification"] is False


class TestScoreRanges:
    """测试复杂度分数在简单/中等/复杂任务上的范围。"""

    def test_simple_task_score(self, assessor):
        score = assessor.assess("写一个 hello world 函数")["complexity_score"]
        assert 1.0 <= score <= 3.0

    def test_moderate_task_score(self, assessor):
        score = assessor.assess(
            "设计并实现一个带单元测试的 REST API 服务，包含用户认证、数据验证和错误处理，需要团队协作完成"
        )["complexity_score"]
        assert 3.0 < score <= 6.0

    def test_complex_task_score(self, assessor):
        desc = (
            "设计一个完整的微服务电商平台，包含用户认证、订单、支付、库存服务，"
            "需要多角色团队协作，并行开发多个模块，并编写详细的架构设计文档"
        )
        score = assessor.assess(desc)["complexity_score"]
        assert score > 6.0


class TestExternalAssessment:
    """测试外部评估直接透传。"""

    def test_external_assessment_passthrough(self, assessor):
        external = {
            "complexity_score": 7.5,
            "task_type": "analysis",
            "features": {"has_explicit_verification": True},
            "recommended_mode": "agent_teams",
        }
        result = assessor.assess("任意描述", external_assessment=external)
        assert result["complexity_score"] == 7.5
        assert result["task_type"] == "analysis"
        assert result["features"]["has_explicit_verification"] is True
        assert result["recommended_mode"] == "agent_teams"
        assert "estimated_tokens" in result

    def test_external_assessment_default_fill(self, assessor):
        result = assessor.assess("描述", external_assessment={"task_type": "default"})
        assert result["complexity_score"] == 3.0
        assert result["recommended_mode"] == "orchestrator_subagent"
