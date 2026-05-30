"""self_evolution 优化测试 — 反思积累 + 交叉算子 + 弱维度检测"""

import json
import pytest
import importlib.util
import tempfile
from pathlib import Path

# 加载 reflection_store
_spec = importlib.util.spec_from_file_location(
    "reflection_store",
    str(Path(__file__).parent.parent / "core" / "reflection_store.py"),
)
_mod = importlib.util.module_from_spec(_spec)
import sys
sys.modules["reflection_store"] = _mod
_spec.loader.exec_module(_mod)

ReflectionStore = _mod.ReflectionStore
EvolutionReflection = _mod.EvolutionReflection


class TestReflectionStore:
    """反思存储测试"""

    def setup_method(self):
        import tempfile
        self.tmp_path = Path(tempfile.mkdtemp()) / "test_reflections.json"
        self.store = ReflectionStore(path=self.tmp_path)

    def test_add_reflection(self):
        ref = EvolutionReflection(
            timestamp=1717000000,
            failure_type="skill_defect",
            weak_dimension="clarity",
            feedback_summary="触发条件过宽",
            mutation_strategy="simplify",
            score_before=0.65,
            score_after=0.62,
            lesson="简化不应删除错误处理步骤",
        )
        self.store.add_reflection("test-skill", ref)

        reflections = self.store.get_reflections("test-skill")
        assert len(reflections) == 1
        assert reflections[0]["weak_dimension"] == "clarity"

    def test_max_reflections_per_skill(self):
        for i in range(35):
            ref = EvolutionReflection(
                timestamp=1717000000 + i,
                failure_type="skill_defect",
                weak_dimension="accuracy",
                feedback_summary=f"feedback {i}",
                mutation_strategy="structural",
                score_before=0.5,
                score_after=0.6,
                lesson=f"lesson {i}",
            )
            self.store.add_reflection("test-skill", ref)

        reflections = self.store.get_reflections("test-skill", limit=100)
        assert len(reflections) == 30  # MAX_REFLECTIONS_PER_SKILL

    def test_pattern_stats(self):
        for _ in range(5):
            ref = EvolutionReflection(
                timestamp=1717000000,
                failure_type="skill_defect",
                weak_dimension="clarity",
                feedback_summary="模糊",
                mutation_strategy="simplify",
                score_before=0.6,
                score_after=0.55,
                lesson="清晰度问题",
            )
            self.store.add_reflection("test-skill", ref)

        stats = self.store.get_pattern_stats("test-skill")
        assert stats.get("clarity_failures", 0) == 5

    def test_lessons_prompt_empty(self):
        result = self.store.get_lessons_prompt("nonexistent")
        assert result == ""

    def test_lessons_prompt_with_data(self):
        ref = EvolutionReflection(
            timestamp=1717000000,
            failure_type="skill_defect",
            weak_dimension="clarity",
            feedback_summary="触发条件过宽",
            mutation_strategy="simplify",
            score_before=0.65,
            score_after=0.70,
            lesson="简化时保留错误处理",
        )
        self.store.add_reflection("test-skill", ref)

        prompt = self.store.get_lessons_prompt("test-skill")
        assert "历史进化经验" in prompt
        assert "clarity" in prompt
        assert "简化时保留错误处理" in prompt

    def test_persistence(self):
        ref = EvolutionReflection(
            timestamp=1717000000,
            failure_type="skill_defect",
            weak_dimension="accuracy",
            feedback_summary="test",
            mutation_strategy="structural",
            score_before=0.5,
            score_after=0.6,
            lesson="test lesson",
        )
        self.store.add_reflection("persist-skill", ref)

        # 重新加载
        store2 = ReflectionStore(path=self.tmp_path)
        reflections = store2.get_reflections("persist-skill")
        assert len(reflections) == 1

    def test_clear(self):
        ref = EvolutionReflection(
            timestamp=1717000000,
            failure_type="optimization",
            weak_dimension="efficiency",
            feedback_summary="test",
            mutation_strategy="simplify",
            score_before=0.5,
            score_after=0.5,
            lesson="test",
        )
        self.store.add_reflection("clear-skill", ref)
        self.store.clear("clear-skill")
        assert self.store.get_reflections("clear-skill") == []

    def test_get_all_skills(self):
        for name in ["skill-a", "skill-b"]:
            ref = EvolutionReflection(
                timestamp=1717000000,
                failure_type="optimization",
                weak_dimension="efficiency",
                feedback_summary="test",
                mutation_strategy="simplify",
                score_before=0.5,
                score_after=0.5,
                lesson="test",
            )
            self.store.add_reflection(name, ref)

        skills = self.store.get_all_skills()
        assert "skill-a" in skills
        assert "skill-b" in skills

    def test_effective_strategy_tracking(self):
        # 成功变异 (score_after > score_before)
        for _ in range(3):
            ref = EvolutionReflection(
                timestamp=1717000000,
                failure_type="optimization",
                weak_dimension="clarity",
                feedback_summary="test",
                mutation_strategy="structural",
                score_before=0.6,
                score_after=0.7,
                lesson="structural works",
            )
            self.store.add_reflection("strategy-skill", ref)

        # 失败变异 (score_after < score_before)
        ref = EvolutionReflection(
            timestamp=1717000000,
            failure_type="optimization",
            weak_dimension="clarity",
            feedback_summary="test",
            mutation_strategy="simplify",
            score_before=0.7,
            score_after=0.6,
            lesson="simplify failed",
        )
        self.store.add_reflection("strategy-skill", ref)

        stats = self.store.get_pattern_stats("strategy-skill")
        assert stats.get("most_effective_strategy") == "structural"


class TestDetectWeakDim:
    """弱维度检测测试 (静态方法)"""

    def _load_optimizer_class(self):
        """加载 SkillOptimizer 类"""
        spec2 = importlib.util.spec_from_file_location(
            "optimizer",
            str(Path(__file__).parent.parent / "pipeline" / "optimizer.py"),
        )
        mod2 = importlib.util.module_from_spec(spec2)
        try:
            spec2.loader.exec_module(mod2)
        except Exception:
            # 可能缺少 openai 依赖，跳过
            pytest.skip("Cannot load optimizer (missing dependencies)")
        return mod2.SkillOptimizer

    def test_detect_clarity(self):
        cls = self._load_optimizer_class()
        result = cls._detect_weak_dim("指令描述模糊，用户无法理解该做什么")
        assert result == "clarity"

    def test_detect_accuracy(self):
        cls = self._load_optimizer_class()
        result = cls._detect_weak_dim("输出结果不正确，有错误")
        assert result == "accuracy"

    def test_detect_completeness(self):
        cls = self._load_optimizer_class()
        result = cls._detect_weak_dim("步骤缺失，缺少关键的错误处理部分")
        assert result == "completeness"

    def test_detect_efficiency(self):
        cls = self._load_optimizer_class()
        result = cls._detect_weak_dim("流程冗余，效率太低")
        assert result == "efficiency"

    def test_detect_safety(self):
        cls = self._load_optimizer_class()
        result = cls._detect_weak_dim("存在安全风险，可能造成危险操作")
        assert result == "safety"

    def test_detect_unknown(self):
        cls = self._load_optimizer_class()
        result = cls._detect_weak_dim("everything is fine")
        assert result == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
