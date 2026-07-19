"""
增强回归测试门限 (RegressionGate)

灵感来源: Self-Harness 论文的保守接受策略
核心规则:
  1. held-in 不退步 (允许小幅度波动, 阈值 5%)
  2. held-out 不退步 (严格不允许)
  3. 至少一个分割的通过数提升
  4. 不能只在一个分割变好、另一个变差（防止过拟合）

与现有系统的关系:
  - 现有 evolution_manager.py Stage 7 已有基础 holdout 检查
  - 本模块增强了门限策略，增加 held-in 检查和 anti-overfitting 约束
  - 可作为 EvolutionManager 的可选增强层

额外检查（基于 Self-Harness 论文）:
  - 失败模式覆盖检查: 进化后的版本是否解决了已知的高频失败模式
  - harness 表面一致性: 修改是否影响了正确的 harness 表面
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RegressionResult:
    """回归测试结果"""
    passed: bool
    held_in_passed: bool
    held_out_passed: bool
    improvement_detected: bool
    anti_overfitting: bool      # True = 两个分割都未退步
    reason: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "held_in_passed": self.held_in_passed,
            "held_out_passed": self.held_out_passed,
            "improvement_detected": self.improvement_detected,
            "anti_overfitting": self.anti_overfitting,
            "reason": self.reason,
            "details": self.details,
        }


@dataclass
class EvolutionScores:
    """一次进化的完整评分"""
    skill_name: str
    baseline_val_score: float      # 基线在 val (held-in) 上的分数
    baseline_holdout_score: float  # 基线在 holdout (held-out) 上的分数
    evolved_val_score: float       # 进化版在 val 上的分数
    evolved_holdout_score: float   # 进化版在 holdout 上的分数
    constraint_passed: bool = True
    constraint_failures: list[str] = field(default_factory=list)


class RegressionGate:
    """
    Self-Harness 风格的回归测试门限

    使用方法:
        gate = RegressionGate()
        result = gate.evaluate(scores)
        if result.passed:
            # 安全部署
        else:
            # 拒绝部署，记录原因
    """

    def __init__(
        self,
        held_in_tolerance: float = 0.05,   # held-in 允许退步 5%
        held_out_tolerance: float = 0.0,    # held-out 严格不允许退步
        min_improvement: float = 0.001,     # 最小提升阈值
    ):
        self.held_in_tolerance = held_in_tolerance
        self.held_out_tolerance = held_out_tolerance
        self.min_improvement = min_improvement

    def evaluate(self, scores: EvolutionScores) -> RegressionResult:
        """
        Self-Harness 三条件门限:
          1. held-in 不退步 (容忍阈值内)
          2. held-out 不退步 (严格)
          3. 至少一个分割提升
          4. 不能只在一个分割变好、另一个变差
        """
        val_delta = scores.evolved_val_score - scores.baseline_val_score
        holdout_delta = scores.evolved_holdout_score - scores.baseline_holdout_score

        # 条件 1: held-in 不退步
        held_in_threshold = -self.held_in_tolerance * scores.baseline_val_score
        held_in_passed = val_delta >= held_in_threshold

        # 条件 2: held-out 不退步
        held_out_threshold = -self.held_out_tolerance * max(scores.baseline_holdout_score, 0.01)
        held_out_passed = holdout_delta >= held_out_threshold

        # 条件 3: 至少一个分割有提升
        improvement_detected = (
            val_delta > self.min_improvement or
            holdout_delta > self.min_improvement
        )

        # 条件 4: anti-overfitting — 不能只在一个分割变好另一个变差
        # (一个明显变好但另一个明显变差 = 过拟合信号)
        anti_overfitting = True
        if val_delta > 0.1 and holdout_delta < -0.05:
            anti_overfitting = False  # val 涨但 holdout 跌 = 过拟合
        if holdout_delta > 0.1 and val_delta < -0.05:
            anti_overfitting = False  # holdout 涨但 val 跌 = 不稳定

        # 约束检查
        constraint_ok = scores.constraint_passed

        # 综合判定
        all_passed = (
            held_in_passed and
            held_out_passed and
            improvement_detected and
            anti_overfitting and
            constraint_ok
        )

        # 生成原因
        reasons = []
        if not held_in_passed:
            reasons.append(f"held-in 退步 {val_delta:+.3f} (阈值 {held_in_threshold:+.3f})")
        if not held_out_passed:
            reasons.append(f"held-out 退步 {holdout_delta:+.3f}")
        if not improvement_detected:
            reasons.append("无显著提升")
        if not anti_overfitting:
            reasons.append("过拟合信号: 一个分割变好但另一个变差")
        if not constraint_ok:
            reasons.append(f"约束失败: {', '.join(scores.constraint_failures)}")

        reason = "; ".join(reasons) if reasons else "全部通过"

        result = RegressionResult(
            passed=all_passed,
            held_in_passed=held_in_passed,
            held_out_passed=held_out_passed,
            improvement_detected=improvement_detected,
            anti_overfitting=anti_overfitting,
            reason=reason,
            details={
                "val_delta": round(val_delta, 4),
                "holdout_delta": round(holdout_delta, 4),
                "baseline_val": round(scores.baseline_val_score, 4),
                "baseline_holdout": round(scores.baseline_holdout_score, 4),
                "evolved_val": round(scores.evolved_val_score, 4),
                "evolved_holdout": round(scores.evolved_holdout_score, 4),
                "held_in_threshold": round(held_in_threshold, 4),
                "held_out_threshold": round(held_out_threshold, 4),
            },
        )

        if all_passed:
            logger.info(
                f"[regression_gate] ✅ PASSED: "
                f"val {scores.baseline_val_score:.3f}→{scores.evolved_val_score:.3f} "
                f"holdout {scores.baseline_holdout_score:.3f}→{scores.evolved_holdout_score:.3f}"
            )
        else:
            logger.warning(f"[regression_gate] ❌ REJECTED: {reason}")

        return result

    def evaluate_from_evolution_result(
        self,
        skill_name: str,
        baseline_val: float,
        baseline_holdout: float,
        evolved_val: float,
        evolved_holdout: float,
        constraint_passed: bool = True,
        constraint_failures: Optional[list[str]] = None,
    ) -> RegressionResult:
        """从 evolution_manager 的结果构造评估"""
        scores = EvolutionScores(
            skill_name=skill_name,
            baseline_val_score=baseline_val,
            baseline_holdout_score=baseline_holdout,
            evolved_val_score=evolved_val,
            evolved_holdout_score=evolved_holdout,
            constraint_passed=constraint_passed,
            constraint_failures=constraint_failures or [],
        )
        return self.evaluate(scores)


class FailurePatternGate:
    """
    基于失败模式的额外门限

    检查进化后的版本是否解决了已知的高频失败模式。
    这是 Self-Harness 论文的 "evidence-driven" 思路。
    """

    def __init__(self, failure_tracker=None):
        self.tracker = failure_tracker

    def check(
        self,
        skill_name: str,
        evolved_text: str,
        days: int = 30,
    ) -> dict:
        """
        检查进化版本是否覆盖了已知弱点

        Returns:
            {
                "has_weaknesses": True/False,
                "top_patterns": [...],
                "suggestions": [...],
            }
        """
        if not self.tracker:
            return {"has_weaknesses": False, "top_patterns": [], "suggestions": []}

        weaknesses = self.tracker.get_skill_weaknesses(skill_name, days=days)

        if not weaknesses:
            return {"has_weaknesses": False, "top_patterns": [], "suggestions": []}

        suggestions = []
        for w in weaknesses[:3]:
            pattern = w["pattern"]
            surface = w["harness_surface"]
            count = w["count"]

            # 检查进化文本是否包含了针对此模式的指导
            if pattern == "tool_timeout" and "超时" not in evolved_text and "timeout" not in evolved_text.lower():
                suggestions.append(f"建议在 skill 中增加超时处理指导 (此模式出现 {count} 次)")
            elif pattern == "tool_loop" and "重试" not in evolved_text and "retry" not in evolved_text.lower():
                suggestions.append(f"建议增加重试策略 (此模式出现 {count} 次)")
            elif pattern == "premature_done" and "验证" not in evolved_text and "verify" not in evolved_text.lower():
                suggestions.append(f"建议增加产物验证步骤 (此模式出现 {count} 次)")
            elif pattern == "missing_artifact" and "创建" not in evolved_text and "create" not in evolved_text.lower():
                suggestions.append(f"建议增加文件创建检查点 (此模式出现 {count} 次)")

        return {
            "has_weaknesses": bool(weaknesses),
            "top_patterns": [w["pattern"] for w in weaknesses[:5]],
            "suggestions": suggestions,
        }
