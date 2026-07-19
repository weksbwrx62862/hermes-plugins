"""
A/B 评分矩阵 + 统计显著性检验

借鉴 Skill-insight 的 ab-scoring.ts 设计：
  - 三维权度：capability（能力）+ cost（成本）+ stability（稳定性）
  - 加权合成：capability × 0.55 + cost × 0.35 + stability × 0.10
  - 硬门控：任一维度低于阈值则 reject
  - 统计显著性：Welch's t-test 检验差异是否显著

用于替代简单的 baseline_score vs evolved_score 比较，
提供更精细的版本间质量对比。
"""

import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DimensionScore:
    """单维度评分结果。"""
    name: str
    score_a: Optional[float] = None  # baseline
    score_b: Optional[float] = None  # evolved
    delta: Optional[float] = None    # b - a
    verdict: str = "unavailable"     # good | warning | reject | unavailable
    label: str = ""
    tone: str = "gray"               # green | amber | red | gray


@dataclass
class AbScoringResult:
    """A/B 评分完整结果。"""
    # 总分
    total_score: Optional[float] = None
    grade: str = "insufficient"      # excellent | good | pass | weak | fail | insufficient
    grade_label: str = ""
    decision: str = "insufficient"   # direct-release | monitor-release | reject | insufficient
    decision_label: str = ""
    allow_release: bool = False
    reject_category: Optional[str] = None  # capability | cost | stability | None

    # 三维度
    capability: DimensionScore = field(default_factory=lambda: DimensionScore(name="capability"))
    cost: DimensionScore = field(default_factory=lambda: DimensionScore(name="cost"))
    stability: DimensionScore = field(default_factory=lambda: DimensionScore(name="stability"))

    # 统计信息
    sample_size: int = 0
    confidence: str = "low"          # low | medium | high
    p_value: Optional[float] = None
    significant: bool = False

    # 详细信息
    details: dict = field(default_factory=dict)


# ── 评分策略配置 ──────────────────────────────────────────

@dataclass
class ScoringPolicy:
    """评分策略。"""
    version: str = "hermes-scoring-v1.0"
    min_sample_size: int = 3
    recommended_sample_size: int = 10

    # 权重
    weights_capability: float = 0.55
    weights_cost: float = 0.35
    weights_stability: float = 0.10

    # 硬门控阈值
    capability_ceiling: float = -0.05   # 能力退化不超过 5%
    cost_ceiling: float = 0.30          # 成本增加不超过 30%
    stability_ceiling: float = 0.60     # 稳定性分数不低于 0.60

    # 好/警告阈值
    capability_good_pp: float = 0.05    # 能力提升 5% 以上为好
    cost_good_pct: float = 0.10         # 成本降低 10% 以上为好
    cost_warning_pct: float = 0.20      # 成本增加 20% 以上为警告

    # 显著性
    significance_level: float = 0.05    # p < 0.05 为显著


DEFAULT_POLICY = ScoringPolicy()


# ── 统计工具 ──────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    """计算均值。"""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: list[float], mean: float) -> float:
    """计算样本方差。"""
    if len(values) < 2:
        return 0.0
    return sum((x - mean) ** 2 for x in values) / (len(values) - 1)


def _welch_t_test(
    sample_a: list[float],
    sample_b: list[float],
) -> tuple[float, float]:
    """
    Welch's t-test（不假设等方差）。

    Returns:
        (t_statistic, p_value)
    """
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0

    mean_a, mean_b = _mean(sample_a), _mean(sample_b)
    var_a, var_b = _variance(sample_a, mean_a), _variance(sample_b, mean_b)

    # 标准误差
    se = math.sqrt(var_a / n_a + var_b / n_b) if (var_a / n_a + var_b / n_b) > 0 else 1e-10

    # t 统计量
    t_stat = (mean_b - mean_a) / se

    # 自由度（Welch-Satterthwaite 方程）
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = (
        ((var_a / n_a) ** 2) / (n_a - 1) +
        ((var_b / n_b) ** 2) / (n_b - 1)
    ) if ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)) > 0 else 1e-10
    df = numerator / denominator

    # 近似 p 值（双尾检验，使用正态近似当 df > 30）
    if df > 30:
        # 正态近似
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    else:
        # 简化的 t 分布近似
        p_value = 2 * (1 - _t_cdf(abs(t_stat), df))

    return t_stat, p_value


def _normal_cdf(x: float) -> float:
    """标准正态分布的 CDF（近似）。"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _t_cdf(t: float, df: float) -> float:
    """t 分布的 CDF（近似，使用正态近似当 df 较大时）。"""
    if df > 30:
        return _normal_cdf(t)
    # 简化近似：使用 Abramowitz and Stegun 近似
    x = df / (df + t * t)
    if x <= 0 or x >= 1:
        return 0.5
    # 不完全 Beta 函数的近似
    a = df / 2
    b = 0.5
    # 使用简单的近似
    return 1 - 0.5 * x ** a


# ── 评分引擎 ──────────────────────────────────────────────

class AbScorer:
    """
    A/B 评分引擎。

    三维权重评分 + 统计显著性检验 + 硬门控。
    """

    def __init__(self, policy: ScoringPolicy = None):
        self.policy = policy or DEFAULT_POLICY

    def score(
        self,
        baseline_scores: list[float],
        evolved_scores: list[float],
        baseline_costs: list[float] = None,
        evolved_costs: list[float] = None,
        baseline_durations: list[float] = None,
        evolved_durations: list[float] = None,
        invoke_rate: float = None,
        variance_score: float = None,
    ) -> AbScoringResult:
        """
        执行 A/B 评分。

        Args:
            baseline_scores: 基线能力分数列表
            evolved_scores: 进化能力分数列表
            baseline_costs: 基线 Token 成本列表
            evolved_costs: 进化 Token 成本列表
            baseline_durations: 基线耗时列表
            evolved_durations: 进化耗时列表
            invoke_rate: 技能触发率（0-1）
            variance_score: 方差稳定性分数（0-1）

        Returns:
            AbScoringResult
        """
        result = AbScoringResult()
        result.sample_size = min(len(baseline_scores), len(evolved_scores))

        if result.sample_size < self.policy.min_sample_size:
            result.decision = "insufficient"
            result.decision_label = "样本不足，无法评分"
            return result

        # ── 能力维度 ──────────────────────────────────────
        cap = self._score_capability(baseline_scores, evolved_scores)
        result.capability = cap

        # ── 成本维度 ──────────────────────────────────────
        if baseline_costs and evolved_costs:
            cost = self._score_cost(baseline_costs, evolved_costs)
        elif baseline_durations and evolved_durations:
            cost = self._score_cost(baseline_durations, evolved_durations)
        else:
            cost = DimensionScore(name="cost", verdict="unavailable", label="无成本数据")
        result.cost = cost

        # ── 稳定性维度 ────────────────────────────────────
        stab = self._score_stability(invoke_rate, variance_score)
        result.stability = stab

        # ── 统计显著性 ────────────────────────────────────
        t_stat, p_value = _welch_t_test(baseline_scores, evolved_scores)
        result.p_value = p_value
        result.significant = p_value < self.policy.significance_level

        # 置信度
        if result.sample_size >= self.policy.recommended_sample_size:
            result.confidence = "high"
        elif result.sample_size >= 5:
            result.confidence = "medium"
        else:
            result.confidence = "low"

        # ── 加权总分 ──────────────────────────────────────
        cap_score = cap.score_b if cap.score_b is not None else 0.0
        cost_score = cost.score_b if cost.score_b is not None else 0.5  # 已归一化
        stab_score = stab.score_b if stab.score_b is not None else 0.5

        result.total_score = (
            self.policy.weights_capability * cap_score +
            self.policy.weights_cost * cost_score +
            self.policy.weights_stability * stab_score
        )

        # ── 等级和决策 ────────────────────────────────────
        result.grade, result.grade_label = self._grade(result.total_score)
        result.decision, result.decision_label = self._decide(result, cap, cost, stab)
        result.allow_release = result.decision in ("direct-release", "monitor-release")

        # 硬门控拒绝原因
        if result.decision == "reject":
            if cap.verdict == "reject":
                result.reject_category = "capability"
            elif cost.verdict == "reject":
                result.reject_category = "cost"
            elif stab.verdict == "reject":
                result.reject_category = "stability"

        return result

    def _score_capability(
        self, baseline: list[float], evolved: list[float]
    ) -> DimensionScore:
        """能力维度评分。"""
        mean_a = _mean(baseline)
        mean_b = _mean(evolved)
        delta = mean_b - mean_a
        delta_pp = delta  # percentage points

        dim = DimensionScore(
            name="capability",
            score_a=mean_a,
            score_b=mean_b,
            delta=delta,
        )

        if delta_pp >= self.policy.capability_good_pp:
            dim.verdict = "good"
            dim.label = f"能力提升 +{delta_pp:.1%}"
            dim.tone = "green"
        elif delta_pp >= self.policy.capability_ceiling:
            dim.verdict = "warning"
            dim.label = f"能力变化 {delta_pp:+.1%}"
            dim.tone = "amber"
        else:
            dim.verdict = "reject"
            dim.label = f"能力退化 {delta_pp:+.1%}"
            dim.tone = "red"

        return dim

    def _score_cost(
        self, baseline: list[float], evolved: list[float]
    ) -> DimensionScore:
        """成本维度评分。"""
        mean_a = _mean(baseline)
        mean_b = _mean(evolved)

        if mean_a <= 0:
            return DimensionScore(name="cost", verdict="unavailable", label="基线成本为0")

        delta_pct = (mean_b - mean_a) / mean_a

        # 归一化到 0-1 范围（成本越低越好，所以用 1 - ratio）
        ratio = mean_b / mean_a if mean_a > 0 else 1.0
        normalized = max(0, min(1, 1 - delta_pct))  # 降成本→高分，增成本→低分

        dim = DimensionScore(
            name="cost",
            score_a=mean_a,
            score_b=normalized,
            delta=delta_pct,
        )

        if delta_pct <= -self.policy.cost_good_pct:
            dim.verdict = "good"
            dim.label = f"成本降低 {abs(delta_pct):.1%}"
            dim.tone = "green"
        elif delta_pct <= self.policy.cost_warning_pct:
            dim.verdict = "warning"
            dim.label = f"成本变化 {delta_pct:+.1%}"
            dim.tone = "amber"
        else:
            dim.verdict = "reject"
            dim.label = f"成本增加 {delta_pct:+.1%}"
            dim.tone = "red"

        return dim

    def _score_stability(
        self, invoke_rate: Optional[float], variance_score: Optional[float]
    ) -> DimensionScore:
        """稳定性维度评分。"""
        dim = DimensionScore(name="stability")

        if invoke_rate is None and variance_score is None:
            dim.verdict = "unavailable"
            dim.label = "无稳定性数据"
            return dim

        # 综合稳定性分数
        scores = []
        if invoke_rate is not None:
            scores.append(invoke_rate)
        if variance_score is not None:
            scores.append(1.0 - variance_score)  # 方差越低越好

        stability = _mean(scores) if scores else 0.5
        dim.score_b = stability

        if stability >= 0.8:
            dim.verdict = "good"
            dim.label = f"稳定性良好 {stability:.2f}"
            dim.tone = "green"
        elif stability >= self.policy.stability_ceiling:
            dim.verdict = "warning"
            dim.label = f"稳定性一般 {stability:.2f}"
            dim.tone = "amber"
        else:
            dim.verdict = "reject"
            dim.label = f"稳定性不足 {stability:.2f}"
            dim.tone = "red"

        return dim

    def _grade(self, score: Optional[float]) -> tuple[str, str]:
        """评分等级。"""
        if score is None:
            return "insufficient", "数据不足"
        if score >= 0.85:
            return "excellent", "优秀"
        if score >= 0.70:
            return "good", "良好"
        if score >= 0.55:
            return "pass", "及格"
        if score >= 0.40:
            return "weak", "较弱"
        return "fail", "失败"

    def _decide(
        self,
        result: AbScoringResult,
        cap: DimensionScore,
        cost: DimensionScore,
        stab: DimensionScore,
    ) -> tuple[str, str]:
        """发布决策。"""
        # 硬门控检查
        if cap.verdict == "reject":
            return "reject", f"能力退化 ({cap.label})"
        if cost.verdict == "reject":
            return "reject", f"成本过高 ({cost.label})"
        if stab.verdict == "reject":
            return "reject", f"稳定性不足 ({stab.label})"

        # 基于总分决策
        if result.total_score is None:
            return "insufficient", "数据不足"
        if result.total_score >= 0.70 and result.significant:
            return "direct-release", "显著改进，可直接发布"
        if result.total_score >= 0.55:
            return "monitor-release", "改进但需监控"
        return "reject", "改进不足"


def format_ab_report(result: AbScoringResult) -> str:
    """格式化 A/B 评分报告。"""
    lines = [
        "=" * 50,
        "A/B 评分报告",
        "=" * 50,
        f"总分: {result.total_score:.3f}" if result.total_score else "总分: N/A",
        f"等级: {result.grade_label} ({result.grade})",
        f"决策: {result.decision_label} ({result.decision})",
        f"允许发布: {'✅' if result.allow_release else '❌'}",
        "",
        "--- 三维度评分 ---",
    ]

    for dim in [result.capability, result.cost, result.stability]:
        icon = {"green": "🟢", "amber": "🟡", "red": "🔴", "gray": "⚪"}.get(dim.tone, "⚪")
        lines.append(f"  {icon} {dim.name}: {dim.label}")

    lines.append("")
    lines.append("--- 统计信息 ---")
    lines.append(f"  样本量: {result.sample_size}")
    lines.append(f"  置信度: {result.confidence}")
    if result.p_value is not None:
        lines.append(f"  p值: {result.p_value:.4f}")
        lines.append(f"  显著性: {'✅ 显著' if result.significant else '❌ 不显著'}")

    if result.reject_category:
        lines.append(f"\n  ⚠️ 拒绝原因: {result.reject_category}")

    return "\n".join(lines)
