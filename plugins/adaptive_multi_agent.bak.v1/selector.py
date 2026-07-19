from __future__ import annotations

# 本模块负责模式选择引擎。
# 基于 Thompson Sampling、冷启动先验、成本感知与规则过滤，从评估结果中选择最合适的执行模式。

import logging
import random
import sys
from typing import Any, Dict, List, Optional

from .persistence import load_performance, save_performance
from .subagent import AgentMode, CircuitBreaker

logger = logging.getLogger(__name__)

def _read_from_plugin_context(session_id: str, key: str, default: Any = None) -> Any:
    """从 PluginContext 读取共享状态（若可用）。否则返回 default。"""
    try:
        import sys
        if "plugin_orchestrator.context" not in sys.modules:
            return None
        ctx_mod = sys.modules["plugin_orchestrator.context"]
        get_ctx = getattr(ctx_mod, "get_context", None)
        if get_ctx is None:
            return None
        ctx = get_ctx(session_id) if session_id else None
        if ctx is None:
            return None
        return ctx.shared_get(key, default)
    except Exception:
        return None


def _write_to_plugin_context(session_id: str, key: str, value: Any) -> None:
    """向 PluginContext 写入共享状态（若可用）。"""
    try:
        import sys
        if "plugin_orchestrator.context" not in sys.modules:
            return
        ctx_mod = sys.modules["plugin_orchestrator.context"]
        get_or_create = getattr(ctx_mod, "get_or_create_context", None)
        if get_or_create is None:
            return
        ctx = get_or_create(session_id)
        ctx.shared_set(key, value)
    except Exception:
        pass


MODE_UPGRADE_ORDER = [
    AgentMode.GENERATOR_VERIFIER,
    AgentMode.ORCHESTRATOR_SUBAGENT,
    AgentMode.AGENT_TEAMS,
    AgentMode.MESSAGE_BUS,
    AgentMode.SHARED_STATE,
]

# 模式选择规则表：声明式结构，支持通过 config["mode_rules"] 覆盖阈值与条件
MODE_RULES = [
    {
        "name": "explicit_verification",
        "features": {"has_explicit_verification": True},
        "score_lt": 5,
        "candidates": [AgentMode.GENERATOR_VERIFIER],
    },
    {
        "name": "event_driven",
        "features": {"is_event_driven": True},
        "exclude_features": {"has_roles": True, "needs_collaboration": True},
        "candidates": [AgentMode.MESSAGE_BUS],
    },
    {
        "name": "shared_knowledge",
        "features": {"requires_shared_knowledge": True},
        "score_gt": 4,
        "candidates": [AgentMode.SHARED_STATE, AgentMode.ORCHESTRATOR_SUBAGENT],
    },
    {
        "name": "parallel_fusion",
        "score_gte": 5,
        "any_of": [
            {"task_types": {"analysis", "research", "creative", "complex"}},
            {"features": {"multi_perspective": True}},
            {"features": {"cross_reference": True}},
        ],
        "candidates": [AgentMode.PARALLEL_FUSION, AgentMode.AGENT_TEAMS, AgentMode.SHARED_STATE],
    },
    {
        "name": "has_roles",
        "features": {"has_roles": True},
        "score_gt": 5,
        "candidates": [AgentMode.AGENT_TEAMS, AgentMode.ORCHESTRATOR_SUBAGENT],
    },
    {
        "name": "needs_collaboration",
        "features": {"needs_collaboration": True},
        "score_gt": 5,
        "candidates": [AgentMode.AGENT_TEAMS, AgentMode.ORCHESTRATOR_SUBAGENT],
    },
    {
        "name": "needs_parallelism",
        "features": {"needs_parallelism": True},
        "score_gt": 5,
        "candidates": [AgentMode.AGENT_TEAMS, AgentMode.ORCHESTRATOR_SUBAGENT],
    },
]


class ModeSelectionEngine:
    """模式选择引擎 — Thompson Sampling + 贝叶斯平滑混合策略"""

    def __init__(self, circuit_breakers: Optional[Dict] = None, config: Optional[Dict] = None):
        self.historical_performance: Dict[str, Dict[str, Dict]] = load_performance()
        self.circuit_breakers = circuit_breakers or {}
        self._logger = logging.getLogger("ama.selector")
        self._ts_params: Dict = {}
        self.mode_rules = (config or {}).get("mode_rules", MODE_RULES)
        self._warmup_ts_from_history()

    def _warmup_ts_from_history(self):
        """从历史性能数据预热 Thompson Sampling 先验 Beta"""
        for task_type, modes in self.historical_performance.items():
            for mode_name, stats in modes.items():
                trials = stats.get("trials", 0)
                if trials > 0:
                    successes = stats.get("successes", 0)
                    failures = trials - successes
                    self._ts_params[(task_type, mode_name)] = (
                        1 + successes, 1 + failures
                    )

        # -- 任务类型相似度 fallback：为冷启动提供更好的先验 --
        self._task_type_similarity: Dict[str, List[str]] = {
            "complex": ["research", "analysis", "code_generation"],
            "research": ["analysis", "complex", "fact_checking"],
            "analysis": ["research", "complex", "fact_checking"],
            "code_generation": ["software_dev", "analysis"],
            "software_dev": ["code_generation", "analysis"],
            "fact_checking": ["research", "analysis"],
            "event_driven": ["software_dev", "code_generation"],
            "creative": ["analysis", "code_generation"],
        }

    def _ts_sample(self, task_type: str, mode: AgentMode) -> float:
        """Thompson Sampling: 从 Beta(alpha,beta) 采样模式期望成功率"""
        key = (task_type, mode.value)
        alpha, beta = self._ts_params.get(key, self._get_cold_start_prior(task_type, mode.value))
        # Beta(0,0) 不合法，保证 >=0.01
        return random.betavariate(max(alpha, 0.01), max(beta, 0.01))

    def _get_cold_start_prior(self, task_type: str, mode_value: str) -> tuple:
        """冷启动先验：从相似任务类型借用历史数据。

        如果相似类型也无数据，使用略乐观的 (1.5, 1.0) 先验以鼓励探索。
        """
        similar_types = getattr(self, '_task_type_similarity', {}).get(task_type, [])
        acc_trials, acc_successes = 0, 0
        for sim_type in similar_types:
            sim_key = (sim_type, mode_value)
            if sim_key in self._ts_params:
                a, b = self._ts_params[sim_key]
                acc_trials += a + b - 2
                acc_successes += a - 1

        if acc_trials > 0:
            discount = 0.5  # 相似数据打 5 折
            eff_trials = max(int(acc_trials * discount), 1)
            eff_successes = min(int(acc_successes * discount), eff_trials)
            return (1 + eff_successes, 1 + eff_trials - eff_successes)
        return (1.5, 1.0)  # 完全冷启动：轻微乐观

    def _ts_update(self, task_type: str, mode: AgentMode, success: bool, confidence: float = 1.0):
        """更新 Thompson Sampling 的后验 Beta 分布"""
        key = (task_type, mode.value)
        alpha, beta = self._ts_params.get(key, (1, 1))
        if success:
            self._ts_params[key] = (alpha + confidence, beta)
        else:
            self._ts_params[key] = (alpha, beta + confidence)

    def select_mode(self, assessment: Dict, context: Optional[Dict] = None) -> AgentMode:
        score = assessment["complexity_score"]
        task_type = assessment["task_type"]
        features = assessment["features"]

        candidates = self._apply_rules(score, task_type, features)

        # ── Router→AMA 反向联动：弱模型时降级模式 ──
        try:
            # 优先从 PluginContext 读取（如果 plugin-orchestrator 已安装）
            model_quality = _read_from_plugin_context(session_id=None, key="model_quality", default=3)
            if model_quality is None:
                from plugins.model_router import get_active_model_quality
                model_quality = get_active_model_quality()
            if model_quality is not None and model_quality <= 2:
                heavy_modes = {AgentMode.AGENT_TEAMS, AgentMode.SHARED_STATE, AgentMode.MESSAGE_BUS, AgentMode.PARALLEL_FUSION}
                light_candidates = [m for m in candidates if m not in heavy_modes]
                if light_candidates:
                    self._logger.info("[AMA] Router→AMA联动: 模型质量=%d≤2, 排除重型模式", model_quality)
                    candidates = light_candidates
        except ImportError:
            pass

        # ── 成本感知降级：低复杂度任务慎用高成本模式 ──
        # 借鉴 RouteLLM，但仅在无特征信号时生效（避免覆盖显式路由决策）
        # 阈值提高至 7（原 6），因为我们已将 shared_state/fusion 阈值降至 4-5
        if score < 7 and len(candidates) > 1:
            cheap_modes = {AgentMode.GENERATOR_VERIFIER, AgentMode.ORCHESTRATOR_SUBAGENT}
            has_expensive = any(m not in cheap_modes for m in candidates)
            if has_expensive:
                cheap_candidates = [m for m in candidates if m in cheap_modes]
                # 仅在 TS 数据证明高成本模式无效时才强制降级
                if cheap_candidates:
                    expensive_perf = [
                        m for m in candidates
                        if m not in cheap_modes and self._ts_sample(task_type, m) > 0.7
                    ]
                    if not expensive_perf:
                        self._logger.info(
                            "[AMA] 成本感知: 复杂度=%.1f<7, 无高效高成本模式, 降级 %s",
                            score, [m.value for m in cheap_candidates],
                        )
                        candidates = cheap_candidates

        available = [m for m in candidates if self._is_mode_available(m)]
        if not available:
            available = candidates

        # 单候选直接返回，无需 TS
        if len(available) == 1:
            return available[0]

        selected, self._last_ts_samples = self._ts_select(available, task_type)
        return selected

    def _ts_select(self, candidates: List[AgentMode], task_type: str):
        samples = [(self._ts_sample(task_type, m), m) for m in candidates]
        samples.sort(key=lambda x: x[0], reverse=True)
        selected = samples[0][1]

        if all(s < 0.6 for s, _ in samples):
            selected = self._select_best_performer(candidates, task_type)
        return selected, samples

    def _is_mode_available(self, mode: AgentMode) -> bool:
        cb = self.circuit_breakers.get(mode)
        if cb is None:
            return True
        return cb.is_available()

    def _rule_matches(
        self, rule: Dict, complexity_score: float, task_type: str, features: Dict
    ) -> bool:
        """判断单条声明式规则是否命中。"""
        # 分数边界
        if "score_lt" in rule and not (complexity_score < rule["score_lt"]):
            return False
        if "score_lte" in rule and not (complexity_score <= rule["score_lte"]):
            return False
        if "score_gt" in rule and not (complexity_score > rule["score_gt"]):
            return False
        if "score_gte" in rule and not (complexity_score >= rule["score_gte"]):
            return False

        # 排除特征：任一被显式命中则规则不成立
        for key, expected in rule.get("exclude_features", {}).items():
            if features.get(key) == expected:
                return False

        # any_of 子条件：任一满足即可
        if "any_of" in rule:
            return any(
                self._sub_condition_matches(sub, task_type, features)
                for sub in rule["any_of"]
            )

        # 任务类型约束
        if "task_types" in rule and task_type not in rule["task_types"]:
            return False

        # 必备特征：全部满足
        for key, expected in rule.get("features", {}).items():
            if features.get(key) != expected:
                return False

        return True

    def _sub_condition_matches(self, sub: Dict, task_type: str, features: Dict) -> bool:
        """判断 any_of 中的单条子条件是否命中。"""
        if "task_types" in sub and task_type in sub["task_types"]:
            return True
        if "features" in sub:
            if all(features.get(k) == v for k, v in sub["features"].items()):
                return True
        return False

    def _apply_rules(
        self, complexity_score: float, task_type: str, features: Dict
    ) -> List[AgentMode]:
        """遍历 MODE_RULES 返回第一个匹配规则的候选模式；无匹配则按复杂度分级回退。"""
        for rule in self.mode_rules:
            if self._rule_matches(rule, complexity_score, task_type, features):
                self._logger.debug("[AMA] 规则命中: %s", rule["name"])
                return rule["candidates"]

        # ── 复杂度分级回退：每个区间保留3个候选（含探索模式）──
        if complexity_score <= 3:
            # 简单任务：单一模式即可
            return [AgentMode.GENERATOR_VERIFIER]
        elif complexity_score <= 6:
            # 中等任务：编排为主，生成验证兜底
            return [AgentMode.ORCHESTRATOR_SUBAGENT, AgentMode.GENERATOR_VERIFIER]
        elif complexity_score <= 8:
            # 复杂任务：加入团队模式
            return [AgentMode.AGENT_TEAMS, AgentMode.ORCHESTRATOR_SUBAGENT, AgentMode.GENERATOR_VERIFIER]
        else:
            # 极高复杂度：全模式开放，让 TS 决策
            return [
                AgentMode.SHARED_STATE, AgentMode.PARALLEL_FUSION,
                AgentMode.AGENT_TEAMS, AgentMode.MESSAGE_BUS,
            ]

    def _select_best_performer(
        self, candidates: List[AgentMode], task_type: str
    ) -> AgentMode:
        if task_type in self.historical_performance:
            perf = self.historical_performance[task_type]
            scored = [
                (mode, self._calculate_cost_aware_score(perf.get(mode.value), mode.value))
                for mode in candidates
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]
        # 无历史数据时，给新模式探索加分（冷启动）
        return candidates[0]

    def _calculate_performance_score(self, mode_stats: Optional[Dict]) -> float:
        if not mode_stats or mode_stats.get("trials", 0) == 0:
            # 冷启动：无数据模式给予探索加分，高于旧模式的初始分
            return 0.7

        trials = mode_stats.get("trials", 1)
        successes = mode_stats.get("successes", 0)

        # 贝叶斯平滑：加入先验（3次成功/5次试验），避免小样本极端值
        prior_successes = 3
        prior_trials = 5
        smoothed_success_rate = (successes + prior_successes) / (trials + prior_trials)

        # 效率指标使用 sigmoid 压缩到 [0, 1]，避免无上界膨胀
        avg_tokens = max(mode_stats.get("avg_tokens", 1000), 1)
        avg_time = max(mode_stats.get("avg_time", 10), 0.1)
        # sigmoid: 值越小效率越高，用 1/(1+x/基准) 压缩
        token_efficiency = 1.0 / (1.0 + avg_tokens / 2000.0)
        time_efficiency = 1.0 / (1.0 + avg_time / 30.0)

        # 统计显著性权重：试验次数越多，评分越可信
        confidence = min(trials / 20.0, 1.0)
        # 可信评分与先验评分的加权混合
        raw_score = smoothed_success_rate * 0.6 + token_efficiency * 0.2 + time_efficiency * 0.2
        prior_score = 0.5
        final_score = confidence * raw_score + (1.0 - confidence) * prior_score

        # 冷启动探索衰减：试验次数少的模式获得额外加分
        exploration_bonus = 0.1 * max(0, 1.0 - trials / 10.0)
        return final_score + exploration_bonus

    # ── 模式成本权重（借鉴 RouteLLM 成本感知路由）──
    # 每种模式的相对成本系数（基于 token 消耗和 API 调用次数）
    # generator_verifier 最便宜（2次调用），parallel_fusion 最贵（N×L + 1次调用）
    MODE_COST_MAP: Dict[str, float] = {
        "generator_verifier": 1.0,      # 基准：2次调用（生成+验证）
        "orchestrator_subagent": 2.0,    # 编排+分解+多个子代理
        "agent_teams": 3.0,              # 3个角色串行
        "message_bus": 2.5,              # 事件驱动，多次调用
        "shared_state": 4.0,             # 多轮迭代，每轮多个代理
        "parallel_fusion": 5.0,          # N代理×L层+裁决，最贵
    }

    def _calculate_cost_aware_score(
        self, mode_stats: Optional[Dict], mode_value: str
    ) -> float:
        """成本感知评分：quality / cost 比率（借鉴 RouteLLM）。

        RouteLLM 的核心发现：不只看质量，看 质量/成本 比率。
        便宜模式的质量/成本比可能高于贵模式。
        """
        quality_score = self._calculate_performance_score(mode_stats)
        cost = self.MODE_COST_MAP.get(mode_value, 1.0)

        # 成本惩罚：sigmoid 压缩，避免极端惩罚
        # cost_penalty = 1.0 / (1.0 + cost / 5.0) → cost=1时0.83, cost=5时0.5
        cost_penalty = 1.0 / (1.0 + cost / 5.0)

        # 质量权重 70%，成本权重 30%（可调）
        return quality_score * 0.7 + cost_penalty * 0.3

    def record_performance(
        self,
        task_type: str,
        mode: AgentMode,
        success: bool,
        token_usage: int,
        time_taken: float,
    ) -> None:
        if task_type not in self.historical_performance:
            self.historical_performance[task_type] = {}
        if mode.value not in self.historical_performance[task_type]:
            self.historical_performance[task_type][mode.value] = {
                "trials": 0,
                "successes": 0,
                "avg_tokens": 0,
                "avg_time": 0,
            }

        stats = self.historical_performance[task_type][mode.value]
        total = stats["trials"]

        # 指数衰减：旧数据权重随试验次数增加而降低
        # 衰减因子 0.95 意味着 100 次前的数据权重仅为 0.95^100 ≈ 0.006
        decay = 0.95
        stats["trials"] += 1
        stats["avg_tokens"] = (
            stats["avg_tokens"] * total * decay + token_usage
        ) / (total * decay + 1)
        stats["avg_time"] = (
            stats["avg_time"] * total * decay + time_taken
        ) / (total * decay + 1)
        if success:
            stats["successes"] += 1

        save_performance(task_type, mode.value, stats)

        # 同步更新 Thompson Sampling 后验
        self._ts_update(
            task_type, mode,
            success=success,
            confidence=min(1.0, stats.get("trials", 0) / 10.0),  # 经验越多置信度越高
        )

