"""
策略多样化优化器

包装现有 SkillOptimizer，使用其 _diversify() 和 _audit() 方法。
K 候选 + Auditor 检查 + 对比更新。
"""

from __future__ import annotations

import logging
from typing import Optional

from self_evolution.core.evolution_provider import EvalDataset
from self_evolution.pipeline.base_optimizer import SkillOptimizerBase, OptimizeResult

logger = logging.getLogger(__name__)


class DiversifyOptimizer(SkillOptimizerBase):
    """策略多样化优化器：K 候选 + Auditor 检查 + 对比更新。"""

    name = "diversify"

    def __init__(self, client=None, model: str = None, k_candidates: int = 4):
        self._client = client
        self._model = model
        self._k_candidates = k_candidates

    def optimize(
        self,
        skill_text: str,
        dataset: EvalDataset,
        iterations: int = 5,
        temperature: float = 0.7,
    ) -> OptimizeResult:
        from self_evolution.pipeline.optimizer import SkillOptimizer
        optimizer = SkillOptimizer(
            client=self._client,
            model=self._model,
            k_candidates=self._k_candidates,
        )
        evolved_text, best_score, iters, audit_report = optimizer.optimize(
            skill_text, dataset, iterations
        )
        return OptimizeResult(
            evolved_text=evolved_text,
            best_score=best_score,
            iterations_used=iters,
            audit_report=audit_report,
        )

    def score(self, skill_text: str, dataset: EvalDataset) -> float:
        from self_evolution.pipeline.optimizer import SkillOptimizer
        optimizer = SkillOptimizer(client=self._client, model=self._model)
        return optimizer._score_on_split(skill_text, dataset.val)
