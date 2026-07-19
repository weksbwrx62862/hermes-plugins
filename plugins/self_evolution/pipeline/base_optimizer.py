"""
模块化优化器基类

所有优化器必须实现 SkillOptimizerBase 接口，统一返回 OptimizeResult。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from self_evolution.core.evolution_provider import EvalDataset, FitnessScore


@dataclass
class OptimizeResult:
    """优化器统一返回结果。"""
    evolved_text: str
    best_score: float
    iterations_used: int
    audit_report: dict = field(default_factory=dict)


class SkillOptimizerBase(ABC):
    """技能优化器基类。所有优化器必须实现此接口。"""

    name: str = "base"

    @abstractmethod
    def optimize(
        self,
        skill_text: str,
        dataset: EvalDataset,
        iterations: int = 5,
        temperature: float = 0.7,
    ) -> OptimizeResult:
        """执行优化循环，返回优化结果。"""
        ...

    @abstractmethod
    def score(
        self,
        skill_text: str,
        dataset: EvalDataset,
    ) -> float:
        """评估技能在数据集上的分数。"""
        ...

    def audit(self, evolved_text: str, audit_report: dict) -> dict:
        """对优化结果执行审计检查（可选覆盖）。"""
        return audit_report
