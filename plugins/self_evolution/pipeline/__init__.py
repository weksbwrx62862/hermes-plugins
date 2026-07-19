"""
自进化管线 — 模块化优化器框架

提供可插拔的优化器架构：
  - SkillOptimizerBase: 优化器 ABC 接口
  - OptimizeResult: 统一返回结果
  - GEPAOptimizer: GEPA 风格单候选循环优化器
  - DiversifyOptimizer: K 候选 + Auditor 检查优化器
"""

from self_evolution.pipeline.base_optimizer import SkillOptimizerBase, OptimizeResult
from self_evolution.pipeline.gepa_optimizer import GEPAOptimizer
from self_evolution.pipeline.diversify_optimizer import DiversifyOptimizer

__all__ = [
    "SkillOptimizerBase",
    "OptimizeResult",
    "GEPAOptimizer",
    "DiversifyOptimizer",
]
