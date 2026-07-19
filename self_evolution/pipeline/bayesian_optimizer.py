"""
贝叶斯优化器 — 使用高斯过程优化技能参数

优势：
  - 比网格搜索效率高 3-5x
  - 自动探索-利用平衡
  - 适合连续参数空间
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

from bayes_opt import BayesianOptimization

logger = logging.getLogger(__name__)


class BayesianSkillOptimizer:
    """贝叶斯技能优化器
    
    使用贝叶斯优化搜索最优的技能参数组合（如 temperature、max_tokens 等）。
    
    典型用法：
        optimizer = BayesianSkillOptimizer(eval_fn)
        best_params, best_score = optimizer.optimize(
            pbounds={'temperature': (0.1, 1.0), 'max_tokens': (100, 2000)},
            init_points=5,
            n_iter=25,
        )
    """
    
    def __init__(
        self,
        eval_fn: Callable[..., float],
        random_state: int = 42,
        verbose: int = 2,
    ):
        """
        参数:
            eval_fn: 评估函数，接受关键字参数，返回分数（越高越好）
            random_state: 随机种子
            verbose: 日志级别 (0=静默, 1=进度, 2=详细)
        """
        self._eval_fn = eval_fn
        self._random_state = random_state
        self._verbose = verbose
    
    def optimize(
        self,
        pbounds: Dict[str, Tuple[float, float]],
        init_points: int = 5,
        n_iter: int = 25,
        acq: str = 'ucb',
        kappa: float = 2.576,
        xi: float = 0.0,
    ) -> Tuple[Dict[str, float], float]:
        """执行贝叶斯优化
        
        参数:
            pbounds: 参数边界字典，如 {'temperature': (0.1, 1.0)}
            init_points: 初始随机探索点数
            n_iter: 贝叶斯优化迭代次数
            acq: 采集函数 ('ucb', 'ei', 'poi')
            kappa: UCB 探索系数（越大越探索）
            xi: EI/POI 探索系数
        
        返回:
            (最优参数字典, 最优分数)
        """
        logger.info("开始贝叶斯优化 | init_points=%d | n_iter=%d | acq=%s", 
                    init_points, n_iter, acq)
        
        # 创建贝叶斯优化器
        optimizer = BayesianOptimization(
            f=self._eval_fn,
            pbounds=pbounds,
            random_state=self._random_state,
            verbose=self._verbose,
        )
        
        # 执行优化
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )
        
        # 获取最优结果
        best = optimizer.max
        best_params = best['params']
        best_score = best['target']
        
        logger.info("贝叶斯优化完成 | best_score=%.4f | best_params=%s", 
                    best_score, best_params)
        
        return best_params, best_score
    
    def optimize_with_early_stopping(
        self,
        pbounds: Dict[str, Tuple[float, float]],
        init_points: int = 5,
        n_iter: int = 50,
        patience: int = 10,
        min_improvement: float = 0.001,
    ) -> Tuple[Dict[str, float], float, int]:
        """带早停的贝叶斯优化
        
        参数:
            pbounds: 参数边界
            init_points: 初始探索点
            n_iter: 最大迭代次数
            patience: 早停耐心值（连续多少轮无改进则停止）
            min_improvement: 最小改进阈值
        
        返回:
            (最优参数, 最优分数, 实际迭代次数)
        """
        logger.info("开始带早停的贝叶斯优化 | max_iter=%d | patience=%d", n_iter, patience)
        
        optimizer = BayesianOptimization(
            f=self._eval_fn,
            pbounds=pbounds,
            random_state=self._random_state,
            verbose=self._verbose,
        )
        
        # 初始随机探索
        optimizer.maximize(init_points=init_points, n_iter=0)
        
        best_score = optimizer.max['target']
        no_improvement_count = 0
        iterations_used = init_points
        
        for i in range(n_iter):
            # 单步优化
            optimizer.maximize(init_points=0, n_iter=1)
            iterations_used += 1
            
            current_score = optimizer.max['target']
            
            # 检查是否有改进
            if current_score - best_score > min_improvement:
                best_score = current_score
                no_improvement_count = 0
                logger.debug("第 %d 轮: 分数提升至 %.4f", i + 1, current_score)
            else:
                no_improvement_count += 1
                logger.debug("第 %d 轮: 无改进 (%d/%d)", 
                           i + 1, no_improvement_count, patience)
            
            # 早停检查
            if no_improvement_count >= patience:
                logger.info("早停触发: 连续 %d 轮无改进", patience)
                break
        
        best = optimizer.max
        logger.info("贝叶斯优化完成 | best_score=%.4f | iterations=%d", 
                    best['target'], iterations_used)
        
        return best['params'], best['target'], iterations_used


def create_skill_eval_fn(
    base_skill_text: str,
    param_template: Callable[[str, Dict], str],
    evaluate_fn: Callable[[str, str], float],
    test_cases: list,
) -> Callable[..., float]:
    """创建技能评估函数（用于贝叶斯优化）
    
    参数:
        base_skill_text: 基础技能文本
        param_template: 参数化模板函数 (skill_text, params) -> modified_skill_text
        evaluate_fn: 评估函数 (skill_text, test_case) -> score
        test_cases: 测试用例列表
    
    返回:
        评估函数 (**params) -> avg_score
    """
    def eval_fn(**params):
        # 应用参数
        modified_skill = param_template(base_skill_text, params)
        
        # 评估所有测试用例
        scores = []
        for test_case in test_cases:
            try:
                score = evaluate_fn(modified_skill, test_case)
                scores.append(score)
            except Exception as e:
                logger.warning("评估失败: %s", e)
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    return eval_fn


# 示例：参数化技能模板
def skill_param_template(skill_text: str, params: Dict) -> str:
    """技能参数化模板示例
    
    将参数注入到技能文本中（如 temperature、max_tokens 等）。
    这是一个示例实现，实际应根据技能格式调整。
    """
    # 简单示例：在技能文本末尾添加参数说明
    param_str = " ".join(f"{k}={v}" for k, v in params.items())
    return f"{skill_text}\n\n# 优化参数: {param_str}"
