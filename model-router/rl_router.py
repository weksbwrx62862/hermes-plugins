"""
强化学习路由 — 使用 Q-learning 优化模型选择

优势：
  - 根据历史奖励自动学习最优策略
  - 平衡探索与利用
  - 适应用户偏好
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QLearningRouter:
    """Q-learning 模型路由器
    
    使用 Q-learning 算法学习最优的模型选择策略。
    
    典型用法：
        router = QLearningRouter()
        model = router.select_model(models, task_type="chat", complexity=3)
        router.record_reward(model_name, reward=1.0)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01,
        state_file: str = "~/.hermes/model_router_qtable.json",
    ):
        """
        参数:
            learning_rate: 学习率 (0-1)
            discount_factor: 折扣因子 (0-1)
            exploration_rate: 探索率 (0-1)
            exploration_decay: 探索率衰减
            min_exploration_rate: 最小探索率
            state_file: Q-table 持久化文件路径
        """
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._exploration_rate = exploration_rate
        self._exploration_decay = exploration_decay
        self._min_exploration_rate = min_exploration_rate
        self._state_file = os.path.expanduser(state_file)
        
        # Q-table: (state, action) -> q_value
        self._q_table: Dict[Tuple[str, str], float] = defaultdict(float)
        
        # 历史记录
        self._history: List[Dict] = []
        
        # 加载持久化的 Q-table
        self._load_q_table()
        
        # 线程锁
        self._lock = threading.Lock()
    
    def _load_q_table(self) -> None:
        """加载 Q-table"""
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._q_table = defaultdict(float, {
                        tuple(k.split('|')): v 
                        for k, v in data.get('q_table', {}).items()
                    })
                    self._exploration_rate = data.get('exploration_rate', self._exploration_rate)
                    logger.info("加载 Q-table: %d 条记录", len(self._q_table))
        except Exception as e:
            logger.warning("加载 Q-table 失败: %s", e)
    
    def _save_q_table(self) -> None:
        """保存 Q-table"""
        try:
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            data = {
                'q_table': {f"{k[0]}|{k[1]}": v for k, v in self._q_table.items()},
                'exploration_rate': self._exploration_rate,
                'last_save': time.time(),
            }
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.debug("保存 Q-table: %d 条记录", len(self._q_table))
        except Exception as e:
            logger.warning("保存 Q-table 失败: %s", e)
    
    def _get_state(self, task_type: str, complexity: int) -> str:
        """获取状态表示
        
        参数:
            task_type: 任务类型
            complexity: 复杂度 (1-5)
        
        返回:
            状态字符串
        """
        # 离散化复杂度
        if complexity <= 2:
            level = "low"
        elif complexity <= 3:
            level = "medium"
        else:
            level = "high"
        
        return f"{task_type}_{level}"
    
    def select_model(
        self,
        models: List[Dict[str, Any]],
        task_type: str = "chat",
        complexity: int = 3,
        strategy: str = "auto",
    ) -> Tuple[Dict[str, Any], str]:
        """选择模型
        
        参数:
            models: 可用模型列表
            task_type: 任务类型
            complexity: 复杂度 (1-5)
            strategy: 策略 ("auto", "cheapest", "fastest", "smartest")
        
        返回:
            (选中的模型, 选择原因)
        """
        if not models:
            return None, "无可用模型"
        
        with self._lock:
            state = self._get_state(task_type, complexity)
            
            # ε-greedy 策略
            if np.random.random() < self._exploration_rate:
                # 探索：随机选择
                idx = np.random.randint(0, len(models))
                selected = models[idx]
                reason = f"探索 (ε={self._exploration_rate:.3f})"
            else:
                # 利用：选择 Q 值最高的模型
                best_model = None
                best_q = float('-inf')
                
                for model in models:
                    action = model.get("name", "")
                    q_value = self._q_table.get((state, action), 0.0)
                    
                    # 结合策略权重
                    if strategy == "cheapest":
                        score = q_value - model.get("cost", 3) * 0.5
                    elif strategy == "fastest":
                        score = q_value + model.get("speed", 3) * 0.5
                    elif strategy == "smartest":
                        score = q_value + model.get("quality", 3) * 0.5
                    else:  # auto
                        score = q_value
                    
                    if score > best_q:
                        best_q = score
                        best_model = model
                
                selected = best_model or models[0]
                reason = f"利用 (Q={best_q:.3f})"
            
            return selected, reason
    
    def record_reward(
        self,
        model_name: str,
        reward: float,
        task_type: str = "chat",
        complexity: int = 3,
    ) -> None:
        """记录奖励，更新 Q-table
        
        参数:
            model_name: 模型名称
            reward: 奖励值 (正=成功, 负=失败)
            task_type: 任务类型
            complexity: 复杂度 (1-5)
        """
        with self._lock:
            state = self._get_state(task_type, complexity)
            action = model_name
            
            # Q-learning 更新规则
            current_q = self._q_table.get((state, action), 0.0)
            
            # 简化：假设下一状态的 Q 值为当前最大 Q 值
            recent_models = [m.get("name", "") for m in self._history[-10:] if m.get("model")]
            if recent_models:
                max_next_q = max(self._q_table.get((state, a), 0.0) for a in set(recent_models))
            else:
                max_next_q = 0.0
            
            # 更新 Q 值
            new_q = current_q + self._learning_rate * (
                reward + self._discount_factor * max_next_q - current_q
            )
            self._q_table[(state, action)] = new_q
            
            # 记录历史
            self._history.append({
                'model': model_name,
                'state': state,
                'reward': reward,
                'q_value': new_q,
                'timestamp': time.time(),
            })
            
            # 衰减探索率
            self._exploration_rate = max(
                self._min_exploration_rate,
                self._exploration_rate * self._exploration_decay
            )
            
            logger.info(
                "记录奖励: model=%s, reward=%.2f, q_value=%.3f, exploration=%.3f",
                model_name, reward, new_q, self._exploration_rate
            )
            
            # 定期保存
            if len(self._history) % 10 == 0:
                self._save_q_table()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'q_table_size': len(self._q_table),
            'exploration_rate': self._exploration_rate,
            'history_size': len(self._history),
            'states': len(set(k[0] for k in self._q_table.keys())),
            'actions': len(set(k[1] for k in self._q_table.keys())),
        }


# 全局实例
_rl_router: Optional[QLearningRouter] = None
_rl_router_lock = threading.Lock()


def get_rl_router() -> QLearningRouter:
    """获取全局 Q-learning 路由器"""
    global _rl_router
    if _rl_router is None:
        with _rl_router_lock:
            if _rl_router is None:
                _rl_router = QLearningRouter()
    return _rl_router


def rl_select_model(
    models: List[Dict[str, Any]],
    task_type: str = "chat",
    complexity: int = 3,
    strategy: str = "auto",
) -> Tuple[Dict[str, Any], str]:
    """使用 Q-learning 选择模型"""
    router = get_rl_router()
    return router.select_model(models, task_type, complexity, strategy)


def rl_record_reward(
    model_name: str,
    reward: float,
    task_type: str = "chat",
    complexity: int = 3,
) -> None:
    """记录奖励"""
    router = get_rl_router()
    router.record_reward(model_name, reward, task_type, complexity)
