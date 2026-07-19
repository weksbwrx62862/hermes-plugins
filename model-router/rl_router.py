"""
强化学习路由 — Q-learning 与 Contextual Bandit（LinUCB）双引擎

支持通过配置切换算法：
  - Q-learning：基于状态（任务类型+复杂度等级）维护 Q-table
  - LinUCB：基于上下文特征（任务类型 one-hot、复杂度、质量、成本、延迟）在线学习

配置方式（~/.hermes/config.yaml）：
  plugins:
    model-router:
      use_rl_router: true
      rl_algorithm: "linucb"   # "qlearning" | "linucb"，默认 "qlearning"
      linucb_alpha: 1.0        # LinUCB 探索参数
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 配置读取（避免与 __init__.py 循环依赖）
# ─────────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """读取 ~/.hermes/config.yaml，失败时返回空字典。"""
    try:
        import yaml
    except ImportError:
        return {}
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.debug("rl_router 读取 config.yaml 失败: %s", exc)
        return {}


def _get_model_router_config() -> dict[str, Any]:
    return _load_config().get("plugins", {}).get("model-router", {})


def _get_rl_algorithm() -> str:
    """返回当前启用的 RL 算法，默认保留 Q-learning 行为。"""
    algo = _get_model_router_config().get("rl_algorithm", "qlearning")
    if algo in ("qlearning", "linucb"):
        return algo
    return "qlearning"


# ─────────────────────────────────────────────────────────────────────────────
# Q-learning 路由器（原有实现，保持不变）
# ─────────────────────────────────────────────────────────────────────────────

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
        self._q_table: dict[tuple[str, str], float] = defaultdict(float)

        # 历史记录
        self._history: list[dict] = []

        # 加载持久化的 Q-table
        self._load_q_table()

        # 线程锁
        self._lock = threading.Lock()

    def _load_q_table(self) -> None:
        """加载 Q-table"""
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._q_table = defaultdict(float, {
                        tuple(k.split("|")): v
                        for k, v in data.get("q_table", {}).items()
                    })
                    self._exploration_rate = data.get("exploration_rate", self._exploration_rate)
                    logger.info("加载 Q-table: %d 条记录", len(self._q_table))
        except Exception as e:
            logger.warning("加载 Q-table 失败: %s", e)

    def _save_q_table(self) -> None:
        """保存 Q-table"""
        try:
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            data = {
                "q_table": {f"{k[0]}|{k[1]}": v for k, v in self._q_table.items()},
                "exploration_rate": self._exploration_rate,
                "last_save": time.time(),
            }
            with open(self._state_file, "w", encoding="utf-8") as f:
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
        models: list[dict[str, Any]],
        task_type: str = "chat",
        complexity: int = 3,
        strategy: str = "auto",
    ) -> tuple[dict[str, Any], str]:
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
                best_q = float("-inf")

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

    def compute_reward(
        self,
        success: bool,
        token_usage: int = 0,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ) -> float:
        """综合成功率、token 效率、延迟、成本计算奖励，范围 [-1, 1]。"""
        base = 1.0 if success else -1.0
        # token 效率：用量越小奖励越高
        token_factor = min(1.0, 1000.0 / max(token_usage, 1))
        # 延迟惩罚：延迟超过 10s 开始衰减，30s 降到 0.5
        latency_factor = 1.0
        if latency_ms > 0:
            latency_factor = max(0.5, 1.0 - (latency_ms / 30000.0))
        # 成本惩罚：成本超过 $0.01 开始衰减，$0.05 降到 0.5
        cost_factor = 1.0
        if cost_usd > 0:
            cost_factor = max(0.5, 1.0 - (cost_usd / 0.05))

        reward = base * (
            0.4 + 0.3 * token_factor + 0.2 * latency_factor + 0.1 * cost_factor
        )
        return max(-1.0, min(1.0, reward))

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
                "model": model_name,
                "state": state,
                "reward": reward,
                "q_value": new_q,
                "timestamp": time.time(),
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

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        return {
            "q_table_size": len(self._q_table),
            "exploration_rate": self._exploration_rate,
            "history_size": len(self._history),
            "states": len(set(k[0] for k in self._q_table.keys())),
            "actions": len(set(k[1] for k in self._q_table.keys())),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Contextual Bandit 路由器（LinUCB）
# ─────────────────────────────────────────────────────────────────────────────

class ContextualBanditRouter:
    """基于 LinUCB 的上下文 bandit 模型路由器。

    特征向量：任务类型 one-hot + 复杂度 + 模型质量 + 成本 + 延迟。
    对每个臂（模型）维护一个线性模型 theta_a，通过 UCB 得分做探索-利用平衡，
    并针对新模型/新任务类型提供冷启动探索奖励。

    用法：
        router = ContextualBanditRouter(alpha=1.0)
        context = {"task_type": "code", "complexity": 4}
        arm = router.select_arm(context, arms=models)  # 返回模型 dict
        router.update(context, arm["name"], reward=1.0)
    """

    # 与 __init__.py 保持一致的任务类型集合
    TASK_TYPES = [
        "classify", "extract", "simple_qa", "long_doc",
        "code", "math", "complex_reasoning", "agent",
    ]

    def __init__(
        self,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        cold_start_bonus: float = 0.5,
        new_task_bonus: float = 0.3,
        state_file: str = "~/.hermes/model_router_linucb.json",
    ):
        """
        参数:
            alpha: UCB 探索参数，越大越倾向探索
            lambda_reg: A 矩阵初始正则化系数
            cold_start_bonus: 新模型冷启动探索奖励
            new_task_bonus: 新任务类型冷启动探索奖励
            state_file: LinUCB 状态持久化文件路径
        """
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.cold_start_bonus = cold_start_bonus
        self.new_task_bonus = new_task_bonus
        self._state_file = os.path.expanduser(state_file)

        self._feature_dim = len(self.TASK_TYPES) + 4  # one-hot + 4 个数值特征
        self._A: dict[str, np.ndarray] = {}   # arm -> d x d
        self._b: dict[str, np.ndarray] = {}   # arm -> d
        self._arm_counts: dict[str, int] = defaultdict(int)
        self._task_counts: dict[str, int] = defaultdict(int)
        self._arm_features: dict[str, np.ndarray] = {}  # 最近一次见到的臂特征
        self._lock = threading.Lock()

        self._load_state()

    # ── 工具方法 ──

    def _task_type_index(self, task_type: str) -> int:
        """返回任务类型在 one-hot 向量中的下标，未知类型返回 -1。"""
        try:
            return self.TASK_TYPES.index(task_type)
        except ValueError:
            return -1

    def _build_feature(
        self,
        context: dict[str, Any],
        arm: dict[str, Any],
    ) -> np.ndarray:
        """构建上下文+臂的特征向量。

        维度：len(TASK_TYPES) + 4
          - 任务类型 one-hot（8 维）
          - 复杂度归一化值
          - 模型质量归一化值
          - 模型成本归一化值
          - 模型延迟代理归一化值（由 speed 反推）
        """
        task_type = context.get("task_type", "simple_qa")
        complexity = float(context.get("complexity", 3))

        # 任务类型 one-hot
        onehot = np.zeros(len(self.TASK_TYPES), dtype=float)
        idx = self._task_type_index(task_type)
        if idx >= 0:
            onehot[idx] = 1.0

        # 复杂度：假设输入为 1-5，归一化到 [0, 1]
        complexity_norm = max(0.0, min(1.0, (complexity - 1.0) / 4.0))

        # 臂相关特征
        quality = float(arm.get("quality", 3))
        cost = float(arm.get("cost", 3))
        speed = float(arm.get("speed", 3))

        quality_norm = max(0.0, min(1.0, quality / 5.0))
        cost_norm = max(0.0, min(1.0, cost / 10.0))
        # speed 1-5，延迟代理 = (6 - speed) / 5，speed 越快延迟越低
        latency_norm = max(0.0, min(1.0, (6.0 - speed) / 5.0))

        x = np.concatenate([
            onehot,
            np.array([complexity_norm, quality_norm, cost_norm, latency_norm], dtype=float),
        ])
        return x

    def _get_A(self, arm_name: str) -> np.ndarray:
        """获取臂的 A 矩阵，不存在时初始化为 lambda * I。"""
        if arm_name not in self._A:
            self._A[arm_name] = self.lambda_reg * np.eye(self._feature_dim)
            self._b[arm_name] = np.zeros(self._feature_dim)
        return self._A[arm_name]

    def _get_b(self, arm_name: str) -> np.ndarray:
        if arm_name not in self._b:
            self._b[arm_name] = np.zeros(self._feature_dim)
        return self._b[arm_name]

    def _ucb_score(self, x: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
        """计算 LinUCB 得分：theta^T x + alpha * sqrt(x^T A^{-1} x)。"""
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
        theta = A_inv @ b
        pred = float(theta @ x)
        uncertainty = float(np.sqrt(x @ A_inv @ x))
        return pred + self.alpha * uncertainty

    # ── 核心接口 ──

    def select_arm(
        self,
        context: dict[str, Any],
        arms: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """根据上下文选择推荐模型（臂）。

        参数:
            context: 包含 task_type、complexity 的字典；也可通过 "arms" / "models"
                     直接传入候选模型列表。
            arms: 候选模型列表，若提供则优先使用；否则读取 context["arms"] 或
                  context["models"]。

        返回:
            选中的模型 dict
        """
        if arms is None:
            arms = context.get("arms") or context.get("models") or []
        if not arms:
            raise ValueError("select_arm 需要提供候选模型列表")

        task_type = context.get("task_type", "simple_qa")

        with self._lock:
            best_arm = None
            best_score = float("-inf")

            for arm in arms:
                arm_name = arm.get("name", "")
                if not arm_name:
                    continue

                x = self._build_feature(context, arm)
                self._arm_features[arm_name] = x

                A = self._get_A(arm_name)
                b = self._get_b(arm_name)
                score = self._ucb_score(x, A, b)

                # 冷启动保护：新臂或新任务类型增加探索奖励
                if self._arm_counts[arm_name] == 0:
                    score += self.cold_start_bonus
                if self._task_counts[task_type] == 0:
                    score += self.new_task_bonus

                if score > best_score:
                    best_score = score
                    best_arm = arm

            return best_arm or arms[0]

    def update(
        self,
        context: dict[str, Any],
        arm: str,
        reward: float,
    ) -> None:
        """在线更新指定臂的 LinUCB 参数。

        参数:
            context: 包含 task_type、complexity 的字典
            arm: 选中的模型名称
            reward: 观测到的奖励（建议范围 [-1, 1]）
        """
        task_type = context.get("task_type", "simple_qa")

        with self._lock:
            # 优先使用 select_arm 时缓存的臂特征；否则用零填充模型特征
            x = self._arm_features.get(arm)
            if x is None:
                x = self._build_feature(context, {"name": arm, "quality": 3, "cost": 3, "speed": 3})

            A = self._get_A(arm)
            b_vec = self._get_b(arm)

            A_new = A + np.outer(x, x)
            b_new = b_vec + reward * x

            self._A[arm] = A_new
            self._b[arm] = b_new
            self._arm_counts[arm] += 1
            self._task_counts[task_type] += 1

            logger.debug(
                "LinUCB 更新: arm=%s, reward=%.2f, pulls=%d, task=%s",
                arm, reward, self._arm_counts[arm], task_type
            )

            # 每 10 次更新持久化一次
            total_pulls = sum(self._arm_counts.values())
            if total_pulls % 10 == 0:
                self._save_state()

    def record_reward(
        self,
        model_name: str,
        reward: float,
        task_type: str = "chat",
        complexity: int = 3,
    ) -> None:
        """兼容 Q-learning 风格的奖励记录接口。"""
        context = {"task_type": task_type, "complexity": complexity}
        self.update(context, model_name, reward)

    def compute_reward(
        self,
        success: bool,
        token_usage: int = 0,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ) -> float:
        """计算综合奖励（与 Q-learning 保持一致语义）。"""
        base = 1.0 if success else -1.0
        token_factor = min(1.0, 1000.0 / max(token_usage, 1))
        latency_factor = 1.0
        if latency_ms > 0:
            latency_factor = max(0.5, 1.0 - (latency_ms / 30000.0))
        cost_factor = 1.0
        if cost_usd > 0:
            cost_factor = max(0.5, 1.0 - (cost_usd / 0.05))
        reward = base * (0.4 + 0.3 * token_factor + 0.2 * latency_factor + 0.1 * cost_factor)
        return max(-1.0, min(1.0, reward))

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        return {
            "arms": len(self._A),
            "feature_dim": self._feature_dim,
            "alpha": self.alpha,
            "total_pulls": sum(self._arm_counts.values()),
            "arm_counts": dict(self._arm_counts),
            "task_counts": dict(self._task_counts),
        }

    # ── 持久化 ──

    def _load_state(self) -> None:
        """从文件加载 LinUCB 状态。"""
        try:
            if not os.path.exists(self._state_file):
                return
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            loaded_dim = data.get("feature_dim", self._feature_dim)
            if loaded_dim != self._feature_dim:
                logger.warning(
                    "LinUCB 状态维度不匹配: 文件=%d, 当前=%d，忽略历史状态",
                    loaded_dim, self._feature_dim,
                )
                return

            for arm, A_list in data.get("A", {}).items():
                self._A[arm] = np.array(A_list, dtype=float)
                self._b[arm] = np.array(data["b"].get(arm, []), dtype=float)
            self._arm_counts = defaultdict(int, data.get("arm_counts", {}))
            self._task_counts = defaultdict(int, data.get("task_counts", {}))
            self.alpha = data.get("alpha", self.alpha)
            logger.info("加载 LinUCB 状态: %d 个臂", len(self._A))
        except Exception as e:
            logger.warning("加载 LinUCB 状态失败: %s", e)

    def _save_state(self) -> None:
        """保存 LinUCB 状态到文件。"""
        try:
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            data = {
                "A": {arm: A.tolist() for arm, A in self._A.items()},
                "b": {arm: b.tolist() for arm, b in self._b.items()},
                "arm_counts": dict(self._arm_counts),
                "task_counts": dict(self._task_counts),
                "feature_dim": self._feature_dim,
                "alpha": self.alpha,
                "last_save": time.time(),
            }
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug("保存 LinUCB 状态: %d 个臂", len(self._A))
        except Exception as e:
            logger.warning("保存 LinUCB 状态失败: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# 全局实例与兼容接口
# ─────────────────────────────────────────────────────────────────────────────

_rl_router: Optional[QLearningRouter] = None
_linucb_router: Optional[ContextualBanditRouter] = None
_rl_router_lock = threading.Lock()


def get_rl_router() -> QLearningRouter:
    """获取全局 Q-learning 路由器"""
    global _rl_router
    if _rl_router is None:
        with _rl_router_lock:
            if _rl_router is None:
                _rl_router = QLearningRouter()
    return _rl_router


def get_linucb_router() -> ContextualBanditRouter:
    """获取全局 LinUCB 路由器"""
    global _linucb_router
    if _linucb_router is None:
        with _rl_router_lock:
            if _linucb_router is None:
                cfg = _get_model_router_config()
                alpha = float(cfg.get("linucb_alpha", 1.0))
                _linucb_router = ContextualBanditRouter(alpha=alpha)
    return _linucb_router


def _get_active_router() -> QLearningRouter | ContextualBanditRouter:
    """根据配置返回当前使用的 RL 路由器。"""
    if _get_rl_algorithm() == "linucb":
        return get_linucb_router()
    return get_rl_router()


def rl_select_model(
    models: list[dict[str, Any]],
    task_type: str = "chat",
    complexity: int = 3,
    strategy: str = "auto",
) -> tuple[dict[str, Any], str]:
    """根据配置使用 Q-learning 或 LinUCB 选择模型。"""
    router = _get_active_router()
    if isinstance(router, ContextualBanditRouter):
        context = {"task_type": task_type, "complexity": complexity}
        selected = router.select_arm(context, arms=models)
        reason = f"LinUCB (alpha={router.alpha})"
        return selected, reason
    return router.select_model(models, task_type, complexity, strategy)


def rl_compute_reward(
    success: bool,
    token_usage: int = 0,
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
) -> float:
    """计算综合奖励（综合考虑成功、token 效率、延迟、成本）。"""
    router = _get_active_router()
    return router.compute_reward(success, token_usage, latency_ms, cost_usd)


def rl_record_reward(
    model_name: str,
    reward: float,
    task_type: str = "chat",
    complexity: int = 3,
) -> None:
    """记录奖励（根据当前算法路由到 Q-learning 或 LinUCB）。"""
    router = _get_active_router()
    router.record_reward(model_name, reward, task_type, complexity)
