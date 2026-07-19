"""
model-router 强化学习路由单元测试

覆盖：
  - ContextualBanditRouter（LinUCB）核心接口
  - 50 次反馈后策略收敛到稳定模型
  - 冷启动保护（新臂/新任务类型）
  - 特征向量维度与取值范围
  - 兼容接口（rl_select_model / rl_compute_reward / rl_record_reward）
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# 目录名含连字符，无法直接 import，使用 importlib 加载 rl_router.py
_rl_router_path = Path(__file__).resolve().parent.parent / "rl_router.py"
_spec = importlib.util.spec_from_file_location("rl_router", str(_rl_router_path))
rl_router = importlib.util.module_from_spec(_spec)
sys.modules["rl_router"] = rl_router
_spec.loader.exec_module(rl_router)

ContextualBanditRouter = rl_router.ContextualBanditRouter
QLearningRouter = rl_router.QLearningRouter
rl_select_model = rl_router.rl_select_model
rl_compute_reward = rl_router.rl_compute_reward
rl_record_reward = rl_router.rl_record_reward


@pytest.fixture
def premium_models():
    """构造一组特征明显向 premium 倾斜的候选模型。"""
    return [
        {"name": "slow", "cost": 5, "speed": 2, "quality": 2},
        {"name": "mid", "cost": 3, "speed": 3, "quality": 3},
        {"name": "premium", "cost": 0, "speed": 5, "quality": 5},
    ]


class TestContextualBanditRouter:
    """ContextualBanditRouter（LinUCB）测试。"""

    def test_feature_dimension_and_range(self, premium_models):
        """验证特征向量维度与归一化取值范围。"""
        router = ContextualBanditRouter(alpha=1.0, state_file="/tmp/linucb_test.json")
        context = {"task_type": "code", "complexity": 4}
        arm = premium_models[0]

        x = router._build_feature(context, arm)

        # one-hot（8 维） + 4 个数值特征 = 12 维
        assert len(x) == 12
        # one-hot 分量只有对应任务类型为 1，其余为 0
        assert x[router.TASK_TYPES.index("code")] == 1.0
        assert sum(x[: len(router.TASK_TYPES)]) == 1.0
        # 数值特征均归一化到 [0, 1]
        for val in x[len(router.TASK_TYPES) :]:
            assert 0.0 <= val <= 1.0

    def test_cold_start_new_arm_exploration(self, premium_models):
        """冷启动保护：新模型（未被拉动过）首次会有探索奖励。"""
        router = ContextualBanditRouter(
            alpha=0.0, cold_start_bonus=10.0, state_file="/tmp/linucb_test.json"
        )
        context = {"task_type": "code", "complexity": 4}

        # alpha=0 时 UCB 得分仅由预测值和冷启动奖励构成；
        # 所有臂预测值相同，新臂因 cold_start_bonus 最高应被选中
        selected = router.select_arm(context, arms=premium_models)
        assert selected["name"] in {"slow", "mid", "premium"}
        assert router._arm_counts[selected["name"]] == 0  # select 不计入 count

    def test_cold_start_new_task_exploration(self, premium_models):
        """冷启动保护：新任务类型会给所有候选臂增加探索奖励。"""
        router = ContextualBanditRouter(
            alpha=0.0,
            cold_start_bonus=0.0,
            new_task_bonus=10.0,
            state_file="/tmp/linucb_test.json",
        )
        # 使用一个此前未见过任务类型（非 "simple_qa"），确保新任务奖励生效
        context = {"task_type": "agent", "complexity": 3}

        selected = router.select_arm(context, arms=premium_models)
        assert selected["name"] in {"slow", "mid", "premium"}
        assert router._task_counts["agent"] == 0  # select 不计入 count

    def test_update_increments_counts(self, premium_models):
        """update 后臂计数与任务计数递增。"""
        router = ContextualBanditRouter(state_file="/tmp/linucb_test.json")
        context = {"task_type": "code", "complexity": 4}
        arm = router.select_arm(context, arms=premium_models)

        router.update(context, arm["name"], reward=1.0)

        assert router._arm_counts[arm["name"]] == 1
        assert router._task_counts["code"] == 1

    def test_converges_to_best_arm_after_50_rounds(self, premium_models):
        """模拟 50 次反馈，验证 LinUCB 收敛到最优模型 premium。

        收敛定义：最后连续 10 次选择同一模型。
        """
        router = ContextualBanditRouter(
            alpha=0.2,
            cold_start_bonus=0.1,
            new_task_bonus=0.1,
            state_file="/tmp/linucb_converge_test.json",
        )
        context = {"task_type": "code", "complexity": 4}
        target = "premium"

        choices = []
        for _ in range(50):
            arm = router.select_arm(context, arms=premium_models)
            name = arm["name"]
            # premium 给高奖励，其他臂给负奖励，拉大差距
            reward = 1.0 if name == target else -0.5
            router.update(context, name, reward)
            choices.append(name)

        # 最后 10 次选择一致视为收敛
        last_10 = choices[-10:]
        assert len(set(last_10)) == 1, f"最后 10 次选择未收敛: {last_10}"
        assert last_10[0] == target, f"收敛到的模型不是 {target}: {last_10[0]}"

    def test_compute_reward_range(self):
        """奖励计算结果在 [-1, 1] 范围内。"""
        router = ContextualBanditRouter(state_file="/tmp/linucb_test.json")
        reward = router.compute_reward(
            success=True, token_usage=500, latency_ms=5000, cost_usd=0.01
        )
        assert -1.0 <= reward <= 1.0

        reward_fail = router.compute_reward(success=False)
        assert -1.0 <= reward_fail <= 0.0


class TestCompatibilityInterfaces:
    """模块级兼容接口测试（默认配置下走 Q-learning）。"""

    def test_rl_select_model_returns_model(self):
        models = [
            {"name": "a", "cost": 1, "speed": 3, "quality": 3},
            {"name": "b", "cost": 2, "speed": 4, "quality": 4},
        ]
        selected, reason = rl_select_model(models, task_type="simple_qa", complexity=2)
        assert selected is not None
        assert selected["name"] in {"a", "b"}
        assert isinstance(reason, str)

    def test_rl_compute_reward_success_positive(self):
        reward = rl_compute_reward(success=True, token_usage=100, latency_ms=1000)
        assert reward > 0

    def test_rl_compute_reward_failure_negative(self):
        reward = rl_compute_reward(success=False)
        assert reward < 0

    def test_rl_record_reward_does_not_raise(self):
        # 记录奖励不应抛异常；默认配置走 Q-learning
        rl_record_reward("test-model", reward=0.5, task_type="code", complexity=3)


class TestQLearningRouter:
    """Q-learning 路由器回归测试。"""

    def test_qlearning_select_and_update(self):
        router = QLearningRouter(state_file="/tmp/qlearning_test.json")
        models = [
            {"name": "cheap", "cost": 1, "speed": 5, "quality": 2},
            {"name": "smart", "cost": 5, "speed": 2, "quality": 5},
        ]
        selected, reason = router.select_model(models, task_type="code", complexity=4)
        assert selected["name"] in {"cheap", "smart"}

        router.record_reward(selected["name"], reward=1.0, task_type="code", complexity=4)
        stats = router.get_stats()
        assert stats["history_size"] == 1
