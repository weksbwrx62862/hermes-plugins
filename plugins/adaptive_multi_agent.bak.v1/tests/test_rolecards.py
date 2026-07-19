"""rolecards 模块单元测试。"""

import pytest

from adaptive_multi_agent.rolecards import (
    MODE_DEFAULT_ROLES,
    get_mode_role_context,
    inject_role_to_goal,
)
from adaptive_multi_agent.subagent import AgentMode


class TestInjectRoleToGoal:
    """测试 inject_role_to_goal 能正确注入角色上下文。"""

    def test_injects_builder_context(self):
        goal = "实现一个快速排序函数"
        result = inject_role_to_goal(goal, role_id="builder")
        assert "构建者" in result
        assert "Builder" in result
        assert goal in result

    def test_unknown_role_returns_original_goal(self):
        goal = "未知角色测试"
        result = inject_role_to_goal(goal, role_id="not_a_role")
        assert result == goal


class TestGetModeRoleContext:
    """测试 get_mode_role_context 返回非空角色上下文。"""

    @pytest.mark.parametrize("mode", list(AgentMode))
    def test_returns_non_empty_context(self, mode):
        ctx = get_mode_role_context(mode)
        assert ctx
        assert mode.cn in ctx
        # 至少包含一个默认角色的中文名称
        for role_id in MODE_DEFAULT_ROLES[mode]:
            # ROLE_LIBRARY 中的 role_name 为中文
            from adaptive_multi_agent.rolecards import ROLE_LIBRARY
            role_name = ROLE_LIBRARY[role_id].role_name
            assert role_name in ctx


class TestModeDefaultRoles:
    """测试 MODE_DEFAULT_ROLES 包含已知模式。"""

    def test_contains_all_six_modes(self):
        assert set(MODE_DEFAULT_ROLES.keys()) == set(AgentMode)

    def test_each_mode_has_roles(self):
        for mode, roles in MODE_DEFAULT_ROLES.items():
            assert isinstance(roles, list)
            assert len(roles) > 0
