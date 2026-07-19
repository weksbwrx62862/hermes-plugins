"""pre_llm_call_hook 单元测试

验证置信度过滤、core/pool 字段注入与域推荐列表，所有外部依赖均使用 mock。
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INIT_PATH = os.path.join(ROOT, "__init__.py")


def _load_init():
    """动态加载 skill-router 入口模块"""
    spec = importlib.util.spec_from_file_location("skill_router_init", _INIT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["skill_router_init"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_init()


class _MockPluginContext:
    """模拟 plugin_context，提供 shared_get / shared_set"""

    def __init__(self):
        self._state: dict = {}

    def shared_get(self, key: str, default=None):
        return self._state.get(key, default)

    def shared_set(self, key: str, value) -> None:
        self._state[key] = value


def _patch_dependencies(module, core_skills=None):
    """用 mock 替换 hook 依赖的外部函数"""
    module._load_config = lambda: module._DEFAULT_CONFIG.copy()
    module._domain_config_mod._CONFIG_LOADER = module._load_config
    module._load_skill_index = lambda: {
        "hermes-agent": {"category": "general", "description": "通用助手"},
        "skill-creator": {"category": "dev", "description": "创建技能"},
        "web-search-china": {"category": "search", "description": "中文搜索"},
        "finance-assistant": {"category": "finance", "description": "金融助手"},
    }
    module._get_core_skills = lambda: set(core_skills) if core_skills is not None else {"hermes-agent", "skill-creator"}


def _make_result(name: str, score: float, confidence: str, tier: str = "pool") -> dict:
    """构造 search_skills 返回结果项"""
    return {
        "name": name,
        "category": "test",
        "description": f"技能 {name}",
        "score": score,
        "tier": tier,
        "confidence": confidence,
    }


def test_low_confidence_returns_empty(module):
    """仅 low 置信度 pool 技能且无 core 技能时应返回空"""
    _patch_dependencies(module, core_skills=[])
    module.search_skills = lambda query, top_k=None: [
        _make_result("web-search-china", 0.2, "low"),
    ]

    result = module._pre_llm_call_hook(user_message="我想查点资料")
    assert result == {}


def test_medium_confidence_includes_pool(module):
    """medium 置信度时应注入 pool 技能"""
    _patch_dependencies(module)
    module.search_skills = lambda query, top_k=None: [
        _make_result("web-search-china", 0.35, "medium"),
    ]

    result = module._pre_llm_call_hook(user_message="我想查点资料")
    assert "context" in result
    assert "web-search-china" in result["context"]
    assert result["context_merge"]["skill_count"] == 1


def test_high_confidence_includes_pool(module):
    """high 置信度时应正常注入 pool 技能"""
    _patch_dependencies(module)
    module.search_skills = lambda query, top_k=None: [
        _make_result("finance-assistant", 0.75, "high"),
    ]

    result = module._pre_llm_call_hook(user_message="查看股票行情")
    assert "finance-assistant" in result["context"]
    assert result["context_merge"]["skill_count"] == 1


def test_core_skills_are_injected(module):
    """core 技能应始终出现在上下文 Core 行中"""
    _patch_dependencies(module)
    module.search_skills = lambda query, top_k=None: [
        _make_result("finance-assistant", 0.75, "high"),
    ]

    result = module._pre_llm_call_hook(user_message="查看股票行情")
    assert "Core:" in result["context"]
    assert "hermes-agent" in result["context"]
    assert "skill-creator" in result["context"]


def test_core_results_not_repeated_in_pool(module):
    """search_skills 返回的 core 结果不应重复出现在 pool 行中"""
    _patch_dependencies(module)
    module.search_skills = lambda query, top_k=None: [
        _make_result("hermes-agent", 0.9, "high", tier="core"),
        _make_result("finance-assistant", 0.7, "high", tier="pool"),
    ]

    result = module._pre_llm_call_hook(user_message="查看股票行情")
    lines = result["context"].splitlines()
    core_line = [line for line in lines if line.startswith("Core:")][0]
    pool_lines = [line for line in lines if line.startswith("  -") and "Core:" not in line]
    assert "hermes-agent" in core_line
    assert not any("hermes-agent" in line for line in pool_lines)
    assert any("finance-assistant" in line for line in pool_lines)


def test_recommended_domains_in_context_merge(module):
    """命中域的查询应在 context_merge 中返回推荐域列表"""
    _patch_dependencies(module)
    module.search_skills = lambda query, top_k=None: [
        _make_result("finance-assistant", 0.75, "high"),
    ]

    result = module._pre_llm_call_hook(user_message="查看股票行情")
    assert "finance" in result["context_merge"]["skill_domains"]


def test_slash_command_returns_empty(module):
    """斜杠命令应直接跳过技能注入"""
    assert module._pre_llm_call_hook(user_message="/help") == {}


def test_short_message_returns_empty(module):
    """过短用户消息应跳过"""
    assert module._pre_llm_call_hook(user_message="hi") == {}
