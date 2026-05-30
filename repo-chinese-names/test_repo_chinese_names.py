"""repo-chinese-names 插件测试 v1.1.0"""

import json
import pytest
import importlib.util
import tempfile
from pathlib import Path
from unittest.mock import patch

# 直接从文件路径加载
_spec = importlib.util.spec_from_file_location(
    "repo_chinese_names",
    str(Path(__file__).parent / "__init__.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_chinese_name = _mod.get_chinese_name
format_repo_list = _mod.format_repo_list
REPO_CN_MAP = _mod.REPO_CN_MAP
REPO_CATEGORIES = _mod.REPO_CATEGORIES
get_category = _mod.get_category
list_categories = _mod.list_categories
list_by_category = _mod.list_by_category
search_repos = _mod.search_repos
get_all_mappings = _mod.get_all_mappings
get_unmapped_repos = _mod.get_unmapped_repos


class TestGetChineseName:
    """get_chinese_name 测试"""

    def test_known_repo(self):
        assert get_chinese_name("model-router") == "模型路由器"
        assert get_chinese_name("skill-router") == "技能路由器"
        assert get_chinese_name("omnimem") == "五层记忆系统"
        assert get_chinese_name("hermes-agent") == "Hermes智能体"

    def test_unknown_repo(self):
        assert get_chinese_name("unknown-repo") == "unknown-repo"
        assert get_chinese_name("") == ""

    def test_all_known_repos(self):
        """测试所有已知仓库都有映射"""
        for name in REPO_CN_MAP:
            result = get_chinese_name(name)
            assert result == REPO_CN_MAP[name]
            assert isinstance(result, str)
            assert len(result) > 0

    def test_fuzzy_case_insensitive(self):
        """模糊匹配: 大小写不兼容"""
        assert get_chinese_name("Model-Router", fuzzy=True) == "模型路由器"
        assert get_chinese_name("MODEL-ROUTER", fuzzy=True) == "模型路由器"

    def test_fuzzy_partial_match(self):
        """模糊匹配: 部分匹配"""
        # "model-router" 包含在 "model-router-v2" 中
        assert get_chinese_name("model-router-v2", fuzzy=True) == "模型路由器"

    def test_fuzzy_disabled_by_default(self):
        """默认不启用模糊匹配"""
        assert get_chinese_name("Model-Router") == "Model-Router"  # 原样返回


class TestFormatRepoList:
    """format_repo_list 测试"""

    def test_format_known_repos(self):
        repos = [
            {"name": "model-router", "description": "Router for models"},
            {"name": "omnimem", "description": "Memory system"},
        ]
        result = format_repo_list(repos)
        assert "模型路由器" in result
        assert "五层记忆系统" in result

    def test_format_unknown_repo(self):
        repos = [{"name": "my-unknown-repo", "description": "Something"}]
        result = format_repo_list(repos)
        assert "my-unknown-repo" in result

    def test_format_empty_list(self):
        result = format_repo_list([])
        assert result == "" or result.strip() == ""

    def test_format_mixed_repos(self):
        repos = [
            {"name": "model-router", "description": "Router"},
            {"name": "unknown-xyz", "description": "Unknown"},
        ]
        result = format_repo_list(repos)
        assert "模型路由器" in result
        assert "unknown-xyz" in result

    def test_format_with_category(self):
        repos = [{"name": "model-router", "description": "Router"}]
        result = format_repo_list(repos, show_category=True)
        assert "模型路由器" in result
        assert "[Hermes 插件]" in result


class TestRepoCNMap:
    """映射表测试"""

    def test_map_is_dict(self):
        assert isinstance(REPO_CN_MAP, dict)

    def test_map_not_empty(self):
        assert len(REPO_CN_MAP) > 0

    def test_all_values_are_strings(self):
        for key, value in REPO_CN_MAP.items():
            assert isinstance(key, str), f"Key {key!r} is not a string"
            assert isinstance(value, str), f"Value for {key!r} is not a string"
            assert len(key) > 0, f"Empty key found"
            assert len(value) > 0, f"Empty value for {key!r}"

    def test_hermes_plugins_present(self):
        """测试 Hermes 插件仓库都在映射中"""
        expected = ["model-router", "skill-router", "skill-pool",
                    "adaptive-multi-agent", "self-evolution",
                    "deepseek-cache-optimizer", "omnimem", "dev-lifecycle"]
        for name in expected:
            assert name in REPO_CN_MAP, f"{name} missing from REPO_CN_MAP"

    def test_finance_repos_present(self):
        """测试金融量化仓库在映射中"""
        expected = ["QuantDinger", "FinceptTerminal", "Hyper-Alpha-Arena", "YMOS"]
        for name in expected:
            assert name in REPO_CN_MAP, f"{name} missing from REPO_CN_MAP"

    def test_no_duplicate_keys_across_categories(self):
        """确保跨分类没有重复 key"""
        all_keys = []
        for cat_info in REPO_CATEGORIES.values():
            all_keys.extend(cat_info["repos"].keys())
        assert len(all_keys) == len(set(all_keys)), "存在重复的仓库名"


class TestCategories:
    """分类管理测试"""

    def test_list_categories(self):
        cats = list_categories()
        assert "hermes-plugins" in cats
        assert "finance" in cats
        assert "tools" in cats
        assert cats["hermes-plugins"] == "Hermes 插件"

    def test_get_category(self):
        assert get_category("model-router") == "Hermes 插件"
        assert get_category("QuantDinger") == "金融量化"
        assert get_category("OCH") == "工具平台"
        assert get_category("unknown-repo") is None

    def test_list_by_category(self):
        plugins = list_by_category("hermes-plugins")
        assert "model-router" in plugins
        assert "omnimem" in plugins
        assert len(plugins) > 5

    def test_category_matches_map(self):
        """分类中的仓库都应该在 REPO_CN_MAP 中"""
        for cat_key, cat_info in REPO_CATEGORIES.items():
            for repo_name in cat_info["repos"]:
                assert repo_name in REPO_CN_MAP, \
                    f"{repo_name} in category {cat_key} but not in REPO_CN_MAP"


class TestSearch:
    """搜索测试"""

    def test_search_by_english(self):
        results = search_repos("router")
        names = [r[0] for r in results]
        assert "model-router" in names
        assert "skill-router" in names

    def test_search_by_chinese(self):
        results = search_repos("路由")
        names = [r[0] for r in results]
        assert "model-router" in names

    def test_search_with_category(self):
        results = search_repos("model-router")
        assert len(results) >= 1
        # 应该包含分类信息
        assert results[0][2] == "Hermes 插件"

    def test_search_no_match(self):
        results = search_repos("xyznonexistent")
        assert len(results) == 0


class TestUnmappedRepos:
    """未映射仓库检测测试"""

    def test_all_mapped(self):
        repos = list(REPO_CN_MAP.keys())
        assert get_unmapped_repos(repos) == []

    def test_some_unmapped(self):
        repos = ["model-router", "my-new-repo", "another-one"]
        unmapped = get_unmapped_repos(repos)
        assert "my-new-repo" in unmapped
        assert "another-one" in unmapped
        assert "model-router" not in unmapped


class TestExternalConfig:
    """外部配置测试"""

    def test_get_all_mappings_includes_builtin(self):
        all_map = get_all_mappings()
        assert "model-router" in all_map
        assert len(all_map) >= len(REPO_CN_MAP)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
