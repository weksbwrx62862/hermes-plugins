"""ProviderSelector 单元测试。"""

import importlib.util
import pathlib

import pytest


def _load_provider_selector():
    """通过文件路径加载 provider_selector 模块（目录名含连字符，无法直接 import）。"""
    module_path = (
        pathlib.Path(__file__).resolve().parent.parent / "provider_selector.py"
    )
    spec = importlib.util.spec_from_file_location(
        "provider_selector", str(module_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ProviderSelector


ProviderSelector = _load_provider_selector()


def _dep(**kwargs):
    """构造一个候选 deployment，填充默认值。"""
    defaults = {
        "provider": "openai",
        "model": "deepseek-v4-flash",
        "key": "sk-test",
        "base_url": "https://api.test.com/v1",
        "latency_ms": 100,
        "quality": 5,
        "cost": 1,
        "rpm_limit": 100,
        "tpm_limit": 10000,
        "region": "us",
    }
    defaults.update(kwargs)
    return defaults


class TestProviderSelector:
    def test_select_best_provider(self):
        """正常选择最优 Provider：综合评分最高者被选中。"""
        selector = ProviderSelector()
        deployments = [
            _dep(provider="openai", latency_ms=200, quality=3, cost=2, rpm_limit=100),
            _dep(provider="deepseek", latency_ms=100, quality=5, cost=1, rpm_limit=100),
            _dep(provider="stepfun", latency_ms=300, quality=2, cost=3, rpm_limit=100),
        ]
        selected, chain = selector.select("deepseek-v4-flash", deployments)

        assert selected["provider"] == "deepseek"
        assert [d["provider"] for d in chain] == ["openai", "stepfun"]

    def test_switch_when_quota_exhausted(self):
        """首选 Provider 配额耗尽时，自动切换到次优可用 Provider。"""
        selector = ProviderSelector()
        deployments = [
            _dep(provider="openai", latency_ms=50, quality=5, cost=1,
                  rpm_limit=100, rpm_used=100),
            _dep(provider="deepseek", latency_ms=100, quality=5, cost=1,
                  rpm_limit=100, rpm_used=10),
        ]
        selected, chain = selector.select("deepseek-v4-flash", deployments)

        assert selected["provider"] == "deepseek"
        # 配额耗尽的 openai 不应出现在降级链首位
        assert chain[0]["provider"] != "openai" or chain[0].get("exhausted")

    def test_fallback_chain_order(self):
        """降级链按综合评分从高到低排列，且不包含已选中 deployment。"""
        selector = ProviderSelector()
        deployments = [
            _dep(provider="a", latency_ms=10, quality=5, cost=1, rpm_limit=100),
            _dep(provider="b", latency_ms=50, quality=4, cost=1, rpm_limit=100),
            _dep(provider="c", latency_ms=100, quality=3, cost=2, rpm_limit=100),
            _dep(provider="d", latency_ms=200, quality=2, cost=3, rpm_limit=100),
        ]
        selected, chain = selector.select("deepseek-v4-flash", deployments)

        assert selected["provider"] == "a"
        assert [d["provider"] for d in chain] == ["b", "c", "d"]
        assert all(d["provider"] != "a" for d in chain)

    def test_region_preference_bonus(self):
        """Region 偏好应提升对应 deployment 的综合评分。"""
        selector = ProviderSelector()
        deployments = [
            _dep(provider="us", latency_ms=100, quality=5, cost=1, region="us"),
            _dep(provider="cn", latency_ms=110, quality=5, cost=1, region="cn"),
        ]
        # 无偏好时 us 略优
        selected, _ = selector.select("m", deployments)
        assert selected["provider"] == "us"

        # 偏好 cn 后应选中 cn
        selected, _ = selector.select(
            "m", deployments, region_preferences=["cn"]
        )
        assert selected["provider"] == "cn"

    def test_infer_provider_from_model_name(self):
        """兼容当前 flat 配置：deployment 缺少 provider 时从模型名推断。"""
        selector = ProviderSelector()
        deployments = [
            {
                "model": "qwen-plus",
                "key": "sk-test",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "latency_ms": 80,
                "quality": 4,
                "cost": 0,
                "rpm_limit": 100,
                "tpm_limit": 10000,
                "region": "cn",
            }
        ]
        selected, chain = selector.select("qwen-plus", deployments)

        assert selected["provider"] == "dashscope"
        assert chain == []

    def test_infer_provider_for_multiple_known_prefixes(self):
        """多个已知模型前缀均能正确推断 provider。"""
        selector = ProviderSelector()
        cases = [
            ("step-3.7-flash", "stepfun"),
            ("sensenova-6.7-flash-lite", "sensenova"),
            ("deepseek-chat", "deepseek"),
            ("agnes-2.0-flash", "custom:agnes"),
        ]
        for model, expected in cases:
            deployments = [{"model": model, "latency_ms": 100, "quality": 3,
                            "cost": 1, "rpm_limit": 100, "tpm_limit": 10000}]
            selected, _ = selector.select(model, deployments)
            assert selected["provider"] == expected, f"{model} 推断失败"

    def test_empty_deployments_returns_none(self):
        """无候选 deployment 时返回 None 与空降级链。"""
        selector = ProviderSelector()
        selected, chain = selector.select("m", [])
        assert selected is None
        assert chain == []
