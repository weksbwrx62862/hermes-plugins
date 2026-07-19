"""域分组配置化单元测试

验证 _DEFAULT_DOMAIN_CONFIG、_load_domain_config、_classify_domain、
_filter_skills_by_domain、_validate_domain_config 的行为符合预期。
"""

import importlib.util
import logging
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INIT_PATH = os.path.join(ROOT, "__init__.py")


def _load_init():
    """通过 importlib 加载 skill-router 的 __init__.py。

    目录名含连字符，无法作为普通包导入，因此使用文件路径加载。
    """
    spec = importlib.util.spec_from_file_location("skill_router_init", _INIT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["skill_router_init"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_init()


# ── 默认配置等价性 ──

def test_default_domain_config_contains_all_domains(module):
    """默认配置应包含与原常量一致的域集合"""
    cfg = module._DEFAULT_DOMAIN_CONFIG
    assert cfg["top_k"] == 3
    # 原 keywords 映射不含 other，map 映射包含 other
    expected_keyword_domains = {
        "code", "ml", "creative", "productivity", "research", "finance",
        "lark", "wechat-article", "finance-dianjin", "insurance-agent", "crypto",
    }
    expected_map_domains = expected_keyword_domains | {"other"}
    assert set(cfg["keywords"].keys()) == expected_keyword_domains
    assert set(cfg["map"].keys()) == expected_map_domains


def test_default_domain_config_finance_keywords(module):
    """finance 域默认关键词应与原硬编码一致"""
    assert "股票" in module._DEFAULT_DOMAIN_CONFIG["keywords"]["finance"]
    assert "fund" in module._DEFAULT_DOMAIN_CONFIG["keywords"]["finance"]


def test_default_domain_config_aggregators(module):
    """默认聚合器映射应与原硬编码一致"""
    aggregators = module._DEFAULT_DOMAIN_CONFIG["aggregators"]
    assert aggregators["finance"] == "finance-assistant"
    assert aggregators["finance-dianjin"] == "finance-assistant"
    assert aggregators["wechat-article"] == "wechat-article-main"


def test_default_domain_config_subdomains(module):
    """默认子域配置应包含 finance-dianjin 下 6 个子域"""
    subdomains = module._DEFAULT_DOMAIN_CONFIG["subdomains"]["finance-dianjin"]
    expected_subs = {
        "corporate-banker", "credit-review-expert", "credit-risk-manager",
        "investment-advisor", "investment-researcher", "wealth-copilot",
    }
    assert set(subdomains.keys()) == expected_subs
    assert "credit-due-diligence" in subdomains["corporate-banker"]["prefixes"]
    assert "尽调" in subdomains["corporate-banker"]["keywords"]


def test_default_domain_config_special_rules(module):
    """默认特殊规则应与原硬编码一致"""
    rules = module._DEFAULT_DOMAIN_CONFIG["special_rules"]
    assert rules["lark"]["name_prefix"] == "lark-"
    assert "finance-assistant" in rules["finance"]["always_include"]
    assert "stock" in rules["finance"]["name_keywords"]


# ── 自定义配置生效 ──

def test_classify_domain_with_custom_keywords(module):
    """自定义 keywords 可改变 _classify_domain 结果"""
    default_cfg = module._DEFAULT_DOMAIN_CONFIG
    custom_cfg = module._merge_domain_config(
        default_cfg,
        {"keywords": {"custom-domain": ["自定义关键词"]}}
    )
    # 默认查询不应命中任何域
    assert module._classify_domain("不含关键词的查询", default_cfg) is None
    # 自定义关键词应命中 custom-domain
    assert module._classify_domain("包含自定义关键词的查询", custom_cfg) == "custom-domain"


def test_filter_skills_by_domain_with_custom_map(module):
    """自定义 map 可改变候选技能集"""
    skills = {
        "skill-a": {"category": "custom-category"},
        "skill-b": {"category": "other-category"},
    }
    default_cfg = module._DEFAULT_DOMAIN_CONFIG
    custom_cfg = module._merge_domain_config(
        default_cfg,
        {"map": {"custom-domain": ["custom-category"]}}
    )
    names, aggregator = module._filter_skills_by_domain(
        "查询", "custom-domain", skills, top_k=5, domain_config=custom_cfg
    )
    assert names == {"skill-a"}
    assert aggregator is None


# ── 特殊规则 ──

def test_lark_special_rule_filters_by_prefix(module):
    """lark 特殊规则按 name_prefix 过滤 productivity 技能"""
    skills = {
        "lark-calendar": {"category": "productivity"},
        "lark-mail": {"category": "productivity"},
        "other-productivity": {"category": "productivity"},
    }
    cfg = module._DEFAULT_DOMAIN_CONFIG
    names, aggregator = module._filter_skills_by_domain(
        "飞书日程", "lark", skills, top_k=5, domain_config=cfg
    )
    assert names == {"lark-calendar", "lark-mail"}


def test_lark_prefix_rule_is_configurable(module):
    """lark 前缀规则可通过 special_rules 配置修改"""
    skills = {
        "feishu-calendar": {"category": "productivity"},
        "lark-calendar": {"category": "productivity"},
    }
    default_cfg = module._DEFAULT_DOMAIN_CONFIG
    custom_cfg = module._merge_domain_config(
        default_cfg,
        {"special_rules": {"lark": {"name_prefix": "feishu-"}}}
    )
    names, _ = module._filter_skills_by_domain(
        "飞书日程", "lark", skills, top_k=5, domain_config=custom_cfg
    )
    assert names == {"feishu-calendar"}


def test_finance_special_rule_includes_name_keywords(module):
    """finance 特殊规则补充名称关键词匹配技能"""
    skills = {
        "stock-quote-analysis": {"category": "finance"},
        "some-stock-helper": {"category": "other"},
        "finance-assistant": {"category": "finance"},
    }
    cfg = module._DEFAULT_DOMAIN_CONFIG
    names, aggregator = module._filter_skills_by_domain(
        "股票行情", "finance", skills, top_k=5, domain_config=cfg
    )
    assert "stock-quote-analysis" in names
    assert "some-stock-helper" in names
    assert "finance-assistant" in names
    assert aggregator == "finance-assistant"


def test_finance_always_include_is_configurable(module):
    """finance 强制包含项可通过 special_rules 配置修改"""
    skills = {
        "custom-finance-entry": {"category": "finance"},
        # finance-assistant 不属于 finance category，仅通过 always_include 引入
        "finance-assistant": {"category": "other"},
    }
    default_cfg = module._DEFAULT_DOMAIN_CONFIG
    custom_cfg = module._merge_domain_config(
        default_cfg,
        {"special_rules": {"finance": {"name_keywords": [], "always_include": ["custom-finance-entry"]}}}
    )
    names, _ = module._filter_skills_by_domain(
        "理财", "finance", skills, top_k=5, domain_config=custom_cfg
    )
    assert "custom-finance-entry" in names
    assert "finance-assistant" not in names


# ── 子域过滤 ──

def test_subdomain_filter_by_keywords_and_prefixes(module):
    """finance-dianjin 子域按 keywords + prefixes 过滤生效"""
    skills = {
        "credit-due-diligence": {"category": "corporate-banker"},
        "stock-quote-analysis": {"category": "investment-advisor"},
        "fund-diagnosis": {"category": "investment-advisor"},
    }
    cfg = module._DEFAULT_DOMAIN_CONFIG
    # 命中 corporate-banker 子域关键词
    names, _ = module._filter_skills_by_domain(
        "尽调报告", "finance-dianjin", skills, top_k=5, domain_config=cfg
    )
    assert names == {"credit-due-diligence"}

    # 命中 investment-advisor 子域关键词
    names, _ = module._filter_skills_by_domain(
        "选股诊断", "finance-dianjin", skills, top_k=5, domain_config=cfg
    )
    assert names == {"stock-quote-analysis", "fund-diagnosis"}


def test_subdomain_prefixes_are_configurable(module):
    """子域 prefixes 可通过配置覆盖"""
    skills = {
        "custom-due-diligence": {"category": "corporate-banker"},
        "credit-due-diligence": {"category": "corporate-banker"},
    }
    default_cfg = module._DEFAULT_DOMAIN_CONFIG
    custom_cfg = module._merge_domain_config(
        default_cfg,
        {
            "subdomains": {
                "finance-dianjin": {
                    "corporate-banker": {
                        "keywords": ["尽调"],
                        "prefixes": ["custom-due-diligence"],
                    }
                }
            }
        }
    )
    names, _ = module._filter_skills_by_domain(
        "尽调", "finance-dianjin", skills, top_k=5, domain_config=custom_cfg
    )
    assert names == {"custom-due-diligence"}


# ── 配置校验 ──

def test_validate_domain_config_warns_unknown_aggregator(module, caplog):
    """未知聚合器技能名应触发 warning"""
    cfg = module._DEFAULT_DOMAIN_CONFIG.copy()
    cfg["aggregators"] = {**cfg.get("aggregators", {}), "new-domain": "nonexistent-skill"}
    skills = {"finance-assistant": {"category": "finance"}}
    with caplog.at_level(logging.WARNING, logger="skill_router_init"):
        module._validate_domain_config(cfg, skills)
    assert "nonexistent-skill" in caplog.text
    assert "new-domain" in caplog.text


def test_validate_domain_config_warns_unmatched_subdomain_prefixes(module, caplog):
    """子域 prefixes 无匹配技能时应触发 warning"""
    cfg = module._DEFAULT_DOMAIN_CONFIG.copy()
    cfg["subdomains"] = {
        "finance-dianjin": {
            "no-match": {
                "keywords": ["尽调"],
                "prefixes": ["zzz-no-such-prefix"],
            }
        }
    }
    skills = {"credit-due-diligence": {"category": "corporate-banker"}}
    with caplog.at_level(logging.WARNING, logger="skill_router_init"):
        module._validate_domain_config(cfg, skills)
    assert "zzz-no-such-prefix" in caplog.text


# ── 配置加载 ──

def test_merge_domain_config_deep_merge(module):
    """_merge_domain_config 对字典递归合并，列表直接替换"""
    default = {
        "top_k": 3,
        "keywords": {"finance": ["股票"], "ml": ["模型"]},
        "subdomains": {"finance-dianjin": {"corporate-banker": {"keywords": ["尽调"], "prefixes": ["p1"]}}},
    }
    override = {
        "keywords": {"finance": ["基金"]},
        "subdomains": {"finance-dianjin": {"corporate-banker": {"keywords": ["授信"]}}},
    }
    merged = module._merge_domain_config(default, override)
    assert merged["top_k"] == 3
    assert merged["keywords"]["finance"] == ["基金"]
    assert merged["keywords"]["ml"] == ["模型"]
    assert merged["subdomains"]["finance-dianjin"]["corporate-banker"]["keywords"] == ["授信"]
    assert merged["subdomains"]["finance-dianjin"]["corporate-banker"]["prefixes"] == ["p1"]
