"""域分组配置与过滤模块

管理域关键词、映射、聚合器、子域、特殊规则，以及基于域的候选技能过滤逻辑。
"""

import logging
import re
from typing import Any, Callable, Dict, Optional, Set, Tuple

logger = logging.getLogger("skill_router_init.domain_config")

# 默认域配置加载器，由入口模块注入，避免循环导入
_CONFIG_LOADER: Optional[Callable[[], Dict[str, Any]]] = None


def set_config_loader(loader: Callable[[], Dict[str, Any]]) -> None:
    """注入默认配置加载器"""
    global _CONFIG_LOADER
    _CONFIG_LOADER = loader


# ── 域分组默认配置（P1: 先选域再选技能，13 域）──
_DEFAULT_DOMAIN_CONFIG = {
    "top_k": 3,
    "keywords": {
        "code": ["代码", "编程", "开发", "bug", "debug", "git", "github", "pr", "commit", "deploy", "docker", "kubernetes", "ci", "cd", "plugin", "插件", "脚本", "script", "python", "javascript", "typescript", "rust", "go", "java"],
        "ml": ["模型", "训练", "微调", "推理", "inference", "train", "finetune", "lora", "gpu", "cuda", "vllm", "gguf", "量化", "benchmark", "评估"],
        "creative": ["图", "画", "设计", "视频", "音乐", "动画", "ascii", "pixel", "漫画", "illustration", "diagram", "svg", "艺术", "art", "手绘", "sketch", "excalidraw"],
        "productivity": ["文件", "文档", "邮件", "日历", "笔记", "notion", "obsidian", "pdf", "表格", "sheet", "drive", "ppt", "幻灯片", "演示文稿", "演示"],
        "research": ["论文", "搜索", "研究", "arxiv", "知识", "wiki", "新闻", "news", "report"],
        "finance": ["股票", "基金", "加密", "货币", "投资", "理财", "行情", "价格", "收益", "分析", "portfolio", "stock", "fund", "crypto", "etf"],
        "lark": ["飞书", "lark", "feishu", "审批", "考勤", "打卡", "通讯录", "多维表格", "云空间", "知识库", "日程", "会议", "妙记", "幻灯片", "画板", "妙搭", "日历", "邮箱"],
        "wechat-article": ["公众号", "微信", "wechat", "排版", "配图", "审稿", "发布", "推送", "选题", "爆款"],
        "finance-dianjin": ["尽调", "信贷", "授信", "核保", "理赔", "财富", "资产配置", "组合", "风险评估", "授信审查", "贷后", "风控", "反欺诈", "建模", "评分卡", "xgb"],
        "insurance-agent": ["保险", "代理人", "保单", "投保", "健康告知", "核保", "保障缺口", "续期"],
        "crypto": ["加密货币", "bitcoin", "btc", "eth", "crypto", "defi", "nft", "区块链"],
    },
    "map": {
        "code": ["software-development", "devops", "github", "autonomous-ai-agents", "mcp", "dogfood"],
        "ml": ["mlops", "training", "inference", "models", "evaluation", "data-science"],
        "creative": ["creative", "media"],
        "productivity": ["productivity", "note-taking", "email"],
        "research": ["research", "red-teaming"],
        "finance": ["finance"],
        "lark": ["productivity"],  # lark-* 技能全在 productivity 下，通过 name 前缀二次过滤
        "wechat-article": ["wechat-article"],
        "finance-dianjin": [
            "corporate-banker", "credit-review-expert", "credit-risk-manager",
            "financial-engineering-expert", "investment-advisor", "investment-researcher",
            "wealth-copilot", "L2-1_opportunity", "L2-2_outreach", "L2-3_strategy",
            "L2-4_qa", "L2-5_diagnosis", "L2-6_allocation", "L2-7_summary", "L2-8_companion",
        ],
        "insurance-agent": ["_disabled/insurance-agent"],
        "crypto": ["crypto"],
        "other": ["gaming", "leisure", "smart-home", "social-media", "cloud"],
    },
    "aggregators": {
        "finance": "finance-assistant",           # 统一金融助手
        "finance-dianjin": "finance-assistant",    # 金融助手作为入口
        "wechat-article": "wechat-article-main",   # 公众号一条龙入口
    },
    "subdomains": {
        "finance-dianjin": {
            "corporate-banker": {
                "keywords": ["尽调", "贷前", "访前", "路演", "授信申请", "财报分析", "股权穿透"],
                "prefixes": [
                    "credit-due-diligence", "credit-industry-analysis", "equity-penetration-analysis",
                    "financial-report-analysis", "pre-visit-plan", "product-roadshow",
                    "submit-credit-application", "visit-memo",
                ],
            },
            "credit-review-expert": {
                "keywords": ["授信审查", "关联交易", "押品", "准入", "进件"],
                "prefixes": [
                    "admission-rules-scan", "ai-risk-planning", "credit-case-intake-check",
                    "credit-collateral-risk-mgmt", "credit-related-party-detection",
                    "pre-visit-credit-analysis", "reviewer-visit-memo",
                ],
            },
            "credit-risk-manager": {
                "keywords": ["风控", "贷后", "风险监测", "风险暴露", "大额", "政策分析"],
                "prefixes": [
                    "credit-industry-rule-gen", "credit-large-exposure-mgmt", "credit-policy-analysis",
                    "credit-risk-cot", "credit-risk-extraction", "loan-risk-monitor",
                    "post-loan-management", "vlm-verifier",
                ],
            },
            "investment-advisor": {
                "keywords": ["热点", "可比公司", "资金流向", "股东", "诊断", "选股", "技术分析"],
                "prefixes": [
                    "a-market-hotspot-discovery", "comparable-company-analysis", "fund-diagnosis",
                    "fund-multi-factor-filter", "stock-fund-analysis", "stock-multi-factor-filter",
                    "stock-quote-analysis", "stock-shareholder-analysis", "stock-tech-analysis",
                ],
            },
            "investment-researcher": {
                "keywords": ["研究员", "宏观", "固收", "量化", "策略", "估值", "景气", "行业分析", "周报", "日报", "转债", "快评"],
                "prefixes": [
                    "announcement-analysis", "company-deep-analysis", "company-one-page-analysis",
                    "convertible-bond-valuation", "fixed-income", "global-finance-brief",
                    "global-macro-linkage", "industry-deep-analysis", "industry-one-page-analysis",
                    "macro-", "market-", "policy-flash", "quant-", "sector-allocation",
                    "strategy-daily", "valuation-prosperity",
                ],
            },
            "wealth-copilot": {
                "keywords": ["财富", "理财", "资产配置", "客户画像", "商机", "话术", "合规", "投教", "持仓诊断"],
                "prefixes": [
                    "client-", "maturity-client", "hot-product", "market-event", "market-morning",
                    "outreach-script", "wechat-moments", "client-segmentation", "competitor-product",
                    "sales-sop", "compliance-risk", "market-insight", "product-knowledge",
                    "fund-deep-research", "portfolio-health", "portfolio-risk", "asset-allocation-optimizer",
                    "family-financial", "investment-simulation", "smart-product", "daily-work-review",
                    "service-summary", "tomorrow-plan", "investor-education", "market-hotspot-digest",
                    "portfolio-alert",
                ],
            },
        },
    },
    "special_rules": {
        "lark": {"name_prefix": "lark-"},
        "finance": {
            "name_keywords": ["stock", "fund", "finance", "portfolio", "market", "crypto", "macro", "analysis", "technical", "news"],
            "always_include": ["finance-assistant"],
            # 个股价格/行情查询应路由到 stock-quote-analysis，而非 finance
            "exclude_when_single_stock": {
                "keywords": ["股价", "个股", "这只股票"],
                "redirect_to": "stock-quote-analysis",
            },
        },
        "creative": {
            # "生成音乐/直接生成歌曲"查询应路由到 heartmula，而非 songwriting-and-ai-music
            "redirect_music_gen": {
                "keywords": [
                    "生成音乐", "生成歌曲", "生成一首",
                    "输出音频", "文本转音乐",
                    "HeartMuLa", "Suno 替代",
                    # 短片段关键词，覆盖"歌词生成完整歌曲""给我生成一段音乐"等变体
                    "生成.*歌", "生成.*音乐", "Suno",
                ],
                "redirect_to": "heartmula",
            },
        },
    },
}


def _merge_domain_config(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并域配置：字典递归合并，列表直接替换"""
    result: Dict[str, Any] = {}
    for key, default_value in default.items():
        if key in override:
            override_value = override[key]
            if isinstance(default_value, dict) and isinstance(override_value, dict):
                result[key] = _merge_domain_config(default_value, override_value)
            else:
                result[key] = override_value
        else:
            result[key] = default_value
    for key, override_value in override.items():
        if key not in default:
            result[key] = override_value
    return result


def _load_domain_config(config_loader: Optional[Callable[[], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """加载域分组配置，与默认值深度合并"""
    loader = config_loader or _CONFIG_LOADER
    if loader is None:
        raise RuntimeError("未设置域配置加载器，请先调用 set_config_loader")
    plugin_config = loader()
    user_domain_config = plugin_config.get("domain_config", {})
    return _merge_domain_config(_DEFAULT_DOMAIN_CONFIG, user_domain_config)


def _validate_domain_config(domain_config: Dict[str, Any], skills: Dict[str, Dict[str, Any]]) -> None:
    """校验域配置中引用的技能是否存在于当前索引"""
    # 校验聚合器技能名
    aggregators = domain_config.get("aggregators", {})
    for domain, skill_name in aggregators.items():
        if skill_name not in skills:
            logger.warning("域 '%s' 的聚合器技能 '%s' 不在当前技能索引中", domain, skill_name)

    # 校验子域前缀是否至少能匹配到一个技能
    subdomains = domain_config.get("subdomains", {})
    all_skill_names = set(skills.keys())
    for domain, sub_domain_config in subdomains.items():
        for sub_name, sub_cfg in sub_domain_config.items():
            prefixes = sub_cfg.get("prefixes", [])
            if not prefixes:
                continue
            if not any(any(name.startswith(p) for name in all_skill_names) for p in prefixes):
                logger.warning(
                    "子域 '%s/%s' 的前缀 %s 未匹配到任何技能",
                    domain, sub_name, prefixes
                )


def _classify_domain(query: str, domain_config: Dict[str, Any]) -> Optional[str]:
    """根据查询文本快速分类到功能域

    使用关键词匹配，O(1) 复杂度。返回域名称或 None（无法分类）。
    """
    query_lower = query.lower()
    scores: Dict[str, int] = {}
    for domain, keywords in domain_config.get("keywords", {}).items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[domain] = score
    if not scores:
        return None
    return max(scores.items(), key=lambda item: item[1])[0]


def _get_domain_aggregator(domain: str, domain_config: Dict[str, Any]) -> Optional[str]:
    """获取域的聚合器入口技能名（如 finance → finance-assistant）"""
    return domain_config.get("aggregators", {}).get(domain)


def _filter_skills_by_domain(
    query: str,
    domain: str,
    skills: Dict[str, Dict[str, Any]],
    top_k: int,
    domain_config: Dict[str, Any],
) -> Tuple[Optional[Set[str]], Optional[str]]:
    """根据域配置过滤候选技能集

    返回 (domain_skill_names, domain_aggregator)：(None, None) 表示该域未配置。
    """
    domain_map = domain_config.get("map", {})
    if domain not in domain_map:
        return None, None

    domain_categories = set(domain_map[domain])
    domain_aggregator = _get_domain_aggregator(domain, domain_config)

    # 按 category 收集域内技能
    domain_only: Set[str] = {
        name for name, info in skills.items()
        if info.get("category") in domain_categories
    }

    special_rules = domain_config.get("special_rules", {})

    # lark 特殊规则：按名称前缀二次过滤
    if domain in special_rules and "name_prefix" in special_rules[domain]:
        prefix = special_rules[domain]["name_prefix"]
        domain_only = {n for n in domain_only if n.startswith(prefix)}

    # 通用重定向规则：匹配 redirect_* 配置项
    for rule_key, rule_cfg in special_rules.get(domain, {}).items():
        if rule_key.startswith("redirect_") and isinstance(rule_cfg, dict):
            redirect_keywords = rule_cfg.get("keywords", [])
            redirect_to = rule_cfg.get("redirect_to")
            if redirect_keywords and redirect_to:
                query_lower_for_redirect = query.lower()
                matched = False
                for kw in redirect_keywords:
                    if ".*" in kw or ".+" in kw:
                        # 正则模式匹配
                        if re.search(kw, query, re.IGNORECASE):
                            matched = True
                            break
                    else:
                        # 子串匹配
                        if kw.lower() in query_lower_for_redirect:
                            matched = True
                            break
                if matched and redirect_to in skills:
                    logger.debug("域重定向 (%s): %s → %s", rule_key, query, redirect_to)
                    return {redirect_to}, redirect_to

    # finance 特殊规则：个股价格查询重定向到 stock-quote-analysis
    if domain in special_rules and "exclude_when_single_stock" in special_rules[domain]:
        redirect_cfg = special_rules[domain]["exclude_when_single_stock"]
        redirect_keywords = redirect_cfg.get("keywords", [])
        redirect_to = redirect_cfg.get("redirect_to")
        query_lower_for_redirect = query.lower()
        if any(kw in query_lower_for_redirect for kw in redirect_keywords) and redirect_to:
            if redirect_to in skills:
                logger.debug("个股查询重定向: %s → %s", query, redirect_to)
                return {redirect_to}, redirect_to

    # finance 特殊规则：补充名称关键词匹配，并强制包含指定技能
    if domain in special_rules and "name_keywords" in special_rules[domain]:
        name_keywords = special_rules[domain]["name_keywords"]
        for name, info in skills.items():
            name_lower = name.lower()
            if any(kw in name_lower for kw in name_keywords):
                domain_only.add(name)
        always_include = special_rules[domain].get("always_include", [])
        for name in always_include:
            if name in skills:
                domain_only.add(name)

    # 子域细化：命中子域关键词后按 prefixes 进一步缩小
    subdomains = domain_config.get("subdomains", {})
    if domain in subdomains:
        query_lower = query.lower()
        sub_domain_config = subdomains[domain]
        matched_sub: Optional[str] = None
        for sub_name, sub_cfg in sub_domain_config.items():
            keywords = sub_cfg.get("keywords", [])
            if any(kw in query_lower for kw in keywords):
                matched_sub = sub_name
                break
        if matched_sub and matched_sub in sub_domain_config:
            prefixes = sub_domain_config[matched_sub].get("prefixes", [])
            if prefixes:
                domain_only = {
                    n for n in domain_only
                    if any(n.startswith(p) for p in prefixes)
                }
                logger.debug("%s 子域 %s 过滤后剩余 %d 个技能", domain, matched_sub, len(domain_only))

    # Top-K 截断：避免同类技能全量注入，同时保留向量检索空间
    domain_top_k = domain_config.get("top_k", 3)
    domain_only_sorted = sorted(domain_only)
    if len(domain_only_sorted) > domain_top_k * 4:
        domain_skill_names: Set[str] = set(domain_only_sorted[:domain_top_k * 4])
    else:
        domain_skill_names = domain_only

    # 候选不足时加入未分类技能兜底
    if len(domain_skill_names) < top_k:
        domain_skill_names |= {
            name for name, info in skills.items()
            if info.get("category") == "uncategorized"
        }

    logger.debug(
        "域分类 v2: %s → %d/%d 个候选 (聚合器=%s)",
        domain, len(domain_skill_names), len(skills), domain_aggregator or "无",
    )

    return domain_skill_names, domain_aggregator
