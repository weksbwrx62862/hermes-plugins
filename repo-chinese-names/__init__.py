"""
repo-chinese-names 插件 v1.1.0

GitHub 仓库中文名映射，让仓库列表显示中文名。

v1.1.0 优化:
  - 分类管理: 仓库按类别分组 (插件/金融/工具/其他)
  - 外部配置: 支持从 ~/.hermes/repo-names.json 加载自定义映射
  - 模糊匹配: 大小写不兼容 + 部分名称匹配
  - CLI 管理: add/remove/list/search/categories 命令
  - 统计追踪: 记录映射使用频率
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 内置映射表 (分类) ──────────────────────────────────────────────────────────

REPO_CATEGORIES = {
    "hermes-plugins": {
        "label": "Hermes 插件",
        "repos": {
            "model-router": "模型路由器",
            "skill-router": "技能路由器",
            "skill-pool": "技能池索引",
            "adaptive-multi-agent": "自适应多智能体",
            "self-evolution": "自进化引擎",
            "deepseek-cache-optimizer": "缓存优化器",
            "omnimem": "五层记忆系统",
            "dev-lifecycle": "开发生命周期",
            "log-translator": "日志翻译器",
            "hermes-plugins": "Hermes插件仓库",
            "hermes-agent": "Hermes智能体",
        },
    },
    "finance": {
        "label": "金融量化",
        "repos": {
            "QuantDinger": "量化交易平台",
            "FinceptTerminal": "金融分析终端",
            "Hyper-Alpha-Arena": "AI合约交易",
            "YMOS": "人机投研系统",
            "trading-discipline": "交易纪律系统",
        },
    },
    "tools": {
        "label": "工具平台",
        "repos": {
            "OCH": "多智能体平台",
            "GitNexus": "代码知识图谱",
            "awesome-codex-skills": "Codex技能精选",
            "Expert-Suite-Skills": "AI专家套件",
            "project-analysis-rules": "项目分析规则",
            "TrendRadar": "热点监控雷达",
        },
    },
}

# 展平为 {repo_name: chinese_name} — 兼容旧接口
REPO_CN_MAP: Dict[str, str] = {}
REPO_NAME_TO_CATEGORY: Dict[str, str] = {}

for cat_key, cat_info in REPO_CATEGORIES.items():
    for repo_name, cn_name in cat_info["repos"].items():
        REPO_CN_MAP[repo_name] = cn_name
        REPO_NAME_TO_CATEGORY[repo_name] = cat_key

# ── 外部配置 ──────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(os.environ.get(
    "HERMES_REPO_NAMES_CONFIG",
    os.path.expanduser("~/.hermes/repo-names.json")
))

_external_map: Dict[str, str] = {}
_external_categories: Dict[str, str] = {}  # repo_name -> category_key

def _load_external_config():
    """加载外部配置文件"""
    global _external_map, _external_categories
    if not _CONFIG_PATH.exists():
        return
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # 简单格式: {"repo": "中文名"}
            if all(isinstance(v, str) for v in data.values()):
                _external_map = data
            # 分类格式: {"category_key": {"label": "...", "repos": {...}}}
            else:
                for cat_key, cat_info in data.items():
                    if isinstance(cat_info, dict) and "repos" in cat_info:
                        for repo_name, cn_name in cat_info["repos"].items():
                            _external_map[repo_name] = cn_name
                            _external_categories[repo_name] = cat_key
    except (json.JSONDecodeError, OSError):
        pass

_load_external_config()

def save_external_config(repo_name: str, cn_name: str, category: str = "custom"):
    """保存映射到外部配置文件"""
    _external_map[repo_name] = cn_name
    _external_categories[repo_name] = category

    # 读取现有配置
    existing = {}
    if _CONFIG_PATH.exists():
        try:
            existing = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    # 按分类格式写入
    if not isinstance(existing, dict) or not all(isinstance(v, str) for v in existing.values()):
        # 已经是分类格式
        if category not in existing:
            existing[category] = {"label": category, "repos": {}}
        if "repos" not in existing[category]:
            existing[category]["repos"] = {}
        existing[category]["repos"][repo_name] = cn_name
    else:
        # 简单格式，直接添加
        existing[repo_name] = cn_name

    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2),
                            encoding="utf-8")

def remove_external_config(repo_name: str) -> bool:
    """从外部配置移除映射"""
    if repo_name not in _external_map:
        return False
    del _external_map[repo_name]
    _external_categories.pop(repo_name, None)

    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if repo_name in data:
                    del data[repo_name]
                else:
                    for cat_info in data.values():
                        if isinstance(cat_info, dict) and "repos" in cat_info:
                            cat_info["repos"].pop(repo_name, None)
            _CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        except (json.JSONDecodeError, OSError):
            pass
    return True

# ── 统计 ──────────────────────────────────────────────────────────────────────

_usage_stats: Counter = Counter()

# ── 核心 API ──────────────────────────────────────────────────────────────────

def get_chinese_name(repo_name: str, fuzzy: bool = False) -> str:
    """
    获取仓库中文名，无映射则返回原名。

    Args:
        repo_name: 仓库名
        fuzzy: 是否启用模糊匹配 (大小写不兼容 + 部分匹配)
    """
    if not repo_name:
        return repo_name

    # 1. 精确匹配 (内置 + 外部)
    result = REPO_CN_MAP.get(repo_name) or _external_map.get(repo_name)
    if result:
        _usage_stats[repo_name] += 1
        return result

    if not fuzzy:
        return repo_name

    # 2. 大小写不兼容匹配
    lower = repo_name.lower()
    for name, cn in {**REPO_CN_MAP, **_external_map}.items():
        if name.lower() == lower:
            _usage_stats[name] += 1
            return cn

    # 3. 部分匹配 (仓库名包含在映射名中，或映射名包含在仓库名中)
    for name, cn in {**REPO_CN_MAP, **_external_map}.items():
        if name.lower() in lower or lower in name.lower():
            _usage_stats[name] += 1
            return cn

    return repo_name


def get_category(repo_name: str) -> Optional[str]:
    """获取仓库所属分类"""
    cat = REPO_NAME_TO_CATEGORY.get(repo_name) or _external_categories.get(repo_name)
    if cat:
        return REPO_CATEGORIES.get(cat, {}).get("label", cat)
    return None


def list_by_category(category_key: str) -> Dict[str, str]:
    """列出某个分类下的所有仓库"""
    cat_info = REPO_CATEGORIES.get(category_key)
    if cat_info:
        return dict(cat_info["repos"])
    # 外部配置
    return {k: v for k, v in _external_map.items()
            if _external_categories.get(k) == category_key}


def list_categories() -> Dict[str, str]:
    """列出所有分类 {key: label}"""
    cats = {k: v["label"] for k, v in REPO_CATEGORIES.items()}
    # 添加外部分类
    for cat in set(_external_categories.values()):
        if cat not in cats:
            cats[cat] = cat
    return cats


def search_repos(query: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    搜索仓库 (按中文名或英文名)。

    Returns: [(repo_name, chinese_name, category_label), ...]
    """
    query_lower = query.lower()
    results = []

    all_map = {**REPO_CN_MAP, **_external_map}
    all_cats = {**REPO_NAME_TO_CATEGORY, **_external_categories}

    for name, cn in all_map.items():
        if query_lower in name.lower() or query_lower in cn:
            cat_key = all_cats.get(name)
            cat_label = REPO_CATEGORIES.get(cat_key, {}).get("label", cat_key) if cat_key else None
            results.append((name, cn, cat_label))

    return results


def format_repo_list(repos: list[dict], show_category: bool = False) -> str:
    """
    格式化仓库列表，显示中文名。

    Args:
        repos: [{"name": ..., "description": ..., "primaryLanguage": {...}}, ...]
        show_category: 是否显示分类标签
    """
    lines = []
    for repo in repos:
        name = repo.get("name", "")
        cn_name = get_chinese_name(name)
        desc = repo.get("description", "")
        lang = repo.get("primaryLanguage", {}).get("name", "N/A")

        parts = [f"{name} ({cn_name})"]
        if show_category:
            cat = get_category(name)
            if cat:
                parts.append(f"[{cat}]")
        parts.append(f"| {lang} | {desc}")

        lines.append(" ".join(parts))
    return "\n".join(lines)


def get_usage_stats() -> Dict[str, int]:
    """获取映射使用统计"""
    return dict(_usage_stats.most_common())


def get_all_mappings() -> Dict[str, str]:
    """获取所有映射 (内置 + 外部)"""
    return {**REPO_CN_MAP, **_external_map}


def get_unmapped_repos(repo_names: List[str]) -> List[str]:
    """找出未映射的仓库名"""
    all_map = {**REPO_CN_MAP, **_external_map}
    return [name for name in repo_names if name not in all_map]


# ── 插件元信息 ──────────────────────────────────────────────────────────────────

PLUGIN_NAME = "repo-chinese-names"
PLUGIN_VERSION = "1.1.0"


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="GitHub 仓库中文名管理")
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="列出所有映射")
    p_list.add_argument("--category", "-c", help="按分类筛选")
    p_list.add_argument("--json", action="store_true", help="JSON 格式输出")

    # add
    p_add = sub.add_parser("add", help="添加映射")
    p_add.add_argument("repo", help="仓库名")
    p_add.add_argument("chinese", help="中文名")
    p_add.add_argument("--category", "-c", default="custom", help="分类")

    # remove
    p_rm = sub.add_parser("remove", help="移除外部映射")
    p_rm.add_argument("repo", help="仓库名")

    # search
    p_search = sub.add_parser("search", help="搜索仓库")
    p_search.add_argument("query", help="搜索关键词")

    # categories
    sub.add_parser("categories", help="列出所有分类")

    # lookup
    p_lookup = sub.add_parser("lookup", help="查询仓库中文名")
    p_lookup.add_argument("repo", help="仓库名")
    p_lookup.add_argument("--fuzzy", "-f", action="store_true", help="模糊匹配")

    # unmapped
    p_unmapped = sub.add_parser("unmapped", help="找出未映射的仓库")
    p_unmapped.add_argument("repos", nargs="+", help="仓库名列表")

    # stats
    sub.add_parser("stats", help="显示使用统计")

    args = parser.parse_args()

    if args.command == "list":
        mappings = get_all_mappings()
        if args.category:
            mappings = list_by_category(args.category)
        if args.json:
            print(json.dumps(mappings, ensure_ascii=False, indent=2))
        else:
            for name, cn in sorted(mappings.items()):
                cat = get_category(name)
                cat_str = f" [{cat}]" if cat else ""
                print(f"  {name:30s} → {cn}{cat_str}")
            print(f"\n共 {len(mappings)} 个映射")

    elif args.command == "add":
        save_external_config(args.repo, args.chinese, args.category)
        print(f"✅ 已添加: {args.repo} → {args.chinese} (分类: {args.category})")

    elif args.command == "remove":
        if remove_external_config(args.repo):
            print(f"✅ 已移除: {args.repo}")
        else:
            print(f"❌ 未找到外部映射: {args.repo}")

    elif args.command == "search":
        results = search_repos(args.query)
        if results:
            for name, cn, cat in results:
                cat_str = f" [{cat}]" if cat else ""
                print(f"  {name} → {cn}{cat_str}")
        else:
            print(f"未找到匹配 '{args.query}' 的仓库")

    elif args.command == "categories":
        cats = list_categories()
        for key, label in cats.items():
            repos = list_by_category(key)
            print(f"  {key} ({label}): {len(repos)} 个仓库")

    elif args.command == "lookup":
        cn = get_chinese_name(args.repo, fuzzy=args.fuzzy)
        cat = get_category(args.repo)
        print(f"  {args.repo} → {cn}")
        if cat:
            print(f"  分类: {cat}")

    elif args.command == "unmapped":
        unmapped = get_unmapped_repos(args.repos)
        if unmapped:
            print(f"未映射的仓库 ({len(unmapped)}):")
            for name in unmapped:
                print(f"  - {name}")
        else:
            print("所有仓库都已映射 ✅")

    elif args.command == "stats":
        stats = get_usage_stats()
        if stats:
            print("映射使用统计 (Top 10):")
            for name, count in list(stats.items())[:10]:
                cn = get_chinese_name(name)
                print(f"  {name:30s} ({cn}): {count} 次")
        else:
            print("暂无使用统计")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# ── Hermes 插件接口 ──────────────────────────────────────────────────────────

def register(ctx=None):
    """Hermes 插件注册入口"""
    pass  # 纯工具库，无需注册
