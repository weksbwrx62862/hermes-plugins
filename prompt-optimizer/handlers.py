"""prompt-optimizer 插件 — 工具处理器。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

try:
    from .core import (
        score_prompt, generate_optimization_guide, build_compare_framework,
        TEMPLATES, OPTIMIZATION_FRAMEWORK,
    )
    from .store import PromptStore, init_db
except ImportError:
    from core import (
        score_prompt, generate_optimization_guide, build_compare_framework,
        TEMPLATES, OPTIMIZATION_FRAMEWORK,
    )
    from store import PromptStore, init_db

logger = logging.getLogger("plugins.prompt-optimizer")

_store: Optional[PromptStore] = None


def _get_store() -> PromptStore:
    global _store
    if _store is None:
        init_db()
        _store = PromptStore()
    return _store


# ─────────────────────────────────────────────────────
# prompt_optimize
# ─────────────────────────────────────────────────────

def handle_prompt_optimize(args: Dict[str, Any] = None, **kwargs) -> str:
    if not args or not isinstance(args, dict):
        return json.dumps({"success": False, "error": "args 参数缺失"}, ensure_ascii=False)
    action = args.get("action")
    if not action:
        return json.dumps({"success": False, "error": "action 参数缺失"}, ensure_ascii=False)

    if action == "template":
        topic = args.get("topic", "通用")
        mode = args.get("mode", "user")
        if topic not in TEMPLATES:
            topic = "通用"
        template = TEMPLATES[topic][mode]
        mode_label = "用户提示词" if mode == "user" else "系统提示词"
        return (
            f"📝 {topic}场景 - {mode_label}优化模板\n"
            f"{'=' * 40}\n\n"
            f"{template}\n\n"
            f"{'=' * 40}\n"
            f"使用方法：将 [方括号] 中的内容替换为你的实际需求。"
        )

    prompt = args.get("prompt", "")
    mode = args.get("mode", "user")

    if not prompt:
        return "❌ 请提供待优化的提示词（prompt 参数）"

    # 分析并生成优化指导
    guide = generate_optimization_guide(prompt, mode)
    analysis = score_prompt(prompt, mode)

    mode_label = "用户提示词" if mode == "user" else "系统提示词"
    result_parts = [
        f"🔍 {mode_label}优化分析",
        f"{'=' * 40}",
        f"📊 当前评分: {analysis['total']}/10",
        f"📏 长度: {analysis['length']} 字符",
        f"🏗️ 有结构: {'✅' if analysis['has_structure'] else '❌'}",
        "",
        "📊 各维度评分:",
    ]

    dims = OPTIMIZATION_FRAMEWORK["dimensions" if mode == "user" else "system_dimensions"]
    for dim in dims:
        score = analysis["scores"][dim["key"]]
        bar = "█" * score + "░" * (10 - score)
        result_parts.append(f"  {dim['name']:12} [{bar}] {score}/10")

    if guide["suggestions"]:
        result_parts.append("")
        result_parts.append(f"⚠️ 缺少 {guide['missing_count']} 个维度:")
        for s in guide["suggestions"]:
            result_parts.append(f"  • {s['dimension']}: {s['suggestion']}")
            result_parts.append(f"    示例: {s['example']}")

    # 如果是 save 操作，保存到 Prompt Garden
    if action == "save":
        name = args.get("name", "")
        if not name:
            return "❌ 保存时必须提供 name 参数"
        tags = args.get("tags", "")
        store = _get_store()
        save_result = store.save(
            name=name, prompt=prompt, mode=mode,
            description=f"优化前原始版本 (评分: {analysis['total']})",
            tags=tags,
        )
        result_parts.append("")
        result_parts.append(f"💾 已保存到 Prompt Garden: {save_result}")

    result_parts.append("")
    result_parts.append("💡 优化建议：请根据上述缺失维度，补全提示词的对应部分。")
    result_parts.append("   使用 prompt_optimize(action='template', topic='写作') 获取对应场景模板。")

    return "\n".join(result_parts)


# ─────────────────────────────────────────────────────
# prompt_analyze
# ─────────────────────────────────────────────────────

def handle_prompt_analyze(args: Dict[str, Any], **kwargs) -> str:
    prompt = args.get("prompt", "")
    mode = args.get("mode", "user")

    if not prompt:
        return "❌ 请提供待分析的提示词（prompt 参数）"

    analysis = score_prompt(prompt, mode)
    mode_label = "用户提示词" if mode == "user" else "系统提示词"

    dims = OPTIMIZATION_FRAMEWORK["dimensions" if mode == "user" else "system_dimensions"]

    parts = [
        f"📊 {mode_label}质量分析报告",
        f"{'=' * 40}",
        f"📏 长度: {analysis['length']} 字符",
        f"🏗️ 有结构: {'✅' if analysis['has_structure'] else '❌'}",
        f"📈 综合评分: {analysis['total']}/10",
        "",
        "各维度详情:",
    ]

    for dim in dims:
        score = analysis["scores"][dim["key"]]
        bar = "█" * score + "░" * (10 - score)
        level = "优秀" if score >= 8 else "良好" if score >= 6 else "一般" if score >= 4 else "需改进"
        parts.append(f"  {dim['name']:12} [{bar}] {score}/10 ({level})")

    # 评级
    total = analysis["total"]
    if total >= 8:
        grade = "🏆 A — 优秀，提示词结构完整，可直接使用"
    elif total >= 6:
        grade = "🥈 B — 良好，建议补全部分维度"
    elif total >= 4:
        grade = "🥉 C — 一般，需要较多优化"
    else:
        grade = "⚠️ D — 需大幅优化，建议使用模板重写"

    parts.append(f"\n{grade}")

    if analysis["missing"]:
        parts.append(f"\n缺少维度: {', '.join(m['name'] for m in analysis['missing'])}")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────
# prompt_compare
# ─────────────────────────────────────────────────────

def handle_prompt_compare(args: Dict[str, Any], **kwargs) -> str:
    prompt_a = args.get("prompt_a", "")
    prompt_b = args.get("prompt_b", "")
    task_context = args.get("task_context", "")

    if not prompt_a or not prompt_b:
        return "❌ 请提供两个版本的提示词（prompt_a 和 prompt_b）"

    score_a = score_prompt(prompt_a)
    score_b = score_prompt(prompt_b)

    framework = build_compare_framework()

    parts = [
        "⚖️ 提示词 A/B 对比报告",
        "=" * 40,
    ]

    if task_context:
        parts.append(f"📋 任务上下文: {task_context}")
        parts.append("")

    parts.append("📊 综合评分对比:")
    parts.append(f"  版本 A: {score_a['total']}/10 ({len(prompt_a)} 字符)")
    parts.append(f"  版本 B: {score_b['total']}/10 ({len(prompt_b)} 字符)")
    diff = score_b["total"] - score_a["total"]
    if diff > 0:
        parts.append(f"  📈 版本 B 优于版本 A (+{diff})")
    elif diff < 0:
        parts.append(f"  📉 版本 A 优于版本 B ({diff})")
    else:
        parts.append(f"  ➡️ 两版本评分持平")

    parts.append("")
    parts.append("📊 各维度对比:")

    dims = OPTIMIZATION_FRAMEWORK["dimensions"]
    for dim in dims:
        sa = score_a["scores"][dim["key"]]
        sb = score_b["scores"][dim["key"]]
        winner = "A" if sa > sb else "B" if sb > sa else "="
        bar_a = "█" * sa + "░" * (10 - sa)
        bar_b = "█" * sb + "░" * (10 - sb)
        parts.append(f"  {dim['name']}:")
        parts.append(f"    A [{bar_a}] {sa}/10")
        parts.append(f"    B [{bar_b}] {sb}/10  {'← B更优' if sb > sa else '← A更优' if sa > sb else '持平'}")

    # 差异维度
    diff_dims = []
    for dim in dims:
        sa = score_a["scores"][dim["key"]]
        sb = score_b["scores"][dim["key"]]
        if abs(sa - sb) >= 2:
            diff_dims.append((dim["name"], sa, sb))

    if diff_dims:
        parts.append("")
        parts.append("🔑 关键差异维度:")
        for name, sa, sb in diff_dims:
            better = "B" if sb > sa else "A"
            parts.append(f"  • {name}: {better} 版本明显更优 (差 {abs(sa - sb)} 分)")

    # 对比框架说明
    parts.append("")
    parts.append("📐 对比维度说明:")
    for d in framework["dimensions"]:
        parts.append(f"  • {d['name']}: {d['desc']}")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────
# prompt_garden
# ─────────────────────────────────────────────────────

def handle_prompt_garden(args: Dict[str, Any], **kwargs) -> str:
    action = args["action"]
    store = _get_store()

    if action == "save":
        name = args.get("name", "")
        prompt = args.get("prompt", "")
        if not name or not prompt:
            return "❌ save 需要 name 和 prompt 参数"
        result = store.save(
            name=name, prompt=prompt, mode=args.get("mode", "user"),
            description=args.get("description", ""),
            tags=args.get("tags", ""),
        )
        return f"💾 保存结果: {json.dumps(result, ensure_ascii=False)}"

    elif action == "list":
        tag = args.get("tags")
        mode = args.get("mode")
        items = store.list_all(tag=tag, mode=mode)
        if not items:
            return "📭 Prompt Garden 为空，使用 prompt_garden(action='save') 开始积累你的提示词资产"
        parts = [f"🌱 Prompt Garden ({len(items)} 条)", "=" * 40]
        for item in items:
            mode_icon = "🤖" if item["mode"] == "system" else "👤"
            tags_str = f" [{item['tags']}]" if item["tags"] else ""
            parts.append(
                f"  {mode_icon} {item['name']} (v{item['version']}){tags_str}"
                f"  — {item['description'][:40] if item['description'] else '无描述'}"
            )
        return "\n".join(parts)

    elif action == "get":
        name = args.get("name", "")
        if not name:
            return "❌ get 需要 name 参数"
        item = store.get(name, mode=args.get("mode", "user"))
        if not item:
            return f"❌ 未找到: {name}"
        return (
            f"📄 {item['name']} (v{item['version']}, {item['mode']}模式)\n"
            f"{'=' * 40}\n"
            f"📝 描述: {item['description'] or '无'}\n"
            f"🏷️ 标签: {item['tags'] or '无'}\n"
            f"{'=' * 40}\n"
            f"{item['prompt']}"
        )

    elif action == "search":
        query = args.get("query", "")
        if not query:
            return "❌ search 需要 query 参数"
        results = store.search(query)
        if not results:
            return f"🔍 未找到匹配 '{query}' 的提示词"
        parts = [f"🔍 搜索结果: '{query}' ({len(results)} 条)", "=" * 40]
        for item in results:
            mode_icon = "🤖" if item["mode"] == "system" else "👤"
            parts.append(f"  {mode_icon} {item['name']} (v{item['version']}) — {item['description'][:50] if item['description'] else '无描述'}")
        return "\n".join(parts)

    elif action == "delete":
        name = args.get("name", "")
        if not name:
            return "❌ delete 需要 name 参数"
        deleted = store.delete(name, mode=args.get("mode", "user"))
        return f"🗑️ 已删除: {name}" if deleted else f"❌ 未找到: {name}"

    elif action == "export":
        data = store.export_all()
        return f"📦 导出数据:\n{data}"

    elif action == "history":
        name = args.get("name", "")
        if not name:
            return "❌ history 需要 name 参数"
        versions = store.history(name, mode=args.get("mode", "user"))
        if not versions:
            return f"❌ 未找到: {name}"
        parts = [f"📜 版本历史: {name}", "=" * 40]
        for v in versions:
            current = " ← 当前" if v["is_current"] else ""
            parts.append(f"  v{v['version']}{current} ({v['hash']})")
            # 显示前 100 字符
            preview = v["prompt"][:100].replace("\n", " ")
            parts.append(f"    {preview}...")
        return "\n".join(parts)

    return f"❌ 未知操作: {action}"


# ─────────────────────────────────────────────────────
# pre_llm_call hook — 自动检测低质量提示词
# ─────────────────────────────────────────────────────

# 指令型关键词（中文 + 英文）
_INSTRUCTION_PATTERNS_ZH = [
    r"帮我", r"写一", r"生成", r"创建", r"翻译", r"分析", r"总结",
    r"解释", r"设计", r"优化", r"编写", r"给出", r"提供", r"列出",
    r"比较", r"评估", r"推荐", r"搜索", r"查找", r"计算", r"预测",
    r"模拟", r"制定", r"规划", r"描述", r"概述", r"梳理",
]
_INSTRUCTION_PATTERNS_EN = [
    r"write ", r"create ", r"generate ", r"translate ", r"analyze ",
    r"summarize ", r"explain ", r"design ", r"build ", r"make ",
    r"help me", r"list ", r"compare ", r"evaluate ", r"recommend ",
]


def _looks_like_prompt_instruction(message: str) -> bool:
    """检测消息是否像一条提示词指令（而不是闲聊/问答/确认）。"""
    import re
    text = message.strip()
    # 太短不算，太长通常是上下文
    if len(text) < 6 or len(text) > 500:
        return False
    # 已经有结构的不算（说明用户已经很会写了）
    if re.search(r'^\d+[.、)\s]', text, re.MULTILINE):
        return False
    if re.search(r'^#{1,3}\s', text, re.MULTILINE):
        return False
    # 多个约束/维度已经明确的不算（含逗号分隔的多条要求）
    comma_count = text.count('，') + text.count(',')
    if comma_count >= 3:
        return False
    # 包含指令关键词
    text_lower = text.lower()
    for p in _INSTRUCTION_PATTERNS_ZH:
        if re.search(p, text):
            return True
    for p in _INSTRUCTION_PATTERNS_EN:
        if p in text_lower:
            return True
    return False


def handle_pre_llm_call(**kwargs) -> Optional[Dict[str, Any]]:
    """pre_llm_call hook：自动检测低质量提示词并注入优化建议。"""
    import re

    user_message = kwargs.get("user_message", "")
    if not user_message or not _looks_like_prompt_instruction(user_message):
        return None

    analysis = score_prompt(user_message, mode="user")
    total = analysis["total"]

    # 高于 6 分不打扰
    if total >= 6.0:
        return None

    missing_names = [m["name"] for m in analysis["missing"]]
    missing_str = "、".join(missing_names) if missing_names else "无"

    hint = (
        f"[Prompt Optimizer 自动检测] 用户指令质量评分: {total}/10 | "
        f"缺失维度: {missing_str} | "
        f"提示：可以使用 prompt_optimize 工具优化此指令，"
        f"或主动建议用户补全角色、对象、结构、约束等维度。"
    )
    return {"context": hint}
