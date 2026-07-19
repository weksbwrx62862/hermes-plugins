"""prompt-optimizer 插件 — 优化引擎与分析框架。

核心理念（来自 prompt-optimizer 项目）：
- 提示词应该被设计，不是被试出来
- 六维优化：角色、对象、结构、风格、约束、输出目标
- System Prompt 和 User Prompt 分开处理
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────
# 六维优化框架
# ─────────────────────────────────────────────────────

OPTIMIZATION_FRAMEWORK = {
    "dimensions": [
        {"key": "role", "name": "角色(Role)", "desc": "明确 AI 应该扮演什么身份/专家"},
        {"key": "object", "name": "对象(Object)", "desc": "明确目标受众/处理对象是谁"},
        {"key": "structure", "name": "结构(Structure)", "desc": "要求分步骤、分层次、有逻辑"},
        {"key": "style", "name": "风格(Style)", "desc": "语言风格：口语化/专业/学术/轻松"},
        {"key": "constraints", "name": "约束(Constraints)", "desc": "边界条件：长度、格式、禁止项"},
        {"key": "output", "name": "输出目标(Output)", "desc": "明确期望的输出格式和标准"},
    ],
    "system_dimensions": [
        {"key": "identity", "name": "身份(Identity)", "desc": "你是谁，什么角色"},
        {"key": "expertise", "name": "专业领域(Expertise)", "desc": "擅长什么，知识范围"},
        {"key": "behavior", "name": "行为准则(Behavior)", "desc": "如何回应，沟通风格"},
        {"key": "boundaries", "name": "边界(Boundaries)", "desc": "不做什么，拒绝什么"},
        {"key": "output_format", "name": "输出规范(Output)", "desc": "默认输出格式和结构"},
        {"key": "examples", "name": "示例(Examples)", "desc": "提供行为示例或参考"},
    ],
}


def detect_missing_dimensions(prompt: str, mode: str = "user") -> List[Dict[str, Any]]:
    """检测提示词缺少哪些维度。"""
    dims = OPTIMIZATION_FRAMEWORK["dimensions"] if mode == "user" else OPTIMIZATION_FRAMEWORK["system_dimensions"]
    missing = []

    prompt_lower = prompt.lower()

    # User prompt 检测规则
    if mode == "user":
        # Role: 是否有角色指示
        role_patterns = [r"你是", r"作为", r"扮演", r"假设你", r"你是一个", r"act as", r"you are", r"role"]
        has_role = any(re.search(p, prompt_lower) for p in role_patterns)

        # Object: 是否指定了受众/对象
        object_patterns = [r"面向", r"针对", r"给.*的", r"目标.*读者", r"受众", r"用户", r"客户"]
        has_object = any(re.search(p, prompt_lower) for p in object_patterns)

        # Structure: 是否有结构要求
        structure_patterns = [r"第[一二三四五六七八九十]", r"\d+[.、]", r"步骤", r"首先.*然后", r"分.*部分", r"大纲", r"step", r"1\.", r"1、"]
        has_structure = any(re.search(p, prompt_lower) for p in structure_patterns)

        # Style: 是否指定了风格
        style_patterns = [r"风格", r"语气", r"口语", r"专业", r"轻松", r"严肃", r"幽默", r"学术", r"简洁", r"详细"]
        has_style = any(re.search(p, prompt_lower) for p in style_patterns)

        # Constraints: 是否有约束
        constraint_patterns = [r"不要", r"不能", r"避免", r"限制", r"字数", r"不超过", r"至少", r"必须", r"禁止"]
        has_constraints = any(re.search(p, prompt_lower) for p in constraint_patterns)

        # Output: 是否指定了输出格式
        output_patterns = [r"输出", r"返回", r"格式", r"以.*形式", r"json", r"markdown", r"表格", r"列表", r"输出为"]
        has_output = any(re.search(p, prompt_lower) for p in output_patterns)

        checks = [has_role, has_object, has_structure, has_style, has_constraints, has_output]
    else:
        # System prompt 检测规则
        identity_patterns = [r"你是", r"你是.*助手", r"你是.*专家", r"身份", r"角色"]
        has_identity = any(re.search(p, prompt_lower) for p in identity_patterns)

        expertise_patterns = [r"擅长", r"精通", r"专业", r"领域", r"知识", r"能力"]
        has_expertise = any(re.search(p, prompt_lower) for p in expertise_patterns)

        behavior_patterns = [r"回复.*风格", r"沟通", r"语气", r"态度", r"行为", r"响应方式"]
        has_behavior = any(re.search(p, prompt_lower) for p in behavior_patterns)

        boundary_patterns = [r"不要", r"不能", r"拒绝", r"不做", r"禁止", r"超出.*范围"]
        has_boundaries = any(re.search(p, prompt_lower) for p in boundary_patterns)

        output_patterns = [r"输出.*格式", r"回复.*格式", r"默认.*格式", r"使用.*格式"]
        has_output = any(re.search(p, prompt_lower) for p in output_patterns)

        example_patterns = [r"例如", r"示例", r"比如", r"参考", r"example"]
        has_examples = any(re.search(p, prompt_lower) for p in example_patterns)

        checks = [has_identity, has_expertise, has_behavior, has_boundaries, has_output, has_examples]

    for dim, has in zip(dims, checks):
        if not has:
            missing.append(dim)

    return missing


def score_prompt(prompt: str, mode: str = "user") -> Dict[str, Any]:
    """对提示词进行六维评分（1-10）。"""
    dims = OPTIMIZATION_FRAMEWORK["dimensions"] if mode == "user" else OPTIMIZATION_FRAMEWORK["system_dimensions"]
    missing = detect_missing_dimensions(prompt, mode)

    # 基础分
    length = len(prompt)
    word_count = len(prompt.split())
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', prompt))

    scores = {}
    missing_keys = {m["key"] for m in missing}

    for dim in dims:
        base = 3  # 基础分
        if dim["key"] not in missing_keys:
            base += 4  # 有这个维度 +4

        # 长度加分（太短扣分）
        if length > 200:
            base += 1
        if length > 500:
            base += 1
        if length < 20:
            base -= 1

        # 结构加分
        if dim["key"] in ("structure", "output") and re.search(r'\n', prompt):
            base += 1

        scores[dim["key"]] = max(1, min(10, base))

    total = sum(scores.values()) / len(scores)

    return {
        "scores": scores,
        "total": round(total, 1),
        "missing": missing,
        "length": length,
        "has_structure": "\n" in prompt,
    }


def generate_optimization_guide(prompt: str, mode: str = "user") -> Dict[str, Any]:
    """生成优化指导（不直接改写，而是告诉 agent 如何优化）。"""
    analysis = score_prompt(prompt, mode)
    missing = analysis["missing"]

    suggestions = []
    for m in missing:
        suggestions.append({
            "dimension": m["name"],
            "issue": f"缺少{m['name']}维度",
            "suggestion": m["desc"],
            "example": _get_example(m["key"], mode),
        })

    return {
        "original_score": analysis["total"],
        "missing_count": len(missing),
        "missing_dimensions": [m["name"] for m in missing],
        "suggestions": suggestions,
        "framework": OPTIMIZATION_FRAMEWORK["dimensions" if mode == "user" else "system_dimensions"],
    }


def _get_example(key: str, mode: str) -> str:
    """获取各维度的示例。"""
    examples = {
        "role": "你是一位资深科技自媒体编辑，擅长写公众号爆款文章",
        "object": "面向 25-40 岁的互联网从业者，有一定技术背景",
        "structure": "分三个部分：1) 用真实案例引入 2) 分析核心观点 3) 给出可执行建议",
        "style": "语言口语化但保持专业，避免学术腔，多用短句",
        "constraints": "字数 1500-2000 字，不要使用'总之'开头，避免 AI 味",
        "output": "输出为 Markdown 格式，包含标题、小标题和要点列表",
        "identity": "你是一个专业的代码审查助手",
        "expertise": "精通 Python/TypeScript/Go，熟悉设计模式和性能优化",
        "behavior": "回复简洁直接，先给结论再给理由，代码示例优先",
        "boundaries": "不做代码编写，只审查和建议；超出技术范围的问题礼貌拒绝",
        "output_format": "默认使用 Markdown，代码块标注语言，关键问题用 ⚠️ 标记",
        "examples": "例如：当发现未处理异常时，回复'⚠️ L3: 第42行 except 为空，建议添加日志记录'",
    }
    return examples.get(key, "")


def build_compare_framework() -> Dict[str, Any]:
    """返回 A/B 对比框架。"""
    return {
        "dimensions": [
            {"key": "completeness", "name": "完整性", "desc": "哪个覆盖了更多必要信息"},
            {"key": "clarity", "name": "清晰度", "desc": "哪个意图更明确、更少歧义"},
            {"key": "structure", "name": "结构性", "desc": "哪个组织更有层次和逻辑"},
            {"key": "controllability", "name": "可控性", "desc": "哪个输出更可预测、更稳定"},
            {"key": "reusability", "name": "可复用性", "desc": "哪个更容易迁移到类似场景"},
            {"key": "efficiency", "name": "效率性", "desc": "哪个用更少的字表达更明确的意图"},
        ],
        "method": (
            "对比方法：\n"
            "1. 分别对 A/B 进行六维评分\n"
            "2. 逐维度对比差异\n"
            "3. 识别各自优势和不足\n"
            "4. 给出综合判断和改进建议"
        ),
    }


# ─────────────────────────────────────────────────────
# 场景模板库
# ─────────────────────────────────────────────────────

TEMPLATES = {
    "写作": {
        "user": (
            "## 任务\n请写一篇关于 [主题] 的文章。\n\n"
            "## 要求\n"
            "- 面向 [目标读者]\n"
            "- 风格：[口语化/专业/轻松]\n"
            "- 结构：[开头用真实案例引入 → 分析核心观点 → 给出可执行建议]\n"
            "- 字数：[1500-2000字]\n"
            "- 禁止：[不要用'总之'开头，避免 AI 味]\n\n"
            "## 输出格式\nMarkdown 格式，包含标题和小标题"
        ),
        "system": (
            "你是一位资深自媒体编辑，擅长写公众号和博客文章。\n"
            "你的写作风格：口语化但专业，善用短句和案例，结构清晰。\n"
            "你不使用模板化表达（如'在这个快速发展的时代'）。\n"
            "你的文章特点：开头有吸引力、观点有深度、结尾可执行。"
        ),
    },
    "代码": {
        "user": (
            "## 任务\n[描述编程任务]\n\n"
            "## 技术栈\n- 语言：[Python/TypeScript/Go 等]\n"
            "- 框架：[如有]\n\n"
            "## 要求\n"
            "- 遵循 [代码规范]\n"
            "- 包含错误处理\n"
            "- 添加必要注释\n"
            "- 编写对应测试\n\n"
            "## 输出格式\n完整可运行的代码，包含使用示例"
        ),
        "system": (
            "你是一个资深软件工程师，精通多种编程语言和设计模式。\n"
            "你写的代码：简洁、可读、有良好的错误处理和测试覆盖。\n"
            "你优先使用标准库，避免不必要的依赖。\n"
            "你的回复格式：先给出方案思路，再给出代码实现。"
        ),
    },
    "翻译": {
        "user": (
            "## 任务\n将以下内容翻译为 [目标语言]。\n\n"
            "## 要求\n"
            "- 风格：[正式/口语化/技术文档]\n"
            "- 术语处理：[保留原文/翻译/加注]\n"
            "- 保持原文格式\n\n"
            "## 原文\n[粘贴原文]"
        ),
        "system": (
            "你是一个专业翻译，精通 [源语言] 和 [目标语言]。\n"
            "你的翻译原则：信达雅，优先准确，兼顾流畅。\n"
            "专业术语处理：首次出现时标注原文。\n"
            "你不添加原文没有的内容。"
        ),
    },
    "分析": {
        "user": (
            "## 任务\n分析以下 [数据/问题/文本]。\n\n"
            "## 分析角度\n"
            "- [角度1]\n"
            "- [角度2]\n"
            "- [角度3]\n\n"
            "## 要求\n"
            "- 用数据支撑观点\n"
            "- 给出可行建议\n"
            "- 标注置信度\n\n"
            "## 输出格式\n结构化报告，包含摘要、详细分析、结论和建议"
        ),
        "system": (
            "你是一个数据分析专家，擅长从复杂信息中提取关键洞察。\n"
            "你的分析方法：先总结核心发现，再展开详细论证。\n"
            "你区分事实和推测，对不确定的结论标注置信度。\n"
            "你的建议总是具体可执行的。"
        ),
    },
    "通用": {
        "user": (
            "## 角色\n你是一个 [角色描述]。\n\n"
            "## 任务\n[具体任务描述]\n\n"
            "## 要求\n"
            "- [要求1]\n"
            "- [要求2]\n"
            "- [要求3]\n\n"
            "## 约束\n"
            "- [不要做什么]\n"
            "- [限制条件]\n\n"
            "## 输出格式\n[期望的输出格式]"
        ),
        "system": (
            "你是一个 [身份描述]。\n"
            "你的专业领域：[领域]。\n"
            "你的沟通风格：[风格]。\n"
            "你的行为准则：[准则]。\n"
            "你不做的事情：[边界]。\n"
            "你的输出格式：[默认格式]。"
        ),
    },
}
