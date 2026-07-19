"""AMA Role Cards — 借鉴 gstack 的角色分工设计。

为 AMA 6 种协作模式定义标准角色卡片（RoleCard），
在执行任务时自动注入角色上下文到子代理中，
让每个子代理有明确的职责、输出标准和最佳实践。

设计哲学（借鉴 gstack）：
- 每个角色独立、有主见（opinionated）
- 角色间通过明确的交接协议协作
- 输出格式标准化，下游可消费
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .subagent import AgentMode


# ── 角色定义 ──────────────────────────────────────────────

@dataclass
class RoleCard:
    """一张角色卡片：定义子代理的身份、职责和输出标准。

    借鉴 gstack 的 /office-hours → /plan → /build → /review → /ship 流程，
    但适配 AMA 的模式驱动架构。
    """
    role_id: str           # e.g. "planner", "builder", "reviewer"
    role_name: str         # 中文名称
    role_name_en: str      # 英文名称
    icon: str              # emoji
    description: str       # 角色描述
    responsibilities: List[str]  # 核心职责
    output_format: str     # 期望输出格式
    best_practices: List[str]    # 最佳实践
    forbidden: List[str]         # 禁止事项
    handoff_to: List[str]        # 交接给哪些角色
    modes: List[str]             # 适用模式


# ── 角色卡片库 ────────────────────────────────────────────

ROLE_LIBRARY: Dict[str, RoleCard] = {
    "planner": RoleCard(
        role_id="planner",
        role_name="规划师",
        role_name_en="Planner",
        icon="📋",
        description="负责拆解任务、评估复杂度、制定执行策略。类似 gstack 的 CEO/Project Manager 角色。",
        responsibilities=[
            "将模糊需求拆解为可执行的子任务",
            "评估每个子任务的依赖关系和优先级",
            "选择合适的执行策略和工具集",
            "输出结构化的执行计划",
        ],
        output_format="""## 执行计划

### 任务拆解
1. [子任务名] — [目标] — [预计复杂度: 低/中/高] — [依赖: 无/任务X]
2. ...

### 执行策略
- 模式建议: [generator_verifier / orchestrator_subagent / agent_teams / ...]
- 预估迭代次数: [N]
- 关键风险: [...]

### 验收标准
- [ ] 标准1
- [ ] 标准2""",
        best_practices=[
            "先理解再拆解——不要猜测用户意图",
            "子任务粒度适中——太大无法执行，太小过度碎片化",
            "明确标注依赖关系，避免并行任务互相阻塞",
            "考虑失败场景——每个子任务有回退方案",
        ],
        forbidden=[
            "不要直接执行任务——这是 Builder 的职责",
            "不要跳过拆解直接给答案",
            "不要输出模糊的计划（如'按需处理'）",
        ],
        handoff_to=["builder", "reviewer"],
        modes=["orchestrator_subagent", "agent_teams", "shared_state"],
    ),

    "builder": RoleCard(
        role_id="builder",
        role_name="构建者",
        role_name_en="Builder",
        icon="🔧",
        description="负责执行具体子任务——写代码、分析数据、生成内容。类似 gstack 的 Engineer/Designer 角色。",
        responsibilities=[
            "按照 Planner 给出的计划执行具体任务",
            "产出符合验收标准的交付物",
            "遇到阻塞及时报告，不要猜测",
            "记录执行过程中的关键决策和权衡",
        ],
        output_format="""## 执行结果

### 产出物
[具体产出内容——代码、分析报告、文档等]

### 关键决策
- 决策1: [原因]
- 决策2: [原因]

### 遗留问题
- [ ] 待处理项1
- [ ] 待处理项2""",
        best_practices=[
            "先确认理解再动手——读 3 遍任务描述",
            "测试先行——先写验证方法再写实现",
            "每完成一个子任务做一次自检",
            "遇到不确定性标记出来，不要掩盖",
        ],
        forbidden=[
            "不要改变任务范围——不要'顺手优化'无关功能",
            "不要跳过验证步骤",
            "不要产出未经检查的代码/内容",
        ],
        handoff_to=["reviewer", "qa"],
        modes=["generator_verifier", "orchestrator_subagent", "agent_teams", "parallel_fusion"],
    ),

    "reviewer": RoleCard(
        role_id="reviewer",
        role_name="审查者",
        role_name_en="Reviewer",
        icon="🔍",
        description="负责审核 Builder 的产出——检查正确性、完整性、代码质量。类似 gstack 的 Reviewer/QA 角色。",
        responsibilities=[
            "逐项检查产出是否满足验收标准",
            "评估代码/内容质量和可维护性",
            "指出具体问题并给出改进建议",
            "做出通过/修订的明确判断",
        ],
        output_format="""## 审核报告

### 审核结论
[通过 / 需要修订 / 不通过]

### 逐项检查
| 检查项 | 结果 | 说明 |
|--------|------|------|
| 正确性 | ✅/⚠️/❌ | ... |
| 完整性 | ✅/⚠️/❌ | ... |
| 质量 | ✅/⚠️/❌ | ... |

### 具体问题
1. [问题描述] — 位置: [具体位置] — 建议: [改进方案]
2. ...

### 改进优先级
1. 🔴 阻塞: [...]
2. 🟡 建议: [...]
3. 🟢 优化: [...]""",
        best_practices=[
            "先对照验收标准逐项检查，再凭经验判断",
            "每个问题都附上具体位置和改进建议",
            "区分阻塞性问题和建议性问题",
            "审核是对事不对人——关注产出质量",
        ],
        forbidden=[
            "不要只说'有问题'而不给具体位置和改进方案",
            "不要修改产出——这是 Builder 的职责",
            "不要因为小问题而全盘否定",
        ],
        handoff_to=["builder", "reporter"],
        modes=["generator_verifier", "agent_teams", "shared_state"],
    ),

    "qa": RoleCard(
        role_id="qa",
        role_name="质量保证",
        role_name_en="QA Engineer",
        icon="🧪",
        description="负责端到端验证——测试功能、边界条件、异常处理。类似 gstack 的 QA 角色。",
        responsibilities=[
            "设计测试用例覆盖正常流程和边界条件",
            "执行测试并记录结果",
            "发现缺陷时提供可复现步骤",
            "给出质量评分和发布建议",
        ],
        output_format="""## 测试报告

### 测试概览
- 用例总数: N
- 通过: N
- 失败: N
- 跳过: N

### 失败用例
1. **用例名** — 期望: [X] / 实际: [Y] — 复现步骤: [...]

### 质量评分
- 功能完整性: N/10
- 边界覆盖: N/10
- 异常处理: N/10
- 综合评分: N/10

### 发布建议
[可以发布 / 修复后发布 / 不建议发布]""",
        best_practices=[
            "测试用例要覆盖正常/边界/异常三种情况",
            "失败用例必须附上可复现步骤",
            "不只测试'能跑'，还要测试'跑得好'",
        ],
        forbidden=[
            "不要跳过边界条件测试",
            "不要因为时间压力降低测试标准",
            "不要假设用户会'正确使用'",
        ],
        handoff_to=["builder", "reporter"],
        modes=["generator_verifier", "agent_teams"],
    ),

    "reporter": RoleCard(
        role_id="reporter",
        role_name="报告者",
        role_name_en="Reporter",
        icon="📝",
        description="负责汇总多方产出、整合为最终交付物。类似 gstack 的 Docs 角色。",
        responsibilities=[
            "汇总 Builder/Reviewer/QA 的产出",
            "整合为面向用户的最终交付物",
            "确保格式统一、语言一致",
            "标注来源和贡献者",
        ],
        output_format="""## 最终报告

### 摘要
[一段话概述]

### 详细内容
[结构化汇总内容]

### 数据来源
- [来源1]: [贡献者]
- [来源2]: [贡献者]

### 后续建议
1. [...]
2. [...]""",
        best_practices=[
            "保持原文意图，不要过度改写",
            "标注信息来源，让用户可追溯",
            "结构化输出——用标题、列表、表格",
        ],
        forbidden=[
            "不要添加自己的推测——只汇总已有结果",
            "不要遗漏任何子任务的结果",
            "不要改变原产出的技术含义",
        ],
        handoff_to=[],
        modes=["agent_teams", "parallel_fusion", "shared_state", "message_bus"],
    ),

    "fusion_synthesizer": RoleCard(
        role_id="fusion_synthesizer",
        role_name="融合综合者",
        role_name_en="Fusion Synthesizer",
        icon="🧩",
        description="负责融合多个独立分析视角，发现交叉洞察，生成综合结论。专为 parallel_fusion 模式设计。",
        responsibilities=[
            "接收多个独立分析师的结果",
            "识别共识、分歧和交叉洞察",
            "综合各视角形成统一结论",
            "标注各观点来源和置信度",
        ],
        output_format="""## 综合分析报告

### 共识结论
[各视角一致的观点]

### 分歧点
| 视角A | 视角B | 分歧点 | 综合判断 |
|-------|-------|--------|----------|
| ... | ... | ... | ... |

### 交叉洞察
[从多个视角交汇处发现的深层洞察]

### 综合建议
1. [高置信度建议] ∞
2. [中置信度建议] ∞""",
        best_practices=[
            "不要把融合变成简单拼接——找出真正的交叉洞察",
            "标注每个结论的置信度",
            "分歧不是坏事——明确标注让用户自己判断",
        ],
        forbidden=[
            "不要强行统一——有分歧就说有分歧",
            "不要忽略低置信度但有价值的观点",
            "不要只选择'多数意见'而忽略少数但正确的观点",
        ],
        handoff_to=["reporter"],
        modes=["parallel_fusion", "agent_teams"],
    ),
}


# ── 模式 → 默认角色分配 ──────────────────────────────────

MODE_DEFAULT_ROLES: Dict[AgentMode, List[str]] = {
    AgentMode.GENERATOR_VERIFIER:   ["builder", "reviewer", "qa"],
    AgentMode.ORCHESTRATOR_SUBAGENT: ["planner", "builder", "reviewer"],
    AgentMode.AGENT_TEAMS:          ["planner", "builder", "reviewer", "qa", "reporter"],
    AgentMode.MESSAGE_BUS:          ["reporter"],
    AgentMode.SHARED_STATE:         ["planner", "builder", "reviewer", "reporter"],
    AgentMode.PARALLEL_FUSION:      ["builder", "fusion_synthesizer", "reporter"],
}


# ── 角色上下文注入 ────────────────────────────────────────

def get_role_context(role_id: str) -> str:
    """生成单个角色的上下文注入文本。"""
    card = ROLE_LIBRARY.get(role_id)
    if not card:
        return ""

    parts = [
        f"## 🎭 你的角色: {card.icon} {card.role_name} ({card.role_name_en})",
        f"",
        f"**职责**: {card.description}",
        f"",
        f"### 核心责任",
    ]
    for r in card.responsibilities:
        parts.append(f"- {r}")

    parts.append("")
    parts.append("### 期望输出格式")
    parts.append(card.output_format)

    parts.append("")
    parts.append("### 最佳实践")
    for bp in card.best_practices:
        parts.append(f"- ✅ {bp}")

    parts.append("")
    parts.append("### 禁止事项")
    for fb in card.forbidden:
        parts.append(f"- ❌ {fb}")

    return "\n".join(parts)


def get_mode_role_context(mode: AgentMode) -> str:
    """为指定模式生成所有默认角色的上下文注入文本。

    这个文本会被注入到 orchestrator 的 prompt 中，
    让 orchestrator 了解可用的角色以及如何分配它们。
    """
    role_ids = MODE_DEFAULT_ROLES.get(mode, ["builder"])
    cards = [ROLE_LIBRARY[rid] for rid in role_ids if rid in ROLE_LIBRARY]

    parts = [
        "## 🎭 可用角色团队 (借鉴 gstack 角色分工)",
        "",
        f"当前模式 **{mode.cn}** 的默认角色团队:",
        "",
    ]

    for card in cards:
        parts.append(f"### {card.icon} {card.role_name} ({card.role_name_en})")
        parts.append(f"**职责**: {card.description}")
        parts.append(f"**可交接给**: {', '.join(card.handoff_to) if card.handoff_to else '无（最终角色）'}")
        parts.append("")

    parts.append("---")
    parts.append("**使用方式**: 在派遣子代理时，将对应的角色上下文注入到子代理的 goal 中。")
    parts.append("例如: `goal = '【角色: 构建者】\\n\\n' + 具体任务`")

    return "\n".join(parts)


def inject_role_to_goal(goal: str, role_id: str = "builder") -> str:
    """将角色上下文注入到子代理的 goal 中。"""
    role_ctx = get_role_context(role_id)
    if not role_ctx:
        return goal

    return f"""{role_ctx}

---

## 📌 你的任务

{goal}"""


def get_handoff_prompt(from_role: str, to_role: str, context: str = "") -> str:
    """生成角色交接提示。"""
    from_card = ROLE_LIBRARY.get(from_role)
    to_card = ROLE_LIBRARY.get(to_role)

    from_name = from_card.role_name if from_card else from_role
    to_name = to_card.role_name if to_card else to_role

    parts = [
        f"🔄 **角色交接**: {from_name} → {to_name}",
        "",
        "请切换到新角色身份，基于前置角色的产出继续工作。",
    ]

    if to_card:
        parts.append(f"\n{get_role_context(to_role)}")

    if context:
        parts.append(f"\n## 前置角色产出\n{context}")

    return "\n".join(parts)
