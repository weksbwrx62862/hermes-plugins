"""AMA Workflow Templates — 借鉴 gstack 的结构化流程设计。

为常见任务类型预定义多阶段工作流（WorkflowTemplate），
每个阶段有明确的角色分配、阶段目标和交付物标准。

设计哲学（借鉴 gstack）：
- 结构化流程：每个阶段有明确的输入/输出
- 角色分工：不同阶段分配不同角色
- 交付物驱动：每个阶段有验收标准
- 可组合：子流程可以嵌套
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .subagent import AgentMode


@dataclass
class WorkflowStage:
    """工作流的一个阶段。"""
    id: str                  # e.g. "plan", "build", "review"
    name: str                # 中文名
    role_id: str             # 主角色（从 rolecards.ROLE_LIBRARY）
    description: str         # 阶段目标
    goal_template: str       # 给子代理的 goal 模板（用 {task} {context} 占位）
    success_criteria: List[str]  # 验收标准
    next_stage: Optional[str] = None  # 下一阶段 ID
    fallback_stage: Optional[str] = None  # 失败时回退到哪个阶段


@dataclass
class WorkflowTemplate:
    """一个完整工作流模板。"""
    id: str                          # e.g. "software_dev"
    name: str                        # 中文名
    task_types: List[str]            # 匹配的任务类型
    description: str                 # 说明
    stages: List[WorkflowStage]      # 阶段顺序列表
    default_mode: AgentMode          # 默认执行模式
    max_iterations: int = 5          # 默认最大迭代次数


# ── 工作流模板库 ──────────────────────────────────────────

WORKFLOW_LIBRARY: Dict[str, WorkflowTemplate] = {
    "software_dev": WorkflowTemplate(
        id="software_dev",
        name="软件开发",
        task_types=["software_dev", "code_generation"],
        description="完整的软件开发流程：需求分析→架构设计→编码实现→代码审查→测试验证。",
        default_mode=AgentMode.ORCHESTRATOR_SUBAGENT,
        max_iterations=5,
        stages=[
            WorkflowStage(
                id="plan",
                name="需求分析",
                role_id="planner",
                description="分析需求、拆解任务、明确验收标准",
                goal_template="""## 【需求分析阶段】

请分析以下任务需求，输出：
1. 功能拆解清单（逐项列出需要实现的功能）
2. 技术架构方案（技术栈、模块划分、数据流）
3. 验收标准（每项功能的可验证标准）

### 任务
{task}

### 上下文
{context}

### 输出格式
请使用结构化 Markdown，包含上述三个部分。""",
                success_criteria=["功能清单完整", "技术方案可行", "验收标准可衡量"],
                next_stage="build",
            ),
            WorkflowStage(
                id="build",
                name="编码实现",
                role_id="builder",
                description="按照设计方案编码实现",
                goal_template="""## 【编码实现阶段】

请基于上一阶段的方案，完成编码实现。
- 严格遵循验收标准
- 每个功能附带必要的注释
- 含错误处理和边界情况

### 前置方案
{context}

### 任务
{task}

### 要求
- 代码规范：遵循 PEP8/PEP257
- 测试覆盖：核心逻辑有单元测试
- 错误处理：输入验证 + 异常捕获""",
                success_criteria=["代码编译/语法正确", "核心功能完整", "有测试覆盖"],
                next_stage="review",
                fallback_stage="plan",
            ),
            WorkflowStage(
                id="review",
                name="代码审查",
                role_id="reviewer",
                description="审查代码质量和功能正确性",
                goal_template="""## 【代码审查阶段】

请逐项审查代码：
1. 功能完整性——是否满足所有需求
2. 代码质量——可读性、可维护性、性能
3. 安全性——输入验证、数据泄露
4. 测试覆盖——是否覆盖核心路径和边界

### 原始需求
{task}

### 待审查代码
{context}

### 输出格式
请使用结构化审核报告格式：
- 通过/需修订/不通过
- 逐项检查表
- 具体问题列表（含位置和建议）""",
                success_criteria=["功能100%覆盖", "代码风格达标", "无安全问题"],
                next_stage=None,  # 终审通过则结束
                fallback_stage="build",
            ),
        ],
    ),

    "debugging": WorkflowTemplate(
        id="debugging",
        name="调试修复",
        task_types=["debugging", "fact_checking"],
        description="结构化调试流程：复现问题→定位根因→修复实现→验证通过。",
        default_mode=AgentMode.GENERATOR_VERIFIER,
        max_iterations=4,
        stages=[
            WorkflowStage(
                id="diagnose",
                name="问题诊断",
                role_id="reviewer",
                description="分析错误信息，定位根因",
                goal_template="""## 【问题诊断阶段】

请分析以下问题，输出：
1. 问题复现步骤
2. 根因分析（代码/逻辑/配置/环境）
3. 修复方案

### 问题描述
{task}

### 错误信息/上下文
{context}

### 要求
- 先尝试复现，再分析根因
- 区分直接原因和根本原因
- 修复方案要有优先级""",
                success_criteria=["问题可复现", "根因定位准确", "修复方案可行"],
                next_stage="fix",
            ),
            WorkflowStage(
                id="fix",
                name="修复实现",
                role_id="builder",
                description="按诊断方案实施修复",
                goal_template="""## 【修复实现阶段】

请基于诊断结果实施修复：

### 诊断结论
{context}

### 修复要求
1. 最小改动原则——只改必须改的地方
2. 不引入新问题——修改后检查相关功能
3. 添加回归测试防止再犯

### 任务
{task}""",
                success_criteria=["修复方案已实施", "不引入新问题", "有回归测试"],
                next_stage="verify",
                fallback_stage="diagnose",
            ),
            WorkflowStage(
                id="verify",
                name="验证确认",
                role_id="qa",
                description="验证修复有效且无回归",
                goal_template="""## 【验证确认阶段】

请验证修复效果：
1. 确认原问题不再复现
2. 检查相关功能无回归
3. 运行测试全部通过

### 原始问题
{task}

### 修复内容
{context}

### 输出
- 通过/不通过
- 验证详情""",
                success_criteria=["原问题修复", "无回归", "测试通过"],
                next_stage=None,
                fallback_stage="fix",
            ),
        ],
    ),

    "research": WorkflowTemplate(
        id="research",
        name="研究调研",
        task_types=["research", "analysis"],
        description="结构化研究流程：信息收集→交叉验证→综合分析→输出报告。推荐并行融合模式。",
        default_mode=AgentMode.PARALLEL_FUSION,
        max_iterations=3,
        stages=[
            WorkflowStage(
                id="gather",
                name="信息收集",
                role_id="builder",
                description="多角度收集信息，确保覆盖全面",
                goal_template="""## 【信息收集阶段】

请从多个角度收集相关资料：

### 研究主题
{task}

### 上下文
{context}

### 要求
1. 多渠道信息来源
2. 区分一手/二手资料
3. 记录每个信息的来源和时间

### 输出
结构化信息摘要，含来源标注""",
                success_criteria=["至少3个独立来源", "信息相互印证", "来源可追溯"],
                next_stage="analyze",
            ),
            WorkflowStage(
                id="analyze",
                name="分析综合",
                role_id="fusion_synthesizer",
                description="交叉分析，识别共识和分歧",
                goal_template="""## 【分析综合阶段】

请综合各视角的信息：

### 研究主题
{task}

### 收集到的信息
{context}

### 分析维度
1. 共识点——各方一致认同的结论
2. 分歧点——观点不一之处
3. 发现缺口——信息不足/矛盾之处
4. 深层洞察——信息交叉处的新发现

### 输出
综合分析报告""",
                success_criteria=["共识和分歧都标注", "有深度分析", "结论有置信度标注"],
                next_stage="report",
            ),
            WorkflowStage(
                id="report",
                name="输出报告",
                role_id="reporter",
                description="整合为最终研究报告",
                goal_template="""## 【报告输出阶段】

请将分析结果整理为最终报告：

### 任务
{task}

### 分析结果
{context}

### 输出要求
- 结构化 Markdown
- 含目录和摘要
- 关键发现加粗标注
- 末尾附来源清单""",
                success_criteria=["结构清晰", "关键发现突出", "来源完整"],
                next_stage=None,
            ),
        ],
    ),

    "writing": WorkflowTemplate(
        id="writing",
        name="内容创作",
        task_types=["creative", "writing"],
        description="结构化创作流程：构思大纲→撰写初稿→审核润色→定稿输出。",
        default_mode=AgentMode.GENERATOR_VERIFIER,
        max_iterations=4,
        stages=[
            WorkflowStage(
                id="outline",
                name="大纲策划",
                role_id="planner",
                description="确定结构框架和核心论点",
                goal_template="""## 【大纲策划阶段】

请为以下创作任务设计大纲：

### 主题
{task}

### 上下文/约束
{context}

### 输出要求
1. 核心论点（1-3个）
2. 结构框架（章节划分）
3. 关键信息点（每节要点）
4. 目标读者定位""",
                success_criteria=["结构合理", "核心论点明确", "信息层次清晰"],
                next_stage="draft",
            ),
            WorkflowStage(
                id="draft",
                name="初稿撰写",
                role_id="builder",
                description="根据大纲撰写初稿",
                goal_template="""## 【初稿撰写阶段】

请根据大纲撰写内容：

### 大纲
{context}

### 任务
{task}

### 要求
1. 语言流畅自然
2. 避免 AI 味的表达
3. 有具体例子和数据支撑论点
4. 保持一致的叙事风格""",
                success_criteria=["内容完整覆盖大纲", "语言自然", "有具体例证"],
                next_stage="polish",
                fallback_stage="outline",
            ),
            WorkflowStage(
                id="polish",
                name="审核润色",
                role_id="reviewer",
                description="审核内容质量，优化表达",
                goal_template="""## 【审核润色阶段】

请审核并润色以下内容：

### 创作要求
{task}

### 待审核文本
{context}

### 审核要点
1. 逻辑是否通顺
2. 表达是否自然（去除 AI 味）
3. 事实是否准确
4. 风格是否一致
5. 结构是否合理

### 输出
- 修改版文本
- 修改说明清单""",
                success_criteria=["表达自然", "逻辑连贯", "风格一致"],
                next_stage=None,
                fallback_stage="draft",
            ),
        ],
    ),

    "data_analysis": WorkflowTemplate(
        id="data_analysis",
        name="数据分析",
        task_types=["analysis", "data_analysis"],
        description="结构化数据分析流程：数据加载→清洗→分析→可视化→报告。",
        default_mode=AgentMode.ORCHESTRATOR_SUBAGENT,
        max_iterations=4,
        stages=[
            WorkflowStage(
                id="prepare",
                name="数据准备",
                role_id="builder",
                description="加载数据、清洗预处理、探查性分析",
                goal_template="""## 【数据准备阶段】

请准备数据进行分析：

### 分析任务
{task}

### 数据源
{context}

### 步骤
1. 加载数据（检查格式和编码）
2. 数据清洗（缺失值、异常值、重复值）
3. 探查性分析（统计摘要、分布情况）
4. 输出清洗后的数据集

### 输出
- 数据质量报告
- 清洗数据集""",
                success_criteria=["数据加载无误", "清洗策略合理", "质量报告完整"],
                next_stage="analyze",
            ),
            WorkflowStage(
                id="analyze",
                name="深度分析",
                role_id="builder",
                description="执行核心分析，发现模式和洞察",
                goal_template="""## 【深度分析阶段】

请执行核心分析：

### 分析任务
{task}

### 已清洗数据
{context}

### 分析方法
1. 按任务需求选择合适的分析方法
2. 统计分析 + 可视化相结合
3. 关键指标计算
4. 模式/趋势/异常检测""",
                success_criteria=["方法选择合理", "分析深度充分", "发现具业务价值"],
                next_stage="report",
            ),
            WorkflowStage(
                id="report",
                name="报告输出",
                role_id="reporter",
                description="整合分析结果，输出最终报告",
                goal_template="""## 【报告输出阶段】

请整合分析结果为最终报告：

### 分析任务
{task}

### 分析结果
{context}

### 输出要求
- 执行摘要
- 方法说明
- 关键发现（图表+解读）
- 结论与建议""",
                success_criteria=["结构清晰", "关键发现突出", "有 actionable 建议"],
                next_stage=None,
            ),
        ],
    ),
}


# ── 工作流匹配与使用 ─────────────────────────────────────

def match_workflow(task_type: str) -> Optional[WorkflowTemplate]:
    """根据任务类型匹配合适的工作流模板。"""
    for wf_id, wf in WORKFLOW_LIBRARY.items():
        if task_type in wf.task_types:
            return wf
    return None


def get_workflow_stage_goal(
    stage: WorkflowStage,
    task: str,
    context: str = "",
    prev_result: str = "",
) -> str:
    """生成工作流阶段的 goal 文本。"""
    goal = stage.goal_template.format(
        task=task,
        context=context or prev_result or "无",
    )
    return goal


def list_workflows() -> List[Dict]:
    """列出所有可用工作流模板。"""
    return [
        {
            "id": wf.id,
            "name": wf.name,
            "task_types": wf.task_types,
            "description": wf.description,
            "stages": [s.name for s in wf.stages],
            "default_mode": wf.default_mode.value,
        }
        for wf in WORKFLOW_LIBRARY.values()
    ]
