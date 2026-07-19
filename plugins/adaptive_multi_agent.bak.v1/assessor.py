from __future__ import annotations

# 本模块负责任务复杂度评估与需求澄清。
# 包含 TaskComplexityAssessor（规则引擎评分）和 RequirementClarifier（多轮澄清+LLM 评分）
# 以及相关的提示词模板与特征常量。

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .subagent import AgentMode

logger = logging.getLogger(__name__)

CLARIFY_PROMPT_TEMPLATE = """你是一个任务需求分析专家。用户给出了一个任务描述，你需要判断需求是否足够明确，如果不明确则提出澄清问题。

## 评分标准（7个特征维度）

| 特征 | 说明 | 加分 | 示例关键词 |
|------|------|------|-----------|
| has_explicit_verification | 是否需要验证/检查结果 | +0.5 | 验证、检查、测试、verify、validate |
| needs_parallelism | 是否需要并行处理 | +2.0 | 并行、同时、批量、parallel、concurrent |
| has_roles | 是否涉及多角色分工 | +1.5 | 角色、分工、团队、role、team |
| is_event_driven | 是否是事件驱动 | +1.0 | 事件、监控、实时、event、monitor |
| needs_collaboration | 是否需要协作 | +2.0 | 协作、共同、collaborate、together |
| iterative_potential | 是否需要迭代改进 | +1.0 | 迭代、改进、循环、iterate、refine |
| requires_shared_knowledge | 是否需要共享知识 | +1.0 | 共享、知识库、整合、shared、integrate |

评分公式：基础分 1.0 + 各特征加分（上限 10.0）
任务类型：code_generation / research / fact_checking / software_dev / event_driven / default
模式映射：score≤3→generator_verifier, 3<score≤6→orchestrator_subagent, 6<score≤8→agent_teams, score>8→shared_state/message_bus

## 你的任务

1. 分析用户描述中哪些特征维度信息缺失或不明确
2. 如果需求已充分明确（能确定大部分特征维度），设置 needs_clarification=false
3. 如果需求不明确，生成 2-4 个简洁的澄清问题

## 输出格式（严格 JSON）

```json
{{{{
  "needs_clarification": true/false,
  "questions": ["问题1", "问题2"],
  "extracted_features": {{{{
    "has_explicit_verification": false,
    "needs_parallelism": false,
    "has_roles": false,
    "is_event_driven": false,
    "needs_collaboration": false,
    "iterative_potential": false,
    "requires_shared_knowledge": false
  }}}},
  "clarified_task": "补充完善后的任务描述",
  "reasoning": "简要说明你的判断依据"
}}}}
```

## 用户任务描述
{task_description}

## 额外上下文
{context}

## 历史问答
{previous_qa}"""

SCORE_PROMPT_TEMPLATE = """你是一个任务复杂度评估专家。请用 Rubric 维度评分法对任务进行结构化评分。

## 5 维度 Rubric 评分（每维度 1-5 分）

| 维度 | 1分 | 3分 | 5分 |
|------|-----|-----|-----|
| steps | 单步操作，无依赖 | 3-5步，有顺序依赖 | 10+步，有分支和并行 |
| domain | 纯文本/对话 | 单领域（如纯后端） | 跨领域全栈（前端+后端+DB+部署） |
| verification | 无需验证 | 需要测试/验证 | 多轮验证+回归测试 |
| collaboration | 单人可完成 | 需要2个角色配合 | 需要多角色并行协作 |
| uncertainty | 需求完全明确 | 部分模糊需澄清 | 高度不确定需探索 |

## 评分公式

dim_avg = (steps + domain + verification + collaboration + uncertainty) / 5
complexity_score = min(dim_avg * 2 + 0.5, 10.0)  # 缩放到 [1, 10]

## 特征判断（基于任务描述推断）

从以下特征中，标记为 true 的：
- has_explicit_verification: 需要验证结果
- needs_parallelism: 需要并行处理
- has_roles: 需要多角色
- is_event_driven: 事件驱动
- needs_collaboration: 需要协作
- iterative_potential: 需要迭代
- requires_shared_knowledge: 需要共享知识

## 任务描述（经澄清后）
{clarified_task}

## 额外上下文
{context}

## 澄清历史
{clarification_history}

## 输出格式（严格 JSON）
```json
{{
  "rubric": {{
    "steps": 3,
    "domain": 3,
    "verification": 2,
    "collaboration": 1,
    "uncertainty": 2
  }},
  "complexity_score": 5.5,
  "task_type": "code_generation",
  "features": {{
    "has_explicit_verification": true,
    "needs_parallelism": false,
    "has_roles": false,
    "is_event_driven": false,
    "needs_collaboration": false,
    "iterative_potential": false,
    "requires_shared_knowledge": false
  }},
  "reasoning": "50字以内的评分依据"
}}
```"""""

LLM_REFINE_PROMPT_TEMPLATE = """你是一个任务复杂度评分专家。规则引擎给出的初步评分为 {rule_score} 分，请你基于语义理解重新评估任务复杂度。

## 评分体系（基础分 1.0，上限 10.0）

### 一、7个显性特征维度

| 特征 | 说明 | 加分 |
|------|------|------|
| has_explicit_verification | 是否需要验证/检查结果 | +0.5 |
| needs_parallelism | 是否需要并行处理 | +2.0 |
| has_roles | 是否涉及多角色分工 | +1.5 |
| is_event_driven | 是否是事件驱动 | +1.0 |
| needs_collaboration | 是否需要协作 | +2.0 |
| iterative_potential | 是否需要迭代改进 | +1.0 |
| requires_shared_knowledge | 是否需要共享知识 | +1.0 |

### 二、隐性复杂度信号（规则引擎可能漏判，需你特别关注）

| 信号类别 | 说明 | 加分 | 示例 |
|----------|------|------|------|
| 领域复杂度 | 涉及安全/认证/加密/数据库/分布式/并发/微服务/API/算法/架构 | +0.5~1.0 | 设计认证系统、加密模块、数据库迁移 |
| 输出复杂度 | 要求报告/文档/分析/对比/方案/评估/选型 | +1.0 | 技术选型报告、竞品对比 |
| 范围信号 | 完整系统/全栈/端到端/生产级/平台级 | +1.0~1.5 | 完整电商系统、生产级部署 |
| 多组件 | 需要同时处理多个子任务或组件 | +1.0 | 同时处理认证+存储+API |
| 子任务数 | 描述中列出 3+ 个明确子项 | +1.0~1.5 | 包括注册、登录、权限、Token刷新 |

### 评分公式
最终分 = min(1.0 + 显性特征加分 + 隐性信号加分, 10.0)

### 模式映射
- score <= 3: generator_verifier
- 3 < score <= 6: orchestrator_subagent
- 6 < score <= 8: agent_teams
- score > 8: shared_state / message_bus

## 任务描述
{task}

## 额外上下文
{context}

## 输出格式（严格 JSON，不要包含其他内容）

```json
{{
  "complexity_score": 5.5,
  "task_type": "code_generation",
  "features": {{
    "has_explicit_verification": true,
    "needs_parallelism": false,
    "has_roles": false,
    "is_event_driven": false,
    "needs_collaboration": false,
    "iterative_potential": false,
    "requires_shared_knowledge": false
  }},
  "recommended_mode": "orchestrator_subagent",
  "reasoning": "50字以内的评分依据"
}}
```"""

class TaskComplexityAssessor:

    TASK_PATTERNS = {
        "code_generation": [
            "代码", "生成", "函数", "class", "def", "写代码",
            "code", "generate", "function", "implement", "write code",
        ],
        "fact_checking": [
            "验证", "检查", "事实", "正确", "错误",
            "verify", "check", "fact", "correct", "validate",
        ],
        "software_dev": [
            "项目", "开发", "实现", "测试", "发布",
            "project", "develop", "implement", "test", "deploy", "build",
        ],
        "event_driven": [
            "监控", "警报", "事件", "实时", "通知",
            "monitor", "alert", "event", "realtime", "notification",
        ],
        # 高优先级类型放在后面（后匹配覆盖前匹配）
        "complex": [
            "综合", "全面", "系统", "深入", "详细", "完整", "多维度",
            "comprehensive", "thorough", "systematic", "in-depth", "detailed", "holistic",
        ],
        "analysis": [
            "分析", "解读", "剖析", "洞察", "趋势", "数据", "报告",
            "analyze", "interpret", "insight", "trend", "data", "report", "breakdown",
        ],
        "creative": [
            "创作", "设计", "构思", "策划", "创意", "方案", "文案",
            "create", "design", "brainstorm", "plan", "creative", "proposal", "copywriting",
        ],
        "research": [
            "研究", "调研", "搜索", "资料", "对比", "评测",
            "research", "investigate", "search", "review", "compare", "evaluate",
        ],
    }

    FEATURE_KEYWORDS = {
        "has_explicit_verification": [
            "验证", "检查", "测试", "标准", "规范",
            "verify", "check", "test", "standard", "validate",
        ],
        "needs_parallelism": [
            "同时", "并行", "多个", "分别", "批量",
            "parallel", "concurrent", "multiple", "batch", "simultaneously",
        ],
        "has_roles": [
            "角色", "分工", "负责", "团队",
            "role", "assign", "responsible", "team",
        ],
        "is_event_driven": [
            "事件", "监控", "警报", "实时", "监听",
            "event", "monitor", "alert", "realtime", "listen",
        ],
        "needs_collaboration": [
            "协作", "一起", "共同", "互相",
            "collaborate", "together", "joint", "cooperative",
        ],
        "iterative_potential": [
            "迭代", "改进", "循环", "多次",
            "iterate", "improve", "loop", "refine",
        ],
        "requires_shared_knowledge": [
            "共享", "知识库", "整合", "综合",
            "shared", "knowledge base", "integrate", "synthesize",
        ],
        # 新增：推理深度特征
        "reasoning_depth": [
            "为什么", "原因", "推导", "证明", "逻辑", "因果",
            "假设", "权衡", "取舍", "利弊",
            "why", "reason", "derive", "prove", "logic", "tradeoff",
        ],
        # 新增：跨域引用特征
        "cross_reference": [
            "结合", "整合", "融合", "综合", "兼顾",
            "跨", "对比", "映射", "关联",
            "combine", "integrate", "cross", "relate",
        ],
    }

    # 否定前缀词：当这些词紧邻关键词时，该关键词匹配无效
    NEGATION_PREFIXES = [
        "不", "没", "无", "非", "未", "别", "勿", "莫",
        "not", "no", "non", "un", "dis", "never", "without",
    ]

    # 否定词和关键词之间允许出现的助动词/连接词
    NEGATION_BRIDGE_WORDS = [
        "需", "需要", "要", "用", "必", "会", "能", "得", "是",
        "必", "须", "该", "应", "可", "经", "经过", "被",
    ]

    # 复杂度信号关键词：这些词暗示任务有隐性复杂度
    COMPLEXITY_SIGNALS = {
        "domain_complexity": [
            "安全", "认证", "权限", "加密", "数据库", "分布式",
            "并发", "微服务", "API", "协议", "算法", "架构",
            "财务", "量化", "回测", "风控", "策略", "因子",
            "security", "auth", "encrypt", "database", "distributed",
            "concurrent", "microservice", "algorithm", "architecture",
        ],
        "output_complexity": [
            "报告", "文档", "设计文档", "表格", "对比", "调查",
            "分析报告", "分析", "总结", "方案", "推荐", "评估",
            "计算", "趋势", "走势", "筛选", "排名",
            "report", "document", "comparison", "analysis", "evaluation",
        ],
        "scope_signals": [
            "完整的", "全面的", "整个", "系统", "项目", "平台",
            "完整的", "全栈", "端到端", "生产级",
            "complete", "comprehensive", "full", "system", "platform",
            "production", "end-to-end",
        ],
        "multi_component": [
            "多个", "多种", "多项", "各类", "分别",
            "同时", "一并", "都", "各",
            "multiple", "various", "each", "all", "both",
        ],
        # 新增：推理深度信号（需要多步推理的任务）
        "reasoning_depth": [
            "为什么", "原因", "推导", "证明", "逻辑", "因果",
            "假设", "如果", "否则", "权衡", "取舍", "利弊",
            "why", "reason", "derive", "prove", "logic", "tradeoff",
        ],
        # 新增：跨域引用信号（需要整合多个领域知识）
        "cross_reference": [
            "结合", "整合", "融合", "综合", "兼顾",
            "跨", "之间", "对比", "映射", "关联",
            "combine", "integrate", "cross", "between", "relate",
        ],
        # 新增：多视角信号（适合 Fusion 的任务特征）
        "multi_perspective": [
            "多角度", "多维度", "全面分析", "深入分析", "综合分析",
            "利弊", "优缺点", "对比分析", "可行性", "风险评估",
            "multi-angle", "holistic", "comprehensive analysis", "pros and cons",
            "feasibility", "risk assessment", "trade-off",
        ],
    }

    def assess(
        self,
        task_description: str,
        context: Optional[Dict] = None,
        external_assessment: Optional[Dict] = None,
    ) -> Dict:
        if external_assessment:
            return {
                "complexity_score": external_assessment.get("complexity_score", 3.0),
                "task_type": external_assessment.get("task_type", "default"),
                "estimated_tokens": self._estimate_tokens(
                    external_assessment.get("features", {}),
                    external_assessment.get("complexity_score", 3.0),
                ),
                "features": external_assessment.get("features", {}),
                "recommended_mode": external_assessment.get(
                    "recommended_mode", "orchestrator_subagent"
                ),
            }

        features = self._extract_features(task_description, context)
        complexity_score = self._calculate_score(features, task_description)
        task_type = self._identify_task_type(task_description)
        estimated_tokens = self._estimate_tokens(features, complexity_score)
        recommended_mode = self._preliminary_recommendation(
            complexity_score, task_type, features
        )

        return {
            "complexity_score": complexity_score,
            "task_type": task_type,
            "estimated_tokens": estimated_tokens,
            "features": {k: v for k, v in features.items() if isinstance(v, (bool, int, float))},
            "recommended_mode": recommended_mode.value,
        }

    def _extract_features(self, task_description: str, context: Optional[Dict]) -> Dict:
        combined = (task_description + " " + str(context or "")).lower()

        features = {}
        for feature_name, keywords in self.FEATURE_KEYWORDS.items():
            features[feature_name] = self._keyword_match(combined, keywords)

        features["context_size"] = len(str(context or ""))
        features["uncertainty_level"] = self._assess_uncertainty(task_description)
        features["task_length"] = len(task_description)

        # 新增：检测推理深度和跨域引用（来自 COMPLEXITY_SIGNALS）
        features["reasoning_depth"] = self._keyword_match(combined, self.COMPLEXITY_SIGNALS["reasoning_depth"])
        features["cross_reference"] = self._keyword_match(combined, self.COMPLEXITY_SIGNALS["cross_reference"])
        # 新增：多视角信号（适合 Fusion 的任务）
        features["multi_perspective"] = self._keyword_match(combined, self.COMPLEXITY_SIGNALS["multi_perspective"])

        return features

    def _keyword_match(self, text: str, keywords: List[str]) -> bool:
        """关键词匹配，带否定前缀过滤"""
        for kw in keywords:
            idx = text.find(kw)
            while idx != -1:
                if idx == 0 or not self._is_negated(text, idx, kw):
                    return True
                idx = text.find(kw, idx + len(kw))
        return False

    def _is_negated(self, text: str, idx: int, keyword: str) -> bool:
        """检查关键词是否被否定前缀修饰（支持中文隔词否定）"""
        # 检查关键词前方 1-6 个字符内是否有否定词
        prefix_start = max(0, idx - 6)
        prefix = text[prefix_start:idx]
        for neg in self.NEGATION_PREFIXES:
            neg_idx = prefix.find(neg)
            if neg_idx == -1:
                continue
            # 否定词和关键词之间的内容
            gap = prefix[neg_idx + len(neg):]
            # 如果中间内容为空或仅包含助动词/连接词，则视为否定
            if not gap:
                return True
            gap_stripped = gap.strip()
            if not gap_stripped:
                return True
            # 检查中间内容是否全部由助动词组成
            all_bridge = True
            remaining = gap_stripped
            for bridge in sorted(self.NEGATION_BRIDGE_WORDS, key=len, reverse=True):
                while bridge in remaining:
                    remaining = remaining.replace(bridge, "", 1)
            if not remaining.strip():
                return True
        return False

    def _calculate_score(self, features: Dict, task_description: str) -> float:
        score = 1.0
        if features["needs_parallelism"]:
            score += 2.0
        if features["has_roles"]:
            score += 1.5
        if features["is_event_driven"]:
            score += 1.0
        if features["needs_collaboration"]:
            score += 2.0
        if features["requires_shared_knowledge"]:
            score += 1.0
        if features["has_explicit_verification"]:
            score += 0.5
        if features["iterative_potential"]:
            score += 1.0

        # ── 新增：隐性复杂度检测 ──
        combined = task_description.lower()

        # 领域复杂度：安全/认证/数据库/分布式等 → 任务天然复杂
        for kw in self.COMPLEXITY_SIGNALS["domain_complexity"]:
            if kw in combined:
                score += 0.5
                break  # 只加一次

        # 输出复杂度：要求报告/文档/对比/评估等 → 需综合分析
        if self._keyword_match(combined, self.COMPLEXITY_SIGNALS["output_complexity"]):
            score += 1.0

        # 范围信号：完整系统/平台/端到端 → 大规模任务
        if self._keyword_match(combined, self.COMPLEXITY_SIGNALS["scope_signals"]):
            score += 1.5

        # 多组件：需要同时处理多个事物 → 并行潜力和拆分需求
        if self._keyword_match(combined, self.COMPLEXITY_SIGNALS["multi_component"]):
            score += 1.0

        # 新增：推理深度：需要多步推理 → generator-verifier 更合适
        if self._keyword_match(combined, self.COMPLEXITY_SIGNALS["reasoning_depth"]):
            score += 0.8

        # 新增：跨域引用：需要整合多领域知识 → 共享状态/团队模式更合适
        if self._keyword_match(combined, self.COMPLEXITY_SIGNALS["cross_reference"]):
            score += 1.2

        # 新增：多视角信号：需要全面分析 → 并行融合更合适
        if self._keyword_match(combined, self.COMPLEXITY_SIGNALS["multi_perspective"]):
            score += 1.5

        # ── 子任务数量检测 ──
        numbered_items = len(re.findall(
            r'(?:^|\n)\s*(?:\d+[.)]\s|[•\-*]\s)', task_description
        ))
        # 也检测数字+顿号模式：1、 2、 3、
        cn_numbered = len(re.findall(r'\d+、', task_description))
        subtask_count = max(numbered_items, cn_numbered)
        if subtask_count >= 3:
            score += 1.5  # 3+ 子任务，需要拆分
        elif subtask_count >= 1:
            score += 0.5

        # ── 多动作动词检测（"分析→计算→对比→生成"类流水线） ──
        ACTION_VERBS = ["分析", "计算", "对比", "生成", "筛选", "评估",
                       "采集", "清洗", "建模", "回测", "部署", "测试"]
        verb_count = sum(1 for v in ACTION_VERBS if v in task_description)
        if verb_count >= 3:
            score += 1.0  # 多步流水线

        # ── 模糊度 ──
        score += features["uncertainty_level"] * 0.3

        # ── 任务长度（降低阈值） ──
        task_len = len(task_description)
        if task_len > 100:
            score += 0.5
        if task_len > 300:
            score += 1.0
        if task_len > 600:
            score += 1.0
        if task_len > 1000:
            score += 1.0

        return min(score, 10.0)

    def _assess_uncertainty(self, task_description: str) -> int:
        vague_words = [
            "可能", "大概", "也许", "或许", "看看", "试试", "研究下", "探索",
            "maybe", "perhaps", "possibly", "explore", "investigate",
        ]
        return min(sum(1 for w in vague_words if w in task_description), 5)

    def _identify_task_type(self, task_description: str) -> str:
        lower = task_description.lower()
        for task_type, keywords in self.TASK_PATTERNS.items():
            if self._keyword_match(lower, keywords):
                return task_type
        return "default"

    def _estimate_tokens(self, features: Dict, complexity_score: float) -> int:
        base = 1000
        multiplier = 1 + (complexity_score / 5)
        return int(base * multiplier)

    def _preliminary_recommendation(
        self, score: float, task_type: str, features: Dict
    ) -> AgentMode:
        # 纯事件驱动（无角色/协作）→ message_bus；有协作特征时不覆盖
        if features["is_event_driven"] and not features.get("has_roles") and not features.get("needs_collaboration"):
            return AgentMode.MESSAGE_BUS
        if features["requires_shared_knowledge"] and score > 5:
            return AgentMode.SHARED_STATE
        # 跨域引用 + 高复杂度 → 共享状态（需要跨域知识整合）
        if features.get("cross_reference") and score > 6:
            return AgentMode.SHARED_STATE
        # 推理深度 + 中等复杂度 → generator-verifier（需要验证推理链）
        if features.get("reasoning_depth") and score <= 6:
            return AgentMode.GENERATOR_VERIFIER
        if features["has_explicit_verification"] and score <= 5:
            return AgentMode.GENERATOR_VERIFIER
        # 并行融合：高复杂度 + 需要多视角分析的任务
        fusion_task_match = task_type in ("analysis", "research", "creative", "complex")
        fusion_signal = features.get("multi_perspective") or (features.get("cross_reference") and score >= 7)
        if score >= 7 and (fusion_task_match or fusion_signal):
            return AgentMode.PARALLEL_FUSION
        if features["has_roles"] and score > 5:
            return AgentMode.AGENT_TEAMS
        if features["needs_collaboration"] and score > 5:
            return AgentMode.AGENT_TEAMS
        if score <= 3:
            return AgentMode.GENERATOR_VERIFIER
        elif score <= 6:
            return AgentMode.ORCHESTRATOR_SUBAGENT
        elif score <= 9:
            return AgentMode.AGENT_TEAMS
        else:
            return AgentMode.SHARED_STATE

class RequirementClarifier:
    """多轮需求澄清 + LLM 评分"""

    def clarify_and_score(
        self,
        ctx,
        task_description: str,
        context: Optional[str] = None,
        max_rounds: int = 3,
        **kwargs,
    ) -> Dict:
        previous_qa = []
        questions_asked = []
        clarified_task = task_description

        for round_num in range(max_rounds):
            prompt = self._build_clarify_prompt(
                task_description, context, previous_qa
            )
            sr = ctx.dispatch_tool("delegate_task", {"goal": prompt}, **kwargs)
            raw = sr if isinstance(sr, str) else str(sr)
            clarify_result = self._parse_clarify_response(raw)

            if not clarify_result.get("needs_clarification", False):
                clarified_task = clarify_result.get("clarified_task", clarified_task)
                break

            questions = clarify_result.get("questions", [])
            questions_asked.extend(questions)

            for q in questions:
                previous_qa.append({"question": q, "answer": ""})

            clarified_task = clarify_result.get("clarified_task", clarified_task)

            if self._is_sufficient(previous_qa, clarify_result.get("extracted_features", {})):
                break

        score_prompt = self._build_score_prompt(
            task_description, context, previous_qa, clarified_task
        )
        sr2 = ctx.dispatch_tool("delegate_task", {"goal": score_prompt}, **kwargs)
        raw2 = sr2 if isinstance(sr2, str) else str(sr2)
        score_result = self._parse_score_response(raw2)

        return {
            "clarified_task": score_result.get("clarified_task", clarified_task),
            "complexity_score": score_result.get("complexity_score", 3.0),
            "task_type": score_result.get("task_type", "default"),
            "features": score_result.get("features", {}),
            "recommended_mode": score_result.get("recommended_mode", "orchestrator_subagent"),
            "clarification_rounds": len(previous_qa),
            "questions_asked": questions_asked,
            "reasoning": score_result.get("reasoning", ""),
        }

    def _build_clarify_prompt(
        self,
        task_description: str,
        context: Optional[str],
        previous_qa: List[Dict],
    ) -> str:
        qa_text = ""
        for qa in previous_qa:
            qa_text += f"问: {qa['question']}\n"
            if qa.get("answer"):
                qa_text += f"答: {qa['answer']}\n"
            else:
                qa_text += "答: (待用户回答)\n"

        return CLARIFY_PROMPT_TEMPLATE.format(
            task_description=task_description,
            context=context or "无",
            previous_qa=qa_text or "无",
        )

    def _build_score_prompt(
        self,
        task_description: str,
        context: Optional[str],
        previous_qa: List[Dict],
        clarified_task: str,
    ) -> str:
        history_text = ""
        for qa in previous_qa:
            history_text += f"问: {qa['question']}\n答: {qa.get('answer', '(待回答)')}\n"

        return SCORE_PROMPT_TEMPLATE.format(
            clarified_task=clarified_task or task_description,
            context=context or "无",
            clarification_history=history_text or "无",
        )

    @staticmethod
    def _parse_clarify_response(llm_response: str) -> Dict:
        json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        try:
            data = json.loads(llm_response)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        return {
            "needs_clarification": True,
            "questions": ["请更详细地描述您的任务目标"],
            "extracted_features": {},
            "clarified_task": llm_response,
        }

    @staticmethod
    def _parse_score_response(llm_response: str) -> Dict:
        json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "complexity_score" in data:
                    return data
            except json.JSONDecodeError:
                pass

        try:
            data = json.loads(llm_response)
            if isinstance(data, dict) and "complexity_score" in data:
                return data
        except json.JSONDecodeError:
            pass

        return {
            "complexity_score": 3.0,
            "task_type": "default",
            "features": {},
            "recommended_mode": "orchestrator_subagent",
            "clarified_task": "",
            "reasoning": "LLM 评分解析失败，使用默认值",
        }

    @staticmethod
    def _is_sufficient(previous_qa: List[Dict], features: Dict) -> bool:
        if not features:
            return False
        determined = sum(1 for v in features.values() if isinstance(v, bool))
        return determined >= 5

