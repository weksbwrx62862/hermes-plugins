from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .persistence import (
    load_performance,
    save_execution_transaction,
    save_performance,
)
from .subagent import (
    AgentMode,
    CircuitBreaker,
    RetryPolicy,
    SubagentConfig,
    SubagentRegistry,
    SubagentResult,
    SubagentStatus,
    SubtaskItem,
    TaskResultStore,
    _MODE_CN,
    MODE_CN_SHORT,
    TASK_TYPE_CN,
    get_template_subtasks,
    validate_subtask_dag,
)


MODE_UPGRADE_ORDER = [
    AgentMode.GENERATOR_VERIFIER,
    AgentMode.ORCHESTRATOR_SUBAGENT,
    AgentMode.AGENT_TEAMS,
    AgentMode.MESSAGE_BUS,
    AgentMode.SHARED_STATE,
]

@dataclass
class SwitchContext:
    """模式切换上下文"""
    failure_reason: str = ""
    intermediate_result: str = ""
    source_mode: str = ""
    target_mode: str = ""
    error_category: Optional[str] = None
    token_usage: int = 0
    time_taken: float = 0.0


DEFAULT_EVENT_TOPOLOGY = {
    "events": ["task_start", "data_received", "result_ready"],
    "subscribers": {
        "task_start": ["planner", "researcher"],
        "data_received": ["analyzer", "processor"],
        "result_ready": ["validator"],
    },
    "transitions": {
        "task_start": "data_received",
        "data_received": "result_ready",
    },
}

TEMPLATE_TOPOLOGIES = {
    "code_generation": {
        "events": ["analyze", "implement", "verify"],
        "subscribers": {
            "analyze": ["architect"],
            "implement": ["developer"],
            "verify": ["reviewer"],
        },
        "transitions": {
            "analyze": "implement",
            "implement": "verify",
        },
    },
    "research": {
        "events": ["search", "synthesize", "report"],
        "subscribers": {
            "search": ["researcher"],
            "synthesize": ["analyst"],
            "report": ["writer"],
        },
        "transitions": {
            "search": "synthesize",
            "synthesize": "report",
        },
    },
    "event_driven": {
        "events": ["trigger", "process", "respond"],
        "subscribers": {
            "trigger": ["listener"],
            "process": ["handler"],
            "respond": ["notifier"],
        },
        "transitions": {
            "trigger": "process",
            "process": "respond",
        },
    },
}


def _validate_event_topology(topology: Dict) -> List[str]:
    """校验事件拓扑合法性，返回错误列表"""
    errors = []
    events = topology.get("events", [])
    subscribers = topology.get("subscribers", {})
    transitions = topology.get("transitions", {})

    if not events:
        errors.append("events 列表为空")
        return errors

    for event_name in subscribers:
        if event_name not in events:
            errors.append(f"subscribers 引用了不存在的 event: {event_name}")

    for from_event, to_event in transitions.items():
        if from_event not in events:
            errors.append(f"transitions 的 key 不在 events 中: {from_event}")
        if to_event not in events:
            errors.append(f"transitions 的 value 不在 events 中: {to_event}")

    for event_name in events:
        if event_name not in subscribers or not subscribers[event_name]:
            errors.append(f"event '{event_name}' 没有订阅者")

    return errors


def _detect_topology_cycle(transitions: Dict[str, str]) -> List[str]:
    """DFS 检测 transitions 图中的环，返回环路径"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in transitions}
    cycles = []

    def dfs(node, path):
        color[node] = GRAY
        path.append(node)
        neighbor = transitions.get(node)
        if neighbor and neighbor in color:
            if color[neighbor] == GRAY:
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
            elif color[neighbor] == WHITE:
                dfs(neighbor, path)
        path.pop()
        color[node] = BLACK

    for node in list(transitions.keys()):
        if color.get(node, WHITE) == WHITE:
            dfs(node, [])

    return cycles


def _break_topology_cycle(transitions: Dict[str, str], cycles: List[str]) -> Dict[str, str]:
    """移除环中最弱的边（最长的 transition，假设越长越弱）"""
    result = dict(transitions)
    for cycle in cycles:
        if len(cycle) >= 2:
            from_node = cycle[-2]
            if from_node in result:
                del result[from_node]
    return result


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
        "research": [
            "研究", "调研", "分析", "搜索", "资料",
            "research", "investigate", "analyze", "search", "review",
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
        if features["has_explicit_verification"] and score <= 5:
            return AgentMode.GENERATOR_VERIFIER
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


class ModeSelectionEngine:
    """模式选择引擎 — Thompson Sampling + 贝叶斯平滑混合策略"""

    def __init__(self, circuit_breakers: Optional[Dict] = None):
        self.historical_performance: Dict[str, Dict[str, Dict]] = load_performance()
        self.circuit_breakers = circuit_breakers or {}
        self._logger = logging.getLogger("ama.selector")
        self._ts_params: Dict = {}
        self._warmup_ts_from_history()

    def _warmup_ts_from_history(self):
        """从历史性能数据预热 Thompson Sampling 先验 Beta"""
        for task_type, modes in self.historical_performance.items():
            for mode_name, stats in modes.items():
                trials = stats.get("trials", 0)
                if trials > 0:
                    successes = stats.get("successes", 0)
                    failures = trials - successes
                    self._ts_params[(task_type, mode_name)] = (
                        1 + successes, 1 + failures
                    )

    def _ts_sample(self, task_type: str, mode: AgentMode) -> float:
        """Thompson Sampling: 从 Beta(α,β) 采样模式期望成功率"""
        key = (task_type, mode.value)
        alpha, beta = self._ts_params.get(key, (1, 1))
        # Beta(0,0) 不合法，保证 ≥0.01
        return random.betavariate(max(alpha, 0.01), max(beta, 0.01))

    def _ts_update(self, task_type: str, mode: AgentMode, success: bool, confidence: float = 1.0):
        """更新 Thompson Sampling 的后验 Beta 分布"""
        key = (task_type, mode.value)
        alpha, beta = self._ts_params.get(key, (1, 1))
        if success:
            self._ts_params[key] = (alpha + confidence, beta)
        else:
            self._ts_params[key] = (alpha, beta + confidence)

    def select_mode(self, assessment: Dict, context: Optional[Dict] = None) -> AgentMode:
        score = assessment["complexity_score"]
        task_type = assessment["task_type"]
        features = assessment["features"]

        candidates = self._apply_rules(score, task_type, features)

        # ── Router→AMA 反向联动：弱模型时降级模式 ──
        try:
            from plugins.model_router import get_active_model_quality
            model_quality = get_active_model_quality()
            if model_quality <= 2:
                heavy_modes = {AgentMode.AGENT_TEAMS, AgentMode.SHARED_STATE, AgentMode.MESSAGE_BUS}
                light_candidates = [m for m in candidates if m not in heavy_modes]
                if light_candidates:
                    self._logger.info("[AMA] Router→AMA联动: 模型质量=%d≤2, 排除重型模式", model_quality)
                    candidates = light_candidates
        except ImportError:
            pass

        available = [m for m in candidates if self._is_mode_available(m)]
        if not available:
            available = candidates

        # 单候选直接返回，无需 TS
        if len(available) == 1:
            return available[0]

        selected, self._last_ts_samples = self._ts_select(available, task_type)
        return selected

    def _ts_select(self, candidates: List[AgentMode], task_type: str):
        samples = [(self._ts_sample(task_type, m), m) for m in candidates]
        samples.sort(key=lambda x: x[0], reverse=True)
        selected = samples[0][1]

        if all(s < 0.6 for s, _ in samples):
            selected = self._select_best_performer(candidates, task_type)
        return selected, samples

    def _is_mode_available(self, mode: AgentMode) -> bool:
        cb = self.circuit_breakers.get(mode)
        if cb is None:
            return True
        return cb.is_available()

    def _apply_rules(
        self, complexity_score: float, task_type: str, features: Dict
    ) -> List[AgentMode]:
        if features.get("has_explicit_verification") and complexity_score < 5:
            return [AgentMode.GENERATOR_VERIFIER]
        if features.get("is_event_driven") and not features.get("has_roles") and not features.get("needs_collaboration"):
            return [AgentMode.MESSAGE_BUS]
        if features.get("requires_shared_knowledge") and complexity_score > 5:
            return [AgentMode.SHARED_STATE]
        if features.get("has_roles") and complexity_score > 5:
            return [AgentMode.AGENT_TEAMS, AgentMode.ORCHESTRATOR_SUBAGENT]
        if features.get("needs_collaboration") and complexity_score > 5:
            return [AgentMode.AGENT_TEAMS, AgentMode.ORCHESTRATOR_SUBAGENT]
        if features.get("needs_parallelism") and complexity_score > 5:
            return [AgentMode.AGENT_TEAMS, AgentMode.ORCHESTRATOR_SUBAGENT]

        if complexity_score <= 3:
            return [AgentMode.GENERATOR_VERIFIER]
        elif complexity_score <= 6:
            return [AgentMode.ORCHESTRATOR_SUBAGENT, AgentMode.GENERATOR_VERIFIER]
        elif complexity_score <= 9:
            return [AgentMode.AGENT_TEAMS, AgentMode.ORCHESTRATOR_SUBAGENT]
        else:
            return [AgentMode.SHARED_STATE, AgentMode.MESSAGE_BUS]

    def _select_best_performer(
        self, candidates: List[AgentMode], task_type: str
    ) -> AgentMode:
        if task_type in self.historical_performance:
            perf = self.historical_performance[task_type]
            scored = [
                (mode, self._calculate_performance_score(perf.get(mode.value)))
                for mode in candidates
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]
        # 无历史数据时，给新模式探索加分（冷启动）
        return candidates[0]

    def _calculate_performance_score(self, mode_stats: Optional[Dict]) -> float:
        if not mode_stats or mode_stats.get("trials", 0) == 0:
            # 冷启动：无数据模式给予探索加分，高于旧模式的初始分
            return 0.7

        trials = mode_stats.get("trials", 1)
        successes = mode_stats.get("successes", 0)

        # 贝叶斯平滑：加入先验（3次成功/5次试验），避免小样本极端值
        prior_successes = 3
        prior_trials = 5
        smoothed_success_rate = (successes + prior_successes) / (trials + prior_trials)

        # 效率指标使用 sigmoid 压缩到 [0, 1]，避免无上界膨胀
        avg_tokens = max(mode_stats.get("avg_tokens", 1000), 1)
        avg_time = max(mode_stats.get("avg_time", 10), 0.1)
        # sigmoid: 值越小效率越高，用 1/(1+x/基准) 压缩
        token_efficiency = 1.0 / (1.0 + avg_tokens / 2000.0)
        time_efficiency = 1.0 / (1.0 + avg_time / 30.0)

        # 统计显著性权重：试验次数越多，评分越可信
        confidence = min(trials / 20.0, 1.0)
        # 可信评分与先验评分的加权混合
        raw_score = smoothed_success_rate * 0.6 + token_efficiency * 0.2 + time_efficiency * 0.2
        prior_score = 0.5
        final_score = confidence * raw_score + (1.0 - confidence) * prior_score

        # 冷启动探索衰减：试验次数少的模式获得额外加分
        exploration_bonus = 0.1 * max(0, 1.0 - trials / 10.0)
        return final_score + exploration_bonus

    def record_performance(
        self,
        task_type: str,
        mode: AgentMode,
        success: bool,
        token_usage: int,
        time_taken: float,
    ) -> None:
        if task_type not in self.historical_performance:
            self.historical_performance[task_type] = {}
        if mode.value not in self.historical_performance[task_type]:
            self.historical_performance[task_type][mode.value] = {
                "trials": 0,
                "successes": 0,
                "avg_tokens": 0,
                "avg_time": 0,
            }

        stats = self.historical_performance[task_type][mode.value]
        total = stats["trials"]

        # 指数衰减：旧数据权重随试验次数增加而降低
        # 衰减因子 0.95 意味着 100 次前的数据权重仅为 0.95^100 ≈ 0.006
        decay = 0.95
        stats["trials"] += 1
        stats["avg_tokens"] = (
            stats["avg_tokens"] * total * decay + token_usage
        ) / (total * decay + 1)
        stats["avg_time"] = (
            stats["avg_time"] * total * decay + time_taken
        ) / (total * decay + 1)
        if success:
            stats["successes"] += 1

        save_performance(task_type, mode.value, stats)

        # 同步更新 Thompson Sampling 后验
        self._ts_update(
            task_type, mode,
            success=success,
            confidence=min(1.0, stats.get("trials", 0) / 10.0),  # 经验越多置信度越高
        )


_PYTHON_EXCEPTION_MARKERS = [
    "not supported between instances of",
    "TypeError:",
    "AttributeError:",
    "ValueError: NoneType",
    "cannot compare",
    "unorderable types",
]


def _is_python_exception_string(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return any(marker in text for marker in _PYTHON_EXCEPTION_MARKERS)


class AdaptiveMultiAgentEngine:

    def __init__(self, config: Optional[Dict] = None):
        self.assessor = TaskComplexityAssessor()
        self.circuit_breakers: Dict[AgentMode, CircuitBreaker] = {
            mode: CircuitBreaker() for mode in AgentMode
        }
        self.selector = ModeSelectionEngine(circuit_breakers=self.circuit_breakers)
        self.session_mode_override: Optional[AgentMode] = None
        self.registry = SubagentRegistry()
        from .subagent import PluginRegistry
        self.plugin_registry = PluginRegistry()
        self.result_store = TaskResultStore()
        self.retry_policy = RetryPolicy()
        self._human_input_mode = "NEVER"
        self._switch_cooldown: Dict[str, float] = {}
        self._switch_cooldown_seconds = 30.0
        self._lifecycle_hooks: Dict[str, List] = {
            "on_started": [], "on_progress": [], "on_completed": [],
            "on_failed": [], "on_timeout": [], "on_cancelled": [],
        }
        self._logger = logging.getLogger("ama.engine")
        self.config = {
            "allow_mode_switch": True,
            "switch_threshold": {"max_tokens": 50000, "max_time": 300},
            "default_mode": "auto",
            "max_concurrent_children": 3,
            "llm_refine_enabled": True,
            "llm_refine_range": (3.0, 7.0),
        }
        if config:
            self.config.update(config)
            if "default_mode" in config and config["default_mode"] != "auto":
                self.session_mode_override = AgentMode(config["default_mode"])

    def update_config(self, cfg: Dict) -> None:
        self.config.update(cfg)

    @staticmethod
    def _extract_token_usage(delegate_result_str: str) -> int:
        """从 delegate_task 返回 JSON 提取 tokens.total，失败回退 len(result)//4"""
        try:
            data = json.loads(delegate_result_str)
            if isinstance(data, dict):
                results = data.get("results", [])
                total = 0
                for r in results:
                    tokens = r.get("tokens", {})
                    total += tokens.get("total", 0)
                if total > 0:
                    return total
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return len(delegate_result_str) // 4

    @staticmethod
    def _extract_tool_traces(delegate_result_str: str) -> List[Dict[str, Any]]:
        """从 delegate_task 返回结果中提取工具调用追踪"""
        traces = []
        try:
            data = json.loads(delegate_result_str)
            if isinstance(data, dict):
                results = data.get("results", [])
                for r in results:
                    tool_calls = r.get("tool_calls", [])
                    for tc in tool_calls:
                        traces.append({
                            "tool_name": tc.get("name", ""),
                            "args_summary": str(tc.get("args", {}))[:200],
                            "status": "completed" if not tc.get("error") else "failed",
                        })
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return traces

    def register_hook(self, event: str, callback) -> None:
        if event in self._lifecycle_hooks:
            self._lifecycle_hooks[event].append(callback)

    def _fire_hook(self, event: str, result: SubagentResult) -> None:
        for cb in self._lifecycle_hooks.get(event, []):
            try:
                cb(result)
            except Exception as e:
                self._logger.warning("生命周期钩子 %s 执行失败: %s", event, e)

    def diagnose(self) -> Dict:
        """诊断 AMA 内部状态：TS 参数、性能历史、熔断器、会话覆盖等"""

        # ── Thompson Sampling 参数 ──
        ts_params = {}
        for (task_type, mode_name), (alpha, beta_val) in self.selector._ts_params.items():
            ts_params.setdefault(task_type, {})[mode_name] = {
                "alpha": round(alpha, 2),
                "beta": round(beta_val, 2),
                "expected": round(alpha / (alpha + beta_val), 3) if (alpha + beta_val) > 0 else 0.5,
                "trials_equivalent": int(alpha + beta_val - 2),  # 等效试验次数（减去先验）
            }

        # ── 性能历史 ──
        perf_summary = {}
        for task_type, modes in self.selector.historical_performance.items():
            for mode_name, stats in modes.items():
                trials = stats.get("trials", 0)
                if trials > 0:
                    perf_summary[f"{task_type}/{mode_name}"] = {
                        "trials": trials,
                        "success_rate": round(stats.get("successes", 0) / trials, 3),
                        "avg_time": round(stats.get("avg_time", 0), 1),
                        "avg_tokens": stats.get("avg_tokens", 0),
                    }

        # ── 熔断器状态 ──
        cb_status = {}
        for mode, cb in self.circuit_breakers.items():
            now_ts = time.time()
            cooling = 0
            if not cb.is_available() and cb._last_failure_time is not None:
                cooling = max(0, int(cb.recovery_timeout - (now_ts - cb._last_failure_time)))
            cb_status[mode.cn] = {
                "available": cb.is_available(),
                "state": cb._state,
                "failures": cb._failure_count,
                "threshold": cb.failure_threshold,
                "cooldown_seconds": cooling,
            }

        # ── 摘要 ──
        ts_count = sum(len(modes) for modes in ts_params.values())
        perf_count = len(perf_summary)
        cb_blocked = sum(1 for s in cb_status.values() if not s["available"])
        lines = [f"[AMA诊断] TS参数: {ts_count}组 | 性能数据: {perf_count}条 | 熔断: {cb_blocked}个断路"]
        lines.append(f"  会话覆盖: {self.session_mode_override.value if self.session_mode_override else '无'}")

        return {
            "summary": "\n".join(lines),
            "ts_params": ts_params,
            "performance": perf_summary,
            "circuit_breakers": cb_status,
            "session_override": self.session_mode_override.value if self.session_mode_override else None,
            "config": {
                "allow_mode_switch": self.config["allow_mode_switch"],
                "llm_refine_enabled": self.config.get("llm_refine_enabled", True),
                "llm_refine_range": self.config.get("llm_refine_range", (3.0, 7.0)),
            },
        }

    def generate_mermaid_diagram(self, trace_id: Optional[str] = None) -> str:
        """基于执行记录生成 Mermaid 流程图"""
        from .persistence import get_execution_by_trace_id, get_stats as get_persistence_stats

        if trace_id:
            records = []
            record = get_execution_by_trace_id(trace_id)
            if record:
                records = [record]
        else:
            stats = get_persistence_stats()
            recent = stats.get("recent_executions", [])
            records = recent[:1] if recent else []

        if not records:
            return "graph TD\n    A[无执行记录] --> B[请先执行任务]"

        record = records[0]
        mode = record.get("mode_used", "unknown")
        success = bool(record.get("success", 0))

        mode_flows = {
            "generator_verifier": [
                ("评估任务", "生成初稿"),
                ("生成初稿", "验证结果"),
                ("验证结果", "通过?"),
                ("通过?", "END[完成]"),
                ("通过?", "生成初稿"),
            ],
            "orchestrator_subagent": [
                ("评估任务", "分解子任务"),
                ("分解子任务", "并行执行子代理"),
                ("并行执行子代理", "综合结果"),
                ("综合结果", "END[完成]"),
            ],
            "agent_teams": [
                ("评估任务", "PM分配任务"),
                ("PM分配任务", "Engineer+Reviewer并行"),
                ("Engineer+Reviewer并行", "综合结果"),
                ("综合结果", "END[完成]"),
            ],
            "message_bus": [
                ("评估任务", "规划事件拓扑"),
                ("规划事件拓扑", "事件循环处理"),
                ("事件循环处理", "END[完成]"),
            ],
            "shared_state": [
                ("评估任务", "初始化共享状态"),
                ("初始化共享状态", "迭代收敛"),
                ("迭代收敛", "收敛?"),
                ("收敛?", "END[完成]"),
                ("收敛?", "迭代收敛"),
            ],
        }

        edges = mode_flows.get(mode, [("评估任务", "执行任务"), ("执行任务", "END[完成]")])

        lines = ["graph TD"]
        for i, (src, dst) in enumerate(edges):
            src_id = f"N{i}"
            if dst == "END[完成]":
                dst_id = "END"
                lines.append(f"    {dst_id}[完成]")
            else:
                dst_id = f"N{i+1}" if i < len(edges) - 1 else "END"
            if src in ("通过?", "收敛?"):
                label = "是" if dst.startswith("END") else "否"
                lines.append(f'    {src_id}{{{src}}} -->|{label}| {dst_id}')
            else:
                lines.append(f"    {src_id}[{src}] --> {dst_id}")

        status_color = "#90EE90" if success else "#FFB6C1"
        lines.append(f"    style END fill:{status_color},stroke:#333")

        return "\n".join(lines)

    def _llm_refine_assessment(self, ctx, task: str, context: str, rule_assessment: Dict, **exec_kwargs) -> Dict:
        """LLM 二次评估：当规则分落在模糊区间时，用大模型重新评分"""
        if not self.config.get("llm_refine_enabled", True):
            return rule_assessment

        try:
            prompt = LLM_REFINE_PROMPT_TEMPLATE.format(
                rule_score=rule_assessment["complexity_score"],
                task=task,
                context=context or "无",
            )
            raw = ctx.dispatch_tool("delegate_task", {
                "goal": prompt,
            }, **exec_kwargs)
            raw_str = raw if isinstance(raw, str) else str(raw)

            # delegate_task 返回 {"results": [{"summary": "..."}]}，需要提取 summary
            llm_text = raw_str
            try:
                wrapper = json.loads(raw_str)
                if isinstance(wrapper, dict) and "results" in wrapper:
                    summaries = [r.get("summary", "") for r in wrapper["results"] if r.get("summary")]
                    if summaries:
                        llm_text = summaries[0]
            except (json.JSONDecodeError, TypeError):
                pass

            llm_result = RequirementClarifier._parse_score_response(llm_text)

            llm_score = llm_result.get("complexity_score", rule_assessment["complexity_score"])

            # 防止 LLM 输出异常值（分数必须在 [1, 10] 之间）
            if not (1.0 <= llm_score <= 10.0):
                self._logger.warning("LLM refine 分数异常: %s，回退规则评分", llm_score)
                return rule_assessment

            merged_features = {**rule_assessment.get("features", {}), **llm_result.get("features", {})}

            return {
                **rule_assessment,
                "complexity_score": llm_score,
                "task_type": llm_result.get("task_type", rule_assessment["task_type"]),
                "features": merged_features,
                "recommended_mode": llm_result.get("recommended_mode", rule_assessment["recommended_mode"]),
                "llm_refined": True,
                "rule_score": rule_assessment["complexity_score"],
                "refine_reasoning": llm_result.get("reasoning", ""),
            }
        except Exception as e:
            self._logger.warning("LLM refine 失败，回退规则评分: %s", e)
            return rule_assessment

    def execute(
        self,
        ctx,
        task: str,
        context: Optional[str] = None,
        force_mode: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        start_time = time.time()
        self._human_input_mode = kwargs.get("human_input_mode", "NEVER")

        assessment = self.assessor.assess(
            task,
            {"context": context} if context else None,
            external_assessment=kwargs.get("external_assessment"),
        )

        # ── 获取当前活跃模型名（供反馈闭环使用） ──
        try:
            from plugins.model_router import _active_model as _rm
            self._active_model_name = _rm.get("name", "") if _rm else ""
        except ImportError:
            self._active_model_name = ""

        # ── LLM 二次评估：规则分落在模糊区间时触发 ──
        if not kwargs.get("external_assessment"):
            lo, hi = self.config.get("llm_refine_range", (3.0, 7.0))
            rule_score = assessment["complexity_score"]
            if lo <= rule_score <= hi:
                assessment = self._llm_refine_assessment(
                    ctx, task, context or "", assessment,
                    **{k: v for k, v in kwargs.items()
                       if k in ("parent_agent", "session_id", "timeout_seconds", "subagent_type")},
                )

        # ── 联动 Model Router：推送任务权重 ──
        session_id = kwargs.get("session_id", "")
        if session_id:
            try:
                from plugins.model_router import set_task_weight
                rec_strategy = set_task_weight(session_id, assessment["complexity_score"])
                self._logger.info(
                    "[AMA→Router] session=%s | AMA评分=%.1f | 推荐策略=%s",
                    session_id[:8], assessment["complexity_score"], rec_strategy,
                )
            except ImportError:
                pass  # Model Router 未安装则跳过

        if force_mode:
            selected_mode = AgentMode(force_mode)
        elif self.session_mode_override:
            selected_mode = self.session_mode_override
        else:
            selected_mode = self.selector.select_mode(assessment)

        # ── 可视化：输出选中模式（对齐 Model Router 风格） ──
        features_active = [k for k, v in assessment.get("features", {}).items()
                           if v and k not in ("context_size", "uncertainty_level", "task_length")]
        rule_score = assessment.get("rule_score", assessment["complexity_score"])
        llm_refined = assessment.get("llm_refined", False)
        score_part = f"规则={rule_score:.1f}" + (f"→LLM={assessment['complexity_score']:.1f}" if llm_refined else "")
        session_tag = f"[{kwargs.get('session_id', '')[:8]}]" if kwargs.get('session_id') else ""

        # 选型原因：强制模式 / 会话覆盖 / TS采样 / 规则引擎
        if force_mode:
            reason = f"强制指定 mode={force_mode}"
        elif self.session_mode_override:
            reason = f"会话覆盖 mode={self.session_mode_override.value}"
        else:
            # TS 采样结果速览
            candidates = self.selector._apply_rules(
                assessment["complexity_score"], assessment["task_type"], assessment["features"]
            )
            last_samples = getattr(self.selector, '_last_ts_samples', [])
            ts_info = ",".join(
                f"{MODE_CN_SHORT.get(m.value, m.value)}={s:.2f}"
                for s, m in last_samples[:3]
            ) if last_samples else "N/A"
            reason = f"TS采样({ts_info})"
        self._logger.info(
            "[AMA] %s | 类型=%s | 复杂度=%s | 选中: %s | 原因: %s | 特征=%s",
            session_tag,
            assessment["task_type"],
            score_part,
            selected_mode.cn,
            reason,
            ",".join(features_active) if features_active else "无",
        )

        self._last_assessment = assessment
        result = self._execute_mode(ctx, task, context, selected_mode, **kwargs)

        switched = False
        original_mode = selected_mode
        switch_reason = None

        if self.config["allow_mode_switch"] and not result.get("success"):
            switched_result = self._try_switch_mode(
                ctx, task, context, selected_mode, result, **kwargs
            )
            if switched_result is not None:
                switched = True
                switch_reason = f"{selected_mode.value} 执行失败，升级切换"
                result = switched_result

        time_taken = time.time() - start_time
        mode_used = result.get("mode", selected_mode.value)
        if isinstance(mode_used, AgentMode):
            mode_used = mode_used.value

        success = result.get("success", False)
        token_usage = result.get("token_usage", 0)

        self.selector.record_performance(
            assessment["task_type"],
            AgentMode(mode_used) if isinstance(mode_used, str) else mode_used,
            success,
            token_usage,
            time_taken,
        )

        mode_key = AgentMode(mode_used) if isinstance(mode_used, str) else mode_used
        if success:
            self.circuit_breakers[mode_key].record_success()
        else:
            self.circuit_breakers[mode_key].record_failure()

        perf_stats = self.selector.historical_performance.get(
            assessment["task_type"], {}
        ).get(mode_key.value, {"trials": 0, "successes": 0, "avg_tokens": 0, "avg_time": 0})

        save_execution_transaction(
            task_type=assessment["task_type"],
            mode=mode_key.value,
            stats=perf_stats,
            session_id=kwargs.get("session_id"),
            task=task,
            complexity_score=assessment["complexity_score"],
            mode_used=mode_used,
            original_mode=original_mode.value,
            success=success,
            token_usage=token_usage,
            time_taken=time_taken,
            switched_modes=switched,
            switch_reason=switch_reason,
            trace_id=result.get("trace_id", ""),
            status=result.get("status", ""),
            error_category=result.get("error_category"),
            retries_attempted=result.get("retries_attempted", 0),
            timeout_seconds=kwargs.get("timeout_seconds"),
        )

        # ── 反馈闭环：AMA 执行结果回流 Router 选型 ──
        try:
            from plugins.model_router import record_model_feedback, get_active_model_quality
            model_name = getattr(self, "_active_model_name", "") or "unknown"
            record_model_feedback(model_name, success, token_usage)
        except ImportError:
            pass

        # ── 选型诊断摘要（供 agent 消费，可自然融入对话） ──
        diag_type = TASK_TYPE_CN.get(assessment["task_type"], assessment["task_type"])
        diag_mode = _MODE_CN.get(mode_used if isinstance(mode_used, AgentMode) else AgentMode(mode_used), mode_used if isinstance(mode_used, str) else mode_used.value)
        diag_orig = original_mode.cn
        diag_score = f"规则={rule_score:.1f}" + (f"→LLM精修={assessment['complexity_score']:.1f}" if llm_refined else f"={assessment['complexity_score']:.1f}")
        diag_line = f"任务类型={diag_type} | 复杂度={diag_score} | 选中模式={diag_mode} | 原因={reason}"
        diag_switched = f" | ⚠️ 模式切换: {diag_orig}→{diag_mode}（{switch_reason}）" if switched else ""
        diagnosis = {
            "summary": f"[AMA选型] {diag_line}{diag_switched}",
            "task_type": diag_type,
            "task_type_raw": assessment["task_type"],
            "complexity_score": assessment["complexity_score"],
            "rule_score": rule_score,
            "llm_refined": llm_refined,
            "selected_mode": diag_mode,
            "selected_mode_raw": mode_used,
            "original_mode": diag_orig,
            "reason": reason,
            "features": features_active,
            "switched": switched,
            "switch_reason": switch_reason,
            "ts_samples": ts_info if not (force_mode or self.session_mode_override) else None,
        }

        return {
            "result": result.get("result", ""),
            "success": success,
            "token_usage": token_usage,
            "time_taken": time_taken,
            "mode_used": mode_used,
            "complexity_score": assessment["complexity_score"],
            "task_type": assessment["task_type"],
            "diagnosis": diagnosis,
            "switched_modes": switched,
            "original_mode": original_mode.value,
            "switch_reason": switch_reason,
            "metadata": result.get("metadata", {}),
            "task_id": result.get("task_id", ""),
            "trace_id": result.get("trace_id", ""),
            "status": result.get("status", ""),
            "error_category": result.get("error_category"),
            "retries_attempted": result.get("retries_attempted", 0),
        }

    def _execute_mode(
        self,
        ctx,
        task: str,
        context: Optional[str],
        mode: AgentMode,
        **kwargs,
    ) -> Dict:
        # 优先从插件注册表查找
        plugin = self.registry.get_plugin(mode)
        if plugin is not None:
            from .graph import ModeGraph
            graph = ModeGraph(mode.value)
            plugin.register_graph(graph)
            compiled = graph.compile()
            return compiled.execute(ctx, task, context, **kwargs)

        # 降级到内置模式
        if mode == AgentMode.GENERATOR_VERIFIER:
            return self._run_generator_verifier(ctx, task, context, **kwargs)
        elif mode == AgentMode.ORCHESTRATOR_SUBAGENT:
            return self._run_orchestrator_subagent(ctx, task, context, **kwargs)
        elif mode == AgentMode.AGENT_TEAMS:
            return self._run_agent_teams(ctx, task, context, **kwargs)
        elif mode == AgentMode.MESSAGE_BUS:
            return self._run_message_bus(ctx, task, context, **kwargs)
        elif mode == AgentMode.SHARED_STATE:
            return self._run_shared_state(ctx, task, context, **kwargs)
        return {"success": False, "result": f"未知模式: {mode}"}

    def _execute_subagent(
        self,
        ctx,
        goal: str,
        config: Optional[SubagentConfig] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> SubagentResult:
        mode = kwargs.get("_mode", AgentMode.GENERATOR_VERIFIER)
        if config is None:
            config = self.registry.get(mode)

        result = SubagentResult(
            trace_id=kwargs.get("trace_id", str(uuid.uuid4())),
            status=SubagentStatus.PENDING,
        )
        self.result_store.put(result.task_id, result)

        timeout = kwargs.get("timeout_seconds") or config.timeout_seconds
        cancel_event = result.cancel_event

        result.status = SubagentStatus.RUNNING
        if self._human_input_mode == "ALWAYS":
            try:
                approval = ctx.dispatch_tool("ask_user", {
                    "question": f"即将执行子代理任务: {goal[:200]}...\n是否继续？",
                })
                if isinstance(approval, str) and any(kw in approval.lower() for kw in ["否", "取消", "no", "cancel", "skip"]):
                    result.status = SubagentStatus.CANCELLED
                    self._fire_hook("on_cancelled", result)
                    return result
            except Exception:
                pass
        self._fire_hook("on_started", result)

        start = time.time()
        retries = 0

        while retries <= config.max_retries:
            try:
                if cancel_event.is_set():
                    result.status = SubagentStatus.CANCELLED
                    self._fire_hook("on_cancelled", result)
                    return result

                elapsed = time.time() - start
                if elapsed > timeout:
                    result.status = SubagentStatus.TIMED_OUT
                    result.elapsed_seconds = elapsed
                    cancel_event.set()
                    self._fire_hook("on_timeout", result)
                    return result

                args = {"goal": goal}
                if context:
                    args["context"] = context
                if config.tools:
                    args["toolsets"] = config.tools

                delegate_result = ctx.dispatch_tool("delegate_task", args, **kwargs)

                tool_traces = self._extract_tool_traces(delegate_result)
                result.tool_trace = tool_traces

                # 检测子代理是否返回了底层 Python 异常（如 float/None 比较 bug）
                if _is_python_exception_string(delegate_result):
                    result.status = SubagentStatus.FAILED
                    result.result = delegate_result
                    result.error_category = "internal_error"
                    result.elapsed_seconds = time.time() - start
                    self._logger.error(
                        "子代理返回 Python 异常 (mode=%s, goal=%.100s): %s",
                        mode.value, goal, delegate_result[:500],
                    )
                    self._fire_hook("on_failed", result)
                    return result

                result.result = delegate_result
                result.token_usage = self._extract_token_usage(delegate_result)
                result.status = SubagentStatus.COMPLETED
                result.elapsed_seconds = time.time() - start
                self._fire_hook("on_completed", result)
                return result

            except Exception as e:
                error_category = self.retry_policy.classify_error(e)
                result.error_category = error_category

                if self.retry_policy.should_retry(error_category) and retries < config.max_retries:
                    retries += 1
                    result.retries_attempted = retries
                    result.status = SubagentStatus.RETRYING
                    wait = self.retry_policy.get_wait_time(retries)
                    self._logger.info(
                        "子代理重试 %d/%d，等待 %.1fs，错误: %s",
                        retries, config.max_retries, wait, error_category,
                    )
                    time.sleep(wait)
                    continue

                result.status = SubagentStatus.FAILED
                result.result = str(e)
                result.elapsed_seconds = time.time() - start
                self._fire_hook("on_failed", result)
                return result

        result.status = SubagentStatus.FAILED
        result.error_category = "max_retries_exceeded"
        result.elapsed_seconds = time.time() - start
        self._fire_hook("on_failed", result)
        return result

    def _execute_subagent_parallel(
        self,
        ctx,
        tasks: List[Dict],
        **kwargs,
    ) -> List[SubagentResult]:
        """并行执行多个子代理任务，利用 delegate_task 的 tasks 参数"""
        mode = kwargs.get("_mode", AgentMode.ORCHESTRATOR_SUBAGENT)
        config = self.registry.get(mode)

        parallel_tasks = []
        for t in tasks:
            args = {"goal": t.get("goal", t.get("description", ""))}
            if t.get("context"):
                args["context"] = t["context"]
            parallel_tasks.append(args)

        results = []
        try:
            delegate_result_str = ctx.dispatch_tool(
                "delegate_task", {"tasks": parallel_tasks}, **kwargs
            )
            parsed = self._parse_delegate_results(delegate_result_str)
            for i, item in enumerate(parsed):
                is_error = isinstance(item, dict) and "error" in item and "result" not in item
                if is_error:
                    sr = SubagentResult(
                        trace_id=str(uuid.uuid4()),
                        status=SubagentStatus.FAILED,
                        result=item.get("error", str(item)),
                        error_category="subagent_error",
                        token_usage=item.get("tokens", {}).get("total", 0),
                    )
                else:
                    sr = SubagentResult(
                        trace_id=str(uuid.uuid4()),
                        status=SubagentStatus.COMPLETED,
                        result=item.get("result", str(item)),
                        token_usage=item.get("tokens", {}).get("total", 0),
                    )
                if sr.token_usage == 0:
                    sr.token_usage = len(str(item)) // 4
                results.append(sr)
        except Exception as e:
            for _ in tasks:
                sr = SubagentResult(
                    status=SubagentStatus.FAILED,
                    error_category=self.retry_policy.classify_error(e),
                    result=str(e),
                )
                results.append(sr)

        return results

    @staticmethod
    def _parse_delegate_results(delegate_result_str: str) -> List[Dict]:
        """解析并行调用返回的 {"results": [...]} 格式"""
        try:
            data = json.loads(delegate_result_str)
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return [{"result": delegate_result_str}]

    def _delegate(
        self,
        ctx,
        goal: str,
        context: Optional[str] = None,
        toolsets: Optional[list] = None,
        **kwargs,
    ) -> str:
        """向后兼容的委托方法，内部转为 _execute_subagent 调用"""
        mode = kwargs.get("_mode", AgentMode.GENERATOR_VERIFIER)
        config = self.registry.get(mode)
        if toolsets:
            config = SubagentConfig(
                name=config.name, description=config.description,
                system_prompt=config.system_prompt, tools=toolsets,
                disallowed_tools=config.disallowed_tools, model=config.model,
                max_turns=config.max_turns, timeout_seconds=config.timeout_seconds,
                priority=config.priority, max_retries=config.max_retries,
            )
        sr = self._execute_subagent(ctx, goal, config, context, _mode=mode, **kwargs)
        return sr.result or ""

    def _run_generator_verifier(
        self, ctx, task: str, context: Optional[str], **kwargs
    ) -> Dict:
        start_time = time.time()
        tokens_used = 0
        max_iterations = 5
        result_text = ""
        success = False
        trace_id = str(uuid.uuid4())

        for i in range(max_iterations):
            gen_goal = f"完成以下任务: {task}"
            if i > 0:
                gen_goal += f"\n\n之前版本: {result_text}\n请根据验证反馈改进。"
            if context:
                gen_goal += f"\n\n上下文: {context}"

            gen_sr = self._execute_subagent(
                ctx, gen_goal, context=context,
                _mode=AgentMode.GENERATOR_VERIFIER, trace_id=trace_id, **kwargs,
            )
            tokens_used += gen_sr.token_usage
            if gen_sr.status != SubagentStatus.COMPLETED:
                break
            gen_result = gen_sr.result or ""

            verify_goal = (
                f"验证以下结果是否正确完成了任务。\n\n"
                f"原始任务: {task}\n\n"
                f"生成结果: {gen_result}\n\n"
                f"请以 JSON 格式返回验证结果：\n"
                f'{{"passed": true/false, "feedback": "验证反馈说明"}}\n\n'
                f"如果 JSON 格式不便，也可以直接回复'通过'或指出问题。"
            )
            verify_sr = self._execute_subagent(
                ctx, verify_goal, context=context,
                _mode=AgentMode.GENERATOR_VERIFIER, trace_id=trace_id, **kwargs,
            )
            tokens_used += verify_sr.token_usage

            passed, feedback = self._parse_verification_result(verify_sr.result or "")
            result_text = gen_result

            if passed:
                success = True
                break

        return {
            "result": result_text,
            "success": success,
            "token_usage": tokens_used,
            "time_taken": time.time() - start_time,
            "mode": AgentMode.GENERATOR_VERIFIER.value,
            "metadata": {"iterations": i + 1, "converged": success},
            "task_id": gen_sr.task_id,
            "trace_id": trace_id,
            "status": SubagentStatus.COMPLETED.value if success else SubagentStatus.FAILED.value,
            "error_category": None,
            "retries_attempted": gen_sr.retries_attempted,
        }

    @staticmethod
    def _parse_verification_result(result_str: str) -> tuple:
        if not result_str or _is_python_exception_string(result_str):
            return False, result_str or ""
        try:
            data = json.loads(result_str)
            if isinstance(data, dict) and "passed" in data:
                return bool(data["passed"]), data.get("feedback", "")
        except (json.JSONDecodeError, TypeError):
            pass
        passed = any(kw in result_str for kw in ["通过", "满意", "pass", "approved"])
        if not passed:
            passed = "正确" in result_str and "不正确" not in result_str and "错误" not in result_str
        return passed, result_str

    def _run_orchestrator_subagent(
        self, ctx, task: str, context: Optional[str], **kwargs
    ) -> Dict:
        start_time = time.time()
        tokens_used = 0
        trace_id = str(uuid.uuid4())

        decompose_goal = (
            f"将以下任务分解为 2-4 个独立的子任务，每个子任务用一句话描述。\n"
            f"必须严格返回以下 JSON 格式，不要添加其他内容：\n"
            f'```json\n{{"subtasks": [{{"id": "1", "description": "子任务描述"}}, ...]}}\n```\n\n'
            f"任务: {task}"
        )
        if context:
            decompose_goal += f"\n\n上下文: {context}"

        decompose_sr = self._execute_subagent(
            ctx, decompose_goal,
            _mode=AgentMode.ORCHESTRATOR_SUBAGENT, trace_id=trace_id, **kwargs,
        )
        tokens_used += decompose_sr.token_usage
        subtasks = self._parse_subtasks_json(decompose_sr.result or "", task_type=self._last_assessment.get("task_type", "default") if hasattr(self, '_last_assessment') else "default")

        parallel_tasks = [
            {"goal": st.get("description", str(st)), "context": f"这是大任务的一部分: {task}"}
            for st in subtasks
        ]
        parallel_results = self._execute_subagent_parallel(
            ctx, parallel_tasks, _mode=AgentMode.ORCHESTRATOR_SUBAGENT, **kwargs,
        )

        subtask_results = {}
        for idx, (st, sr) in enumerate(zip(subtasks, parallel_results)):
            tokens_used += sr.token_usage
            subtask_results[st.get("id", str(idx))] = sr.result or ""

        synth_goal = (
            f"综合以下子任务结果，完成最终任务。\n\n"
            f"任务: {task}\n\n"
            f"子任务结果:\n"
        )
        for sid, res in subtask_results.items():
            synth_goal += f"\n--- 子任务 {sid} ---\n{res}\n"

        synth_sr = self._execute_subagent(
            ctx, synth_goal,
            _mode=AgentMode.ORCHESTRATOR_SUBAGENT, trace_id=trace_id, **kwargs,
        )
        tokens_used += synth_sr.token_usage

        return {
            "result": synth_sr.result or "",
            "success": bool(synth_sr.result) and synth_sr.status == SubagentStatus.COMPLETED,
            "token_usage": tokens_used,
            "time_taken": time.time() - start_time,
            "mode": AgentMode.ORCHESTRATOR_SUBAGENT.value,
            "metadata": {
                "subtasks_executed": len(subtasks),
                "subtask_ids": list(subtask_results.keys()),
            },
            "task_id": decompose_sr.task_id,
            "trace_id": trace_id,
            "status": SubagentStatus.COMPLETED.value,
            "error_category": None,
            "retries_attempted": decompose_sr.retries_attempted,
        }

    @staticmethod
    def _parse_subtasks_json(raw_str: str, task_type: str = "default") -> List[Dict]:
        """多层容错解析子任务 JSON，集成 DAG 校验"""
        raw_items = None

        json_match = re.search(r'```json\s*(.*?)\s*```', raw_str, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "subtasks" in data:
                    raw_items = data["subtasks"]
                elif isinstance(data, list):
                    raw_items = data
            except json.JSONDecodeError:
                pass

        if raw_items is None:
            try:
                data = json.loads(raw_str)
                if isinstance(data, dict) and "subtasks" in data:
                    raw_items = data["subtasks"]
                elif isinstance(data, list):
                    raw_items = data
            except json.JSONDecodeError:
                pass

        if raw_items is None:
            items = re.findall(r'"id"\s*:\s*"(\d+)"\s*,\s*"description"\s*:\s*"([^"]*)"', raw_str)
            if items:
                raw_items = [{"id": mid, "description": desc} for mid, desc in items]

        if raw_items:
            subtask_items = []
            for item in raw_items:
                if isinstance(item, dict):
                    subtask_items.append(SubtaskItem(
                        id=str(item.get("id", "")),
                        description=item.get("description", str(item)),
                        dependencies=[str(d) for d in item.get("dependencies", [])],
                        expected_output=item.get("expected_output", ""),
                    ))
                elif isinstance(item, SubtaskItem):
                    subtask_items.append(item)

            field_errors = []
            for st in subtask_items:
                field_errors.extend(st.validate())

            if not field_errors:
                dag_errors = validate_subtask_dag(subtask_items)
                if not dag_errors:
                    return [
                        {"id": st.id, "description": st.description,
                         "dependencies": st.dependencies, "expected_output": st.expected_output}
                        for st in subtask_items
                    ]
                logging.getLogger("ama.engine").warning("子任务 DAG 校验失败: %s，使用模板", dag_errors)
            else:
                logging.getLogger("ama.engine").warning("子任务字段校验失败: %s，使用模板", field_errors)

        template = get_template_subtasks(task_type)
        logging.getLogger("ama.engine").info("使用 %s 类型的模板子任务", task_type)
        return [
            {"id": st.id, "description": st.description,
             "dependencies": st.dependencies, "expected_output": st.expected_output}
            for st in template
        ]

    def _run_agent_teams(
        self, ctx, task: str, context: Optional[str], **kwargs
    ) -> Dict:
        start_time = time.time()
        tokens_used = 0
        trace_id = str(uuid.uuid4())

        pm_goal = (
            f"角色: product_manager - 负责规划和需求分析\n\n"
            f"任务: {task}\n\n"
        )
        if context:
            pm_goal += f"上下文: {context}\n\n"
        pm_goal += "请输出需求分析和规划方案。"

        pm_sr = self._execute_subagent(
            ctx, pm_goal,
            _mode=AgentMode.AGENT_TEAMS, trace_id=trace_id, **kwargs,
        )
        tokens_used += pm_sr.token_usage
        pm_result = pm_sr.result or ""

        parallel_tasks = [
            {
                "goal": (
                    f"角色: engineer - 负责实现\n\n"
                    f"任务: {task}\n\n"
                    f"产品经理的规划: {pm_result}\n\n"
                    f"请根据规划实现任务。"
                ),
                "context": context,
            },
            {
                "goal": (
                    f"角色: reviewer - 负责质量检查\n\n"
                    f"任务: {task}\n\n"
                    f"产品经理的规划: {pm_result}\n\n"
                    f"请根据规划制定质量检查标准。"
                ),
                "context": context,
            },
        ]
        parallel_results = self._execute_subagent_parallel(
            ctx, parallel_tasks, _mode=AgentMode.AGENT_TEAMS, **kwargs,
        )

        shared_results = {"product_manager": pm_result}
        for idx, sr in enumerate(parallel_results):
            tokens_used += sr.token_usage
            role = "engineer" if idx == 0 else "reviewer"
            shared_results[role] = sr.result or ""

        final_result = shared_results.get("reviewer", shared_results.get("engineer", ""))
        return {
            "result": final_result,
            "success": bool(final_result) and all(
                sr.status == SubagentStatus.COMPLETED for sr in parallel_results
            ) and pm_sr.status == SubagentStatus.COMPLETED,
            "token_usage": tokens_used,
            "time_taken": time.time() - start_time,
            "mode": AgentMode.AGENT_TEAMS.value,
            "metadata": {
                "team_size": 3,
                "team_members": list(shared_results.keys()),
            },
            "task_id": pm_sr.task_id,
            "trace_id": trace_id,
            "status": SubagentStatus.COMPLETED.value,
            "error_category": None,
            "retries_attempted": pm_sr.retries_attempted,
        }

    def _run_message_bus(
        self, ctx, task: str, context: Optional[str], **kwargs
    ) -> Dict:
        start_time = time.time()
        tokens_used = 0
        trace_id = str(uuid.uuid4())

        _task_type = getattr(self, '_last_assessment', {}).get("task_type", "default")
        topology = self._plan_event_topology(ctx, task, context, trace_id=trace_id, _task_type=_task_type, **kwargs)

        subscribers = topology.get("subscribers", DEFAULT_EVENT_TOPOLOGY["subscribers"])
        transitions = topology.get("transitions", DEFAULT_EVENT_TOPOLOGY["transitions"])

        event_queue = [{"type": topology.get("events", ["task_start"])[0], "task": task, "context": context}]
        results = {}
        max_events = 15
        event_count = 0

        while event_queue and event_count < max_events:
            event = event_queue.pop(0)
            event_count += 1

            if event["type"] in subscribers:
                event_subs = subscribers[event["type"]]
                if len(event_subs) > 1:
                    parallel_tasks = []
                    for subscriber in event_subs:
                        parallel_tasks.append({
                            "goal": (
                                f"订阅者: {subscriber}\n"
                                f"处理事件类型: {event['type']}\n"
                                f"任务: {event.get('task', '')}\n"
                                f"数据: {json.dumps(event, ensure_ascii=False)}"
                            ),
                            "context": None,
                        })
                    parallel_results = self._execute_subagent_parallel(
                        ctx, parallel_tasks, _mode=AgentMode.MESSAGE_BUS, **kwargs,
                    )
                    for idx, (subscriber, sr) in enumerate(zip(event_subs, parallel_results)):
                        results[f"{subscriber}_{event_count}"] = sr.result or ""
                        tokens_used += sr.token_usage
                    next_event_type = transitions.get(event["type"])
                    if next_event_type and parallel_results:
                        event_queue.append({
                            "type": next_event_type,
                            "task": task,
                            "data": parallel_results[0].result or "",
                        })
                else:
                    subscriber = event_subs[0]
                    goal = (
                        f"订阅者: {subscriber}\n"
                        f"处理事件类型: {event['type']}\n"
                        f"任务: {event.get('task', '')}\n"
                        f"数据: {json.dumps(event, ensure_ascii=False)}"
                    )
                    sr = self._execute_subagent(
                        ctx, goal,
                        _mode=AgentMode.MESSAGE_BUS, trace_id=trace_id, **kwargs,
                    )
                    results[f"{subscriber}_{event_count}"] = sr.result or ""
                    tokens_used += sr.token_usage
                    next_event_type = transitions.get(event["type"])
                    if next_event_type:
                        event_queue.append({
                            "type": next_event_type,
                            "task": task,
                            "data": sr.result or "",
                        })

        final_parts = [v for v in results.values()]
        final_result = "\n\n".join(final_parts) if final_parts else "无结果"

        return {
            "result": final_result,
            "success": event_count > 0,
            "token_usage": tokens_used,
            "time_taken": time.time() - start_time,
            "mode": AgentMode.MESSAGE_BUS.value,
            "metadata": {
                "events_processed": event_count,
                "subscribers": list(subscribers.keys()),
            },
            "task_id": "",
            "trace_id": trace_id,
            "status": SubagentStatus.COMPLETED.value if event_count > 0 else SubagentStatus.FAILED.value,
            "error_category": None,
            "retries_attempted": 0,
        }

    def _plan_event_topology(
        self, ctx, task: str, context: Optional[str], **kwargs
    ) -> Dict:
        """通过子智能体动态生成事件拓扑"""
        topology_goal = (
            f"为以下任务设计事件驱动处理流程。\n\n"
            f"任务: {task}\n\n"
            f"请返回 JSON 格式的事件拓扑定义：\n"
            f'```json\n{{'
            f'"events": ["event1", "event2", ...],'
            f'"subscribers": {{"event1": ["subscriber1", ...], ...}},'
            f'"transitions": {{"event1": "event2", ...}}'
            f'}}\n```\n\n'
            f"如果无法确定，请返回空 JSON。"
        )
        if context:
            topology_goal += f"\n\n上下文: {context}"

        try:
            sr = self._execute_subagent(
                ctx, topology_goal,
                _mode=AgentMode.MESSAGE_BUS, **kwargs,
            )
            raw = sr.result or ""
            json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
            if json_match:
                raw = json_match.group(1)
            data = json.loads(raw)
            if isinstance(data, dict) and "events" in data and "subscribers" in data:
                errors = _validate_event_topology(data)
                if errors:
                    self._logger.warning("LLM 生成的拓扑校验失败: %s", errors)
                else:
                    cycles = _detect_topology_cycle(data.get("transitions", {}))
                    if cycles:
                        self._logger.warning("拓扑存在环: %s，尝试断环", cycles)
                        data["transitions"] = _break_topology_cycle(data.get("transitions", {}), cycles)
                    return data
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        task_type = kwargs.get("_task_type", "default")
        template = TEMPLATE_TOPOLOGIES.get(task_type, DEFAULT_EVENT_TOPOLOGY)
        self._logger.info("使用模板拓扑: %s", task_type)
        return template

    def _run_shared_state(
        self, ctx, task: str, context: Optional[str], **kwargs
    ) -> Dict:
        start_time = time.time()
        tokens_used = 0
        trace_id = str(uuid.uuid4())

        shared_store = {
            "task": task,
            "context": context,
            "findings": [],
            "drafts": [],
            "validated": False,
            "round": 0,
        }

        agents = ["explorer", "analyzer", "synthesizer", "validator"]
        max_rounds = 6
        converged = False
        round_num = 0

        for round_num in range(max_rounds):
            shared_store["round"] = round_num + 1

            for agent in agents:
                state_summary = json.dumps(
                    {k: v for k, v in shared_store.items() if k != "context"},
                    ensure_ascii=False,
                )
                goal = f"Agent 角色: {agent}\n共享状态: {state_summary}\n\n"

                if agent == "explorer":
                    goal += "请探索和调研任务相关信息，输出你的发现。"
                elif agent == "analyzer":
                    goal += "请分析已有发现，输出你的分析草稿。"
                elif agent == "synthesizer":
                    goal += "请综合所有发现和分析，输出综合草稿。"
                elif agent == "validator":
                    goal += "请验证综合结果是否满足任务要求。如果满意请回复'验证通过'，否则指出问题。"

                sr = self._execute_subagent(
                    ctx, goal,
                    _mode=AgentMode.SHARED_STATE, trace_id=trace_id, **kwargs,
                )
                tokens_used += sr.token_usage
                result = sr.result or ""

                if agent == "explorer":
                    shared_store["findings"].append(result)
                elif agent == "analyzer":
                    shared_store["drafts"].append(result)
                elif agent == "synthesizer":
                    shared_store["drafts"].append(result)
                elif agent == "validator":
                    if "验证通过" in result or "通过" in result or "满意" in result:
                        shared_store["validated"] = True
                        shared_store["final_result"] = result

            if shared_store["validated"]:
                converged = True
                break

        final_result = shared_store.get("final_result", "未产生结果")

        return {
            "result": final_result,
            "success": converged,
            "token_usage": tokens_used,
            "time_taken": time.time() - start_time,
            "mode": AgentMode.SHARED_STATE.value,
            "metadata": {
                "rounds_used": round_num + 1,
                "converged": converged,
            },
            "task_id": "",
            "trace_id": trace_id,
            "status": SubagentStatus.COMPLETED.value if converged else SubagentStatus.FAILED.value,
            "error_category": None,
            "retries_attempted": 0,
        }

    def _smart_switch_strategy(self, failed_mode: AgentMode, error_category: Optional[str]) -> List[AgentMode]:
        """基于错误类型的智能切换策略"""
        downgrade_triggers = {"context_overflow", "timeout"}
        if error_category in downgrade_triggers:
            idx = MODE_UPGRADE_ORDER.index(failed_mode)
            candidates = []
            for offset in range(1, len(MODE_UPGRADE_ORDER)):
                downgrade_idx = idx - offset
                if downgrade_idx >= 0:
                    mode = MODE_UPGRADE_ORDER[downgrade_idx]
                    if self.circuit_breakers[mode].is_available():
                        candidates.append(mode)
            return candidates

        upgrade_triggers = {"verification_failed", "internal_error", "json_parse_error"}
        if error_category in upgrade_triggers:
            idx = MODE_UPGRADE_ORDER.index(failed_mode)
            candidates = []
            for offset in range(1, len(MODE_UPGRADE_ORDER)):
                upgrade_idx = idx + offset
                if upgrade_idx < len(MODE_UPGRADE_ORDER):
                    mode = MODE_UPGRADE_ORDER[upgrade_idx]
                    if self.circuit_breakers[mode].is_available():
                        candidates.append(mode)
            return candidates

        idx = MODE_UPGRADE_ORDER.index(failed_mode)
        candidates = []
        for offset in range(1, len(MODE_UPGRADE_ORDER)):
            upgrade_idx = idx + offset
            if upgrade_idx < len(MODE_UPGRADE_ORDER):
                mode = MODE_UPGRADE_ORDER[upgrade_idx]
                if self.circuit_breakers[mode].is_available():
                    candidates.append(mode)
        for offset in range(1, len(MODE_UPGRADE_ORDER)):
            downgrade_idx = idx - offset
            if downgrade_idx >= 0:
                mode = MODE_UPGRADE_ORDER[downgrade_idx]
                if self.circuit_breakers[mode].is_available():
                    candidates.append(mode)
        return candidates

    def _try_switch_mode(
        self,
        ctx,
        task: str,
        context: Optional[str],
        failed_mode: AgentMode,
        failed_result: Dict,
        max_switches: int = 2,
        **kwargs,
    ) -> Optional[Dict]:
        switches = 0
        error_category = failed_result.get("error_category")

        switch_ctx = SwitchContext(
            failure_reason=failed_result.get("result", "")[:500],
            intermediate_result=failed_result.get("result", "")[:2000],
            source_mode=failed_mode.value,
            error_category=error_category,
            token_usage=failed_result.get("token_usage", 0),
            time_taken=failed_result.get("time_taken", 0),
        )

        candidates = self._smart_switch_strategy(failed_mode, error_category)

        for mode in candidates:
            if switches >= max_switches:
                break

            cooldown_key = f"{failed_mode.value}->{mode.value}"
            last_switch_time = self._switch_cooldown.get(cooldown_key, 0)
            if time.time() - last_switch_time < self._switch_cooldown_seconds:
                self._logger.info("模式切换冷却中: %s，跳过", cooldown_key)
                continue

            switch_ctx.target_mode = mode.value
            switch_kwargs = dict(kwargs)
            switch_kwargs["switch_context"] = switch_ctx
            enhanced_context = context or ""
            if switch_ctx.intermediate_result:
                enhanced_context += f"\n\n[前次模式 {switch_ctx.source_mode} 的中间结果]\n{switch_ctx.intermediate_result}"

            switches += 1
            self._switch_cooldown[cooldown_key] = time.time()
            result = self._execute_mode(ctx, task, enhanced_context, mode, **switch_kwargs)
            if result.get("success"):
                return result

        return None
