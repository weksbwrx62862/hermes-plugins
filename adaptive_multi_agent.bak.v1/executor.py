from __future__ import annotations

# 本模块负责各执行模式的实际调度与运行。
# 包含进度上报、共享记忆、质量评分、事件拓扑、模式切换上下文以及 generator-verifier、
# orchestrator-subagent、agent-teams、message-bus、shared-state、parallel-fusion 等模式的执行实现。

import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .checkpoint import AMACheckpoint, CheckpointState
from .diagnostics import _is_python_exception_string
from .errors import ErrorCategory
from .rolecards import inject_role_to_goal
from .selector import MODE_UPGRADE_ORDER
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
    TerminationGuard,
    MaxMessagesTermination,
    TokenBudgetTermination,
    TimeoutTermination,
    _MODE_CN,
    MODE_CN_SHORT,
    TASK_TYPE_CN,
    get_template_subtasks,
    validate_subtask_dag,
)

logger = logging.getLogger(__name__)

class ProgressReporter:
    """子任务进度上报器。

    在任务执行过程中向 IM 渠道推送进度更新，
    让用户实时看到子智能体的工作状态。
    """

    def __init__(self, ctx=None, enabled: bool = True):
        self._ctx = ctx
        self._enabled = enabled
        self._logger = logging.getLogger("ama.progress")
        self._last_update = 0.0
        self._min_interval = 5.0  # 最小推送间隔（秒），防止刷屏

    def set_ctx(self, ctx):
        """设置 PluginContext（延迟注入）"""
        self._ctx = ctx

    def report(self, stage: str, detail: str, progress: str = ""):
        """推送进度更新到 IM。

        Args:
            stage: 阶段名（如 "生成中", "审核中", "已完成"）
            detail: 详细描述
            progress: 进度信息（如 "2/5"）
        """
        if not self._enabled or not self._ctx:
            return

        now = time.time()
        if now - self._last_update < self._min_interval:
            return  # 防止刷屏

        try:
            emoji = {
                "规划中": "📋", "执行中": "⚙️", "生成中": "✍️",
                "审核中": "🔍", "已完成": "✅", "失败": "❌",
                "重试中": "🔄", "等待中": "⏳",
            }.get(stage, "📢")

            msg = f"{emoji} **{stage}**"
            if progress:
                msg += f" [{progress}]"
            msg += f"\n{detail}"

            self._ctx.dispatch_tool("send_message", {
                "action": "send",
                "message": msg,
            })
            self._last_update = now
            self._logger.info("[progress] %s: %s", stage, detail[:100])
        except Exception as e:
            self._logger.debug("[progress] 推送失败: %s", e)


_progress_reporter = ProgressReporter()


def _classify_error(exception_or_result) -> str:
    """将异常或执行结果统一分类为 ErrorCategory.value。

    分类规则：
    - TimeoutError / 消息含 timeout/timed out/deadline → timeout
    - ValueError / 参数校验失败 / invalid / validation → validation
    - LLM 调用失败 / delegate_task 返回错误 / 模型相关 → llm_error
    - 子代理返回 failed / 异常 → subagent_failure
    - 用户取消 / KeyboardInterrupt / cancel → cancelled
    - 其他 → unknown
    """
    # 字典结果（如子代理返回包、模式执行结果）
    if isinstance(exception_or_result, dict):
        result = exception_or_result
        status = result.get("status", "")
        if status == "cancelled" or result.get("cancelled"):
            return ErrorCategory.cancelled.value
        if status == "timed_out" or status == "timeout" or result.get("timed_out"):
            return ErrorCategory.timeout.value
        if status == "failed" or result.get("failed") or result.get("success") is False:
            msg = str(result.get("result", result.get("error", ""))).lower()
            if "cancel" in msg or "取消" in msg:
                return ErrorCategory.cancelled.value
            if "timeout" in msg or "timed out" in msg or "deadline" in msg:
                return ErrorCategory.timeout.value
            if "validation" in msg or "参数校验" in msg or "invalid" in msg or "参数错误" in msg:
                return ErrorCategory.validation.value
            if "llm" in msg or "delegate_task" in msg or "模型" in msg or "api error" in msg:
                return ErrorCategory.llm_error.value
            if "subagent" in msg or "子代理" in msg:
                return ErrorCategory.subagent_failure.value
            return ErrorCategory.subagent_failure.value
        return ErrorCategory.unknown.value

    # 字符串消息
    if isinstance(exception_or_result, str):
        msg = exception_or_result.lower()
        if "cancel" in msg or "取消" in msg:
            return ErrorCategory.cancelled.value
        if "timeout" in msg or "timed out" in msg or "deadline" in msg:
            return ErrorCategory.timeout.value
        if "validation" in msg or "参数校验" in msg or "invalid" in msg or "参数错误" in msg:
            return ErrorCategory.validation.value
        if "llm" in msg or "delegate_task" in msg or "模型" in msg or "api error" in msg:
            return ErrorCategory.llm_error.value
        if "subagent" in msg or "子代理" in msg:
            return ErrorCategory.subagent_failure.value
        return ErrorCategory.unknown.value

    # 异常对象
    if isinstance(exception_or_result, Exception):
        exc = exception_or_result
        exc_type = type(exc).__name__
        msg = str(exc).lower()

        if isinstance(exc, KeyboardInterrupt) or "cancel" in msg or "取消" in msg:
            return ErrorCategory.cancelled.value
        if isinstance(exc, TimeoutError) or "timeout" in msg or "timed out" in msg or "deadline" in msg:
            return ErrorCategory.timeout.value
        if isinstance(exc, ValueError) or "validation" in msg or "参数校验" in msg or "invalid" in msg or "参数错误" in msg:
            return ErrorCategory.validation.value
        if "llm" in msg or "delegate_task" in msg or "模型" in msg or "api error" in msg:
            return ErrorCategory.llm_error.value
        if "subagent" in msg or "子代理" in msg:
            return ErrorCategory.subagent_failure.value
        return ErrorCategory.unknown.value

    return ErrorCategory.unknown.value


class SharedMemory:
    """子智能体共享记忆池。

    在 orchestrator-subagent 模式下，各子智能体可以：
    - write(key, value): 写入发现/结论
    - read(): 读取所有共享知识
    - context_str(): 生成注入 prompt 的上下文字符串

    借鉴 MetaGPT 的 pub/sub 消息池设计，
    让并行执行的子智能体能共享中间发现，提高任务连贯性。
    """

    def __init__(self):
        self._store: Dict[str, str] = {}
        self._lock = threading.Lock()

    def write(self, key: str, value: str):
        """写入共享记忆"""
        with self._lock:
            self._store[key] = value

    def read(self) -> Dict[str, str]:
        """读取所有共享记忆"""
        with self._lock:
            return dict(self._store)

    def context_str(self) -> str:
        """生成注入 prompt 的上下文字符串"""
        with self._lock:
            if not self._store:
                return ""
            lines = ["【共享记忆池】以下是其他子任务的发现，请参考："]
            for k, v in self._store.items():
                lines.append(f"- {k}: {v[:200]}")
            return "\n".join(lines)

    def summary(self) -> Dict:
        """返回摘要"""
        with self._lock:
            return {"entries": len(self._store), "keys": list(self._store.keys())}


class QualityScorer:
    """结果质量评分器。

    对 generator-verifier 模式的输出进行多维度评分：
    - completeness: 完整性（是否覆盖所有要求）
    - correctness: 正确性（是否有明显错误）
    - clarity: 清晰度（表达是否清楚）
    - relevance: 相关性（是否紧扣任务）

    综合分 = weighted average，用于决定是否需要继续迭代。
    """

    # 各维度权重
    WEIGHTS = {
        "completeness": 0.35,
        "correctness": 0.35,
        "clarity": 0.15,
        "relevance": 0.15,
    }

    # 早期终止阈分（高于此分直接通过）
    EARLY_PASS_THRESHOLD = 85

    @classmethod
    def calculate_score(cls, dimensions: Dict[str, float]) -> float:
        """计算加权综合分"""
        total = 0.0
        weight_sum = 0.0
        for dim, score in dimensions.items():
            w = cls.WEIGHTS.get(dim, 0.1)
            total += score * w
            weight_sum += w
        return total / weight_sum if weight_sum > 0 else 0.0

    @classmethod
    def should_early_pass(cls, score: float) -> bool:
        """是否应该早期终止（分数足够高）"""
        return score >= cls.EARLY_PASS_THRESHOLD

    @classmethod
    def parse_from_verifier(cls, result_str: str) -> Optional[Dict[str, float]]:
        """从 Verifier 输出中解析质量维度分数。

        支持 JSON 格式：
        {"scores": {"completeness": 90, "correctness": 85, "clarity": 80, "relevance": 95}}
        """
        if not result_str:
            return None
        try:
            data = json.loads(result_str)
            if isinstance(data, dict) and "scores" in data:
                scores = data["scores"]
                if isinstance(scores, dict):
                    return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
        except (json.JSONDecodeError, TypeError):
            pass
        return None


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


class ModeExecutor:
    """模式执行器：封装各 AgentMode 的调度与执行逻辑。"""

    def __init__(self, engine):
        self._engine = engine

    def __getattr__(self, name):
        """未在 ModeExecutor 中定义的属性委托给 AdaptiveMultiAgentEngine 实例。"""
        return getattr(self._engine, name)

    def execute_mode(
        self,
        ctx,
        task: str,
        context: Optional[str],
        mode: AgentMode,
        **kwargs,
    ) -> Dict:
        # ── 自动启用 DAG：高复杂度任务或 orchestrator 模式 ──
        complexity_score = self._last_assessment.get("complexity_score", 5.0) if hasattr(self, '_last_assessment') else 5.0
        use_dag = kwargs.get("use_dag", self.config.get("use_dag", False))
        
        # 自动启用 DAG 条件：
        # 1. 配置启用了 DAG
        # 2. 复杂度 >= 6.0 且是 orchestrator 模式
        # 3. 任务描述包含多个子任务信号（数字列表、多步骤等）
        auto_dag = (
            use_dag
            or (complexity_score >= 6.0 and mode == AgentMode.ORCHESTRATOR_SUBAGENT)
            or (mode == AgentMode.ORCHESTRATOR_SUBAGENT and len(re.findall(r'\d+[.)]\s|\d+、', task)) >= 3)
        )
        
        if auto_dag and mode == AgentMode.ORCHESTRATOR_SUBAGENT and not kwargs.get("resume_state"):
            try:
                logger.info("[AMA] 自动启用 DAG 模式 (复杂度=%.1f)", complexity_score)
                return self._execute_with_dag(ctx, task, context, **kwargs)
            except Exception as e:
                logger.warning("DAG 执行失败，降级到传统模式: %s", e)
        
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
        elif mode == AgentMode.PARALLEL_FUSION:
            return self._run_parallel_fusion(ctx, task, context, **kwargs)
        return {"success": False, "result": f"未知模式: {mode}"}

    def resume_from_checkpoint(
        self,
        checkpoint_state: CheckpointState,
        ctx,
        **kwargs,
    ) -> Dict:
        """从检查点恢复执行。

        根据 checkpoint.mode 选择对应执行模式，将 results_so_far 注入上下文，
        并把 checkpoint_state 透传给具体模式方法以跳过 completed_steps。
        """
        try:
            mode = AgentMode(checkpoint_state.mode)
        except (ValueError, TypeError):
            mode = AgentMode.GENERATOR_VERIFIER

        # 将已完成的中间结果注入上下文
        resume_context = kwargs.get("context") or ""
        if checkpoint_state.results_so_far:
            results_text = json.dumps(checkpoint_state.results_so_far, ensure_ascii=False)
            resume_context += f"\n\n【断点恢复：已完成的中间结果】\n{results_text}"

        kwargs["resume_state"] = checkpoint_state
        return self.execute_mode(ctx, checkpoint_state.task, resume_context, mode, **kwargs)


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
                    result.error_category = ErrorCategory.subagent_failure.value
                    result.elapsed_seconds = time.time() - start
                    logger.error(
                        "子代理返回 Python 异常 (mode=%s, goal=%.100s): %s",
                        mode.value, goal, delegate_result[:500],
                    )
                    self._fire_hook("on_failed", result)
                    return result

                # 提取子智能体的实际输出（delegate_task 返回 {"results": [...]} 格式）
                actual_result = self._extract_single_result(delegate_result)
                result.result = actual_result
                result.token_usage = self._extract_token_usage(delegate_result)
                result.status = SubagentStatus.COMPLETED
                result.elapsed_seconds = time.time() - start
                self._fire_hook("on_completed", result)

                # ── gstack 角色注入 + MULTICA 技能记录 ──
                mode_value = kwargs.get("_mode", mode)
                self.skill_registry.record(
                    goal=goal,
                    mode=mode_value if isinstance(mode_value, AgentMode) else mode,
                    result=result,
                )
                return result

            except Exception as e:
                retry_category = self.retry_policy.classify_error(e)
                result.error_category = _classify_error(e)

                if self.retry_policy.should_retry(retry_category) and retries < config.max_retries:
                    retries += 1
                    result.retries_attempted = retries
                    result.status = SubagentStatus.RETRYING
                    wait = self.retry_policy.get_wait_time(retries)
                    logger.info(
                        "子代理重试 %d/%d，等待 %.1fs，错误: %s",
                        retries, config.max_retries, wait, retry_category,
                    )
                    time.sleep(wait)
                    continue

                result.status = SubagentStatus.FAILED
                result.result = str(e)
                result.elapsed_seconds = time.time() - start
                self._fire_hook("on_failed", result)
                return result

        result.status = SubagentStatus.FAILED
        result.error_category = ErrorCategory.subagent_failure.value
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
                        error_category=ErrorCategory.subagent_failure.value,
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


    @staticmethod
    def _extract_single_result(delegate_result_str: str) -> str:
        """从 delegate_task 的 {"results": [...]} 响应中提取单个子智能体的实际输出。

        delegate_task 返回格式:
        {"results": [{"status": "completed", "result": "实际输出", ...}]}
        """
        try:
            data = json.loads(delegate_result_str)
            if isinstance(data, dict) and "results" in data:
                results = data["results"]
                if isinstance(results, list) and len(results) > 0:
                    first = results[0]
                    if isinstance(first, dict):
                        return first.get("result") or first.get("summary") or delegate_result_str
        except (json.JSONDecodeError, TypeError):
            pass
        return delegate_result_str


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
        """生成-验证模式（Adversarial Agent Team 风格）

        Generator 和 Verifier 使用独立的角色配置：
        - Generator: 专注于产出，不自我审查
        - Verifier: 独立审核，只找问题不帮忙修改

        增强特性（借鉴 ChatDev/SWE-agent）：
        - 质量评分：Verifier 输出多维度分数 (completeness/correctness/clarity/relevance)
        - 早期终止：分数 ≥ 85 直接通过，节省 token
        - 结构化反馈：带评分的反馈帮助 Generator 精准改进
        - 断点恢复：每轮生成/验证后保存 checkpoint

        状态流转：INIT → PLANNING → EXECUTING → REVIEWING → (REVISING → EXECUTING → REVIEWING)* → DONE/FAILED
        """
        from .subagent import _GENERATOR_CONFIG, _VERIFIER_CONFIG, TaskStateMachine, TaskState

        start_time = time.time()
        tokens_used = 0
        max_iterations = 5
        result_text = ""
        success = False
        all_issues = []
        quality_scores = []  # 记录每轮质量分数
        error_category: Optional[str] = None

        # ── 断点恢复状态注入 ──
        resume_state = kwargs.get("resume_state")
        if isinstance(resume_state, CheckpointState):
            trace_id = resume_state.trace_id or str(uuid.uuid4())
            result_text = resume_state.results_so_far.get("result_text", "")
            all_issues = resume_state.results_so_far.get("all_issues", [])
            quality_scores = resume_state.results_so_far.get("quality_scores", [])
            completed_steps = list(resume_state.completed_steps)
            # 从 pending_steps 推导应从哪一轮继续（round_num 是 checkpoint 序号，不能当迭代索引）
            start_i = self._resume_start_index(completed_steps, resume_state.pending_steps)
        else:
            trace_id = kwargs.get("trace_id") or str(uuid.uuid4())
            start_i = 0
            completed_steps = []

        # 用于 checkpoint 的元信息
        complexity_score = self._last_assessment.get("complexity_score", 5.0) if hasattr(self, '_last_assessment') else 5.0
        task_type = self._last_assessment.get("task_type", "default") if hasattr(self, '_last_assessment') else "default"

        # 初始化状态机 + 进度上报
        task_id = str(uuid.uuid4())
        sm = TaskStateMachine(task_id)
        _progress_reporter.set_ctx(ctx)

        def _on_state_change(from_s, to_s, tid):
            _progress_reporter.report(to_s.cn, f"任务状态: {from_s.cn} → {to_s.cn}")

        sm._on_transition = _on_state_change

        # 构建阶段计划
        pending_steps = []
        for i in range(max_iterations):
            pending_steps.append(f"generate_{i}")
            pending_steps.append(f"verify_{i}")
        pending_steps = [s for s in pending_steps if s not in completed_steps]

        def _save_checkpoint(round_num: int, extra_results: Optional[Dict] = None) -> None:
            results_so_far = {
                "result_text": result_text,
                "all_issues": all_issues,
                "quality_scores": quality_scores,
            }
            if extra_results:
                results_so_far.update(extra_results)
            AMACheckpoint.save(
                trace_id=trace_id,
                round_num=round_num,
                task=task,
                mode=AgentMode.GENERATOR_VERIFIER.value,
                task_type=task_type,
                complexity_score=complexity_score,
                completed_steps=completed_steps,
                pending_steps=pending_steps,
                results_so_far=results_so_far,
            )

        # INIT → PLANNING
        sm.transition(TaskState.PLANNING)

        # 非恢复场景保存初始 checkpoint
        if not isinstance(resume_state, CheckpointState):
            _save_checkpoint(round_num=0)

        for i in range(start_i, max_iterations):
            # PLANNING → EXECUTING
            sm.transition(TaskState.EXECUTING)

            step_gen = f"generate_{i}"
            gen_result = result_text  # 若本步生成被跳过，沿用已有结果

            if step_gen not in completed_steps:
                # ── Generator 阶段 ──────────────────────────
                _progress_reporter.report(
                    "生成中", f"任务: {task[:60]}...", f"{i+1}/{max_iterations}"
                )

                gen_goal = f"【生成任务】\n\n{task}"
                if i > 0 and all_issues:
                    # 带上 Verifier 的反馈要求改进（结构化反馈）
                    issues_text = "\n".join(f"- {iss}" for iss in all_issues[-1])
                    gen_goal += f"\n\n【审核反馈】请严格按以下问题改进：\n{issues_text}"
                    if quality_scores:
                        last_score = quality_scores[-1]
                        gen_goal += f"\n\n【质量评分】上轮综合分: {last_score['total']:.0f}/100"
                        for dim, val in last_score.get("dimensions", {}).items():
                            gen_goal += f"\n  - {dim}: {val:.0f}"
                        gen_goal += "\n请重点提升低分维度。"
                if context:
                    gen_goal += f"\n\n【上下文】{context}"

                # ── 注入 Builder 角色上下文（借鉴 gstack）──
                gen_goal = inject_role_to_goal(gen_goal, role_id="builder")

                gen_sr = self._execute_subagent(
                    ctx, gen_goal, context=context,
                    config=_GENERATOR_CONFIG,
                    _mode=AgentMode.GENERATOR_VERIFIER, trace_id=trace_id, **kwargs,
                )
                tokens_used += gen_sr.token_usage
                if gen_sr.status != SubagentStatus.COMPLETED:
                    sm.transition(TaskState.FAILED)
                    _progress_reporter.report("失败", f"生成阶段失败: {gen_sr.status.value}")
                    error_category = gen_sr.error_category or ErrorCategory.subagent_failure.value
                    _save_checkpoint(round_num=i + 1, extra_results={"error": error_category})
                    AMACheckpoint.mark_interrupted(trace_id, error_category)
                    break
                gen_result = gen_sr.result or ""
                result_text = gen_result

                completed_steps.append(step_gen)
                pending_steps.remove(step_gen)
                _save_checkpoint(round_num=i + 1)

            # EXECUTING → REVIEWING
            sm.transition(TaskState.REVIEWING)

            step_verify = f"verify_{i}"
            if step_verify not in completed_steps:
                # ── Verifier 阶段（独立审核 + 质量评分）────────────────
                _progress_reporter.report(
                    "审核中", f"审核第 {i+1} 轮产出...", f"{i+1}/{max_iterations}"
                )

                verify_goal = (
                    f"【审核任务】\n\n"
                    f"原始任务要求：\n{task}\n\n"
                    f"待审核产出：\n{gen_result}\n\n"
                    f"请严格审核以上产出是否满足任务要求。\n"
                    f"审核完成后，请返回以下 JSON 格式（必须包含 scores 字段）：\n"
                    f'```json\n{{"passed": true/false, "feedback": "审核意见", "issues": ["问题1", ...], '
                    f'"scores": {{"completeness": 0-100, "correctness": 0-100, "clarity": 0-100, "relevance": 0-100}}}}\n```'
                )
                if context:
                    verify_goal += f"\n\n【任务上下文】{context}"

                # ── 注入 Reviewer 角色上下文（借鉴 gstack）──
                verify_goal = inject_role_to_goal(verify_goal, role_id="reviewer")

                verify_sr = self._execute_subagent(
                    ctx, verify_goal, context=None,  # Verifier 不共享上下文，保持独立性
                    config=_VERIFIER_CONFIG,
                    _mode=AgentMode.GENERATOR_VERIFIER, trace_id=trace_id, **kwargs,
                )
                tokens_used += verify_sr.token_usage

                # 解析验证结果
                passed, feedback, issues = self._parse_verification_result_v2(verify_sr.result or "")
                result_text = gen_result
                all_issues.append(issues)

                # 解析质量评分（如果有）
                dimensions = QualityScorer.parse_from_verifier(verify_sr.result or "")
                if dimensions:
                    total_score = QualityScorer.calculate_score(dimensions)
                    quality_scores.append({"total": total_score, "dimensions": dimensions})

                    logger.info(
                        "[generator_verifier] 迭代 %d/%d | passed=%s | quality=%.0f | issues=%d | dims=%s",
                        i + 1, max_iterations, passed, total_score, len(issues),
                        {k: f"{v:.0f}" for k, v in dimensions.items()},
                    )

                    # 早期终止：分数足够高直接通过（即使 Verifier 没说 passed）
                    if not passed and QualityScorer.should_early_pass(total_score):
                        logger.info(
                            "[generator_verifier] 质量分 %.0f ≥ %d，早期终止",
                            total_score, QualityScorer.EARLY_PASS_THRESHOLD,
                        )
                        passed = True
                else:
                    logger.info(
                        "[generator_verifier] 迭代 %d/%d | passed=%s | issues=%d | feedback=%.100s",
                        i + 1, max_iterations, passed, len(issues), feedback,
                    )

                completed_steps.append(step_verify)
                pending_steps.remove(step_verify)

                if passed:
                    # REVIEWING → DONE
                    sm.transition(TaskState.DONE)
                    score_info = ""
                    if quality_scores:
                        score_info = f" | 质量分: {quality_scores[-1]['total']:.0f}/100"
                    _progress_reporter.report(
                        "已完成", f"通过审核（第 {i+1} 轮）{score_info}", f"{i+1}/{max_iterations}"
                    )
                    success = True
                    _save_checkpoint(round_num=i + 1)
                    AMACheckpoint.mark_completed(trace_id)
                    break
                else:
                    # REVIEWING → REVISING → EXECUTING (下一轮)
                    sm.transition(TaskState.REVISING)
                    _progress_reporter.report(
                        "重试中", f"审核未通过: {feedback[:80]}...", f"{i+1}/{max_iterations}"
                    )
                    _save_checkpoint(round_num=i + 1)

        if not success:
            sm.transition(TaskState.FAILED)
            _progress_reporter.report("失败", f"达到最大迭代次数 ({max_iterations})")
            if not error_category:
                error_category = ErrorCategory.subagent_failure.value
            _save_checkpoint(round_num=max_iterations)
            AMACheckpoint.mark_interrupted(trace_id, error_category)

        return {
            "result": result_text,
            "success": success,
            "token_usage": tokens_used,
            "time_taken": time.time() - start_time,
            "mode": AgentMode.GENERATOR_VERIFIER.value,
            "metadata": {
                "iterations": i + 1,
                "converged": success,
                "adversarial": True,  # 标记使用了对抗式独立角色
                "all_issues": all_issues,
                "task_state": sm.summary(),  # 状态机摘要
            },
            "task_id": gen_sr.task_id if 'gen_sr' in locals() else task_id,
            "trace_id": trace_id,
            "status": SubagentStatus.COMPLETED.value if success else SubagentStatus.FAILED.value,
            "error_category": error_category,
            "retries_attempted": gen_sr.retries_attempted if 'gen_sr' in locals() else 0,
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


    @staticmethod
    def _resume_start_index(completed_steps: List[str], pending_steps: List[str]) -> int:
        """根据已完成/待执行步骤推导恢复起始轮次。

        将 generate_i / verify_i / subtask_i / synthesize 等步骤名映射为迭代索引，
        优先从 pending_steps 取最小轮次；无 pending 时从 completed_steps 取最大轮次+1。
        """
        import re

        def _step_index(step: str) -> Optional[int]:
            m = re.match(r"^(generate|verify|subtask|round|layer|aggregate|synthesize|finalize|process|respond)(?:_|$)(\d+)?", step)
            if not m:
                return None
            num = m.group(2)
            return int(num) if num is not None else 0

        indices = [_step_index(s) for s in pending_steps if _step_index(s) is not None]
        if indices:
            return min(indices)
        indices = [_step_index(s) for s in completed_steps if _step_index(s) is not None]
        return max(indices) + 1 if indices else 0

    @staticmethod
    def _parse_verification_result_v2(result_str: str) -> tuple:
        """解析 Verifier 的审核结果（v2 格式，支持 issues 列表）。

        Returns:
            (passed, feedback, issues)
        """
        if not result_str or _is_python_exception_string(result_str):
            return False, result_str or "", []

        # 尝试 JSON 解析
        try:
            data = json.loads(result_str)
            if isinstance(data, dict):
                passed = bool(data.get("passed", False))
                feedback = data.get("feedback", "")
                issues = data.get("issues", [])
                if isinstance(issues, str):
                    issues = [issues]
                return passed, feedback, issues
        except (json.JSONDecodeError, TypeError):
            pass

        # 降级：从文本中判断
        # 先检查否定形式
        negated = any(kw in result_str for kw in ["不通过", "未通过", "未pass", "not pass", "failed"])
        if negated:
            passed = False
        else:
            passed = any(kw in result_str for kw in ["通过", "满意", "pass", "approved"])
            if not passed:
                passed = "正确" in result_str and "不正确" not in result_str and "错误" not in result_str

        # 提取可能的问题列表
        issues = []
        for line in result_str.split("\n"):
            line = line.strip()
            if line.startswith(("- ", "• ", "* ", "1.", "2.", "3.", "4.", "5.")):
                issues.append(line.lstrip("-•*0123456789. "))

        return passed, result_str, issues


    def _execute_with_dag(
        self,
        ctx,
        task: str,
        context: Optional[str],
        **kwargs,
    ) -> Dict:
        """使用 DAG 编排器执行任务（增强版：per-subtask 模式选择）
        
        优势：
          - 并行执行无依赖任务
          - 自动拓扑排序
          - 故障隔离
          - 每个子任务独立评估复杂度并选择最优模式
        """
        from .dag_orchestrator import DAGOrchestrator
        
        logger.info("使用 DAG 编排器执行任务: %s", task[:50])
        
        # 创建 DAG 编排器
        dag = DAGOrchestrator(max_workers=self.config.get("dag_max_workers", 4))
        
        # ── 增强任务分解：依赖分析 + per-subtask 复杂度评估 ──
        complexity_score = self._last_assessment.get("complexity_score", 5.0) if hasattr(self, '_last_assessment') else 5.0
        decompose_goal = (
            f"将以下任务分解为 2-4 个子任务，并分析依赖关系和每个子任务的复杂度。\n\n"
            f"## 要求\n"
            f"1. 每个子任务用一句话描述\n"
            f"2. 标注子任务间的依赖关系（哪些子任务需要等其他子任务完成）\n"
            f"3. 评估每个子任务的复杂度（1-10分）\n"
            f"4. 为每个子任务推荐执行模式\n\n"
            f"## 可选模式\n"
            f"- generator_verifier: 简单生成+验证（复杂度≤3）\n"
            f"- orchestrator_subagent: 中等复杂，需要多步骤（复杂度4-6）\n"
            f"- agent_teams: 需要多角色协作（复杂度7-8）\n"
            f"- message_bus: 事件驱动/异步处理（复杂度≥8）\n"
            f"- shared_state: 需要共享知识库（复杂度≥8）\n\n"
            f"## 输出格式（严格 JSON，不要添加其他内容）\n"
            f'```json\n{{"subtasks": [{{"id": "1", "description": "子任务描述", "dependencies": [], "complexity": 5.0, "recommended_mode": "orchestrator_subagent", "reasoning": "50字以内推荐理由"}}, ...]}}\n```\n\n'
            f"## 任务\n"
            f"整体复杂度: {complexity_score:.1f}/10\n"
            f"任务: {task}"
        )
        if context:
            decompose_goal += f"\n\n上下文: {context}"
        
        # 使用子代理分解任务
        decompose_sr = self._execute_subagent(
            ctx, decompose_goal,
            _mode=AgentMode.ORCHESTRATOR_SUBAGENT, **kwargs,
        )
        
        subtasks = self._parse_subtasks_json(
            decompose_sr.result or "",
            task_type=self._last_assessment.get("task_type", "default") if hasattr(self, '_last_assessment') else "default"
        )
        
        if not subtasks:
            raise ValueError("任务分解失败")
        
        # ── 为每个子任务创建 DAG 节点（per-subtask 模式选择 + 依赖感知）──
        for i, subtask in enumerate(subtasks):
            subtask_desc = subtask.get("description", str(subtask))
            subtask_mode = subtask.get("recommended_mode", "orchestrator_subagent")
            subtask_deps = subtask.get("dependencies", [])
            node_name = subtask.get("id", f"subtask_{i}")
            
            # 定义节点处理函数（per-subtask 模式选择）
            def make_handler(subtask_desc: str, subtask_mode: str):
                def handler(ctx, task: str, context: Optional[str] = None, **kwargs):
                    # 选择子任务推荐的模式
                    try:
                        mode = AgentMode(subtask_mode)
                    except ValueError:
                        mode = AgentMode.ORCHESTRATOR_SUBAGENT
                    
                    # 执行子任务
                    result = self._execute_subagent(
                        ctx, subtask_desc,
                        context=f"这是大任务的一部分: {task}",
                        _mode=mode, **kwargs,
                    )
                    return {"success": result.status == SubagentStatus.COMPLETED, "result": result.result}
                return handler
            
            # 添加节点（支持依赖关系）
            dag.add_node(
                node_name,
                make_handler(subtask_desc, subtask_mode),
                dependencies=subtask_deps,
            )
        
        # 执行 DAG
        dag_result = dag.execute(ctx, task, context, **kwargs)
        
        # 汇总结果
        success = dag_result.success
        results = dag_result.results
        errors = dag_result.errors
        
        # 合并所有子任务结果
        all_results = []
        for node_name, result in results.items():
            if result and result.get("success"):
                all_results.append(result.get("result", ""))
        
        combined_result = "\n\n".join(all_results) if all_results else "所有子任务执行失败"
        
        return {
            "success": success,
            "result": combined_result,
            "mode": AgentMode.ORCHESTRATOR_SUBAGENT.value,  # DAG 是 orchestrator 的变体
            "subtask_count": len(subtasks),
            "execution_order": dag_result.execution_order,
            "parallel_groups": dag_result.parallel_groups,
        }


    def _run_orchestrator_subagent(
        self, ctx, task: str, context: Optional[str], **kwargs
    ) -> Dict:
        """协调-子代理模式（增强版：per-subtask 模式选择）

        增强特性（借鉴 MetaGPT 消息池 + CAMEL Workforce）：
        - 共享记忆池：子智能体完成后提取关键发现，供后续综合阶段参考
        - 上下文传递：综合阶段能看到所有子任务的关键发现
        - per-subtask 模式选择：根据子任务复杂度自动选择最优模式
        - 依赖感知：支持子任务间的依赖关系
        - 断点恢复：decompose/每个子任务/synthesize 阶段后保存 checkpoint
        """
        start_time = time.time()
        tokens_used = 0

        # ── 断点恢复状态注入 ──
        resume_state = kwargs.get("resume_state")
        if isinstance(resume_state, CheckpointState):
            trace_id = resume_state.trace_id or str(uuid.uuid4())
            completed_steps = list(resume_state.completed_steps)
            subtasks = resume_state.results_so_far.get("subtasks", [])
            subtask_results = resume_state.results_so_far.get("subtask_results", {})
        else:
            trace_id = kwargs.get("trace_id") or str(uuid.uuid4())
            completed_steps = []
            subtasks = []
            subtask_results = {}

        # 初始化共享记忆池，并恢复已完成的子任务结果
        shared_mem = SharedMemory()
        for sid, res in subtask_results.items():
            self._extract_findings_to_memory(shared_mem, sid, res)

        # checkpoint 元信息
        complexity_score = self._last_assessment.get("complexity_score", 5.0) if hasattr(self, '_last_assessment') else 5.0
        task_type = self._last_assessment.get("task_type", "default") if hasattr(self, '_last_assessment') else "default"

        def _save_checkpoint(round_num: int, extra_results: Optional[Dict] = None) -> None:
            results_so_far = {
                "subtasks": subtasks,
                "subtask_results": subtask_results,
            }
            if extra_results:
                results_so_far.update(extra_results)
            AMACheckpoint.save(
                trace_id=trace_id,
                round_num=round_num,
                task=task,
                mode=AgentMode.ORCHESTRATOR_SUBAGENT.value,
                task_type=task_type,
                complexity_score=complexity_score,
                completed_steps=completed_steps,
                pending_steps=pending_steps,
                results_so_far=results_so_far,
            )

        def _build_pending_steps() -> List[str]:
            steps = [f"subtask_{st.get('id', str(i))}" for i, st in enumerate(subtasks)]
            steps.append("synthesize")
            return [s for s in steps if s not in completed_steps]

        pending_steps: List[str] = []

        # ── 增强任务分解：依赖分析 + per-subtask 复杂度评估 ──
        step_decompose = "decompose"
        if step_decompose not in completed_steps:
            workflow_info = ""
            if self.current_workflow:
                wf = self.current_workflow
                stages_desc = "\n".join(f"  {i+1}. {s.name}（角色: {s.role_id}）" for i, s in enumerate(wf.stages))
                workflow_info = f"\n## 工作流模板（参考）\n名称: {wf.name}\n阶段:\n{stages_desc}\n\n可参考以上工作流结构来规划子任务，但请根据实际需求灵活调整。\n"
            decompose_goal = (
                f"将以下任务分解为 2-4 个子任务，并分析依赖关系和每个子任务的复杂度。\n\n"
                f"## 要求\n"
                f"1. 每个子任务用一句话描述\n"
                f"2. 标注子任务间的依赖关系（哪些子任务需要等其他子任务完成）\n"
                f"3. 评估每个子任务的复杂度（1-10分）\n"
                f"4. 为每个子任务推荐执行模式\n\n"
                f"## 可选模式\n"
                f"- generator_verifier: 简单生成+验证（复杂度≤3）\n"
                f"- orchestrator_subagent: 中等复杂，需要多步骤（复杂度4-6）\n"
                f"- agent_teams: 需要多角色协作（复杂度7-8）\n"
                f"- message_bus: 事件驱动/异步处理（复杂度≥8）\n"
                f"- shared_state: 需要共享知识库（复杂度≥8）\n\n"
                f"## 输出格式（严格 JSON，不要添加其他内容）\n"
                f'```json\n{{"subtasks": [{{"id": "1", "description": "子任务描述", "dependencies": [], "complexity": 5.0, "recommended_mode": "orchestrator_subagent", "reasoning": "50字以内推荐理由"}}, ...]}}\n```\n\n'
                f"## 任务\n"
                f"整体复杂度: {complexity_score:.1f}/10\n"
                f"任务: {task}"
            )
            if context:
                decompose_goal += f"\n\n上下文: {context}"
            decompose_goal += workflow_info

            decompose_sr = self._execute_subagent(
                ctx, decompose_goal,
                _mode=AgentMode.ORCHESTRATOR_SUBAGENT, trace_id=trace_id, **kwargs,
            )
            tokens_used += decompose_sr.token_usage
            subtasks = self._parse_subtasks_json(decompose_sr.result or "", task_type=task_type)
            completed_steps.append(step_decompose)
            pending_steps = _build_pending_steps()
            _save_checkpoint(round_num=1)
        else:
            pending_steps = _build_pending_steps()

        # ── per-subtask 模式选择 + 依赖感知执行 ──
        # 按依赖关系分组：无依赖的并行执行，有依赖的串行执行
        no_deps = [st for st in subtasks if not st.get("dependencies")]
        has_deps = [st for st in subtasks if st.get("dependencies")]

        # 执行无依赖的子任务
        if no_deps:
            for st in no_deps:
                step_name = f"subtask_{st.get('id', '')}"
                if step_name in completed_steps:
                    continue
                st_mode = st.get("recommended_mode", "orchestrator_subagent")
                shared_ctx = shared_mem.context_str()
                try:
                    mode = AgentMode(st_mode)
                except ValueError:
                    mode = AgentMode.ORCHESTRATOR_SUBAGENT
                sr = self._execute_subagent(
                    ctx, st.get("description", str(st)),
                    context=f"这是大任务的一部分: {task}\n\n{shared_ctx}\n\n完成子任务后，请在结果末尾用以下格式提取关键发现：\n[FINDINGS] key1=value1; key2=value2; ...",
                    _mode=mode, trace_id=trace_id, **kwargs,
                )
                tokens_used += sr.token_usage
                result_text = sr.result or ""
                st_id = st.get("id", "")
                subtask_results[st_id] = result_text
                self._extract_findings_to_memory(shared_mem, st_id, result_text)
                completed_steps.append(step_name)
                pending_steps.remove(step_name)
                _save_checkpoint(round_num=2)

        # 执行有依赖的子任务（串行，按依赖顺序）
        if has_deps:
            sorted_deps = self._topological_sort_subtasks(has_deps)
            for st in sorted_deps:
                step_name = f"subtask_{st.get('id', '')}"
                if step_name in completed_steps:
                    continue
                st_mode = st.get("recommended_mode", "orchestrator_subagent")
                dep_results = []
                for dep_id in st.get("dependencies", []):
                    if dep_id in subtask_results:
                        dep_results.append(f"【依赖任务 {dep_id} 的结果】\n{subtask_results[dep_id][:500]}")
                dep_ctx = "\n\n".join(dep_results) if dep_results else ""
                shared_ctx = shared_mem.context_str()
                try:
                    mode = AgentMode(st_mode)
                except ValueError:
                    mode = AgentMode.ORCHESTRATOR_SUBAGENT
                sr = self._execute_subagent(
                    ctx, st.get("description", str(st)),
                    context=f"这是大任务的一部分: {task}\n\n{dep_ctx}\n\n{shared_ctx}\n\n完成子任务后，请在结果末尾用以下格式提取关键发现：\n[FINDINGS] key1=value1; key2=value2; ...",
                    _mode=mode, trace_id=trace_id, **kwargs,
                )
                tokens_used += sr.token_usage
                result_text = sr.result or ""
                st_id = st.get("id", "")
                subtask_results[st_id] = result_text
                self._extract_findings_to_memory(shared_mem, st_id, result_text)
                completed_steps.append(step_name)
                pending_steps.remove(step_name)
                _save_checkpoint(round_num=3)

        # 综合阶段：注入共享记忆
        step_synth = "synthesize"
        if step_synth not in completed_steps:
            shared_ctx_final = shared_mem.context_str()
            synth_goal = (
                f"综合以下子任务结果，完成最终任务。\n\n"
                f"任务: {task}\n\n"
                f"子任务结果:\n"
            )
            for sid, res in subtask_results.items():
                synth_goal += f"\n--- 子任务 {sid} ---\n{res}\n"

            if shared_ctx_final:
                synth_goal += f"\n\n{shared_ctx_final}"

            synth_sr = self._execute_subagent(
                ctx, synth_goal,
                _mode=AgentMode.ORCHESTRATOR_SUBAGENT, trace_id=trace_id, **kwargs,
            )
            tokens_used += synth_sr.token_usage
            completed_steps.append(step_synth)
            pending_steps.remove(step_synth)
            _save_checkpoint(round_num=4, extra_results={"final_result": synth_sr.result or ""})
            AMACheckpoint.mark_completed(trace_id)
            return {
                "result": synth_sr.result or "",
                "success": bool(synth_sr.result) and synth_sr.status == SubagentStatus.COMPLETED,
                "token_usage": tokens_used,
                "time_taken": time.time() - start_time,
                "mode": AgentMode.ORCHESTRATOR_SUBAGENT.value,
                "metadata": {
                    "subtasks_executed": len(subtasks),
                    "subtask_ids": list(subtask_results.keys()),
                    "shared_memory": shared_mem.summary(),
                },
                "task_id": getattr(decompose_sr, 'task_id', '') if 'decompose_sr' in locals() else '',
                "trace_id": trace_id,
                "status": SubagentStatus.COMPLETED.value,
                "error_category": None,
                "retries_attempted": getattr(decompose_sr, 'retries_attempted', 0) if 'decompose_sr' in locals() else 0,
            }

        # 若 synthesize 已完成（恢复场景），直接返回已保存结果
        final_result = resume_state.results_so_far.get("final_result", "") if isinstance(resume_state, CheckpointState) else ""
        AMACheckpoint.mark_completed(trace_id)
        return {
            "result": final_result,
            "success": bool(final_result),
            "token_usage": tokens_used,
            "time_taken": time.time() - start_time,
            "mode": AgentMode.ORCHESTRATOR_SUBAGENT.value,
            "metadata": {
                "subtasks_executed": len(subtasks),
                "subtask_ids": list(subtask_results.keys()),
                "shared_memory": shared_mem.summary(),
                "resumed": True,
            },
            "task_id": "",
            "trace_id": trace_id,
            "status": SubagentStatus.COMPLETED.value,
            "error_category": None,
            "retries_attempted": 0,
        }


    @staticmethod
    def _topological_sort_subtasks(subtasks: List[Dict]) -> List[Dict]:
        """对有依赖的子任务进行拓扑排序（Kahn 算法）
        
        Args:
            subtasks: 有依赖关系的子任务列表
            
        Returns:
            拓扑排序后的子任务列表
        """
        from collections import defaultdict, deque
        
        # 构建图和入度表
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        task_map = {st.get("id", str(i)): st for i, st in enumerate(subtasks)}
        
        for st in subtasks:
            st_id = st.get("id", str(subtasks.index(st)))
            in_degree[st_id] = len(st.get("dependencies", []))
            for dep_id in st.get("dependencies", []):
                graph[dep_id].append(st_id)
        
        # BFS 拓扑排序
        queue = deque([st_id for st_id in in_degree if in_degree[st_id] == 0])
        sorted_ids = []
        
        while queue:
            current = queue.popleft()
            sorted_ids.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 返回排序后的子任务
        return [task_map[st_id] for st_id in sorted_ids if st_id in task_map]


    @staticmethod
    def _extract_findings_to_memory(mem: SharedMemory, task_id: str, result: str):
        """从子任务结果中提取关键发现写入共享记忆。

        支持两种格式：
        1. [FINDINGS] key1=value1; key2=value2; ...
        2. 自动提取：取结果的前 200 字符作为摘要
        """
        if not result:
            return

        # 格式 1：显式 [FINDINGS] 标记
        findings_match = re.search(r'\[FINDINGS\]\s*(.+)', result, re.DOTALL)
        if findings_match:
            findings_text = findings_match.group(1).strip()
            for pair in findings_text.split(";"):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    mem.write(f"task_{task_id}_{k.strip()}", v.strip())
            return

        # 格式 2：自动摘要（取前 200 字符）
        summary = result[:200].replace("\n", " ").strip()
        if summary:
            mem.write(f"task_{task_id}_summary", summary)


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
                    # 从原始数据中提取额外字段（complexity, recommended_mode, reasoning）
                    result = []
                    for i, st in enumerate(subtask_items):
                        raw = raw_items[i] if i < len(raw_items) else {}
                        entry = {
                            "id": st.id,
                            "description": st.description,
                            "dependencies": st.dependencies,
                            "expected_output": st.expected_output,
                        }
                        # 添加新增字段（如果有）
                        if isinstance(raw, dict):
                            if "complexity" in raw:
                                entry["complexity"] = raw["complexity"]
                            if "recommended_mode" in raw:
                                entry["recommended_mode"] = raw["recommended_mode"]
                            if "reasoning" in raw:
                                entry["reasoning"] = raw["reasoning"]
                        result.append(entry)
                    return result
                logger.warning("子任务 DAG 校验失败: %s，使用模板", dag_errors)
            else:
                logger.warning("子任务字段校验失败: %s，使用模板", field_errors)

        template = get_template_subtasks(task_type)
        logger.info("使用 %s 类型的模板子任务", task_type)
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
        trace_id = kwargs.get("trace_id") or str(uuid.uuid4())

        # checkpoint 元信息
        complexity_score = self._last_assessment.get("complexity_score", 5.0) if hasattr(self, '_last_assessment') else 5.0
        task_type = self._last_assessment.get("task_type", "default") if hasattr(self, '_last_assessment') else "default"
        AMACheckpoint.save(
            trace_id=trace_id,
            round_num=0,
            task=task,
            mode=AgentMode.AGENT_TEAMS.value,
            task_type=task_type,
            complexity_score=complexity_score,
            completed_steps=[],
            pending_steps=["plan", "execute", "review", "finalize"],
            results_so_far={},
        )

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
        success = bool(final_result) and all(
            sr.status == SubagentStatus.COMPLETED for sr in parallel_results
        ) and pm_sr.status == SubagentStatus.COMPLETED
        if success:
            AMACheckpoint.mark_completed(trace_id)
        else:
            AMACheckpoint.mark_interrupted(trace_id, "agent_teams_failed")

        return {
            "result": final_result,
            "success": success,
            "token_usage": tokens_used,
            "time_taken": time.time() - start_time,
            "mode": AgentMode.AGENT_TEAMS.value,
            "metadata": {
                "team_size": 3,
                "team_members": list(shared_results.keys()),
            },
            "task_id": pm_sr.task_id,
            "trace_id": trace_id,
            "status": SubagentStatus.COMPLETED.value if success else SubagentStatus.FAILED.value,
            "error_category": None,
            "retries_attempted": pm_sr.retries_attempted,
        }


    def _run_message_bus(
        self, ctx, task: str, context: Optional[str], **kwargs
    ) -> Dict:
        start_time = time.time()
        tokens_used = 0
        trace_id = kwargs.get("trace_id") or str(uuid.uuid4())

        # checkpoint 元信息
        complexity_score = self._last_assessment.get("complexity_score", 5.0) if hasattr(self, '_last_assessment') else 5.0
        task_type = self._last_assessment.get("task_type", "default") if hasattr(self, '_last_assessment') else "default"
        AMACheckpoint.save(
            trace_id=trace_id,
            round_num=0,
            task=task,
            mode=AgentMode.MESSAGE_BUS.value,
            task_type=task_type,
            complexity_score=complexity_score,
            completed_steps=[],
            pending_steps=["plan_topology", "process_events", "aggregate"],
            results_so_far={},
        )

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
        success = event_count > 0
        if success:
            AMACheckpoint.mark_completed(trace_id)
        else:
            AMACheckpoint.mark_interrupted(trace_id, "message_bus_no_events")

        return {
            "result": final_result,
            "success": success,
            "token_usage": tokens_used,
            "time_taken": time.time() - start_time,
            "mode": AgentMode.MESSAGE_BUS.value,
            "metadata": {
                "events_processed": event_count,
                "subscribers": list(subscribers.keys()),
            },
            "task_id": "",
            "trace_id": trace_id,
            "status": SubagentStatus.COMPLETED.value if success else SubagentStatus.FAILED.value,
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
                    logger.warning("LLM 生成的拓扑校验失败: %s", errors)
                else:
                    cycles = _detect_topology_cycle(data.get("transitions", {}))
                    if cycles:
                        logger.warning("拓扑存在环: %s，尝试断环", cycles)
                        data["transitions"] = _break_topology_cycle(data.get("transitions", {}), cycles)
                    return data
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        task_type = kwargs.get("_task_type", "default")
        template = TEMPLATE_TOPOLOGIES.get(task_type, DEFAULT_EVENT_TOPOLOGY)
        logger.info("使用模板拓扑: %s", task_type)
        return template


    def _run_shared_state(
        self, ctx, task: str, context: Optional[str], **kwargs
    ) -> Dict:
        start_time = time.time()
        tokens_used = 0
        trace_id = kwargs.get("trace_id") or str(uuid.uuid4())

        # checkpoint 元信息
        complexity_score = self._last_assessment.get("complexity_score", 5.0) if hasattr(self, '_last_assessment') else 5.0
        task_type = self._last_assessment.get("task_type", "default") if hasattr(self, '_last_assessment') else "default"
        AMACheckpoint.save(
            trace_id=trace_id,
            round_num=0,
            task=task,
            mode=AgentMode.SHARED_STATE.value,
            task_type=task_type,
            complexity_score=complexity_score,
            completed_steps=[],
            pending_steps=["round_{i}/{agent}" for i in range(6) for agent in ["explorer", "analyzer", "synthesizer", "validator"]] + ["finalize"],
            results_so_far={},
        )

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
        if converged:
            AMACheckpoint.mark_completed(trace_id)
        else:
            AMACheckpoint.mark_interrupted(trace_id, "shared_state_not_converged")

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


    def _run_parallel_fusion(
        self, ctx, task: str, context: Optional[str], **kwargs
    ) -> Dict:
        """并行融合模式：多层 MoA 架构，参考模型并行回答 + 迭代优化 + 裁决综合。

        基于 Mixture-of-Agents (MoA) 方法论（arXiv:2406.04692）：
        Layer 1: 参考模型并行生成多样化回答
        Layer 2: 参考模型基于 Layer 1 结果优化回答（迭代改进层）
        Layer 3: 裁决模型综合分析所有回答，生成最终输出

        关键发现：多层迭代 > 单层 + 更强裁决
        """
        start_time = time.time()
        tokens_used = 0
        trace_id = kwargs.get("trace_id") or str(uuid.uuid4())

        # checkpoint 元信息
        complexity_score = self._last_assessment.get("complexity_score", 5.0) if hasattr(self, '_last_assessment') else 5.0
        task_type = self._last_assessment.get("task_type", "default") if hasattr(self, '_last_assessment') else "default"
        AMACheckpoint.save(
            trace_id=trace_id,
            round_num=0,
            task=task,
            mode=AgentMode.PARALLEL_FUSION.value,
            task_type=task_type,
            complexity_score=complexity_score,
            completed_steps=[],
            pending_steps=["generate_layers", "aggregate"],
            results_so_far={},
        )

        # 并行融合配置
        fusion_config = kwargs.get("fusion_config", {})
        num_reference_agents = fusion_config.get("num_agents", 3)
        num_layers = fusion_config.get("num_layers", 2)  # 参考层层数（不含裁决层），默认 2

        # ── 终止守卫（借鉴 AutoGen 可组合条件）──
        max_tokens = fusion_config.get("max_tokens", 50000)
        max_time = fusion_config.get("max_time", 600)
        termination_guard = TerminationGuard(
            TokenBudgetTermination(max_tokens) | TimeoutTermination(max_time)
        )
        fusion_start_time = time.time()

        total_steps = num_layers + 1  # 参考层 + 裁决层
        logger.info(
            "[AMA] 启动并行融合模式 (trace=%s, 参考代理数=%d, 层数=%d)",
            trace_id[:8], num_reference_agents, num_layers,
        )
        _progress_reporter.report(
            "执行中",
            f"并行融合模式：{num_reference_agents}个代理 × {num_layers}层 + 裁决层",
            f"1/{total_steps}",
        )

        # ── 参考代理角色定义 ──
        reference_roles = [
            "分析专家 - 擅长深度分析和逻辑推理",
            "创意专家 - 擅长创新思维和多角度思考",
            "批判性思考者 - 擅长发现问题和风险评估",
            "领域专家 - 擅长专业知识和行业经验",
            "系统架构师 - 擅长全局视角和系统性思考",
        ][:num_reference_agents]

        # ── 模型多样性配置（借鉴 MoA：异构模型 > 同构模型）──
        # 支持为每个参考代理指定不同的模型
        # fusion_config.reference_models = ["deepseek-v4", "kimi-k2.6", "qwen3.6"]
        reference_models = fusion_config.get("reference_models", [])
        if reference_models:
            logger.info("[AMA] 模型多样性: %s", reference_models)

        # ── 多层迭代生成 ──
        current_responses: List[Dict[str, str]] = []
        all_layer_responses: List[List[Dict[str, str]]] = []  # 保留所有层的结果供裁决

        for layer_idx in range(num_layers):
            step = layer_idx + 1
            is_first_layer = (layer_idx == 0)

            # ── 终止守卫检查 ──
            guard_ctx = {
                "tokens_used": tokens_used,
                "elapsed": time.time() - fusion_start_time,
            }
            if termination_guard.should_terminate(guard_ctx):
                logger.warning(
                    "[AMA] 终止守卫触发：tokens=%d, elapsed=%.1fs, 跳过剩余层",
                    tokens_used, guard_ctx["elapsed"],
                )
                break

            if is_first_layer:
                # Layer 1: 初始回答，不带前序上下文
                _progress_reporter.report(
                    "生成中",
                    f"Layer {step}/{num_layers}：{num_reference_agents}个代理并行生成初始回答",
                    f"{step}/{total_steps}",
                )
            else:
                # Layer 2+: 基于前一层结果优化回答
                _progress_reporter.report(
                    "优化中",
                    f"Layer {step}/{num_layers}：基于上一轮结果迭代优化",
                    f"{step}/{total_steps}",
                )

            # 构建本轮任务
            parallel_tasks = []
            for i, role in enumerate(reference_roles):
                if is_first_layer:
                    # 第一层：原始任务
                    goal = (
                        f"角色: {role}\n\n"
                        f"任务: {task}\n\n"
                    )
                    if context:
                        goal += f"上下文: {context}\n\n"
                    goal += (
                        f"请从你的专业角度独立分析并回答这个问题。\n"
                        f"要求：\n"
                        f"1. 完整回答问题，不要遗漏关键点\n"
                        f"2. 提供你的推理过程和依据\n"
                        f"3. 如果有不确定的地方，明确指出\n"
                        f"4. 保持独立思考，不要受其他观点影响"
                    )
                else:
                    # 后续层：看到上一层所有回答后改进
                    prev_summary = self._build_layer_summary(current_responses)
                    goal = (
                        f"角色: {role}\n\n"
                        f"原始任务: {task}\n\n"
                    )
                    if context:
                        goal += f"上下文: {context}\n\n"
                    goal += (
                        f"以下是其他专家的上一轮回答，请参考后改进你的答案：\n\n"
                        f"{prev_summary}\n\n"
                        f"要求：\n"
                        f"1. 保留你上一轮回答中仍然正确的部分\n"
                        f"2. 借鉴其他专家的优秀观点补充你的回答\n"
                        f"3. 指出其他专家回答中的错误或遗漏\n"
                        f"4. 你的最终回答应该比上一轮更完整、更准确"
                    )
                task_entry = {"goal": goal, "context": context}
                # 模型多样性：为每个代理指定不同模型
                if reference_models and i < len(reference_models):
                    task_entry["model"] = reference_models[i]
                parallel_tasks.append(task_entry)

            # 并行执行本轮所有代理
            layer_results = self._execute_subagent_parallel(
                ctx, parallel_tasks,
                _mode=AgentMode.PARALLEL_FUSION, trace_id=trace_id, **kwargs,
            )

            # 收集本轮结果
            current_responses = []
            layer_failed = 0
            for i, sr in enumerate(layer_results):
                tokens_used += sr.token_usage
                if sr.status == SubagentStatus.COMPLETED and sr.result:
                    current_responses.append({
                        "role": reference_roles[i],
                        "response": sr.result,
                        "layer": layer_idx + 1,
                    })
                else:
                    layer_failed += 1

            all_layer_responses.append(list(current_responses))
            logger.info(
                "[AMA] Layer %d 完成: %d 成功, %d 失败",
                layer_idx + 1, len(current_responses), layer_failed,
            )

            # 如果本轮全部失败，提前终止
            if len(current_responses) < 1:
                logger.warning("[AMA] Layer %d 全部失败，提前终止", layer_idx + 1)
                break

        # ── 最终裁决层 ──
        _progress_reporter.report(
            "审核中",
            f"裁决模型正在综合 {sum(len(r) for r in all_layer_responses)} 个回答...",
            f"{total_steps}/{total_steps}",
        )

        # 汇总所有层的回答
        all_responses_flat = []
        for layer_idx, layer_resps in enumerate(all_layer_responses):
            for resp in layer_resps:
                resp["layer"] = layer_idx + 1
                all_responses_flat.append(resp)

        if len(all_responses_flat) < 1:
            AMACheckpoint.mark_interrupted(trace_id, ErrorCategory.subagent_failure.value)
            return {
                "result": "并行融合失败：所有参考代理均未能生成有效回答",
                "success": False,
                "token_usage": tokens_used,
                "time_taken": time.time() - start_time,
                "mode": AgentMode.PARALLEL_FUSION.value,
                "metadata": {
                    "reference_agents": num_reference_agents,
                    "layers_used": len(all_layer_responses),
                    "successful": 0,
                },
                "task_id": "",
                "trace_id": trace_id,
                "status": SubagentStatus.FAILED.value,
                "error_category": ErrorCategory.subagent_failure.value,
                "retries_attempted": 0,
            }

        # 构建裁决提示词（包含多层信息）
        aggregator_prompt = self._build_multilayer_aggregator_prompt(
            task, all_layer_responses, all_responses_flat,
        )

        aggregator_goal = (
            f"你是裁决模型，负责综合多个专家的多轮回答生成最终答案。\n\n"
            f"任务: {task}\n\n"
            f"以下是 {num_layers} 轮迭代中 {len(all_responses_flat)} 个回答：\n\n"
            f"{aggregator_prompt}\n\n"
            f"请执行以下步骤：\n"
            f"1. 分析各回答的共识点（多数专家认同的观点）\n"
            f"2. 识别各回答的分歧点（不同观点和争议）\n"
            f"3. 关注后轮回答对前轮的改进（说明哪些观点在迭代中被修正）\n"
            f"4. 评估各回答的独特见解和补充信息\n"
            f"5. 检查各回答中的潜在错误或偏见\n"
            f"6. 综合所有优点，生成一个更高质量的最终答案\n\n"
            f"输出要求：\n"
            f"- 最终答案应该比任何单一回答更完整、更准确\n"
            f"- 保留各回答中的优秀观点\n"
            f"- 修正发现的错误\n"
            f"- 补充遗漏的关键信息\n"
            f"- 保持清晰的结构和逻辑"
        )

        aggregator_sr = self._execute_subagent(
            ctx, aggregator_goal,
            _mode=AgentMode.PARALLEL_FUSION, trace_id=trace_id, **kwargs,
        )
        tokens_used += aggregator_sr.token_usage

        final_result = aggregator_sr.result or "裁决模型未能生成最终答案"

        _progress_reporter.report(
            "已完成",
            f"并行融合完成（{num_layers}层+裁决），共使用 {tokens_used} tokens",
            "✓",
        )

        success = aggregator_sr.status == SubagentStatus.COMPLETED
        if success:
            AMACheckpoint.mark_completed(trace_id)
        else:
            AMACheckpoint.mark_interrupted(trace_id, "aggregator_failed")

        return {
            "result": final_result,
            "success": success,
            "token_usage": tokens_used,
            "time_taken": time.time() - start_time,
            "mode": AgentMode.PARALLEL_FUSION.value,
            "metadata": {
                "reference_agents": num_reference_agents,
                "layers_used": len(all_layer_responses),
                "total_responses": len(all_responses_flat),
                "per_layer_counts": [len(r) for r in all_layer_responses],
                "roles_used": [role for role in reference_roles],
            },
            "task_id": "",
            "trace_id": trace_id,
            "status": SubagentStatus.COMPLETED.value if success else SubagentStatus.FAILED.value,
            "error_category": None,
            "retries_attempted": 0,
        }


    def _build_layer_summary(self, responses: List[Dict[str, str]]) -> str:
        """构建上一轮回答的摘要，用于注入下一轮的 prompt"""
        parts = []
        for i, resp in enumerate(responses, 1):
            # 截断过长的回答，避免上下文溢出
            text = resp["response"]
            if len(text) > 3000:
                text = text[:3000] + "\n...(截断)"
            parts.append(f"--- {resp['role']} 的回答 ---\n{text}\n")
        return "\n".join(parts)


    def _build_multilayer_aggregator_prompt(
        self,
        task: str,
        layer_responses: List[List[Dict[str, str]]],
        all_flat: List[Dict[str, str]],
    ) -> str:
        """构建多层裁决提示词，区分不同层的回答"""
        parts = []
        for layer_idx, layer_resps in enumerate(layer_responses):
            if not layer_resps:
                continue
            parts.append(f"\n{'='*40}")
            parts.append(f"第 {layer_idx + 1} 轮回答")
            parts.append(f"{'='*40}\n")
            for i, resp in enumerate(layer_resps, 1):
                text = resp["response"]
                if len(text) > 3000:
                    text = text[:3000] + "\n...(截断)"
                parts.append(f"--- 专家 {i}: {resp['role']} ---\n{text}\n")
        return "\n".join(parts)


    def _build_aggregator_prompt(
        self, task: str, responses: List[Dict[str, str]]
    ) -> str:
        """构建裁决模型的提示词（单层兼容）"""
        parts = []
        for i, resp in enumerate(responses, 1):
            parts.append(
                f"--- 专家 {i}: {resp['role']} ---\n"
                f"{resp['response']}\n"
                f"--- 专家 {i} 结束 ---\n"
            )
        return "\n".join(parts)


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


    def try_switch_mode(
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
                logger.info("模式切换冷却中: %s，跳过", cooldown_key)
                continue

            switch_ctx.target_mode = mode.value
            switch_kwargs = dict(kwargs)
            switch_kwargs["switch_context"] = switch_ctx
            enhanced_context = context or ""
            if switch_ctx.intermediate_result:
                enhanced_context += f"\n\n[前次模式 {switch_ctx.source_mode} 的中间结果]\n{switch_ctx.intermediate_result}"

            switches += 1
            self._switch_cooldown[cooldown_key] = time.time()
            result = self.execute_mode(ctx, task, enhanced_context, mode, **switch_kwargs)
            if result.get("success"):
                return result

        return None


