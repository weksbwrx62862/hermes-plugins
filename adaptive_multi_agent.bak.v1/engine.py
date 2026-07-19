from __future__ import annotations

# AdaptiveMultiAgentEngine 现在作为 facade，负责初始化各子模块、配置管理、生命周期钩子
# 以及公共方法 execute/assess/diagnose 的委托。具体评估、选择、执行、诊断逻辑分别位于
# assessor.py、selector.py、executor.py、diagnostics.py。

import json
import logging
import time
from typing import Any, Dict, List, Optional

from .checkpoint import AMACheckpoint, CheckpointState
from .persistence import save_execution_transaction
from .skill_registry import get_skill_registry
from .workflows import match_workflow
from .subagent import (
    AgentMode,
    CircuitBreaker,
    PluginRegistry,
    RetryPolicy,
    SubagentConfig,
    SubagentRegistry,
    SubagentResult,
    SubagentStatus,
    TaskResultStore,
    _MODE_CN,
    MODE_CN_SHORT,
    TASK_TYPE_CN,
)
from .assessor import (
    TaskComplexityAssessor,
    RequirementClarifier,
    LLM_REFINE_PROMPT_TEMPLATE,
)
from .selector import (
    ModeSelectionEngine,
    _read_from_plugin_context,
    _write_to_plugin_context,
)
from .executor import ModeExecutor
from . import diagnostics

logger = logging.getLogger(__name__)


class AdaptiveMultiAgentEngine:

    def __init__(self, config: Optional[Dict] = None):
        self.assessor = TaskComplexityAssessor()
        self.config = {
            "allow_mode_switch": True,
            "switch_threshold": {"max_tokens": 50000, "max_time": 300},
            "default_mode": "auto",
            "max_concurrent_children": 3,
            "llm_refine_enabled": True,
            "llm_refine_range": (4.0, 6.0),  # 收窄至 [4,6]：减少 LLM 调用开销
            "use_dag": True,
            "dag_max_workers": 4,
        }
        if config:
            self.config.update(config)
        self.circuit_breakers: Dict[AgentMode, CircuitBreaker] = {
            mode: CircuitBreaker() for mode in AgentMode
        }
        self.selector = ModeSelectionEngine(
            circuit_breakers=self.circuit_breakers,
            config={"mode_rules": self.config.get("mode_rules")} if self.config.get("mode_rules") else None,
        )
        self.session_mode_override: Optional[AgentMode] = None
        if config and "default_mode" in config and config["default_mode"] != "auto":
            self.session_mode_override = AgentMode(config["default_mode"])
        self.registry = SubagentRegistry()
        self.plugin_registry = PluginRegistry()
        self.result_store = TaskResultStore()
        self.executor = ModeExecutor(self)
        self.retry_policy = RetryPolicy()
        self._human_input_mode = "NEVER"
        self.skill_registry = get_skill_registry()
        self.current_workflow = None
        self._switch_cooldown: Dict[str, float] = {}
        self._switch_cooldown_seconds = 30.0
        self._lifecycle_hooks: Dict[str, List] = {
            "on_started": [], "on_progress": [], "on_completed": [],
            "on_failed": [], "on_timeout": [], "on_cancelled": [],
        }


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
                logger.warning("生命周期钩子 %s 执行失败: %s", event, e)


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
                logger.warning("LLM refine 分数异常: %s，回退规则评分", llm_score)
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
            logger.warning("LLM refine 失败，回退规则评分: %s", e)
            return rule_assessment


    def execute(
        self,
        ctx,
        task: str,
        context: Optional[str] = None,
        force_mode: Optional[str] = None,
        resume_from: Optional[Any] = None,
        **kwargs,
    ) -> Dict:
        # ── 断点恢复路径 ──
        if resume_from is not None:
            cp: Optional[CheckpointState] = None
            if isinstance(resume_from, str):
                cp = AMACheckpoint.load_latest(resume_from)
            elif isinstance(resume_from, CheckpointState):
                cp = resume_from

            if not cp:
                return {
                    "success": False,
                    "result": f"未找到 trace={resume_from} 的检查点",
                    "mode_used": "",
                    "trace_id": resume_from if isinstance(resume_from, str) else "",
                }

            result = self.executor.resume_from_checkpoint(cp, ctx, **kwargs)
            if result.get("success"):
                AMACheckpoint.mark_completed(cp.trace_id)
            else:
                AMACheckpoint.mark_interrupted(cp.trace_id, result.get("error_category") or "resume_failed")
            return {
                **result,
                "resumed": True,
                "trace_id": cp.trace_id,
            }

        start_time = time.time()
        self._human_input_mode = kwargs.get("human_input_mode", "NEVER")

        # ── 轨迹记录：开始 ──
        from .trajectory import get_recorder
        recorder = get_recorder()

        assessment = self.assessor.assess(
            task,
            {"context": context} if context else None,
            external_assessment=kwargs.get("external_assessment"),
        )

        # ── 获取当前活跃模型名（供反馈闭环使用） ──
        try:
            # 优先从 PluginContext 读取
            active_model = _read_from_plugin_context(
                session_id=kwargs.get("session_id"),
                key="model_selection", default="",
            )
            if active_model:
                self._active_model_name = active_model
            else:
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
                # 优先通过 PluginContext 写入（如果 orchestrator 已安装）
                _write_to_plugin_context(session_id, "ama_task_weight", assessment["complexity_score"])
                _write_to_plugin_context(session_id, "task_complexity", assessment["complexity_score"])
                # 也通知 model-router 的全局变量（向后兼容）
                from plugins.model_router import set_task_weight
                rec_strategy = set_task_weight(session_id, assessment["complexity_score"])
                logger.info(
                    "[AMA→Router] session=%s | AMA评分=%.1f | 推荐策略=%s",
                    session_id[:8], assessment["complexity_score"], rec_strategy,
                )
            except ImportError:
                pass  # Model Router 未安装则跳过

        # ── 工作流匹配：为结构化任务推荐最佳流程（借鉴 gstack）──
        self.current_workflow = match_workflow(assessment["task_type"])
        if self.current_workflow:
            wf = self.current_workflow
            mode_from_workflow = wf.default_mode
            # 若非强制或会话覆盖模式，且工作流推荐的模式复杂度合适，则使用工作流的默认模式
            if not force_mode and not self.session_mode_override:
                ctx.dispatch_tool("send_message", {
                    "action": "send",
                    "message": (
                        f"📋 **工作流**: {wf.name}\n"
                        f"阶段: {' → '.join(s.name for s in wf.stages)}\n"
                        f"推荐模式: {mode_from_workflow.cn}\n"
                    ),
                })
                # 使用工作流推荐的模式（覆盖 TS 采样）
                selected_mode = mode_from_workflow
            del mode_from_workflow

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
        logger.info(
            "[AMA] %s | 类型=%s | 复杂度=%s | 选中: %s | 原因: %s | 特征=%s",
            session_tag,
            assessment["task_type"],
            score_part,
            selected_mode.cn,
            reason,
            ",".join(features_active) if features_active else "无",
        )

        self._last_assessment = assessment

        # ── 轨迹记录：开始记录 ──
        trajectory_id = recorder.start(
            task=task,
            context=context,
            mode=selected_mode.value,
            complexity_score=assessment["complexity_score"],
            task_type=assessment["task_type"],
            metadata={"session_id": kwargs.get("session_id", "")},
        )
        recorder.add_step(
            "assess",
            f"复杂度评估: {assessment['complexity_score']:.1f}/10, 推荐模式: {selected_mode.value}",
            input_data=task[:500],
            output_data=assessment,
        )

        result = self.executor.execute_mode(ctx, task, context, selected_mode, **kwargs)

        switched = False
        original_mode = selected_mode
        switch_reason = None

        if self.config["allow_mode_switch"] and not result.get("success"):
            switched_result = self.executor.try_switch_mode(
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

        # ── 更新执行状态：成功完成 / 中断可恢复 ──
        trace_id = result.get("trace_id", "")
        if trace_id:
            if success:
                AMACheckpoint.mark_completed(trace_id)
            else:
                AMACheckpoint.mark_interrupted(trace_id, result.get("error_category") or "execute_failed")

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

        # 新增：TS 探索状态
        ts_exploration = ""
        if not (force_mode or self.session_mode_override):
            mode_counts = {}
            for (tt, mv), (a, b) in self.selector._ts_params.items():
                if tt == assessment["task_type"]:
                    trials = a + b - 2
                    mode_counts[MODE_CN_SHORT.get(mv, mv)] = max(trials, 0)
            if mode_counts:
                ts_exploration = " | 探索: " + " ".join(
                    f"{k}×{v}" for k, v in sorted(mode_counts.items(), key=lambda x: -x[1])[:4]
                )

        diag_line = f"任务类型={diag_type} | 复杂度={diag_score} | 选中模式={diag_mode} | 原因={reason}"
        diag_switched = f" | ⚠️ 模式切换: {diag_orig}→{diag_mode}（{switch_reason}）" if switched else ""
        diagnosis = {
            "summary": f"[AMA选型] {diag_line}{diag_switched}{ts_exploration}",
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
            "ts_exploration": ts_exploration.strip() if ts_exploration else None,
        }

        # ── 轨迹记录：完成 ──
        recorder.add_step(
            "execute",
            f"执行模式: {mode_used}, 耗时: {time_taken:.1f}s",
            output_data=result.get("result", "")[:1000] if result.get("result") else None,
            success=success,
            error=result.get("error"),
            duration_ms=time_taken * 1000,
        )
        trajectory = recorder.finish(
            final_result=result.get("result", ""),
            success=success,
            error=result.get("error"),
        )
        # 将 trajectory_id 附加到返回结果
        result_trajectory_id = trajectory.trajectory_id if trajectory else ""

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
            "trajectory_id": result_trajectory_id,
            "status": result.get("status", ""),
            "error_category": result.get("error_category"),
            "retries_attempted": result.get("retries_attempted", 0),
        }


    def assess(self, task: str, context: Optional[Dict] = None, external_assessment: Optional[Dict] = None) -> Dict:
        """评估任务复杂度并返回评分结果。"""
        return self.assessor.assess(
            task,
            {"context": context} if context else None,
            external_assessment=external_assessment,
        )

    def diagnose(self) -> Dict:
        """诊断 AMA 内部状态：TS 参数、性能历史、熔断器、会话覆盖等。"""
        return diagnostics.diagnose(self)

    def generate_mermaid_diagram(self, trace_id: Optional[str] = None) -> str:
        """基于执行记录生成 Mermaid 流程图。"""
        return diagnostics.generate_mermaid_diagram(self, trace_id)

