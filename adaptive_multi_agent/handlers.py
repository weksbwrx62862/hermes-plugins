from __future__ import annotations

import json
from typing import Any, Dict, Optional

from tools.registry import tool_error, tool_result

from .engine import AdaptiveMultiAgentEngine, AgentMode, RequirementClarifier
from .persistence import get_stats
from .subagent import _MODE_CN, TASK_TYPE_CN

_engine: Optional[AdaptiveMultiAgentEngine] = None
_clarifier: Optional[RequirementClarifier] = None
_plugin_ctx: Optional[object] = None  # 由 __init__.register() 注入


def _load_hermes_config() -> Dict:
    """从 ~/.hermes/config.yaml 加载 ama 配置项"""
    try:
        import yaml
        from pathlib import Path
        config_path = Path.home() / ".hermes" / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f) or {}
            return full_config.get("ama", {})
    except Exception:
        pass
    return {}


def _get_engine() -> AdaptiveMultiAgentEngine:
    global _engine
    if _engine is None:
        config = _load_hermes_config()
        _engine = AdaptiveMultiAgentEngine(config=config)
    return _engine


def _get_clarifier() -> RequirementClarifier:
    global _clarifier
    if _clarifier is None:
        _clarifier = RequirementClarifier()
    return _clarifier


def handle_ama_execute(args: Dict, **kwargs) -> str:
    task = args.get("task", "")
    if not task:
        return tool_error("task 参数不能为空")

    context = args.get("context")
    force_mode = args.get("force_mode")
    timeout_seconds = args.get("timeout_seconds")
    subagent_type = args.get("subagent_type")
    clarify = args.get("clarify", False)
    human_input_mode = args.get("human_input_mode", "NEVER")

    engine = _get_engine()
    try:
        external_assessment = None
        if clarify:
            clarifier = _get_clarifier()
            if _plugin_ctx:
                clarify_result = clarifier.clarify_and_score(
                    _plugin_ctx, task, context=context
                )
                external_assessment = clarify_result
                task = clarify_result.get("clarified_task", task)

        result = engine.execute(
            ctx=_plugin_ctx,
            task=task,
            context=context,
            force_mode=force_mode,
            session_id=kwargs.get("session_id"),
            parent_agent=kwargs.get("parent_agent"),
            timeout_seconds=timeout_seconds,
            subagent_type=subagent_type,
            external_assessment=external_assessment,
            human_input_mode=human_input_mode,
        )
        return tool_result(result)
    except Exception as e:
        return tool_error(f"ama_execute 执行失败: {e}")


def handle_ama_assess(args: Dict, **kwargs) -> str:
    task = args.get("task", "")
    if not task:
        return tool_error("task 参数不能为空")

    context = args.get("context")
    clarify = args.get("clarify", False)
    engine = _get_engine()
    try:
        external_assessment = None
        if clarify:
            clarifier = _get_clarifier()
            if _plugin_ctx:
                clarify_result = clarifier.clarify_and_score(
                    _plugin_ctx, task, context=context
                )
                external_assessment = clarify_result

        assessment = engine.assessor.assess(
            task,
            {"context": context} if context else None,
            external_assessment=external_assessment,
        )

        # ── 选型诊断 ──
        rec_mode = assessment.get("recommended_mode", "")
        rec_mode_cn = _MODE_CN.get(AgentMode(rec_mode) if rec_mode in [m.value for m in AgentMode] else None, rec_mode)
        task_type_cn = TASK_TYPE_CN.get(assessment.get("task_type", ""), assessment.get("task_type", ""))
        features_active = [k for k, v in assessment.get("features", {}).items()
                           if v and k not in ("context_size", "uncertainty_level", "task_length")]

        assessment["diagnosis"] = {
            "summary": (
                f"[AMA评估] 任务类型={task_type_cn} | 复杂度={assessment['complexity_score']:.1f}/10 | "
                f"推荐模式={rec_mode_cn} | 特征={','.join(features_active) if features_active else '无'}"
            ),
            "task_type_cn": task_type_cn,
            "recommended_mode_cn": rec_mode_cn,
            "features_active": features_active,
        }
        return tool_result(assessment)
    except Exception as e:
        return tool_error(f"ama_assess 评估失败: {e}")


def handle_ama_switch_mode(args: Dict, **kwargs) -> str:
    mode_str = args.get("mode", "")
    if not mode_str:
        return tool_error("mode 参数不能为空")

    try:
        mode = AgentMode(mode_str)
    except ValueError:
        return tool_error(
            f"无效模式: {mode_str}，可选值: {[m.value for m in AgentMode]}"
        )

    engine = _get_engine()
    engine.session_mode_override = mode
    return tool_result({
        "success": True,
        "current_mode": mode.value,
        "message": f"已切换到 {mode.value} 模式",
    })


def handle_ama_stats(args: Dict, **kwargs) -> str:
    detail = args.get("detail", False)
    period = args.get("period", "all")
    try:
        stats = get_stats(detail=detail, period=period)
        # ── 格式化摘要 ──
        total = stats.get("total_executions", 0)
        period_label = {"day": "今天", "week": "最近7天", "month": "最近30天"}.get(period, "全部")
        lines = [f"[AMA统计] {period_label} | 总执行: {total}次"]
        for mode, cnt in sorted(stats.get("mode_usage", {}).items(), key=lambda x: -x[1]):
            rate = stats.get("success_rates", {}).get(mode, {})
            lines.append(f"  {_MODE_CN.get(AgentMode(mode) if mode in [m.value for m in AgentMode] else None, mode)}: {cnt}次, 成功率={rate.get('rate', 0):.0%}")
        sw = stats.get("switch_events", 0)
        if sw:
            lines.append(f"  模式切换: {sw}次 ({stats.get('switch_rate', 0):.1%})")
        stats["summary"] = "\n".join(lines)
        if detail:
            engine = _get_engine()
            stats["mermaid_diagram"] = engine.generate_mermaid_diagram()
        return tool_result(stats)
    except Exception as e:
        return tool_error(f"ama_stats 查询失败: {e}")


def handle_ama_cancel(args: Dict, **kwargs) -> str:
    task_id = args.get("task_id", "")
    if not task_id:
        return tool_error("task_id 参数不能为空")

    engine = _get_engine()
    try:
        cancelled = engine.result_store.request_cancel(task_id)
        if cancelled:
            return tool_result({
                "success": True,
                "task_id": task_id,
                "message": f"已请求取消任务 {task_id}",
            })
        else:
            return tool_result({
                "success": False,
                "task_id": task_id,
                "message": f"任务 {task_id} 不存在或已完成，无法取消",
            })
    except Exception as e:
        return tool_error(f"ama_cancel 执行失败: {e}")


def handle_ama_diagnose(args: Dict, **kwargs) -> str:
    engine = _get_engine()
    try:
        diag = engine.diagnose()
        # 过滤可选项
        if not args.get("include_ts_params", True):
            diag.pop("ts_params", None)
        if not args.get("include_circuit_breakers", True):
            diag.pop("circuit_breakers", None)
        trace_id_query = args.get("trace_id")
        if trace_id_query:
            from .persistence import get_execution_by_trace_id
            exec_record = get_execution_by_trace_id(trace_id_query)
            if exec_record:
                diag["execution_detail"] = exec_record
        diag["mode_flow_diagram"] = engine.generate_mermaid_diagram(trace_id_query)
        return tool_result(diag)
    except Exception as e:
        return tool_error(f"ama_diagnose 执行失败: {e}")


def handle_ama_clarify(args: Dict, **kwargs) -> str:
    task = args.get("task", "")
    if not task:
        return tool_error("task 参数不能为空")

    context = args.get("context")
    max_rounds = args.get("max_rounds", 3)

    clarifier = _get_clarifier()
    if not _plugin_ctx:
        return tool_error("ama_clarify 需要 ctx 上下文（插件 ctx 未注入）")

    try:
        result = clarifier.clarify_and_score(
            _plugin_ctx, task, context=context, max_rounds=max_rounds
        )
        return tool_result(result)
    except Exception as e:
        return tool_error(f"ama_clarify 执行失败: {e}")


def handle_post_tool_call(tool_name: str = "", args: Any = None, result: Any = None, **kwargs) -> Optional[str]:
    if not tool_name.startswith("ama_"):
        return None

    engine = _get_engine()
    threshold = engine.config.get("switch_threshold", {})

    tool_result_str = result or ""
    try:
        result_data = json.loads(tool_result_str) if isinstance(tool_result_str, str) else tool_result_str
    except (json.JSONDecodeError, TypeError):
        return None

    if tool_name == "ama_execute" and isinstance(result_data, dict):
        token_usage = result_data.get("token_usage", 0)
        time_taken = result_data.get("time_taken", 0)
        max_tokens = threshold.get("max_tokens", 50000)
        max_time = threshold.get("max_time", 300)

        alerts = []
        if token_usage > max_tokens:
            alerts.append(f"token 消耗 ({token_usage}) 超过阈值 ({max_tokens})")
        if time_taken > max_time:
            alerts.append(f"执行时间 ({time_taken:.1f}s) 超过阈值 ({max_time}s)")

        status = result_data.get("status", "")
        error_category = result_data.get("error_category")
        retries = result_data.get("retries_attempted", 0)

        if status == "failed":
            alerts.append(f"任务执行失败，错误类别: {error_category or 'unknown'}")
        if retries > 0:
            alerts.append(f"子代理重试了 {retries} 次")

        if alerts:
            return json.dumps({
                "ama_monitoring": True,
                "alerts": alerts,
                "suggestion": "考虑使用更轻量的模式或拆分任务",
            }, ensure_ascii=False)

    return None


def handle_on_session_start(session_id: str, **kwargs) -> None:
    engine = _get_engine()
    engine.session_mode_override = None


def handle_on_session_end(session_id: str, **kwargs) -> None:
    engine = _get_engine()
    engine.session_mode_override = None
    engine.result_store.cleanup(3600)


def _reset_engine() -> None:
    global _engine
    _engine = None


def _reset_clarifier() -> None:
    global _clarifier
    _clarifier = None
