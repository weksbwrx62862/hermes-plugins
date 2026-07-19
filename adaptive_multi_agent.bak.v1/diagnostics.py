from __future__ import annotations

# 本模块负责 AMA 内部状态诊断与执行流程可视化。
# 包含 diagnose()、generate_mermaid_diagram() 以及 Python 异常字符串检测工具。

import time
from typing import Any, Dict, List, Optional

from .persistence import get_execution_by_trace_id, get_stats as get_persistence_stats

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


def recent_errors(engine=None, limit: int = 20) -> List[Dict[str, Any]]:
    """查询最近 N 条失败记录并按 error_category 分组统计。

    Args:
        engine: AdaptiveMultiAgentEngine 实例（保留接口一致性，当前未使用）。
        limit: 最近失败记录数量上限。

    Returns:
        按 error_category 分组后的统计列表，每项包含 category 与 count。
    """
    # engine 参数保留用于未来按引擎/会话过滤；当前忽略
    del engine
    from .persistence import get_persistence

    pers = get_persistence()
    conn = pers._connect()
    try:
        rows = conn.execute(
            """SELECT error_category, COUNT(*) as cnt
               FROM (
                   SELECT error_category
                   FROM ama_executions
                   WHERE success = 0
                   ORDER BY created_at DESC
                   LIMIT ?
               )
               GROUP BY error_category
               ORDER BY cnt DESC""",
            (limit,),
        ).fetchall()
        return [{"category": r["error_category"] or "unknown", "count": r["cnt"]} for r in rows]
    finally:
        conn.close()


def diagnose(engine) -> Dict:
    """诊断 AMA 内部状态：TS 参数、性能历史、熔断器、会话覆盖等"""

    # ── Thompson Sampling 参数 ──
    ts_params = {}
    for (task_type, mode_name), (alpha, beta_val) in engine.selector._ts_params.items():
        ts_params.setdefault(task_type, {})[mode_name] = {
            "alpha": round(alpha, 2),
            "beta": round(beta_val, 2),
            "expected": round(alpha / (alpha + beta_val), 3) if (alpha + beta_val) > 0 else 0.5,
            "trials_equivalent": int(alpha + beta_val - 2),  # 等效试验次数（减去先验）
        }

    # ── 性能历史 ──
    perf_summary = {}
    for task_type, modes in engine.selector.historical_performance.items():
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
    for mode, cb in engine.circuit_breakers.items():
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
    lines.append(f"  会话覆盖: {engine.session_mode_override.value if engine.session_mode_override else '无'}")

    return {
        "summary": "\n".join(lines),
        "ts_params": ts_params,
        "performance": perf_summary,
        "circuit_breakers": cb_status,
        "session_override": engine.session_mode_override.value if engine.session_mode_override else None,
        "recent_errors": recent_errors(engine, limit=20),
        "config": {
            "allow_mode_switch": engine.config["allow_mode_switch"],
            "llm_refine_enabled": engine.config.get("llm_refine_enabled", True),
            "llm_refine_range": engine.config.get("llm_refine_range", (3.0, 7.0)),
        },
    }


def generate_mermaid_diagram(engine, trace_id: Optional[str] = None) -> str:
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

