"""omni_record_action 工具处理器。"""

from __future__ import annotations

import json
import logging
from typing import Any

from omnimem.core.action_memory import ActionRecord

logger = logging.getLogger(__name__)

# ★ 低价值 routine tool_call 的工具名黑名单：这些工具的 success 调用不单独记录
_ROUTINE_TOOLS = frozenset({
    "terminal", "read_file", "write_file", "search_files", "patch",
    "skill_view", "skill_search", "skills_list", "skill_manage",
    "process", "mcp_codegraph_codegraph_explore", "mcp_codegraph_codegraph_search",
    "mcp_codegraph_codegraph_node", "mcp_codegraph_codegraph_files",
    "mcp_codegraph_codegraph_impact", "mcp_codegraph_codegraph_callers",
    "mcp_codegraph_codegraph_callees", "mcp_codegraph_codegraph_status",
    "skill_feedback",
})


def _should_skip_action(record: ActionRecord) -> bool:
    """★ 过滤低价值 action 记忆，防止 event 膨胀。

    仅记录有学习价值的行为：
      - 有 lesson_learned 的行为（无论 outcome）
      - failure / partial 的行为（即使是 routine 工具）
      - 非 tool_call 类型（decision / spawn / error_handling 始终记录）
      - tool_call 但不在 routine 黑名单中的（如 MCP 工具、飞书工具等）
    """
    # 非 tool_call 类型始终记录
    if record.action_type != "tool_call":
        return False

    # 有教训记录，值得记录
    if record.lesson_learned:
        return False

    # 失败/部分失败，值得记录
    if record.outcome in ("failure", "partial"):
        return False

    # routine 工具的成功调用，跳过
    if record.tool_name in _ROUTINE_TOOLS:
        return True

    return False


def handle_record_action(provider: Any, args: dict[str, Any]) -> str:
    """处理 omni_record_action 工具调用。

    记录 Agent 的一次工具调用/决策行为。

    ★ 过滤策略：只记录有学习价值的行为，跳过 routine tool_call 的 success 记录，
    防止 event 记忆膨胀。

    Args:
        provider: OmniMemProvider 实例
        args: {
            "action_type": "tool_call" | "decision" | "spawn" | "error_handling",
            "tool_name": str,           # 工具名
            "tool_args_summary": str,   # 参数摘要 (选填)
            "tool_result_summary": str, # 结果摘要 (选填)
            "decision_rationale": str,  # 决策理由 (选填)
            "outcome": "success"|"failure"|"partial", # 结果
            "lesson_learned": str,      # 教训 (选填)
            "parent_task_id": str,      # 所属任务 (选填)
            "agent_role": str,          # 代理角色 (选填)
            "duration_ms": int,         # 耗时 (选填)
            "turn_index": int,          # 轮次 (选填)
        }

    Returns:
        JSON: {"status": "stored", "action_id": "...", "memory_id": "..."}
        或 {"status": "filtered", "reason": "..."} — 被过滤的低价值 action
    """
    if not hasattr(provider, "_action_memory") or provider._action_memory is None:
        return json.dumps({
            "status": "unavailable",
            "reason": "ActionMemoryService not initialized. Check omnimem config.",
        })

    try:
        record = ActionRecord.from_args(args)

        # ★ 过滤低价值 action
        if _should_skip_action(record):
            return json.dumps({
                "status": "filtered",
                "reason": f"Routine {record.tool_name} success — low value",
                "action_type": record.action_type,
                "tool_name": record.tool_name,
            }, ensure_ascii=False)

        mid = provider._action_memory.record_action(record)

        # ★ OPT: 记录行为溯源链
        if hasattr(provider, '_trace_chain') and provider._trace_chain:
            try:
                provider._trace_chain.record_derivation(
                    parent_node_ids=[f"action-{provider._session_id}-{record.turn_index or 0}"],
                    child_node_id=mid,
                    child_layer="L1",
                    ref_path=str(provider._data_dir / "refs" / f"action-{mid}.md"),
                )
            except Exception as e:
                logger.warning("TraceChain record_action failed: %s", e)

        return json.dumps({
            "status": "stored",
            "action_id": record.action_id,
            "memory_id": mid,
            "action_type": record.action_type,
            "outcome": record.outcome,
        }, ensure_ascii=False)

    except Exception as e:
        logger.warning("omni_record_action failed: %s", e)
        return json.dumps({
            "status": "error",
            "reason": str(e),
        })
