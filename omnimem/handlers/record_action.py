"""omni_record_action 工具处理器。"""

from __future__ import annotations

import json
import logging
from typing import Any

from omnimem.core.action_memory import ActionRecord

logger = logging.getLogger(__name__)


def handle_record_action(provider: Any, args: dict[str, Any]) -> str:
    """处理 omni_record_action 工具调用。

    记录 Agent 的一次工具调用/决策行为。

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
    """
    if not hasattr(provider, "_action_memory") or provider._action_memory is None:
        return json.dumps({
            "status": "unavailable",
            "reason": "ActionMemoryService not initialized. Check omnimem config.",
        })

    try:
        record = ActionRecord.from_args(args)
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
