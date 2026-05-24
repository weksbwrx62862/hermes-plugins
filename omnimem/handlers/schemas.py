"""OmniMem 工具 Schema 定义。

从 provider.py 中提取的 get_tool_schemas() 纯数据函数。
"""

from typing import Any


def get_tool_schemas() -> list[dict[str, Any]]:
    """OmniMem 暴露 7 个工具给 Agent。"""
    return [
        {
            "name": "omni_memorize",
            "description": (
                "Store a memory in OmniMem. Use for important facts, decisions, "
                "corrections, user preferences, agent actions, reasoning chains, "
                "or any information worth recalling in future sessions. "
                "Specify the type (fact/preference/correction/skill/procedural/"
                "event/action/reasoning) and confidence level (1-5). "
                "ACTION: agent tool calls, decisions, operations. "
                "REASONING: lessons learned, debugging insights, pitfalls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to store",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": [
                            "fact",
                            "preference",
                            "correction",
                            "skill",
                            "procedural",
                            "event",
                            "action",
                            "reasoning",
                        ],
                        "default": "fact",
                        "description": "Type of memory",
                    },
                    "confidence": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3,
                        "description": "Confidence level (5=certain, 1=uncertain)",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["personal", "project", "shared"],
                        "default": "personal",
                    },
                    "privacy": {
                        "type": "string",
                        "enum": ["public", "team", "personal", "secret"],
                        "default": "personal",
                    },
                },
                "required": ["content"],
            },
        },
        {
            "name": "omni_recall",
            "description": (
                "Search OmniMem for relevant memories. Use before answering "
                "questions about past context, user preferences, or decisions. "
                "Supports semantic and keyword search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["rag", "llm"],
                        "default": "rag",
                        "description": (
                            "rag: fast vector+BM25 hybrid (milliseconds). "
                            "llm: deep reasoning with intent prediction (seconds)."
                        ),
                    },
                    "max_tokens": {
                        "type": "integer",
                        "default": 1500,
                        "description": "Maximum tokens in results",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "omni_compact",
            "description": (
                "Manually trigger context compaction with OmniMem's progressive "
                "compression engine. Useful when you notice context is getting long."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {
                        "type": "integer",
                        "default": 4000,
                        "description": "Target token budget for compressed context",
                    },
                },
            },
        },
        {
            "name": "omni_reflect",
            "description": (
                "Reflect on accumulated memories to generate deeper insights. "
                "Consolidates raw facts into observations and mental models. "
                "Use when you need to synthesize patterns from past experiences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic or question to reflect on",
                    },
                    "disposition": {
                        "type": "object",
                        "properties": {
                            "skepticism": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "default": 3,
                                "description": "Skepticism level (5=very cautious, 1=trusting)",
                            },
                            "literalness": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "default": 2,
                                "description": "Literalness level (5=exact/verifiable, 1=speculative)",
                            },
                            "empathy": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "default": 4,
                                "description": "Empathy level (5=feeling-focused, 1=fact-focused)",
                            },
                        },
                        "description": "Reflection personality: adjusts tone and emphasis of reflection output",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "omni_govern",
            "description": (
                "Manage memory governance: resolve conflicts, set privacy levels, "
                "trigger forgetting/archive, or view memory provenance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "resolve_conflict",
                            "set_privacy",
                            "archive",
                            "reactivate",
                            "provenance",
                            "forgetting_status",
                            "lora_train",
                            "shade_switch",
                            "shade_list",
                            "kv_cache_stats",
                            "consolidation_stats",
                            "sync_status",
                            "sync_instances",
                            "export_memories",
                            "import_memories",
                            "wiki_upgrade",
                            "forgetting_heat",
                        ],
                        "description": "Governance action to perform",
                    },
                    "target": {
                        "type": "string",
                        "description": "Memory ID or query for the action",
                    },
                    "params": {
                        "type": "object",
                        "description": 'Additional parameters for the action. set_privacy: {"level": "public|team|personal|secret"}. resolve_conflict/scan_conflicts: no params needed.',
                        "properties": {
                            "level": {
                                "type": "string",
                                "description": "Privacy level for set_privacy: public, team, personal, or secret",
                                "enum": ["public", "team", "personal", "secret"],
                            },
                        },
                    },
                },
                "required": ["action"],
            },
        },
        {
            "name": "omni_detail",
            "description": (
                "Fetch full details of a specific memory by ID. Use when you need "
                "more context than the summary provided in prefetch. "
                "Prefetch only injects concise summaries; this tool lets you "
                "lazily load the full content when needed.\n\n"
                "ALSO: Use action='list' to see all memories injected this turn "
                "(with their IDs for detail lookup). Use action='events' to query "
                "the session event log at a specific time range (getEvents pattern)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get", "list", "events"],
                        "description": (
                            "get: fetch full detail of a memory by ID. "
                            "list: show all memories injected this turn. "
                            "events: query session event log at a time range."
                        ),
                        "default": "list",
                    },
                    "memory_id": {
                        "type": "string",
                        "description": "Memory ID (required for action='get')",
                    },
                    "from_turn": {
                        "type": "integer",
                        "description": "Start turn number for events query (default: 0)",
                    },
                    "to_turn": {
                        "type": "integer",
                        "description": "End turn number for events query (default: current)",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional filter query for events",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "memory",
            "description": (
                "Save durable information to persistent memory that survives across sessions. "
                "Memory is injected into future turns, so keep it compact and focused on facts "
                "that will still matter later.\n\n"
                "WHEN TO SAVE:\n"
                "- User corrects you or says 'remember this'\n"
                "- User shares a preference, habit, or personal detail\n"
                "- You discover something about the environment\n"
                "- You learn a convention or API quirk specific to this setup\n\n"
                "TWO TARGETS:\n"
                "- 'memory': your notes (environment facts, conventions, lessons learned)\n"
                "- 'user': who the user is (preferences, communication style, habits)\n\n"
                "ACTIONS: add (new entry), replace (update existing), remove (delete)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "replace", "remove"],
                        "description": "The action to perform.",
                    },
                    "target": {
                        "type": "string",
                        "enum": ["memory", "user"],
                        "description": "'memory' for notes, 'user' for user profile.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Entry content. Required for 'add' and 'replace'.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Substring identifying entry for replace/remove.",
                    },
                },
                "required": ["action", "target"],
            },
        },
        {
            "name": "omni_record_action",
            "description": (
                "Record an agent action: tool call, decision, subagent spawn, "
                "or error handling. Captures what the agent did, why, the outcome, "
                "and lessons learned. Use for building an action memory that helps "
                "the agent learn from past execution patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["tool_call", "decision", "spawn", "error_handling"],
                        "description": "Type of action",
                        "default": "tool_call",
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the tool called",
                    },
                    "tool_args_summary": {
                        "type": "string",
                        "description": "Brief summary of tool arguments",
                    },
                    "tool_result_summary": {
                        "type": "string",
                        "description": "Brief summary of tool result",
                    },
                    "decision_rationale": {
                        "type": "string",
                        "description": "Why this action was chosen",
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["success", "failure", "partial", "unknown"],
                        "description": "Result of the action",
                        "default": "unknown",
                    },
                    "lesson_learned": {
                        "type": "string",
                        "description": "What was learned from this action",
                    },
                    "parent_task_id": {
                        "type": "string",
                        "description": "Task ID this action belongs to",
                    },
                    "agent_role": {
                        "type": "string",
                        "description": "Agent role: leaf, orchestrator, reviewer",
                    },
                    "duration_ms": {
                        "type": "integer",
                        "description": "Execution duration in milliseconds",
                    },
                    "turn_index": {
                        "type": "integer",
                        "description": "Turn number when this action occurred",
                    },
                },
                "required": ["action_type"],
            },
        },
    ]
