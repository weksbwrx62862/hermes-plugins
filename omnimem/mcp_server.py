from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any


class OmniMemMCPServer:
    name = "omnimem"

    def __init__(self, storage_dir: str | None = None, config: dict | None = None):
        from omnimem.sdk import OmniMemSDK

        self._sdk = OmniMemSDK(storage_dir=storage_dir, config=config)

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "omni_memorize",
                "description": (
                    "Store a memory in OmniMem. Use for important facts, decisions, "
                    "corrections, user preferences, or any information worth recalling "
                    "in future sessions. Specify the type (fact/preference/correction/"
                    "skill/procedural) and confidence level (1-5)."
                ),
                "inputSchema": {
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
                            "description": "Type of memory. action: agent operations. reasoning: lessons learned.",
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
                "inputSchema": {
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
                "name": "omni_reflect",
                "description": (
                    "Reflect on accumulated memories to generate deeper insights. "
                    "Consolidates raw facts into observations and mental models. "
                    "Use when you need to synthesize patterns from past experiences."
                ),
                "inputSchema": {
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
                "inputSchema": {
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
                            ],
                            "description": "Governance action to perform",
                        },
                        "target": {
                            "type": "string",
                            "description": "Memory ID or query for the action",
                        },
                        "params": {
                            "type": "object",
                            "description": "Additional parameters for the action",
                            "properties": {
                                "level": {
                                    "type": "string",
                                    "enum": ["public", "team", "personal", "secret"],
                                    "description": "Privacy level for set_privacy",
                                },
                            },
                        },
                    },
                    "required": ["action"],
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "omni_memorize":
            result = self._sdk.memorize(**arguments)
        elif name == "omni_recall":
            result = self._sdk.recall(**arguments)
        elif name == "omni_reflect":
            result = self._sdk.reflect(**arguments)
        elif name == "omni_govern":
            result = self._sdk.govern(**arguments)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
        return json.dumps(result, ensure_ascii=False)

    def close(self) -> None:
        self._sdk.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniMem MCP Server")
    parser.add_argument("--storage-dir", default=None)
    args = parser.parse_args()

    try:
        import mcp.server.stdio
        import mcp.types as mcp_types
        from mcp.server import Server
    except ImportError:
        raise SystemExit("pip install omnimem[mcp]")

    mcp_impl = OmniMemMCPServer(storage_dir=args.storage_dir)
    server = Server("omnimem")

    @server.list_tools()
    async def _list_tools():
        return [
            mcp_types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in mcp_impl.list_tools()
        ]

    @server.call_tool()
    async def _call_tool(name, arguments):
        result = mcp_impl.call_tool(name, arguments)
        return [mcp_types.TextContent(type="text", text=result)]

    async def _run():
        async with mcp.server.stdio.stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())
