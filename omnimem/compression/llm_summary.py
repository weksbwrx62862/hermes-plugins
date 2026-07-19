"""LLM Summary — LLM 结构化摘要。

第四层压缩，需要 LLM 调用：
  - 6字段结构化摘要（参考 ReMe）
  - 目标/进度/决策/关键信息/开放问题/下一步

此模块提供接口，实际 LLM 调用由上层 (provider) 负责。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StructuredSummary:
    """6字段结构化摘要（参考 ReMe 设计）。"""

    goal: str = ""  # 当前目标
    progress: str = ""  # 进度状态
    decisions: str = ""  # 关键决策
    key_info: str = ""  # 关键信息
    open_issues: str = ""  # 开放问题
    next_steps: str = ""  # 下一步

    def to_text(self) -> str:
        """渲染为文本。"""
        parts = []
        if self.goal:
            parts.append(f"Goal: {self.goal}")
        if self.progress:
            parts.append(f"Progress: {self.progress}")
        if self.decisions:
            parts.append(f"Decisions: {self.decisions}")
        if self.key_info:
            parts.append(f"Key Info: {self.key_info}")
        if self.open_issues:
            parts.append(f"Open Issues: {self.open_issues}")
        if self.next_steps:
            parts.append(f"Next Steps: {self.next_steps}")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, str]:
        return {
            "goal": self.goal,
            "progress": self.progress,
            "decisions": self.decisions,
            "key_info": self.key_info,
            "open_issues": self.open_issues,
            "next_steps": self.next_steps,
        }


# LLM 摘要的 prompt 模板
_SUMMARY_PROMPT = """Analyze the following conversation/messages and produce a structured summary with these 6 fields:

1. **Goal**: What is the user trying to accomplish?
2. **Progress**: What has been done so far?
3. **Decisions**: What key decisions were made?
4. **Key Info**: What important facts/preferences/constraints were established?
5. **Open Issues**: What problems remain unresolved?
6. **Next Steps**: What should happen next?

Messages:
{messages}

Respond in JSON format with keys: goal, progress, decisions, key_info, open_issues, next_steps
"""


def llm_summarize(messages: str, llm_call_fn: Callable[[str], str] | None = None) -> StructuredSummary:
    """使用 LLM 生成结构化摘要。

    Args:
        messages: 消息文本
        llm_call_fn: LLM 调用函数 (prompt) -> response_text

    Returns:
        结构化摘要
    """
    if llm_call_fn is None:
        # 无 LLM 调用函数时，使用简单提取
        return _extract_without_llm(messages)

    try:
        prompt = _SUMMARY_PROMPT.format(messages=messages[:3000])
        response = llm_call_fn(prompt)
        return _parse_llm_response(response)
    except Exception as e:
        logger.warning("LLM summary failed: %s", e)
        return _extract_without_llm(messages)


async def async_llm_summarize(
    content: str,
    llm_fn: Callable[[str], str] | Callable[[str], object] | None = None,
) -> StructuredSummary:
    """异步 LLM 摘要 — 使用 asyncio.to_thread 包装同步 LLM 调用。

    与同步 llm_summarize 逻辑一致，但 LLM 调用异步执行，避免阻塞事件循环。
    支持传入同步函数或异步函数（协程函数）。

    Args:
        content: 消息文本
        llm_fn: LLM 调用函数，可为同步 (prompt) -> str 或异步 (prompt) -> str。
                None 时使用无 LLM 的简单提取。

    Returns:
        结构化摘要（与同步版本返回类型一致）
    """
    if llm_fn is None:
        # 无 LLM 调用函数时，使用简单提取
        return _extract_without_llm(content)

    try:
        prompt = _SUMMARY_PROMPT.format(messages=content[:3000])
        # ★ 判断 llm_fn 是同步还是异步函数
        if asyncio.iscoroutinefunction(llm_fn):
            response = await llm_fn(prompt)
        else:
            # 同步函数包装为异步，避免阻塞事件循环
            response = await asyncio.to_thread(llm_fn, prompt)
        return _parse_llm_response(response)
    except Exception as e:
        logger.warning("Async LLM summary failed: %s", e)
        return _extract_without_llm(content)


def _extract_without_llm(messages: str) -> StructuredSummary:
    """无 LLM 时的简单提取。"""
    lines = messages.strip().split("\n")
    # 提取决策
    decisions = []
    for line in lines:
        lower = line.lower()
        if any(m in lower for m in ["决定", "选择", "decided", "chose"]):
            decisions.append(line[:100])

    return StructuredSummary(
        goal=lines[0][:100] if lines else "",
        progress="See conversation history",
        decisions="; ".join(decisions[:3]) if decisions else "",
        key_info="",
        open_issues="",
        next_steps="",
    )


def _parse_llm_response(response: str) -> StructuredSummary:
    """解析 LLM 响应为结构化摘要。"""
    import json

    try:
        # 尝试直接解析 JSON
        data = json.loads(response)
        return StructuredSummary(
            goal=data.get("goal", ""),
            progress=data.get("progress", ""),
            decisions=data.get("decisions", ""),
            key_info=data.get("key_info", ""),
            open_issues=data.get("open_issues", ""),
            next_steps=data.get("next_steps", ""),
        )
    except json.JSONDecodeError:
        # 尝试从 markdown 代码块中提取 JSON
        import re

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return StructuredSummary(
                    goal=data.get("goal", ""),
                    progress=data.get("progress", ""),
                    decisions=data.get("decisions", ""),
                    key_info=data.get("key_info", ""),
                    open_issues=data.get("open_issues", ""),
                    next_steps=data.get("next_steps", ""),
                )
            except json.JSONDecodeError:
                logger.debug("LLMSummary: markdown JSON block parse failed, using raw response as key_info")

        # 最终回退：将整个响应作为 key_info
        return StructuredSummary(key_info=response[:500])
