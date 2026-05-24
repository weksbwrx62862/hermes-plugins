"""CoreBlock — 常驻 LLM 上下文的工作记忆块。

参考 Letta Core Memory Block 设计，分为三个区域：
  - Identity Block: Soul/Identity/User 三元人格
  - Context Block: 当前任务上下文（动态更新）
  - Plan Block: 当前计划/待办（动态更新）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoreBlock:
    """常驻 LLM 上下文的工作记忆块。"""

    identity_block: str = ""
    context_block: str = ""
    plan_block: str = ""

    def to_prompt_text(self) -> str:
        """渲染为 system prompt 片段。"""
        parts = []
        if self.identity_block:
            parts.append(f"### Identity\n{self.identity_block}")
        if self.context_block:
            parts.append(f"### Current Context\n{self.context_block}")
        if self.plan_block:
            parts.append(f"### Plan\n{self.plan_block}")
        return "\n\n".join(parts)

    def update_context(self, context: str) -> None:
        """更新当前上下文。"""
        self.context_block = context

    def update_plan(self, plan: str) -> None:
        """更新当前计划。"""
        self.plan_block = plan
