"""LLM-as-Memory-Manager — LLM 驱动的记忆决策引擎。

替代传统规则去重机制，通过 LLM 理解语义关系来决定记忆的增删改操作。
核心能力：
  - 语义理解：判断新内容与已有记忆的深层关系（补充/矛盾/重复/无关）
  - 矛盾检测：当新内容与已有记忆矛盾时，返回 UPDATE 并标记旧内容过时
  - 决策解释：每次决策附带 reason，便于审计和调试

四种决策：
  ADD    — 新增记忆（与已有记忆无关或为补充信息）
  UPDATE — 更新已有记忆（内容矛盾或需要合并）
  DELETE — 删除已有记忆（旧内容完全过时/错误）
  NONE   — 无需操作（已有记忆已充分覆盖）
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from omnimem.utils.llm_client import AsyncLLMClient

logger = logging.getLogger(__name__)


class MemoryAction(str, Enum):
    """记忆决策动作枚举。"""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NONE = "NONE"


@dataclass
class MemoryDecision:
    """LLM 记忆决策结果。

    Attributes:
        action: 决策动作（ADD/UPDATE/DELETE/NONE）
        target_memory_id: 目标记忆 ID（UPDATE/DELETE 时必填）
        reason: 决策理由
        updated_content: 更新后的内容（UPDATE 时必填，合并/修正后的完整内容）
    """

    action: MemoryAction = MemoryAction.NONE
    target_memory_id: str = ""
    reason: str = ""
    updated_content: str = ""

    def to_dedup_result(self) -> dict[str, Any]:
        """转换为与 _semantic_dedup 兼容的返回格式。"""
        if self.action == MemoryAction.ADD:
            return {"action": "create"}
        if self.action == MemoryAction.UPDATE:
            return {
                "action": "update",
                "existing_id": self.target_memory_id,
                "reason": self.reason,
                "updated_content": self.updated_content,
            }
        if self.action == MemoryAction.DELETE:
            return {
                "action": "delete",
                "existing_id": self.target_memory_id,
                "reason": self.reason,
            }
        return {
            "action": "skip",
            "existing_id": self.target_memory_id,
            "reason": self.reason,
        }


_DECISION_SYSTEM_PROMPT = """你是一个记忆管理决策引擎。你的任务是分析新内容与已有记忆的关系，并做出最优决策。

## 决策选项

1. **ADD** — 新增记忆
   - 新内容与已有记忆无关
   - 新内容是对已有记忆的有价值补充（非重复）
   - 已有记忆无法通过简单更新来涵盖新信息

2. **UPDATE** — 更新已有记忆
   - 新内容与已有记忆矛盾（如旧信息过时、错误）
   - 新内容是对已有记忆的精确修正或扩展
   - 合并后的内容应包含新旧信息的完整版本
   - ⚠️ 矛盾检测：当新内容明确否定或修正已有记忆时，必须返回 UPDATE，
     并在 updated_content 中标注旧内容过时（如"[已过时: 原内容xxx] 新内容yyy"）

3. **DELETE** — 删除已有记忆
   - 已有记忆完全错误或过时，无保留价值
   - 新内容完全替代旧内容，无需合并

4. **NONE** — 无需操作
   - 新内容与已有记忆语义完全重复，无新增信息
   - 已有记忆已充分覆盖新内容

## 输出格式

严格输出 JSON，不要包含任何其他文字：
```json
{
  "action": "ADD|UPDATE|DELETE|NONE",
  "target_memory_id": "UPDATE/DELETE时填写目标记忆ID，否则为空",
  "reason": "决策理由的简要说明",
  "updated_content": "UPDATE时填写合并/修正后的完整内容，否则为空"
}
```

## 关键原则

- 宁可 ADD 也不要错误地 UPDATE：如果新信息只是相关但独立，应 ADD
- 矛盾必须标记：检测到矛盾时，UPDATE 的 updated_content 必须明确标注旧内容过时
- 保守删除：DELETE 仅在旧内容完全无价值时使用
- 短内容宽容：对于简短的事实（<30字），相似度阈值应更高才判定为重复"""


_DECISION_USER_TEMPLATE = """## 新内容
内容：{content}
类型：{memory_type}

## 已有相似记忆
{memories_section}

请分析新内容与已有记忆的关系，做出决策。"""


class LLMMemoryManager:
    """LLM 驱动的记忆决策管理器。

    通过 LLM 分析新内容与已有记忆的语义关系，替代传统的规则去重机制。
    当 LLM 客户端不可用时，自动回退到规则判断。
    """

    def __init__(self, llm_client: AsyncLLMClient | None, config: Any = None):
        self._llm_client = llm_client
        self._config = config
        self._max_candidates = 5
        self._max_content_length = 500
        self._llm_timeout = 10.0
        self._llm_max_tokens = 400
        if config:
            self._max_candidates = getattr(config, "get", lambda _k, d: d)("llm_memory_max_candidates", 5)
            self._max_content_length = getattr(config, "get", lambda _k, d: d)("llm_memory_max_content_length", 500)
            self._llm_timeout = getattr(config, "get", lambda _k, d: d)("llm_memory_timeout", 10.0)
            self._llm_max_tokens = getattr(config, "get", lambda _k, d: d)("llm_memory_max_tokens", 400)

    @property
    def is_available(self) -> bool:
        """LLM 客户端是否可用。"""
        return self._llm_client is not None

    def decide(
        self,
        content: str,
        memory_type: str,
        existing_memories: list[dict[str, Any]],
    ) -> MemoryDecision:
        """调用 LLM 分析新内容与已有记忆的关系，返回决策。

        Args:
            content: 待存储的新内容
            memory_type: 记忆类型（fact/preference/skill 等）
            existing_memories: 已有相似记忆列表，每项包含
                content、memory_id、type、confidence 等字段

        Returns:
            MemoryDecision 决策结果
        """
        if not self._llm_client:
            logger.debug("LLMMemoryManager: LLM 客户端不可用，跳过决策")
            return MemoryDecision(action=MemoryAction.ADD, reason="LLM 客户端不可用，默认新增")

        if not existing_memories:
            return MemoryDecision(action=MemoryAction.ADD, reason="无已有相似记忆，直接新增")

        prompt = self._build_decision_prompt(content, memory_type, existing_memories)
        try:
            response = self._llm_client.call_sync(
                prompt=prompt,
                system=_DECISION_SYSTEM_PROMPT,
                max_tokens=self._llm_max_tokens,
                temperature=0.1,
                use_cache=False,
            )
            if not response or not response.content:
                logger.warning("LLMMemoryManager: LLM 返回为空，默认新增")
                return MemoryDecision(action=MemoryAction.ADD, reason="LLM 返回为空")
            return self._parse_decision(response.content)
        except Exception as e:
            logger.warning("LLMMemoryManager: LLM 调用失败 (%s)，默认新增", e)
            return MemoryDecision(action=MemoryAction.ADD, reason=f"LLM 调用异常: {e}")

    def _build_decision_prompt(
        self,
        content: str,
        memory_type: str,
        existing_memories: list[dict[str, Any]],
    ) -> str:
        """构建决策 prompt，包含已有记忆列表和决策选项。

        Args:
            content: 新内容
            memory_type: 记忆类型
            existing_memories: 已有相似记忆列表

        Returns:
            完整的用户 prompt
        """
        truncated_content = content[:self._max_content_length]
        if len(content) > self._max_content_length:
            truncated_content += "...(已截断)"

        memories_lines = []
        for i, m in enumerate(existing_memories[:self._max_candidates], 1):
            mem_content = m.get("content", "")[:self._max_content_length]
            if len(m.get("content", "")) > self._max_content_length:
                mem_content += "...(已截断)"
            mem_id = m.get("memory_id", "unknown")
            mem_type = m.get("type", m.get("memory_type", "unknown"))
            mem_confidence = m.get("confidence", "N/A")
            memories_lines.append(
                f"{i}. [ID: {mem_id}] 类型: {mem_type} | 置信度: {mem_confidence}\n   内容: {mem_content}"
            )

        memories_section = "\n".join(memories_lines) if memories_lines else "（无相似记忆）"

        return _DECISION_USER_TEMPLATE.format(
            content=truncated_content,
            memory_type=memory_type,
            memories_section=memories_section,
        )

    def _parse_decision(self, llm_response: str) -> MemoryDecision:
        """解析 LLM 返回的决策 JSON。

        支持多种格式容错：
          - 纯 JSON
          - Markdown 代码块包裹的 JSON
          - JSON 前后带有多余文字

        Args:
            llm_response: LLM 的原始返回文本

        Returns:
            MemoryDecision 解析后的决策
        """
        json_str = llm_response.strip()

        # 尝试提取 Markdown 代码块中的 JSON
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", json_str, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1).strip()

        # 尝试提取第一个完整的 JSON 对象
        brace_start = json_str.find("{")
        brace_end = json_str.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            json_str = json_str[brace_start : brace_end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("LLMMemoryManager: JSON 解析失败 (%s)，原始响应: %s", e, llm_response[:200])
            return MemoryDecision(action=MemoryAction.ADD, reason="JSON 解析失败，默认新增")

        action_str = data.get("action", "ADD").upper()
        try:
            action = MemoryAction(action_str)
        except ValueError:
            logger.warning("LLMMemoryManager: 未知动作 '%s'，默认新增", action_str)
            return MemoryDecision(action=MemoryAction.ADD, reason=f"未知动作 {action_str}")

        decision = MemoryDecision(
            action=action,
            target_memory_id=data.get("target_memory_id", ""),
            reason=data.get("reason", ""),
            updated_content=data.get("updated_content", ""),
        )

        # 校验：UPDATE/DELETE 必须有 target_memory_id
        if action in (MemoryAction.UPDATE, MemoryAction.DELETE) and not decision.target_memory_id:
            logger.warning(
                "LLMMemoryManager: %s 决策缺少 target_memory_id，回退为 ADD",
                action.value,
            )
            decision.action = MemoryAction.ADD
            decision.reason = f"原决策 {action.value} 缺少目标 ID，回退为新增"

        # 校验：UPDATE 必须有 updated_content
        if action == MemoryAction.UPDATE and not decision.updated_content:
            logger.warning("LLMMemoryManager: UPDATE 决策缺少 updated_content，回退为 ADD")
            decision.action = MemoryAction.ADD
            decision.reason = "UPDATE 决策缺少更新内容，回退为新增"

        logger.info(
            "LLMMemoryManager: 决策=%s, target=%s, reason=%s",
            decision.action.value,
            decision.target_memory_id,
            decision.reason[:100],
        )
        return decision
