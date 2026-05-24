"""Agent 行为记忆模块。

对标 Memori / neo4j agent-memory：不只是记对话文本，而是记录 Agent 的
工具调用链路、决策路径、成功经验和失败教训。

提供:
  - ActionRecord: 行为记忆数据模型
  - ActionMemoryService: 记录 + 检索 + 经验提取
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ────────────── 数据模型 ──────────────


@dataclass
class ActionRecord:
    """一次 Agent 行为（工具调用 / 决策 / 子代理创建）的结构化记录。"""

    action_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_task_id: str = ""         # 所属任务 ID
    agent_role: str = "leaf"         # 执行者角色: leaf / orchestrator / reviewer
    action_type: str = "tool_call"   # tool_call / decision / spawn / error_handling
    tool_name: str = ""              # 工具名（action_type=tool_call 时）
    tool_args_summary: str = ""      # 参数摘要（脱敏截断）
    tool_result_summary: str = ""    # 结果摘要
    decision_rationale: str = ""     # 决策理由（为什么要这么做）
    outcome: str = "unknown"         # success / failure / partial / unknown
    lesson_learned: str = ""         # 从这次行为中学到的教训
    duration_ms: int = 0             # 执行耗时（毫秒）
    turn_index: int = 0              # 发生在第几轮迭代
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_content(self) -> str:
        """转换为可存储的文本。"""
        parts = [
            f"Agent ({self.agent_role}) 执行 {self.action_type}",
        ]
        if self.tool_name:
            parts.append(f"  工具: {self.tool_name}")
        if self.tool_args_summary:
            parts.append(f"  参数: {self.tool_args_summary}")
        if self.tool_result_summary:
            parts.append(f"  结果: {self.tool_result_summary}")
        if self.decision_rationale:
            parts.append(f"  理由: {self.decision_rationale}")
        if self.duration_ms:
            parts.append(f"  耗时: {self.duration_ms}ms")
        parts.append(f"  结果: {self.outcome}")
        if self.lesson_learned:
            parts.append(f"  教训: {self.lesson_learned}")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "parent_task_id": self.parent_task_id,
            "agent_role": self.agent_role,
            "action_type": self.action_type,
            "tool_name": self.tool_name,
            "tool_args_summary": self.tool_args_summary,
            "tool_result_summary": self.tool_result_summary,
            "decision_rationale": self.decision_rationale,
            "outcome": self.outcome,
            "lesson_learned": self.lesson_learned,
            "duration_ms": self.duration_ms,
            "turn_index": self.turn_index,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> ActionRecord:
        """从工具参数构造。"""
        return cls(
            action_id=args.get("action_id", uuid.uuid4().hex[:12]),
            parent_task_id=args.get("parent_task_id", ""),
            agent_role=args.get("agent_role", "leaf"),
            action_type=args.get("action_type", "tool_call"),
            tool_name=args.get("tool_name", ""),
            tool_args_summary=cls._truncate(args.get("tool_args_summary", ""), 200),
            tool_result_summary=cls._truncate(args.get("tool_result_summary", ""), 300),
            decision_rationale=args.get("decision_rationale", ""),
            outcome=args.get("outcome", "unknown"),
            lesson_learned=args.get("lesson_learned", ""),
            duration_ms=args.get("duration_ms", 0),
            turn_index=args.get("turn_index", 0),
        )

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    @staticmethod
    def from_memory_entry(entry: dict[str, Any]) -> ActionRecord | None:
        """从存储条目恢复。缺少关键元数据时返回 None。"""
        try:
            meta = entry.get("metadata", {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            if not meta or not isinstance(meta, dict):
                return None
            if "action_type" not in meta and "tool_name" not in meta:
                return None
            return ActionRecord(
                action_id=entry.get("memory_id", ""),
                parent_task_id=meta.get("parent_task_id", ""),
                agent_role=meta.get("agent_role", "leaf"),
                action_type=meta.get("action_type", "tool_call"),
                tool_name=meta.get("tool_name", ""),
                tool_args_summary=meta.get("tool_args_summary", ""),
                tool_result_summary=meta.get("tool_result_summary", ""),
                decision_rationale=meta.get("decision_rationale", ""),
                outcome=meta.get("outcome", "unknown"),
                lesson_learned=meta.get("lesson_learned", ""),
                duration_ms=meta.get("duration_ms", 0),
                turn_index=meta.get("turn_index", 0),
                timestamp=str(entry.get("stored_at", "")),
            )
        except Exception as e:
            logger.warning("ActionRecord from_memory_entry failed: %s", e)
            return None


# ────────────── 服务 ──────────────


class ActionMemoryService:
    """Agent 行为记忆的读写查询服务。"""

    def __init__(self, store: Any, index: Any, retriever: Any,
                 wing_room: Any, provenance: Any, forgetting: Any) -> None:
        self._store = store
        self._index = index
        self._retriever = retriever
        self._wing_room = wing_room
        self._provenance = provenance
        self._forgetting = forgetting

    def record_action(self, record: ActionRecord) -> str:
        """记录一次 Agent 行为。返回 memory_id。"""
        content = record.to_content()
        wing = self._wing_room.resolve_wing_from_privacy("private", "action")
        room = "action"
        metadata = record.to_dict()

        mid = self._store.add(
            wing=wing,
            room=room,
            content=content,
            memory_type="action",
            confidence=self._outcome_confidence(record.outcome),
            privacy="personal",
            metadata=json.dumps(metadata, ensure_ascii=False),
        )

        self._index.add(
            memory_id=mid,
            wing=wing,
            hall="action",
            room=room,
            content=content,
            summary=content[:200].replace("\n", " "),
            type="action",
            confidence=self._outcome_confidence(record.outcome),
            privacy="personal",
            scope="personal",
            stored_at=record.timestamp,
            provenance=json.dumps({"action_id": record.action_id}),
        )

        self._store.flush()
        self._forgetting.record_access(mid)

        logger.warning("ActionMemoryService: recorded %s id=%s", record.action_type, mid)
        return mid

    def query_actions(
        self,
        parent_task_id: str | None = None,
        agent_role: str | None = None,
        tool_name: str | None = None,
        outcome: str | None = None,
        limit: int = 20,
    ) -> list[ActionRecord]:
        """按维度检索行为记忆。"""
        results = self._retriever.search("agent action", max_tokens=3000)
        records: list[ActionRecord] = []

        for r in results:
            entry = self._store.get(r.get("memory_id", ""))
            if not entry:
                continue
            if entry.get("type") != "action":
                continue
            rec = ActionRecord.from_memory_entry(entry)
            if rec is None:
                continue

            # 过滤
            if parent_task_id and rec.parent_task_id != parent_task_id:
                continue
            if agent_role and rec.agent_role != agent_role:
                continue
            if tool_name and rec.tool_name != tool_name:
                continue
            if outcome and rec.outcome != outcome:
                continue

            records.append(rec)
            if len(records) >= limit:
                break

        return records

    def get_task_chain(self, parent_task_id: str) -> list[ActionRecord]:
        """获取某个任务的完整执行链路，按 turn_index 排序。"""
        records = self.query_actions(parent_task_id=parent_task_id, limit=100)
        records.sort(key=lambda r: r.turn_index)
        return records

    def get_failures(self, limit: int = 20) -> list[ActionRecord]:
        """获取所有失败的行为，用于经验提取。"""
        return self.query_actions(outcome="failure", limit=limit)

    def learn_from_failures(self, limit: int = 20) -> list[dict[str, str]]:
        """从失败中提取教训列表。"""
        failures = self.get_failures(limit=limit)
        lessons: list[dict[str, str]] = []
        for f in failures:
            lesson = f.lesson_learned or self._auto_extract_lesson(f)
            if lesson:
                lessons.append({
                    "tool": f.tool_name,
                    "error": f.tool_result_summary,
                    "lesson": lesson,
                    "task_id": f.parent_task_id,
                })
        return lessons

    # ── 内部方法 ──

    @staticmethod
    def _outcome_confidence(outcome: str) -> int:
        return {"success": 5, "failure": 3, "partial": 3, "unknown": 2}.get(outcome, 3)

    @staticmethod
    def _auto_extract_lesson(record: ActionRecord) -> str:
        """当用户未提供 lesson_learned 时自动生成。"""
        result = record.tool_result_summary.lower()
        if "timeout" in result or "timed out" in result:
            return f"工具 {record.tool_name} 超时，考虑减少参数或增加超时时间"
        if "permission" in result or "access denied" in result:
            return f"工具 {record.tool_name} 权限不足，检查凭据或权限配置"
        if "not found" in result:
            return f"工具 {record.tool_name} 目标不存在，检查参数是否正确"
        if "rate limit" in result:
            return f"工具 {record.tool_name} 被限流，需要添加重试或降低频率"
        return f"工具 {record.tool_name} 执行失败: {record.tool_result_summary[:80]}"
