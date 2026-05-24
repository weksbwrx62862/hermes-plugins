"""MemoryStoreService — 存储服务层，封装所有存储操作。

OPT-4: Extracts storage-related helper methods from OmniMemProvider to reduce
main class size and improve maintainability.

Responsibilities:
  - Signal-driven memory storage (correction, reinforcement, fact)
  - Auto-checkpoint and emergency save
  - Session memory extraction
  - Delegation record storage
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class MemoryStoreService:
    """存储服务层，封装所有与 DrawerClosetStore 的交互。

    将 OmniMemProvider 中的 _store_*, _auto_checkpoint, _emergency_save 等方法
    提取到此类，使主类专注于工具路由和生命周期管理。
    """

    def __init__(
        self,
        store: Any,
        perception: Any,
        provenance: Any,
        session_id: str = "",
        turn_count: int = 0,
    ):
        """Initialize store service.

        Args:
            store: DrawerClosetStore instance
            perception: PerceptionEngine instance (for _extract_core_fact)
            provenance: ProvenanceTracker instance
            session_id: Current session ID
            turn_count: Starting turn count
        """
        self._store = store
        self._perception = perception
        self._provenance = provenance
        self._session_id = session_id
        self._turn_count = turn_count
        self._last_save_turn = 0

    # ─── Properties ──────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @turn_count.setter
    def turn_count(self, value: int) -> None:
        self._turn_count = value

    @property
    def last_save_turn(self) -> int:
        return self._last_save_turn

    @last_save_turn.setter
    def last_save_turn(self, value: int) -> None:
        self._last_save_turn = value

    # ─── Core fact extraction delegate ───────────────────────

    def extract_core_fact(self, text: str) -> str:
        """Delegate to perception engine for core fact extraction."""
        return str(self._perception._extract_core_fact(text))

    # ─── Signal-driven storage ───────────────────────────────

    def store_correction(self, signals: Any, user_content: str) -> str | None:
        """存储纠错记忆 — 精炼版：只存纠正目标而非整段对话。

        Returns:
            memory_id if stored, None otherwise
        """
        core = signals.correction_target or self.extract_core_fact(user_content)
        result = self._store.add(
            wing="personal",
            room="correction",
            content=f"纠正: {core}",
            memory_type="correction",
            confidence=4,
            privacy="personal",
            provenance=self._provenance.track(core, source=self._session_id, method="correction"),
        )
        return str(result) if result else None

    def store_reinforcement(self, signals: Any, user_content: str) -> str | None:
        """存储正反馈记忆 — 精炼版：只存强化目标。"""
        core = signals.reinforcement_target or self.extract_core_fact(user_content)
        result = self._store.add(
            wing="personal",
            room="reinforcement",
            content=f"确认: {core}",
            memory_type="preference",
            confidence=5,
            privacy="personal",
            provenance=self._provenance.track(
                core, source=self._session_id, method="reinforcement"
            ),
        )
        return str(result) if result else None

    def store_fact(self, signals: Any, user_content: str) -> str | None:
        """存储一般事实 — 精炼版：用感知引擎提炼的 fact_content。"""
        content = signals.fact_content or self.extract_core_fact(user_content)
        mem_type = "preference" if signals.has_preference else "fact"
        result = self._store.add(
            wing="personal",
            room=mem_type,
            content=content,
            memory_type=mem_type,
            confidence=3,
            privacy="personal",
            provenance=self._provenance.track(content, source=self._session_id, method="fact"),
        )
        return str(result) if result else None

    # ─── Auto checkpoint ─────────────────────────────────────

    def auto_checkpoint(self, user_content: str, save_interval: int = 15) -> bool:
        """定期自动存档 — 精炼版：只存核心事实而非原文。

        Args:
            user_content: Raw user content
            save_interval: Turns between auto-saves

        Returns:
            True if checkpoint was created
        """
        self._turn_count += 1
        if self._turn_count - self._last_save_turn < save_interval:
            return False

        core = self.extract_core_fact(user_content) if user_content else f"turn-{self._turn_count}"
        self._store.add(
            wing="auto",
            room=f"turn-{self._turn_count}",
            content=f"[Turn {self._turn_count}] {core}",
            memory_type="event",
            confidence=2,
            privacy="personal",
            provenance=self._provenance.track(
                core, source=self._session_id, method="auto_checkpoint"
            ),
        )
        self._last_save_turn = self._turn_count
        logger.warning("Auto checkpoint at turn %d", self._turn_count)
        return True

    # ─── Emergency save ──────────────────────────────────────

    def emergency_save(self, messages: list[dict[str, Any]]) -> str:
        """压缩前紧急保存 — 精炼版：只存关键事实摘要，不存原文。

        Returns:
            Status message describing what was saved
        """
        saved_parts = []
        for msg in messages[-20:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            if not content or role != "user":
                continue
            core = self.extract_core_fact(content)
            if core and len(core) > 5:
                saved_parts.append(core)
            if len(saved_parts) >= 10:
                break

        if saved_parts:
            self._store.add(
                wing="auto",
                room=f"compress-save-{self._turn_count}",
                content=f"[Emergency save turn {self._turn_count}] " + " | ".join(saved_parts),
                memory_type="event",
                confidence=2,
                privacy="personal",
                provenance=self._provenance.track(
                    "pre_compress", source=self._session_id, method="auto_emergency"
                ),
            )

        return f"Emergency saved {len(saved_parts)} core facts before compression."

    # ─── Session memory extraction ───────────────────────────

    def extract_session_memories(
        self,
        messages: list[dict[str, Any]],
        strip_system_injections: Callable[[str], str],
        should_store: Callable[[str], bool],
        memorize_fn: Callable[[dict[str, Any]], None],
    ) -> int:
        """会话结束时从完整对话中提取遗漏的记忆。

        Args:
            messages: Full message list
            strip_system_injections: Function to clean prefetch injections
            should_store: Function to filter storable content
            memorize_fn: Callback to store extracted memory

        Returns:
            Number of memories extracted
        """
        count = 0
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role != "user" or not isinstance(content, str):
                continue
            clean = strip_system_injections(content)
            if len(clean) < 50:
                continue
            implicit = self._perception.extract_implicit_memories(clean)
            for mem in implicit:
                if len(mem) > 100:
                    refined = self.extract_core_fact(mem)
                    if refined and len(refined) < len(mem):
                        mem = refined
                if not should_store(mem):
                    continue
                memorize_fn(
                    {
                        "content": mem,
                        "memory_type": "fact",
                        "confidence": 2,
                        "scope": "personal",
                        "privacy": "personal",
                    }
                )
                count += 1
        return count

    # ─── Delegation storage ──────────────────────────────────

    def store_delegation(self, task: str, result: str, child_session_id: str = "") -> str | None:
        """存储子 Agent 委托记录。"""
        mem_result = self._store.add(
            wing="delegation",
            room=child_session_id[:8] if child_session_id else "unknown",
            content=f"Delegated: {task[:200]}\nResult: {result[:300]}",
            memory_type="event",
            confidence=3,
            privacy="team",
            provenance=self._provenance.track(task, source=self._session_id, method="delegation"),
        )
        return str(mem_result) if mem_result else None
