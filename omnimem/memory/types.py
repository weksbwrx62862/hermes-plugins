"""记忆数据模型。"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypedDict

try:
    from typing import NotRequired  # Python 3.11+
except ImportError:
    from typing_extensions import NotRequired  # Python 3.10 backport


class MemoryType(str, enum.Enum):
    """记忆类型枚举。"""

    FACT = "fact"
    PREFERENCE = "preference"
    CORRECTION = "correction"
    SKILL = "skill"
    PROCEDURAL = "procedural"
    EVENT = "event"
    ACTION = "action"       # Agent 的工具调用/决策行为
    REASONING = "reasoning" # 推理链条/踩过的坑/经验教训


class PrivacyLevel(str, enum.Enum):
    """隐私级别枚举。"""

    PUBLIC = "public"
    TEAM = "team"
    PERSONAL = "personal"
    SECRET = "secret"


@dataclass
class MemoryEntry:
    """一条记忆的完整数据模型。"""

    memory_id: str
    content: str
    memory_type: MemoryType = MemoryType.FACT
    confidence: int = 3
    privacy: PrivacyLevel = PrivacyLevel.PERSONAL
    scope: str = "personal"
    wing: str = ""
    room: str = ""
    provenance: dict[str, Any] | None = None
    stored_at: datetime | None = None
    trust: float = 0.5  # 信任评分 0.0-1.0，反馈驱动 ±0.05/0.10
    heat: str = "neutral"  # 热度分类: neutral/hot/warm/cold
    upgraded_to_wiki: bool = False  # 是否已升级到 Wiki
    wiki_page_path: str = ""  # Wiki 页面路径（升级后填写）
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于序列化）。"""
        result: dict[str, Any] = {
            "memory_id": self.memory_id,
            "content": self.content,
            "type": self.memory_type.value,
            "confidence": self.confidence,
            "trust": self.trust,
            "privacy": self.privacy.value,
            "scope": self.scope,
            "wing": self.wing,
            "room": self.room,
            "stored_at": self.stored_at.isoformat() if self.stored_at else None,
            "heat": self.heat,
        }
        if self.upgraded_to_wiki:
            result["upgraded_to_wiki"] = True
            result["wiki_page_path"] = self.wiki_page_path
        if self.provenance:
            result["provenance"] = self.provenance
        result.update(self.metadata)
        return result


class RetrievalResult(TypedDict):
    """检索结果。"""

    memory_id: str
    content: str
    score: float
    source: str  # 检索来源：vector/bm25/catalog/graph
    wing: str
    room: str
    type: str  # 记忆类型
    degraded: NotRequired[bool]  # 降级标记


class SignalResult(TypedDict):
    """信号分析结果。"""

    category: str
    confidence: float
    keywords: list[str]
    reasoning: NotRequired[str]


class GovernActionArgs(TypedDict):
    """治理动作参数。"""

    action: str
    target: NotRequired[str]
    params: NotRequired[dict[str, Any]]


class StoreEntry(TypedDict):
    """存储条目。"""

    memory_id: str
    wing: str
    room: str
    content: str
    type: str
    confidence: int
    privacy: str
    stored_at: str
    provenance: NotRequired[str]
