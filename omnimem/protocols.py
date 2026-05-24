"""核心组件 Protocol 接口定义。

为 Store / Index / Retriever / Dedup / Provider 定义结构化子类型协议，
支持 isinstance() 运行时检查与静态类型推导。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StoreProtocol(Protocol):
    """记忆存储协议 — DrawerClosetStore 的结构化子类型。"""

    def add(
        self,
        wing: str,
        room: str,
        content: str,
        memory_type: str = "fact",
        confidence: int = 3,
        privacy: str = "personal",
        provenance: dict[str, Any] | None = None,
        vc: str = "",
        memory_id: str = "",
        **kwargs: Any,
    ) -> str: ...

    def get(self, memory_id: str) -> dict[str, Any] | None: ...

    def search(
        self,
        wing: str = "",
        room: str = "",
        memory_type: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]: ...

    def search_by_content(self, query: str, limit: int = 20) -> list[dict[str, Any]]: ...


@runtime_checkable
class IndexProtocol(Protocol):
    """三层索引协议 — ThreeLevelIndex 的结构化子类型。"""

    def add(
        self,
        memory_id: str,
        wing: str,
        hall: str,
        room: str,
        content: str,
        summary: str = "",
        type: str = "fact",
        confidence: int = 3,
        privacy: str = "personal",
        scope: str = "personal",
        stored_at: str = "",
        provenance: str = "",
        metadata: str = "",
    ) -> None: ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """混合检索协议 — HybridRetriever 的结构化子类型。"""

    def add(self, content: str, memory_id: str, metadata: dict[str, Any]) -> None: ...

    def search(
        self,
        query: str,
        max_tokens: int = 1500,
        mode: str = "rag",
        top_k: int = 10,
        store: Any = None,
        enable_trace: bool = False,
    ) -> list[dict[str, Any]]: ...

    def warmup(self) -> None: ...

    def flush(self) -> None: ...


@runtime_checkable
class DedupProtocol(Protocol):
    """语义去重协议 — SemanticDedupService 的结构化子类型。"""

    def semantic_dedup(
        self,
        content: str,
        memory_type: str,
        candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...


@runtime_checkable
class ProviderProtocol(Protocol):
    """记忆提供者协议 — OmniMemProvider 的结构化子类型。"""

    _store: StoreProtocol
    _index: IndexProtocol
    _retriever: RetrieverProtocol
    _saga: Any
    _semantic_dedup: DedupProtocol
    _wing_room: Any
    _conflict_resolver: Any
    _provenance: Any
    _forgetting: Any
    _temporal_decay: Any
    _privacy: Any
    _context_manager: Any
    _knowledge_graph: Any
    _consolidation: Any
    _kv_cache: Any

    def _should_store(self, content: str) -> bool: ...
