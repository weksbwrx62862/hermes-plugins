"""DeepMemoryFacade — L3 深层记忆。

封装: ConsolidationEngine, KnowledgeGraph, ReflectEngine
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omnimem.deep.consolidation import ConsolidationEngine
from omnimem.deep.knowledge_graph import KnowledgeGraph
from omnimem.deep.reflect import ReflectEngine


class DeepMemoryFacade:

    def __init__(self, data_dir: Path, config: Any, recall_fn: Any, llm_fn: Any = None, llm_client: Any = None):
        deep_dir = data_dir / "deep"
        self._consolidation = ConsolidationEngine(
            deep_dir,
            fact_threshold=config.get("fact_threshold", 10),
            llm_client=llm_client,
        )
        self._knowledge_graph = KnowledgeGraph(deep_dir)
        self._reflect_engine = ReflectEngine(
            deep_dir,
            consolidation_engine=self._consolidation,
            recall_fn=recall_fn,
            llm_fn=llm_fn,
            llm_client=llm_client,
        )

    @property
    def consolidation(self) -> ConsolidationEngine:
        return self._consolidation

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        return self._knowledge_graph

    @property
    def reflect_engine(self) -> ReflectEngine:
        return self._reflect_engine

    def close(self) -> None:
        """关闭深层记忆资源。"""
        if self.knowledge_graph:
            self.knowledge_graph.close()
        if self.consolidation:
            self.consolidation.close()
        if hasattr(self, "reflect_engine") and self.reflect_engine:
            self.reflect_engine.close()
