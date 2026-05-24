"""向量库工厂函数。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omnimem.retrieval.vector_store import ChromaDBStore, QdrantStore, VectorStore, _CachedEmbeddingFunction

logger = logging.getLogger(__name__)


def create_vector_store(backend: str = "chromadb", **kwargs: Any) -> VectorStore:
    if backend == "chromadb":
        persist_dir = kwargs.pop("persist_dir", kwargs.pop("data_dir", "/tmp/omnimem/retrieval/chroma"))
        if not isinstance(persist_dir, Path):
            persist_dir = Path(persist_dir)
        collection_name = kwargs.pop("collection_name", "omnimem")
        embedding_fn = kwargs.pop("embedding_fn", None)
        if embedding_fn is None:
            try:
                cache_path = persist_dir.parent / "embedding_cache.json"
                embedding_fn = _CachedEmbeddingFunction(cache_path=cache_path)
            except Exception as e:
                logger.warning("vector_factory create embedding_fn failed: %s", e)
        return ChromaDBStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
            embedding_fn=embedding_fn,
        )
    elif backend == "qdrant":
        collection_name = kwargs.pop("collection_name", "omnimem")
        url = kwargs.pop("qdrant_url", kwargs.pop("url", "localhost:6333"))
        api_key = kwargs.pop("api_key", None)
        return QdrantStore(
            collection_name=collection_name,
            url=url,
            api_key=api_key,
        )
    elif backend == "faiss":
        from omnimem.retrieval.faiss_store import FAISSStore
        persist_dir = kwargs.pop("persist_dir", kwargs.pop("data_dir", "/tmp/omnimem/retrieval/faiss"))
        if not isinstance(persist_dir, Path):
            persist_dir = Path(persist_dir)
        embedding_fn = kwargs.pop("embedding_fn", None)
        if embedding_fn is None:
            try:
                cache_path = persist_dir.parent / "embedding_cache.json"
                embedding_fn = _CachedEmbeddingFunction(cache_path=cache_path)
            except Exception as e:
                logger.warning("vector_factory create embedding_fn for faiss failed: %s", e)
        return FAISSStore(
            persist_dir=persist_dir,
            embedding_fn=embedding_fn,
        )
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")
