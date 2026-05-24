"""VectorRetriever — 向量检索。

使用 VectorStore 抽象接口，支持多后端切换（chromadb/qdrant）。
默认使用 ChromaDBStore（向后兼容）。
OPT-3: 支持自定义 Embedding Function，实现嵌入结果缓存。
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from omnimem.retrieval.vector_store import (
    ChromaDBStore,
    VectorStore,
    _CachedEmbeddingFunction,
    _emit,
)
from omnimem.retrieval.vector_factory import create_vector_store

logger = logging.getLogger(__name__)


class VectorRetriever:
    """向量检索，委托 VectorStore 抽象接口。"""

    def __init__(self, backend: str = "chromadb", data_dir: Path | None = None, embedding_model_path: str = ""):
        self._backend = backend
        self._data_dir = data_dir or Path("/tmp/omnimem/retrieval")
        self._store: VectorStore | None = None
        self._initialized = False
        self._embedding_fn: Any = None
        self._encoder = None
        self._embedding_model_path = embedding_model_path
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)
        if self._backend == "faiss":
            from omnimem.retrieval.faiss_store import FAISSStore
            try:
                cache_path = self._data_dir / "embedding_cache.json"
                self._embedding_fn = _CachedEmbeddingFunction(cache_path=cache_path, model_path=self._embedding_model_path)
            except Exception as e:
                logger.warning("Failed to create CachedEmbeddingFunction for faiss: %s, using default", e)
                self._embedding_fn = None
            self._store = FAISSStore(
                persist_dir=self._data_dir / "faiss",
                embedding_fn=self._embedding_fn,
            )
        elif self._backend == "chromadb":
            try:
                cache_path = self._data_dir / "embedding_cache.json"
                self._embedding_fn = _CachedEmbeddingFunction(cache_path=cache_path, model_path=self._embedding_model_path)
            except Exception as e:
                logger.warning("Failed to create CachedEmbeddingFunction: %s, using default", e)
                self._embedding_fn = None
            self._store = ChromaDBStore(
                collection_name="omnimem",
                persist_dir=self._data_dir / "chroma",
                embedding_fn=self._embedding_fn,
            )
        else:
            self._store = create_vector_store(
                backend=self._backend,
                persist_dir=self._data_dir / "chroma",
                data_dir=self._data_dir / "chroma",
            )
        self._initialized = True

    _CHUNK_SIZE = 500
    _CHUNK_OVERLAP = 100

    def add(self, content: str, memory_id: str, metadata: dict[str, Any]) -> None:
        self._add_single(content, memory_id, metadata)

    def add_batch(self, documents: list[dict[str, Any]]) -> None:
        self._ensure_initialized()
        if self._store is None:
            return
        all_ids: list[str] = []
        all_docs: list[str] = []
        all_metas: list[dict[str, str]] = []
        for doc in documents:
            content = doc.get("content", "")
            memory_id = doc.get("memory_id", "")
            if not content or not memory_id:
                continue
            meta = {
                k: str(v)
                for k, v in doc.items()
                if k not in ("content", "memory_id") and v is not None
            }
            if len(content) > self._CHUNK_SIZE:
                chunks = self._split_chunks(content, self._CHUNK_SIZE, self._CHUNK_OVERLAP)
                for i, chunk in enumerate(chunks):
                    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                    all_ids.append(f"{memory_id}_chunk{chunk_hash}")
                    all_docs.append(chunk)
                    all_metas.append(dict(meta, _parent_id=memory_id, _chunk_idx=str(i)))
            else:
                all_ids.append(memory_id)
                all_docs.append(content)
                all_metas.append(meta)
        if not all_ids:
            return
        try:
            self._store.add(ids=all_ids, documents=all_docs, metadatas=all_metas)
        except Exception as e:
            logger.warning("Vector add_batch failed: %s", e)

    def add_batch_optimized(self, entries: list[dict[str, Any]]) -> None:
        self._ensure_initialized()
        if self._store is None:
            return
        all_ids: list[str] = []
        all_docs: list[str] = []
        all_metas: list[dict[str, str]] = []
        for entry in entries:
            content = entry.get("content", "")
            memory_id = entry.get("memory_id", "")
            if not content or not memory_id:
                continue
            meta = {
                k: str(v)
                for k, v in entry.items()
                if k not in ("content", "memory_id") and v is not None
            }
            if len(content) > self._CHUNK_SIZE:
                chunks = self._split_chunks(content, self._CHUNK_SIZE, self._CHUNK_OVERLAP)
                for i, chunk in enumerate(chunks):
                    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                    all_ids.append(f"{memory_id}_chunk{chunk_hash}")
                    all_docs.append(chunk)
                    all_metas.append(dict(meta, _parent_id=memory_id, _chunk_idx=str(i)))
            else:
                all_ids.append(memory_id)
                all_docs.append(content)
                all_metas.append(meta)
        if not all_ids:
            return
        if self._embedding_fn is not None:
            try:
                embeddings = self._embedding_fn(all_docs)
                if isinstance(self._store, ChromaDBStore) and self._store._collection is not None:
                    self._store._collection.upsert(
                        ids=all_ids,
                        embeddings=embeddings,
                        documents=all_docs,
                        metadatas=all_metas,
                    )
                    self._store._persist_client()
                    return
            except Exception as e:
                logger.warning("Vector add_batch_optimized embedding pre-compute failed: %s", e)
        try:
            self._store.add(ids=all_ids, documents=all_docs, metadatas=all_metas)
        except Exception as e:
            logger.warning("Vector add_batch_optimized fallback failed: %s", e)

    def _add_single(self, content: str, memory_id: str, metadata: dict[str, Any]) -> None:
        self._ensure_initialized()
        if self._store is None:
            return
        try:
            meta = {k: str(v) for k, v in metadata.items() if v is not None}
            if len(content) > self._CHUNK_SIZE:
                chunks = self._split_chunks(content, self._CHUNK_SIZE, self._CHUNK_OVERLAP)
                ids = [
                    f"{memory_id}_chunk{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
                    for chunk in chunks
                ]
                metas = [
                    dict(meta, _parent_id=memory_id, _chunk_idx=str(i)) for i in range(len(chunks))
                ]
                self._store.add(ids=ids, documents=chunks, metadatas=metas)
            else:
                self._store.add(
                    ids=[memory_id],
                    documents=[content],
                    metadatas=[meta],
                )
            # ★ R25修复Minor-3：写入后立即 persist 确保向量索引可搜
            if isinstance(self._store, ChromaDBStore):
                self._store._persist_client()
        except Exception as e:
            logger.warning("Vector add failed for %s: %s", memory_id, e)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        self._ensure_initialized()
        if self._store is None:
            return []
        try:
            count = self._store.count()
            if count == 0:
                return []
            results = self._store.query(
                query_texts=[query],
                n_results=min(top_k, count),
            )
            output = []
            if results and results.get("documents"):
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
                ids = results.get("ids", [[]])[0] if results.get("ids") else [""] * len(docs)
                for doc_id, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
                    entry = dict(meta) if meta else {}
                    entry["content"] = doc
                    sim = 1.0 - dist
                    # ★ R29修复Minor-3：按类型动态调整相似度阈值
                    # secret/skill/procedural 内容与自然语言查询语义距离大，
                    # 使用统一阈值会导致这些类型零命中
                    # ★ BUG FIX: ChromaDB cosine 距离范围为 [0,2]，sim 通常在 [0, 0.5]
                    # 原阈值 0.25 太高，降到 0.05
                    mem_type = entry.get("type", "fact")
                    type_thresholds = {"secret": 0.03, "skill": 0.05, "procedural": 0.05}
                    threshold = type_thresholds.get(mem_type, 0.05)
                    if sim < threshold:
                        continue
                    entry["score"] = sim
                    if "memory_id" not in entry and doc_id:
                        entry["memory_id"] = doc_id
                    output.append(entry)

            output = self._merge_chunk_results(output)
            return output
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            return [{"degraded": True, "content": "", "score": 0.0, "memory_id": ""}]

    def count(self) -> int:
        self._ensure_initialized()
        if self._store is None:
            return 0
        try:
            return self._store.count()
        except Exception:
            return 0

    def warmup(self) -> None:
        """预热：启动时预加载模型和初始化 ChromaDB，避免首次搜索延迟。"""
        logger.info("VectorRetriever warmup: initializing...")
        t0 = time.time()
        success = True
        try:
            self._ensure_initialized()
            if self._embedding_fn is not None:
                self._embedding_fn(["warmup"])
                elapsed = time.time() - t0
                logger.info("SentenceTransformer model loaded in %.1fs", elapsed)
                _emit(f"[OmniMem] 嵌入模型就绪 ({elapsed:.1f}s), 内存缓存: {self._embedding_fn.cache_size} 条")
            if self._store is not None:
                doc_count = self._store.count()
                logger.info("ChromaDB initialized in %.1fs, docs=%d", time.time() - t0, doc_count)
                _emit(f"[OmniMem] ChromaDB 就绪: {doc_count} 条文档")
        except Exception as e:
            success = False
            logger.warning("VectorRetriever warmup failed (non-fatal): %s", e)
            _emit(f"[OmniMem] ⚠ 向量检索引擎预热失败: {e}")

        if success:
            logger.info("VectorRetriever warmup complete in %.1fs", time.time() - t0)

    def embed_text(self, text: str) -> list[float]:
        self._ensure_initialized()
        if self._embedding_fn is None:
            return []
        try:
            vecs = self._embedding_fn([text])
            return vecs[0] if vecs else []
        except Exception as e:
            logger.warning("VectorRetriever embed_text failed: %s", e)
            return []

    def flush(self) -> None:
        try:
            if isinstance(self._store, ChromaDBStore):
                self._store._persist_client()
        except Exception as e:
            logger.warning("ChromaDB persist failed: %s", e)
        if self._embedding_fn:
            try:
                self._embedding_fn.persist()
            except Exception as e:
                logger.warning("Embedding cache persist failed: %s", e)

    def delete(self, memory_id: str) -> None:
        """从向量索引中删除指定条目（包括分块）。

        ChromaDB 的分块 ID 格式为 {memory_id}_chunk{hash}，
        需要先查询所有匹配的 ID 再删除。
        """
        self._ensure_initialized()
        if self._store is None:
            return
        try:
            if isinstance(self._store, ChromaDBStore) and self._store._collection is not None:
                # 查询所有以 memory_id 开头的 ID（含分块）
                all_ids = self._store._collection.get(ids=None, include=[])["ids"]
                ids_to_delete = [
                    i for i in all_ids
                    if i == memory_id or i.startswith(f"{memory_id}_chunk")
                ]
                if ids_to_delete:
                    self._store.delete(ids_to_delete)
            else:
                self._store.delete([memory_id])
        except Exception as e:
            logger.warning("Vector delete failed for %s: %s", memory_id, e)

    def _split_chunks(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        if self._encoder is not None:
            max_tokens = chunk_size // 2
            overlap_tokens = overlap // 2
            return self._split_by_tokens(text, max_tokens, overlap_tokens)
        return self._split_by_chars(text, chunk_size, overlap)

    @staticmethod
    def _split_by_chars(text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                best_end = end
                para_pos = text.rfind("\n\n", start + chunk_size // 3, end)
                if para_pos > start:
                    best_end = para_pos + 2
                else:
                    for sep in ["。", "！", "？", ".", "!", "?"]:
                        sep_pos = text.rfind(sep, start + chunk_size // 2, end)
                        if sep_pos > start:
                            best_end = sep_pos + 1
                            break
                    else:
                        for sep in ["\n", "；", ";"]:
                            sep_pos = text.rfind(sep, start + chunk_size * 2 // 3, end)
                            if sep_pos > start:
                                best_end = sep_pos + 1
                                break
                        else:
                            comma_pos = text.rfind("，", start + chunk_size * 3 // 4, end)
                            if comma_pos > start:
                                best_end = comma_pos + 1
                end = best_end
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap if end < len(text) else end
            if start <= end - chunk_size:
                start = end - overlap
        return chunks if chunks else [text[:chunk_size]]

    def _split_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
        if self._encoder is None:
            return [text]
        tokens = self._encoder.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self._encoder.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            start = end - overlap_tokens if end < len(tokens) else end
        return chunks if chunks else [text]

    @staticmethod
    def _merge_chunk_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parent_chunks: dict[str, list[dict[str, Any]]] = {}
        non_chunked: dict[str, dict[str, Any]] = {}

        for r in results:
            parent_id = r.get("_parent_id", "")
            if parent_id:
                parent_chunks.setdefault(parent_id, []).append(r)
            else:
                mid = r.get("memory_id", "")
                if mid not in non_chunked or r["score"] > non_chunked[mid]["score"]:
                    non_chunked[mid] = r

        merged_list = []

        for parent_id, chunks in parent_chunks.items():
            chunks.sort(key=lambda c: int(c.get("_chunk_idx", "0")))

            scores = [c["score"] for c in chunks]
            best_score = max(scores)
            avg_score = sum(scores) / len(scores) if scores else 0
            total_chunks = max(int(chunks[-1].get("_chunk_idx", "0")) + 1, len(chunks))
            hit_ratio = len(chunks) / total_chunks if total_chunks > 0 else 1.0

            score_std = (
                (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5
                if len(scores) > 1
                else 0
            )
            has_star_chunk = (
                len(chunks) >= 2
                and best_score > avg_score + score_std
                and best_score > 0.6
            )

            if has_star_chunk:
                for chunk in chunks:
                    chunk_entry = {
                        k: v for k, v in chunk.items() if k not in ("_parent_id", "_chunk_idx")
                    }
                    chunk_entry["memory_id"] = f"{parent_id}_chunk{chunk.get('_chunk_idx', '0')}"
                    if chunk["score"] == best_score:
                        chunk_entry["score"] = min(chunk["score"] * 2.0, 1.0)
                    else:
                        chunk_entry["score"] = chunk["score"] * 0.9
                    merged_list.append(chunk_entry)
            else:
                merged_score = best_score * (1.0 + 1.0 * hit_ratio)
                combined_content = "\n".join(c.get("content", "") for c in chunks)
                merged = {
                    k: v
                    for k, v in chunks[0].items()
                    if k not in ("_parent_id", "_chunk_idx", "content", "score")
                }
                merged["memory_id"] = parent_id
                merged["content"] = combined_content
                merged["score"] = merged_score
                merged_list.append(merged)

        merged_list.extend(non_chunked.values())

        return sorted(merged_list, key=lambda x: x["score"], reverse=True)
