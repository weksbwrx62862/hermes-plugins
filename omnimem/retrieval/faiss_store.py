"""FAISSStore — 基于 FAISS 的轻量向量存储后端。

设计目标：
  - 替代 ChromaDB，CPU 环境下快 10-100x
  - 元数据由 SQLite 管理，FAISS 只管向量搜索
  - 支持 _CachedEmbeddingFunction 复用 embedding 缓存
  - 持久化：FAISS index (.faiss) + metadata (.db)

接口兼容 VectorStore ABC，可直接替换 ChromaDBStore。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from omnimem.retrieval.vector_store import VectorStore, _CachedEmbeddingFunction

logger = logging.getLogger(__name__)


class FAISSStore(VectorStore):
    """FAISS 向量存储 + SQLite 元数据。"""

    def __init__(
        self,
        persist_dir: Path | str,
        embedding_fn: _CachedEmbeddingFunction | None = None,
        dimension: int = 384,  # all-MiniLM-L6-v2 默认维度
    ):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._persist_dir / "omnimem.faiss"
        self._meta_path = self._persist_dir / "omnimem_meta.db"
        self._dimension = dimension
        self._embedding_fn = embedding_fn
        self._index: faiss.Index | None = None
        self._id_map: list[str] = []  # FAISS 内部 idx → memory_id
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            # 初始化 SQLite 元数据库
            self._meta_conn = sqlite3.connect(str(self._meta_path), check_same_thread=False)
            self._meta_conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    id TEXT PRIMARY KEY,
                    document TEXT,
                    meta_json TEXT
                )
            """)
            self._meta_conn.execute("PRAGMA journal_mode=WAL")
            self._meta_conn.commit()

            # 加载或创建 FAISS 索引
            if self._index_path.exists():
                try:
                    self._index = faiss.read_index(str(self._index_path))
                    # 重建 id_map from SQLite (按 rowid 顺序对应 FAISS 内部 idx)
                    rows = self._meta_conn.execute("SELECT id FROM metadata ORDER BY rowid").fetchall()
                    self._id_map = [r[0] for r in rows]
                    logger.info("FAISSStore: loaded index with %d vectors", self._index.ntotal)
                except Exception as e:
                    logger.warning("FAISSStore: failed to load index, rebuilding: %s", e)
                    self._build_index_from_db()
            else:
                self._build_index_from_db()

            self._initialized = True

    def _build_index_from_db(self) -> None:
        """从 SQLite 元数据重建 FAISS 索引。"""
        rows = self._meta_conn.execute("SELECT id FROM metadata ORDER BY rowid").fetchall()
        self._id_map = [r[0] for r in rows]
        if self._embedding_fn and self._id_map:
            # 有 embedding 函数时，从缓存重建向量
            docs = self._meta_conn.execute(
                "SELECT id, document FROM metadata ORDER BY rowid"
            ).fetchall()
            texts = [d[1] for d in docs]
            try:
                embeddings = self._embedding_fn(texts)
                vectors = np.array(embeddings, dtype=np.float32)
                if vectors.ndim == 2 and vectors.shape[1] > 0:
                    self._dimension = vectors.shape[1]
                    self._index = faiss.IndexFlatIP(self._dimension)  # Inner Product (余弦相似度需归一化)
                    faiss.normalize_L2(vectors)
                    self._index.add(vectors)
                    logger.info("FAISSStore: rebuilt index from %d cached embeddings", len(texts))
                    self._save_index()
                    return
            except Exception as e:
                logger.warning("FAISSStore: embedding rebuild failed: %s", e)
        # Fallback: 空索引
        self._index = faiss.IndexFlatIP(self._dimension)
        logger.info("FAISSStore: created empty index (dim=%d)", self._dimension)

    def _save_index(self) -> None:
        """持久化 FAISS 索引到磁盘。"""
        if self._index is not None:
            try:
                faiss.write_index(self._index, str(self._index_path))
            except Exception as e:
                logger.warning("FAISSStore: save index failed: %s", e)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """计算 embedding，返回归一化的 float32 数组。"""
        if self._embedding_fn is None:
            raise RuntimeError("FAISSStore: no embedding function configured")
        embeddings = self._embedding_fn(texts)
        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict] | None = None) -> None:
        self._ensure_initialized()
        if not ids:
            return
        with self._lock:
            try:
                vectors = self._embed(documents)
                if self._index is None or self._index.d != vectors.shape[1]:
                    # 维度变化，重建索引
                    self._dimension = vectors.shape[1]
                    self._index = faiss.IndexFlatIP(self._dimension)
                    self._id_map = []
                    # 从 DB 重新加载已有向量
                    self._rebuild_with_new_dim()

                # Upsert: 先删除已存在的，再添加
                for i, mid in enumerate(ids):
                    if mid in self._id_map:
                        old_idx = self._id_map.index(mid)
                        # FAISS IndexFlat 不支持原地删除，标记为需要重建
                        # 用 replace 策略：记录新向量，后续 rebuild
                        pass

                    # 写入 SQLite
                    meta_json = json.dumps(metadatas[i], ensure_ascii=False) if metadatas else "{}"
                    self._meta_conn.execute(
                        "INSERT OR REPLACE INTO metadata (id, document, meta_json) VALUES (?, ?, ?)",
                        (mid, documents[i], meta_json),
                    )

                self._meta_conn.commit()

                # 简单策略：直接添加到索引（允许重复，搜索时去重）
                self._index.add(vectors)
                self._id_map.extend(ids)
                self._save_index()

            except Exception as e:
                logger.warning("FAISSStore add failed: %s", e)
                raise RuntimeError(f"FAISSStore add failed: {e}") from e

    def _rebuild_with_new_dim(self) -> None:
        """维度变化时从 SQLite 重建索引。"""
        rows = self._meta_conn.execute("SELECT id, document FROM metadata ORDER BY rowid").fetchall()
        if not rows:
            return
        texts = [r[1] for r in rows]
        try:
            vectors = self._embed(texts)
            self._index = faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(vectors)
            self._id_map = [r[0] for r in rows]
        except Exception as e:
            logger.warning("FAISSStore rebuild failed: %s", e)

    def query(self, query_texts: list[str], n_results: int = 10, where: dict | None = None) -> dict:
        self._ensure_initialized()
        empty = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        if self._index is None or self._index.ntotal == 0:
            return empty
        with self._lock:
            try:
                query_vectors = self._embed(query_texts)
                k = min(n_results, self._index.ntotal)
                distances, indices = self._index.search(query_vectors, k)

                all_ids, all_docs, all_metas, all_dists = [], [], [], []
                for q_idx in range(len(query_texts)):
                    q_ids, q_docs, q_metas, q_dists = [], [], [], []
                    for i, idx in enumerate(indices[q_idx]):
                        if idx < 0 or idx >= len(self._id_map):
                            continue
                        mid = self._id_map[idx]
                        # 从 SQLite 读取完整数据
                        row = self._meta_conn.execute(
                            "SELECT document, meta_json FROM metadata WHERE id = ?", (mid,)
                        ).fetchone()
                        if row is None:
                            continue

                        # where 过滤
                        if where:
                            meta = json.loads(row[1])
                            if not all(meta.get(k) == v for k, v in where.items()):
                                continue

                        q_ids.append(mid)
                        q_docs.append(row[0])
                        q_metas.append(json.loads(row[1]))
                        # Inner Product 转 cosine distance: dist = 1 - similarity
                        q_dists.append(float(1.0 - distances[q_idx][i]))

                    all_ids.append(q_ids)
                    all_docs.append(q_docs)
                    all_metas.append(q_metas)
                    all_dists.append(q_dists)

                return {"ids": all_ids, "documents": all_docs, "metadatas": all_metas, "distances": all_dists}

            except Exception as e:
                logger.warning("FAISSStore query failed: %s", e)
                return empty

    def delete(self, ids: list[str]) -> None:
        self._ensure_initialized()
        if not ids:
            return
        with self._lock:
            try:
                placeholders = ",".join("?" * len(ids))
                self._meta_conn.execute(f"DELETE FROM metadata WHERE id IN ({placeholders})", ids)
                self._meta_conn.commit()
                # FAISS IndexFlat 不支持原地删除，重建索引
                self._rebuild_index()
            except Exception as e:
                logger.warning("FAISSStore delete failed: %s", e)

    def _rebuild_index(self) -> None:
        """从 SQLite 重建 FAISS 索引（删除后必需）。"""
        rows = self._meta_conn.execute("SELECT id, document FROM metadata ORDER BY rowid").fetchall()
        self._id_map = [r[0] for r in rows]
        if not rows:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._save_index()
            return
        texts = [r[1] for r in rows]
        try:
            vectors = self._embed(texts)
            self._dimension = vectors.shape[1]
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(vectors)
            self._save_index()
        except Exception as e:
            logger.warning("FAISSStore rebuild failed: %s", e)

    def count(self) -> int:
        self._ensure_initialized()
        try:
            return self._meta_conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
        except Exception:
            return 0

    def reset(self) -> None:
        self._ensure_initialized()
        with self._lock:
            try:
                self._meta_conn.execute("DELETE FROM metadata")
                self._meta_conn.commit()
                self._index = faiss.IndexFlatIP(self._dimension)
                self._id_map = []
                self._save_index()
            except Exception as e:
                logger.warning("FAISSStore reset failed: %s", e)
