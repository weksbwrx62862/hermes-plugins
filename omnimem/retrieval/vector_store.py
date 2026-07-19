"""VectorStore 抽象接口与多后端适配器。"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any

# ★ 嵌入缓存命中率指标（失败时降级为 no-op，不影响主流程）
try:
    from omnimem.utils.metrics import record_cache_hit, record_cache_miss
except Exception:  # pragma: no cover - 监控模块不可用时降级
    def record_cache_hit() -> None:
        pass

    def record_cache_miss() -> None:
        pass

# ★ 抑制 ChromaDB 0.6.x telemetry PostHog capture() 签名不兼容的噪音日志
#    这是 ChromaDB 内部问题，不影响功能，无需暴露给用户
class _ChromaDBTelemetryFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # 过滤 PostHog telemetry 失败消息和 chromadb telemetry DEBUG 噪音
        if "Failed to send telemetry event" in msg:
            return False
        if record.name.startswith("chromadb.telemetry"):
            return False
        return True

_telemetry_filter = _ChromaDBTelemetryFilter()
for _logger_name in ("chromadb.telemetry.product.posthog", "chromadb.telemetry"):
    logging.getLogger(_logger_name).addFilter(_telemetry_filter)
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _emit(msg: str) -> None:
    """向用户输出状态消息。

    优先写入 stderr（CLI 中可见），
    非 TTY 环境（gateway/cron/background）降级为 logger.info。
    """
    try:
        import sys
        if sys.stderr.isatty():
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        else:
            logger.info(msg)
    except Exception:
        logger.info(msg)


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: list[str], documents: list[str], metadatas: list[dict] | None = None) -> None:
        ...

    @abstractmethod
    def query(self, query_texts: list[str], n_results: int = 10, where: dict | None = None) -> dict:
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class _CachedEmbeddingFunction:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_path: Path | None = None,
        model_path: str = "",
        cache_ttl: float = 3600,
    ):
        self._model_name = model_name
        self._model_path = model_path
        self._model = None
        # ★ TTL + LRU 缓存：value 为 (embedding, expire_at)
        self._cache: OrderedDict[str, tuple[list[float], float]] = OrderedDict()
        self._max_cache = 1000
        self._cache_ttl = cache_ttl
        self._lock = threading.Lock()
        self._cache_path = cache_path
        # ★ 异步持久化：脏标记 + 持久化进行中标记
        self._dirty = False
        self._persist_in_progress = False
        self._load_cache()

    @staticmethod
    def name() -> str:
        """ChromaDB EmbeddingFunction 协议要求的 name 方法。"""
        return "omnimem_cached_sentence_transformer"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> _CachedEmbeddingFunction:
        """ChromaDB EmbeddingFunction 协议要求的反序列化方法。"""
        return _CachedEmbeddingFunction(
            model_name=config.get("model_name", "all-MiniLM-L6-v2"),
            cache_path=Path(config["cache_path"]) if config.get("cache_path") else None,
        )

    def get_config(self) -> dict[str, Any]:
        """ChromaDB EmbeddingFunction 协议要求的序列化方法。"""
        return {
            "model_name": self._model_name,
            "cache_path": str(self._cache_path) if self._cache_path else "",
        }

    @staticmethod
    def is_legacy() -> bool:
        return False

    @property
    def cache_size(self) -> int:
        """当前内存缓存条目数。"""
        with self._lock:
            return len(self._cache)

    def _load_cache(self) -> None:
        if not self._cache_path or not self._cache_path.exists():
            return
        try:
            import json

            with open(self._cache_path, encoding="utf-8") as f:
                data = json.load(f)
            now = time.time()
            loaded: OrderedDict[str, tuple[list[float], float]] = OrderedDict()
            for k, vec in data.items():
                if isinstance(vec, (list, tuple)) and len(vec) == 2:
                    # 持久化或新格式: (vector, expire_at)
                    embedding = [float(v) for v in vec[0]]
                else:
                    # 旧格式: 仅 vector
                    embedding = [float(v) for v in vec]
                loaded[k] = (embedding, now + self._cache_ttl)
            self._cache = loaded
            count = len(self._cache)
            logger.warning("Loaded %d entries from embedding cache", count)
            if count > 0:
                _emit(f"[OmniMem] 嵌入缓存: 从磁盘加载 {count} 条")
        except Exception as e:
            logger.warning("Embedding cache load failed: %s", e)
            self._cache = OrderedDict()

    def persist(self) -> None:
        """异步持久化嵌入缓存到磁盘（后台线程写入，不阻塞主流程）。

        使用 _dirty 标记避免无变更时的无效写入，
        使用 _persist_in_progress 标记避免并发写入冲突。
        """
        if not self._cache_path:
            return
        # 无变更则跳过
        if not self._dirty:
            return
        # 已有持久化在进行中则跳过（避免并发写入冲突）
        if self._persist_in_progress:
            return
        # 标记持久化进行中
        self._persist_in_progress = True
        # 启动后台 daemon 线程执行磁盘写入
        thread = threading.Thread(target=self._persist_to_disk, daemon=True)
        thread.start()

    def _persist_to_disk(self) -> None:
        """后台线程执行磁盘写入（内部方法）。"""
        try:
            import json
            import os

            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                # 持久化时仅保存向量（加载时重置 TTL），避免过期时间绝对值失效
                data = {k: v[0] for k, v in self._cache.items()}
                self._dirty = False
            # 原子写入：先写临时文件再 rename，避免并发读看到不完整的写入
            tmp_path = self._cache_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(str(tmp_path), str(self._cache_path))
            logger.warning("Saved %d entries to embedding cache (async)", len(data))
        except Exception as e:
            logger.warning("Embedding cache persist failed: %s", e)
        finally:
            self._persist_in_progress = False

    def _get_model(self) -> Any:
        if self._model is None:
            import os
            import time

            # ★ 仅当 skill-router 模型路径与本模型一致时才复用（避免维度不匹配）
            #   skill-router 使用 768 维微调 BERT，OmniMem 使用 384 维 MiniLM，
            #   维度不同会导致 FAISS 索引完全失效。
            try:
                import sys
                sr = sys.modules.get("hermes_plugins.skill_router")
                if sr is None:
                    import importlib
                    sr = importlib.import_module("hermes_plugins.skill_router")
                cached = getattr(sr, "_MODEL_CACHE", None)
                if cached is not None:
                    # 检查模型路径是否一致（resolve 绝对路径比较）
                    sr_model_path = os.path.expanduser(
                        sr._load_config().get("model_path", "") if hasattr(sr, "_load_config") else ""
                    )
                    my_model_path = os.path.abspath(self._model_path or self._model_name)
                    if sr_model_path and os.path.abspath(sr_model_path) == my_model_path:
                        self._model = cached
                        logger.info("[OmniMem] 复用 skill-router 的 embedding 模型（路径一致）")
                        return self._model
                    else:
                        logger.debug("[OmniMem] 跳过 skill-router 模型复用（路径不同: sr=%s, omni=%s）",
                                   sr_model_path, my_model_path)
            except Exception:
                logger.debug("OmniMem skill-router model reuse check failed", exc_info=True)

            try:
                import torch.distributed as dist

                if not hasattr(dist, "is_initialized"):
                    dist.is_initialized = lambda: False
            except Exception:
                logger.debug("OmniMem torch.distributed patch failed", exc_info=True)
            from sentence_transformers import SentenceTransformer

            # ★ GFW 修复：使用中国镜像 + CPU 模式
            if 'HF_ENDPOINT' not in os.environ:
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

            model_path = self._model_path or self._model_name
            _emit(f"[OmniMem] 正在加载嵌入模型: {model_path} ...")
            t0 = time.time()
            try:
                self._model = SentenceTransformer(model_path, device='cpu')
                elapsed = time.time() - t0
                _emit(f"[OmniMem] 嵌入模型加载完成 ({model_path}, {elapsed:.1f}s, CPU)")
            except Exception as e:
                _emit(f"[OmniMem] ⚠ 嵌入模型加载失败: {e}")
                raise
        return self._model

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_query(input)

    def _evict_expired(self, now: float) -> None:
        """清理已过期的缓存条目。"""
        expired = [k for k, (_, expire_at) in self._cache.items() if expire_at <= now]
        for k in expired:
            del self._cache[k]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        results = []
        to_encode = []
        to_encode_idx = []
        now = time.time()

        with self._lock:
            for i, text in enumerate(input):
                text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                cached = self._cache.get(text_hash)
                if cached is not None:
                    vec, expire_at = cached
                    if expire_at > now:
                        # ★ 缓存命中且未过期：更新 LRU 顺序并记录指标
                        self._cache.move_to_end(text_hash)
                        results.append((i, vec))
                        try:
                            record_cache_hit()
                        except Exception:
                            pass
                        continue
                    # 已过期则删除，重新编码
                    del self._cache[text_hash]
                # ★ 缓存未命中：记录指标
                to_encode.append(text)
                to_encode_idx.append((i, text_hash))
                try:
                    record_cache_miss()
                except Exception:
                    pass

        if to_encode:
            model = self._get_model()
            embeddings = model.encode(to_encode, convert_to_numpy=True)
            now = time.time()
            with self._lock:
                for (orig_idx, text_hash), emb in zip(to_encode_idx, embeddings, strict=False):
                    vec = emb.tolist()
                    self._cache[text_hash] = (vec, now + self._cache_ttl)
                    self._cache.move_to_end(text_hash)
                    results.append((orig_idx, vec))

                # ★ TTL 优先淘汰，再按 LRU 容量淘汰
                self._evict_expired(now)
                while len(self._cache) > self._max_cache:
                    self._cache.popitem(last=False)
                # ★ 缓存已变更，标记为脏（下次 persist 时写入磁盘）
                self._dirty = True

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


class ChromaDBStore(VectorStore):
    def __init__(
        self,
        collection_name: str = "omnimem",
        persist_dir: str | Path = "/tmp/omnimem/retrieval/chroma",
        embedding_fn: Any = None,
    ):
        self._collection_name = collection_name
        self._persist_dir = Path(persist_dir)
        self._embedding_fn = embedding_fn
        self._client: Any = None
        self._collection: Any = None
        self._initialized = False
        self._persist_pending = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self._persist_dir))
            # ★ R32修复QUAL-2：ChromaDB 0.6.x 升级后 config_json_str 缺少 _type 字段
            # 旧版 ChromaDB 创建的 collection 配置为 {}，新版期望 {"_type": "CollectionConfigurationInternal"}
            # 在获取 collection 前自动修复
            self._fix_chromadb_config_type()
            try:
                if self._embedding_fn is not None:
                    self._collection = self._client.get_or_create_collection(
                        name=self._collection_name,
                        metadata={"hnsw:space": "cosine"},
                        embedding_function=self._embedding_fn,
                    )
                else:
                    self._collection = self._client.get_or_create_collection(
                        name=self._collection_name,
                        metadata={"hnsw:space": "cosine"},
                    )
            except Exception as e:
                # embedding function 不兼容时，删除旧 collection 重建
                logger.warning("ChromaDB collection incompatible: %s, recreating", e)
                try:
                    self._client.delete_collection(name=self._collection_name)
                except Exception:
                    logger.debug("OmniMem ChromaDB collection delete failed during recreate", exc_info=True)
                if self._embedding_fn is not None:
                    self._collection = self._client.get_or_create_collection(
                        name=self._collection_name,
                        metadata={"hnsw:space": "cosine"},
                        embedding_function=self._embedding_fn,
                    )
                else:
                    self._collection = self._client.get_or_create_collection(
                        name=self._collection_name,
                        metadata={"hnsw:space": "cosine"},
                    )
            logger.warning(
                "ChromaDB collection initialized: %d documents",
                self._collection.count() if self._collection else 0,
            )
        except ImportError:
            logger.warning("chromadb not installed — vector search unavailable")
        except Exception as e:
            logger.warning("ChromaDB init failed: %s", e)
        self._initialized = True

    def _fix_chromadb_config_type(self) -> None:
        """★ R32修复QUAL-2：修复 ChromaDB 0.6.x 升级后 config_json_str 缺少 _type 字段的问题。

        ChromaDB 0.6.x 的 CollectionConfigurationInternal.from_json() 要求
        config_json_str 中包含 "_type" 字段，但旧版创建的 collection
        config_json_str 为 {}，导致 KeyError: '_type'。
        """
        try:
            import sqlite3

            chroma_db_path = self._persist_dir / "chroma.sqlite3"
            if not chroma_db_path.exists():
                return
            conn = sqlite3.connect(str(chroma_db_path))
            try:
                rows = conn.execute(
                    "SELECT name, config_json_str FROM collections"
                ).fetchall()
                for name, config_str in rows:
                    if not config_str:
                        continue
                    try:
                        import json

                        config = json.loads(config_str)
                        if "_type" not in config:
                            config["_type"] = "CollectionConfigurationInternal"
                            conn.execute(
                                "UPDATE collections SET config_json_str = ? WHERE name = ?",
                                (json.dumps(config), name),
                            )
                            logger.info(
                                "ChromaDB config migrated: added _type to collection %s",
                                name,
                            )
                    except (json.JSONDecodeError, Exception):
                        logger.debug("OmniMem ChromaDB config migration row skipped", exc_info=True)
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("ChromaDB config migration skipped: %s", e)

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict] | None = None) -> None:
        self._ensure_initialized()
        if self._collection is None:
            return
        try:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            self._persist_client()
        except Exception as e:
            logger.warning("ChromaDBStore add failed: %s", e)
            raise RuntimeError(f"ChromaDBStore add failed: {e}") from e

    def query(self, query_texts: list[str], n_results: int = 10, where: dict | None = None) -> dict:
        self._ensure_initialized()
        if self._collection is None:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        try:
            count = self._collection.count()
            if count == 0:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            kwargs: dict[str, Any] = {
                "query_texts": query_texts,
                "n_results": min(n_results, count),
                "include": ["documents", "metadatas", "distances"],
            }
            if where is not None:
                kwargs["where"] = where
            return self._collection.query(**kwargs)
        except Exception as e:
            logger.warning("ChromaDBStore query failed: %s", e)
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def delete(self, ids: list[str]) -> None:
        self._ensure_initialized()
        if self._collection is None:
            return
        if not ids:
            return
        try:
            self._collection.delete(ids=ids)
            self._persist_client()
        except Exception as e:
            logger.warning("ChromaDBStore delete failed: %s", e)

    def count(self) -> int:
        self._ensure_initialized()
        if self._collection is None:
            return 0
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    def reset(self) -> None:
        self._ensure_initialized()
        if self._client is None:
            return
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embedding_fn,
            )
        except Exception as e:
            logger.warning("ChromaDBStore reset failed: %s", e)

    def _persist_client(self) -> None:
        try:
            if self._client and hasattr(self._client, "persist"):
                self._client.persist()
                self._persist_pending = False
        except Exception as e:
            logger.warning("ChromaDB persist failed: %s", e)
            self._persist_pending = True

    def close(self) -> None:
        """释放 ChromaDB 客户端和集合。"""
        try:
            self._persist_client()
            self._collection = None
            self._client = None
            self._initialized = False
        except Exception as e:
            logger.warning("ChromaDBStore close failed: %s", e)


class QdrantStore(VectorStore):
    def __init__(
        self,
        collection_name: str = "omnimem",
        url: str | None = None,
        api_key: str | None = None,
    ):
        self._collection_name = collection_name
        self._url = url or "localhost:6333"
        self._api_key = api_key
        self._client: Any = None
        self._initialized = False
        self._point_id_counter = 0

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            kwargs: dict[str, Any] = {"url": self._url}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = QdrantClient(**kwargs)
            try:
                self._client.get_collection(self._collection_name)
            except Exception:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
            logger.warning("Qdrant collection initialized: %s", self._collection_name)
        except ImportError:
            logger.warning("qdrant-client not installed — Qdrant vector search unavailable")
        except Exception as e:
            logger.warning("Qdrant init failed: %s", e)
        self._initialized = True

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict] | None = None) -> None:
        self._ensure_initialized()
        if self._client is None:
            return
        try:
            from qdrant_client.models import PointStruct

            metas = metadatas or [{}] * len(ids)
            points = []
            for pid, doc, meta in zip(ids, documents, metas, strict=False):
                self._point_id_counter += 1
                points.append(
                    PointStruct(
                        id=self._point_id_counter,
                        vector=[0.0] * 384,
                        payload={"_id": pid, "document": doc, **meta},
                    )
                )
            self._client.upsert(collection_name=self._collection_name, points=points)
        except Exception as e:
            logger.warning("QdrantStore add failed: %s", e)

    def query(self, query_texts: list[str], n_results: int = 10, where: dict | None = None) -> dict:
        self._ensure_initialized()
        if self._client is None:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        try:
            dummy_vector = [0.0] * 384
            filter_obj = None
            if where:
                from qdrant_client.models import FieldCondition, Filter, MatchValue

                conditions = []
                for k, v in where.items():
                    conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
                filter_obj = Filter(must=conditions)

            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=dummy_vector,
                limit=n_results,
                query_filter=filter_obj,
            )
            ids = [[str(r.id) for r in results]]
            documents = [[r.payload.get("document", "") for r in results]]
            metadatas = [[{k: v for k, v in r.payload.items() if k != "document"} for r in results]]
            distances = [[1.0 - r.score for r in results]]
            return {
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
                "distances": distances,
            }
        except Exception as e:
            logger.warning("QdrantStore query failed: %s", e)
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def delete(self, ids: list[str]) -> None:
        self._ensure_initialized()
        if self._client is None:
            return
        try:
            from qdrant_client.models import PointIdsList

            scroll_result = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=None,
                limit=10000,
            )
            points_to_delete = []
            for point in scroll_result[0]:
                if point.payload.get("_id") in ids:
                    points_to_delete.append(point.id)
            if points_to_delete:
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=PointIdsList(points=points_to_delete),
                )
        except Exception as e:
            logger.warning("QdrantStore delete failed: %s", e)

    def count(self) -> int:
        self._ensure_initialized()
        if self._client is None:
            return 0
        try:
            info = self._client.get_collection(self._collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def reset(self) -> None:
        self._ensure_initialized()
        if self._client is None:
            return
        try:
            self._client.delete_collection(self._collection_name)
            from qdrant_client.models import Distance, VectorParams

            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            self._point_id_counter = 0
        except Exception as e:
            logger.warning("QdrantStore reset failed: %s", e)
