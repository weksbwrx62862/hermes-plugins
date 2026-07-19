"""路由决策缓存层。

提供基于查询文本 hash 的线程安全 LRU 缓存，支持 TTL、配置变更失效以及
基于 embedding 近似 hash 的语义相似命中（可选）。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class RouteCache:
    """model-router 路由结果缓存。

    缓存内容约定::

        {
            "complexity": int,
            "task_type": str,
            "strategy": str,
            "model_scores_sorted": list[tuple[str, float]],
            "timestamp": float,  # 由缓存自动写入
        }

    参数:
        capacity: LRU 最大条目数，至少为 1。
        ttl: 条目存活时间（秒），默认 60。
        semantic: 是否启用基于 embedding 近似 hash 的语义相似缓存。
    """

    def __init__(self, capacity: int = 128, ttl: float = 60.0, semantic: bool = False):
        self.capacity = max(1, capacity)
        self.ttl = max(0.0, ttl)
        self.semantic = semantic

        self._lock = threading.RLock()
        # 精确缓存：sha256(query) -> 条目
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        # 语义索引：embedding 近似 hash -> set(精确 key)
        self._semantic_index: dict[str, set[str]] = {}

        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash_query(query: str) -> str:
        """计算查询文本的 sha256 hash。"""
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_embedding(embedding: list[float], bins: int = 16, decimals: int = 1) -> str:
        """对 embedding 做量化后生成近似 hash 桶。

        取前 ``bins`` 维并四舍五入到 ``decimals`` 位小数，相近的向量会落入同一桶，
        从而实现语义相似查询的缓存命中。
        """
        quantized = tuple(round(float(v), decimals) for v in embedding[:bins])
        return hashlib.sha256(str(quantized).encode("utf-8")).hexdigest()

    @staticmethod
    def _config_hash(config: dict[str, Any]) -> str:
        """计算配置字典的规范 hash，用于判断配置是否发生变化。"""
        canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _now(self) -> float:
        """单调时钟，避免系统时间回拨影响 TTL 判断。"""
        return time.monotonic()

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """判断条目是否超过 TTL。"""
        return self._now() - entry["timestamp"] > self.ttl

    def _remove(self, key: str) -> None:
        """移除指定 key，并同步清理语义索引。"""
        entry = self._cache.pop(key, None)
        if entry and self.semantic:
            bucket = entry.get("semantic_bucket")
            if bucket:
                bucket_set = self._semantic_index.get(bucket)
                if bucket_set:
                    bucket_set.discard(key)
                    if not bucket_set:
                        del self._semantic_index[bucket]

    def _cleanup_expired(self) -> None:
        """清理所有已过期条目。"""
        expired = [k for k, e in self._cache.items() if self._is_expired(e)]
        for k in expired:
            self._remove(k)

    def _evict_if_needed(self) -> None:
        """超出容量时按 LRU 顺序淘汰最久未访问条目。"""
        while len(self._cache) > self.capacity:
            oldest_key, _ = self._cache.popitem(last=False)
            if self.semantic:
                # popitem 已从 _cache 移除，这里仅清理语义索引
                for bucket, keys in list(self._semantic_index.items()):
                    keys.discard(oldest_key)
                    if not keys:
                        del self._semantic_index[bucket]

    def set(
        self,
        key: str,
        value: dict[str, Any],
        embedding: Optional[list[float]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """写入缓存。

        参数:
            key: 查询文本（内部会计算 sha256 hash）。
            value: 路由结果字典，至少包含 complexity、task_type、strategy、
                   model_scores_sorted；缺少 timestamp 时会自动补入。
            embedding: 可选的查询 embedding 向量，用于语义相似命中。
            config: 生成该路由结果时的相关配置快照，用于配置变更失效。
        """
        if value is None:
            raise ValueError("缓存值不能为 None")

        with self._lock:
            self._cleanup_expired()

            exact_key = self._hash_query(key)
            # 若已存在则先移除旧语义索引，避免同一 key 出现在多个桶
            self._remove(exact_key)

            stored_value = dict(value)
            stored_value.setdefault("timestamp", self._now())

            entry: dict[str, Any] = {
                "value": stored_value,
                "timestamp": self._now(),
                "config_hash": self._config_hash(config) if config else "",
            }

            if self.semantic and embedding:
                bucket = self._hash_embedding(embedding)
                entry["semantic_bucket"] = bucket
                self._semantic_index.setdefault(bucket, set()).add(exact_key)

            self._cache[exact_key] = entry
            self._cache.move_to_end(exact_key)
            self._evict_if_needed()

    def get(self, key: str, embedding: Optional[list[float]] = None) -> Optional[dict[str, Any]]:
        """读取缓存。

        参数:
            key: 查询文本。
            embedding: 可选的查询 embedding 向量，精确未命中时尝试语义相似命中。

        返回:
            缓存值（浅拷贝后的字典）或 None。
        """
        with self._lock:
            self._cleanup_expired()

            exact_key = self._hash_query(key)
            entry = self._cache.get(exact_key)
            if entry and not self._is_expired(entry):
                self._cache.move_to_end(exact_key)
                self._hits += 1
                return dict(entry["value"])

            if self.semantic and embedding:
                bucket = self._hash_embedding(embedding)
                candidates = list(self._semantic_index.get(bucket, set()))
                for cand_key in candidates:
                    cand_entry = self._cache.get(cand_key)
                    if cand_entry and not self._is_expired(cand_entry):
                        self._hits += 1
                        return dict(cand_entry["value"])

            self._misses += 1
            return None

    def invalidate_all(self) -> int:
        """使所有缓存失效。

        返回:
            被清空的条目数。
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._semantic_index.clear()
            return count

    def invalidate_by_config(self, config: dict[str, Any]) -> int:
        """使与当前配置 hash 不一致的缓存失效。

        参数:
            config: 当前相关配置快照。

        返回:
            被失效的条目数。
        """
        current_hash = self._config_hash(config)
        removed = 0
        with self._lock:
            mismatch_keys = [
                k for k, e in self._cache.items() if e.get("config_hash") != current_hash
            ]
            for k in mismatch_keys:
                self._remove(k)
                removed += 1
        return removed

    @property
    def stats(self) -> dict[str, int]:
        """返回缓存统计信息。"""
        with self._lock:
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "semantic_buckets": len(self._semantic_index),
            }
