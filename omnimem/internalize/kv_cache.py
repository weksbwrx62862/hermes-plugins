"""KVCacheManager — L4 内化记忆：KV Cache 预填充。

参考 MemOS 的 KV Cache 记忆设计 (ActMemory)：
  - 高频访问的记忆模式预填充到 KV Cache
  - 触发条件: access_count > 10 自动缓存
  - 文件系统持久化
  - 与检索引擎集成：缓存命中时跳过检索直接返回

Phase 4 完整实现。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnimem.utils.experimental import experimental_class
from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)


@experimental_class
class KVCacheManager:
    """KV Cache 预填充管理器。

    核心机制 (MemOS ActMemory):
      1. 监控记忆访问频率
      2. 当 access_count > threshold (默认10) 时自动预填充
      3. 预填充内容持久化到文件系统
      4. 推理时先查 KV Cache，命中则跳过检索
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        auto_preload_threshold: int = 10,
        max_cache_size: int = 100,
    ):
        """初始化 KVCacheManager。

        Args:
            data_dir: 数据目录，用于持久化
            auto_preload_threshold: 自动预填充的访问次数阈值
            max_cache_size: 最大缓存条目数
        """
        self._data_dir = data_dir
        self._auto_threshold = auto_preload_threshold
        self._max_cache_size = max_cache_size
        self._cache: dict[str, dict[str, Any]] = {}
        self._access_counts: dict[str, int] = {}
        self._conn: sqlite3.Connection | None = None
        self._preload_count = 0
        self._lock = threading.RLock()

        if data_dir:
            self._init_db(data_dir)

    def _init_db(self, data_dir: Path) -> None:
        """初始化 KV Cache 持久化数据库。"""
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "kv_cache.db"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="kv_cache_entries",
            create_sql="""
                CREATE TABLE IF NOT EXISTS kv_cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    access_count INTEGER DEFAULT 0,
                    preloaded_at TEXT,
                    last_accessed TEXT,
                    source_memory_ids TEXT
                )
            """,
            migrations=[],
        )
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_access_count ON kv_cache_entries(access_count DESC)
        """)
        self._conn.commit()

        # 从持久化存储中恢复缓存
        self._restore_from_db()

    def _restore_from_db(self) -> None:
        """从数据库恢复已缓存的条目。"""
        if not self._conn:
            return
        try:
            rows = self._conn.execute(
                "SELECT cache_key, content, metadata, access_count, source_memory_ids FROM kv_cache_entries ORDER BY access_count DESC LIMIT ?",
                (self._max_cache_size,),
            ).fetchall()
            for row in rows:
                key, content, metadata_str, access_count, source_ids_str = row
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                try:
                    source_ids = json.loads(source_ids_str) if source_ids_str else []
                except (json.JSONDecodeError, TypeError):
                    source_ids = []

                self._cache[key] = {
                    "key": key,
                    "content": content,
                    "metadata": metadata,
                    "source_memory_ids": source_ids,
                }
                self._access_counts[key] = access_count
            logger.info("KV Cache: restored %d entries from disk", len(self._cache))
        except Exception as e:
            logger.warning("KV Cache restore failed: %s", e)

    # ─── 公开接口 ─────────────────────────────────────────────

    def preload(self, patterns: list[dict[str, Any]]) -> int:
        """预加载高频模式到 KV Cache。

        Args:
            patterns: 高频记忆模式列表，每个包含 key, content, metadata

        Returns:
            预加载的模式数量
        """
        count = 0
        for pattern in patterns:
            key = pattern.get("key", "")
            content = pattern.get("content", "")
            if not key or not content:
                continue

            self._cache[key] = pattern
            self._access_counts[key] = self._access_counts.get(key, 0)

            # 持久化
            self._persist_entry(key, pattern)
            count += 1

        self._preload_count += count
        if count > 0:
            logger.info("KV Cache: preloaded %d patterns", count)
        return count

    def get(self, key: str) -> Any | None:
        """从 KV Cache 获取预填充内容。"""
        if key in self._cache:
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
            # 不立即写 SQLite，close() 时批量刷盘
            return self._cache[key]
        return None

    def is_cached(self, key: str) -> bool:
        """检查是否已缓存。"""
        return key in self._cache

    def get_hot_patterns(self, top_k: int = 10) -> list[dict[str, Any]]:
        """获取最热访问模式。"""
        sorted_items = sorted(
            self._access_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        results = []
        for key, count in sorted_items[:top_k]:
            if key in self._cache:
                entry = dict(self._cache[key])
                entry["access_count"] = count
                results.append(entry)
        return results

    def check_and_auto_preload(
        self,
        key: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        source_memory_ids: list[str] | None = None,
    ) -> bool:
        """检查访问频率并自动预填充。

        当 access_count > threshold 时自动触发预填充。
        这是 MemOS ActMemory 的核心机制。

        Args:
            key: 缓存键
            content: 记忆内容
            metadata: 元数据
            source_memory_ids: 来源记忆ID列表

        Returns:
            是否触发了预填充
        """
        # 更新访问计数
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        current_count = self._access_counts[key]

        # 如果已经缓存，只更新计数
        if key in self._cache:
            self._update_access_count(key)
            return False

        # 达到阈值，触发预填充
        if current_count >= self._auto_threshold:
            pattern = {
                "key": key,
                "content": content,
                "metadata": metadata or {},
                "source_memory_ids": source_memory_ids or [],
            }
            self._cache[key] = pattern
            self._persist_entry(key, pattern)
            logger.info(
                "KV Cache: auto-preloaded key '%s' (access_count=%d)",
                key[:50],
                current_count,
            )
            return True

        return False

    def search_cache(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """在缓存中搜索匹配的内容。

        简单的关键词匹配，用于推理加速时的缓存查找。
        """
        results = []
        query_lower = query.lower()
        for key, entry in self._cache.items():
            content = entry.get("content", "").lower()
            if query_lower in content or any(
                kw in content for kw in query_lower.split() if len(kw) >= 2
            ):
                result = dict(entry)
                result["access_count"] = self._access_counts.get(key, 0)
                results.append(result)
                if len(results) >= limit:
                    break
        return results

    def clear(self) -> None:
        """清空缓存。"""
        self._cache.clear()
        self._access_counts.clear()
        if self._conn:
            try:
                self._conn.execute("DELETE FROM kv_cache_entries")
                self._conn.commit()
            except Exception as e:
                logger.warning("KVCacheManager clear failed: %s", e)

    def get_stats(self) -> dict[str, Any]:
        """获取 KV Cache 统计。"""
        total_accesses = sum(self._access_counts.values())
        hot_patterns = self.get_hot_patterns(top_k=3)
        return {
            "cached_entries": len(self._cache),
            "total_accesses": total_accesses,
            "auto_preload_threshold": self._auto_threshold,
            "max_cache_size": self._max_cache_size,
            "total_preloaded": self._preload_count,
            "top_patterns": [
                {"key": p.get("key", "")[:50], "access_count": p.get("access_count", 0)}
                for p in hot_patterns
            ],
        }

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            # 保存当前访问计数
            self._flush_access_counts()
            self._conn.close()
            self._conn = None

    # ─── 持久化 ───────────────────────────────────────────────

    def _persist_entry(self, key: str, pattern: dict[str, Any]) -> None:
        """持久化缓存条目。"""
        if not self._conn:
            return
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO kv_cache_entries
                       (cache_key, content, metadata, access_count, preloaded_at, last_accessed, source_memory_ids)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        key,
                        pattern.get("content", ""),
                        json.dumps(pattern.get("metadata", {}), ensure_ascii=False),
                        self._access_counts.get(key, 0),
                        now,
                        now,
                        json.dumps(pattern.get("source_memory_ids", []), ensure_ascii=False),
                    ),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning("KV Cache persist failed: %s", e)

    def _update_access_count(self, key: str) -> None:
        """更新访问计数（批量写入优化）。"""
        if not self._conn:
            return
        with self._lock:
            try:
                now = datetime.now(timezone.utc).isoformat()
                self._conn.execute(
                    "UPDATE kv_cache_entries SET access_count = ?, last_accessed = ? WHERE cache_key = ?",
                    (self._access_counts.get(key, 0), now, key),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning("KV Cache update access count failed for %s: %s", key, e)

    def _flush_access_counts(self) -> None:
        """批量刷新所有访问计数。"""
        if not self._conn:
            return
        with self._lock:
            try:
                now = datetime.now(timezone.utc).isoformat()
                for key, count in self._access_counts.items():
                    self._conn.execute(
                        "UPDATE kv_cache_entries SET access_count = ?, last_accessed = ? WHERE cache_key = ?",
                        (count, now, key),
                    )
                self._conn.commit()
            except Exception as e:
                logger.warning("KVCacheManager _flush_access_counts failed: %s", e)
