"""skill-router 全局状态管理

集中管理模型、后端、查询缓存、嵌入缓存、BM25 缓存、配置缓存等全局状态，
便于单测替换与避免重复加载。
"""

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Set, Tuple

# 使用 skill_router_init 前缀，使日志能被入口模块的 caplog 捕获
logger = logging.getLogger("skill_router_init.state")


class CacheManager:
    """统一 TTL 缓存管理器，线程安全

    封装带过期时间的缓存逻辑：
      - get(): 获取缓存值，过期或未设置时返回 None
      - set(value): 写入缓存并刷新时间戳
      - invalidate(): 主动使缓存失效
    """

    def __init__(self, ttl: float):
        self._ttl = ttl
        self._value: Optional[Any] = None
        self._timestamp: float = 0.0
        self._lock = threading.Lock()

    def get(self) -> Optional[Any]:
        """获取缓存值，过期或未设置时返回 None"""
        with self._lock:
            if self._value is None:
                return None
            if time.time() - self._timestamp >= self._ttl:
                return None
            return self._value

    def set(self, value: Any) -> None:
        """设置缓存值并更新时间戳"""
        with self._lock:
            self._value = value
            self._timestamp = time.time()

    def invalidate(self) -> None:
        """使缓存失效"""
        with self._lock:
            self._value = None
            self._timestamp = 0.0


class QueryCache:
    """LRU + TTL 查询缓存，线程安全

    基于 OrderedDict 实现 LRU 淘汰策略：
      - get(): 命中时移动到末尾（最近使用），过期返回 None
      - set(): 写入缓存，超容量时淘汰最旧条目（头部）
      - clear(): 清空所有缓存
    """

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self._max_size = max_size
        self._ttl = ttl
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值，过期返回 None，命中时更新 LRU 顺序"""
        with self._lock:
            if key not in self._cache:
                return None
            value, timestamp = self._cache[key]
            if time.time() - timestamp >= self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """写入缓存，超容量时淘汰最旧条目"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """返回当前缓存条目数"""
        with self._lock:
            return len(self._cache)


class PluginState:
    """插件全局状态容器

    集中持有所有缓存与锁，所有函数通过 module.plugin_state 访问，
    测试时可直接替换内部缓存。
    """

    def __init__(self):
        # 配置与索引缓存（TTL 300 秒）
        self._config_cache = CacheManager(ttl=300)
        self._skill_index_cache = CacheManager(ttl=300)
        self._embedding_cache = CacheManager(ttl=300)
        self._bm25_cache = CacheManager(ttl=300)

        # 模型与后端缓存（无 TTL，懒加载一次常驻）
        self._model_cache: Optional[Any] = None
        self._model_lock = threading.Lock()
        self._skillrouter_backend: Optional[Any] = None
        self._backend_lock = threading.Lock()

        # 查询缓存（在 register 阶段初始化）
        self._query_cache: Optional[QueryCache] = None

        # SkillRouter 后端增量嵌入缓存
        self._sr_embeddings_cache: Dict[str, Tuple[Any, float]] = {}
        self._sr_embeddings_lock = threading.Lock()

    # ── 模型缓存 ──
    def get_model_cache(self) -> Optional[Any]:
        return self._model_cache

    def set_model_cache(self, value: Any) -> None:
        self._model_cache = value

    def get_model_lock(self) -> threading.Lock:
        return self._model_lock

    # ── SkillRouter 后端缓存 ──
    def get_skillrouter_backend(self) -> Optional[Any]:
        return self._skillrouter_backend

    def set_skillrouter_backend(self, value: Any) -> None:
        self._skillrouter_backend = value

    def get_backend_lock(self) -> threading.Lock:
        return self._backend_lock

    # ── 查询缓存 ──
    def create_query_cache(self, max_size: int = 1000, ttl: int = 300) -> QueryCache:
        self._query_cache = QueryCache(max_size=max_size, ttl=ttl)
        return self._query_cache

    def get_query_cache(self) -> Optional[QueryCache]:
        return self._query_cache

    def get_query_cache_item(self, key: str) -> Optional[Any]:
        cache = self._query_cache
        if cache is None:
            return None
        return cache.get(key)

    def set_query_cache_item(self, key: str, value: Any) -> None:
        cache = self._query_cache
        if cache is None:
            return
        cache.set(key, value)

    def clear_query_cache(self) -> None:
        cache = self._query_cache
        if cache is not None:
            cache.clear()

    # ── SkillRouter 增量嵌入缓存 ──
    def get_sr_embedding(self, name: str) -> Optional[Tuple[Any, float]]:
        return self._sr_embeddings_cache.get(name)

    def set_sr_embedding(self, name: str, embedding: Any, mtime: float) -> None:
        self._sr_embeddings_cache[name] = (embedding, mtime)

    def remove_sr_embedding(self, name: str) -> None:
        self._sr_embeddings_cache.pop(name, None)

    def get_sr_embedding_names(self) -> Set[str]:
        return set(self._sr_embeddings_cache.keys())

    def get_sr_embeddings_lock(self) -> threading.Lock:
        return self._sr_embeddings_lock

    # ── 嵌入缓存 ──
    def get_embedding_cache(self) -> Optional[Any]:
        return self._embedding_cache.get()

    def set_embedding_cache(self, value: Any) -> None:
        self._embedding_cache.set(value)

    def invalidate_embedding_cache(self) -> None:
        self._embedding_cache.invalidate()

    # ── BM25 缓存 ──
    def get_bm25_cache(self) -> Optional[Any]:
        return self._bm25_cache.get()

    def set_bm25_cache(self, value: Any) -> None:
        self._bm25_cache.set(value)

    def invalidate_bm25_cache(self) -> None:
        self._bm25_cache.invalidate()

    # ── 配置缓存 ──
    def get_config_cache(self) -> Optional[Any]:
        return self._config_cache.get()

    def set_config_cache(self, value: Any) -> None:
        self._config_cache.set(value)

    def invalidate_config_cache(self) -> None:
        self._config_cache.invalidate()

    # ── 技能索引缓存 ──
    def get_skill_index_cache(self) -> Optional[Any]:
        return self._skill_index_cache.get()

    def set_skill_index_cache(self, value: Any) -> None:
        self._skill_index_cache.set(value)

    def invalidate_skill_index_cache(self) -> None:
        self._skill_index_cache.invalidate()


# 模块级全局实例，入口层通过该实例访问状态
plugin_state = PluginState()
