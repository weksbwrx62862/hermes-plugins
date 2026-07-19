"""PluginState 单元测试

验证缓存的 get/set/clear、TTL 行为以及锁的基本存在性。
"""

import importlib.util
import os
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE_PATH = os.path.join(ROOT, "state.py")


def _load_state():
    """动态加载 state 模块"""
    spec = importlib.util.spec_from_file_location("state", _STATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["state"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_state()


@pytest.fixture
def state(module):
    """每个测试使用全新的 PluginState 实例"""
    return module.PluginState()


def test_config_cache_get_set_and_invalidate(state):
    """配置缓存应支持写入、读取和失效"""
    assert state.get_config_cache() is None
    state.set_config_cache({"enabled": True})
    assert state.get_config_cache() == {"enabled": True}
    state.invalidate_config_cache()
    assert state.get_config_cache() is None


def test_skill_index_cache_round_trip(state):
    """技能索引缓存应支持完整读写周期"""
    index = {"skill-a": {"category": "test"}}
    state.set_skill_index_cache(index)
    assert state.get_skill_index_cache() is index


def test_cache_manager_ttl_expires(module):
    """CacheManager 应在 TTL 到期后返回 None"""
    cache = module.CacheManager(ttl=0.05)
    cache.set("value")
    assert cache.get() == "value"
    time.sleep(0.06)
    assert cache.get() is None


def test_query_cache_lru_and_ttl(module):
    """QueryCache 应支持 TTL 与 LRU 淘汰"""
    cache = module.QueryCache(max_size=2, ttl=0.05)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    assert cache.get("k1") == "v1"
    cache.set("k3", "v3")  # 触发淘汰最久未使用的 k2
    assert cache.get("k1") == "v1"
    assert cache.get("k2") is None
    time.sleep(0.06)
    assert cache.get("k1") is None


def test_query_cache_create_and_clear(state):
    """PluginState 应能创建并清空查询缓存"""
    assert state.get_query_cache() is None
    cache = state.create_query_cache(max_size=10, ttl=60)
    assert cache is not None
    state.set_query_cache_item("key", "value")
    assert state.get_query_cache_item("key") == "value"
    state.clear_query_cache()
    assert state.get_query_cache_item("key") is None


def test_locks_exist(state):
    """PluginState 应暴露模型锁、后端锁与 SR 嵌入锁"""
    assert isinstance(state.get_model_lock(), threading.Lock)
    assert isinstance(state.get_backend_lock(), threading.Lock)
    assert isinstance(state.get_sr_embeddings_lock(), threading.Lock)


def test_sr_embeddings_round_trip(state):
    """SkillRouter 增量嵌入缓存应支持读写删"""
    state.set_sr_embedding("skill-a", [0.1, 0.2], 123.0)
    assert state.get_sr_embedding("skill-a") == ([0.1, 0.2], 123.0)
    assert state.get_sr_embedding_names() == {"skill-a"}
    state.remove_sr_embedding("skill-a")
    assert state.get_sr_embedding("skill-a") is None
    assert state.get_sr_embedding_names() == set()
