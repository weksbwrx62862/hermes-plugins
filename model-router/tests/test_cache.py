"""
RouteCache 单元测试。

覆盖：
  - 基本 set/get
  - TTL 过期
  - LRU 淘汰
  - 配置变更失效
  - 并发安全
  - 语义相似缓存（可选）
"""

import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

# 目录名含连字符，无法直接 import，使用 importlib 加载 cache.py
_cache_path = Path(__file__).resolve().parent.parent / "cache.py"
_cache_spec = importlib.util.spec_from_file_location("route_cache", str(_cache_path))
route_cache = importlib.util.module_from_spec(_cache_spec)
sys.modules["route_cache"] = route_cache
_cache_spec.loader.exec_module(route_cache)

RouteCache = route_cache.RouteCache


def _sample_value():
    """构造一个符合缓存内容约定的示例值。"""
    return {
        "complexity": 3,
        "task_type": "code",
        "strategy": "auto",
        "model_scores_sorted": [
            ("model-a", 95.0),
            ("model-b", 80.0),
            ("model-c", 60.0),
        ],
    }


class TestBasicOperations:
    """基本读写操作测试。"""

    def test_set_and_get(self):
        """正常 set 后 get 应返回缓存值并包含 timestamp。"""
        cache = RouteCache()
        value = _sample_value()

        cache.set("hello world", value)
        result = cache.get("hello world")

        assert result is not None
        assert result["complexity"] == 3
        assert result["task_type"] == "code"
        assert result["strategy"] == "auto"
        assert result["model_scores_sorted"][0] == ("model-a", 95.0)
        assert "timestamp" in result

    def test_get_missing_returns_none(self):
        """未命中的 key 返回 None。"""
        cache = RouteCache()
        assert cache.get("not exists") is None

    def test_set_overwrites_existing(self):
        """同一 key 重复 set 应覆盖旧值。"""
        cache = RouteCache()
        cache.set("q", {"complexity": 1, "task_type": "simple_qa", "strategy": "cheapest", "model_scores_sorted": []})
        cache.set("q", _sample_value())

        result = cache.get("q")
        assert result["complexity"] == 3


class TestTTL:
    """TTL 过期测试。"""

    def test_ttl_expiration(self):
        """超过 TTL 后缓存应自动失效。"""
        cache = RouteCache(ttl=0.05)
        cache.set("q", _sample_value())

        assert cache.get("q") is not None
        time.sleep(0.08)
        assert cache.get("q") is None

    def test_zero_ttl_immediate_expiration(self):
        """TTL 为 0 时写入后立即过期。"""
        cache = RouteCache(ttl=0.0)
        cache.set("q", _sample_value())
        assert cache.get("q") is None


class TestLRU:
    """LRU 淘汰测试。"""

    def test_lru_eviction(self):
        """容量满时最久未访问的条目应被淘汰。"""
        cache = RouteCache(capacity=2)
        cache.set("a", _sample_value())
        cache.set("b", _sample_value())
        cache.get("a")  # 访问 a，提升其热度
        cache.set("c", _sample_value())  # 应淘汰 b

        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_capacity_at_least_one(self):
        """容量会被强制限制为至少 1。"""
        cache = RouteCache(capacity=0)
        cache.set("a", _sample_value())
        cache.set("b", _sample_value())

        # 只有最后写入的一个保留
        assert cache.get("a") is None
        assert cache.get("b") is not None


class TestConfigInvalidation:
    """配置变更失效测试。"""

    def test_invalidate_all(self):
        """invalidate_all 清空所有缓存。"""
        cache = RouteCache()
        cache.set("a", _sample_value())
        cache.set("b", _sample_value())

        removed = cache.invalidate_all()

        assert removed == 2
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_invalidate_by_config_mismatch(self):
        """配置 hash 不一致的条目应被失效。"""
        cache = RouteCache()
        old_config = {"scoring": {"strategy_multiplier": 6}}
        new_config = {"scoring": {"strategy_multiplier": 10}}

        cache.set("a", _sample_value(), config=old_config)
        cache.set("b", _sample_value(), config=old_config)

        removed = cache.invalidate_by_config(new_config)

        assert removed == 2
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_invalidate_by_config_keeps_match(self):
        """配置 hash 一致的条目应保留。"""
        cache = RouteCache()
        config = {"scoring": {"strategy_multiplier": 6}}

        cache.set("a", _sample_value(), config=config)
        removed = cache.invalidate_by_config(config)

        assert removed == 0
        assert cache.get("a") is not None


class TestConcurrency:
    """并发安全测试。"""

    def test_concurrent_set_get(self):
        """多线程并发 set/get 不应抛异常且结果一致。"""
        cache = RouteCache(capacity=128)
        errors = []

        def worker(idx):
            try:
                key = f"query-{idx % 16}"
                value = {
                    "complexity": idx % 5 + 1,
                    "task_type": "code",
                    "strategy": "auto",
                    "model_scores_sorted": [(f"model-{idx}", float(idx))],
                }
                cache.set(key, value)
                result = cache.get(key)
                if result is None:
                    errors.append(f"worker {idx} get returned None")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"worker {idx}: {exc}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cache._hits > 0


class TestSemanticSimilarity:
    """语义相似缓存测试（可选能力）。"""

    def test_semantic_similar_embedding_hit(self):
        """不同查询文本但 embedding 相似时应命中同一缓存。"""
        cache = RouteCache(semantic=True)
        value = _sample_value()
        embedding = [0.1, 0.2, 0.3, 0.4]

        cache.set("original query", value, embedding=embedding)
        result = cache.get("different query", embedding=[0.11, 0.19, 0.31, 0.39])

        assert result is not None
        assert result["task_type"] == "code"

    def test_semantic_exact_key_priority(self):
        """精确 key 命中时直接返回，不依赖 embedding。"""
        cache = RouteCache(semantic=True)
        cache.set("q", _sample_value())
        assert cache.get("q") is not None

    def test_semantic_different_bucket_miss(self):
        """embedding 差异较大时不应误命中。"""
        cache = RouteCache(semantic=True)
        cache.set("q", _sample_value(), embedding=[0.1, 0.9])
        assert cache.get("q2", embedding=[0.9, 0.1]) is None
