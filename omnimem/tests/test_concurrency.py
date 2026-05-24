"""SQLite 并发安全测试。"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from omnimem.internalize.kv_cache import KVCacheManager
from omnimem.memory.index import ThreeLevelIndex
from omnimem.memory.meta_store import MetaStore


class TestSQLiteConcurrency(unittest.TestCase):

    def test_meta_store_concurrent_writes(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        meta = MetaStore(tmpdir / "meta")
        errors = []
        num_threads = 5
        writes_per_thread = 20

        def writer(thread_id: int) -> None:
            try:
                for i in range(writes_per_thread):
                    meta.add(
                        memory_id=f"t{thread_id}-m{i}",
                        wing="personal",
                        hall="facts",
                        room="test",
                        type="fact",
                        confidence=3,
                        privacy="personal",
                        stored_at=f"2025-01-01T00:{thread_id:02d}:{i:02d}Z",
                        summary=f"Thread {thread_id} memory {i}",
                        content_preview=f"Content from thread {thread_id} item {i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(tid,)) for tid in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        self.assertEqual(len(errors), 0, f"Concurrent write errors: {errors}")
        self.assertEqual(meta.count(), num_threads * writes_per_thread)

    def test_kv_cache_concurrent_access(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        kv = KVCacheManager(data_dir=tmpdir / "kv", auto_preload_threshold=3, max_cache_size=50)
        errors = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(10):
                    key = f"key-t{thread_id}-{i}"
                    kv.check_and_auto_preload(
                        key=key,
                        content=f"Content from thread {thread_id} item {i}",
                        metadata={"source": "test"},
                    )
            except Exception as e:
                errors.append(e)

        def reader(thread_id: int) -> None:
            try:
                for i in range(10):
                    key = f"key-t{thread_id}-{i}"
                    kv.get(key)
                    kv.search_cache(f"thread {thread_id}")
            except Exception as e:
                errors.append(e)

        threads = []
        for tid in range(3):
            threads.append(threading.Thread(target=writer, args=(tid,)))
            threads.append(threading.Thread(target=reader, args=(tid,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")

    def test_index_concurrent_commits(self) -> None:
        tmpdir = Path(tempfile.mkdtemp())
        index = ThreeLevelIndex(tmpdir / "idx")
        errors = []
        num_threads = 5
        writes_per_thread = 10

        def writer(thread_id: int) -> None:
            try:
                for i in range(writes_per_thread):
                    index.add(
                        memory_id=f"idx-t{thread_id}-{i}",
                        wing="personal",
                        hall="facts",
                        room="test",
                        content=f"Index content from thread {thread_id} item {i}",
                        type="fact",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(tid,)) for tid in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        index.flush()
        self.assertEqual(len(errors), 0, f"Concurrent commit errors: {errors}")
        total = len(index.search_all_for_retrieval(limit=1000))
        self.assertGreater(total, 0)
        index.close()
