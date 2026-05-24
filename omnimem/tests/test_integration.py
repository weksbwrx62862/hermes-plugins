"""集成测试：端到端记忆写入→检索→精炼。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from omnimem.context.manager import ContextManager
from omnimem.governance.decay import TemporalDecay
from omnimem.governance.privacy import PrivacyManager
from omnimem.memory.drawer_closet import DrawerClosetStore
from omnimem.memory.index import ThreeLevelIndex
from omnimem.perception.engine import PerceptionEngine


class TestIntegration(unittest.TestCase):
    """集成测试：端到端记忆写入→检索→精炼。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.store = DrawerClosetStore(Path(self.tmpdir) / "palace")
        self.index = ThreeLevelIndex(Path(self.tmpdir) / "index")
        self.cm = ContextManager()
        self.pe = PerceptionEngine()

    def tearDown(self) -> None:
        self.index.close()

    def test_write_perceive_refine(self) -> None:
        mid = self.store.add(
            wing="personal",
            room="python",
            content="用户喜欢Python编程语言",
            memory_type="preference",
            confidence=5,
            privacy="personal",
        )
        self.assertTrue(mid)

        signals = self.pe.detect_signals("我喜欢Python")
        self.assertTrue(signals.has_preference)

        results = self.store.search_by_content("Python")
        self.assertTrue(len(results) > 0)

        refined = self.cm.refine_prefetch_results(results)
        self.assertIn("### Relevant Memories", refined)

    def test_multi_type_storage_and_search(self) -> None:
        self.store.add(wing="personal", room="r1", content="事实1", memory_type="fact")
        self.store.add(wing="personal", room="r2", content="偏好1", memory_type="preference")
        self.store.add(wing="personal", room="r3", content="纠正1", memory_type="correction")

        facts = self.store.search(memory_type="fact")
        prefs = self.store.search(memory_type="preference")
        corrections = self.store.search(memory_type="correction")

        self.assertTrue(any(f["type"] == "fact" for f in facts))
        self.assertTrue(any(p["type"] == "preference" for p in prefs))
        self.assertTrue(any(c["type"] == "correction" for c in corrections))

    def test_privacy_filter_integration(self) -> None:
        self.store.add(
            wing="personal", room="r1", content="公开", memory_type="fact", privacy="public"
        )
        self.store.add(
            wing="personal", room="r2", content="秘密", memory_type="fact", privacy="secret"
        )

        all_items = self.store.search(limit=10)
        pm = PrivacyManager()
        filtered = pm.filter(all_items)
        secret_items = [r for r in filtered if r.get("privacy") == "secret"]
        self.assertEqual(len(secret_items), 1)
        self.assertTrue(secret_items[0].get("_encrypted"))
        self.assertTrue(secret_items[0]["content"].startswith("[加密记忆"))

    def test_temporal_decay_integration(self) -> None:
        self.store.add(
            wing="personal",
            room="r1",
            content="新事件",
            memory_type="event",
            confidence=3,
            privacy="personal",
        )
        self.store.add(
            wing="personal",
            room="r2",
            content="旧事件",
            memory_type="event",
            confidence=3,
            privacy="personal",
        )

        results = self.store.search(memory_type="event")
        now = datetime.now(timezone.utc)
        results[0]["stored_at"] = now.isoformat()
        results[0]["score"] = 1.0
        results[0]["type"] = "event"
        results[1]["stored_at"] = (now - timedelta(days=180)).isoformat()
        results[1]["score"] = 1.0
        results[1]["type"] = "event"

        td = TemporalDecay()
        decayed = td.apply(results)
        self.assertGreater(decayed[0]["score"], decayed[1]["score"])
