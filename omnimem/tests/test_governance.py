"""治理引擎模块测试。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from omnimem.governance.conflict import ConflictResolver, ConflictResult
from omnimem.governance.decay import TemporalDecay
from omnimem.governance.forgetting import ForgettingCurve
from omnimem.governance.privacy import PrivacyManager
from omnimem.memory.drawer_closet import DrawerClosetStore


class TestConflictResolver(unittest.TestCase):
    """ConflictResolver 测试。"""

    def test_no_conflict(self) -> None:
        cr = ConflictResolver(strategy="latest")
        result = cr.check("普通事实内容")
        self.assertFalse(result.has_conflict)

    def test_negation_without_existing(self) -> None:
        cr = ConflictResolver(strategy="latest")
        result = cr.check("纠正: 应该是Python")
        self.assertFalse(result.has_conflict)

    def test_negation_with_contradiction(self) -> None:
        cr = ConflictResolver(strategy="latest")
        existing = [{"content": "项目使用Java开发", "memory_id": "old-1"}]
        result = cr.check("项目使用Python开发，不对，应该不是Java", existing)
        self.assertIsInstance(result, ConflictResult)

    def test_mutual_exclusive(self) -> None:
        cr = ConflictResolver(strategy="latest")
        existing = [{"content": "使用AWS部署服务", "memory_id": "old-1"}]
        result = cr.check("使用腾讯云部署", existing)
        self.assertTrue(result.has_conflict)
        self.assertEqual(result.conflict_type, "semantic_contradiction")

    def test_resolve_latest(self) -> None:
        cr = ConflictResolver(strategy="latest")
        conflict = ConflictResult(has_conflict=True, existing_id="old-1")
        result = cr.resolve("新内容", conflict)
        self.assertEqual(result.action, "accept")

    def test_compute_overlap(self) -> None:
        overlap = ConflictResolver._compute_overlap("Python编程", "Python开发")
        self.assertGreater(overlap, 0)

    def test_compute_overlap_no_overlap(self) -> None:
        overlap = ConflictResolver._compute_overlap("量子计算", "烹饪食谱")
        self.assertLess(overlap, 0.3)


class TestTemporalDecay(unittest.TestCase):
    """TemporalDecay 测试。"""

    def test_fact_no_decay(self) -> None:
        td = TemporalDecay()
        results = [
            {
                "content": "事实",
                "type": "fact",
                "score": 1.0,
                "stored_at": (datetime.now(timezone.utc) - timedelta(days=100)).isoformat(),
            }
        ]
        decayed = td.apply(results)
        self.assertEqual(decayed[0]["score"], 1.0)

    def test_event_decay(self) -> None:
        td = TemporalDecay()
        results = [
            {
                "content": "事件",
                "type": "event",
                "score": 1.0,
                "stored_at": (datetime.now(timezone.utc) - timedelta(days=90)).isoformat(),
            }
        ]
        decayed = td.apply(results)
        self.assertLess(decayed[0]["score"], 1.0)
        self.assertIn("decay_factor", decayed[0])

    def test_preference_half_life(self) -> None:
        td = TemporalDecay()
        half_life = td.get_half_life("preference")
        self.assertEqual(half_life, 180)

    def test_custom_half_life(self) -> None:
        td = TemporalDecay(custom_half_lives={"event": 30})
        self.assertEqual(td.get_half_life("event"), 30)

    def test_sorted_by_score(self) -> None:
        td = TemporalDecay()
        results = [
            {
                "content": "旧事件",
                "type": "event",
                "score": 1.0,
                "stored_at": (datetime.now(timezone.utc) - timedelta(days=180)).isoformat(),
            },
            {
                "content": "新事件",
                "type": "event",
                "score": 1.0,
                "stored_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            },
        ]
        decayed = td.apply(results)
        self.assertGreater(decayed[0]["score"], decayed[1]["score"])


class TestForgettingCurve(unittest.TestCase):
    """ForgettingCurve 测试。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.fc = ForgettingCurve(Path(self.tmpdir))

    def tearDown(self) -> None:
        self.fc.close()

    def test_default_stage(self) -> None:
        stage = self.fc.get_stage("new-memory")
        self.assertEqual(stage, "active")

    def test_archive(self) -> None:
        self.fc.archive("mem-1")
        self.fc.flush()
        stage = self.fc.get_stage("mem-1")
        self.assertEqual(stage, "archived")

    def test_archive_twice_to_forgotten(self) -> None:
        self.fc.archive("mem-2")
        self.fc.flush()
        self.fc.archive("mem-2")
        self.fc.flush()
        stage = self.fc.get_stage("mem-2")
        self.assertEqual(stage, "forgotten")

    def test_forgotten_not_archived_again(self) -> None:
        self.fc.archive("mem-3")
        self.fc.flush()
        self.fc.archive("mem-3")
        self.fc.flush()
        self.fc.archive("mem-3")
        self.fc.flush()
        stage = self.fc.get_stage("mem-3")
        self.assertEqual(stage, "forgotten")

    def test_reactivate(self) -> None:
        self.fc.archive("mem-4")
        self.fc.flush()
        self.fc.reactivate("mem-4")
        self.fc.flush()
        stage = self.fc.get_stage("mem-4")
        self.assertEqual(stage, "active")

    def test_get_stage_by_age(self) -> None:
        self.assertEqual(self.fc.get_stage_by_age(0), "active")
        self.assertEqual(self.fc.get_stage_by_age(5), "active")
        self.assertEqual(self.fc.get_stage_by_age(10), "consolidating")
        self.assertEqual(self.fc.get_stage_by_age(60), "archived")
        self.assertEqual(self.fc.get_stage_by_age(120), "forgotten")

    def test_record_access(self) -> None:
        self.fc.record_access("mem-access")
        self.fc.flush()
        stage = self.fc.get_stage("mem-access")
        self.assertEqual(stage, "active")

    def test_get_status(self) -> None:
        self.fc.archive("s-1")
        self.fc.flush()
        status = self.fc.get_status()
        self.assertIn("stages", status)
        self.assertIn("active", status["stages"])
        self.assertIn("archived", status["stages"])


class TestPrivacyManager(unittest.TestCase):
    """PrivacyManager 测试。"""

    def test_default_level(self) -> None:
        pm = PrivacyManager()
        self.assertEqual(pm.get("any-id"), "personal")

    def test_set_and_get(self) -> None:
        pm = PrivacyManager()
        pm.set("m1", "public")
        self.assertEqual(pm.get("m1"), "public")

    def test_filter_secret(self) -> None:
        pm = PrivacyManager()
        results = [
            {"content": "公开", "privacy": "public"},
            {"content": "秘密", "privacy": "secret"},
        ]
        filtered = pm.filter(results)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["privacy"], "public")
        self.assertEqual(filtered[1]["privacy"], "secret")
        self.assertTrue(filtered[1].get("_encrypted"))

    def test_filter_by_max_privacy(self) -> None:
        pm = PrivacyManager()
        results = [
            {"content": "公开", "privacy": "public"},
            {"content": "团队", "privacy": "team"},
            {"content": "个人", "privacy": "personal"},
        ]
        filtered = pm.filter(results, max_privacy="team")
        self.assertTrue(all(r["privacy"] in ("public", "team") for r in filtered))

    def test_bind_store_and_persist(self) -> None:
        tmpdir = tempfile.mkdtemp()
        store = DrawerClosetStore(Path(tmpdir))
        pm = PrivacyManager()
        pm.bind_store(store)
        mid = store.add(wing="personal", room="r", content="隐私测试", privacy="personal")
        pm.set(mid, "team")
        result = store.get(mid)
        self.assertEqual(result["privacy"], "team")  # type: ignore[index]

    def test_invalid_level_ignored(self) -> None:
        pm = PrivacyManager()
        pm.set("m1", "invalid_level")
        self.assertEqual(pm.get("m1"), "personal")
