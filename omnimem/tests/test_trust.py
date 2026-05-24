"""信任评分和反馈测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omnimem.governance.feedback import FeedbackCollector


class TestTrustScoring(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.fb = FeedbackCollector(Path(self.tmpdir))

    def test_default_trust(self) -> None:
        trust = self.fb.get_memory_trust("nonexistent")
        self.assertGreaterEqual(trust, 0.0)
        self.assertLessEqual(trust, 1.0)

    def test_click_increases_trust(self) -> None:
        self.fb.record_click("q1", "m1")
        trust = self.fb.get_memory_trust("m1")
        self.assertGreater(trust, 0.0)

    def test_multiple_clicks(self) -> None:
        for _ in range(5):
            self.fb.record_click("q", "frequent")
        trust = self.fb.get_memory_trust("frequent")
        self.assertGreater(trust, 0.1)

    def test_no_clicks_low_trust(self) -> None:
        """有展示但未点击的记忆信任低。"""
        self.fb.record_shown("q", [
            {"memory_id": "shown_only", "_source": "vector"}
        ])
        trust = self.fb.get_memory_trust("shown_only")
        # 展示不增加信任，应保持默认接近0.5
        self.assertLess(trust, 0.55)

    def test_trust_bounded(self) -> None:
        for _ in range(30):
            self.fb.record_click("q", "maxed_out")
        trust = self.fb.get_memory_trust("maxed_out")
        self.assertLessEqual(trust, 1.0)
