"""dev-lifecycle 插件 — telemetry.py 遥测模块测试。"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "/home/xxh/.hermes/plugins/dev-lifecycle")

from telemetry import TelemetryRecorder, TelemetryEvent


class TestTelemetryRecorder(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "workflow.db"

    def test_record_and_report(self):
        rec = TelemetryRecorder(db_path=self.db_path)
        rec.record(TelemetryEvent(skill_name="grill-me", stage="ideate", project_type="python"))
        report = rec.report()
        self.assertEqual(report["total_events"], 1)
        self.assertEqual(report["skill_usage"]["grill-me"], 1)
        rec.close()

    def test_empty_report(self):
        rec = TelemetryRecorder(db_path=self.db_path)
        report = rec.report()
        self.assertEqual(report["total_events"], 0)
        rec.close()

    def test_skill_stats(self):
        rec = TelemetryRecorder(db_path=self.db_path)
        rec.record(TelemetryEvent(skill_name="grill-me", stage="ideate", project_type="python", duration=10.0))
        stats = rec.get_skill_stats("grill-me")
        self.assertTrue(stats["found"])
        self.assertEqual(stats["total_uses"], 1)
        rec.close()


if __name__ == "__main__":
    unittest.main()
