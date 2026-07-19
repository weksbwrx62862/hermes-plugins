"""dev-lifecycle 插件 — gates.py 质量门禁测试。"""

import sys
import unittest

sys.path.insert(0, "/home/xxh/.hermes/plugins/dev-lifecycle")

from gates import QualityGateManager, GateResult, GateCheck


class TestGateResult(unittest.TestCase):

    def test_gate_result_defaults(self):
        result = GateResult(passed=True)
        self.assertTrue(result.passed)
        self.assertEqual(result.failures, [])
        self.assertEqual(result.suggestions, [])


class TestQualityGateManager(unittest.TestCase):

    def test_ideate_to_build_passed(self):
        mgr = QualityGateManager()
        ctx = {"prd_path": "/prd.md", "issues_count": 3, "plan_path": "/plan.md"}
        result = mgr.check("ideate", "build", ctx)
        self.assertTrue(result.passed)

    def test_ideate_to_build_failed(self):
        mgr = QualityGateManager()
        result = mgr.check("ideate", "build", {})
        self.assertFalse(result.passed)
        self.assertGreater(len(result.failures), 0)

    def test_build_to_deliver_passed(self):
        mgr = QualityGateManager()
        ctx = {"tests_passed": True, "review_completed": True, "blocking_bugs": 0}
        result = mgr.check("build", "deliver", ctx)
        self.assertTrue(result.passed)

    def test_build_to_deliver_failed(self):
        mgr = QualityGateManager()
        ctx = {"tests_passed": False}
        result = mgr.check("build", "deliver", ctx)
        self.assertFalse(result.passed)

    def test_unknown_transition_passes(self):
        mgr = QualityGateManager()
        result = mgr.check("unknown_a", "unknown_b", {})
        self.assertTrue(result.passed)

    def test_custom_gate(self):
        mgr = QualityGateManager()
        custom = GateCheck(name="custom", check_fn=lambda ctx: (True, "ok"), description="test")
        mgr.register_gate("a", "b", custom)
        result = mgr.check("a", "b", {})
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
