"""基准测试的单元验证 — 确认 benchmarks/fullstack_benchmark.py 可正常导入运行。"""

from __future__ import annotations

import json
import tempfile
import unittest

from omnimem.benchmarks.fullstack_benchmark import FullStackBenchmark


class TestFullStackBenchmark(unittest.TestCase):

    def test_runs_without_error(self) -> None:
        bm = FullStackBenchmark(tempfile.mkdtemp())
        results = bm.run_all()
        self.assertIsInstance(results, dict)
        self.assertIn("write_throughput", results)
        self.assertIn("dedup_accuracy", results)
        self.assertIn("entity_extraction", results)
        self.assertIn("trust_scoring", results)
        self.assertIn("security_fencing", results)

    def test_write_throughput_positive(self) -> None:
        bm = FullStackBenchmark(tempfile.mkdtemp())
        results = bm.run_all()
        self.assertGreater(results["write_throughput"]["ops"], 0)
        self.assertGreater(results["write_throughput"]["throughput_ops_per_sec"], 100)

    def test_dedup_accuracy_valid(self) -> None:
        bm = FullStackBenchmark(tempfile.mkdtemp())
        results = bm.run_all()
        acc = results["dedup_accuracy"]["accuracy"]
        self.assertGreaterEqual(acc, 0.5)

    def test_save_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bm = FullStackBenchmark(tmpdir)
            bm.run_all()
            out = f"{tmpdir}/bench_results.json"
            bm.save_results(out)
            with open(out) as f:
                data = json.load(f)
            self.assertIn("write_throughput", data)
