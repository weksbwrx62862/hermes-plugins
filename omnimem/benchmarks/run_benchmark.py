import json
import sys
from pathlib import Path

from omnimem.benchmarks.l3_l4_benchmark import L3L4Benchmark

if __name__ == "__main__":
    storage_dir = sys.argv[1] if len(sys.argv) > 1 else ".omnimem_bench"
    bench = L3L4Benchmark(storage_dir)
    results = bench.run_all()
    bench.save_results(Path(storage_dir) / "benchmark_results.json")
    print(json.dumps(results, ensure_ascii=False, indent=2))
