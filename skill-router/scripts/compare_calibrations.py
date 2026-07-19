#!/usr/bin/env python3
"""对比不同 hybrid_calibration 策略下的混合检索分数分布

复用 scripts/evaluate.py 的评估逻辑，分别运行 minmax / sigmoid / zscore，
输出各策略的 Top-1/Top-3 准确率与置信度分布。
"""

import os
import sys

# 将插件根目录加入路径，确保能 import evaluate
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import evaluate  # noqa: E402


def _run_calibration(name: str) -> None:
    """设置校准策略并运行 evaluate.main()"""
    evaluate.HYBRID_CALIBRATION = name
    print(f"\n{'=' * 60}")
    print(f"校准策略: {name}")
    print("=" * 60)
    evaluate.main()


def main() -> None:
    """依次运行三种校准策略"""
    for strategy in ("minmax", "sigmoid", "zscore"):
        _run_calibration(strategy)


if __name__ == "__main__":
    main()
