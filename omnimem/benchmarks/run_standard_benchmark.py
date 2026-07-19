"""标准化基准测试统一入口 — 提供运行和快速基准测试的公共接口。

支持 LOCOMO 和 LongMemEval 两种基准测试，输出 JSON 报告。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from omnimem.benchmarks.cost_tracker import CostTracker
from omnimem.benchmarks.locomo_adapter import (
    LOCOMODataset,
    LOCOMOEvaluator,
    LOCOMOQuestion,
    LOCOMOQuestionType,
)
from omnimem.benchmarks.longmemeval_adapter import (
    LongMemEvalCapability,
    LongMemEvalDataset,
    LongMemEvalEvaluator,
    LongMemEvalQuestion,
)

logger = logging.getLogger(__name__)

_SUPPORTED_BENCHMARKS = ("locomo", "longmemeval", "five_dimension")

_QUICK_SAMPLE_SIZE = 10


def run_benchmark(
    benchmark_name: str,
    provider: Any,
    output_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """运行指定基准测试。

    Args:
        benchmark_name: 基准测试名称，支持 "locomo" 或 "longmemeval"
        provider: OmniMemProvider 实例
        output_path: 报告输出路径（JSON），为 None 则不保存文件
        dataset_path: 数据集文件路径，为 None 则使用内置示例数据

    Returns:
        评估结果字典，包含 results 和 metrics

    Raises:
        ValueError: 不支持的基准测试名称
        FileNotFoundError: 数据集文件不存在
    """
    benchmark_name = benchmark_name.lower().strip()
    if benchmark_name not in _SUPPORTED_BENCHMARKS:
        raise ValueError(
            f"不支持的基准测试: '{benchmark_name}'，"
            f"可选: {', '.join(_SUPPORTED_BENCHMARKS)}"
        )

    logger.info("开始运行基准测试: %s", benchmark_name)
    start_time = time.time()

    if benchmark_name == "locomo":
        dataset = _load_locomo_dataset(dataset_path)
        evaluator = LOCOMOEvaluator(provider)
    elif benchmark_name == "five_dimension":
        return run_five_dimension_benchmark(provider, output_path, dataset_path)
    else:
        dataset = _load_longmemeval_dataset(dataset_path)
        evaluator = LongMemEvalEvaluator(provider)

    report = evaluator.run(dataset)
    report["elapsed_seconds"] = round(time.time() - start_time, 2)
    report["timestamp"] = time.time()

    if output_path is not None:
        _save_report(report, output_path)

    logger.info(
        "基准测试 %s 完成: 总题数=%d, 总准确率=%.2f%%, 耗时=%.1fs",
        benchmark_name,
        report["total_questions"],
        report["metrics"].get("overall_accuracy", 0) * 100,
        report["elapsed_seconds"],
    )

    return report


def run_quick_benchmark(provider: Any) -> dict[str, Any]:
    """运行快速基准测试（子集），用于日常验证。

    从 LOCOMO 和 LongMemEval 各抽取少量样本运行，
    快速验证记忆系统基本功能是否正常。

    Returns:
        快速基准测试结果摘要
    """
    logger.info("开始快速基准测试")
    start_time = time.time()

    locomo_ds = _build_quick_locomo_dataset()
    longmemeval_ds = _build_quick_longmemeval_dataset()

    locomo_evaluator = LOCOMOEvaluator(provider)
    longmemeval_evaluator = LongMemEvalEvaluator(provider)

    locomo_report = locomo_evaluator.run(locomo_ds)
    longmemeval_report = longmemeval_evaluator.run(longmemeval_ds)

    elapsed = round(time.time() - start_time, 2)

    summary = {
        "benchmark": "quick",
        "elapsed_seconds": elapsed,
        "timestamp": time.time(),
        "locomo": {
            "total_questions": locomo_report["total_questions"],
            "metrics": locomo_report["metrics"],
        },
        "longmemeval": {
            "total_questions": longmemeval_report["total_questions"],
            "metrics": longmemeval_report["metrics"],
        },
    }

    logger.info(
        "快速基准测试完成: LOCOMO 准确率=%.2f%%, LongMemEval 准确率=%.2f%%, 耗时=%.1fs",
        locomo_report["metrics"].get("overall_accuracy", 0) * 100,
        longmemeval_report["metrics"].get("overall_accuracy", 0) * 100,
        elapsed,
    )

    return summary


def _load_locomo_dataset(dataset_path: str | Path | None) -> LOCOMODataset:
    """加载 LOCOMO 数据集，未指定路径时使用内置示例。"""
    if dataset_path is not None:
        return LOCOMODataset.load_from_json(dataset_path)
    return _build_quick_locomo_dataset()


def _load_longmemeval_dataset(dataset_path: str | Path | None) -> LongMemEvalDataset:
    """加载 LongMemEval 数据集，未指定路径时使用内置示例。"""
    if dataset_path is not None:
        return LongMemEvalDataset.load_from_json(dataset_path)
    return _build_quick_longmemeval_dataset()


def _build_quick_locomo_dataset() -> LOCOMODataset:
    """构建快速 LOCOMO 示例数据集。"""
    questions = [
        LOCOMOQuestion(
            question_id="quick-sh-01",
            question="用户最喜欢的编程语言是什么？",
            answer="Python",
            question_type=LOCOMOQuestionType.SINGLE_HOP,
            evidence_ids=["demo-001"],
        ),
        LOCOMOQuestion(
            question_id="quick-sh-02",
            question="用户养了什么宠物？",
            answer="一只橘猫叫小橘",
            question_type=LOCOMOQuestionType.SINGLE_HOP,
            evidence_ids=["demo-002"],
        ),
        LOCOMOQuestion(
            question_id="quick-mh-01",
            question="用户用什么编程语言开发项目，同时用什么版本控制工具管理代码？",
            answer="用户使用 Python 开发项目，使用 Git 管理代码",
            question_type=LOCOMOQuestionType.MULTI_HOP,
            evidence_ids=["demo-001", "demo-003"],
        ),
        LOCOMOQuestion(
            question_id="quick-tp-01",
            question="用户之前使用的编辑器主题是什么，后来换成了什么？",
            answer="用户之前使用浅色主题，后来换成了深色主题",
            question_type=LOCOMOQuestionType.TEMPORAL,
            evidence_ids=["demo-004"],
        ),
        LOCOMOQuestion(
            question_id="quick-od-01",
            question="请总结用户的技术偏好。",
            answer="用户偏好 Python 编程、深色主题编辑器、Git 版本控制，对函数式编程感兴趣",
            question_type=LOCOMOQuestionType.OPEN_DOMAIN,
            evidence_ids=["demo-001", "demo-003", "demo-004", "demo-005"],
        ),
        LOCOMOQuestion(
            question_id="quick-sh-03",
            question="用户每天早上喝什么？",
            answer="咖啡",
            question_type=LOCOMOQuestionType.SINGLE_HOP,
            evidence_ids=["demo-006"],
        ),
        LOCOMOQuestion(
            question_id="quick-mh-02",
            question="用户的宠物和生活习惯之间有什么联系？",
            answer="用户养了一只橘猫，每天早上喝咖啡",
            question_type=LOCOMOQuestionType.MULTI_HOP,
            evidence_ids=["demo-002", "demo-006"],
        ),
        LOCOMOQuestion(
            question_id="quick-tp-02",
            question="用户最近开始学习什么新技术？",
            answer="深度学习技术",
            question_type=LOCOMOQuestionType.TEMPORAL,
            evidence_ids=["demo-007"],
        ),
        LOCOMOQuestion(
            question_id="quick-od-02",
            question="用户的生活方式有什么特点？",
            answer="用户每天早上喝咖啡，每天工作8小时，周末喜欢阅读技术书籍",
            question_type=LOCOMOQuestionType.OPEN_DOMAIN,
            evidence_ids=["demo-006", "demo-008", "demo-009"],
        ),
        LOCOMOQuestion(
            question_id="quick-sh-04",
            question="用户使用什么 IDE？",
            answer="VS Code",
            question_type=LOCOMOQuestionType.SINGLE_HOP,
            evidence_ids=["demo-010"],
        ),
    ]
    return LOCOMODataset(questions=questions, name="locomo_quick")


def _build_quick_longmemeval_dataset() -> LongMemEvalDataset:
    """构建快速 LongMemEval 示例数据集。"""
    questions = [
        LongMemEvalQuestion(
            question_id="quick-ie-01",
            question="用户最喜欢的编程语言是什么？",
            answer="Python",
            capability=LongMemEvalCapability.INFORMATION_EXTRACTION,
            evidence_ids=["demo-001"],
        ),
        LongMemEvalQuestion(
            question_id="quick-ie-02",
            question="用户的宠物叫什么名字？",
            answer="小橘",
            capability=LongMemEvalCapability.INFORMATION_EXTRACTION,
            evidence_ids=["demo-002"],
        ),
        LongMemEvalQuestion(
            question_id="quick-msr-01",
            question="结合用户的技术偏好和生活习惯，用户可能用什么方式记录学习笔记？",
            answer="用户可能使用 VS Code 编辑器配合 Git 版本控制来管理 Markdown 格式的学习笔记",
            capability=LongMemEvalCapability.MULTI_SESSION_REASONING,
            evidence_ids=["demo-001", "demo-003", "demo-010"],
        ),
        LongMemEvalQuestion(
            question_id="quick-tr-01",
            question="用户最近开始学习什么技术？这和之前的技术偏好有什么变化？",
            answer="用户最近开始学习深度学习，从传统编程转向 AI/ML 方向",
            capability=LongMemEvalCapability.TEMPORAL_REASONING,
            evidence_ids=["demo-001", "demo-007"],
        ),
        LongMemEvalQuestion(
            question_id="quick-ku-01",
            question="用户之前使用浅色主题编辑器，现在呢？",
            answer="用户现在使用深色主题编辑器",
            capability=LongMemEvalCapability.KNOWLEDGE_UPDATE,
            evidence_ids=["demo-004"],
        ),
        LongMemEvalQuestion(
            question_id="quick-rf-01",
            question="用户昨天晚餐吃了什么？",
            answer="无法回答，没有相关记忆",
            capability=LongMemEvalCapability.REFUSAL,
            evidence_ids=[],
        ),
        LongMemEvalQuestion(
            question_id="quick-ie-03",
            question="用户每天工作多长时间？",
            answer="8小时",
            capability=LongMemEvalCapability.INFORMATION_EXTRACTION,
            evidence_ids=["demo-008"],
        ),
        LongMemEvalQuestion(
            question_id="quick-msr-02",
            question="用户的技术兴趣和生活习惯如何影响其周末安排？",
            answer="用户周末喜欢阅读技术书籍，对函数式编程感兴趣",
            capability=LongMemEvalCapability.MULTI_SESSION_REASONING,
            evidence_ids=["demo-005", "demo-009"],
        ),
        LongMemEvalQuestion(
            question_id="quick-tr-02",
            question="用户的编辑器主题偏好发生了什么变化？",
            answer="从浅色主题换成了深色主题",
            capability=LongMemEvalCapability.TEMPORAL_REASONING,
            evidence_ids=["demo-004"],
        ),
        LongMemEvalQuestion(
            question_id="quick-ku-02",
            question="用户对函数式编程的态度是什么？",
            answer="用户对函数式编程感兴趣",
            capability=LongMemEvalCapability.KNOWLEDGE_UPDATE,
            evidence_ids=["demo-005"],
        ),
    ]
    return LongMemEvalDataset(questions=questions, name="longmemeval_quick")


def _save_report(report: dict[str, Any], output_path: str | Path) -> None:
    """将评估报告保存为 JSON 文件。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("基准测试报告已保存: %s", output)


def main() -> None:
    """命令行入口：运行指定基准测试。"""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("用法: python -m omnimem.benchmarks.run_standard_benchmark <benchmark_name> [dataset_path] [output_path]")
        print(f"支持的基准测试: {', '.join(_SUPPORTED_BENCHMARKS)}, quick")
        sys.exit(1)

    benchmark_name = sys.argv[1]

    if benchmark_name == "quick":
        from omnimem.provider import OmniMemProvider
        provider = OmniMemProvider()
        provider.initialize(session_id="bench-quick")
        result = run_quick_benchmark(provider)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if benchmark_name == "five_dimension":
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        dataset_path = sys.argv[3] if len(sys.argv) > 3 else None
        from omnimem.provider import OmniMemProvider
        provider = OmniMemProvider()
        provider.initialize(session_id="bench-five-dimension")
        report = run_five_dimension_benchmark(provider, output_path, dataset_path)
        print(json.dumps(report.get("dimensions", {}), ensure_ascii=False, indent=2))
        return

    dataset_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    from omnimem.provider import OmniMemProvider
    provider = OmniMemProvider()
    provider.initialize(session_id=f"bench-{benchmark_name}")

    report = run_benchmark(
        benchmark_name=benchmark_name,
        provider=provider,
        output_path=output_path,
        dataset_path=dataset_path,
    )

    print(json.dumps(report.get("metrics", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


# ★ P2-2 五维评测 — 对照论文《Are We Ready For An Agent-Native Memory System?》的 5 个评测维度

_FIVE_DIMENSION_MAP: dict[str, list[str]] = {
    "task_performance": ["locomo_overall", "longmemeval_overall"],
    "evidence_retrieval": ["locomo_single_hop", "longmemeval_information_extraction"],
    "dynamic_update": ["longmemeval_knowledge_update", "longmemeval_temporal_reasoning"],
    "long_term_stability": ["locomo_temporal", "locomo_multi_hop"],
    "operational_cost": ["cost_tracker"],
}


def run_five_dimension_benchmark(
    provider: Any,
    output_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """运行五维基准测试，输出统一报告。

    对照论文建议的五维评估框架：
      1. task_performance — 端到端任务效果 (LOCOMO + LongMemEval)
      2. evidence_retrieval — 证据检索保真度
      3. dynamic_update — 动态更新与时序推理
      4. long_term_stability — 长程稳定性
      5. operational_cost — 操作成本

    Args:
        provider: OmniMemProvider 实例
        output_path: 报告输出路径（JSON）
        dataset_path: 数据集路径（可选）

    Returns:
        五维评测报告
    """
    logger.info("=== 开始五维基准测试 ===")
    start_time = time.time()

    # 1. 运行 LOCOMO + LongMemEval
    locomo_ds = _load_locomo_dataset(dataset_path)
    longmemeval_ds = _load_longmemeval_dataset(dataset_path)

    locomo_evaluator = LOCOMOEvaluator(provider)
    longmemeval_evaluator = LongMemEvalEvaluator(provider)

    locomo_report = locomo_evaluator.run(locomo_ds)
    longmemeval_report = longmemeval_evaluator.run(longmemeval_ds)

    # 2. 运行成本追踪（在 benchmark 执行期间记录的操作成本）
    cost_tracker = CostTracker()
    cost_tracker.record(
        "benchmark_locomo",
        latency_ms=locomo_report.get("elapsed_seconds", 0) * 1000,
        result_count=locomo_report.get("total_questions", 0),
    )
    cost_tracker.record(
        "benchmark_longmemeval",
        latency_ms=longmemeval_report.get("elapsed_seconds", 0) * 1000,
        result_count=longmemeval_report.get("total_questions", 0),
    )

    # 3. 聚合五维得分
    locomo_metrics = locomo_report.get("metrics", {})
    longmemeval_metrics = longmemeval_report.get("metrics", {})

    dimensions = {
        "task_performance": {
            "name": "端到端任务效果",
            "score": round(
                (locomo_metrics.get("overall_accuracy", 0)
                 + longmemeval_metrics.get("overall_accuracy", 0)) / 2 * 100,
                2,
            ),
            "locomo_accuracy": round(locomo_metrics.get("overall_accuracy", 0) * 100, 2),
            "longmemeval_accuracy": round(longmemeval_metrics.get("overall_accuracy", 0) * 100, 2),
        },
        "evidence_retrieval": {
            "name": "证据检索保真度",
            "score": round(
                locomo_metrics.get("single_hop_accuracy", locomo_metrics.get("overall_accuracy", 0)) * 100,
                2,
            ),
            "locomo_single_hop": round(
                locomo_metrics.get("single_hop_accuracy", 0) * 100, 2,
            ),
            "longmemeval_information_extraction": round(
                longmemeval_metrics.get("information_extraction_accuracy", 0) * 100, 2,
            ),
        },
        "dynamic_update": {
            "name": "动态更新与时序推理",
            "score": round(
                (longmemeval_metrics.get("knowledge_update_accuracy", 0)
                 + longmemeval_metrics.get("temporal_reasoning_accuracy", 0)) / 2 * 100,
                2,
            ),
            "knowledge_update": round(
                longmemeval_metrics.get("knowledge_update_accuracy", 0) * 100, 2,
            ),
            "temporal_reasoning": round(
                longmemeval_metrics.get("temporal_reasoning_accuracy", 0) * 100, 2,
            ),
        },
        "long_term_stability": {
            "name": "长程稳定性（跨会话推理）",
            "score": round(
                (locomo_metrics.get("multi_hop_accuracy", 0)
                 + locomo_metrics.get("temporal_accuracy", 0)) / 2 * 100,
                2,
            ),
            "locomo_multi_hop": round(
                locomo_metrics.get("multi_hop_accuracy", 0) * 100, 2,
            ),
            "locomo_temporal": round(
                locomo_metrics.get("temporal_accuracy", 0) * 100, 2,
            ),
        },
        "operational_cost": {
            "name": "操作成本",
            "cost_summary": cost_tracker.report(),
            "total_operations": cost_tracker.report()["total_operations"],
        },
    }

    elapsed = round(time.time() - start_time, 2)

    report = {
        "benchmark": "five_dimension",
        "elapsed_seconds": elapsed,
        "timestamp": time.time(),
        "dimensions": dimensions,
        "average_score": round(
            sum(
                d.get("score", 0) for d in dimensions.values()
                if "score" in d
            ) / 4,  # cost 维度无 score
            2,
        ),
        "locomo_detail": {
            k: round(v * 100, 2) if isinstance(v, float) else v
            for k, v in locomo_metrics.items()
        },
        "longmemeval_detail": {
            k: round(v * 100, 2) if isinstance(v, float) else v
            for k, v in longmemeval_metrics.items()
        },
    }

    logger.info(
        "五维评测完成: 平均得分=%.2f%%, 耗时=%.1fs",
        report["average_score"],
        elapsed,
    )

    if output_path is not None:
        _save_report(report, output_path)

    return report
