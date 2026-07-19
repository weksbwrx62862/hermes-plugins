"""LOCOMO 基准测试适配器 — 评估长上下文记忆系统的多跳推理能力。

支持四种问题类型：单跳、多跳、时序、开放域。
使用 LLM-as-Judge 评估答案正确性，按类型统计准确率。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from omnimem.utils.llm_client import AsyncLLMClient

logger = logging.getLogger(__name__)


class LOCOMOQuestionType(str, Enum):
    """LOCOMO 问题类型枚举。"""
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    OPEN_DOMAIN = "open_domain"


@dataclass
class LOCOMOQuestion:
    """LOCOMO 单条问题数据。"""
    question_id: str
    question: str
    answer: str
    question_type: LOCOMOQuestionType
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class LOCOMODataset:
    """LOCOMO 数据集容器。"""
    questions: list[LOCOMOQuestion] = field(default_factory=list)
    name: str = "locomo"

    @classmethod
    def load_from_json(cls, path: str | Path) -> LOCOMODataset:
        """从 JSON 文件加载 LOCOMO 数据集。

        JSON 格式要求：
        {
            "name": "locomo_v1",
            "questions": [
                {
                    "question_id": "q001",
                    "question": "...",
                    "answer": "...",
                    "question_type": "single_hop",
                    "evidence_ids": ["mem-001", "mem-002"]
                }
            ]
        }
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LOCOMO 数据集文件不存在: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"LOCOMO 数据集 JSON 解析失败: {e}") from e

        questions = []
        for item in data.get("questions", []):
            try:
                q_type = LOCOMOQuestionType(item["question_type"])
            except ValueError:
                logger.warning(
                    "跳过未知问题类型 '%s' (question_id=%s)",
                    item.get("question_type"), item.get("question_id"),
                )
                continue

            questions.append(LOCOMOQuestion(
                question_id=item["question_id"],
                question=item["question"],
                answer=item["answer"],
                question_type=q_type,
                evidence_ids=item.get("evidence_ids", []),
            ))

        if not questions:
            raise ValueError("LOCOMO 数据集为空或所有问题的类型均无效")

        logger.info("LOCOMO 数据集加载完成: %d 道题目, 来源=%s", len(questions), path)
        return cls(questions=questions, name=data.get("name", "locomo"))

    @classmethod
    def load_from_hf(cls, split: str = "test") -> LOCOMODataset:
        """从 HuggingFace 加载 LOCOMO 数据集（可选）。

        需要安装 datasets 库: pip install datasets
        数据集: kernel-loopy/LOCOMO
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "加载 HuggingFace 数据集需要 datasets 库，"
                "请安装: pip install datasets"
            ) from None

        try:
            ds = load_dataset("kernel-loopy/LOCOMO", split=split)
        except Exception as e:
            raise RuntimeError(f"从 HuggingFace 加载 LOCOMO 数据集失败: {e}") from e

        questions = []
        for idx, item in enumerate(ds):
            q_type_str = item.get("question_type", "single_hop")
            try:
                q_type = LOCOMOQuestionType(q_type_str)
            except ValueError:
                q_type = LOCOMOQuestionType.SINGLE_HOP

            questions.append(LOCOMOQuestion(
                question_id=item.get("question_id", f"hf-{idx:04d}"),
                question=item["question"],
                answer=item.get("answer", ""),
                question_type=q_type,
                evidence_ids=item.get("evidence_ids", []),
            ))

        logger.info("LOCOMO 数据集从 HuggingFace 加载完成: %d 道题目, split=%s", len(questions), split)
        return cls(questions=questions, name=f"locomo_hf_{split}")


_JUDGE_SYSTEM_PROMPT = """你是一个严格的答案评判专家。你的任务是判断预测答案是否与标准答案语义一致。

评判标准：
- 核心信息一致即为正确，不要求措辞完全相同
- 如果预测答案包含标准答案的核心事实，视为正确
- 如果预测答案遗漏关键信息或包含矛盾信息，视为错误
- 只回答 "correct" 或 "incorrect"，不要解释"""

_JUDGE_USER_TEMPLATE = """标准答案: {gold_answer}
预测答案: {prediction}

请判断预测答案是否正确。"""


class LOCOMOEvaluator:
    """LOCOMO 基准测试评估器。

    对每个问题调用 OmniMemProvider 的 recall 接口获取预测答案，
    使用 LLM-as-Judge 评估正确性，按问题类型统计准确率。
    """

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self._llm_client = self._init_judge_client()

    def _init_judge_client(self) -> AsyncLLMClient | None:
        """初始化 Judge 用的 LLM 客户端，复用 provider 的凭证。"""
        existing = getattr(self._provider, "_llm_client", None)
        if existing and isinstance(existing, AsyncLLMClient):
            api_key = getattr(existing, "_api_key", "")
            base_url = getattr(existing, "_base_url", "")
            model = getattr(existing, "_model", "glm-5.1")
            if api_key and base_url:
                return AsyncLLMClient(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    max_concurrent=3,
                    timeout=30.0,
                    cache_ttl=0.0,
                )

        creds = AsyncLLMClient.load_credentials_from_hermes_config()
        if creds.get("api_key") and creds.get("base_url"):
            return AsyncLLMClient(
                api_key=creds["api_key"],
                base_url=creds["base_url"],
                model=creds.get("model", "glm-5.1"),
                max_concurrent=3,
                timeout=30.0,
                cache_ttl=0.0,
            )

        logger.warning("LOCOMOEvaluator: 无法初始化 Judge LLM 客户端，将使用关键词匹配")
        return None

    def _judge_answer(self, prediction: str, gold_answer: str) -> bool:
        """使用 LLM-as-Judge 判断预测答案是否正确。"""
        if not prediction.strip():
            return False

        if self._llm_client is None:
            return self._keyword_match(prediction, gold_answer)

        prompt = _JUDGE_USER_TEMPLATE.format(gold_answer=gold_answer, prediction=prediction)
        try:
            result = self._llm_client.call_sync(
                prompt=prompt,
                system=_JUDGE_SYSTEM_PROMPT,
                max_tokens=10,
                temperature=0.0,
                use_cache=False,
            )
            verdict = result.content.strip().lower() if result and result.content else ""
            return verdict.startswith("correct")
        except Exception as e:
            logger.warning("LLM Judge 调用失败，回退到关键词匹配: %s", e)
            return self._keyword_match(prediction, gold_answer)

    @staticmethod
    def _keyword_match(prediction: str, gold_answer: str) -> bool:
        """关键词匹配回退方案：标准答案的核心词出现在预测中即视为正确。"""
        gold_words = set(gold_answer.lower().split())
        if not gold_words:
            return False
        pred_lower = prediction.lower()
        matched = sum(1 for w in gold_words if w in pred_lower)
        return matched / len(gold_words) >= 0.5

    def _recall_question(self, question: str) -> tuple[str, float, int]:
        """对单个问题执行 recall，返回 (预测答案, 延迟ms, token数)。"""
        start = time.perf_counter()
        try:
            raw = self._provider._handle_recall({"query": question, "mode": "rag"})
            latency_ms = (time.perf_counter() - start) * 1000

            data = json.loads(raw)
            if data.get("status") == "found" and data.get("memories"):
                memories = data["memories"]
                parts = []
                total_chars = 0
                for mem in memories:
                    content = mem.get("content", "") or mem.get("summary", "")
                    if content:
                        parts.append(content)
                        total_chars += len(content)
                prediction = "\n".join(parts)
                token_count = total_chars // 4
                return prediction, latency_ms, token_count
            else:
                return "", latency_ms, 0
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("LOCOMO recall 失败 (query=%s): %s", question[:50], e)
            return "", latency_ms, 0

    def run(self, dataset: LOCOMODataset) -> dict[str, Any]:
        """运行 LOCOMO 评估。

        对每个问题执行 recall + Judge 评估，收集详细结果。
        """
        results: list[dict[str, Any]] = []
        total = len(dataset.questions)

        for i, q in enumerate(dataset.questions):
            prediction, latency_ms, token_count = self._recall_question(q.question)
            is_correct = self._judge_answer(prediction, q.answer) if prediction else False

            results.append({
                "question_id": q.question_id,
                "question_type": q.question_type.value,
                "question": q.question,
                "gold_answer": q.answer,
                "prediction": prediction[:500],
                "is_correct": is_correct,
                "latency_ms": round(latency_ms, 3),
                "token_count": token_count,
                "evidence_ids": q.evidence_ids,
            })

            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info("LOCOMO 评估进度: %d/%d", i + 1, total)

        return {
            "benchmark": "locomo",
            "dataset_name": dataset.name,
            "total_questions": total,
            "results": results,
            "metrics": self.compute_metrics(results),
        }

    def compute_metrics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """计算汇总指标，按问题类型统计准确率。"""
        if not results:
            return {
                "overall_accuracy": 0.0,
                "single_hop_accuracy": 0.0,
                "multi_hop_accuracy": 0.0,
                "temporal_accuracy": 0.0,
                "open_domain_accuracy": 0.0,
                "avg_latency_ms": 0.0,
                "avg_token_count": 0.0,
            }

        type_stats: dict[str, list[bool]] = {
            t.value: [] for t in LOCOMOQuestionType
        }
        latencies: list[float] = []
        token_counts: list[int] = []

        for r in results:
            q_type = r["question_type"]
            type_stats.setdefault(q_type, []).append(r["is_correct"])
            latencies.append(r["latency_ms"])
            token_counts.append(r["token_count"])

        total_correct = sum(r["is_correct"] for r in results)
        overall_accuracy = total_correct / len(results)

        def _type_accuracy(type_key: str) -> float:
            items = type_stats.get(type_key, [])
            if not items:
                return 0.0
            return sum(items) / len(items)

        return {
            "overall_accuracy": round(overall_accuracy, 4),
            "single_hop_accuracy": round(_type_accuracy(LOCOMOQuestionType.SINGLE_HOP.value), 4),
            "multi_hop_accuracy": round(_type_accuracy(LOCOMOQuestionType.MULTI_HOP.value), 4),
            "temporal_accuracy": round(_type_accuracy(LOCOMOQuestionType.TEMPORAL.value), 4),
            "open_domain_accuracy": round(_type_accuracy(LOCOMOQuestionType.OPEN_DOMAIN.value), 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
            "avg_token_count": round(sum(token_counts) / len(token_counts), 1),
        }
