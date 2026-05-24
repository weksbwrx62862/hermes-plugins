"""LongMemEval 基准测试适配器 — 评估长时记忆系统的五大核心能力。

支持五种能力维度：信息提取、多会话推理、时序推理、知识更新、拒绝回答。
使用 LLM-as-Judge 评估答案正确性，按能力维度统计准确率。
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


class LongMemEvalCapability(str, Enum):
    """LongMemEval 能力维度枚举。"""
    INFORMATION_EXTRACTION = "information_extraction"
    MULTI_SESSION_REASONING = "multi_session_reasoning"
    TEMPORAL_REASONING = "temporal_reasoning"
    KNOWLEDGE_UPDATE = "knowledge_update"
    REFUSAL = "refusal"


@dataclass
class LongMemEvalQuestion:
    """LongMemEval 单条问题数据。"""
    question_id: str
    question: str
    answer: str
    capability: LongMemEvalCapability
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class LongMemEvalDataset:
    """LongMemEval 数据集容器。"""
    questions: list[LongMemEvalQuestion] = field(default_factory=list)
    name: str = "longmemeval"

    @classmethod
    def load_from_json(cls, path: str | Path) -> LongMemEvalDataset:
        """从 JSON 文件加载 LongMemEval 数据集。

        JSON 格式要求：
        {
            "name": "longmemeval_v1",
            "questions": [
                {
                    "question_id": "q001",
                    "question": "...",
                    "answer": "...",
                    "capability": "information_extraction",
                    "evidence_ids": ["mem-001"]
                }
            ]
        }
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LongMemEval 数据集文件不存在: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"LongMemEval 数据集 JSON 解析失败: {e}") from e

        questions = []
        for item in data.get("questions", []):
            try:
                cap = LongMemEvalCapability(item["capability"])
            except ValueError:
                logger.warning(
                    "跳过未知能力维度 '%s' (question_id=%s)",
                    item.get("capability"), item.get("question_id"),
                )
                continue

            questions.append(LongMemEvalQuestion(
                question_id=item["question_id"],
                question=item["question"],
                answer=item["answer"],
                capability=cap,
                evidence_ids=item.get("evidence_ids", []),
            ))

        if not questions:
            raise ValueError("LongMemEval 数据集为空或所有问题的能力维度均无效")

        logger.info("LongMemEval 数据集加载完成: %d 道题目, 来源=%s", len(questions), path)
        return cls(questions=questions, name=data.get("name", "longmemeval"))

    @classmethod
    def load_from_hf(cls, split: str = "test") -> LongMemEvalDataset:
        """从 HuggingFace 加载 LongMemEval 数据集（可选）。

        需要安装 datasets 库: pip install datasets
        数据集: xlangai/LongMemEval
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "加载 HuggingFace 数据集需要 datasets 库，"
                "请安装: pip install datasets"
            )

        try:
            ds = load_dataset("xlangai/LongMemEval", split=split)
        except Exception as e:
            raise RuntimeError(f"从 HuggingFace 加载 LongMemEval 数据集失败: {e}") from e

        questions = []
        for idx, item in enumerate(ds):
            cap_str = item.get("capability", "information_extraction")
            try:
                cap = LongMemEvalCapability(cap_str)
            except ValueError:
                cap = LongMemEvalCapability.INFORMATION_EXTRACTION

            questions.append(LongMemEvalQuestion(
                question_id=item.get("question_id", f"hf-{idx:04d}"),
                question=item["question"],
                answer=item.get("answer", ""),
                capability=cap,
                evidence_ids=item.get("evidence_ids", []),
            ))

        logger.info("LongMemEval 数据集从 HuggingFace 加载完成: %d 道题目, split=%s", len(questions), split)
        return cls(questions=questions, name=f"longmemeval_hf_{split}")


_JUDGE_SYSTEM_PROMPT = """你是一个严格的答案评判专家。你的任务是判断预测答案是否与标准答案语义一致。

评判标准：
- 核心信息一致即为正确，不要求措辞完全相同
- 如果预测答案包含标准答案的核心事实，视为正确
- 如果预测答案遗漏关键信息或包含矛盾信息，视为错误
- 对于拒绝类问题，如果预测答案合理拒绝回答，视为正确
- 只回答 "correct" 或 "incorrect"，不要解释"""

_JUDGE_USER_TEMPLATE = """标准答案: {gold_answer}
预测答案: {prediction}

请判断预测答案是否正确。"""


class LongMemEvalEvaluator:
    """LongMemEval 基准测试评估器。

    对每个问题调用 OmniMemProvider 的 recall 接口获取预测答案，
    使用 LLM-as-Judge 评估正确性，按能力维度统计准确率。
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

        logger.warning("LongMemEvalEvaluator: 无法初始化 Judge LLM 客户端，将使用关键词匹配")
        return None

    def _judge_answer(self, prediction: str, gold_answer: str, capability: LongMemEvalCapability) -> bool:
        """使用 LLM-as-Judge 判断预测答案是否正确。"""
        if not prediction.strip():
            return False

        if capability == LongMemEvalCapability.REFUSAL:
            return self._judge_refusal(prediction, gold_answer)

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

    def _judge_refusal(self, prediction: str, gold_answer: str) -> bool:
        """评估拒绝类问题：预测答案应合理拒绝回答。"""
        refusal_indicators = [
            "不知道", "无法", "不能", "没有", "不确定",
            "don't know", "cannot", "unable", "not sure", "no information",
        ]
        pred_lower = prediction.lower()
        has_refusal = any(ind in pred_lower for ind in refusal_indicators)

        if self._llm_client is None:
            return has_refusal

        prompt = (
            f"标准答案: {gold_answer}\n"
            f"预测答案: {prediction}\n\n"
            f"这是一个拒绝类问题，预测答案是否合理地表示无法回答？"
            f"只回答 \"correct\" 或 \"incorrect\"。"
        )
        try:
            result = self._llm_client.call_sync(
                prompt=prompt,
                system="你是一个评判专家，判断预测答案是否合理地拒绝了回答。",
                max_tokens=10,
                temperature=0.0,
                use_cache=False,
            )
            verdict = result.content.strip().lower() if result and result.content else ""
            return verdict.startswith("correct")
        except Exception as e:
            logger.warning("LLM Judge (refusal) 调用失败，回退到关键词匹配: %s", e)
            return has_refusal

    @staticmethod
    def _keyword_match(prediction: str, gold_answer: str) -> bool:
        """关键词匹配回退方案。"""
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
            logger.error("LongMemEval recall 失败 (query=%s): %s", question[:50], e)
            return "", latency_ms, 0

    def run(self, dataset: LongMemEvalDataset) -> dict[str, Any]:
        """运行 LongMemEval 评估。

        对每个问题执行 recall + Judge 评估，收集详细结果。
        """
        results: list[dict[str, Any]] = []
        total = len(dataset.questions)

        for i, q in enumerate(dataset.questions):
            prediction, latency_ms, token_count = self._recall_question(q.question)
            is_correct = self._judge_answer(prediction, q.answer, q.capability) if prediction else False

            results.append({
                "question_id": q.question_id,
                "capability": q.capability.value,
                "question": q.question,
                "gold_answer": q.answer,
                "prediction": prediction[:500],
                "is_correct": is_correct,
                "latency_ms": round(latency_ms, 3),
                "token_count": token_count,
                "evidence_ids": q.evidence_ids,
            })

            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info("LongMemEval 评估进度: %d/%d", i + 1, total)

        return {
            "benchmark": "longmemeval",
            "dataset_name": dataset.name,
            "total_questions": total,
            "results": results,
            "metrics": self.compute_metrics(results),
        }

    def compute_metrics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """计算汇总指标，按能力维度统计准确率。"""
        if not results:
            return {
                "overall_accuracy": 0.0,
                "information_extraction_accuracy": 0.0,
                "multi_session_reasoning_accuracy": 0.0,
                "temporal_reasoning_accuracy": 0.0,
                "knowledge_update_accuracy": 0.0,
                "refusal_accuracy": 0.0,
                "avg_latency_ms": 0.0,
                "avg_token_count": 0.0,
            }

        cap_stats: dict[str, list[bool]] = {
            c.value: [] for c in LongMemEvalCapability
        }
        latencies: list[float] = []
        token_counts: list[int] = []

        for r in results:
            cap = r["capability"]
            cap_stats.setdefault(cap, []).append(r["is_correct"])
            latencies.append(r["latency_ms"])
            token_counts.append(r["token_count"])

        total_correct = sum(r["is_correct"] for r in results)
        overall_accuracy = total_correct / len(results)

        def _cap_accuracy(cap_key: str) -> float:
            items = cap_stats.get(cap_key, [])
            if not items:
                return 0.0
            return sum(items) / len(items)

        return {
            "overall_accuracy": round(overall_accuracy, 4),
            "information_extraction_accuracy": round(
                _cap_accuracy(LongMemEvalCapability.INFORMATION_EXTRACTION.value), 4
            ),
            "multi_session_reasoning_accuracy": round(
                _cap_accuracy(LongMemEvalCapability.MULTI_SESSION_REASONING.value), 4
            ),
            "temporal_reasoning_accuracy": round(
                _cap_accuracy(LongMemEvalCapability.TEMPORAL_REASONING.value), 4
            ),
            "knowledge_update_accuracy": round(
                _cap_accuracy(LongMemEvalCapability.KNOWLEDGE_UPDATE.value), 4
            ),
            "refusal_accuracy": round(
                _cap_accuracy(LongMemEvalCapability.REFUSAL.value), 4
            ),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
            "avg_token_count": round(sum(token_counts) / len(token_counts), 1),
        }
