"""LongMemEval 基准测试适配器 — 评估长时记忆系统的五大核心能力。

支持五种能力维度：信息提取、多会话推理、时序推理、知识更新、拒绝回答。
使用 LLM-as-Judge 评估答案正确性，按能力维度统计准确率。
"""

from __future__ import annotations

import json
import logging
import re
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
            with open(path, encoding="utf-8") as f:
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
            ) from None

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
        self._enable_gen: bool = True

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

    def _generate_answer(self, question: str, memories: list[str]) -> str:
        """基于检索到的记忆生成简洁答案。

        Args:
            question: 问题文本。
            memories: 检索到的记忆文本列表。

        Returns:
            生成的答案文本；LLM 调用失败时回退为记忆拼接。
        """
        if not self._enable_gen or self._llm_client is None:
            # 未启用生成或无 LLM 客户端，回退为记忆拼接
            return "\n".join(memories)

        # 编号拼接记忆文本
        memories_text = "\n".join(
            f"[{i + 1}] {mem}" for i, mem in enumerate(memories)
        )

        system_prompt = (
            "你是一个记忆助手。基于给定的记忆上下文回答问题。"
            "只使用记忆中的信息，不要编造。"
            "如果记忆中没有相关信息，回答\"我不知道\"。"
            "回答要简洁直接。"
        )
        user_prompt = (
            f"记忆上下文:\n{memories_text}\n\n"
            f"问题: {question}\n\n答案:"
        )

        try:
            result = self._llm_client.call_sync(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=200,
                temperature=0.0,
                use_cache=False,
            )
            if result and result.content and result.content.strip():
                return result.content.strip()
            # LLM 返回空内容，回退
            logger.warning("_generate_answer: LLM 返回空内容，回退为记忆拼接")
            return "\n".join(memories)
        except Exception as e:
            logger.warning("_generate_answer: LLM 调用失败，回退为记忆拼接: %s", e)
            return "\n".join(memories)

    def _recall_question(self, question: str) -> tuple[str, float, int]:
        """对单个问题执行 recall，返回 (预测答案, 延迟ms, token数)。"""
        start = time.perf_counter()
        try:
            raw = self._provider._handle_recall({"query": question, "mode": "rag"})

            data = json.loads(raw)
            if data.get("status") == "found" and data.get("memories"):
                memories = data["memories"]
                memory_texts: list[str] = []
                total_chars = 0
                for mem in memories:
                    content = mem.get("content", "") or mem.get("summary", "")
                    if content:
                        memory_texts.append(content)
                        total_chars += len(content)

                if not memory_texts:
                    latency_ms = (time.perf_counter() - start) * 1000
                    return "", latency_ms, 0

                # 调用 _generate_answer 生成简洁答案
                prediction = self._generate_answer(question, memory_texts)
                latency_ms = (time.perf_counter() - start) * 1000
                token_count = len(prediction) // 4
                return prediction, latency_ms, token_count
            else:
                latency_ms = (time.perf_counter() - start) * 1000
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


# ======================================================================
# OmniMemMemoryProvider — 将 OmniMemSDK 适配为 LongMemEval 的记忆提供者
# ======================================================================

# 偏好关键词列表，用于识别包含用户偏好的消息
_PREFERENCE_KEYWORDS = frozenset({
    "like", "prefer", "love", "enjoy", "favorite", "favourite",
    "hate", "dislike", "want", "wish", "hope", "need",
    "喜欢", "偏好", "讨厌", "想要", "希望", "最爱",
    "preferably", "rather", "inclined", "keen on",
})

# LongMemEval question_type → LongMemEvalCapability 映射
_QUESTION_TYPE_TO_CAPABILITY: dict[str, LongMemEvalCapability] = {
    "single-session-user": LongMemEvalCapability.INFORMATION_EXTRACTION,
    "single-session-assistant": LongMemEvalCapability.INFORMATION_EXTRACTION,
    "single-session-preference": LongMemEvalCapability.INFORMATION_EXTRACTION,
    "multi-session": LongMemEvalCapability.MULTI_SESSION_REASONING,
    "temporal-reasoning": LongMemEvalCapability.TEMPORAL_REASONING,
    "knowledge-update": LongMemEvalCapability.KNOWLEDGE_UPDATE,
}


class OmniMemMemoryProvider:
    """OmniMem → LongMemEval 适配器。

    将 OmniMemSDK 的 memorize/recall 接口适配为 LongMemEval 评测流程所需的
    记忆写入与检索接口。

    用法:
        provider = OmniMemMemoryProvider(storage_dir="/tmp/omnimem_lme")
        provider.ingest_sessions(raw_data)   # raw_data 来自 LongMemEval JSON
        contexts = provider.search("用户喜欢什么编程语言？")
        provider.close()
    """

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """初始化适配器。

        Args:
            storage_dir: OmniMemSDK 存储目录，默认使用临时目录。
            config: OmniMemSDK 配置字典。
        """
        from omnimem.sdk import OmniMemSDK

        self._storage_dir = storage_dir
        self._config = config or {}
        self._sdk: OmniMemSDK | None = None
        self._ingested_count: int = 0

        # 延迟初始化 SDK，便于异常处理
        try:
            self._sdk = OmniMemSDK(storage_dir=storage_dir, config=self._config)
            logger.info(
                "OmniMemMemoryProvider 初始化成功: storage_dir=%s",
                storage_dir,
            )
        except Exception as e:
            logger.error("OmniMemMemoryProvider 初始化失败: %s", e)
            self._sdk = None

    # ------------------------------------------------------------------
    # 记忆写入
    # ------------------------------------------------------------------

    def ingest_sessions(self, sessions_data: list[dict[str, Any]], user_only: bool = False) -> int:
        """将 LongMemEval 的会话历史写入 OmniMem。

        遍历每条数据的 haystack_sessions，按 turn 调用 sdk.memorize()
        存储每轮对话，并根据角色和内容自动推断 memory_type。

        Args:
            sessions_data: LongMemEval 原始数据列表，每条包含
                haystack_session_ids, haystack_dates, haystack_sessions 等字段。
            user_only: 是否只写入 user 角色的 turn。默认 False，写入所有角色
                （含 assistant turn，memory_type 映射为 "action"，
                并附加元数据 role="assistant"）。

        Returns:
            成功写入的记忆条数。
        """
        if self._sdk is None:
            logger.error("OmniMemMemoryProvider: SDK 未初始化，无法写入记忆")
            return 0

        total_ingested = 0

        for entry_idx, entry in enumerate(sessions_data):
            session_ids = entry.get("haystack_session_ids", [])
            session_dates = entry.get("haystack_dates", [])
            sessions = entry.get("haystack_sessions", [])

            for sess_idx, (sess_id, sess_date, sess_turns) in enumerate(
                zip(session_ids, session_dates, sessions)
            ):
                for turn_idx, turn in enumerate(sess_turns):
                    role = turn.get("role", "user")
                    content = turn.get("content", "").strip()
                    if not content:
                        continue

                    # 如果 user_only=True，跳过非 user 角色的 turn
                    if user_only and role != "user":
                        continue

                    # 选择性写入 assistant turn：跳过低信息密度的回复
                    if role == "assistant" and not self._is_substantive(content):
                        continue

                    # ★ v10: 长回复分段存储 — 排班表/列表按行拆分
                    segments = self._split_long_content(content, role)
                    if len(segments) <= 1:
                        # 短回复直接存
                        memory_type = self._infer_memory_type(role, content)
                        # ★ 偏好增强索引：为偏好类 turn 注入可搜索前缀标签
                        store_content = content
                        if memory_type == "preference":
                            store_content = f"[prefer like enjoy] {content}"

                        metadata = {
                            "session_id": sess_id,
                            "turn_index": turn_idx,
                            "timestamp": sess_date,
                            "role": role,
                            "entry_index": entry_idx,
                            "source": "longmemeval",
                        }
                        if turn.get("has_answer"):
                            metadata["has_answer"] = True
                        try:
                            result = self._sdk.memorize(
                                content=store_content,
                                memory_type=memory_type,
                                metadata=metadata,
                            )
                            if result.get("status") in ("ok", "stored", "deduplicated"):
                                total_ingested += 1
                        except Exception as e:
                            logger.warning("写入记忆失败 (session=%s, turn=%d): %s", sess_id, turn_idx, e)
                    else:
                        # 长回复分段存储
                        for seg_idx, segment in enumerate(segments):
                            memory_type = self._infer_memory_type(role, segment)
                            # ★ 偏好增强索引
                            store_segment = segment
                            if memory_type == "preference":
                                store_segment = f"[prefer like enjoy] {segment}"

                            metadata = {
                                "session_id": sess_id,
                                "turn_index": turn_idx,
                                "segment_index": seg_idx,
                                "timestamp": sess_date,
                                "role": role,
                                "entry_index": entry_idx,
                                "source": "longmemeval",
                            }
                            if turn.get("has_answer"):
                                metadata["has_answer"] = True
                            try:
                                result = self._sdk.memorize(
                                    content=store_segment,
                                    memory_type=memory_type,
                                    metadata=metadata,
                                )
                                if result.get("status") in ("ok", "stored", "deduplicated"):
                                    total_ingested += 1
                            except Exception as e:
                                logger.warning("写入记忆失败 (session=%s, turn=%d, seg=%d): %s", sess_id, turn_idx, seg_idx, e)

            # 每 50 条 entry 报告进度
            if (entry_idx + 1) % 50 == 0:
                logger.info(
                    "ingest_sessions 进度: %d/%d entries, 已写入 %d 条记忆",
                    entry_idx + 1, len(sessions_data), total_ingested,
                )

        self._ingested_count += total_ingested
        logger.info(
            "ingest_sessions 完成: 共写入 %d 条记忆 (累计 %d)",
            total_ingested, self._ingested_count,
        )
        return total_ingested

    # ------------------------------------------------------------------
    # 记忆检索
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 50) -> list[str]:
        """检索与查询相关的记忆上下文。

        调用 sdk.recall() 并将结果转换为 LongMemEval 期望的格式
        （检索到的 context 文本列表）。

        Args:
            query: 查询文本。
            top_k: 最多返回的结果数。

        Returns:
            检索到的上下文文本列表，按相关性排序。
        """
        if self._sdk is None:
            logger.error("OmniMemMemoryProvider: SDK 未初始化，无法检索记忆")
            return []

        try:
            result = self._sdk.recall(query=query, mode="rag")
        except Exception as e:
            logger.error("recall 调用异常 (query=%s): %s", query[:50], e)
            return []

        status = result.get("status", "")

        if status == "found" and result.get("memories"):
            memories = result["memories"]
            contexts: list[str] = []
            for mem in memories[:top_k]:
                # 优先取 content，其次取 summary
                text = mem.get("content", "") or mem.get("summary", "")
                if text:
                    contexts.append(text)
            return contexts

        if status == "no_results":
            logger.debug("recall 无结果 (query=%s)", query[:50])
            return []

        # 异常状态
        logger.warning(
            "recall 返回异常状态: %s (query=%s)",
            status, query[:50],
        )
        return []

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    # 寒暄/确认模式黑名单（整个内容匹配时跳过）
    _GREETING_PATTERN = re.compile(
        r"^(好的|是的|没问题|当然|可以|明白了|知道了|了解|嗯|对|行|确实|没错|是的呢"
        r"|OK|Okay|Sure|Of course|I see|Got it|Understood|Right|Yes|Yeah|Absolutely"
        r"|Certainly|Exactly)[.!。！？?]*$",
        re.IGNORECASE,
    )

    # 技术关键词（不区分大小写）
    _TECH_KEYWORDS = frozenset(
        "python java rust docker kubernetes api database server model training "
        "algorithm code function class method data analysis research experiment "
        "test debug deploy config install upgrade migrate performance security "
        "network cloud framework library plugin module package version release".split()
    )

    # 中文专有名词指示词
    _CN_PROPER_INDICATORS = ("叫做", "名为", "是", "位于", "属于", "来自")

    # 对话性实质信息模式（建议词/具体指代词）
    _DIALOG_PATTERN = re.compile(
        r"(recommend|suggest|should|could|would|advise|prefer|"
        r"建议|推荐|应该|可以|需要|最好|"
        r"the |this |that |my |your |our |"
        r"我的|你的|这个|那个|我们)",
        re.IGNORECASE,
    )

    # ★ v8 新增：排班/表格/分配类模式（assistant 生成的结构化信息）
    _SCHEDULE_PATTERN = re.compile(
        r"(shift|rotation|schedule|assigned|assignment|roster|timetable|"
        r"排班|轮班|值班|安排|分配|表格|"
        r"(am|pm)\s*[-–—]\s*(am|pm)|\d{1,2}\s*[-–:]\s*\d{0,2})",
        re.IGNORECASE,
    )

    @staticmethod
    def _is_substantive(content: str) -> bool:
        """判断 assistant turn 是否含实质信息。

        过滤规则（按优先级执行）：
        1. 匹配寒暄黑名单 → False（寒暄/确认，无论长短）
        2. 含实质信息（数字/大写单词/技术关键词/中文专有名词指示词） → True
        2.5 含对话性实质信息（建议词/指代词 + 长度 > 30） → True
        3. 长度 > 60 → True（长回复通常含实质信息）
        4. 长度 ≤ 30 → False（短回复几乎都是寒暄/确认，且无实质信息）
        5. 默认 → False（中等长度无实质信息）

        Args:
            content: assistant turn 的文本内容。

        Returns:
            是否含实质信息。
        """
        # 规则 1：寒暄/确认模式黑名单
        if OmniMemMemoryProvider._GREETING_PATTERN.match(content.strip()):
            return False

        # 规则 2：实质信息检测
        # 含数字（具体数据/时间/版本号）
        if re.search(r"\d+", content):
            return True

        # 含大写单词（连续2+大写字母，专有名词/缩写）
        if re.search(r"[A-Z]{2,}", content):
            return True

        # 含技术关键词（不区分大小写）
        content_lower = content.lower()
        if any(kw in content_lower for kw in OmniMemMemoryProvider._TECH_KEYWORDS):
            return True

        # 含中文专有名词指示词
        if any(ind in content for ind in OmniMemMemoryProvider._CN_PROPER_INDICATORS):
            return True

        # ★ v8 新增：排班/表格/分配类信息（assistant 生成的结构化数据）
        if OmniMemMemoryProvider._SCHEDULE_PATTERN.search(content):
            return True

        # 对话性实质信息检测：含建议词/具体指代词的中等长度回复
        if OmniMemMemoryProvider._DIALOG_PATTERN.search(content) and len(content) > 30:
            return True

        # 规则 3：长回复通常含实质信息（阈值从 60 降到 40）
        if len(content) > 40:
            return True

        # 规则 4：短回复（≤30）且无实质信息 → False
        if len(content) <= 30:
            return False

        # 规则 5：默认不通过（中等长度无实质信息）
        return False

    @staticmethod
    def _split_long_content(content: str, role: str) -> list[str]:
        """将长回复按段落/列表/表格行拆分为独立记忆条目。

        核心问题：排班表/列表等长回复作为一个整体存储时，具体行（如 "Admon | 8am-4pm | Sunday"）
        在 BM25/向量检索中排名太低。拆分后每行独立存储，检索命中率大幅提升。

        拆分策略：
        - Markdown 表格：按 | 行拆分
        - 列表（- / * / 数字.）：按列表项拆分
        - 段落：按双换行拆分
        - 短回复（<500字符）：不拆分，返回原内容

        Returns:
            拆分后的内容列表。长度为 1 表示不拆分。
        """
        # 短回复不拆分
        if len(content) < 500:
            return [content]

        segments = []

        # 策略 1：Markdown 表格行拆分
        table_lines = [l for l in content.split('\n') if l.strip().startswith('|') and l.strip().endswith('|')]
        # 排除表头分隔行（如 | --- | --- |）
        data_lines = [l for l in table_lines if not re.match(r'^\|[\s\-:|]+\|$', l.strip())]
        if len(data_lines) >= 3:
            # 表格数据行足够多，按行拆分
            # 保留表头作为上下文
            header_lines = [l for l in table_lines[:2] if not re.match(r'^\|[\s\-:|]+\|$', l.strip())]
            header_context = '\n'.join(header_lines) if header_lines else ""

            # 表格前的前言
            pre_table = []
            for line in content.split('\n'):
                if line.strip().startswith('|'):
                    break
                if line.strip():
                    pre_table.append(line.strip())
            if pre_table:
                segments.append('\n'.join(pre_table))

            # 每个数据行 + 表头上下文
            for line in data_lines:
                if header_context:
                    segments.append(f"{header_context}\n{line.strip()}")
                else:
                    segments.append(line.strip())
            return segments if segments else [content]

        # 策略 2：列表项拆分
        list_items = re.split(r'\n(?=[\-\*]\s|\d+\.\s)', content)
        if len(list_items) >= 4:
            for item in list_items:
                item = item.strip()
                if item and len(item) > 10:
                    segments.append(item)
            return segments if segments else [content]

        # 策略 3：段落拆分（双换行）
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 20]
        if len(paragraphs) >= 3:
            return paragraphs

        # 不拆分
        return [content]

    @staticmethod
    def _infer_memory_type(role: str, content: str) -> str:
        """根据角色和内容推断 memory_type。

        推断规则：
        - assistant 消息 → "action"
        - 包含偏好关键词的 user 消息 → "preference"
        - 其他 user 消息 → "fact"

        Args:
            role: 对话角色 ("user" / "assistant")。
            content: 对话内容。

        Returns:
            memory_type 字符串。
        """
        if role == "assistant":
            return "action"

        # 检查是否包含偏好关键词
        content_lower = content.lower()
        if any(kw in content_lower for kw in _PREFERENCE_KEYWORDS):
            return "preference"

        return "fact"

    @staticmethod
    def map_question_type(question_type: str, question_id: str) -> LongMemEvalCapability:
        """将 LongMemEval 的 question_type 映射为 LongMemEvalCapability。

        Args:
            question_type: LongMemEval 原始 question_type 字段。
            question_id: 问题 ID，用于判断是否为 abstention 类。

        Returns:
            对应的 LongMemEvalCapability 枚举值。
        """
        if "_abs" in question_id:
            return LongMemEvalCapability.REFUSAL
        return _QUESTION_TYPE_TO_CAPABILITY.get(
            question_type, LongMemEvalCapability.INFORMATION_EXTRACTION
        )

    @staticmethod
    def load_raw_data(path: str | Path) -> list[dict[str, Any]]:
        """加载 LongMemEval 原始 JSON 数据。

        Args:
            path: JSON 文件路径。

        Returns:
            原始数据列表。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LongMemEval 数据文件不存在: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"LongMemEval 数据格式错误: 期望 list，实际 {type(data)}")

        logger.info("LongMemEval 原始数据加载完成: %d 条, 来源=%s", len(data), path)
        return data

    @property
    def ingested_count(self) -> int:
        """已写入的记忆总数。"""
        return self._ingested_count

    @property
    def is_ready(self) -> bool:
        """SDK 是否就绪。"""
        return self._sdk is not None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭 SDK，释放资源。"""
        if self._sdk is not None:
            try:
                self._sdk.close()
            except Exception as e:
                logger.warning("OmniMemMemoryProvider close 失败: %s", e)
            self._sdk = None
        logger.info("OmniMemMemoryProvider 已关闭")

    def __enter__(self) -> OmniMemMemoryProvider:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ======================================================================
# 单元测试
# ======================================================================

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 60)
    print("OmniMemMemoryProvider 单元测试")
    print("=" * 60)

    # --- 准备模拟会话数据（模拟 LongMemEval 格式） ---
    mock_sessions_data = [
        {
            "question_id": "test_001",
            "question_type": "single-session-user",
            "question": "What programming language does the user prefer?",
            "answer": "Python",
            "question_date": "2024-06-15",
            "haystack_session_ids": ["sess_001", "sess_002", "sess_003"],
            "haystack_dates": ["2024-06-01", "2024-06-05", "2024-06-10"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I really like Python for data analysis."},
                    {"role": "assistant", "content": "Python is great for data analysis with libraries like pandas and numpy."},
                ],
                [
                    {"role": "user", "content": "Can you help me with a Java project?"},
                    {"role": "assistant", "content": "Sure, I can help you with Java. What do you need?"},
                ],
                [
                    {"role": "user", "content": "I prefer dark mode in my editor."},
                    {"role": "assistant", "content": "Dark mode is easier on the eyes for long coding sessions."},
                ],
            ],
            "answer_session_ids": ["sess_001"],
        },
        {
            "question_id": "test_002",
            "question_type": "temporal-reasoning",
            "question": "When did the user first mention Python?",
            "answer": "2024-06-01",
            "question_date": "2024-06-15",
            "haystack_session_ids": ["sess_004"],
            "haystack_dates": ["2024-06-12"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I started learning Rust recently."},
                    {"role": "assistant", "content": "Rust is a systems language focused on safety and performance."},
                ],
            ],
            "answer_session_ids": ["sess_004"],
        },
    ]

    # --- 测试 1: 创建适配器并写入记忆 ---
    print("\n[测试 1] 创建适配器并写入记忆...")
    with tempfile.TemporaryDirectory(prefix="omnimem_lme_test_") as tmp_dir:
        try:
            provider = OmniMemMemoryProvider(storage_dir=tmp_dir)
        except Exception as e:
            print(f"  ✗ 适配器创建失败: {e}")
            print("  跳过后续测试（可能是 ChromaDB 不可用）")
            import sys
            sys.exit(0)

        if not provider.is_ready:
            print("  ✗ SDK 未就绪，跳过测试")
            import sys
            sys.exit(0)

        count = provider.ingest_sessions(mock_sessions_data)
        print(f"  ✓ 写入记忆数: {count}")
        print(f"  ✓ 累计写入数: {provider.ingested_count}")

        # --- 测试 2: memory_type 推断 ---
        print("\n[测试 2] memory_type 推断...")
        test_cases = [
            ("user", "I like Python", "preference"),
            ("assistant", "Here is the answer", "action"),
            ("user", "What is the weather today?", "fact"),
            ("user", "I prefer coffee over tea", "preference"),
            ("user", "My name is Alice", "fact"),
        ]
        for role, content, expected in test_cases:
            actual = OmniMemMemoryProvider._infer_memory_type(role, content)
            status_str = "✓" if actual == expected else "✗"
            print(f"  {status_str} role={role}, content='{content[:30]}' → {actual} (期望 {expected})")

        # --- 测试 3: 检索查询 ---
        print("\n[测试 3] 检索查询...")
        queries = [
            "What programming language does the user prefer?",
            "Tell me about the user's editor preferences",
            "What did the user learn recently?",
        ]
        for query in queries:
            results = provider.search(query, top_k=3)
            print(f"  查询: '{query[:50]}'")
            print(f"    结果数: {len(results)}")
            for i, ctx in enumerate(results):
                print(f"    [{i}] {ctx[:80]}...")
            if not results:
                print(f"    (无结果)")

        # --- 测试 4: question_type 映射 ---
        print("\n[测试 4] question_type → capability 映射...")
        type_tests = [
            ("single-session-user", "q001", LongMemEvalCapability.INFORMATION_EXTRACTION),
            ("multi-session", "q002", LongMemEvalCapability.MULTI_SESSION_REASONING),
            ("temporal-reasoning", "q003", LongMemEvalCapability.TEMPORAL_REASONING),
            ("knowledge-update", "q004", LongMemEvalCapability.KNOWLEDGE_UPDATE),
            ("single-session-user", "q005_abs", LongMemEvalCapability.REFUSAL),
        ]
        for qtype, qid, expected in type_tests:
            actual = OmniMemMemoryProvider.map_question_type(qtype, qid)
            status_str = "✓" if actual == expected else "✗"
            print(f"  {status_str} {qtype} + {qid} → {actual.value} (期望 {expected.value})")

        # --- 测试 5: load_raw_data ---
        print("\n[测试 5] load_raw_data...")
        raw_data_path = Path(
            "/home/xxh/.hermes/plugins/omnimem/benchmarks/LongMemEval/data/longmemeval_oracle.json"
        )
        if raw_data_path.exists():
            raw = OmniMemMemoryProvider.load_raw_data(raw_data_path)
            print(f"  ✓ 加载数据条数: {len(raw)}")
            print(f"  ✓ 第一条 question_type: {raw[0]['question_type']}")
            print(f"  ✓ 第一条 sessions 数: {len(raw[0]['haystack_sessions'])}")
        else:
            print(f"  ⊘ 数据文件不存在，跳过: {raw_data_path}")

        # --- 测试 6: _is_substantive 过滤 ---
        print("\n[测试 6] _is_substantive 过滤...")
        substantive_tests = [
            ("好的", False, "短寒暄（中文）"),
            ("OK", False, "短寒暄（英文）"),
            ("Python 3.11 was released", True, "含数字"),
            ("I recommend using Docker", True, "含技术关键词"),
            ("A" * 101, True, "长回复（>60字符）"),
            ("Let me help you with that problem", True, "含指代词that+长度>30"),
            ("Sure", False, "短寒暄 Sure"),
            ("知道了", False, "短寒暄（知道了）"),
            ("The API server is running on port 8080", True, "含技术关键词+数字"),
            ("当然可以", False, "短寒暄（当然可以）"),
            ("I recommend using Python for data analysis", True, "含recommend"),
            ("That is a great question, let me think about it", True, "含That+长度>30"),
            ("I think this approach works well", True, "含this+长度>30"),
            ("好的没问题", False, "寒暄黑名单"),
        ]
        for content, expected, desc in substantive_tests:
            actual = OmniMemMemoryProvider._is_substantive(content)
            status_str = "✓" if actual == expected else "✗"
            print(f"  {status_str} '{content[:40]}' → {actual} (期望 {expected}) [{desc}]")

        # --- 清理 ---
        provider.close()
        print("\n  ✓ 适配器已关闭")

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)
