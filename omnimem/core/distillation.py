"""
DistillationEngine — OPT-1: LLM 驱动的记忆蒸馏。

定期将 auto-captured raw facts 喂给 LLM 进行摘要、提炼、去重合并，
产出高质量的精炼记忆，替代机械规则的 _extract_core_fact()。

架构：
  - 实时链路保持 Perception Engine 规则提取（低延迟）
  - 定期后台运行 LLM 蒸馏（高质量）
  - 支持自定义蒸馏模型（distill_model 配置）

与 ReflectEngine 的区别：
  - Reflect: 对已有记忆做推理反思，产出心智模型
  - Distill: 对原始对话片段做压缩提炼，产出精炼事实
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# FTS5 特殊字符转义
_FTS5_SPECIAL = set("+-*()\"'^~&|!@: ")


def _escape_fts5_query(text: str) -> str:
    """转义 FTS5 特殊字符以用于 MATCH 查询。"""
    return "".join(f"\"{c}\"" if c in _FTS5_SPECIAL else c for c in text)


class DistillationEngine:
    """LLM 蒸馏引擎：对 auto-captured raw facts 进行语义提炼。

    使用方法：
      engine = DistillationEngine(llm_fn=..., store=..., config=...)
      engine.distill_recent_facts()  # 返回蒸馏结果
    """

    _DEFAULT_SYSTEM_PROMPT = (
        "你是一个记忆提炼引擎。你的任务是将对话中捕获的原始片段压缩为简洁、"
        "准确、无冗余的事实陈述。\n\n"
        "提炼要求：\n"
        "1. 每个事实一句话即可，去掉语气词、套话、冗余修饰\n"
        "2. 合并重复/同类信息，不保留多个版本\n"
        "3. 保留关键参数和决策，丢弃过程性闲聊\n"
        "4. 使用中文输出\n"
        "5. 只输出精炼后的事实列表，每行一个事实，不要编号\n"
        "6. 如果输入内容中没有任何值得长期记忆的信息，输出空"
    )

    def __init__(
        self,
        llm_fn: Callable[..., str | None] | None = None,
        store: Any = None,
        memorize_fn: Callable[[dict[str, Any]], str] | None = None,
        config: Any | None = None,
    ):
        """初始化蒸馏引擎。

        Args:
            llm_fn: LLM 调用函数，签名 (prompt, system, max_tokens, model=None) -> str
            store: DrawerClosetStore 实例，用于读写记忆
            memorize_fn: 存储蒸馏结果的回调
            config: OmniMemConfig 实例
        """
        self._llm_fn = llm_fn
        self._store = store
        self._memorize_fn = memorize_fn
        self._config = config
        self._last_distill_turn = 0
        self._last_distill_time = 0.0
        self._distill_count = 0
        self._min_interval_seconds = 30  # 两次蒸馏之间最小间隔（避免频繁调用 LLM）

    def distill_recent_facts(
        self,
        turn_count: int = 0,
        wing_filter: str = "auto",
        max_facts: int = 15,
        min_facts: int = 3,
    ) -> dict[str, Any]:
        """蒸馏最近 auto-captured 的事实。

        Args:
            turn_count: 当前轮次（用于频率控制）
            wing_filter: 要蒸馏的 wing（"auto" 表示 auto_checkpoint 条目）
            max_facts: 最多取多少条原始事实
            min_facts: 最少需要多少条才触发蒸馏

        Returns:
            {"status": "distilled"|"skipped"|"no_facts"|"llm_unavailable",
             "input_count": int, "distilled_count": int, "facts": [...]}
        """
        if not self._llm_fn:
            logger.warning("DistillationEngine: no LLM function configured, distillation unavailable")
            return {"status": "llm_unavailable", "input_count": 0, "distilled_count": 0, "facts": []}

        # 频率控制
        now = time.time()
        if now - self._last_distill_time < self._min_interval_seconds:
            return {"status": "skipped", "reason": "too_soon",
                    "next_available_in": round(self._min_interval_seconds - (now - self._last_distill_time), 1)}

        if turn_count and self._last_distill_turn >= turn_count:
            return {"status": "skipped", "reason": "already_distilled_this_turn"}

        # 获取最近的 auto-captured facts
        raw_facts = self._get_recent_auto_facts(wing_filter, max_facts)
        logger.info("Distillation started: %d raw facts", len(raw_facts))
        if len(raw_facts) < min_facts:
            return {"status": "no_facts", "input_count": len(raw_facts),
                    "distilled_count": 0, "facts": []}

        # 构建蒸馏 prompt
        facts_text = "\n".join(
            f"- {f.get('content', f.get('summary', ''))[:300]}"
            for f in raw_facts
        )

        distill_model = self._get_distill_model()
        prompt = (
            f"以下是对话过程中自动捕获的原始记忆片段，请提炼为简洁的事实陈述：\n\n"
            f"{facts_text}\n\n"
            f"请输出精炼后的事实列表（每行一个事实，不要编号，丢弃不值得长期记忆的内容）。"
        )

        try:
            raw_response = self._llm_fn(
                prompt=prompt,
                system=self._DEFAULT_SYSTEM_PROMPT,
                max_tokens=600,
                model=distill_model,
            )
        except Exception as e:
            logger.warning("DistillationEngine LLM call failed: %s", e)
            return {"status": "llm_unavailable", "input_count": len(raw_facts),
                    "distilled_count": 0, "facts": []}

        if not raw_response or not raw_response.strip():
            return {"status": "distilled", "input_count": len(raw_facts),
                    "distilled_count": 0, "facts": []}

        # 解析蒸馏结果
        distilled = self._parse_distilled_facts(raw_response.strip())

        # 存储蒸馏结果
        stored_count = 0
        if distilled and self._memorize_fn:
            for fact in distilled:
                try:
                    result = self._memorize_fn({
                        "content": fact,
                        "memory_type": "fact",
                        "confidence": 3,
                        "scope": "personal",
                        "privacy": "personal",
                    })
                    if result and "error" not in str(result).lower():
                        stored_count += 1
                except Exception as e:
                    logger.warning("DistillationEngine store failed for '%s': %s", fact[:50], e)

        self._last_distill_time = now
        self._last_distill_turn = turn_count
        self._distill_count += 1

        logger.info("Distillation completed: %d distilled facts", stored_count)

        return {
            "status": "distilled",
            "input_count": len(raw_facts),
            "distilled_count": stored_count,
            "facts": distilled[:20],
        }

    def _get_distill_model(self) -> str | None:
        """获取蒸馏专用模型名。空字符串=使用主模型。"""
        if self._config:
            model = self._config.get("distill_model", "")
            return model if model else None  # None 表示使用默认
        return None

    def _get_recent_auto_facts(
        self, wing_filter: str, limit: int
    ) -> list[dict[str, Any]]:
        """从 Store 获取最近自动捕获的事实。"""
        if not self._store:
            return []

        facts: list[dict[str, Any]] = []

        # 方法1：通过 MetaStore 按 wing 查询
        meta = getattr(self._store, "_meta_store", None)
        if meta:
            try:
                results = meta.search(wing=wing_filter, limit=limit)
                facts.extend(results)
            except Exception as e:
                logger.warning("DistillationEngine meta_store query failed: %s", e)

        # 方法2：如果 meta_store 不可用，用 search_by_content 兜底
        if not facts:
            try:
                # 搜索 auto checkpoint 的标记
                facts = self._store.search_by_content("[Turn ", limit=limit)
                # 补充紧急保存的条目
                emergency = self._store.search_by_content("[Emergency save", limit=limit)
                seen_ids = {f.get("memory_id") for f in facts}
                for e in emergency:
                    if e.get("memory_id") not in seen_ids:
                        facts.append(e)
                        seen_ids.add(e.get("memory_id"))
                        if len(facts) >= limit:
                            break
            except Exception as e:
                logger.warning("DistillationEngine search_by_content failed: %s", e)

        # 方法3：从 closet_index 获取
        if not facts:
            try:
                closet = getattr(self._store, "_closet_index", {})
                for mid, entry in closet.items():
                    wing = entry.get("wing", "")
                    if wing == wing_filter:
                        facts.append(dict(entry))
                        if len(facts) >= limit:
                            break
            except Exception as e:
                logger.warning("DistillationEngine closet_index scan failed: %s", e)

        return facts

    @staticmethod
    def _parse_distilled_facts(raw: str) -> list[str]:
        """解析 LLM 蒸馏输出为事实列表。"""
        facts: list[str] = []

        # 先尝试 JSON 数组
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, str) and len(item.strip()) > 5:
                            facts.append(item.strip())
                        elif isinstance(item, dict):
                            content = item.get("content", item.get("fact", ""))
                            if content and len(str(content).strip()) > 5:
                                facts.append(str(content).strip())
                    return facts
            except (json.JSONDecodeError, ValueError):
                pass

        # 解析为行列表
        for line in raw.split("\n"):
            line = line.strip()
            # 去除编号（如 "1.", "1)", "一、", "- " 等）
            line = re.sub(r"^(?:\d+[.、)）]\s*|[一-龥]、|\-\s+|•\s+|·\s+)", "", line).strip()
            if len(line) > 5 and not line.startswith(("#", "//", "```")):
                facts.append(line)

        return facts

    def get_stats(self) -> dict[str, Any]:
        """获取蒸馏统计。"""
        return {
            "total_distillations": self._distill_count,
            "last_distill_turn": self._last_distill_turn,
            "last_distill_time": self._last_distill_time,
        }

    def close(self) -> None:
        pass
