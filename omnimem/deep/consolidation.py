"""ConsolidationEngine — L3 深层记忆：Consolidation 管线。

参考 Hindsight 的仿生 Consolidation 设计，实现四阶段自动升华：
  Stage 1: world_facts      — 客观事实（原始写入）
  Stage 2: experience_facts — 主观经验（带上下文标注）
  Stage 3: observations     — 自动整合的观察（跨记忆关联）
  Stage 4: mental_models    — 反思形成的心智模型（抽象规律）

升华触发：
  - 累积 fact 数量达到阈值（默认 10 条）
  - on_session_end 时批量处理
  - omni_reflect 工具触发

持久化：
  - 各阶段记忆存储在 SQLite，保持溯源链
  - 观察和心智模型可被检索引擎召回
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)


# ─── 数据模型 ────────────────────────────────────────────────


@dataclass
class ConsolidationResult:
    """Consolidation 结果。"""

    observation: str = ""
    mental_model: str = ""
    facts_consolidated: int = 0
    observations_generated: int = 0
    models_generated: int = 0


@dataclass
class ConsolidatedItem:
    """一条 Consolidation 产出。"""

    item_id: str = ""
    stage: str = ""  # world_facts / experience_facts / observations / mental_models
    content: str = ""
    source_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "stage": self.stage,
            "content": self.content,
            "source_ids": self.source_ids,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }


# ─── 关键词提取 ──────────────────────────────────────────────


def _extract_keywords(texts: list[str], top_k: int = 10) -> list[str]:
    """从文本列表中提取高频关键词。"""
    from collections import Counter

    # 中文停用词
    zh_stopwords = {
        "然后",
        "因为",
        "所以",
        "但是",
        "不过",
        "而且",
        "或者",
        "虽然",
        "如果",
        "那么",
        "这个",
        "那个",
        "什么",
        "怎么",
        "已经",
        "正在",
        "可以",
        "应该",
        "需要",
        "没有",
        "不是",
    }
    word_count: Counter[str] = Counter()
    for text in texts:
        # 中文分词：2-4字组合，排除停用词
        zh = [w for w in re.findall(r"[\u4e00-\u9fff]{2,4}", text) if w not in zh_stopwords]
        word_count.update(zh)
        # 英文分词
        en = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        # 排除停用词
        stopwords = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
        }
        en = [w for w in en if w not in stopwords]
        word_count.update(en)
    return [w for w, _ in word_count.most_common(top_k)]


def _cluster_by_topic(facts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按主题对事实进行简单聚类（基于关键词重叠）。"""
    clusters: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        content = fact.get("content", "")
        # 提取主题关键词
        keywords = _extract_keywords([content], top_k=3)
        topic = keywords[0] if keywords else "general"
        if topic not in clusters:
            clusters[topic] = []
        clusters[topic].append(fact)
    return clusters


def _generate_observation(facts: list[dict[str, Any]]) -> str:
    """从一组相关事实生成观察。"""
    if not facts:
        return ""
    contents = [f.get("content", "") for f in facts]
    keywords = _extract_keywords(contents, top_k=5)
    parts = [f"关于{keywords[0]}，观察到：" if keywords else "观察到："]
    for c in contents[:5]:
        parts.append(f"  - {c[:150]}")
    if len(contents) > 5:
        parts.append(f"  ... 共 {len(contents)} 条相关事实")
    return "\n".join(parts)


def _generate_mental_model(observations: list[str]) -> str:
    """从观察生成心智模型（抽象规律）。"""
    if not observations:
        return ""
    keywords = _extract_keywords(observations, top_k=3)
    parts = []
    if keywords:
        parts.append(f"核心规律：{keywords[0]} 相关经验呈系统性模式")
    parts.append("基于以下观察抽象：")
    for obs in observations[:3]:
        # 截取观察的第一行
        first_line = obs.split("\n")[0] if obs else ""
        parts.append(f"  - {first_line[:100]}")
    return "\n".join(parts)


# ─── ConsolidationEngine ────────────────────────────────────


class ConsolidationEngine:
    """Consolidation 管线：事实 → 经验 → 观察 → 心智模型。"""

    def __init__(self, data_dir: Path | None = None, fact_threshold: int = 10, llm_client: Any | None = None):
        self._data_dir = data_dir
        self._fact_threshold = fact_threshold
        self._llm_client = llm_client
        self._pending: list[dict[str, Any]] = []
        self._conn: sqlite3.Connection | None = None
        self._consolidation_count = 0
        self._lock = threading.RLock()

        if data_dir:
            self._init_db(data_dir)

    def _init_db(self, data_dir: Path) -> None:
        """初始化 Consolidation 数据库。"""
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "consolidation.db"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="consolidation_items",
            create_sql="""
                CREATE TABLE IF NOT EXISTS consolidation_items (
                    item_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_ids TEXT,
                    confidence REAL DEFAULT 0.5,
                    created_at TEXT,
                    metadata TEXT
                )
            """,
            migrations=[],
        )
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_stage ON consolidation_items(stage)
        """)
        self._conn.commit()

    def submit(self, memory_id: str, content: str, memory_type: str = "fact") -> None:
        """提交一条记忆到 Consolidation 队列。"""
        logger.warning("Consolidation submit: %d items", len(self._pending))
        self._pending.append(
            {
                "memory_id": memory_id,
                "content": content,
                "type": memory_type,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def should_process(self) -> bool:
        """检查是否达到处理阈值。"""
        return len(self._pending) >= self._fact_threshold

    def process_pending(self) -> int:
        """处理待 Consolidation 的记忆：执行四阶段升华管线。

        Returns:
            处理的记忆数量
        """
        if not self._pending:
            return 0

        count = len(self._pending)
        facts = list(self._pending)
        self._pending.clear()

        # ─── Stage 1: world_facts → experience_facts ───
        experience_facts = self._annotate_experience(facts)

        # ─── Stage 2: experience_facts → observations ───
        observations = self._consolidate_observations(experience_facts)

        # ─── Stage 3: observations → mental_models ───
        mental_models = self._abstract_models(observations)

        # 持久化
        self._persist_items(experience_facts, "experience_facts")
        self._persist_items(observations, "observations")
        self._persist_items(mental_models, "mental_models")

        self._consolidation_count += count
        logger.info(
            "Consolidation: processed %d facts → %d experience → %d observations → %d models",
            count,
            len(experience_facts),
            len(observations),
            len(mental_models),
        )
        return count

    def reflect(self, query: str) -> ConsolidationResult:
        """对积累的记忆进行反思。

        查询已有的观察和心智模型，如果没有则从原始事实中生成。
        """
        # 尝试从已有观察中检索
        existing_obs = self._query_items("observations", query)
        existing_models = self._query_items("mental_models", query)

        if existing_models:
            best_model = existing_models[0]
            return ConsolidationResult(
                observation="\n".join(o.get("content", "") for o in existing_obs[:3]),
                mental_model=best_model.get("content", ""),
                facts_consolidated=0,
                observations_generated=len(existing_obs),
                models_generated=len(existing_models),
            )

        if existing_obs:
            # 有观察但无心智模型，尝试生成
            obs_contents = [o.get("content", "") for o in existing_obs]
            model = _generate_mental_model(obs_contents)
            return ConsolidationResult(
                observation="\n".join(obs_contents[:3]),
                mental_model=model,
                facts_consolidated=0,
                observations_generated=len(existing_obs),
                models_generated=1 if model else 0,
            )

        # 没有已处理的观察，返回空
        return ConsolidationResult(
            observation="",
            mental_model="",
            facts_consolidated=0,
        )

    def get_observations(self, topic: str = "", limit: int = 20) -> list[dict[str, Any]]:
        """获取观察列表。"""
        return self._query_items("observations", topic, limit)

    def get_mental_models(self, topic: str = "", limit: int = 10) -> list[dict[str, Any]]:
        """获取心智模型列表。"""
        return self._query_items("mental_models", topic, limit)

    def get_stats(self) -> dict[str, int]:
        """获取 Consolidation 统计。"""
        stats = {
            "total_consolidated": self._consolidation_count,
            "pending": len(self._pending),
        }
        if self._conn:
            for stage in ("world_facts", "experience_facts", "observations", "mental_models"):
                try:
                    row = self._conn.execute(
                        "SELECT COUNT(*) FROM consolidation_items WHERE stage = ?",
                        (stage,),
                    ).fetchone()
                    stats[stage] = int(row[0]) if row else 0
                except Exception:
                    stats[stage] = 0
        return stats

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    # ─── 内部方法 ─────────────────────────────────────────────

    def _generate_observation_with_llm(self, facts: list[dict[str, Any]]) -> str | None:
        if self._llm_client is None:
            return None
        contents = [f.get("content", "") for f in facts]
        evidence = "\n".join(f"{i + 1}. {c[:200]}" for i, c in enumerate(contents[:10]))
        prompt = f"请从以下事实中归纳出一条观察结论（2-3句话，中文）：\n\n{evidence}"
        system = "你是一个记忆整合引擎，从多条事实中提炼核心观察。输出必须是连贯的自然语言。"
        try:
            result = self._llm_client.call_sync(
                prompt=prompt, system=system, max_tokens=400, temperature=0.4,
            )
            content = result.content if result else None
            if content and content.strip():
                return content.strip()
        except Exception as e:
            logger.warning("ConsolidationEngine LLM call failed: %s", e)
        return None

    def _generate_model_with_llm(self, observations: list[str]) -> str | None:
        if self._llm_client is None:
            return None
        evidence = "\n".join(f"{i + 1}. {o[:200]}" for i, o in enumerate(observations[:5]))
        prompt = f"请从以下观察中抽象出一个心智模型（1-2句话描述规律或模式，中文）：\n\n{evidence}"
        system = "你是一个深度反思引擎，从观察中提炼抽象规律。输出必须是连贯的自然语言。"
        try:
            result = self._llm_client.call_sync(
                prompt=prompt, system=system, max_tokens=400, temperature=0.4,
            )
            content = result.content if result else None
            if content and content.strip():
                return content.strip()
        except Exception as e:
            logger.warning("ConsolidationEngine LLM call failed: %s", e)
        return None

    def _annotate_experience(self, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stage 1: 为事实添加上下文标注（经验化）。"""
        experience = []
        for fact in facts:
            content = fact.get("content", "")
            memory_type = fact.get("type", "fact")
            # 标注上下文
            context_tag = ""
            if memory_type == "correction":
                context_tag = "[纠错经验] "
            elif memory_type == "preference":
                context_tag = "[偏好经验] "
            elif memory_type == "skill":
                context_tag = "[技能经验] "
            else:
                context_tag = "[事实经验] "

            experience.append(
                {
                    "item_id": f"exp-{fact.get('memory_id', '')}",
                    "content": context_tag + content,
                    "source_ids": [fact.get("memory_id", "")],
                    "confidence": 0.8 if memory_type in ("correction", "preference") else 0.6,
                    "type": "experience_fact",
                }
            )
        return experience

    def _consolidate_observations(
        self, experience_facts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if len(experience_facts) < 2:
            return []

        observations: list[dict[str, Any]] = []
        clusters = _cluster_by_topic(experience_facts)

        for topic, cluster_facts in clusters.items():
            if len(cluster_facts) < 2:
                continue

            obs_content = self._generate_observation_with_llm(cluster_facts)
            if not obs_content:
                logger.warning("LLM observation generation failed, degrading to rule-based")
                obs_content = _generate_observation(cluster_facts)
            if obs_content:
                source_ids = [f.get("item_id", "") for f in cluster_facts]
                observations.append(
                    {
                        "item_id": f"obs-{topic}-{len(observations):03d}",
                        "content": obs_content,
                        "source_ids": source_ids,
                        "confidence": min(0.9, 0.5 + 0.1 * len(cluster_facts)),
                        "type": "observation",
                    }
                )

        return observations

    def _abstract_models(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(observations) < 2:
            return []

        models = []
        obs_contents = [o.get("content", "") for o in observations]
        model_content = self._generate_model_with_llm(obs_contents)
        if not model_content:
            logger.warning("LLM model generation failed, degrading to rule-based")
            model_content = _generate_mental_model(obs_contents)

        if model_content:
            source_ids = [o.get("item_id", "") for o in observations]
            models.append(
                {
                    "item_id": f"model-{self._consolidation_count:04d}",
                    "content": model_content,
                    "source_ids": source_ids,
                    "confidence": 0.7,
                    "type": "mental_model",
                }
            )

        return models

    def _persist_items(self, items: list[dict[str, Any]], stage: str) -> None:
        """持久化 Consolidation 产出到 SQLite。"""
        if not self._conn:
            return
        assert self._conn is not None  # narrow type for mypy
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            for item in items:
                try:
                    self._conn.execute(
                        """INSERT OR REPLACE INTO consolidation_items
                           (item_id, stage, content, source_ids, confidence, created_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            item.get("item_id", ""),
                            stage,
                            item.get("content", ""),
                            json.dumps(item.get("source_ids", []), ensure_ascii=False),
                            item.get("confidence", 0.5),
                            now,
                        ),
                    )
                except Exception as e:
                    logger.warning("Consolidation persist failed for %s: %s", item.get("item_id"), e)
            self._conn.commit()

    def _query_items(self, stage: str, keyword: str = "", limit: int = 20) -> list[dict[str, Any]]:
        """从数据库查询指定阶段的 Consolidation 产出。"""
        if not self._conn:
            return []
        assert self._conn is not None  # narrow type for mypy
        try:
            if keyword:
                rows = self._conn.execute(
                    "SELECT * FROM consolidation_items WHERE stage = ? AND content LIKE ? ORDER BY created_at DESC LIMIT ?",
                    (stage, f"%{keyword}%", limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM consolidation_items WHERE stage = ? ORDER BY created_at DESC LIMIT ?",
                    (stage, limit),
                ).fetchall()
            keys = [
                "item_id",
                "stage",
                "content",
                "source_ids",
                "confidence",
                "created_at",
                "metadata",
            ]
            results = []
            for row in rows:
                d = dict(zip(keys, row, strict=False))
                # 解析 source_ids JSON
                if d.get("source_ids"):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        d["source_ids"] = json.loads(d["source_ids"])
                results.append(d)
            return results
        except Exception as e:
            logger.warning("Consolidation query failed: %s", e)
            return []

    def close(self) -> None:
        """关闭数据库连接，处理剩余待升华数据。"""
        if self._pending:
            logger.info(
                "Consolidation: processing %d remaining pending items on close", len(self._pending)
            )
            self.process_pending()
        if self._conn:
            self._conn.close()
            self._conn = None
