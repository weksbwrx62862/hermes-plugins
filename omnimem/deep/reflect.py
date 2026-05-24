"""ReflectEngine — L3 深层记忆：Reflect 工具循环。

参考 Hindsight 的 Reflect 工具循环设计：
  Agent 调用 omni_reflect → Reflect 引擎执行:
    1. search_mental_models — 检索相关心智模型
    2. recall_facts — 检索相关事实
    3. expand_context — 扩展记忆上下文
    4. search_observations — 搜索观察洞察
    5. 生成反思输出（受 Disposition 性格影响）

Disposition 性格系统 (Hindsight):
  - skepticism (1-5): 怀疑度，越高越审慎
  - literalness (1-5): 字面度，越高越精确
  - empathy (1-5): 共情度，越高越关注感受

Phase 3 完整实现。
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)


# ─── 数据模型 ────────────────────────────────────────────────


@dataclass
class Disposition:
    """反思性格参数，控制 ReflectEngine 输出的语气和侧重。

    三个维度:
      skepticism (1-5): 怀疑度，越高越审慎，输出更保守
      literalness (1-5): 字面度，越高越精确，强调可验证性
      empathy (1-5): 共情度，越高越关注人的感受和影响
    """

    skepticism: int = 3  # 怀疑度 1-5
    literalness: int = 2  # 字面度 1-5
    empathy: int = 4  # 共情度 1-5

    def clamp(self) -> Disposition:
        """确保参数在合法范围内。"""
        return Disposition(
            skepticism=max(1, min(5, self.skepticism)),
            literalness=max(1, min(5, self.literalness)),
            empathy=max(1, min(5, self.empathy)),
        )

    def to_dict(self) -> dict[str, int]:
        """将性格参数序列化为字典。"""
        return {
            "skepticism": self.skepticism,
            "literalness": self.literalness,
            "empathy": self.empathy,
        }


@dataclass
class ReflectResult:
    """Reflect 结果。"""

    observation: str = ""
    mental_model: str = ""
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    disposition_used: dict[str, int] | None = None
    reflection_depth: int = 0  # 反思循环深度
    query: str = ""  # 反思查询关键词


@dataclass
class ReflectionContext:
    """反思循环中累积的上下文。"""

    query: str = ""
    mental_models: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    expanded: list[dict[str, Any]] = field(default_factory=list)


# ─── Disposition 影响的反思风格模板 ────────────────────────────


def _apply_disposition(observation: str, model: str, disposition: Disposition) -> tuple[str, str]:
    """根据 Disposition 参数调整反思输出的语气和侧重。

    Returns:
        (adjusted_observation, adjusted_model)
    """
    d = disposition.clamp()

    # ─── 怀疑度修饰 ───
    skepticism_prefixes = {
        1: "",
        2: "初步看来，",
        3: "据现有信息，",
        4: "需要谨慎对待以下判断：",
        5: "在缺乏更多证据的情况下，暂且认为：",
    }
    s_prefix = skepticism_prefixes.get(d.skepticism, "")

    # ─── 共情度修饰 ───
    # ★ 仅在内容涉及人/感受时添加共情后缀，技术/事实类内容不加
    _person_keywords = {
        "用户",
        "人",
        "感受",
        "情感",
        "心情",
        "体验",
        "偏好",
        "性格",
        "user",
        "people",
        "feeling",
        "emotion",
        "experience",
        "person",
    }
    has_person_context = any(kw in observation or kw in model for kw in _person_keywords)
    empathy_suffixes = {
        1: "",
        2: "",
        3: "（考虑当事人感受）" if has_person_context else "",
        4: "（需关注相关人的需求和感受）" if has_person_context else "",
        5: "（优先考虑对人的影响和情感因素）" if has_person_context else "",
    }
    e_suffix = empathy_suffixes.get(d.empathy, "")

    # ─── 字面度修饰 ───
    if d.literalness >= 4:
        # 高字面度：强调可验证性
        if model and not model.endswith("。"):
            model += "。（以上结论基于可验证的事实依据）"
    elif d.literalness <= 2:
        # 低字面度：允许推测
        if model and "可能" not in model and "或许" not in model:
            model = model.replace("核心规律", "可能的规律").replace("规律", "推测")

    adjusted_obs = f"{s_prefix}{observation}{e_suffix}" if s_prefix or e_suffix else observation
    adjusted_model = f"{s_prefix}{model}" if s_prefix else model

    return adjusted_obs, adjusted_model


# ─── ReflectEngine ────────────────────────────────────────────


class ReflectEngine:
    """Reflect 工具循环引擎 — L3 深层记忆反思。

    四步 Reflect 循环 (Hindsight-inspired):
      Step 1: search_mental_models — 从 Consolidation 查找已有心智模型
      Step 2: recall_facts — 检索相关事实（外部检索函数或 Consolidation 观察）
      Step 3: expand_context — 从事实中扩展关联上下文（关键词→更多观察）
      Step 4: search_observations — 搜索观察洞察
      Step 5: 综合生成 + Disposition 修饰 → 生成输出

    Disposition 性格系统:
      - skepticism (1-5): 怀疑度，越高越审慎
      - literalness (1-5): 字面度，越高越精确
      - empathy (1-5): 共情度，越高越关注感受

    生成策略:
      优先使用 LLM 推理归纳（通过 llm_fn），LLM 不可用时回退到规则归纳。
      规则归纳基于关键词提取和短语重组，输出质量有限。
      关键词堆砌检测和后处理确保输出为连贯自然语言。

    持久化:
      反思结果存入 SQLite (reflect.db)，支持历史查询和统计。
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        consolidation_engine: Any | None = None,
        default_disposition: Disposition | None = None,
        recall_fn: Callable[..., Any] | None = None,
        llm_fn: Callable[..., Any] | None = None,
        llm_client: Any | None = None,
    ):
        """初始化 ReflectEngine。

        Args:
            data_dir: 数据目录，用于持久化反思结果
            consolidation_engine: ConsolidationEngine 实例，用于查询观察/模型
            default_disposition: 默认性格参数
            recall_fn: 外部检索函数，签名: (query, limit) -> List[Dict]
            llm_fn: LLM 调用函数，签名: (prompt: str, system: str, max_tokens: int) -> str
                    接收 prompt + system prompt，返回 LLM 文本响应。
                    为 None 时回退到规则归纳。
            llm_client: LLM 客户端实例，用于直接调用 LLM
        """
        self._data_dir = data_dir
        self._consolidation = consolidation_engine
        self._default_disposition = default_disposition or Disposition()
        self._recall_fn = recall_fn
        self._llm_fn = llm_fn
        self._llm_client = llm_client
        self._conn: sqlite3.Connection | None = None
        self._reflection_count = 0
        self._lock = threading.RLock()

        if data_dir:
            self._init_db(data_dir)

    def _init_db(self, data_dir: Path) -> None:
        """初始化反思结果数据库。"""
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "reflect.db"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="reflections",
            create_sql="""
                CREATE TABLE IF NOT EXISTS reflections (
                    reflection_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    observation TEXT,
                    mental_model TEXT,
                    confidence REAL,
                    disposition TEXT,
                    source_ids TEXT,
                    created_at TEXT,
                    metadata TEXT
                )
            """,
            migrations=[],
        )
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reflect_query ON reflections(query)
        """)
        self._conn.commit()

    # ─── 公开接口 ─────────────────────────────────────────────

    def reflect(
        self,
        query: str,
        memories: list[dict[str, Any]] | None = None,
        disposition: dict[str, int] | None = None,
    ) -> ReflectResult:
        """执行完整的 Reflect 循环。

        Args:
            query: 反思主题
            memories: 外部提供的记忆列表（如来自检索引擎）
            disposition: 性格参数 (skepticism, literalness, empathy)

        Returns:
            ReflectResult
        """
        # 解析 Disposition
        disp = self._resolve_disposition(disposition)

        # 构建 Reflect 循环上下文
        ctx = ReflectionContext(query=query)

        # Step 1: search_mental_models — 从 Consolidation 查找已有心智模型
        ctx.mental_models = self._search_mental_models(query)

        # Step 2: recall_facts — 检索相关事实
        ctx.facts = self._recall_facts(query, memories)

        # Step 3: expand_context — 从事实中扩展关联
        ctx.expanded = self._expand_context(query, ctx.facts)

        # Step 4: search_observations — 搜索观察洞察
        ctx.observations = self._search_observations(query)

        # Step 5: 综合生成
        result = self._synthesize(query, ctx, disp)

        # 持久化
        self._persist_reflection(result)

        self._reflection_count += 1
        return result

    def get_reflection_history(self, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        """获取反思历史记录。

        Args:
            query: 可选的查询关键词，为空时返回全部
            limit: 最大返回条数

        Returns:
            反思记录列表，按时间倒序排列
        """
        if not self._conn:
            return []
        try:
            if query:
                rows = self._conn.execute(
                    "SELECT * FROM reflections WHERE query LIKE ? ORDER BY created_at DESC LIMIT ?",
                    (f"%{query}%", limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM reflections ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            keys = [
                "reflection_id",
                "query",
                "observation",
                "mental_model",
                "confidence",
                "disposition",
                "source_ids",
                "created_at",
                "metadata",
            ]
            return [dict(zip(keys, row, strict=False)) for row in rows]
        except Exception as e:
            logger.warning("Reflect history query failed: %s", e)
            return []

    def get_stats(self) -> dict[str, Any]:
        """获取反思统计信息，包含总反思次数和持久化数量。"""
        stats = {
            "total_reflections": self._reflection_count,
        }
        if self._conn:
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM reflections").fetchone()
                stats["persisted"] = row[0] if row else 0
            except Exception:
                stats["persisted"] = 0
        return stats

    def close(self) -> None:
        """关闭 SQLite 数据库连接，释放资源。"""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ─── Reflect 循环四步 ─────────────────────────────────────

    def _search_mental_models(self, query: str) -> list[dict[str, Any]]:
        """Step 1: 查找已有的心智模型。"""
        if self._consolidation:
            return self._consolidation.get_mental_models(topic=query, limit=5)  # type: ignore[no-any-return]
        return []

    def _recall_facts(
        self, query: str, memories: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Step 2: 检索相关事实。"""
        # 优先使用外部传入的记忆
        if memories:
            return memories[:20]

        # 使用外部检索函数
        if self._recall_fn:
            try:
                results = self._recall_fn(query, limit=20)
                if results:
                    return results  # type: ignore[no-any-return]
            except Exception as e:
                logger.warning("Recall function failed: %s", e)

        # 从 Consolidation 查询经验事实
        if self._consolidation:
            return self._consolidation.get_observations(topic=query, limit=20)  # type: ignore[no-any-return]
        return []

    def _expand_context(self, query: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Step 3: 从已有事实中扩展关联上下文。

        从事实内容中提取关键词，查找更广泛的观察。
        """
        if not facts or not self._consolidation:
            return []

        # 使用智能关键词提取替代正则切词
        keywords = set()
        for fact in facts[:10]:
            content = fact.get("content", "")
            kws = self._smart_extract_keywords(content, max_keywords=3)
            keywords.update(kws[:3])

        # 用关键词从 Consolidation 检索更多观察
        expanded = []
        for kw in list(keywords)[:5]:
            obs = self._consolidation.get_observations(topic=kw, limit=5)
            for o in obs:
                if o not in expanded and o not in facts:
                    expanded.append(o)
        return expanded[:15]

    def _search_observations(self, query: str) -> list[dict[str, Any]]:
        """Step 4: 搜索观察洞察。"""
        if self._consolidation:
            return self._consolidation.get_observations(topic=query, limit=10)  # type: ignore[no-any-return]
        return []

    # ─── 综合生成 ─────────────────────────────────────────────

    def _synthesize(
        self, query: str, ctx: ReflectionContext, disposition: Disposition
    ) -> ReflectResult:
        """综合 Reflect 循环四步的结果，生成最终反思输出。"""
        source_ids: list[str] = []
        mental_model = ""
        confidence = 0.0
        depth = 0

        # ─── 收集所有可用内容 ───
        all_contents: list[str] = []

        if ctx.mental_models:
            best_model = ctx.mental_models[0]
            mental_model = best_model.get("content", "")
            confidence = best_model.get("confidence", 0.7)
            source_ids.extend(
                best_model.get("source_ids", [])
                if isinstance(best_model.get("source_ids"), list)
                else []
            )
            depth = 3
            all_contents.append(f"[已有心智模型] {mental_model}")

        # 观察内容（无论有无心智模型都收集，供 LLM 深度推理）
        if ctx.observations:
            source_ids.extend(o.get("item_id", "") for o in ctx.observations[:5])
            for o in ctx.observations[:5]:
                all_contents.append(f"[观察] {o.get('content', '')[:200]}")
            depth = max(depth, 2)

        # 事实内容
        if ctx.facts:
            source_ids.extend(f.get("memory_id", f.get("item_id", "")) for f in ctx.facts[:8])
            for f in ctx.facts[:8]:
                all_contents.append(f"[事实] {f.get('content', '')[:200]}")
            depth = max(depth, 1)

        # 扩展上下文
        if ctx.expanded:
            for e in ctx.expanded[:3]:
                all_contents.append(f"[关联] {e.get('content', '')[:150]}")
                source_ids.append(e.get("item_id", ""))

        # ─── 无数据 ───
        if not all_contents:
            return ReflectResult(
                observation=f"没有找到与 '{query}' 相关的记忆来进行反思。",
                mental_model="",
                confidence=0.0,
                sources=[],
                disposition_used=disposition.to_dict(),
                reflection_depth=0,
                query=query,
            )

        # ─── 尝试 LLM 推理归纳 ───
        llm_result = self._generate_with_llm(query, all_contents, disposition)
        if llm_result is not None:
            llm_obs, llm_model, llm_conf = llm_result
            # LLM 成功 → 使用 LLM 输出
            observation = llm_obs or self._rule_based_observation(query, ctx)
            if llm_model:
                mental_model = llm_model
            # ★ confidence 合并：LLM 置信度与事实支撑度取较大值
            # 避免有事实支撑时 confidence=0.0（LLM 可能对异质信息返回低置信度）
            if llm_conf > 0:
                confidence = llm_conf
            elif ctx.facts and confidence < 0.3:
                confidence = 0.3
            depth = max(depth, 2)
        else:
            # LLM 不可用 → 回退到规则归纳
            logger.warning("LLM unavailable, degrading to rule-based synthesis")
            observation, mental_model, confidence = self._rule_based_synthesize(
                query, ctx, confidence
            )

        # ─── 应用 Disposition 修饰 ───
        observation, mental_model = _apply_disposition(observation, mental_model, disposition)

        # ─── 后处理：检测并拒绝关键词堆砌模式 ───
        mental_model = self._post_process_mental_model(mental_model, confidence)

        return ReflectResult(
            observation=observation,
            mental_model=mental_model,
            confidence=confidence,
            sources=source_ids[:20],
            disposition_used=disposition.to_dict(),
            reflection_depth=depth,
            query=query,
        )

    # ─── LLM 推理归纳 ──────────────────────────────────────────

    def _generate_with_llm(
        self,
        query: str,
        contents: list[str],
        disposition: Disposition,
        max_tokens: int = 800,
    ) -> tuple[str, str, float] | None:
        """使用 LLM 对记忆内容进行推理归纳。

        Returns:
            (observation, mental_model, confidence) 或 None（LLM 不可用时）
        """
        if self._llm_client is None and self._llm_fn is None:
            logger.warning("ReflectEngine: no LLM client available, skipping LLM call")
            return None

        # 构建推理 prompt
        d = disposition.clamp()
        skepticism_hint = {
            1: "大胆做出结论",
            2: "可以做出初步结论",
            3: "基于现有信息谨慎推理",
            4: "需要更多证据支持，仅给出暂时性判断",
            5: "明确标注不确定性，避免过度推断",
        }.get(d.skepticism, "基于现有信息推理")

        evidence_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(contents[:15]))

        prompt = (
            f"请对以下关于「{query}」的记忆内容进行深度反思和归纳推理。\\n\\n"
            f"推理要求：\\n"
            f"- {skepticism_hint}\\n"
            f"- 从表面事实中提炼深层规律和模式\\n"
            f"- 识别矛盾或不确定性\\n"
            f"- 用简洁的中文表达，避免罗列关键词\\n"
            f"- 心智模型必须是完整的因果陈述句，禁止输出逗号分隔的关键词列表\\n\\n"
            f"记忆内容：\\n{evidence_block}\\n\\n"
            f"请按以下格式输出（严格遵守）：\\n"
            f"【观察】\\n"
            f"（对记忆内容的归纳性总结，2-4句话，提炼核心发现而非复述原文）\\n\\n"
            f"【心智模型】\\n"
            f"（从观察中提炼的规律性认知，1-2个完整的因果陈述句，描述因果关系或模式）\\n\\n"
            f"【置信度】\\n"
            f"（0.0-1.0的数字，表示对上述结论的确信程度）"
        )

        system = (
            "你是一个深度反思引擎。你的任务是从记忆片段中进行归纳推理，"
            "提炼出非平凡的观察和心智模型。"
            "不要简单复述或罗列关键词，要进行真正的推理和抽象。"
            "输出必须严格遵守指定的【观察】【心智模型】【置信度】格式。"
        )

        # ★ R15修复：添加重试机制（最多3次，R17增加到3次以应对截断和关键词堆砌）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw = None
                # ★ R25修复ARCH-1：优先尝试 _llm_fn（经过 provider 的凭证管理），
                # 再尝试直接 _llm_client，确保 LLM 路径被正确调用
                if self._llm_fn is not None:
                    try:
                        raw = self._llm_fn(prompt=prompt, system=system, max_tokens=max_tokens)
                    except Exception as e:
                        logger.warning(
                            "ReflectEngine _llm_fn failed (attempt %d/%d): %s: %s",
                            attempt + 1, max_retries, type(e).__name__, e,
                        )
                if not raw and self._llm_client is not None:
                    try:
                        result = self._llm_client.call_sync(
                            prompt=prompt, system=system, max_tokens=max_tokens, temperature=0.5,
                        )
                        raw = result.content if result else None
                    except Exception as e:
                        logger.warning(
                            "ReflectEngine _llm_client failed (attempt %d/%d): %s: %s",
                            attempt + 1, max_retries, type(e).__name__, e,
                        )
                if not raw or not raw.strip():
                    if attempt < max_retries - 1:
                        logger.warning(
                            "ReflectEngine LLM returned empty (attempt %d/%d), retrying...",
                            attempt + 1,
                            max_retries,
                        )
                        continue
                    logger.warning("LLM generation failed, degrading to rule-based")
                    return None

                # ★ R17修复：检测截断 — 如果输出没有结束标记（【置信度】），可能被截断
                has_complete_structure = bool(re.search(r"【置信度】|置信度[：:]", raw))
                if not has_complete_structure and attempt < max_retries - 1:
                    logger.warning(
                        "ReflectEngine LLM output appears truncated (attempt %d/%d), retrying...",
                        attempt + 1,
                        max_retries,
                    )
                    # 截断重试时增加max_tokens
                    max_tokens = min(max_tokens + 200, 1200)
                    continue

                # 解析 LLM 输出
                obs, model, conf = self._parse_llm_output(raw.strip())

                # ★ R17修复：对截断的observation进行修补
                if obs and len(obs) < 15 and attempt < max_retries - 1:
                    logger.warning(
                        "ReflectEngine LLM observation too short (%d chars), likely truncated (attempt %d/%d)",
                        len(obs),
                        attempt + 1,
                        max_retries,
                    )
                    continue

                # 验证mental_model质量
                if model and self._is_keyword_stuffing(model):
                    if attempt < max_retries - 1:
                        logger.warning(
                            "ReflectEngine LLM returned keyword stuffing (attempt %d/%d), retrying...",
                            attempt + 1,
                            max_retries,
                        )
                        # 关键词堆砌重试时，在prompt中增加更明确的反堆砌指令
                        if attempt == max_retries - 2:
                            prompt += (
                                "\n\n★ 重要提醒：你的【心智模型】输出必须是完整的自然语言句子，"
                                "描述因果关系或模式规律，严禁输出关键词列表！"
                            )
                        continue

                return (obs, model, conf)

            except Exception as e:
                logger.warning(
                    "ReflectEngine LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, e
                )
                if attempt < max_retries - 1:
                    continue

        logger.warning("LLM generation failed, degrading to rule-based")
        return None

    @staticmethod
    def _parse_llm_output(raw: str) -> tuple[str, str, float]:
        """解析 LLM 输出为 (observation, mental_model, confidence)。

        支持多种标记格式：
        1. 中文标记：【观察】/【心智模型】/【置信度】
        2. 英文标记：observation:/mental_model:/confidence:
        3. ★ R33修复ARCH-1：JSON 格式 {"observation": "...", "mental_model": "...", "confidence": 0.8}
        """
        observation = ""
        mental_model = ""
        confidence = None

        # ★ R33修复ARCH-1：先尝试 JSON 格式解析
        # 某些 LLM 会忽略中文标记格式要求，返回 JSON 格式
        stripped = raw.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    observation = parsed.get("observation", parsed.get("obs", ""))
                    mental_model = parsed.get("mental_model", parsed.get("model", ""))
                    conf_val = parsed.get("confidence", parsed.get("conf", None))
                    if conf_val is not None:
                        try:
                            confidence = max(0.0, min(1.0, float(conf_val)))
                        except (ValueError, TypeError):
                            confidence = 0.5
                    if observation or mental_model:
                        if confidence is None:
                            confidence = 0.5
                        return (str(observation), str(mental_model), confidence)
            except (json.JSONDecodeError, ValueError):
                pass

        # 尝试中文标记解析
        obs_match = re.search(
            r"(?:【观察】|观察[：:]\s*)\s*\n?(.*?)(?=【心智模型】|心智模型[：:]|\Z)", raw, re.DOTALL
        )
        model_match = re.search(
            r"(?:【心智模型】|心智模型[：:]\s*)\s*\n?(.*?)(?=【置信度】|置信度[：:]|\Z)",
            raw,
            re.DOTALL,
        )
        conf_match = re.search(r"(?:【置信度】|置信度[：:]\s*)\s*\n?([\d.]+)", raw)

        if obs_match:
            observation = obs_match.group(1).strip()
        if model_match:
            mental_model = model_match.group(1).strip()
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5

        # 如果格式解析失败，整体作为观察
        if not observation and not mental_model:
            observation = raw.strip()[:500]

        # 未解析到置信度时使用默认值
        if confidence is None:
            confidence = 0.5

        return (observation, mental_model, confidence)

    # ─── 规则归纳回退 ──────────────────────────────────────────

    def _smart_extract_keywords(self, text: str, max_keywords: int = 6) -> list[str]:
        """智能关键词提取：按标点和语义边界切分，避免破碎分词。

        替代 re.findall(r'[\\u4e00-\\u9fff]{2,4}', text) 这种滑动窗口切词，
        该方法会把"对记忆系统进行"切成"对记忆"、"记忆系"、"系统进行"等碎片。

        策略：
        1. 按标点（中英文逗号、顿号、句号、冒号、分号、空格）分割为片段
        2. 过滤停用词片段
        3. 保留有实际含义的片段（2-12字的中文片段，或英文单词）
        4. 去重并保留顺序
        """
        if not text:
            return []

        # 按标点和空白分割
        segments = re.split(r"[，,、；;：:。\.\s！？!?()\（\）\[\]【】「」\n\r\t]+", text)

        # 停用词和低质量前缀
        zh_stopwords = {
            "关于",
            "问题",
            "情况",
            "使用",
            "进行",
            "需要",
            "可以",
            "已经",
            "其中",
            "以上",
            "以下",
            "这个",
            "那个",
            "就是",
            "还是",
            "而且",
            "因为",
            "所以",
            "如果",
            "虽然",
            "但是",
            "不过",
            "然后",
            "接着",
            "之前",
            "之后",
            "通过",
            "包括",
            "以及",
            "对于",
            "基于",
            "据现有",
            "信息",
            "据现",
            "现有",
            "有信",
            "据现有信息",
            # R17新增：测试轮次标记（R12/R14/R15等）和测试通用词
            "对记忆系统进行",
            "系统进行",
            "进行记忆",
            "记忆系统回归",
            "系统回归",
            "回归测试",
            "测试范围包括",
        }
        # 丢弃以"对记忆"/"系统回"等低质量模式开头的片段
        low_quality_prefixes = ("对记忆", "系统回", "进行记", "记忆系", "统进行")
        en_stopwords = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "that",
            "this",
            "with",
            "from",
            "have",
            "been",
            "was",
            "were",
        }

        keywords = []
        seen = set()
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            # 中文片段：2-15字，非停用词，非低质量前缀，非低质量结尾
            if re.match(r"^[\u4e00-\u9fff]{2,15}$", seg):
                # 跳过以停用词结尾的片段
                zh_bad_endings = ("使用", "进行", "需要", "可以", "关于", "包括", "通过", "基于")
                if (
                    seg not in zh_stopwords
                    and seg not in seen
                    and not any(seg.startswith(p) for p in low_quality_prefixes)
                    and not any(seg.endswith(s) for s in zh_bad_endings)
                ):
                    keywords.append(seg)
                    seen.add(seg)
            # 中英混合片段：提取中文部分（3字以上）
            elif re.search(r"[\u4e00-\u9fff]{3,}", seg):
                zh_bad_endings = ("使用", "进行", "需要", "可以", "关于", "包括", "通过", "基于")
                for m in re.finditer(r"[\u4e00-\u9fff]{3,}", seg):
                    chunk = m.group()
                    if (
                        chunk not in zh_stopwords
                        and chunk not in seen
                        and not any(chunk.startswith(p) for p in low_quality_prefixes)
                        and not any(chunk.endswith(s) for s in zh_bad_endings)
                    ):
                        keywords.append(chunk)
                        seen.add(chunk)
            # 纯英文单词
            elif re.match(r"^[a-zA-Z]{3,}$", seg.lower()):
                if seg.lower() not in en_stopwords and seg.lower() not in seen:
                    keywords.append(seg.lower())
                    seen.add(seg.lower())

            if len(keywords) >= max_keywords:
                break

        return keywords

    def _extract_content_phrases(self, texts: list[str], max_phrases: int = 5) -> list[str]:
        """从文本列表中提取有意义的短句/短语，用于组织连贯的输出。

        区别于关键词提取，这里保留更完整的语义片段。
        """
        phrases = []
        seen = set()
        for text in texts[:10]:
            # 按标点分割，保留有实质内容的片段
            parts = re.split(r"[，,、；;。\.\n]+", text)
            for part in parts:
                part = part.strip()
                # 过滤：太短、纯标记、停用词开头
                if len(part) < 4:
                    continue
                if part.startswith(("R1", "R1", "[", "—", "※", "★")):
                    continue
                if any(part.startswith(sw) for sw in ("关于", "据现", "基于", "核心")):
                    continue
                # 保留6-60字的有意义片段
                if 4 <= len(part) <= 80 and part not in seen:
                    phrases.append(part)
                    seen.add(part)
                if len(phrases) >= max_phrases:
                    return phrases
        return phrases

    def _rule_based_observation(self, query: str, ctx: ReflectionContext) -> str:
        """规则归纳：生成观察文本（LLM 不可用时的回退）。"""
        observation_parts: list[str] = []

        if ctx.mental_models:
            ctx.mental_models[0]
            observation_parts.append("已有心智模型支撑：")
            for o in ctx.observations[:3]:
                observation_parts.append(f"  - {o.get('content', '')[:150]}")

        elif ctx.observations:
            obs_contents = [o.get("content", "") for o in ctx.observations]
            phrases = self._extract_content_phrases(obs_contents)
            observation_parts.append(f"基于 {len(ctx.observations)} 条观察的归纳：")
            if phrases:
                observation_parts.append(f"  主要发现：{phrases[0]}")
                for p in phrases[1:3]:
                    observation_parts.append(f"  - {p}")
            else:
                keywords = self._smart_extract_keywords(" ".join(obs_contents))
                if keywords:
                    observation_parts.append(f"  涉及主题：{'、'.join(keywords[:3])}")

        elif ctx.facts:
            fact_contents = [f.get("content", "") for f in ctx.facts]
            phrases = self._extract_content_phrases(fact_contents)
            observation_parts.append(f"基于 {len(ctx.facts)} 条记忆的归纳：")
            if phrases:
                observation_parts.append(f"  主要发现：{phrases[0]}")
                for p in phrases[1:3]:
                    observation_parts.append(f"  - {p}")
            else:
                keywords = self._smart_extract_keywords(" ".join(fact_contents))
                if keywords:
                    observation_parts.append(f"  涉及主题：{'、'.join(keywords[:3])}")

        if ctx.expanded:
            exp_contents = [e.get("content", "") for e in ctx.expanded[:3]]
            observation_parts.append("\n关联上下文：")
            for ec in exp_contents:
                observation_parts.append(f"  - {ec[:120]}")

        return "\n".join(observation_parts) if observation_parts else ""

    def _rule_based_synthesize(
        self, query: str, ctx: ReflectionContext, base_confidence: float
    ) -> tuple[str, str, float]:
        """规则归纳完整回退（观察 + 心智模型 + 置信度）。

        注意：规则归纳是 LLM 不可用时的降级方案，输出质量有限。
        confidence 基于来源数量和一致性动态计算。
        """
        observation = self._rule_based_observation(query, ctx)

        # 生成心智模型
        if ctx.observations:
            obs_contents = [o.get("content", "") for o in ctx.observations]
            mental_model = self._generate_model_from_observations(obs_contents, query)
            # 动态confidence：基于来源数量
            n_sources = len(ctx.observations) + len(ctx.facts)
            confidence = min(0.45, 0.25 + n_sources * 0.02)
        elif ctx.facts:
            fact_contents = [f.get("content", "") for f in ctx.facts]
            obs = self._generate_observation_from_facts(fact_contents, query)
            if obs:
                observation += f"\n综合观察：{obs}"
            mental_model = self._generate_model_from_facts(fact_contents, query)
            # 动态confidence：基于来源数量和短语质量
            n_sources = len(ctx.facts)
            phrases = self._extract_content_phrases(fact_contents, max_phrases=1)
            base = 0.30 + n_sources * 0.02
            if phrases:
                base += 0.05  # 有完整短语的加分
            confidence = min(0.45, base)
        else:
            mental_model = ""
            confidence = 0.2

        return (observation, mental_model, confidence)

    # ─── 辅助生成 ─────────────────────────────────────────────

    def _generate_model_from_observations(self, observations: list[str], query: str) -> str:
        """从观察生成心智模型（语义句子，非关键词堆砌）。"""
        if not observations:
            return ""

        keywords = self._smart_extract_keywords(" ".join(observations[:5]))
        phrases = self._extract_content_phrases(observations, max_phrases=3)

        # 用有意义的短语组织句子
        if phrases:
            main_point = phrases[0]
            if len(phrases) >= 2:
                return (
                    f"在「{query}」方面，{main_point}；"
                    f"同时{phrases[1]}。"
                    f"基于{len(observations)}条观察，这些方面呈现出关联趋势。"
                )
            else:
                return (
                    f"在「{query}」方面，{main_point}。"
                    f"基于{len(observations)}条观察，该领域有待进一步验证。"
                )
        elif keywords:
            return (
                f"围绕「{'、'.join(keywords[:3])}」等主题，"
                f"已有{len(observations)}条观察记录，但这些信息尚不足以形成完整的因果推断。"
            )
        else:
            return f"关于「{query}」的观察信息有限，需更多数据支撑。"

    def _generate_observation_from_facts(self, facts: list[str], query: str) -> str:
        """从事实列表生成初步观察。"""
        if len(facts) < 2:
            return ""

        # 按语义关键词聚类
        keyword_map: dict[str, list[str]] = {}
        for fact in facts:
            kws = self._smart_extract_keywords(fact, max_keywords=2)
            key = kws[0] if kws else "general"
            keyword_map.setdefault(key, []).append(fact)

        parts = []
        for kw, group in list(keyword_map.items())[:3]:
            if len(group) >= 2:
                phrases = self._extract_content_phrases(group, max_phrases=1)
                if phrases:
                    parts.append(
                        f"在「{kw}」方面，{len(group)}条记忆显示一致趋势（如{phrases[0]}）"
                    )
                else:
                    parts.append(f"在「{kw}」方面，{len(group)}条记忆显示一致趋势")
            else:
                phrases = self._extract_content_phrases(group, max_phrases=1)
                if phrases:
                    parts.append(f"关于「{kw}」有记录显示{phrases[0]}")

        return "；".join(parts) if parts else ""

    def _generate_model_from_facts(self, facts: list[str], query: str) -> str:
        """从事实列表直接生成初步心智模型（语义句子，非关键词堆砌）。

        当 Consolidation 管线尚未产出 observation/mental_model 时，
        reflect 仍然可以从原始事实中生成有意义的初步模型。
        """
        if not facts:
            return ""

        keywords = self._smart_extract_keywords(" ".join(facts[:8]))
        phrases = self._extract_content_phrases(facts, max_phrases=3)

        # 用有意义的短语组织成连贯的句子
        if phrases:
            main_point = phrases[0]
            supporting = phrases[1] if len(phrases) >= 2 else None
            if supporting:
                return (
                    f"关于「{query}」，{main_point}，"
                    f"此外{supporting}。"
                    f"基于{len(facts)}条记忆，上述信息存在关联但因果性待验证。"
                )
            else:
                return (
                    f"关于「{query}」，{main_point}。基于{len(facts)}条记忆，该认知尚需更多验证。"
                )
        elif keywords:
            return (
                f"关于「{query}」的现有记忆涉及{'、'.join(keywords[:3])}等方面，"
                f"但这些信息尚属碎片化，需进一步整理才能形成完整认知。"
            )
        else:
            return f"关于「{query}」的信息不足，无法形成有效推断。"

    # ─── 持久化 ───────────────────────────────────────────────

    def _persist_reflection(self, result: ReflectResult) -> None:
        """持久化反思结果。"""
        if not self._conn:
            return
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            reflection_id = f"ref-{self._reflection_count:04d}-{now[:10]}"
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO reflections
                       (reflection_id, query, observation, mental_model, confidence,
                        disposition, source_ids, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        reflection_id,
                        result.query,
                        result.observation,
                        result.mental_model,
                        result.confidence,
                        json.dumps(result.disposition_used, ensure_ascii=False)
                        if result.disposition_used
                        else "",
                        json.dumps(result.sources, ensure_ascii=False),
                        now,
                    ),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning("Reflect persist failed: %s", e)

    # ─── 内部辅助 ─────────────────────────────────────────────

    def _resolve_disposition(self, disposition: dict[str, int] | None) -> Disposition:
        """解析 Disposition 参数。"""
        if disposition:
            return Disposition(
                skepticism=disposition.get("skepticism", self._default_disposition.skepticism),
                literalness=disposition.get("literalness", self._default_disposition.literalness),
                empathy=disposition.get("empathy", self._default_disposition.empathy),
            ).clamp()
        return self._default_disposition.clamp()

    @staticmethod
    def _is_keyword_stuffing(text: str) -> bool:
        """检测文本是否为关键词堆砌模式。

        关键词堆砌特征：
        1. 短语（≤6字）用逗号/顿号分隔
        2. 没有完整的句子结构（缺少主谓宾）
        3. 分隔符数量占比过高

        Returns:
            True 如果检测到关键词堆砌模式
        """
        if not text or len(text) < 4:
            return True

        cleaned = text.strip()

        # 特征1: 逗号/顿号分隔的短词列表
        separators = r"[，,、；;]+"
        parts = re.split(separators, cleaned)

        if len(parts) < 2:
            return False

        # 检查每个部分是否都是短词（≤6字）
        short_parts = [p.strip() for p in parts if p.strip() and len(p.strip()) <= 6]

        # 如果超过60%的部分是短词 → 关键词堆砌
        if len(short_parts) >= 2 and len(short_parts) / len(parts) > 0.6:
            # 额外检查：排除正常的列举格式（如"1. xxx 2. xxx"）
            has_numbering = bool(re.match(r"^\s*\d+[\.\、]", cleaned))
            if not has_numbering:
                return True

        # 特征2: 分隔符密度过高（每3个字符就有1个分隔符）
        sep_count = len(re.findall(separators, cleaned))
        if sep_count > 0 and len(cleaned) / (sep_count + 1) < 5:
            return True

        # 特征3: 缺少句子结构（没有句号，且没有常见的谓语动词）
        has_sentence_end = any(cleaned.endswith(p) for p in ["。", "！", "？", ".", "!", "?"])
        if not has_sentence_end and len(parts) >= 3:
            # 检查是否缺少动词
            verbs = {
                "是",
                "有",
                "在",
                "为",
                "呈",
                "显示",
                "表明",
                "呈现",
                "涉及",
                "包含",
                "be",
                "has",
                "is",
                "shows",
                "indicates",
            }
            has_verb = any(v in cleaned for v in verbs)
            if not has_verb:
                return True

        return False

    def _post_process_mental_model(self, mental_model: str, confidence: float) -> str:
        """后处理：检测并修复关键词堆砌的心智模型。

        当检测到关键词堆砌时：
        - 如果 confidence < 0.5（规则归纳生成），尝试重新组织为连贯文本
        - 如果 confidence ≥ 0.5（LLM生成但质量差），保留但添加警告标记

        Args:
            mental_model: 原始心智模型文本
            confidence: 当前置信度

        Returns:
            处理后的心智模型文本
        """
        if not mental_model:
            return mental_model

        if not self._is_keyword_stuffing(mental_model):
            return mental_model

        logger.warning(
            "ReflectEngine detected keyword stuffing in mental_model (confidence=%.2f): %s",
            confidence,
            mental_model[:100],
        )

        # 规则归纳生成的低质量输出 → 提取有意义的片段并重组
        if confidence < 0.5:
            # 尝试从mental_model中提取中文短语
            phrases = re.findall(
                r"[\u4e00-\u9fff]{3,}(?:的|了|在|与|和|是|有|为|到|从|对|向)?[\u4e00-\u9fff]*",
                mental_model,
            )
            meaningful = [
                p for p in phrases if len(p) >= 3 and p not in {"关于", "初步认知", "核心要素"}
            ]

            if len(meaningful) >= 2:
                unique = list(dict.fromkeys(meaningful))[:4]
                return (
                    f"当前记忆显示在{unique[0]}与{unique[1]}方面存在关联性，"
                    f"{'、'.join(unique[2:]) + '等' if len(unique) > 2 else ''}"
                    f"这些领域的交互模式在多次记录中反复出现。"
                )
            elif len(meaningful) == 1:
                return (
                    f"当前记忆显示用户关注的主要领域是{meaningful[0]}，"
                    f"但已记录的信息尚不足以形成完整的因果推断。"
                )
            return (
                "当前记忆积累尚处于初步阶段，已记录的信息呈现碎片化特征，"
                "建议通过持续交互积累更多观察以形成更完整的心智模型。"
            )

        # LLM 生成的低质量输出 → 保留原内容但添加标记
        return f"[⚠ 质量警告] {mental_model}"
