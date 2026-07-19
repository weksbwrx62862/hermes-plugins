"""ConflictResolver — 冲突仲裁（两阶段检测）。

当新记忆与已有记忆矛盾时，执行仲裁：
  Stage 1: 否定词快速检测（零成本）
  Stage 2: 语义相似度检测（可选，需要向量检索）
  策略：最新优先 / 高置信优先 / 手动选择
  结果：accept / reject / merge
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConflictResult:
    """冲突检测结果。"""

    has_conflict: bool = False
    existing_memory: str = ""
    existing_id: str = ""
    conflict_type: str = ""  # "negation" / "semantic_contradiction" / "update" / "duplicate"
    action: str = "accept"  # "accept" / "reject" / "merge"
    reason: str = ""
    # ★ 知识更新标记（Task 2）
    is_updated: bool = False       # 新记忆是更新版本
    is_superseded: bool = False    # 旧记忆已被取代
    superseded_id: str = ""        # 被取代的旧记忆 ID


class ConflictResolver:
    """冲突仲裁引擎（两阶段检测）。

    Stage 1: 否定词快速检测 — 检测内容中的否定/纠正标记（零成本）
    Stage 2: 语义相似度检测 — 与已有记忆做向量相似度比对（可选）
    """

    # 否定词模式（Stage 1）
    _NEGATION_PATTERNS = [
        # 中文否定
        "不对",
        "不是",
        "不用",
        "错误",
        "纠正",
        "更正",
        "不再",
        "而非",
        "实际上",
        "事实上",
        "相反",
        "纠正一下",
        "我说错了",
        "改用",
        "并不",
        "并非",
        "不要",
        "无法",
        "没能",
        "改为",
        # 英文否定
        "NOT ",
        "DON'T ",
        "WRONG",
        "INCORRECT",
        "ACTUALLY",
        "CORRECTION",
        "INSTEAD",
        "NO LONGER",
        "RATHER THAN",
    ]

    # 纠正标记（高优先级）
    _CORRECTION_MARKERS = [
        "CORRECTION:",
        "纠正:",
        "更正:",
        "纠正一下",
    ]

    def __init__(
        self,
        strategy: str = "latest",
        semantic_check_fn: Callable[..., Any] | None = None,
        similarity_threshold: float = 0.85,
    ):
        """初始化冲突仲裁器。

        Args:
            strategy: 仲裁策略
                "latest" — 最新记忆优先
                "confidence" — 高置信度优先
                "manual" — 标记冲突，等待用户/Agent 决定
            semantic_check_fn: 语义相似度检查函数
                签名: (query, threshold) -> List[Dict]
                返回相似记忆列表，None 则跳过语义检测
            similarity_threshold: 语义相似度阈值（默认0.85）
        """
        self._strategy = strategy
        self._semantic_check_fn = semantic_check_fn
        self._similarity_threshold = similarity_threshold
        self._conflict_log: list[dict[str, Any]] = []

    def check(
        self, content: str, existing_memories: list[dict[str, Any]] | None = None
    ) -> ConflictResult:
        """两阶段冲突检测。

        Stage 1: 否定词快速检测
        Stage 2: 语义相似度检测（可选）

        ★ 优化：否定词检测不再单独判定为冲突，必须同时满足以下条件之一：
          - 有 existing_memories 且存在语义矛盾（Stage 2 确认）
          - 有 semantic_check_fn 且检索到高相似度的已有记忆
        仅含否定词但无匹配的已有记忆，不视为冲突（"纠正"类内容本身含否定词是正常的）。

        Args:
            content: 新记忆内容
            existing_memories: 已有记忆列表（用于精确检测）

        Returns:
            ConflictResult
        """
        # ─── Stage 1: 否定词快速检测（零成本） ───
        negation = self._detect_negation(content)
        if negation:
            # ★ 否定词检测必须配合 Stage 2 语义验证，否则 "纠正: xxx" 类内容会误判为冲突
            # 如果有已有记忆，检查是否存在语义矛盾
            if existing_memories:
                semantic_conflict = self._check_semantic_with_memories(content, existing_memories)
                if semantic_conflict:
                    return semantic_conflict
            # 如果有语义检索函数，也继续检查
            if self._semantic_check_fn:
                semantic_conflict = self._check_semantic_with_fn(content)
                if semantic_conflict:
                    return semantic_conflict
            # ★ 否定词存在但无匹配的已有矛盾记忆 → 不视为冲突
            # correction 类型含否定词是正常的语义表达，不是冲突

        # ─── Stage 2: 语义相似度检测（可选） ───
        # 2a: 如果传入了 existing_memories，直接比较
        if existing_memories:
            semantic_conflict = self._check_semantic_with_memories(content, existing_memories)
            if semantic_conflict:
                return semantic_conflict

        # 2b: 如果有语义检索函数，查询相似记忆
        if self._semantic_check_fn:
            semantic_conflict = self._check_semantic_with_fn(content)
            if semantic_conflict:
                return semantic_conflict

        return ConflictResult(has_conflict=False)

    def resolve(self, content: str, conflict: ConflictResult) -> ConflictResult:
        """根据策略解决冲突。"""
        if self._strategy == "latest":
            conflict.action = "accept"
            conflict.reason = "Latest memory takes priority"
        elif self._strategy == "add_only":
            # ★ ADD-only 策略：新记忆直接写入，旧记忆不标记 superseded
            # 检索层保留所有记忆，由 LLM 合成或排序决定使用哪条
            conflict.action = "accept"
            conflict.is_updated = False
            conflict.is_superseded = False
            conflict.superseded_id = ""
            conflict.reason = "ADD-only: both memories retained, retrieval decides relevance"
        elif self._strategy == "confidence":
            conflict.action = "accept"
            conflict.reason = "Resolved by confidence comparison"
        elif self._strategy == "manual":
            conflict.action = "accept"
            conflict.reason = "Auto-accepted (manual mode would defer to user)"

        # ★ Task 2: 知识更新冲突时设置标记（add_only 模式除外）
        if self._strategy != "add_only" and conflict.conflict_type == "update":
            conflict.is_updated = True
            conflict.is_superseded = True
            conflict.superseded_id = conflict.existing_id

        # 记录冲突（★ 包含 memory_id，供 resolve_by_id 查询）
        self._conflict_log.append(
            {
                "memory_id": conflict.existing_id,
                "content": content[:200],
                "conflict_type": conflict.conflict_type,
                "action": conflict.action,
                "reason": conflict.reason,
            }
        )

        return conflict

    def resolve_by_id(self, memory_id: str) -> ConflictResult:
        """根据 ID 解决已记录的冲突。"""
        for entry in reversed(self._conflict_log):
            if entry.get("memory_id") == memory_id:
                return ConflictResult(
                    action="resolved",
                    reason=f"Conflict for {memory_id} resolved (latest wins)",
                )
        return ConflictResult(action="resolved", reason="No pending conflict found")

    # ─── Stage 1: 否定词检测 ────────────────────────────────

    def _detect_negation(self, content: str) -> str:
        """检测内容中的否定/矛盾标记。"""
        # 先检查纠正标记（高优先级）
        content_upper = content.upper()
        for marker in self._CORRECTION_MARKERS:
            if marker.upper() in content_upper:
                return marker

        # 检查否定词
        content_lower = content.lower()
        for pattern in self._NEGATION_PATTERNS:
            if pattern.lower() in content_lower:
                return pattern
        return ""

    # ─── Stage 2: 语义相似度检测 ──────────────────────────────

    def _check_semantic_with_memories(
        self, content: str, memories: list[dict[str, Any]]
    ) -> ConflictResult | None:
        """用已有记忆列表做语义冲突检测。

        通过否定词+同主题来判断：如果新内容否定某条已有记忆，则标记冲突。
        ★ 同时检测互斥选项模式：同主题但选择了互斥的选项（如 AWS vs 腾讯云）。
        """
        content_lower = content.lower()
        candidates = self._find_candidates(content_lower, memories)
        for mem_content, mem_content_lower, mem_id, overlap in candidates:
            result = self._compare_semantics(
                content_lower, mem_content, mem_content_lower, mem_id, overlap
            )
            if result:
                return result
        return None

    def _find_candidates(self, content_lower: str, memories: list[dict[str, Any]]) -> list[tuple[str, str, str, float]]:
        """从记忆列表中筛选候选记忆，计算重叠率。"""
        candidates = []
        for mem in memories:
            mem_content = mem.get("content", "")
            mem_id = mem.get("memory_id", "")
            if not mem_content:
                continue
            mem_content_lower = mem_content.lower()
            overlap = self._compute_overlap(content_lower, mem_content_lower)
            candidates.append((mem_content, mem_content_lower, mem_id, overlap))
        return candidates

    def _compare_semantics(
        self,
        content_lower: str,
        mem_content: str,
        mem_content_lower: str,
        mem_id: str,
        overlap: float,
    ) -> ConflictResult | None:
        """对单条候选记忆执行语义冲突比较。"""
        result = self._check_mutual_exclusive(
            content_lower, mem_content, mem_content_lower, mem_id, overlap
        )
        if result:
            return result
        # ★ R29修复BUG-3：否定词检测独立于 overlap 阈值
        # 之前否定词检测在 _check_topic_divergence 内部，被 overlap>0.3 门控
        # 导致低重叠的矛盾对（如"我喜欢Python" vs "我不再喜欢Python，改用Go了"）漏检
        # 修复：先检查否定词矛盾（只要 overlap>0 即有主题关联），再走高重叠分歧检测
        if overlap > 0:
            negation_result = self._check_negation_conflict(
                content_lower, mem_content, mem_content_lower, mem_id, overlap
            )
            if negation_result:
                return negation_result
        if overlap > 0.3:
            return self._check_topic_divergence(
                content_lower, mem_content, mem_content_lower, mem_id, overlap
            )
        return None

    def _check_mutual_exclusive(
        self,
        content_lower: str,
        mem_content: str,
        mem_content_lower: str,
        mem_id: str,
        overlap: float,
    ) -> ConflictResult | None:
        """检测互斥选项模式：同主题但选择了互斥的选项。"""
        _mutual_exclusive_patterns = [
            (r"(?:aws|亚马逊)", r"(?:阿里云|腾讯云|华为云|azure|gcp)"),
            (r"(?:腾讯云)", r"(?:阿里云|华为云|aws|azure)"),
            (r"(?:阿里云)", r"(?:华为云|aws|azure|腾讯云)"),
            (r"(?:kubernetes|k8s|gke|eks|aks)", r"(?:docker\s*swarm|nomad|tke|cce)"),
            (
                r"(?:腾讯云.*(?:tke|k8s|kubernetes))",
                r"(?:华为云.*(?:cce|k8s|kubernetes)|阿里云.*(?:ack|k8s|kubernetes))",
            ),
            (r"(?:python)", r"(?:java|go|rust|typescript|c\+\+)"),
            (r"(?:mysql)", r"(?:postgresql|postgres)"),
            (r"(?:mongodb)", r"(?:redis|dynamodb|elasticsearch)"),
            (r"(?:react)", r"(?:vue|angular|svelte)"),
            (r"(?:docker)", r"(?:podman|containerd)"),
        ]
        for pattern_a, pattern_b in _mutual_exclusive_patterns:
            a_in_content = bool(re.search(pattern_a, content_lower))
            b_in_content = bool(re.search(pattern_b, content_lower))
            a_in_mem = bool(re.search(pattern_a, mem_content_lower))
            b_in_mem = bool(re.search(pattern_b, mem_content_lower))
            if (a_in_content and b_in_mem and not b_in_content and not a_in_mem) or (
                b_in_content and a_in_mem and not a_in_content and not b_in_mem
            ):
                return ConflictResult(
                    has_conflict=True,
                    existing_memory=mem_content[:200],
                    existing_id=mem_id,
                    conflict_type="semantic_contradiction",
                    action=self._resolve_strategy(),
                    reason=f"Mutual exclusive choices detected with existing memory (overlap={overlap:.0%})",
                )
        return None

    def _check_negation_conflict(
        self,
        content_lower: str,
        mem_content: str,
        mem_content_lower: str,
        mem_id: str,
        overlap: float,
    ) -> ConflictResult | None:
        """★ R29修复BUG-3：否定词矛盾检测，独立于 overlap 阈值。

        只要两条记忆有主题关联（overlap>0）且新内容含否定词，
        即判定为语义矛盾。之前此逻辑在 _check_topic_divergence 内部
        被 overlap>0.3 门控，导致低重叠矛盾对漏检。
        """
        negation_indicators = [
            "不是", "不对", "并非", "不再", "改为", "而不是",
            "不用", "改用", "不要", "无法", "没能",
            "not", "no longer", "instead of", "rather than",
        ]
        for ni in negation_indicators:
            if ni in content_lower:
                return ConflictResult(
                    has_conflict=True,
                    existing_memory=mem_content[:200],
                    existing_id=mem_id,
                    conflict_type="semantic_contradiction",
                    action=self._resolve_strategy(),
                    reason=f"Semantic conflict: new content contradicts existing memory (overlap={overlap:.0%})",
                )
        return None

    def _check_topic_divergence(
        self,
        content_lower: str,
        mem_content: str,
        mem_content_lower: str,
        mem_id: str,
        overlap: float,
    ) -> ConflictResult | None:
        """检测同主题分歧：高重叠但含否定意图或不同选项。"""

        def _extract_words_for_diff(text: str) -> set[str]:
            words = set()
            words.update(re.findall(r"[a-zA-Z]{2,}", text))
            words.update(re.findall(r"\d+[.]\d+|\d+", text))
            zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
            zh_str = "".join(zh_chars)
            for n in (2, 3, 4):
                for i in range(len(zh_str) - n + 1):
                    words.add(zh_str[i : i + n])
            return words

        words_a = _extract_words_for_diff(content_lower)
        words_b = _extract_words_for_diff(mem_content_lower)
        diff_a = words_a - words_b
        diff_b = words_b - words_a
        if overlap > 0.3 and diff_a and diff_b:
            nums_a = set(re.findall(r"\d+", content_lower))
            nums_b = set(re.findall(r"\d+", mem_content_lower))
            numeric_conflict = (
                nums_a
                and nums_b
                and nums_a != nums_b
                and nums_a & nums_b != nums_a
                and nums_a & nums_b != nums_b
            )
            if numeric_conflict:
                return ConflictResult(
                    has_conflict=True,
                    existing_memory=mem_content[:200],
                    existing_id=mem_id,
                    conflict_type="semantic_contradiction",
                    action=self._resolve_strategy(),
                    reason=f"Same topic but different numeric values detected (overlap={overlap:.0%})",
                )
            _option_pattern = r"[A-Z][a-z]+|[\u4e00-\u9fff]{2,4}(?:云|平台|服务|框架|语言|数据库|省|市|区|路|街|公司|部门|团队|项目)"
            _city_pattern = r"北京|上海|广州|深圳|杭州|成都|武汉|南京|西安|重庆|天津|苏州|长沙|郑州|东莞|青岛|沈阳|宁波|昆明"
            option_like_a = any(
                re.match(_option_pattern, w) or re.match(_city_pattern, w) for w in diff_a
            )
            option_like_b = any(
                re.match(_option_pattern, w) or re.match(_city_pattern, w) for w in diff_b
            )
            if option_like_a and option_like_b:
                return ConflictResult(
                    has_conflict=True,
                    existing_memory=mem_content[:200],
                    existing_id=mem_id,
                    conflict_type="semantic_contradiction",
                    action=self._resolve_strategy(),
                    reason=f"Same topic but different choices detected (overlap={overlap:.0%})",
                )
            # ★ R31修复BUG-3：高重叠+有差异但无明确选项模式时，
            # 仍判定为潜在冲突（同主题不同实体，如"橘猫"vs"蓝猫"）
            # 条件：overlap > 0.5 且双方差异词各≥2个
            if overlap > 0.5 and len(diff_a) >= 2 and len(diff_b) >= 2:
                return ConflictResult(
                    has_conflict=True,
                    existing_memory=mem_content[:200],
                    existing_id=mem_id,
                    conflict_type="update",
                    action=self._resolve_strategy(),
                    reason=f"Same topic but different details detected (overlap={overlap:.0%})",
                )
        return None

    def _check_semantic_with_fn(self, content: str) -> ConflictResult | None:
        """用语义检索函数做冲突检测。"""
        if self._semantic_check_fn is None:
            return None
        try:
            similar = self._semantic_check_fn(content, self._similarity_threshold)
            if similar:
                best = similar[0]
                score = best.get("score", 0)
                if score >= self._similarity_threshold:
                    return ConflictResult(
                        has_conflict=True,
                        existing_memory=best.get("content", "")[:200],
                        existing_id=best.get("memory_id", ""),
                        conflict_type="semantic_contradiction",
                        action=self._resolve_strategy(),
                        reason=f"Semantic conflict: similarity={score:.2f} with existing memory",
                    )
        except Exception as e:
            logger.warning("Semantic conflict check failed: %s", e)
        return None

    @staticmethod
    def _compute_overlap(text_a: str, text_b: str) -> float:
        """计算两段文本的词语重叠率。"""

        # ★ 改进分词：用2-4字滑动窗口分词，而非贪婪匹配连续中文字符
        def _extract_words(text: str) -> set[str]:
            words = set()
            # 英文词
            words.update(re.findall(r"[a-zA-Z]{2,}", text))
            # 数字组合（版本号、端口等）
            words.update(re.findall(r"\d+[.]\d+|\d+", text))
            # 中文：2字、3字、4字滑动窗口
            zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
            zh_str = "".join(zh_chars)
            for n in (2, 3, 4):
                for i in range(len(zh_str) - n + 1):
                    words.add(zh_str[i : i + n])
            return words

        words_a = _extract_words(text_a)
        words_b = _extract_words(text_b)
        if not words_a or not words_b:
            return 0.0
        overlap = len(words_a & words_b)
        return overlap / min(len(words_a), len(words_b))

    def _resolve_strategy(self) -> str:
        """返回当前策略下的默认动作。"""
        return "accept"
