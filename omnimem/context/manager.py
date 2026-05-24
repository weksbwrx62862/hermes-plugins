"""ContextManager — 上下文管理层：精炼/去重/预算控制/按需加载。

核心设计原则（来自 Anthropic managed-agents 文章）：
  1. 存储 ≠ 上下文注入 — 存储层可以全量保存，注入层必须精炼
  2. 预取只给摘要，细节按需拉取 — lazy provisioning
  3. 记忆是"牲口"不是"宠物" — 可以大胆合并/删除/重建，因为原始数据在 session log

三层架构:
  存储层（全量、不丢失）  → DrawerClosetStore / WingRoomManager
  上下文管理层（精炼、按需）→ 本文件 ContextManager
  上下文窗口（有限的、昂贵的）→ prefetch() 返回值 → Hermes 注入
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ContextBudget:
    """上下文预算配置。"""

    # 预取阶段最大 token 数（注入到上下文的摘要）
    max_prefetch_tokens: int = 300
    # 每条记忆摘要最大字符数
    max_summary_chars: int = 60       # L0：符号摘要（prefetch 注入）
    # L1：结构化概览（recall 返回）
    max_overview_chars: int = 200     # ★ 新增：L1 概览最大字符数
    # 预取最多返回多少条
    max_prefetch_items: int = 8
    # 按需拉取（omni_detail）最大字符数
    max_detail_chars: int = 500       # L2：完整原文（omni_detail 拉取）
    # 去重相似度阈值
    dedup_similarity_threshold: float = 0.7
    # ★ OPT: Mermaid 画布 token 预算比例（参考 TencentDB mmdMaxTokenRatio=0.2）
    mmd_max_token_ratio: float = 0.2
    # ★ OPT: 温和压缩触发比例（参考 TencentDB mildOffloadRatio=0.5）
    mild_offload_ratio: float = 0.5
    # ★ OPT: 激进压缩触发比例（参考 TencentDB aggressiveCompressRatio=0.85）
    aggressive_compress_ratio: float = 0.85


@dataclass
class RefinedItem:
    """精炼后的记忆条目。"""

    summary: str          # L0：符号摘要（≤60 字）
    memory_id: str
    memory_type: str
    confidence: float
    source_type: str = ""  # kv_cache / vector / bm25 / graph
    overview: str = ""     # ★ L1：结构化概览（≤200 字）
    # ★ OPT: 溯源链 node_id，支持从摘要下钻到原文
    trace_node_id: str = ""
    # ★ OPT: 底层原文引用路径
    ref_path: str = ""


class ContextManager:
    """上下文管理层：在存储层和上下文窗口之间做精炼、去重、预算控制。

    职责:
      1. 精炼 (Refine): 将原始记忆压缩为精简摘要再注入
      2. 去重 (Dedup): 同一信息的多次召回只保留最精炼的一条
      3. 预算控制 (Budget): 总注入量不超过配置上限
      4. 按需加载 (Lazy): 预取只给摘要，需要细节时通过 omni_detail 拉取
    """

    # ★ P1方案三：结构化压缩模板 — 将常见长句模式压缩为固定格式短摘要
    _COMPRESSION_TEMPLATES: list[tuple[str, str]] = [
        (r"用户(?:不喜欢|讨厌|反感|不爱)(.+?)(?:，|；|\.|$)", r"用户否定: \1"),
        (r"用户(?:喜欢|爱|偏好|钟爱)(.+?)(?:，|；|\.|$)", r"用户偏好: \1"),
        (r"用户(?:叫|称呼|让\s*叫)(.+?)(?:，|；|\.|$)", r"用户称呼: \1"),
        (r"纠正[:：]\s*(.+?)(?:，|；|\.|$)", r"纠正: \1"),
        (r"确认[:：]\s*(.+?)(?:，|；|\.|$)", r"确认: \1"),
        (r"(?:记住|记住[:：])\s*(.+?)(?:，|；|\.|$)", r"记住: \1"),
    ]

    def __init__(
        self,
        budget: ContextBudget | None = None,
        embedding_fn: Callable[[str], list[float]] | None = None,
    ):
        self._budget = budget or ContextBudget()
        # 本轮已注入的摘要指纹集合，防止重复注入
        self._injected_fingerprints: set[str] = set()
        # ★ 持久指纹：system_prompt_block 注入的指纹，跨轮保留
        #   防止 prefetch 重复注入 system_prompt_block 已经注入的记忆
        self._persistent_fingerprints: set[str] = set()
        # 本轮注入历史（用于 omni_detail 回溯）
        self._injected_items: list[RefinedItem] = []
        # ★ Embedding 语义去重：可选的 embedding 函数（接收 str 返回 List[float]）
        self._embedding_fn = embedding_fn
        # 指纹 → 摘要映射，供 embedding 去重时获取原文
        self._fp_to_summary: dict[str, str] = {}
        # embedding 向量本地缓存（避免同一轮内重复计算）
        self._embedding_cache: dict[str, list[float]] = {}

    def reset_for_new_turn(self) -> None:
        """每轮开始时重置注入状态。

        ★ 保留 _persistent_fingerprints（system_prompt_block 注入的），
        只清空本轮 prefetch 的 fingerprints。
        """
        self._injected_fingerprints = set(self._persistent_fingerprints)
        self._injected_items.clear()

    def add_persistent_fingerprint(self, fingerprint: str) -> None:
        """添加持久指纹（来自 system_prompt_block），跨轮保留。"""
        if fingerprint:
            self._persistent_fingerprints.add(fingerprint)
            self._injected_fingerprints.add(fingerprint)

    def get_injected_fingerprints(self) -> set[str]:
        """返回本轮已注入的摘要指纹集合（含持久指纹）。"""
        return set(self._injected_fingerprints)

    @property
    def max_summary_chars(self) -> int:
        """公共接口：获取摘要最大字符数。"""
        return self._budget.max_summary_chars

    # ─── 精炼 (Refine) ─────────────────────────────────────────

    @staticmethod
    def refine_content(raw_content: str, max_chars: int = 60) -> str:
        """将原始记忆内容精炼为摘要。

        策略:
          1. 如果内容已经是精炼事实（≤ max_chars），直接返回
          2. 如果含结构化标记（如 "CORRECTION:", "REINFORCED:"），提取核心
          3. 如果是整段对话，提取关键句子
          4. 最终回退到截断
        """
        content = raw_content.strip()

        # ★ 优先级0: 先剥离结构化标记（无论长度，这些标记是噪音不是语义）
        structured_prefixes = [
            (r"^CORRECTION:\s*", "纠正: "),
            (r"^REINFORCED:\s*", "确认: "),
            (r"^纠正:\s*", "纠正: "),
            (r"^确认:\s*", "确认: "),
            (r"^\[Pre-compression emergency save\]\s*", ""),
            (r"^\[Emergency save\]\s*", ""),
            (r"^\[Turn \d+\]\s*", ""),
            (r"^\[Checkpoint at turn \d+\]\s*", ""),
        ]
        for pattern, replacement in structured_prefixes:
            new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            if new_content != content:
                content = new_content.strip()
                break  # 只替换第一个匹配的前缀

        # 策略1: 如果内容已经是精炼事实（≤ max_chars），直接返回
        if len(content) <= max_chars:
            return content

        # ★ P1方案三：策略2 — 结构化模板压缩
        # 对常见模式（偏好/纠正/称呼）进行语义级压缩，避免截断切断关键条件
        for pattern, replacement in ContextManager._COMPRESSION_TEMPLATES:
            matched = re.search(pattern, content)
            if matched:
                compressed = re.sub(pattern, replacement, content, count=1)
                compressed = compressed.strip()
                if len(compressed) <= max_chars:
                    return compressed
                # 即使压缩后仍超长，也优先使用压缩版本再截断
                if len(compressed) < len(content):
                    content = compressed
                    break  # 只应用第一个匹配的模板

        # 策略3: 提取关键句子
        # 优先取含信号词的短句
        # ★ R19修复Minor-2: 先将真实换行符替换为空格，避免路径中\n被还原后导致split截断
        #   如 "C:\new\test" 中 \n 被还原为换行后，split 会在冒号后截断 summary
        content_for_split = content.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        sentences = re.split(r"[。！？.!?]", content_for_split)
        for s in sentences:
            s = s.strip()
            if 5 <= len(s) <= max_chars:
                return s

        # 策略4: 截断到 max_chars，尽量在句子边界截断
        # ★ 先替换换行符，避免截断后的 summary 在换行处断裂
        clean_content = content.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        truncated = clean_content[:max_chars]
        # 找最后一个句子结束符
        last_punct = max(
            truncated.rfind("。"),
            truncated.rfind("，"),
            truncated.rfind("."),
            truncated.rfind(","),
            truncated.rfind(" "),
        )
        if last_punct > max_chars // 2:
            truncated = truncated[:last_punct]
        return truncated.strip()

    @staticmethod
    def refine_overview(raw_content: str, max_chars: int = 200) -> str:
        """将原始记忆内容精炼为结构化概览（L1 层）。
        
        内化 OpenViking overview() 的设计：
        - L0 summary: 一句话核心事实（60 字）
        - L1 overview: 结构化概览，保留关键条件和上下文（200 字）
        - L2 full: 完整原文
        
        策略：
        1. 如果内容 ≤ max_chars，直接返回
        2. 提取关键句子（含条件/因果/时序标记的句子）
        3. 拼接关键句子，截断到 max_chars
        """
        content = raw_content.strip()
        if len(content) <= max_chars:
            return content
        
        # 剥离结构化标记
        for pattern, replacement in ContextManager._COMPRESSION_TEMPLATES:
            matched = re.search(pattern, content)
            if matched:
                compressed = re.sub(pattern, replacement, content, count=1)
                if len(compressed.strip()) <= max_chars:
                    return compressed.strip()
        
        # 提取关键句子：含信号词的句子优先
        content_for_split = content.replace("\n", " ").replace("\r", " ")
        sentences = re.split(r"[。！？.!?]", content_for_split)
        
        # 信号词权重
        signal_words = [
            "但是", "不过", "然而", "如果", "除非", "因为", "所以",
            "注意", "重要", "关键", "必须", "不要",
            "but", "however", "if", "because", "important", "note",
            "must", "should", "don't",
        ]
        
        scored_sentences = []
        for s in sentences:
            s = s.strip()
            if len(s) < 3:
                continue
            score = 0
            s_lower = s.lower()
            for sw in signal_words:
                if sw in s_lower:
                    score += 2
            scored_sentences.append((s, score))
        
        # 按信号词权重降序 + 原文顺序排列
        # 使用 enumerate 保持原始顺序索引
        for idx, (s, score) in enumerate(scored_sentences):
            pass  # 已经在列表中
        scored_sentences.sort(key=lambda x: (-x[1], content_for_split.find(x[0])))
        
        # 拼接到预算内
        result_parts = []
        used_chars = 0
        for s, _ in scored_sentences:
            if used_chars + len(s) + 1 <= max_chars:
                result_parts.append(s)
                used_chars += len(s) + 1
            else:
                break
        
        if not result_parts:
            # 回退：取前 max_chars
            return content_for_split[:max_chars].strip()
        
        return "。".join(result_parts) + "。"

    # ─── 去重 (Dedup) ─────────────────────────────────────────

    _SYNONYM_MAP: dict[str, str] = {}

    @classmethod
    def _load_synonym_map(cls) -> dict[str, str]:
        """从外部 JSON 加载同义词映射。

        加载策略：
          1. 尝试从 config/synonyms.json 加载
          2. 将一对多格式（key→list[str]）展开为扁平格式（word→key）
             即每个同义词值映射回其主词 key
          3. key 自身也映射到自身
          4. 加载失败时回退到空字典并记录 warning
        """
        result: dict[str, str] = {}
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "synonyms.json"
        )
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                external = json.load(f)
            if isinstance(external, dict):
                for key, val in external.items():
                    result[key] = key
                    if isinstance(val, list):
                        for v in val:
                            if isinstance(v, str) and v not in result:
                                result[v] = key
                    elif isinstance(val, str) and val not in result:
                        result[val] = key
        except FileNotFoundError:
            logger.warning("synonyms.json not found at %s", config_path)
        except Exception:
            logger.warning("Failed to load synonyms.json from %s", config_path)
        return result

    # 分词词典：从 _SYNONYM_MAP 的 key 中提取，按长度降序排列
    # 用于基于词典的最大匹配分词
    _DICT_WORDS: list[str] | None = None
    _DICT_SET: set[str] | None = None

    @classmethod
    def _get_dict_words(cls) -> list[str]:
        """获取分词词典（惰性初始化，按词长降序排列）。"""
        if cls._DICT_WORDS is None:
            if not cls._SYNONYM_MAP:
                cls._SYNONYM_MAP = cls._load_synonym_map()
            all_words = set(cls._SYNONYM_MAP.keys())
            # 按长度降序排列（长词优先匹配）
            cls._DICT_WORDS = sorted(all_words, key=len, reverse=True)
            cls._DICT_SET = set(cls._DICT_WORDS)
        return cls._DICT_WORDS

    @classmethod
    def _get_dict_set(cls) -> set[str]:
        """获取分词词典集合（惰性初始化，用于 O(1) 快速查找）。"""
        if cls._DICT_SET is None:
            cls._get_dict_words()
        assert cls._DICT_SET is not None
        return cls._DICT_SET

    @classmethod
    def _normalize_word(cls, word: str) -> str:
        """将词归一化：先查同义映射，再返回原词。"""
        if not cls._SYNONYM_MAP:
            cls._SYNONYM_MAP = cls._load_synonym_map()
        syn = cls._SYNONYM_MAP.get(word)
        if isinstance(syn, list):
            return syn[0] if syn else word
        if isinstance(syn, str):
            return syn
        return word

    @classmethod
    def _tokenize_chinese(cls, content: str) -> list[str]:
        """基于词典的最大匹配中文分词。

        策略：
        1. 先按标点/分隔符切分片段
        2. 对每个片段用词典最大匹配分词
        3. 未匹配的中文连续串保留为2-3字块
        4. 对2字块尝试二次拆分（如"我喜"→跳过"我"→匹配"喜欢"）
        5. 提取英文词
        """
        tokens = []

        # 1. 提取英文词（2+字）
        en_words = re.findall(r"[a-zA-Z]{2,}", content)
        for w in en_words:
            tokens.append(w.lower())

        # 2. 按分隔符切分中文片段
        # 分隔符: 冒号、逗号、空格、括号等
        segments = re.split(r"[:：，,；;\s\[\]【】\(\)（）·]", content)

        dict_words = cls._get_dict_words()
        dict_set = cls._get_dict_set()  # 复用类级别缓存集合

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            # 移除英文部分（已提取）
            zh_only = re.sub(r"[a-zA-Z0-9]", "", segment)
            if not zh_only:
                continue

            # 对这个片段做基于词典的最大匹配
            i = 0
            while i < len(zh_only):
                matched = False
                # 尝试从最长词开始匹配
                for word in dict_words:
                    # 只用纯中文词典词
                    if not all("\u4e00" <= c <= "\u9fff" for c in word):
                        continue
                    if zh_only[i : i + len(word)] == word:
                        tokens.append(word)
                        i += len(word)
                        matched = True
                        break
                if not matched:
                    # 未匹配到词典词
                    remaining = len(zh_only) - i
                    if remaining >= 3:
                        # 先尝试3字块中是否包含2字词典词
                        chunk3 = zh_only[i : i + 3]
                        found_sub = False
                        # 尝试位置0的2字词
                        if chunk3[:2] in dict_set:
                            tokens.append(chunk3[:2])
                            # 第3字跳过（通常是助词）
                            i += 3
                            found_sub = True
                        # 尝试位置1的2字词
                        elif not found_sub and chunk3[1:] in dict_set:
                            tokens.append(chunk3[1:])
                            i += 3
                            found_sub = True
                        if not found_sub:
                            tokens.append(chunk3)
                            i += 3
                    elif remaining >= 2:
                        chunk2 = zh_only[i : i + 2]
                        # 尝试拆2字块中的词典词
                        if chunk2[1:] in dict_set:
                            # 第2字是词典词（如 "我喜" → 跳过"我"，保留"喜"... 但"喜"不是词）
                            # 更好的方式：看第2字开头的2字词
                            tokens.append(chunk2[1:])
                            i += 2
                        elif chunk2[:1] in dict_set:
                            tokens.append(chunk2[:1])
                            i += 2
                        else:
                            tokens.append(chunk2)
                            i += 2
                    else:
                        # 剩余1字 — 保留非停用词单字（如"猫"、"徐"等实体词）
                        ch = zh_only[i]
                        if ch not in (
                            "的",
                            "了",
                            "是",
                            "在",
                            "我",
                            "你",
                            "他",
                            "她",
                            "它",
                            "们",
                            "这",
                            "那",
                            "有",
                            "不",
                            "也",
                            "都",
                            "就",
                            "还",
                            "会",
                            "能",
                            "要",
                            "和",
                            "与",
                            "或",
                            "而",
                            "但",
                            "被",
                            "把",
                            "给",
                            "让",
                            "从",
                            "到",
                            "用",
                            "对",
                            "好",
                            "很",
                            "以",
                            "为",
                            "着",
                            "过",
                            "吧",
                            "呢",
                            "啊",
                            "呀",
                            "嘛",
                            "着",
                            "过",
                            "了",
                        ):
                            tokens.append(ch)
                        i += 1

        return tokens

    @classmethod
    def _content_fingerprint(cls, content: str) -> str:
        """生成内容指纹，用于去重。基于关键词集合（含语义归一化）。

        改进：
        1. 使用基于词典的最大匹配分词，而非简单正则
        2. 归一化同义词
        3. 去除低信息量词（辅助语气词、通用停用词）
        """
        # ★ 缓存层：相同内容直接返回
        return cls._cached_fingerprint(content)

    @classmethod
    @lru_cache(maxsize=512)
    def _cached_fingerprint(cls, content: str) -> str:
        """带缓存的指纹计算。lru_cache 自动去重，相同内容 O(1)。"""
        raw_words = cls._tokenize_chinese(content)

        # 归一化同义词
        normalized = set()
        for w in raw_words:
            nw = cls._normalize_word(w)
            if nw:
                normalized.add(nw)

        # 去除归一化后的低信息量词（辅助语气词、通用停用词）
        low_info = {
            "偏好偏好",
            "对",
            "好",
            "很",
            "就",
            "也",
            "都",
            "还",
            "又",
            "才",
            "再",
            "就行",
            "就好",
            "可以了",
            "吧",
            "呢",
            "啊",
            "呀",
            "嘛",
            "着",
            "过",
            "了",
        }
        normalized -= low_info

        return "|".join(sorted(normalized))

    @classmethod
    def _fingerprint_similarity(cls, fp1: str, fp2: str) -> float:
        """两个指纹之间的相似度。带缓存。"""
        # 确保参数顺序一致（对称性），提高缓存命中率
        key = (fp1, fp2) if fp1 <= fp2 else (fp2, fp1)
        return cls._cached_similarity(key[0], key[1])

    @classmethod
    @lru_cache(maxsize=1024)
    def _cached_similarity(cls, fp1: str, fp2: str) -> float:
        """带缓存的指纹相似度计算。

        三级检测:
        1. 子集检测: A⊆B 或 B⊆A → 基于有效覆盖率
        2. 语义归一化后 Jaccard
        3. 宽松覆盖检测: 较短指纹的有效词覆盖率
        """
        if not fp1 or not fp2:
            return 0.0
        words1 = set(fp1.split("|"))
        words2 = set(fp2.split("|"))

        # 去除空串
        words1 = {w for w in words1 if w}
        words2 = {w for w in words2 if w}
        if not words1 or not words2:
            return 0.0

        # 完全相等 → 返回 1.0（必须在子集检测之前）
        if words1 == words2:
            return 1.0

        # 策略1: 子集检测 — 如果短的是长的子集，按覆盖率给高分
        if words1 <= words2:
            # words1 是 words2 的子集
            coverage = len(words1) / len(words2)
            # 子集关系本身就有强信号，给予基础分 + 覆盖奖励
            # coverage >= 0.5 → 0.85; coverage >= 0.33 → 0.75; coverage >= 0.25 → 0.72
            if coverage >= 0.5:
                return 0.85
            elif coverage >= 0.33:
                return 0.75
            elif len(words1) >= 2:
                # 至少2个词完全匹配子集关系
                return 0.72
        if words2 <= words1:
            coverage = len(words2) / len(words1)
            if coverage >= 0.5:
                return 0.85
            elif coverage >= 0.33:
                return 0.75
            elif len(words2) >= 2:
                return 0.72

        # 策略2: 标准 Jaccard
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0.0

        if jaccard >= 0.7:
            return jaccard

        # 策略3: 宽松覆盖检测 — 较短指纹的有效覆盖率
        # 场景: "用户姓名|徐信豪" vs "用户姓名|徐信豪|称呼|老板"
        # Jaccard = 2/4 = 0.5 < 0.7, 但较短指纹的2个词全在较长指纹中
        shorter, longer = (words1, words2) if len(words1) <= len(words2) else (words2, words1)
        overlap = len(shorter & longer)
        coverage_of_shorter = overlap / len(shorter) if shorter else 0.0

        # 如果较短指纹的大部分词都在较长指纹中，视为高相似
        if coverage_of_shorter >= 0.7 and overlap >= 2:
            # 有效覆盖率 × 调整系数
            return max(jaccard, coverage_of_shorter * 0.9)

        return jaccard

    def _is_duplicate(self, item: RefinedItem) -> bool:
        """检查是否与本轮已注入的记忆重复。

        两级去重：
          1. 快速路径：Jaccard 相似度 > 0.7（原有逻辑）
          2. 慢路径：Jaccard 在 0.4-0.7 之间时，使用 Embedding 语义相似度
        """
        new_fp = self._content_fingerprint(item.summary)
        if not new_fp:
            return False

        for existing_fp in self._injected_fingerprints:
            fp_sim = self._fingerprint_similarity(new_fp, existing_fp)
            if fp_sim > self._budget.dedup_similarity_threshold:
                return True

            # ★ Embedding 语义去重慢路径：中等 Jaccard 时用向量语义判断
            if 0.4 < fp_sim <= self._budget.dedup_similarity_threshold:
                if self._embedding_fn is not None:
                    emb_sim = self._embedding_similarity(
                        item.summary,
                        self._fp_to_summary.get(existing_fp, ""),
                    )
                    if emb_sim > 0.92:
                        logger.warning(
                            "Embedding dedup: '%s' ~ '%s' (emb_sim=%.3f)",
                            item.summary[:30],
                            self._fp_to_summary.get(existing_fp, "")[:30],
                            emb_sim,
                        )
                        return True
        return False

    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """计算两条文本的 embedding 余弦相似度（带本地缓存）。"""
        if not text1 or not text2 or self._embedding_fn is None:
            return 0.0

        # 从缓存获取或计算 embedding
        vec1 = self._embedding_cache.get(text1)
        if vec1 is None:
            vec1 = self._embedding_fn(text1)
            if vec1:
                self._embedding_cache[text1] = vec1

        vec2 = self._embedding_cache.get(text2)
        if vec2 is None:
            vec2 = self._embedding_fn(text2)
            if vec2:
                self._embedding_cache[text2] = vec2

        if not vec1 or not vec2:
            return 0.0

        # 余弦相似度
        dot = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    # ─── 核心方法 ─────────────────────────────────────────────

    def refine_prefetch_results(
        self,
        raw_results: list[dict[str, Any]],
    ) -> str:
        """将原始检索结果精炼后格式化为注入文本。

        这是 prefetch() 的唯一出口 — 所有注入到上下文的记忆都经过此方法。

        返回格式:
          ### Relevant Memories
          - [fact] 用户姓名: 徐信豪
          - [preference] 称呼偏好: 老板
        """
        if not raw_results:
            return ""

        refined_items: list[RefinedItem] = []
        total_chars = 0

        for r in raw_results:
            raw_content = r.get("content", "")
            if not raw_content:
                continue

            # 1. 精炼
            summary = self.refine_content(raw_content, self._budget.max_summary_chars)
            item = RefinedItem(
                summary=summary,
                memory_id=r.get("memory_id", ""),
                memory_type=r.get("type", "fact"),
                confidence=r.get("confidence", 0),
                source_type=r.get("source_type", ""),
            )

            # 2. 去重
            if self._is_duplicate(item):
                logger.warning("ContextManager: skipped duplicate '%s'", summary[:30])
                continue

            # 3. 预算检查
            estimated_chars = len(summary) + 20  # 格式开销
            if total_chars + estimated_chars > self._budget.max_prefetch_tokens:
                logger.warning(
                    "ContextManager: budget exceeded, stopping at %d items", len(refined_items)
                )
                break
            if len(refined_items) >= self._budget.max_prefetch_items:
                break

            # 通过所有检查，加入注入列表
            refined_items.append(item)
            fp = self._content_fingerprint(summary)
            if fp:
                self._injected_fingerprints.add(fp)
                self._fp_to_summary[fp] = summary
            self._injected_items.append(item)
            total_chars += estimated_chars

        if not refined_items:
            return ""

        # 格式化输出 — 精简版：不含 confidence 等冗余信息
        parts = ["### Relevant Memories"]
        for item in refined_items:
            parts.append(f"- [{item.memory_type}] {item.summary}")
        return "\n".join(parts)

    def refine_recall_results(
        self,
        raw_results: list[dict[str, Any]],
        max_tokens: int = 1500,
    ) -> list[dict[str, Any]]:
        """将 omni_recall 的原始检索结果精炼后返回。

        与 prefetch 不同，recall 是 Agent 主动调用的，可以返回更多信息。
        但仍然做去重和精炼，只是预算更宽松。
        使用与 prefetch 相同的相似度去重算法（而非精确指纹匹配）。
        """
        if not raw_results:
            return []

        refined = []
        seen_fps: list[str] = []  # 改为列表，以支持相似度比较

        for r in raw_results:
            raw_content = r.get("content", "")
            if not raw_content:
                continue

            summary = self.refine_content(raw_content, max_chars=100)
            fp = self._content_fingerprint(summary)

            # 语义相似度去重（与 prefetch 统一）
            is_dup = False
            if fp:
                for existing_fp in seen_fps:
                    if (
                        self._fingerprint_similarity(fp, existing_fp)
                        > self._budget.dedup_similarity_threshold
                    ):
                        is_dup = True
                        break
                if not is_dup:
                    seen_fps.append(fp)
            if is_dup:
                continue

            refined.append(
                {
                    "content": summary,
                    "original_content": raw_content,  # 保留原文供 omni_detail 使用
                    "type": r.get("type", "fact"),
                    "confidence": r.get("confidence"),
                    "memory_id": r.get("memory_id", ""),
                    "wing": r.get("wing"),
                    "room": r.get("room"),
                    "stored_at": r.get("stored_at"),
                }
            )

        return refined

    # ─── 按需加载 (Lazy Provisioning) ─────────────────────────

    def get_detail_for(self, memory_id: str, store: Any) -> dict[str, Any]:
        """按需拉取某条记忆的完整细节。

        预取阶段只注入摘要，Agent 需要细节时调用此方法。
        类似 Anthropic 的 getEvents() — 不预创建沙箱，按需获取。
        """
        # 从注入历史中查找
        for item in self._injected_items:
            if item.memory_id == memory_id:
                # 从存储层拉取完整内容
                full = store.get(memory_id)
                if full:
                    return {
                        "status": "found",
                        "memory_id": memory_id,
                        "summary": item.summary,
                        "full_content": full.get("content", ""),
                        "type": item.memory_type,
                        "confidence": item.confidence,
                        "privacy": full.get("privacy", "personal"),
                        "metadata": full.get("metadata", {}),
                    }
                break

        # 不在注入历史中，直接从存储层查
        full = store.get(memory_id)
        if full:
            return {
                "status": "found",
                "memory_id": memory_id,
                "summary": self.refine_content(full.get("content", "")),
                "full_content": full.get("content", ""),
                "type": full.get("type", "fact"),
                "confidence": full.get("confidence", 0),
                "privacy": full.get("privacy", "personal"),
                "metadata": full.get("metadata", {}),
            }

        return {"status": "not_found", "memory_id": memory_id}

    def get_injected_items(self) -> list[dict[str, str]]:
        """返回本轮已注入的记忆列表（供 omni_detail 列出可查细节的记忆）。"""
        return [
            {
                "memory_id": item.memory_id,
                "summary": item.summary,
                "type": item.memory_type,
            }
            for item in self._injected_items
            if item.memory_id
        ]
