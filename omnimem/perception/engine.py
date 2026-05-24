"""PerceptionEngine — L0 感知层：信号检测 + 意图预测。

参考 memU 的主动式感知和 ReMe 的会话监听设计。

信号类型:
  - correction: 用户纠正（"不对"/"错了"/"不是这样"）
  - reinforcement: 正反馈（"对"/"很好"/"就是这样"）
  - should_memorize: 值得记忆的信息
  - preference: 用户偏好

意图预测:
  - 根据当前消息预测下一步查询，用于 prefetch 预加载
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerceptionSignals:
    """感知信号集合。"""

    has_correction: bool = False
    has_reinforcement: bool = False
    should_memorize: bool = False
    has_preference: bool = False
    correction_target: str = ""
    reinforcement_target: str = ""
    fact_content: str = ""
    predicted_intent: str = ""


class PerceptionEngine:
    """L0 感知引擎：信号检测 + 意图预测。"""

    # 纠正标记 — 移除"不是"（会误匹配问句"不是...吗"），改用更明确的模式
    _CORRECTION_MARKERS = [
        "不对",
        "错了",
        "不是这样",
        "纠正",
        "更正",
        "不是这样",
        "我说的不是",
        "应该是",
        "我说的是",
        "wrong",
        "incorrect",
        "no,",
        "not like that",
        "actually",
        "that's wrong",
        "I meant",
    ]

    # 正反馈标记
    _REINFORCEMENT_MARKERS = [
        "对",
        "很好",
        "没错",
        "就是这样",
        "正确",
        "yes",
        "correct",
        "right",
        "exactly",
        "good",
        "perfect",
        "great",
    ]

    # 偏好标记
    _PREFERENCE_MARKERS = [
        "我喜欢",
        "我不喜欢",
        "偏好",
        "更希望",
        "I prefer",
        "I like",
        "I don't like",
        "I'd rather",
        "my preference",
    ]

    # 值得记忆的标记
    _MEMORABLE_MARKERS = [
        "记住",
        "记下",
        "记住这个",
        "别忘了",
        "remember",
        "note that",
        "keep in mind",
        "don't forget",
        "important",
        "重要",
    ]

    # ★ 事实提炼规则：从原文中提取精简事实而非存整段原文
    # 预编译正则以避免运行时重复编译/缓存查找开销
    _EXTRACTION_PATTERNS = [
        # 偏好提取: "我喜欢X" → "偏好: X"
        (
            re.compile(
                r"(?:我喜欢|偏好|更希望|习惯|总是)(.{2,30}?)(?:[，。！？,.!?\n]|$)", re.IGNORECASE
            ),
            "偏好",
        ),
        # 纠正提取: "不对，应该是X" → "纠正: X"
        (
            re.compile(
                r"(?:不对|错了|不是|纠正|更正)[，,]?\s*(?:应该是|应该是|改用|改为)?(.{2,40}?)(?:[，。！？,.!?\n]|$)",
                re.IGNORECASE,
            ),
            "纠正",
        ),
        # 称呼偏好: "叫我X" / "称呼X" → "称呼偏好: X"（放在姓名前，优先匹配更具体的）
        (
            re.compile(
                r"(?:叫我|称呼|称我为?)\s*(.+?)(?:就行|就好|可以了|吧|了|着|过|呢|啊|呀|嘛|[，。！？,.!?\n]|$)",
                re.IGNORECASE,
            ),
            "称呼偏好",
        ),
        # 姓名提取: "我姓X" / "我叫X" → "姓名: X"
        (
            re.compile(
                r"(?:我姓|我叫|名字是|姓名是?)\s*(.{1,20}?)(?:[，。！？,.!?\n]|$)", re.IGNORECASE
            ),
            "姓名",
        ),
        # 显式记忆: "记住X" → "X"
        (
            re.compile(
                r"(?:记住|记下|别忘了|remember)\s*(.{2,50}?)(?:[，。！？,.!?\n]|$)", re.IGNORECASE
            ),
            "",
        ),
    ]

    def detect_signals(self, user_content: str, assistant_content: str = "") -> PerceptionSignals:
        """检测对话中的感知信号。

        Args:
            user_content: 用户消息
            assistant_content: 助手消息

        Returns:
            PerceptionSignals
        """
        signals = PerceptionSignals()

        is_injection, is_echo = self._check_garbage(user_content, assistant_content)
        self._check_volume(user_content, assistant_content, signals)
        self._check_keywords(user_content, signals)

        # 注入内容不自动记忆
        if is_injection and not signals.has_correction and not signals.has_preference:
            has_explicit = any(m.lower() in user_content.lower() for m in self._MEMORABLE_MARKERS)
            if not has_explicit:
                signals.should_memorize = False

        # ★ AI 回复 echo 防护：如果 AI 回复包含注入记忆格式，
        # 不自动触发 should_memorize（除非有明确的纠正/偏好信号）
        if is_echo and not signals.has_correction and not signals.has_preference:
            has_explicit = any(m.lower() in user_content.lower() for m in self._MEMORABLE_MARKERS)
            if not has_explicit:
                signals.should_memorize = False

        # 提取事实内容
        if signals.should_memorize:
            signals.fact_content = self._extract_core_fact(user_content)

        return signals

    def _check_garbage(self, user_content: str, assistant_content: str) -> tuple[bool, bool]:
        """系统注入检测 + AI 回复 echo 防护。

        Returns:
            (is_injection, is_echo) 元组
        """
        is_injection = "### Relevant Memories" in user_content or "[cached]" in user_content

        is_echo = False
        if assistant_content:
            echo_markers = [
                "### Relevant Memories",
                "- [fact]",
                "- [preference]",
                "- [correction]",
                "- [cached]",
            ]
            is_echo = any(m in assistant_content for m in echo_markers)

        return is_injection, is_echo

    def _check_volume(
        self, user_content: str, assistant_content: str, signals: PerceptionSignals
    ) -> None:
        """纠正类信号检测（纠正 + 模糊纠正）。

        直接修改 signals 对象。
        """
        is_question = user_content.rstrip().endswith(("吗", "？", "?", "么"))
        if not is_question:
            for marker in self._CORRECTION_MARKERS:
                if marker.lower() in user_content.lower():
                    signals.has_correction = True
                    signals.should_memorize = True
                    raw_target = user_content
                    signals.correction_target = self._extract_core_fact(raw_target)
                    break

        # 模糊纠正检测 — 结合 assistant_content 判断
        if not signals.has_correction and not is_question:
            vague_correction_markers = ["换个", "重做", "不对", "换一个", "不是这个", "不要这个"]
            for marker in vague_correction_markers:
                if marker in user_content and assistant_content:
                    signals.has_correction = True
                    signals.should_memorize = True
                    signals.correction_target = self._extract_core_fact(user_content)
                    break

    def _check_keywords(self, user_content: str, signals: PerceptionSignals) -> None:
        """关键词类信号检测（正反馈 / 偏好 / 姓名 / 值得记忆）。

        直接修改 signals 对象。
        """
        # 正反馈检测 — 单字词需词边界检查
        for marker in self._REINFORCEMENT_MARKERS:
            if len(marker) <= 1:
                if re.search(
                    r"(?:^|[，,\n\s])"
                    + re.escape(marker)
                    + r"(?:[，,。.！!？?了着过的呢吧啊呀嘛\s]|$)",
                    user_content,
                ):
                    signals.has_reinforcement = True
                    signals.reinforcement_target = self._extract_core_fact(user_content)
                    break
            elif marker.lower() in user_content.lower():
                signals.has_reinforcement = True
                signals.reinforcement_target = self._extract_core_fact(user_content)
                break

        # 偏好检测 — 独立于 reinforcement
        if not signals.has_reinforcement:
            for marker in self._PREFERENCE_MARKERS:
                if marker.lower() in user_content.lower():
                    signals.has_preference = True
                    signals.should_memorize = True
                    break

        # 姓名/称呼检测
        for pattern, label in self._EXTRACTION_PATTERNS:
            if pattern.search(user_content) and label in ("姓名", "称呼偏好"):
                signals.has_preference = True
                signals.should_memorize = True
                break

        # 值得记忆的检测
        for marker in self._MEMORABLE_MARKERS:
            if marker.lower() in user_content.lower():
                signals.should_memorize = True
                break

    def _extract_core_fact(self, text: str) -> str:
        """从原文中提取精简核心事实。

        策略优先级:
        1. 用模式匹配提取结构化事实（偏好/纠正/姓名等）
        2. 回退到提取关键句子（含信号词的句子）
        3. 最终回退到截断（但限制在 100 字以内）
        """
        # 策略1: 模式匹配提取
        for pattern, label in self._EXTRACTION_PATTERNS:
            m = pattern.search(text)
            if m:
                extracted = m.group(1).strip()
                # 姓名类允许1个字（如"徐"），其他至少2字
                min_len = 1 if label == "姓名" else 2
                if extracted and len(extracted) >= min_len:
                    return f"{label}: {extracted}" if label else extracted

        # 策略2: 提取含信号词的句子
        sentences = re.split(r"[。！？.!?;\n]", text)
        for s in sentences:
            s = s.strip()
            if len(s) < 5 or len(s) > 100:
                continue
            # 包含任何信号关键词的句子
            all_markers = (
                self._PREFERENCE_MARKERS
                + self._MEMORABLE_MARKERS
                + self._CORRECTION_MARKERS[:6]
                + self._REINFORCEMENT_MARKERS[:6]
            )
            if any(m.lower() in s.lower() for m in all_markers):
                return s[:100]

        # 策略3: 截断，但限制在 100 字以内
        return text[:100].strip()

    def predict_intent(self, message: str) -> str:
        """意图预测：根据当前消息预测下一步查询。

        简单实现：提取消息中的关键词/实体作为预检索查询。
        """
        # 提取问号前的内容（问号在末尾时也能正确提取）
        question_match = re.search(r"(.+?)[？?]$", message.strip())
        if question_match:
            return question_match.group(1).strip()
        # 非末尾问号：提取问号后的子问题
        questions = re.findall(r"[？?](.+)$", message, re.MULTILINE)
        if questions:
            return str(questions[0]).strip()

        # 提取关键实体
        entities = self._extract_entities(message)
        if entities:
            return " ".join(entities[:3])

        # 回退：使用消息前100字符
        return message[:100].strip()

    def extract_implicit_memories(self, content: str) -> list[str]:
        """从内容中提取隐含的记忆点。

        用于 on_session_end 时从完整对话中提取遗漏的记忆。
        """
        memories = []
        sentences = re.split(r"[。！？.!?\n]", content)

        for s in sentences:
            s = s.strip()
            if len(s) < 10:
                continue
            # 包含偏好/决定/纠正的句子
            if any(m in s for m in self._PREFERENCE_MARKERS + self._MEMORABLE_MARKERS):
                memories.append(s)

        return memories

    def close(self) -> None:
        pass

    def _extract_entities(self, text: str) -> list[str]:
        """提取文本中的关键实体。"""
        entities = []
        # 中文实体
        zh = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
        entities.extend(zh[:3])
        # 英文实体
        en = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        entities.extend(en[:3])
        return entities
