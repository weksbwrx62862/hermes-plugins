"""L0 感知引擎模块测试。"""

from __future__ import annotations

import unittest

from omnimem.perception.engine import PerceptionEngine


class TestPerceptionEngine(unittest.TestCase):
    """PerceptionEngine 测试。"""

    def setUp(self) -> None:
        self.pe = PerceptionEngine()

    def test_detect_correction(self) -> None:
        signals = self.pe.detect_signals("不对，应该是Python")
        self.assertTrue(signals.has_correction)
        self.assertTrue(signals.should_memorize)

    def test_detect_reinforcement(self) -> None:
        signals = self.pe.detect_signals("对，就是这样")
        self.assertTrue(signals.has_reinforcement)

    def test_detect_preference(self) -> None:
        signals = self.pe.detect_signals("我喜欢暗色主题")
        self.assertTrue(signals.has_preference)
        self.assertTrue(signals.should_memorize)

    def test_detect_memorable(self) -> None:
        signals = self.pe.detect_signals("记住这个重要信息")
        self.assertTrue(signals.should_memorize)

    def test_no_signal_for_plain_text(self) -> None:
        signals = self.pe.detect_signals("今天天气不错")
        self.assertFalse(signals.has_correction)
        self.assertFalse(signals.has_reinforcement)

    def test_correction_question_not_triggered(self) -> None:
        signals = self.pe.detect_signals("这样不对吗？")
        self.assertFalse(signals.has_correction)

    def test_injection_content_not_memorized(self) -> None:
        signals = self.pe.detect_signals("### Relevant Memories\n- [fact] 测试")
        self.assertFalse(signals.should_memorize)

    def test_extract_core_fact_preference(self) -> None:
        result = self.pe._extract_core_fact("我喜欢Python编程")
        self.assertIn("偏好", result)
        self.assertIn("Python", result)

    def test_extract_core_fact_name(self) -> None:
        result = self.pe._extract_core_fact("我叫徐信豪")
        self.assertIn("姓名", result)
        self.assertIn("徐信豪", result)

    def test_extract_core_fact_correction(self) -> None:
        result = self.pe._extract_core_fact("不对，应该是Python")
        self.assertIn("纠正", result)

    def test_predict_intent_returns_string(self) -> None:
        result = self.pe.predict_intent("什么是量子计算？")
        self.assertIsInstance(result, str)
        self.assertIn("量子计算", result)

    def test_predict_intent_with_entities(self) -> None:
        result = self.pe.predict_intent("Docker和Kubernetes的区别")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_extract_implicit_memories(self) -> None:
        result = self.pe.extract_implicit_memories("我喜欢简洁的代码风格。记住要用类型提示。")
        self.assertTrue(len(result) > 0)
