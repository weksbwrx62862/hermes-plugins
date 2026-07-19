"""实体提取增强测试：裸中文人名识别。"""

from __future__ import annotations

import unittest

from omnimem.deep.kg import extract_entities


class TestBareNameExtraction(unittest.TestCase):

    def test_bare_name_without_context(self) -> None:
        """裸中文名（无上下文标记）应被识别。"""
        entities = extract_entities("徐信豪在杭州创办了 Nous Research")
        self.assertIn("徐信豪", entities)

    def test_common_surnames(self) -> None:
        entities = extract_entities("王小明和李华参加了会议")
        self.assertIn("王小明", entities)

    def test_name_in_sentence(self) -> None:
        entities = extract_entities("项目经理张伟负责代码审查")
        self.assertIn("张伟", entities)

    def test_concept_not_name(self) -> None:
        """概念词不应被误识别为人名。"""
        entities = extract_entities("系统测试完成后部署上线")
        self.assertNotIn("测试", entities)
        self.assertNotIn("部署", entities)

    def test_name_with_tech_terms(self) -> None:
        """人名与技术术语混合时应正确识别。"""
        entities = extract_entities("赵工使用Python开发OmniMem")
        self.assertIn("Python", entities)
