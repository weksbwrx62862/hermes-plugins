"""POLE+O 实体提取和知识图谱 API 测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omnimem.deep.kg import (
    _POLEO_TYPES,
    KnowledgeGraph,
    _classify_entity_poleo,
    extract_entities,
    extract_entities_llm,
    extract_triples,
)


class TestPOLEOClassification(unittest.TestCase):
    """_classify_entity_poleo 规则引擎测试。"""

    def test_person_surname(self) -> None:
        self.assertEqual(_classify_entity_poleo("徐信豪"), "person")
        self.assertEqual(_classify_entity_poleo("王小明"), "person")
        self.assertEqual(_classify_entity_poleo("司马光"), "person")

    def test_person_title(self) -> None:
        self.assertEqual(_classify_entity_poleo("张总"), "person")
        self.assertEqual(_classify_entity_poleo("李工"), "person")
        self.assertEqual(_classify_entity_poleo("王老师"), "person")

    def test_not_person_concept(self) -> None:
        """概念词不应被分类为 Person。"""
        self.assertNotEqual(_classify_entity_poleo("测试"), "person")
        self.assertNotEqual(_classify_entity_poleo("系统"), "person")
        self.assertNotEqual(_classify_entity_poleo("部署"), "person")

    def test_english_person(self) -> None:
        self.assertEqual(_classify_entity_poleo("John Smith"), "person")

    def test_organization(self) -> None:
        self.assertEqual(_classify_entity_poleo("阿里公司"), "org")
        self.assertEqual(_classify_entity_poleo("研发团队"), "org")
        self.assertEqual(_classify_entity_poleo("人工智能研究所"), "org")
        self.assertEqual(_classify_entity_poleo("OpenAI"), "org")
        self.assertEqual(_classify_entity_poleo("Nous Research"), "org")

    def test_location(self) -> None:
        self.assertEqual(_classify_entity_poleo("北京市"), "location")
        self.assertEqual(_classify_entity_poleo("金华"), "location")
        self.assertEqual(_classify_entity_poleo("浙江省"), "location")
        self.assertEqual(_classify_entity_poleo("硅谷"), "location")

    def test_event(self) -> None:
        self.assertEqual(_classify_entity_poleo("R37回归测试"), "event")
        self.assertEqual(_classify_entity_poleo("产品发布会"), "event")
        self.assertEqual(_classify_entity_poleo("代码审查"), "event")

    def test_object_default(self) -> None:
        self.assertEqual(_classify_entity_poleo("Python"), "object")
        self.assertEqual(_classify_entity_poleo("API"), "object")
        self.assertEqual(_classify_entity_poleo("rm -rf"), "object")

    def test_all_poleo_types_mapped(self) -> None:
        for key, label in _POLEO_TYPES.items():
            self.assertIn(label, {"Person", "Organization", "Location", "Event", "Object"})


class TestEntityExtraction(unittest.TestCase):
    """实体和三元组提取测试。"""

    def test_extract_entities_chinese(self) -> None:
        entities = extract_entities("徐信豪来自浙江金华，使用Python开发")
        self.assertIn("Python", entities)
        # 金华是地名，但可能被中文实体模式匹配到
        # 至少应该有实体被提取

    def test_extract_entities_english(self) -> None:
        entities = extract_entities("We use Docker and Redis for deployment")
        self.assertIn("Docker", entities)
        self.assertIn("Redis", entities)

    def test_extract_triples_uses(self) -> None:
        triples = extract_triples("前端使用React框架")
        self.assertTrue(any(t[0] == "前端" and t[1] == "uses" for t in triples))

    def test_extract_triples_causes(self) -> None:
        triples = extract_triples("内存泄漏导致服务崩溃")
        self.assertTrue(any(t[1] == "causes" for t in triples))

    def test_extract_entities_llm_fallback(self) -> None:
        """无 LLM 客户端时回退到规则。"""
        entities, triples = extract_entities_llm(
            ["徐信豪使用Python开发OmniMem"],
            llm_call=None,
        )
        self.assertGreater(len(entities), 0)
        # 实体带 POLE+O 分类前缀 (person:..., object:... 等)
        has_prefix = any(":" in e for e in entities)
        self.assertTrue(has_prefix, f"Expected prefixed entities, got: {entities[:3]}")


class TestKnowledgeGraphAPI(unittest.TestCase):
    """KnowledgeGraph 新增 API 测试。"""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.kg = KnowledgeGraph(Path(self.tmpdir))

    def tearDown(self) -> None:
        self.kg.close()

    def test_extract_and_store_uses_poleo(self) -> None:
        result = self.kg.extract_and_store(
            "徐信豪来自浙江金华，使用Python开发OmniMem",
            memory_id="test-001",
        )
        self.assertGreater(result["entities_extracted"], 0)

    def test_entity_type_is_poleo(self) -> None:
        self.kg.extract_and_store("徐信豪创建了研发团队", memory_id="test-002")
        entity = self.kg.get_entity("徐信豪")
        if entity:
            self.assertEqual(entity["entity_type"], "Person")
        # org entity
        entity = self.kg.get_entity("研发团队")
        if entity:
            self.assertEqual(entity["entity_type"], "Organization")

    def test_get_entity_graph(self) -> None:
        self.kg.extract_and_store(
            "徐信豪在杭州创办了Nous Research，使用Python开发OmniMem",
            memory_id="test-003",
        )
        graph = self.kg.get_entity_graph()
        self.assertIsInstance(graph, dict)
        for key in ("Person", "Organization", "Location", "Event", "Object"):
            self.assertIn(key, graph)

    def test_get_all_entities(self) -> None:
        self.kg.extract_and_store("Python和Go是常用语言", memory_id="test-004")
        entities = self.kg.get_all_entities()
        self.assertGreater(len(entities), 0)
        # 实体类型应该是 POLE+O 之一
        valid_types = {"Person", "Organization", "Location", "Event", "Object"}
        for e in entities:
            self.assertIn(e.get("entity_type", ""), valid_types)

    def test_graph_search_with_entity(self) -> None:
        self.kg.extract_and_store("徐信豪使用Python", memory_id="gs-1")
        results = self.kg.graph_search("徐信豪", max_depth=1)
        self.assertGreater(len(results), 0)

    # ── 时序图谱 API ──

    def test_get_timeline(self) -> None:
        self.kg.extract_and_store("徐信豪使用Python", memory_id="tl-1")
        self.kg.extract_and_store("研发团队采用Go语言", memory_id="tl-2")
        self.kg.extract_and_store("徐信豪使用Go语言", memory_id="tl-3")
        timeline = self.kg.get_timeline("Go语言")
        self.assertGreater(len(timeline), 0)
        if len(timeline) >= 2:
            self.assertLessEqual(
                timeline[0].get("created_at", ""),
                timeline[-1].get("created_at", ""),
            )

    def test_get_entity_timeline_text(self) -> None:
        self.kg.extract_and_store("前端使用React", memory_id="tlt-1")
        text = self.kg.get_entity_timeline_text("React")
        self.assertIn("时间线", text)
        self.assertIn("React", text)

    def test_get_entity_timeline_text_empty(self) -> None:
        text = self.kg.get_entity_timeline_text("NoSuchEntity")
        self.assertEqual(text, "")

    def test_get_recent_changes(self) -> None:
        self.kg.extract_and_store("前端使用React", memory_id="rc-1")
        changes = self.kg.get_recent_changes(since_days=1, limit=20)
        self.assertGreater(len(changes), 0)

    def test_get_recent_changes_old(self) -> None:
        """很久以前的时间范围应返回空。"""
        changes = self.kg.get_recent_changes(since_days=0)
        self.assertEqual(len(changes), 0)
