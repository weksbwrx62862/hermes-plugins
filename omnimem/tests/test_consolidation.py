import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from omnimem.deep.consolidation import (
    ConsolidatedItem,
    ConsolidationEngine,
    ConsolidationResult,
    _cluster_by_topic,
    _extract_keywords,
    _generate_mental_model,
    _generate_observation,
)


class TestExtractKeywords:
    def test_chinese_keywords(self):
        texts = ["\u7528\u6237\u559c\u6b22Python\u7f16\u7a0b", "\u7528\u6237\u504f\u597dPython\u5f00\u53d1"]
        kws = _extract_keywords(texts, top_k=5)
        assert isinstance(kws, list)
        assert len(kws) > 0

    def test_empty_input(self):
        kws = _extract_keywords([], top_k=5)
        assert kws == []

    def test_english_keywords(self):
        texts = ["Python programming language", "Python web development"]
        kws = _extract_keywords(texts, top_k=5)
        assert isinstance(kws, list)


class TestClusterByTopic:
    def test_basic_clustering(self):
        facts = [
            {"content": "\u7528\u6237\u559c\u6b22Python\u7f16\u7a0b"},
            {"content": "\u7528\u6237\u504f\u597dPython\u5f00\u53d1"},
            {"content": "\u732b\u559c\u6b22\u5403\u9c7c"},
        ]
        clusters = _cluster_by_topic(facts)
        assert isinstance(clusters, dict)
        assert len(clusters) >= 1

    def test_empty_input(self):
        clusters = _cluster_by_topic([])
        assert clusters == {}


class TestGenerateObservation:
    def test_basic_observation(self):
        facts = [
            {"content": "\u7528\u6237\u559c\u6b22Python"},
            {"content": "\u7528\u6237\u504f\u597d\u7f16\u7a0b"},
        ]
        result = _generate_observation(facts)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_facts(self):
        result = _generate_observation([])
        assert result == ""


class TestGenerateMentalModel:
    def test_basic_model(self):
        observations = ["\u5173\u4e8ePython\uff0c\u89c2\u5bdf\u5230\uff1a\u7528\u6237\u559c\u6b22\u7f16\u7a0b"]
        result = _generate_mental_model(observations)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_observations(self):
        result = _generate_mental_model([])
        assert result == ""


class TestConsolidatedItem:
    def test_to_dict(self):
        item = ConsolidatedItem(
            item_id="test-001",
            stage="observations",
            content="test content",
            source_ids=["s1", "s2"],
            confidence=0.8,
        )
        d = item.to_dict()
        assert d["item_id"] == "test-001"
        assert d["stage"] == "observations"
        assert d["content"] == "test content"
        assert d["source_ids"] == ["s1", "s2"]
        assert d["confidence"] == 0.8


class TestConsolidationEngine:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mock_llm_client = MagicMock()
        self.engine = ConsolidationEngine(
            data_dir=Path(self.tmpdir),
            fact_threshold=3,
            llm_client=self.mock_llm_client,
        )

    def test_submit(self):
        self.engine.submit("mem-001", "\u6d4b\u8bd5\u5185\u5bb9", "fact")
        assert self.engine.pending_count == 1

    def test_should_process_below_threshold(self):
        self.engine.submit("mem-001", "\u6d4b\u8bd5", "fact")
        assert self.engine.should_process() is False

    def test_should_process_at_threshold(self):
        for i in range(3):
            self.engine.submit(f"mem-{i}", f"\u6d4b\u8bd5\u5185\u5bb9{i}", "fact")
        assert self.engine.should_process() is True

    def test_process_pending_with_llm(self):
        mock_result = MagicMock()
        mock_result.content = "\u6d4b\u8bd5\u89c2\u5bdf\u7ed3\u679c"
        self.mock_llm_client.call_sync.return_value = mock_result

        for i in range(3):
            self.engine.submit(f"mem-{i}", f"\u7528\u6237\u559c\u6b22Python\u7f16\u7a0b{i}", "fact")
        count = self.engine.process_pending()
        assert count == 3
        assert self.engine.pending_count == 0

    def test_process_pending_without_llm(self):
        engine = ConsolidationEngine(
            data_dir=Path(tempfile.mkdtemp()),
            fact_threshold=3,
        )
        for i in range(3):
            engine.submit(f"mem-{i}", f"\u7528\u6237\u559c\u6b22Python\u7f16\u7a0b{i}", "fact")
        count = engine.process_pending()
        assert count == 3
        engine.close()

    def test_get_stats(self):
        self.engine.submit("mem-001", "\u6d4b\u8bd5", "fact")
        stats = self.engine.get_stats()
        assert "pending" in stats
        assert stats["pending"] == 1

    def test_reflect_empty(self):
        result = self.engine.reflect("nonexistent")
        assert isinstance(result, ConsolidationResult)
        assert result.observation == ""

    def test_get_observations(self):
        result = self.engine.get_observations("test")
        assert isinstance(result, list)

    def test_get_mental_models(self):
        result = self.engine.get_mental_models("test")
        assert isinstance(result, list)

    def test_close(self):
        self.engine.close()
        assert self.engine._conn is None


class TestConsolidationResult:
    def test_default_values(self):
        r = ConsolidationResult()
        assert r.observation == ""
        assert r.mental_model == ""
        assert r.facts_consolidated == 0
        assert r.observations_generated == 0
        assert r.models_generated == 0
