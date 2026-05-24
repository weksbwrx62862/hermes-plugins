import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omnimem.deep.reflect import (
    Disposition,
    ReflectEngine,
    ReflectResult,
    ReflectionContext,
    _apply_disposition,
)


class TestDisposition:
    def test_default_values(self):
        d = Disposition()
        assert d.skepticism == 3
        assert d.literalness == 2
        assert d.empathy == 4

    def test_clamp(self):
        d = Disposition(skepticism=0, literalness=10, empathy=-1)
        clamped = d.clamp()
        assert clamped.skepticism == 1
        assert clamped.literalness == 5
        assert clamped.empathy == 1

    def test_to_dict(self):
        d = Disposition(skepticism=2, literalness=3, empathy=5)
        result = d.to_dict()
        assert result == {"skepticism": 2, "literalness": 3, "empathy": 5}


class TestApplyDisposition:
    def test_skepticism_prefix(self):
        d = Disposition(skepticism=4, literalness=2, empathy=1)
        obs, model = _apply_disposition("test observation", "test model", d)
        assert obs.startswith("\u9700\u8981\u8c28\u614e\u5bf9\u5f85")

    def test_empathy_suffix_with_person_context(self):
        d = Disposition(skepticism=1, literalness=2, empathy=4)
        obs, model = _apply_disposition("\u7528\u6237\u611f\u53d7", "model", d)
        assert "\u611f\u53d7" in obs or "\u5173\u6ce8" in obs

    def test_no_modification_low_values(self):
        d = Disposition(skepticism=1, literalness=2, empathy=1)
        obs, model = _apply_disposition("observation", "model", d)
        assert obs == "observation"
        assert model == "model"

    def test_literalness_high_adds_verifiability(self):
        d = Disposition(skepticism=1, literalness=5, empathy=1)
        obs, model = _apply_disposition("obs", "\u6838\u5fc3\u89c4\u5f8b", d)
        assert "\u53ef\u9a8c\u8bc1" in model or "\u4e8b\u5b9e\u4f9d\u636e" in model


class TestReflectEngine:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mock_llm_fn = MagicMock(return_value="\u3010\u89c2\u5bdf\u3011\n\u6d4b\u8bd5\u89c2\u5bdf\n\n\u3010\u5fc3\u667a\u6a21\u578b\u3011\n\u6d4b\u8bd5\u6a21\u578b\n\n\u3010\u7f6e\u4fe1\u5ea6\u3011\n0.8")
        self.engine = ReflectEngine(
            data_dir=Path(self.tmpdir),
            llm_fn=self.mock_llm_fn,
        )

    def test_reflect_with_no_data(self):
        result = self.engine.reflect("nonexistent topic")
        assert isinstance(result, ReflectResult)
        assert result.query == "nonexistent topic"

    def test_reflect_with_memories(self):
        memories = [
            {"memory_id": "m1", "content": "\u7528\u6237\u559c\u6b22Python"},
            {"memory_id": "m2", "content": "\u7528\u6237\u504f\u597d\u7f16\u7a0b"},
        ]
        result = self.engine.reflect("\u7f16\u7a0b", memories=memories)
        assert isinstance(result, ReflectResult)
        assert result.query == "\u7f16\u7a0b"

    def test_reflect_with_disposition(self):
        result = self.engine.reflect(
            "test",
            disposition={"skepticism": 5, "literalness": 1, "empathy": 3},
        )
        assert isinstance(result, ReflectResult)
        assert result.disposition_used is not None

    def test_get_stats(self):
        self.engine.reflect("test query")
        stats = self.engine.get_stats()
        assert "total_reflections" in stats
        assert stats["total_reflections"] >= 1

    def test_get_reflection_history(self):
        self.engine.reflect("history test")
        history = self.engine.get_reflection_history("history")
        assert isinstance(history, list)

    def test_close(self):
        self.engine.close()
        assert self.engine._conn is None

    def test_reflect_without_llm(self):
        engine = ReflectEngine(data_dir=Path(self.tmpdir))
        memories = [{"memory_id": "m1", "content": "\u6d4b\u8bd5\u5185\u5bb9"}]
        result = engine.reflect("test", memories=memories)
        assert isinstance(result, ReflectResult)
        engine.close()


class TestReflectResult:
    def test_default_values(self):
        r = ReflectResult()
        assert r.observation == ""
        assert r.mental_model == ""
        assert r.confidence == 0.0
        assert r.sources == []
        assert r.reflection_depth == 0
        assert r.query == ""

    def test_custom_values(self):
        r = ReflectResult(
            observation="obs",
            mental_model="model",
            confidence=0.9,
            sources=["s1"],
            reflection_depth=3,
            query="test",
        )
        assert r.observation == "obs"
        assert r.confidence == 0.9
