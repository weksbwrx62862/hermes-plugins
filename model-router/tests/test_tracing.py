"""
RouterTracer 单元测试。

覆盖：
  - Span 正确记录所有关键字段
  - 决策理由为中文
  - 与 langfuse 插件 / 客户端的兼容接口（mock 测试）
  - OpenTelemetry 语义模拟（start_span / end_span / set_attribute）
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

# 目录名含连字符，无法直接 import，使用 importlib 加载 tracing.py
_tracing_path = Path(__file__).resolve().parent.parent / "tracing.py"
_spec = importlib.util.spec_from_file_location("tracing", str(_tracing_path))
tracing = importlib.util.module_from_spec(_spec)
sys.modules["tracing"] = tracing
_spec.loader.exec_module(tracing)

RouterTracer = tracing.RouterTracer
SpanKind = tracing.SpanKind


def _contains_chinese(text: str) -> bool:
    """判断字符串是否包含中文字符。"""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


class TestRouterTracer:
    """RouterTracer 核心功能测试。"""

    def test_start_and_end_span(self):
        """start_span / end_span 应正确创建并结束 Span。"""
        tracer = RouterTracer()
        span = tracer.start_span("test-span")

        assert span is tracer.get_current_span()
        assert span.name == "test-span"
        assert span.trace_id is not None
        assert span.span_id is not None
        assert span.end_time is None

        ended = tracer.end_span()
        assert ended is span
        assert span.end_time is not None
        assert ended.duration_ms is not None
        assert tracer.get_current_span() is None

    def test_nested_spans_have_parent(self):
        """嵌套 Span 应正确维护父子关系。"""
        tracer = RouterTracer()
        root = tracer.start_span("root")
        child = tracer.start_span("child")

        assert child.parent_id == root.span_id
        assert child.trace_id == root.trace_id

        tracer.end_span()
        tracer.end_span()

        finished = tracer.get_finished_spans()
        assert len(finished) == 2

    def test_set_attribute_on_current_span(self):
        """set_attribute 应写入当前活跃 Span。"""
        tracer = RouterTracer()
        tracer.start_span("span")
        tracer.set_attribute("custom.key", "value")

        span = tracer.get_current_span()
        assert span.attributes["custom.key"] == "value"

        tracer.end_span()

    def test_set_attribute_without_active_span_is_safe(self):
        """无活跃 Span 时调用 set_attribute 不应抛异常。"""
        tracer = RouterTracer()
        tracer.set_attribute("key", "value")  # 不应抛出

    def test_record_route_decision_records_all_key_fields(self):
        """record_route_decision 应记录所有关键字段。"""
        tracer = RouterTracer()
        candidate_scores = [
            {"model": "deepseek-v4-flash", "score": 9.5},
            {"model": "qwen-plus", "score": 8.2},
        ]
        fallback_chain = [
            {"provider": "openai", "model": "deepseek-v4-flash"},
            {"provider": "dashscope", "model": "qwen-plus"},
        ]

        span = tracer.record_route_decision(
            complexity=4,
            task_type="complex_reasoning",
            strategy="smartest",
            candidate_scores=candidate_scores,
            selected_model="deepseek-v4-pro",
            selected_provider="openai",
            fallback_chain=fallback_chain,
            reason="用户指定使用 DeepSeek；综合评分最高",
            cache_hit=True,
            elapsed_ms=12.3,
        )

        assert span is not None
        attrs = span.attributes
        assert attrs["router.complexity"] == 4
        assert attrs["router.task_type"] == "complex_reasoning"
        assert attrs["router.strategy"] == "smartest"
        assert attrs["router.candidate_scores"] == candidate_scores
        assert attrs["router.selected_model"] == "deepseek-v4-pro"
        assert attrs["router.selected_provider"] == "openai"
        assert attrs["router.fallback_chain"] == fallback_chain
        assert attrs["router.selection_reason"] == "用户指定使用 DeepSeek；综合评分最高"
        assert attrs["router.cache_hit"] is True
        assert attrs["router.elapsed_ms"] == 12.3

    def test_record_route_decision_reason_is_chinese(self):
        """决策理由应以中文写入。"""
        tracer = RouterTracer()
        span = tracer.record_route_decision(
            complexity=3,
            task_type="simple_qa",
            strategy="auto",
            candidate_scores=[{"model": "a", "score": 1.0}],
            selected_model="a",
            selected_provider="stepfun",
            fallback_chain=[],
            reason="阶跃星辰全天最高优先，稳定且免费额度大",
        )

        reason = span.attributes["router.selection_reason"]
        assert _contains_chinese(reason), f"决策理由应包含中文: {reason}"

    def test_flush_clears_finished_spans(self):
        """flush 应返回已结束 Span 并清空内部状态。"""
        tracer = RouterTracer()
        tracer.start_trace("trace")
        tracer.end_span()

        spans = tracer.flush()
        assert len(spans) == 1
        assert tracer.get_finished_spans() == []
        assert tracer._trace_id is None


class TestLangfuseCompatibility:
    """与 Langfuse 的兼容接口测试。"""

    def test_mock_langfuse_client_trace_called(self):
        """传入模拟 Langfuse 客户端时，trace 方法应被调用并携带 metadata。"""
        class MockLangfuse:
            def __init__(self):
                self.calls = []

            def trace(self, **kwargs):
                self.calls.append(kwargs)

        mock_client = MockLangfuse()
        tracer = RouterTracer(langfuse_client=mock_client)
        tracer.record_route_decision(
            complexity=2,
            task_type="chat",
            strategy="cheapest",
            candidate_scores=[{"model": "m", "score": 0.5}],
            selected_model="m",
            selected_provider="mimo",
            fallback_chain=[],
            reason="非高峰期优先使用 MiMo",
            elapsed_ms=5.0,
        )

        assert len(mock_client.calls) == 1
        metadata = mock_client.calls[0]
        assert metadata["name"] == "route-decision"
        assert metadata["trace_id"] is not None
        assert metadata["metadata"]["router.selected_provider"] == "mimo"

    def test_callable_langfuse_client(self):
        """支持传入可调用对象作为 Langfuse 客户端。"""
        received = []

        def callback(metadata):
            received.append(metadata)

        tracer = RouterTracer(langfuse_client=callback)
        tracer.record_route_decision(
            complexity=1,
            task_type="classify",
            strategy="fastest",
            candidate_scores=[{"model": "x", "score": 0.9}],
            selected_model="x",
            selected_provider="nvidia-nim",
            fallback_chain=[],
            reason="低延迟需求，选择最快渠道",
        )

        assert len(received) == 1
        assert received[0]["metadata"]["router.strategy"] == "fastest"

    def test_langfuse_available_with_external_client(self):
        """传入外部客户端时 langfuse_available 应返回 True。"""
        tracer = RouterTracer(langfuse_client=object())
        assert tracer.langfuse_available() is True

    def test_langfuse_not_available_when_plugin_missing(self):
        """未安装插件且无外部客户端时 langfuse_available 应返回 False。"""
        tracer = RouterTracer()
        assert tracer.langfuse_available() is False

    def test_send_to_langfuse_swallows_exception(self):
        """Langfuse 回传失败时不应阻断主流程。"""
        class BadClient:
            def trace(self, **kwargs):
                raise RuntimeError("boom")

        tracer = RouterTracer(langfuse_client=BadClient())
        span = tracer.record_route_decision(
            complexity=3,
            task_type="code",
            strategy="auto",
            candidate_scores=[{"model": "c", "score": 0.7}],
            selected_model="c",
            selected_provider="openai",
            fallback_chain=[],
            reason="代码任务自动路由",
        )
        assert span is not None
