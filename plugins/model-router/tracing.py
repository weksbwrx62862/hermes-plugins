"""
model-router 可观测性追踪模块。

基于 OpenTelemetry 语义实现轻量级 Span / Trace 上下文，不强制依赖
opentelemetry 库；当环境中存在 observability/langfuse 插件或传入外部
Langfuse 客户端时，可将 trace metadata 回传。
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Optional


class SpanKind:
    """模拟 OpenTelemetry SpanKind。"""

    INTERNAL = "INTERNAL"
    SERVER = "SERVER"
    CLIENT = "CLIENT"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"


class _Span:
    """轻量级 Span 上下文对象，语义对齐 OpenTelemetry Span。"""

    def __init__(
        self,
        name: str,
        trace_id: str,
        parent_id: Optional[str] = None,
        kind: str = SpanKind.INTERNAL,
    ) -> None:
        self.name = name
        self.trace_id = trace_id
        self.span_id = self._generate_span_id()
        self.parent_id = parent_id
        self.kind = kind
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.attributes: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.status: dict[str, Any] = {"status_code": "UNSET"}

    @staticmethod
    def _generate_span_id() -> str:
        return uuid.uuid4().hex[:16]

    def set_attribute(self, key: str, value: Any) -> None:
        """设置单个属性。"""
        self.attributes[key] = value

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        """批量设置属性。"""
        self.attributes.update(attrs)

    def add_event(
        self, name: str, attributes: Optional[dict[str, Any]] = None, timestamp: Optional[float] = None
    ) -> None:
        """记录事件。"""
        self.events.append(
            {
                "name": name,
                "attributes": attributes or {},
                "timestamp": timestamp or time.time(),
            }
        )

    def set_status(self, status_code: str, description: Optional[str] = None) -> None:
        """设置 Span 状态。"""
        self.status = {"status_code": status_code, "description": description}

    def end(self, timestamp: Optional[float] = None) -> None:
        """结束 Span。"""
        self.end_time = timestamp or time.time()

    @property
    def duration_ms(self) -> Optional[float]:
        """耗时（毫秒）。"""
        if self.end_time is None:
            return None
        return round((self.end_time - self.start_time) * 1000, 3)

    def to_dict(self) -> dict[str, Any]:
        """导出为可序列化的字典。"""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
        }


class RouterTracer:
    """模型路由决策追踪器。"""

    # OpenTelemetry 语义约定前缀
    _ATTR_COMPLEXITY = "router.complexity"
    _ATTR_TASK_TYPE = "router.task_type"
    _ATTR_STRATEGY = "router.strategy"
    _ATTR_CANDIDATE_SCORES = "router.candidate_scores"
    _ATTR_SELECTED_MODEL = "router.selected_model"
    _ATTR_SELECTED_PROVIDER = "router.selected_provider"
    _ATTR_FALLBACK_CHAIN = "router.fallback_chain"
    _ATTR_REASON = "router.selection_reason"
    _ATTR_ELAPSED_MS = "router.elapsed_ms"
    _ATTR_CACHE_HIT = "router.cache_hit"

    def __init__(
        self,
        tracer_name: str = "model-router",
        langfuse_client: Optional[Any] = None,
    ) -> None:
        self.tracer_name = tracer_name
        self.langfuse_client = langfuse_client
        self._trace_id: Optional[str] = None
        self._span_stack: list[_Span] = []
        self._finished_spans: list[_Span] = []

    @staticmethod
    def _langfuse_plugin_path() -> Optional[Path]:
        """探测 observability/langfuse 插件目录。"""
        # 从本文件向上定位到 plugins 目录
        here = Path(__file__).resolve().parent
        plugins_dir = here.parent
        candidate = plugins_dir / "observability" / "langfuse"
        if candidate.is_dir():
            return candidate
        return None

    def langfuse_available(self) -> bool:
        """是否存在可集成的 langfuse 插件或外部客户端。"""
        if self.langfuse_client is not None:
            return True
        return self._langfuse_plugin_path() is not None

    def start_trace(self, name: str = "model-router-decision", attributes: Optional[dict[str, Any]] = None) -> _Span:
        """开启一条新的 Trace，并创建根 Span。"""
        self._trace_id = uuid.uuid4().hex
        self._finished_spans.clear()
        return self.start_span(name, attributes=attributes)

    def start_span(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        kind: str = SpanKind.INTERNAL,
    ) -> _Span:
        """开启一个 Span；若当前无 Trace，则自动创建 Trace ID。"""
        parent = self._span_stack[-1] if self._span_stack else None
        parent_id = parent.span_id if parent else None
        trace_id = self._trace_id or uuid.uuid4().hex
        if self._trace_id is None:
            self._trace_id = trace_id

        span = _Span(name, trace_id, parent_id=parent_id, kind=kind)
        if attributes:
            span.set_attributes(attributes)
        self._span_stack.append(span)
        return span

    def end_span(self) -> Optional[_Span]:
        """结束当前 Span，返回该 Span。"""
        if not self._span_stack:
            return None
        span = self._span_stack.pop()
        span.end()
        self._finished_spans.append(span)
        return span

    def set_attribute(self, key: str, value: Any) -> None:
        """为当前活跃 Span 设置属性。"""
        current = self.get_current_span()
        if current is not None:
            current.set_attribute(key, value)

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        """为当前活跃 Span 批量设置属性。"""
        current = self.get_current_span()
        if current is not None:
            current.set_attributes(attrs)

    def get_current_span(self) -> Optional[_Span]:
        """获取当前活跃 Span。"""
        return self._span_stack[-1] if self._span_stack else None

    def record_route_decision(
        self,
        *,
        complexity: int,
        task_type: str,
        strategy: str,
        candidate_scores: list[dict[str, Any]],
        selected_model: str,
        selected_provider: str,
        fallback_chain: list[dict[str, Any]],
        reason: str,
        cache_hit: bool = False,
        elapsed_ms: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> _Span:
        """
        记录一次路由决策。

        如果当前没有活跃 Span，会自动创建名为 ``route-decision`` 的 Span。
        结束时若检测到 Langfuse 集成，会自动将 trace metadata 回传。
        """
        if not self._span_stack:
            self.start_span("route-decision")

        attributes: dict[str, Any] = {
            self._ATTR_COMPLEXITY: complexity,
            self._ATTR_TASK_TYPE: task_type,
            self._ATTR_STRATEGY: strategy,
            self._ATTR_CANDIDATE_SCORES: candidate_scores,
            self._ATTR_SELECTED_MODEL: selected_model,
            self._ATTR_SELECTED_PROVIDER: selected_provider,
            self._ATTR_FALLBACK_CHAIN: fallback_chain,
            self._ATTR_REASON: reason,
            self._ATTR_CACHE_HIT: cache_hit,
        }
        if elapsed_ms is not None:
            attributes[self._ATTR_ELAPSED_MS] = elapsed_ms
        if extra:
            attributes.update(extra)

        self.set_attributes(attributes)
        span = self.end_span()

        if span is not None and self.langfuse_available():
            self.send_to_langfuse(span.to_dict())

        return span

    def send_to_langfuse(self, span_dict: dict[str, Any]) -> None:
        """
        将 trace metadata 回传给 Langfuse。

        优先使用构造函数传入的 ``langfuse_client``；否则尝试调用
        observability/langfuse 插件的标准适配接口 ``send_trace``。
        任何回传异常都会被吞掉，避免影响主流程。
        """
        try:
            metadata = {
                "trace_id": span_dict.get("trace_id"),
                "span_id": span_dict.get("span_id"),
                "name": span_dict.get("name"),
                "start_time": span_dict.get("start_time"),
                "end_time": span_dict.get("end_time"),
                "metadata": span_dict.get("attributes", {}),
            }

            if self.langfuse_client is not None:
                # 兼容 Langfuse 原生 trace 接口
                if hasattr(self.langfuse_client, "trace"):
                    self.langfuse_client.trace(**metadata)
                # 兼容简单回调接口
                elif callable(self.langfuse_client):
                    self.langfuse_client(metadata)
                return

            plugin_path = self._langfuse_plugin_path()
            if plugin_path is None:
                return

            # 目录名含连字符的插件无法直接 import，使用 importlib 加载
            import importlib.util

            adapter_path = plugin_path / "adapter.py"
            if not adapter_path.exists():
                return

            spec = importlib.util.spec_from_file_location(
                "observability_langfuse_adapter", str(adapter_path)
            )
            adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter)
            if hasattr(adapter, "send_trace"):
                adapter.send_trace(metadata)
        except Exception:
            # 回传失败不应影响主流程
            pass

    def get_finished_spans(self) -> list[_Span]:
        """获取已结束的 Span 列表。"""
        return list(self._finished_spans)

    def flush(self) -> list[dict[str, Any]]:
        """清空已结束 Span 并返回其字典形式。"""
        spans = [s.to_dict() for s in self._finished_spans]
        self._finished_spans.clear()
        self._trace_id = None
        return spans
