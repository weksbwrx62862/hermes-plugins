"""SearchTrace — 检索决策路径记录器。

内化 OpenViking 的可视化检索轨迹：
- 记录检索流程中的每一步决策
- 各通道检索（vector/bm25/catalog）的结果数和耗时
- RRF 融合的输入/输出/阈值
- 过滤步骤（垃圾查询/时间衰减/隐私/关键词验证）
- Rerank 和 Token 裁剪

用法：
trace = SearchTrace(query="用户偏好")
with trace.step("channel_search", channel="vector") as s:
    vector_results = self._vector_search(query, top_k)
    s.output_count = len(vector_results)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TraceStep:
    """检索轨迹中的一个步骤。"""
    action: str           # 动作名称（channel_search / rrf_fuse / filter / rerank / trim）
    channel: str = ""     # 通道名称（vector / bm25 / catalog / graph / store）
    input_count: int = 0  # 输入结果数
    output_count: int = 0 # 输出结果数
    elapsed_ms: float = 0 # 耗时（毫秒）
    details: dict[str, Any] = field(default_factory=dict)  # 额外信息


class SearchTrace:
    """检索轨迹记录器 — 内化 OpenViking 的可视化检索轨迹。

    记录检索流程中的每一步决策：
    1. 各通道检索（vector/bm25/catalog）的结果数和耗时
    2. RRF 融合的输入/输出/阈值
    3. 过滤步骤（垃圾查询/时间衰减/隐私/关键词验证）
    4. Rerank 和 Token 裁剪
    """

    def __init__(self, query: str):
        self.query = query
        self.steps: list[TraceStep] = []
        self._current_step: TraceStep | None = None
        self._step_start: float = 0

    def step(self, action: str, **kwargs) -> _StepContext:
        """记录一个检索步骤。"""
        return self._StepContext(self, action, **kwargs)

    def add_step(self, action: str, **kwargs) -> None:
        """直接添加一个步骤（不测量耗时）。"""
        self.steps.append(TraceStep(action=action, **kwargs))

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "query": self.query,
            "total_steps": len(self.steps),
            "steps": [
                {
                    "action": s.action,
                    "channel": s.channel,
                    "input": s.input_count,
                    "output": s.output_count,
                    "elapsed_ms": round(s.elapsed_ms, 1),
                    **s.details,
                }
                for s in self.steps
            ],
        }

    class _StepContext:
        """上下文管理器：自动测量步骤耗时。"""

        def __init__(self, trace: SearchTrace, action: str, **kwargs):
            self._trace = trace
            self._step = TraceStep(action=action, **kwargs)

        def __enter__(self) -> TraceStep:
            self._step._start_time = time.time()
            self._trace._current_step = self._step
            return self._step

        def __exit__(self, *args) -> None:
            self._step.elapsed_ms = (time.time() - self._step._start_time) * 1000
            self._trace.steps.append(self._step)
            self._trace._current_step = None
