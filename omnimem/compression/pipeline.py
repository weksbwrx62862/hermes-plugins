"""五层压缩管线 — 串联 micro / collapse / line_compress / llm_summary / priority。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from omnimem.compression.collapse import head_tail_collapse
from omnimem.compression.line_compress import structured_line_compress
from omnimem.compression.llm_summary import llm_summarize
from omnimem.compression.micro import microcompact
from omnimem.compression.priority import priority_compress

logger = logging.getLogger(__name__)


class CompressionPipeline:
    def __init__(
        self,
        llm_call_fn: Callable[[str], str] | None = None,
        config: Any = None,
        mermaid_canvas: Any = None,
        session_key: str = "",
    ) -> None:
        """初始化压缩管线。

        Args:
            llm_call_fn: LLM 调用函数
            config: 配置对象
            mermaid_canvas: MermaidCanvas 实例（可选）
            session_key: 会话标识，用于 Mermaid 压缩
        """
        self._llm_call_fn = llm_call_fn
        self._config = config
        self._mermaid_canvas = mermaid_canvas
        self._session_key = session_key

    def compress(self, content: str, memory_type: str = "", priority: int = 2) -> str:
        """压缩内容。

        Args:
            content: 待压缩内容
            memory_type: 记忆类型
            priority: 优先级

        Returns:
            压缩后的内容
        """
        # ★ OPT: 工具日志走 Mermaid 符号化路径
        if self._mermaid_canvas and self._mermaid_canvas.is_tool_log(content):
            logs = self._mermaid_canvas.parse_tool_logs(content)
            if logs:
                mermaid_text, _ = self._mermaid_canvas.offload_and_compress(
                    logs, session_key=self._session_key or "default"
                )
                if mermaid_text:
                    return mermaid_text

        # 常规压缩路径
        lines = content.split("\n")
        lines = microcompact(lines)
        lines = head_tail_collapse(lines)
        lines = structured_line_compress(lines)
        result = "\n".join(lines)

        if self._llm_call_fn:
            summary = llm_summarize(result, self._llm_call_fn)
            result = summary.to_text()

        items = [{"content": result, "type": memory_type or "fact", "confidence": 3}]
        items = priority_compress(items)
        return items[0]["content"] if items else result
