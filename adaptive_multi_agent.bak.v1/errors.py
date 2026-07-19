from __future__ import annotations

from enum import Enum


class ErrorCategory(Enum):
    """AMA 统一错误分类枚举。

    每个成员同时提供：
    - .value: 英文字符串（用于持久化与接口）
    - .label: 中文标签（用于展示）
    """

    timeout = "timeout"                 # 超时
    validation = "validation"           # 参数校验失败
    llm_error = "llm_error"             # LLM 调用失败
    subagent_failure = "subagent_failure"  # 子代理返回 failed / 异常
    cancelled = "cancelled"             # 用户取消
    unknown = "unknown"                 # 其他未知错误

    @property
    def label(self) -> str:
        return _ERROR_CATEGORY_LABELS.get(self, self.value)


_ERROR_CATEGORY_LABELS: dict = {
    ErrorCategory.timeout: "超时",
    ErrorCategory.validation: "校验错误",
    ErrorCategory.llm_error: "LLM 错误",
    ErrorCategory.subagent_failure: "子代理失败",
    ErrorCategory.cancelled: "已取消",
    ErrorCategory.unknown: "未知错误",
}
