"""CompactAttachment — 压缩后携带的结构化状态。

参考 OpenHarness 的 Attachment 系统，8种附件类型：
  - task_focus: 当前任务焦点
  - verified_work: 已验证的工作结果
  - key_decisions: 关键决策
  - open_questions: 待解决问题
  - user_preferences: 用户偏好
  - error_patterns: 错误模式
  - progress_state: 进度状态
  - context_window: 上下文窗口摘要
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompactAttachment:
    """压缩后携带的结构化状态。"""

    kind: str  # task_focus / verified_work / key_decisions / ...
    title: str  # 简短标题
    body: str  # 内容正文
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_text(self, max_body_len: int = 500) -> str:
        """渲染为文本片段。"""
        body = self.body[:max_body_len]
        return f"[{self.kind}] {self.title}: {body}"


def build_attachments(messages: list[dict[str, Any]]) -> list[CompactAttachment]:
    """从消息列表构建 CompactAttachment。

    在 on_pre_compress 中被调用，将即将被压缩的消息转化为
    结构化附件，确保关键信息不会因压缩而丢失。
    """
    attachments: list[CompactAttachment] = []

    if not messages:
        return attachments

    # 提取关键决策（包含 "决定"/"选择"/"决定" 的用户/助手消息）
    decisions = []
    for msg in messages:
        content = _extract_text(msg)
        if not content:
            continue
        role = msg.get("role", "")
        # 简单关键词检测
        decision_markers = ["决定", "选择", "确认", "decided", "chose", "confirmed"]
        if any(m in content.lower() for m in decision_markers):
            decisions.append(f"[{role}] {content[:200]}")

    if decisions:
        attachments.append(
            CompactAttachment(
                kind="key_decisions",
                title="Decisions before compression",
                body="\n".join(decisions[:5]),
            )
        )

    # 提取用户偏好
    preferences = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = _extract_text(msg)
        if not content:
            continue
        pref_markers = ["我喜欢", "我不喜欢", "偏好", "prefer", "don't like", "i like"]
        if any(m in content.lower() for m in pref_markers):
            preferences.append(content[:200])

    if preferences:
        attachments.append(
            CompactAttachment(
                kind="user_preferences",
                title="User preferences noted",
                body="\n".join(preferences[:5]),
            )
        )

    # 提取错误模式
    errors = []
    for msg in messages:
        content = _extract_text(msg)
        if not content:
            continue
        error_markers = ["错误", "失败", "error", "failed", "bug"]
        if any(m in content.lower() for m in error_markers):
            errors.append(content[:200])

    if errors:
        attachments.append(
            CompactAttachment(
                kind="error_patterns",
                title="Errors encountered",
                body="\n".join(errors[:3]),
            )
        )

    # 进度状态：最后几条消息的摘要
    last_msgs = messages[-3:]
    progress_parts = []
    for msg in last_msgs:
        content = _extract_text(msg)
        if content:
            role = msg.get("role", "")
            progress_parts.append(f"[{role}] {content[:150]}")

    if progress_parts:
        attachments.append(
            CompactAttachment(
                kind="progress_state",
                title="Latest progress",
                body="\n".join(progress_parts),
            )
        )

    return attachments


def _extract_text(msg: dict[str, Any]) -> str:
    """从消息中提取纯文本。"""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text" or isinstance(c, dict):
                parts.append(c.get("text", ""))
        return " ".join(parts)
    return str(content) if content else ""
