"""SoulSystem — 人格三元：Soul / Identity / User。

参考 OpenHarness 的 Soul/Identity/User 三元设计：
  - Soul: 核心价值观、人格基底（极少变化）
  - Identity: 当前身份、角色定位（偶尔更新）
  - User: 用户画像、偏好、习惯（频繁更新）
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SoulSystem:
    """人格三元：Soul(核心价值观) / Identity(身份) / User(用户画像)。"""

    def __init__(self, soul_dir: Path):
        self._soul_dir = soul_dir
        self._soul_dir.mkdir(parents=True, exist_ok=True)

    def load_identity(self) -> str:
        """加载 Soul + Identity + User 拼接文本。"""
        parts = []
        for name in ("soul.md", "identity.md", "user.md"):
            path = self._soul_dir / name
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8").strip()
                    if text:
                        parts.append(text)
                except Exception as e:
                    logger.warning("Failed to read %s: %s", name, e)
        return "\n\n".join(parts)

    def update_user_profile(self, content: str) -> None:
        """追加用户画像信息。"""
        user_path = self._soul_dir / "user.md"
        existing = ""
        if user_path.exists():
            with contextlib.suppress(Exception):
                existing = user_path.read_text(encoding="utf-8").strip()
        updated = f"{existing}\n\n{content}".strip() if existing else content
        user_path.write_text(updated, encoding="utf-8")

    def update_identity(self, content: str) -> None:
        """更新身份信息。"""
        identity_path = self._soul_dir / "identity.md"
        identity_path.write_text(content, encoding="utf-8")

    def set_soul(self, content: str) -> None:
        """设置核心价值观（仅在初始化时调用）。"""
        soul_path = self._soul_dir / "soul.md"
        if not soul_path.exists():
            soul_path.write_text(content, encoding="utf-8")
