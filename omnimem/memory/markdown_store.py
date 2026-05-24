"""MarkdownStore — Markdown + YAML Front Matter 存取。

参考 OpenViking 的文件系统存储设计，以 Markdown 文件作为记忆的
持久化格式，YAML Front Matter 存储元数据。

文件格式:
```
---
memory_id: abc123
type: fact
confidence: 4
privacy: personal
stored_at: 2026-04-15T10:00:00
---

记忆内容...
```
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MarkdownStore:
    """Markdown + YAML Front Matter 存取。"""

    def __init__(self, palace_dir: Path):
        self._palace_dir = palace_dir
        self._palace_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._flush_interval = 10

    def write(
        self,
        wing: str,
        hall: str,
        room: str,
        memory_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """写入一条记忆到 Markdown 文件。"""
        room_dir = self._palace_dir / wing / hall / room
        room_dir.mkdir(parents=True, exist_ok=True)
        file_path = room_dir / f"{memory_id}.md"

        front_matter = {
            "memory_id": memory_id,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            front_matter.update(metadata)

        try:
            import yaml

            fm_str = yaml.dump(front_matter, allow_unicode=True, default_flow_style=False)
        except ImportError:
            fm_str = "\n".join(f"{k}: {v}" for k, v in front_matter.items())

        text = f"---\n{fm_str}---\n\n{content}\n"
        file_path.write_text(text, encoding="utf-8")
        return file_path

    def read(self, file_path: Path) -> dict[str, Any] | None:
        """从 Markdown 文件读取一条记忆。"""
        try:
            text = file_path.read_text(encoding="utf-8")
            if text.startswith("---"):
                parts = text.split("---", 2)
                if len(parts) >= 3:
                    try:
                        import yaml

                        fm = yaml.safe_load(parts[1]) or {}
                    except ImportError:
                        fm = {}
                    content = parts[2].strip()
                    return {**fm, "content": content, "file_path": str(file_path)}
            return {"content": text, "file_path": str(file_path)}
        except Exception as e:
            logger.warning("Failed to read %s: %s", file_path, e)
            return None

    def list_memories(self, wing: str = "", hall: str = "", room: str = "") -> list[Path]:
        """列出所有记忆文件。"""
        base = self._palace_dir
        if wing:
            base = base / wing
        if hall:
            base = base / hall
        if room:
            base = base / room

        if not base.exists():
            return []

        return sorted(p for p in base.rglob("*.md") if not p.name.startswith("_"))

    def flush(self) -> None:
        """刷新缓冲到磁盘。"""
        self._buffer.clear()
