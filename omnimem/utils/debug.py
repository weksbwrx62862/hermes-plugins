"""OmniMem 调试模式工具。"""
from __future__ import annotations

import os


def is_debug_mode() -> bool:
    """检查是否开启调试模式。"""
    return os.environ.get("OMNIMEM_DEBUG", "").strip() in ("1", "true", "yes")
