"""KnowledgeGraph 兼容 shim — 已迁移至 deep/kg/ 子包。

保留此文件以维持旧导入路径 `from omnimem.deep.knowledge_graph import ...` 继续可用。
"""

from __future__ import annotations

from omnimem.deep.kg import *  # noqa: F403
from omnimem.deep.kg import __all__ as _kg_all

__all__ = _kg_all
