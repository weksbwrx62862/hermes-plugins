"""pytest 共享 fixture 配置。"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock agent.memory_provider 模块（Hermes 框架依赖）
_mock_agent = MagicMock()
_mock_agent.memory_provider = MagicMock()
_mock_agent.memory_provider.MemoryProvider = object
sys.modules.setdefault("agent", _mock_agent)
sys.modules.setdefault("agent.memory_provider", _mock_agent.memory_provider)


@pytest.fixture
def omni_tmp_path() -> Generator[Path, None, None]:
    """提供临时目录路径。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=False)
def skip_if_no_chromadb():
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")


@pytest.fixture(autouse=False)
def skip_if_no_sentence_transformers():
    try:
        import sentence_transformers
    except ImportError:
        pytest.skip("sentence-transformers not installed")
