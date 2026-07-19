"""测试公共配置：注入 mock 依赖与临时数据库。"""

import sys
from pathlib import Path

import pytest

# 将插件所在目录加入 PYTHONPATH，使 adaptive_multi_agent 包可被导入
ROOT = Path(__file__).resolve().parent.parent
PLUGINS_DIR = ROOT.parent
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

# 将 mock tools 目录加入 PYTHONPATH，避免真实 Hermes 运行时依赖
_MOCK_TOOLS = Path(__file__).resolve().parent / "_mock_tools"
if str(_MOCK_TOOLS) not in sys.path:
    sys.path.insert(0, str(_MOCK_TOOLS))


@pytest.fixture
def ama_persistence(tmp_path):
    """提供使用临时数据库的 AMAPersistence 实例。"""
    from adaptive_multi_agent.persistence import AMAPersistence

    db_path = tmp_path / "ama_state.db"
    return AMAPersistence(db_path=db_path)
