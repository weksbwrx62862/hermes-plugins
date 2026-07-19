"""shared 插件测试公共配置。"""

import importlib.util
import sys
from pathlib import Path

import pytest

# 将 plugins 目录加入 PYTHONPATH，使 shared / adaptive_multi_agent 等包可被导入
ROOT = Path(__file__).resolve().parent.parent
PLUGINS_DIR = ROOT.parent
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))


def _load_model_router():
    """通过文件路径加载 model-router 插件（目录名含连字符，无法直接 import）。"""
    module_name = "_model_router_for_tests"
    if module_name in sys.modules:
        return sys.modules[module_name]

    file_path = PLUGINS_DIR / "model-router" / "__init__.py"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def model_router_module():
    """返回已加载的 model-router 模块，可用 _estimate_complexity / _detect_task_type。"""
    return _load_model_router()


@pytest.fixture
def ama_assessor():
    """返回 AMA TaskComplexityAssessor 实例。"""
    from adaptive_multi_agent.assessor import TaskComplexityAssessor
    return TaskComplexityAssessor()
