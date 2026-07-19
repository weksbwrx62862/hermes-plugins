"""模块可导入性测试

使用 importlib 动态加载插件的 __init__.py, 检查 register 函数是否存在,
不真实调用 register(ctx)。加载前将 hermes-agent 与 ~/.hermes 加入 sys.path,
保证插件内部的相对导入(如 from plugins.xxx import ...)能解析。
"""

from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

from test_framework import PluginInfo, TestResult


# Hermes 主程序目录, 加入 sys.path 以支持插件内部对 hermes_cli / agent 等模块的导入
HERMES_AGENT_DIR = Path.home() / ".hermes" / "hermes-agent"
# ~/.hermes 目录, 加入 sys.path 以支持 from plugins.xxx import ... 风格的导入
HERMES_HOME = Path.home() / ".hermes"


def _ensure_path_on_syspath(path: Path) -> None:
    """将路径加入 sys.path(若尚未存在)。"""
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_plugin_module(plugin_info: PluginInfo) -> ModuleType:
    """以唯一模块名加载插件 __init__.py, 返回模块对象。

    使用 _test_plugin_<name> 作为模块名, 避免污染 sys.modules 中
    已有的 hermes_plugins.<slug> 命名空间。
    """
    _ensure_path_on_syspath(HERMES_AGENT_DIR)
    _ensure_path_on_syspath(HERMES_HOME)

    init_file = plugin_info.init_path
    if not init_file.exists():
        raise FileNotFoundError(f"插件缺少 __init__.py: {init_file}")

    # 唯一模块名, 避免与已加载模块冲突
    module_name = f"_test_plugin_{plugin_info.name.replace('-', '_')}"

    # 若已存在同名模块(前次测试残留), 先移除以保证干净加载
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name,
        init_file,
        submodule_search_locations=[str(plugin_info.path)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {init_file} 创建模块 spec")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = module_name
    module.__path__ = [str(plugin_info.path)]  # type: ignore[attr-defined]
    # 注册到 sys.modules 以支持插件内部的相对/绝对导入
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_import(plugin_info: PluginInfo) -> TestResult:
    """测试插件 __init__.py 是否可被导入, 并检查 register 函数。

    返回 TestResult, 不抛异常。details 中携带 module 引用供后续测试复用。
    """
    start = time.perf_counter()
    details: dict = {}

    try:
        module = _load_plugin_module(plugin_info)
    except SyntaxError as exc:
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="import",
            status="FAIL",
            message=f"语法错误: {exc}",
            details={"lineno": exc.lineno, "filename": exc.filename},
            duration_ms=(time.perf_counter() - start) * 1000,
            exception="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
    except ImportError as exc:
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="import",
            status="FAIL",
            message=f"导入失败: {exc}",
            details={"exception": str(exc)},
            duration_ms=(time.perf_counter() - start) * 1000,
            exception="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
    except Exception as exc:
        # 捕获其他所有异常(如插件顶层代码执行抛出)
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="import",
            status="FAIL",
            message=f"加载时抛出异常: {exc}",
            details={"exception_type": type(exc).__name__},
            duration_ms=(time.perf_counter() - start) * 1000,
            exception="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )

    # 检查 register 是否可调用
    register_fn = getattr(module, "register", None)
    if register_fn is None:
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="import",
            status="FAIL",
            message="模块未定义 register 函数",
            details={},
            duration_ms=(time.perf_counter() - start) * 1000,
        )
    if not callable(register_fn):
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="import",
            status="FAIL",
            message="register 不是可调用对象",
            details={"register_type": type(register_fn).__name__},
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    # 记录可选函数存在性
    optional_fns = ["unregister", "get_status", "reload", "configure"]
    present_optional = [fn for fn in optional_fns if callable(getattr(module, fn, None))]
    details["optional_functions"] = present_optional
    details["module_name"] = module.__name__

    # 将 module 引用附在 details 上供后续 test_register 复用
    # (TestResult.details 是普通 dict, 可携带任意对象, 但报告生成时会做安全序列化)
    details["_module"] = module

    return TestResult(
        plugin_name=plugin_info.name,
        test_name="import",
        status="PASS",
        message="模块导入成功, register 可调用",
        details=details,
        duration_ms=(time.perf_counter() - start) * 1000,
    )
