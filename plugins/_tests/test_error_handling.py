"""错误处理机制测试

测试插件对异常输入的容错能力:
- 调用 register(None) / register("invalid_ctx") 观察是否抛异常
- 用 ast.parse 统计 __init__.py 中 try/except 块数量
- 通过 logging.Handler 捕获 ERROR 级别日志
"""

from __future__ import annotations

import ast
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from test_framework import PluginInfo, TestResult


class _ErrorLogCapture(logging.Handler):
    """捕获 ERROR 及以上级别日志的 logging.Handler。"""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _count_try_except(source: str) -> int:
    """统计源码中 try 语句块数量(粗略的容错指标)。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            count += 1
    return count


def _safe_call_register(register_fn: Any, ctx_arg: Any) -> Dict[str, Any]:
    """安全调用 register(ctx_arg), 返回结果字典。"""
    try:
        register_fn(ctx_arg)
        return {"raised": False, "exception": None}
    except BaseException as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return {
            "raised": True,
            "exception": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
        }


def test_error_handling(
    plugin_info: PluginInfo,
    module: Any,
) -> TestResult:
    """测试插件的错误处理机制。"""
    start = time.perf_counter()
    details: Dict[str, Any] = {}

    register_fn = getattr(module, "register", None)
    if register_fn is None or not callable(register_fn):
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="error_handling",
            status="FAIL",
            message="register 函数不可调用, 无法测试错误处理",
            details=details,
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    # 1. 统计 try/except 块数量
    try:
        source = Path(plugin_info.init_path).read_text(encoding="utf-8")
        try_count = _count_try_except(source)
    except Exception:
        try_count = 0
        source = ""
    details["try_except_count"] = try_count
    has_try_except = try_count > 0

    # 2. 安装日志捕获 handler, 捕获 root logger 的 ERROR
    capture = _ErrorLogCapture()
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.addHandler(capture)
    # 确保 ERROR 级别能被处理
    if root_logger.level == logging.NOTSET or root_logger.level > logging.ERROR:
        root_logger.setLevel(logging.WARNING)

    try:
        # 3. 调用 register(None)
        result_none = _safe_call_register(register_fn, None)
        details["register_none"] = {
            "raised": result_none["raised"],
            "exception": result_none.get("exception"),
        }

        # 4. 调用 register("invalid_ctx")
        result_invalid = _safe_call_register(register_fn, "invalid_ctx")
        details["register_invalid"] = {
            "raised": result_invalid["raised"],
            "exception": result_invalid.get("exception"),
        }
    finally:
        root_logger.removeHandler(capture)
        root_logger.setLevel(original_level)

    # 5. 收集捕获到的 ERROR 日志
    error_logs = [
        f"{logging.getLevelName(r.levelno)}: {r.getMessage()}"
        for r in capture.records
    ]
    details["error_logs"] = error_logs
    details["error_log_count"] = len(error_logs)

    # 6. 评估状态
    # register(None) 不抛异常 + 有 try/except -> PASS
    # register(None) 抛异常但有 try/except -> WARN
    # register(None) 抛异常且无 try/except -> FAIL
    none_raised = result_none["raised"]
    invalid_raised = result_invalid["raised"]

    if not none_raised and has_try_except:
        status = "PASS"
        msg = f"register(None) 未抛异常, 代码含 {try_count} 个 try/except 块"
    elif not none_raised and not has_try_except:
        status = "PASS"
        msg = "register(None) 未抛异常(虽未检出 try/except, 但容错表现良好)"
    elif none_raised and has_try_except:
        status = "WARN"
        msg = f"register(None) 抛出异常但代码含 {try_count} 个 try/except 块"
    else:
        status = "FAIL"
        msg = "register(None) 抛出异常且代码无 try/except 块, 容错能力不足"

    if invalid_raised:
        msg += f"; register('invalid_ctx') 亦抛异常"
    else:
        msg += f"; register('invalid_ctx') 未抛异常"

    if error_logs:
        msg += f"; 捕获 {len(error_logs)} 条 ERROR 日志"

    return TestResult(
        plugin_name=plugin_info.name,
        test_name="error_handling",
        status=status,
        message=msg,
        details=details,
        duration_ms=(time.perf_counter() - start) * 1000,
    )
