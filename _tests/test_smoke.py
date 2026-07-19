"""核心功能冒烟测试

从已注册工具中选择只读型工具(*_status / *_info / *_list / get_status / status)
进行调用, 验证返回值类型。对有副作用的插件(disk-cleanup / gateway-restart /
self_evolution)跳过实际工具调用, 仅验证工具已注册。
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Dict, List

from test_framework import MockPluginContext, PluginInfo, TestResult


# 跳过实际工具调用的插件集合:
#   - disk-cleanup / gateway-restart / self_evolution: 有副作用(修改系统状态)
#   - skill-router: 工具调用会触发 MiniLM 嵌入模型加载(首次 15s+), 不适合冒烟测试
SIDE_EFFECT_PLUGINS = {"disk-cleanup", "gateway-restart", "self_evolution", "skill-router"}

# 只读工具名后缀/全名(按优先级匹配)
READONLY_TOOL_PATTERNS = (
    "_status", "_info", "_list", "get_status", "status",
    "_stats", "_version", "_health",
)

# 单次工具调用超时(秒), 防止工具内部阻塞(如加载模型/网络请求)导致测试挂起
TOOL_CALL_TIMEOUT_SECONDS = 15.0


def _is_readonly_tool(name: str) -> bool:
    """判断工具名是否像只读工具(无副作用)。"""
    name_lower = name.lower()
    for pat in READONLY_TOOL_PATTERNS:
        if name_lower == pat or name_lower.endswith(pat):
            return True
    return False


def _call_with_timeout(fn: Any, timeout: float) -> Dict[str, Any]:
    """在子线程中执行 fn, 带超时。

    返回 {"done": bool, "value": Any, "exception": BaseException}。
    超时后子线程仍在运行(Python 无法强制终止), 但本函数会返回。
    """
    container: Dict[str, Any] = {"done": False, "value": None, "exception": None}

    def _target():
        try:
            container["value"] = fn()
        except BaseException as exc:
            container["exception"] = exc
        finally:
            container["done"] = True

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    container["done"] = not t.is_alive()
    return container


def _safe_call_tool(handler: Any) -> Dict[str, Any]:
    """安全调用工具函数, 尝试多种参数组合, 每次调用带超时。

    返回 {"ok": bool, "value": Any, "exception": str, "call_style": str}。
    """
    attempts = [
        ("no_args", lambda: handler()),
        ("empty_dict", lambda: handler({})),
        ("empty_str", lambda: handler("")),
        ("none", lambda: handler(None)),
    ]
    last_exc = None
    for style, call in attempts:
        result = _call_with_timeout(call, TOOL_CALL_TIMEOUT_SECONDS)
        if not result["done"]:
            # 超时: 该调用方式阻塞, 尝试下一种(子线程仍在后台运行)
            last_exc = TimeoutError(f"工具调用超过 {TOOL_CALL_TIMEOUT_SECONDS}s 未返回")
            continue
        if result["exception"] is not None:
            exc = result["exception"]
            if isinstance(exc, TypeError):
                # 参数不匹配, 尝试下一种调用方式
                last_exc = exc
                continue
            # 调用成功但内部抛异常, 记录异常类型
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            return {
                "ok": False,
                "value": None,
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": tb,
                "call_style": style,
            }
        return {"ok": True, "value": result["value"], "exception": None, "call_style": style}
    return {
        "ok": False,
        "value": None,
        "exception": f"所有调用方式均失败(最后: {last_exc})",
        "call_style": "none",
    }


def test_smoke(
    plugin_info: PluginInfo,
    ctx: MockPluginContext,
    manifest: Dict[str, Any],
) -> TestResult:
    """对已注册工具进行冒烟测试。"""
    start = time.perf_counter()
    details: Dict[str, Any] = {"tools_count": len(ctx.registered_tools)}

    # 根据插件类型适配: memory_provider / hook_only 不依赖工具注册,
    # 跳过工具冒烟测试, 避免被误报"未注册工具"WARN
    if plugin_info.plugin_kind == "memory_provider":
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="smoke",
            status="PASS",
            message="memory_provider 插件, 跳过工具冒烟测试",
            details=details,
            duration_ms=(time.perf_counter() - start) * 1000,
        )
    if plugin_info.plugin_kind == "hook_only":
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="smoke",
            status="PASS",
            message="hook_only 插件, 跳过工具冒烟测试",
            details=details,
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    # 无工具注册
    if not ctx.registered_tools:
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="smoke",
            status="WARN",
            message="插件未注册任何工具, 无法进行冒烟测试",
            details=details,
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    # 有副作用的插件: 仅验证工具已注册, 不实际调用
    if plugin_info.name in SIDE_EFFECT_PLUGINS:
        details["skipped_call"] = True
        details["reason"] = "插件属于有副作用集合, 跳过实际工具调用"
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="smoke",
            status="PASS",
            message=f"已注册 {len(ctx.registered_tools)} 个工具(跳过实际调用以避免副作用)",
            details=details,
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    # 优先选择只读工具
    readonly_tools: List[str] = [
        name for name in ctx.registered_tools if _is_readonly_tool(name)
    ]
    target_tools = readonly_tools if readonly_tools else list(ctx.registered_tools.keys())[:1]
    details["readonly_tools"] = readonly_tools
    details["target_tools"] = target_tools

    call_results: List[Dict[str, Any]] = []
    any_pass = False
    any_fail = False

    for tool_name in target_tools:
        handler = ctx.registered_tools.get(tool_name)
        if not callable(handler):
            call_results.append({
                "tool": tool_name,
                "status": "FAIL",
                "reason": "handler 不可调用",
            })
            any_fail = True
            continue

        result = _safe_call_tool(handler)
        entry: Dict[str, Any] = {
            "tool": tool_name,
            "call_style": result.get("call_style"),
        }
        if result["ok"]:
            value = result["value"]
            # 验证返回值类型: dict / list / str / None
            if isinstance(value, (dict, list, str)) or value is None:
                entry["status"] = "PASS"
                entry["return_type"] = type(value).__name__
                any_pass = True
            else:
                entry["status"] = "WARN"
                entry["return_type"] = type(value).__name__
                entry["reason"] = "返回值类型非 dict/list/str/None"
                any_pass = True  # 仍算通过(调用成功)
        else:
            entry["status"] = "FAIL"
            entry["exception"] = result.get("exception")
            any_fail = True
        call_results.append(entry)

    details["call_results"] = call_results

    # 综合状态
    if any_fail and not any_pass:
        status = "FAIL"
        message = f"调用的 {len(call_results)} 个工具全部失败"
    elif any_fail:
        status = "WARN"
        message = f"部分工具调用失败({sum(1 for c in call_results if c['status']=='FAIL')}/{len(call_results)})"
    else:
        status = "PASS"
        message = f"成功调用 {len(call_results)} 个工具"

    return TestResult(
        plugin_name=plugin_info.name,
        test_name="smoke",
        status=status,
        message=message,
        details=details,
        duration_ms=(time.perf_counter() - start) * 1000,
    )
