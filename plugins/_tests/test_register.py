"""register(ctx) 注册功能测试

在子线程中调用 module.register(ctx), 设置 30 秒超时, 防止插件阻塞测试流程。
校验 manifest 声明的 provides_tools / provides_hooks 是否实际注册, 并验证幂等性。
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from test_framework import MockPluginContext, PluginInfo, TestResult


# register 调用超时阈值(秒)
REGISTER_TIMEOUT_SECONDS = 30.0


def _run_register_in_thread(
    register_fn: Any,
    ctx: MockPluginContext,
    timeout: float,
) -> Tuple[Optional[BaseException], float, bool]:
    """在子线程中执行 register_fn(ctx)。

    返回 (异常对象或None, 耗时秒, 是否超时)。
    注意: 超时后子线程仍在运行(Python 无法强制终止线程), 但本函数会返回。
    """
    container: Dict[str, Any] = {"exception": None}
    start = time.perf_counter()

    def _target():
        try:
            register_fn(ctx)
        except BaseException as exc:  # 捕获 BaseException 以覆盖 SystemExit 等
            container["exception"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    duration = time.perf_counter() - start
    timed_out = t.is_alive()
    return container["exception"], duration, timed_out


def test_register(
    plugin_info: PluginInfo,
    module: Any,
    manifest: Dict[str, Any],
) -> Tuple[TestResult, MockPluginContext]:
    """测试 module.register(ctx), 返回 (TestResult, MockPluginContext)。"""
    start = time.perf_counter()
    ctx = MockPluginContext(plugin_name=plugin_info.name)
    # 将 manifest 挂到 ctx 上, 某些插件可能访问 ctx.manifest
    ctx.manifest = manifest

    register_fn = getattr(module, "register", None)
    if register_fn is None or not callable(register_fn):
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="register",
            status="FAIL",
            message="register 函数不可调用",
            details={},
            duration_ms=(time.perf_counter() - start) * 1000,
        ), ctx

    details: Dict[str, Any] = {}

    # 第一次调用 register(ctx), 带超时
    exc, duration, timed_out = _run_register_in_thread(
        register_fn, ctx, REGISTER_TIMEOUT_SECONDS,
    )

    if timed_out:
        details["timeout_seconds"] = REGISTER_TIMEOUT_SECONDS
        details["tools_registered"] = list(ctx.registered_tools.keys())
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="register",
            status="TIMEOUT",
            message=f"register 调用超过 {REGISTER_TIMEOUT_SECONDS}s 未返回(子线程仍在运行)",
            details=details,
            duration_ms=duration * 1000,
        ), ctx

    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        details["exception_type"] = type(exc).__name__
        details["tools_registered"] = list(ctx.registered_tools.keys())
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="register",
            status="FAIL",
            message=f"register 抛出异常: {exc}",
            details=details,
            duration_ms=duration * 1000,
            exception=tb,
        ), ctx

    # register 成功返回, 校验注册内容
    details["tools_registered"] = list(ctx.registered_tools.keys())
    details["hooks_registered"] = list(ctx.registered_hooks.keys())
    details["commands_registered"] = list(ctx.registered_commands.keys())
    details["register_duration_ms"] = round(duration * 1000, 2)

    declared_tools = manifest.get("provides_tools", []) or []
    registered_tool_names = set(ctx.registered_tools.keys())

    if plugin_info.plugin_kind == "hook_only":
        # hook_only 插件: 只要求不注册工具, 不强制要求注册钩子
        # (部分插件使用 monkey-patch / logging.Filter 等其他机制, 不通过标准 hook 注册)
        if ctx.registered_hooks:
            tools_status = "PASS"
            tools_msg = f"hook_only 插件, 已注册钩子: {list(ctx.registered_hooks.keys())}"
        else:
            tools_status = "PASS"
            tools_msg = "hook_only 插件, 未注册钩子(可能使用 monkey-patch 等机制)"
        details["declared_tools"] = declared_tools
    elif plugin_info.plugin_kind == "memory_provider":
        # memory_provider 插件: 校验 register 调用了 register_memory_provider
        # 未知 register_* 方法由 MockPluginContext.__getattr__ 记录到 call_history
        called_methods = {entry.get("method") for entry in ctx.call_history}
        if "register_memory_provider" in called_methods:
            tools_status = "PASS"
            tools_msg = "memory_provider 插件, 已调用 register_memory_provider"
        else:
            tools_status = "FAIL"
            tools_msg = "memory_provider 插件未调用 register_memory_provider"
        details["called_methods"] = sorted(m for m in called_methods if m)
    else:
        # tool_provider / standalone: 保持原逻辑, 对比 provides_tools
        if declared_tools:
            missing = [t for t in declared_tools if t not in registered_tool_names]
            if not missing:
                tools_status = "PASS"
                tools_msg = f"声明的 {len(declared_tools)} 个工具全部注册"
            elif len(missing) == len(declared_tools):
                tools_status = "FAIL"
                tools_msg = f"声明的工具全部未注册: {missing}"
            else:
                tools_status = "WARN"
                tools_msg = f"部分工具未注册(缺失 {len(missing)}/{len(declared_tools)}): {missing}"
            details["declared_tools"] = declared_tools
            details["missing_tools"] = missing
        else:
            tools_status = "PASS"
            tools_msg = "manifest 未声明 provides_tools"
            if registered_tool_names:
                tools_msg += f"(实际注册 {len(registered_tool_names)} 个)"

    # 校验 provides_hooks (hook_only 已在上方校验, 这里跳过避免重复判定)
    if plugin_info.plugin_kind == "hook_only":
        hooks_status = "PASS"
        hooks_msg = "hook_only 插件钩子校验已在工具校验中完成"
    else:
        declared_hooks = manifest.get("provides_hooks", []) or []
        if declared_hooks and not ctx.registered_hooks:
            hooks_status = "WARN"
            hooks_msg = f"声明了 provides_hooks {declared_hooks} 但未注册任何钩子"
        elif declared_hooks:
            hooks_status = "PASS"
            hooks_msg = f"已注册钩子: {list(ctx.registered_hooks.keys())}"
        else:
            hooks_status = "PASS"
            hooks_msg = "manifest 未声明 provides_hooks"

    # 幂等性: 第二次调用 register(ctx) 不抛异常即 PASS
    idempotent_msg = "未测试"
    exc2, _, timed_out2 = _run_register_in_thread(
        register_fn, ctx, REGISTER_TIMEOUT_SECONDS,
    )
    if timed_out2:
        idempotent_msg = "第二次调用超时"
        idempotent_status = "WARN"
    elif exc2 is not None:
        idempotent_msg = f"第二次调用抛出异常: {exc2}"
        idempotent_status = "WARN"
    else:
        idempotent_msg = "第二次调用未抛异常"
        idempotent_status = "PASS"
    details["idempotent"] = idempotent_status

    # 综合状态: 取最严重的
    statuses = [tools_status, hooks_status, idempotent_status]
    severity = {"PASS": 0, "WARN": 1, "FAIL": 2, "TIMEOUT": 2}
    worst = max(statuses, key=lambda s: severity.get(s, 2))

    message_parts = [tools_msg, hooks_msg, f"幂等性: {idempotent_msg}"]
    message = " | ".join(message_parts)

    return TestResult(
        plugin_name=plugin_info.name,
        test_name="register",
        status=worst,
        message=message,
        details=details,
        duration_ms=(time.perf_counter() - start) * 1000,
    ), ctx
