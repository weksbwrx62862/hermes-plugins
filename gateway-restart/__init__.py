"""
gateway-restart plugin v1.0.0

允许从 Gateway 进程内部执行 ``hermes gateway restart`` / ``systemctl --user restart hermes-gateway``。

工作原理
========
通过 monkey-patch 拦截 ``tools.terminal_tool.terminal_tool()`` 函数。当检测到当前进程在
Gateway 内运行 (``_HERMES_GATEWAY=1``) 并且命令是 gateway restart 指令时，不执行该子进程，
而是直接向当前进程发送 SIGUSR1 信号，触发 Gateway 内置的优雅重启流程：

    SIGUSR1 → request_restart(via_service=True) → drain 活跃 agent
    → exit(75) → systemd RestartForceExitStatus=75 自动重启

这等价于 ``/restart`` 斜杠命令，不经过子进程、无 SIGTERM 传播问题。

与 venv 补丁的关系
================
此插件与已应用的 venv 补丁功能相同，但独立于 ``site-packages`` 文件存在。
即使 ``hermes update`` 覆盖了 ``terminal_tool.py`` / ``gateway.py`` 中的补丁，
此插件仍会在下次 Gateway 启动时重新应用 monkey-patch。

依赖
====
- Linux (需要 SIGUSR1 信号，POSIX)
- ``tools.terminal_tool`` 模块 (Hermes 核心)
- ``hermes_cli.cron._contains_gateway_lifecycle_command`` (命令检测)
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import signal as _sig
import textwrap
import time
from typing import Callable

logger = logging.getLogger(__name__)

# ── 状态 ──────────────────────────────────────────────────────────────────

_PATCH_APPLIED: bool = False
_PATCH_PENDING: bool = False
_PATCH_RETRY_COUNT: int = 0
_PATCH_MAX_RETRIES: int = 10
_PATCH_RETRY_DELAY: float = 1.0

# ── RESTART 命令检测 ──────────────────────────────────────────────────────

_RESTART_PATTERN = re.compile(
    r"(?i)"
    r"(hermes\s+gateway\s+restart)"
    r"|(systemctl\s+(-\S+\s+)*restart\s+.*hermes)"
    r"|(launchctl\s+(kickstart|restart)\s+.*hermes)",
)

_STOP_ONLY_PATTERN = re.compile(
    r"(?i)"
    r"(hermes\s+gateway\s+stop)"
    r"|(systemctl\s+(-\S+\s+)*(stop|kill)\s+.*hermes)"
    r"|(pkill\s+.*hermes)"
)


def _is_gateway_lifecycle_command(command: str) -> tuple[bool, bool]:
    """检测命令是否为 gateway 生命周期命令。

    Returns:
        (is_lifecycle, is_restart): (是否生命周期命令, 是否重启命令)
    """
    try:
        from hermes_cli.cron import _contains_gateway_lifecycle_command
        if not _contains_gateway_lifecycle_command(command):
            return False, False
    except ImportError:
        logger.debug("gateway-restart: hermes_cli.cron not available, falling back to regex")
        # fallback: 简单模式匹配
        if not (_RESTART_PATTERN.search(command) or _STOP_ONLY_PATTERN.search(command)):
            return False, False

    is_restart = bool(_RESTART_PATTERN.search(command))
    return True, is_restart


# ── PATCHED terminal_tool ──────────────────────────────────────────────────

def _make_patched_terminal_tool(original_fn: Callable) -> Callable:
    """创建一个包装后的 ``terminal_tool`` 函数，拦截 gateway restart 命令。"""

    @functools.wraps(original_fn)
    def wrapper(
        command: str,
        background: bool = False,
        timeout: int | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        force: bool = False,
        workdir: str | None = None,
        pty: bool = False,
        notify_on_complete: bool = False,
        watch_patterns: list[str] | None = None,
    ) -> str:
        # 只在 Gateway 进程内拦截
        if os.environ.get("_HERMES_GATEWAY") == "1" and isinstance(command, str):
            is_lifecycle, is_restart = _is_gateway_lifecycle_command(command)

            if is_lifecycle:
                if is_restart:
                    # RESTART → SIGUSR1 优雅重启
                    _pid = os.getpid()
                    logger.info(
                        "gateway-restart: redirecting to SIGUSR1 (pid=%s) — command: %s",
                        _pid, command[:120],
                    )
                    if hasattr(_sig, "SIGUSR1"):
                        os.kill(_pid, _sig.SIGUSR1)
                        return json.dumps({
                            "output": (
                                "Gateway graceful restart triggered via SIGUSR1"
                                " (same as /restart).\n"
                                "The gateway will drain active agents and restart"
                                " under systemd."
                            ),
                            "exit_code": 0,
                            "error": "",
                            "status": "success",
                        }, ensure_ascii=False)
                    else:
                        return json.dumps({
                            "output": "",
                            "exit_code": 1,
                            "error": (
                                "SIGUSR1 not available on this platform;"
                                " cannot restart from inside gateway."
                            ),
                            "status": "error",
                        }, ensure_ascii=False)
                else:
                    # STOP 仍然阻止
                    return json.dumps({
                        "output": "",
                        "exit_code": 1,
                        "error": (
                            "Blocked: cannot stop the gateway from inside the "
                            "gateway process.  Restart is allowed (redirected to "
                            "graceful SIGUSR1 path).  Use `hermes gateway stop` "
                            "from a separate shell outside the running gateway."
                        ),
                        "status": "error",
                    }, ensure_ascii=False)

        # 非生命周期命令或不在 gateway 内 → 正常执行
        return original_fn(
            command=command,
            background=background,
            timeout=timeout,
            task_id=task_id,
            session_id=session_id,
            force=force,
            workdir=workdir,
            pty=pty,
            notify_on_complete=notify_on_complete,
            watch_patterns=watch_patterns,
        )

    wrapper._gw_restart_patched = True
    return wrapper


# ── PATCHED gateway CLI ────────────────────────────────────────────────────

def _patch_gateway_cli() -> bool:
    """Patch ``hermes_cli/gateway.py`` 中的 restart 子命令。

    如果 venv 中的补丁被 hermes update 覆盖，通过 monkey-patch 恢复。
    """
    try:
        import hermes_cli.gateway as gw

        # 查找 restart 子命令处理函数
        # 在 gateway.py 中，子命令对应的处理函数是 ``_handle_gateway_restart``
        handler_name = None
        for name in dir(gw):
            if "restart" in name.lower() and callable(getattr(gw, name, None)):
                handler_name = name
                break

        if handler_name is None:
            logger.debug("gateway-restart: no restart handler found in hermes_cli.gateway")
            return False

        handler = getattr(gw, handler_name)
        if getattr(handler, "_gw_restart_patched", False):
            logger.debug("gateway-restart: CLI handler already patched")
            return True

        @functools.wraps(handler)
        def _patched_handler(*args, **kwargs):
            if os.getenv("_HERMES_GATEWAY") == "1":
                if hasattr(_sig, "SIGUSR1"):
                    _pid = os.getpid()
                    logger.info(
                        "gateway-restart: CLI redirect to SIGUSR1 (pid=%s)",
                        _pid,
                    )
                    os.kill(_pid, _sig.SIGUSR1)
                    print("✓ Gateway graceful restart triggered via SIGUSR1 (same as /restart).")
                    return 0  # sys.exit(0) 会在网关终止后由系统处理
                else:
                    print(
                        "Error: SIGUSR1 not available on this platform;"
                        " cannot restart from inside gateway.",
                        file=__import__("sys").stderr,
                    )
                    return 1
            return handler(*args, **kwargs)

        _patched_handler._gw_restart_patched = True
        setattr(gw, handler_name, _patched_handler)
        logger.info("gateway-restart: CLI patch applied to %s", handler_name)
        return True

    except Exception as e:
        logger.debug("gateway-restart: CLI patch failed: %s", e)
        return False


# ── APPLY ──────────────────────────────────────────────────────────────────

def _terminal_tool_has_venv_patch() -> bool:
    """检查 ``tools.terminal_tool.terminal_tool`` 函数源码中是否已包含
    venv 级别的 SIGUSR1 重启补丁。如果已有，则无需 monkey-patch。"""
    try:
        import tools.terminal_tool as tt
        import inspect
        if not hasattr(tt, "terminal_tool"):
            return False

        source = inspect.getsource(tt.terminal_tool)
        # 检查是否包含 venv 补丁的特征代码
        return (
            "gateway's built-in SIGUSR1 graceful-restart path" in source
            or "redirecting gateway restart to SIGUSR1" in source
            or "gateway-restart" in source
        )
    except Exception:
        return False


def _apply_monkey_patch() -> bool:
    """应用所有 monkey-patch。

    Returns:
        True 如果所有 patch 都成功应用。
    """
    terminal_patched = False
    cli_patched = False

    # 1. Patch tools.terminal_tool — 仅在 venv 补丁缺失时应用
    try:
        import tools.terminal_tool as tt

        if not hasattr(tt, "terminal_tool"):
            logger.warning("gateway-restart: tools.terminal_tool.terminal_tool not found")
        else:
            original = tt.terminal_tool
            if _terminal_tool_has_venv_patch():
                logger.debug(
                    "gateway-restart: venv patch detected in terminal_tool,"
                    " skipping monkey-patch (will activate after hermes update)"
                )
                terminal_patched = True  # venv 已处理，无需 monkey-patch
            elif getattr(original, "_gw_restart_patched", False):
                logger.debug("gateway-restart: terminal_tool already monkey-patched")
                terminal_patched = True
            else:
                patched = _make_patched_terminal_tool(original)
                tt.terminal_tool = patched
                terminal_patched = True
                logger.info("gateway-restart: monkey-patch applied to tools.terminal_tool.terminal_tool")
    except ImportError:
        logger.debug("gateway-restart: tools.terminal_tool not available yet (lazy load)")

    # 2. Patch CLI gateway
    cli_patched = _patch_gateway_cli()

    return terminal_patched


# ── LAZY PATCH (TRAMPOLINE) ────────────────────────────────────────────────
#
# 为了让 ``register`` 快速返回（< 100ms），真正的 monkey-patch 应用被延迟到
# 首次 ``terminal_tool`` 调用时执行。register 阶段只安装一个轻量级 trampoline
# 包装函数，它不执行 ``inspect.getsource`` / ``import hermes_cli.gateway`` /
# 重试 sleep 等耗时操作；当 trampoline 首次被调用时，才触发真正的 patch 逻辑
# （venv 检查、安装真正的 wrapper、CLI patch），然后用真正的 wrapper 替换自身。


def _apply_lazy_patch(original_fn: Callable) -> None:
    """延迟应用真正的 monkey-patch（在首次 ``terminal_tool`` 调用时执行）。

    与 ``_apply_monkey_patch`` 的区别：本函数由 trampoline 在首次调用时触发，
    此时 ``tools.terminal_tool`` 必然已可导入（否则 trampoline 不会被调用），
    且 ``original_fn`` 是 trampoline 闭包持有的真正原版函数。这里完成 venv 检
    查、安装真正的 wrapper、应用 CLI patch。

    Args:
        original_fn: trampoline 闭包持有的原始 ``terminal_tool`` 函数。
    """
    global _PATCH_APPLIED

    try:
        import tools.terminal_tool as tt

        if _terminal_tool_has_venv_patch():
            logger.debug(
                "gateway-restart: venv patch detected (lazy), restoring original"
            )
            # venv 已自带补丁，恢复原版即可，无需再叠加 monkey-patch
            tt.terminal_tool = original_fn
        elif getattr(original_fn, "_gw_restart_patched", False):
            logger.debug("gateway-restart: original already patched (lazy)")
            tt.terminal_tool = original_fn
        else:
            # 安装真正的拦截 wrapper，替换 trampoline
            real_wrapper = _make_patched_terminal_tool(original_fn)
            tt.terminal_tool = real_wrapper
            logger.info(
                "gateway-restart: monkey-patch applied (lazy) to"
                " tools.terminal_tool.terminal_tool"
            )

        # 应用 CLI patch（hermes_cli.gateway restart 子命令）
        _patch_gateway_cli()
        _PATCH_APPLIED = True
    except Exception as e:
        logger.debug("gateway-restart: lazy patch apply failed: %s", e)
        # 失败时恢复原版，避免 trampoline 残留导致后续调用空转
        try:
            import tools.terminal_tool as tt
            tt.terminal_tool = original_fn
        except Exception:
            pass


def _make_trampoline(original_fn: Callable) -> Callable:
    """创建轻量级 trampoline wrapper。

    首次调用时延迟应用真正的 monkey-patch，之后委托给真正的 wrapper / 原版。
    本函数不执行任何重逻辑，确保 ``register`` 阶段安装它时几乎零开销。
    """

    @functools.wraps(original_fn)
    def trampoline(*args, **kwargs):
        global _PATCH_PENDING

        if _PATCH_PENDING:
            _PATCH_PENDING = False
            _apply_lazy_patch(original_fn)

        # 此时 ``tt.terminal_tool`` 已被 ``_apply_lazy_patch`` 替换为真正的
        # wrapper 或原版；委托给当前函数，避免在 trampoline 内复制拦截逻辑。
        try:
            import tools.terminal_tool as tt
            current = tt.terminal_tool
            if current is trampoline:
                # patch 未成功应用（异常路径），回退到原版
                return original_fn(*args, **kwargs)
            return current(*args, **kwargs)
        except Exception:
            return original_fn(*args, **kwargs)

    trampoline._gw_restart_trampoline = True
    return trampoline


def _install_trampoline() -> None:
    """安装轻量级 trampoline（不应用真正的 patch）。

    真正的 patch 延迟到首次 ``terminal_tool`` 调用时应用。本函数不执行
    ``inspect.getsource`` / ``import hermes_cli.gateway`` / 重试 sleep 等重
    操作，确保 ``register`` 阶段快速返回。即使 ``tools.terminal_tool`` 此
    时尚未导入，也只记录日志后立即返回，不阻塞。
    """
    try:
        import tools.terminal_tool as tt

        if not hasattr(tt, "terminal_tool"):
            logger.debug(
                "gateway-restart: tools.terminal_tool.terminal_tool not found (trampoline)"
            )
            return

        current = tt.terminal_tool
        if getattr(current, "_gw_restart_trampoline", False) or getattr(
            current, "_gw_restart_patched", False
        ):
            logger.debug(
                "gateway-restart: terminal_tool already has trampoline/patch"
            )
            return

        trampoline = _make_trampoline(current)
        tt.terminal_tool = trampoline
        logger.debug("gateway-restart: trampoline installed (lazy patch pending)")
    except ImportError:
        # 模块尚未加载 — 不重试，依赖后续 terminal_tool 调用时的 import 链
        logger.debug(
            "gateway-restart: tools.terminal_tool not available (trampoline deferred)"
        )
    except Exception as e:
        logger.debug("gateway-restart: trampoline install failed: %s", e)


# ── REGISTER ───────────────────────────────────────────────────────────────

def register(ctx=None) -> None:
    """插件入口 — 安装轻量级 trampoline，延迟应用 monkey-patch。

    真正的 patch（venv 检查、CLI patch、安装真正的 wrapper）延迟到首次
    ``terminal_tool`` 调用时应用，避免在 ``register`` 阶段执行
    ``inspect.getsource`` / ``import hermes_cli.gateway`` / 重试 sleep 等
    耗时操作。

    注意：``_HERMES_GATEWAY=1`` 的检查保留在 ``_make_patched_terminal_tool``
    创建的 wrapper 内部，确保只在 Gateway 进程内拦截命令。
    """
    global _PATCH_PENDING

    if _PATCH_APPLIED:
        return

    # 仅在 Linux 上有意义
    if not hasattr(_sig, "SIGUSR1"):
        logger.info(
            "gateway-restart: SIGUSR1 not available on this platform, skipping"
        )
        return

    # 标记 patch 待应用，安装轻量级 trampoline（不重试、不 sleep）
    _PATCH_PENDING = True
    _install_trampoline()


def unregister() -> None:
    """清理（可选 — 通常不需要）"""
    global _PATCH_APPLIED

    if not _PATCH_APPLIED:
        return

    try:
        import tools.terminal_tool as tt

        # 恢复原始函数（如果有 patched flag）
        current = tt.terminal_tool
        if hasattr(current, "_gw_restart_patched"):
            # 原始函数在 closure 中
            if hasattr(current, "__wrapped__"):
                tt.terminal_tool = current.__wrapped__
                logger.info("gateway-restart: terminal_tool restored")
    except Exception:
        pass

    _PATCH_APPLIED = False


def get_status() -> dict:
    """查询插件状态"""
    return {
        "patch_applied": _PATCH_APPLIED,
        "retries_attempted": _PATCH_RETRY_COUNT,
        "max_retries": _PATCH_MAX_RETRIES,
        "platform_supported": hasattr(_sig, "SIGUSR1"),
        "in_gateway": os.environ.get("_HERMES_GATEWAY") == "1",
        "venv_patch_exists": _terminal_tool_has_venv_patch() if not _PATCH_APPLIED else "N/A",
    }
