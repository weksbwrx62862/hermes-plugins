"""插件清单(plugin.yaml)静态校验测试

校验 plugin.yaml 的字段完整性、命名规范与版本格式, 不执行任何插件代码。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import yaml

from test_framework import PluginInfo, TestResult


# 版本号正则: 必须以 X.Y.Z 开头
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+")

# 已知合法钩子集合(从 hermes_cli/plugins.py 的 VALID_HOOKS 推断)
KNOWN_HOOKS = {
    "pre_tool_call", "post_tool_call",
    "transform_terminal_output", "transform_tool_result",
    "transform_llm_output", "pre_llm_call", "post_llm_call",
    "pre_api_request", "post_api_request",
    "on_session_start", "on_session_end", "on_session_finalize", "on_session_reset",
    "subagent_stop", "pre_gateway_dispatch",
    "pre_approval_request", "post_approval_response",
    "transform_request",
    "on_message", "on_request", "on_response",  # 兼容性扩展
}


def _load_manifest(plugin_info: PluginInfo) -> Dict[str, Any]:
    """读取并解析 plugin.yaml, 失败时返回空字典。"""
    try:
        text = Path(plugin_info.manifest_path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def test_manifest(plugin_info: PluginInfo) -> TestResult:
    """对 plugin.yaml 进行静态校验, 返回 TestResult。

    本函数不抛异常, 所有错误都记录到 TestResult 中。
    """
    import time
    start = time.perf_counter()
    details: Dict[str, Any] = {}
    errors: list = []
    warnings: list = []

    # 1. 解析 YAML
    try:
        text = Path(plugin_info.manifest_path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except Exception as exc:
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="manifest",
            status="FAIL",
            message=f"plugin.yaml 解析失败: {exc}",
            details={"exception": str(exc)},
            duration_ms=(time.perf_counter() - start) * 1000,
            exception=traceback_str(exc),
        )

    if not isinstance(data, dict):
        return TestResult(
            plugin_name=plugin_info.name,
            test_name="manifest",
            status="FAIL",
            message="plugin.yaml 顶层结构不是字典",
            details={"raw": repr(data)},
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    details["fields"] = list(data.keys())

    # 2. 必填字段校验: name / version / description
    name = data.get("name")
    version = data.get("version")
    description = data.get("description")

    if not name or not isinstance(name, str) or not name.strip():
        errors.append("name 字段缺失或为空")
    else:
        details["name"] = name

    if not version or not isinstance(version, str) or not version.strip():
        errors.append("version 字段缺失或为空")
    else:
        details["version"] = version

    if not description or not isinstance(description, str) or not description.strip():
        errors.append("description 字段缺失或为空")
    else:
        details["description_len"] = len(description)

    # 3. name 应等于目录名
    if isinstance(name, str) and name != plugin_info.name:
        errors.append(f"name 字段({name})与目录名({plugin_info.name})不一致")

    # 4. 版本格式校验
    if isinstance(version, str) and version.strip():
        if not _VERSION_RE.match(version.strip()):
            errors.append(f"version({version})不符合 X.Y.Z 格式")

    # 5. provides_tools 每项为非空字符串
    provides_tools = data.get("provides_tools", [])
    if provides_tools is not None:
        if not isinstance(provides_tools, list):
            errors.append("provides_tools 不是列表")
        else:
            details["provides_tools"] = provides_tools
            for t in provides_tools:
                if not isinstance(t, str) or not t.strip():
                    errors.append(f"provides_tools 中存在非法项: {t!r}")

    # 6. provides_hooks 每项为非空字符串, 未知钩子记为 WARN
    provides_hooks = data.get("provides_hooks", [])
    if provides_hooks is not None:
        if not isinstance(provides_hooks, list):
            errors.append("provides_hooks 不是列表")
        else:
            details["provides_hooks"] = provides_hooks
            for h in provides_hooks:
                if not isinstance(h, str) or not h.strip():
                    errors.append(f"provides_hooks 中存在非法项: {h!r}")
                elif h not in KNOWN_HOOKS:
                    warnings.append(f"未知钩子: {h}(不属于已知集合, 但仍记录)")

    # 7. dependencies 仅记录不强制安装
    deps = data.get("dependencies", [])
    if deps:
        details["dependencies"] = deps

    # 8. kind 校验(信息性)
    kind = data.get("kind", "standalone")
    details["kind"] = kind

    # 评估状态
    if errors:
        status = "FAIL"
        message = "; ".join(errors)
    elif warnings:
        status = "WARN"
        message = "; ".join(warnings)
    else:
        status = "PASS"
        message = "清单校验通过"

    if warnings and not errors:
        details["warnings"] = warnings

    return TestResult(
        plugin_name=plugin_info.name,
        test_name="manifest",
        status=status,
        message=message,
        details=details,
        duration_ms=(time.perf_counter() - start) * 1000,
    )


def traceback_str(exc: BaseException) -> str:
    """格式化异常 traceback。"""
    import traceback
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
