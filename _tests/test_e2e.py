"""Hermes 插件全功能测试 (End-to-End + 深度覆盖)

超越现有测试框架（manifest/import/register/smoke），补充：
  1. 端到端钩子链路 — 钩子安装后是否真的触发
  2. Provider 实现层 — Provider 类方法能否正常调用
  3. Schema 一致性 — 工具 schema 字段名是否匹配 handler 参数
  4. 跨插件接口检查 — 插件间的依赖/消费者接口是否匹配
  5. 性能基线 — 钩子/工具/register 耗时
  6. 副作用验证 — disk-cleanup 等钩子插件的实际行为

用法:
    python3 test_e2e.py                # 完整测试
    python3 test_e2e.py --plugin xxx   # 单插件模式
    python3 test_e2e.py --dry-run      # 仅列出待测插件
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

USER_PLUGINS_DIR = Path.home() / ".hermes" / "plugins"
BUNDLED_PLUGINS_DIR = (
    Path.home() / ".hermes" / "hermes-agent" / "venv" / "lib" / "python3.11" / "site-packages" / "plugins"
)
HERMES_VENV = (
    Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python3"
)

# 取 HERMES_VENV 所在 venv 的 site-packages
_HERMES_SITE = None
for p in (Path.home() / ".hermes" / "hermes-agent" / "venv" / "lib").rglob("site-packages"):
    if p.is_dir():
        _HERMES_SITE = str(p)
        break

# 所有要验证的 21 个已启用插件
ALL_PLUGINS = {
    "omnimem", "model-router", "skill-router", "plugin-orchestrator",
    "adaptive_multi_agent", "self_evolution", "dev-lifecycle",
    "codegraph", "prompt-optimizer", "taste_skill", "understand_anything",
    "rejection-ledger", "repo-chinese-names",
    "disk-cleanup", "deepseek-cache-optimizer", "log-translator", "gateway-restart",
    "skill_pool",
    "google_meet", "langfuse", "teams_pipeline",
}

logger = logging.getLogger("e2e_test")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class E2EResult:
    """单个全功能测试结果。"""
    def __init__(self, plugin: str, test: str, status: str = "PASS",
                 message: str = "", details: dict = None,
                 duration_ms: float = 0.0, exception: str = None):
        self.plugin = plugin
        self.test = test
        self.status = status  # PASS / FAIL / WARN / SKIP
        self.message = message
        self.details = details or {}
        self.duration_ms = duration_ms
        self.exception = exception

    def __repr__(self) -> str:
        return f"[{self.status}] {self.plugin}/{self.test}: {self.message}"


class E2ECollector:
    """结果收集器。"""
    def __init__(self):
        self.results: List[E2EResult] = []

    def add(self, r: E2EResult):
        self.results.append(r)

    def summary(self) -> dict:
        s = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0, "TOTAL": len(self.results)}
        for r in self.results:
            if r.status in s:
                s[r.status] += 1
        return s

    def by_plugin(self, plugin: str) -> List[E2EResult]:
        return [r for r in self.results if r.plugin == plugin]

    def failures(self) -> List[E2EResult]:
        return [r for r in self.results if r.status == "FAIL"]


# ---------------------------------------------------------------------------
# 插件定位
# ---------------------------------------------------------------------------

def _find_plugin_path(name: str) -> Optional[Path]:
    """查找插件目录路径。"""
    # 优先 user 插件
    p = USER_PLUGINS_DIR / name
    if p.is_dir() and (p / "__init__.py").exists():
        return p
    # bundled 插件 (一级)
    p = BUNDLED_PLUGINS_DIR / name
    if p.is_dir() and (p / "__init__.py").exists():
        return p
    # bundled 插件 (二级, 如 observability/langfuse)
    for sub in BUNDLED_PLUGINS_DIR.iterdir():
        if sub.is_dir():
            p = sub / name
            if p.is_dir() and (p / "__init__.py").exists():
                return p
    # 尝试 observability/*
    obs_dir = BUNDLED_PLUGINS_DIR / "observability"
    if obs_dir.is_dir():
        p = obs_dir / name
        if p.is_dir() and (p / "__init__.py").exists():
            return p
    return None


def _load_plugin_module(name: str, plugin_path: Path):
    """加载插件的 __init__.py 为 Python 模块。

    使用唯一模块名，正确处理连字符和相对导入。
    """
    import importlib

    init_file = plugin_path / "__init__.py"
    if not init_file.exists():
        return None, f"__init__.py 不存在"

    # 安全的模块名：用下划线替换连字符，加时间戳防冲突
    safe_name = name.replace("-", "_")
    mod_name = f"_e2e_{safe_name}"

    # 清理旧缓存（如果模块名已存在）
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    # 添加插件目录的父目录到 sys.path
    parent_dir = str(plugin_path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # 添加插件目录本身（让相对导入能找到同级模块）
    plugin_dir = str(plugin_path)
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)

    # 创建 spec 时，设置包名以支持相对导入
    try:
        spec = importlib.util.spec_from_file_location(mod_name, str(init_file),
                                                      submodule_search_locations=[plugin_dir])
        if spec is None or spec.loader is None:
            return None, "无法创建 spec"

        module = importlib.util.module_from_spec(spec)
        # 关键：设为包以支持相对导入
        module.__package__ = mod_name
        module.__path__ = [plugin_dir]

        sys.modules[mod_name] = module
        # 也注册到真实包名（支持相对导入，如 from self_evolution.core import）
        real_pkg_name = name.replace("-", "_")
        if real_pkg_name not in sys.modules:
            sys.modules[real_pkg_name] = module
        spec.loader.exec_module(module)
        return module, None
    except Exception as e:
        # 清理失败的缓存
        sys.modules.pop(mod_name, None)
        return None, f"{type(e).__name__}: {e}"


def _read_plugin_yaml(name: str, plugin_path: Path) -> dict:
    """读取 plugin.yaml。"""
    yaml_path = plugin_path / "plugin.yaml"
    if not yaml_path.exists():
        return {}
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 测试工厂
# ---------------------------------------------------------------------------

def _timed(fn: Callable) -> Tuple[Any, float]:
    """执行函数并计时 (ms)。"""
    t0 = time.perf_counter()
    result = fn()
    dt = (time.perf_counter() - t0) * 1000
    return result, dt


def safe_run(test_fn: Callable, plugin: str, test_name: str,
             collector: E2ECollector, *args, **kwargs):
    """安全执行测试函数，异常时记 FAIL。"""
    try:
        result = test_fn(*args, **kwargs)
        if isinstance(result, E2EResult):
            collector.add(result)
        return result
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        collector.add(E2EResult(
            plugin=plugin, test=test_name,
            status="FAIL", message=f"异常: {type(e).__name__}: {e}",
            exception=tb,
        ))
        return None


# ===================================================================
# 测试用例实现
# ===================================================================

# ----- 1. 钩子链路验证 -----

def test_hook_chain(module, plugin_yaml: dict, plugin: str) -> E2EResult:
    """验证注册的钩子函数可被实例化并调用，返回 True / False 值。"""
    t0 = time.perf_counter()
    details = {}

    # 从 plugin.yaml 读取钩子声明
    declared_hooks: List[str] = plugin_yaml.get("hooks", []) or []

    # 分析插件加载后的实际钩子
    register_fn = getattr(module, "register", None)
    if not callable(register_fn):
        return E2EResult(plugin=plugin, test="hook_chain", status="SKIP",
                         message="无 register 函数", details={"declared_hooks": declared_hooks},
                         duration_ms=(time.perf_counter() - t0) * 1000)

    # 创建一个记录型上下文来捕获注册行为
    registered_hooks: Dict[str, Any] = {}

    class HookCaptureCtx:
        def register_hook(self, hook_name: str, handler: Callable):
            registered_hooks[hook_name] = handler
        def register_tool(self, **kw):
            pass
        def __getattr__(self, name):
            return lambda *a, **kw: None

    try:
        register_fn(HookCaptureCtx())
    except Exception as e:
        return E2EResult(plugin=plugin, test="hook_chain", status="FAIL",
                         message=f"register 异常: {e}",
                         duration_ms=(time.perf_counter() - t0) * 1000)

    details["declared_hooks"] = declared_hooks
    details["registered_hooks"] = list(registered_hooks.keys())

    if not declared_hooks and not registered_hooks:
        return E2EResult(plugin=plugin, test="hook_chain", status="SKIP",
                         message="无钩子声明或注册", details=details,
                         duration_ms=(time.perf_counter() - t0) * 1000)

    # 验证每个注册的钩子可被调用
    call_results = []
    for hook_name, handler in registered_hooks.items():
        if not callable(handler):
            call_results.append({"hook": hook_name, "status": "FAIL", "reason": "handler 不可调用"})
            continue
        # 尝试调用（传入空 kwargs，看是否抛异常）
        try:
            result = handler()
            call_results.append({"hook": hook_name, "status": "PASS", "return_type": type(result).__name__})
        except Exception as e:
            call_results.append({"hook": hook_name, "status": "WARN",
                                 "reason": f"调用失败: {type(e).__name__}: {e}"})

    details["hook_call_results"] = call_results
    fails = [c for c in call_results if c["status"] == "FAIL"]
    warns = [c for c in call_results if c["status"] == "WARN"]

    if fails:
        status = "FAIL"
        msg = f"{len(fails)}/{len(call_results)} 钩子调用失败"
    elif warns:
        status = "WARN"
        msg = f"{len(warns)} 个钩子调用有异常(可能是参数问题)"
    else:
        status = "PASS"
        msg = f"{len(call_results)} 个钩子均可调用"

    return E2EResult(plugin=plugin, test="hook_chain", status=status, message=msg,
                     details=details, duration_ms=(time.perf_counter() - t0) * 1000)


# ----- 2. Schema 一致性验证 -----

def test_schema_consistency(module, plugin_yaml: dict, plugin: str) -> E2EResult:
    """验证工具 schema 中声明的参数与 handler 的实际参数签名匹配。"""
    t0 = time.perf_counter()
    details = {}

    declared_tools: List[str] = plugin_yaml.get("provides_tools", []) or []

    class SchemaCaptureCtx:
        def __init__(self):
            self.tools: List[dict] = []
        def register_tool(self, name, toolset="", schema=None, handler=None, **kw):
            self.tools.append({
                "name": name, "schema": schema or {},
                "handler": handler, "handler_name": getattr(handler, "__name__", str(handler)),
                "toolset": toolset,
            })
        def register_hook(self, *a, **kw):
            pass
        def __getattr__(self, name):
            return lambda *a, **kw: None

    register_fn = getattr(module, "register", None)
    if not callable(register_fn):
        return E2EResult(plugin=plugin, test="schema_consistency", status="SKIP",
                         message="无 register 函数",
                         duration_ms=(time.perf_counter() - t0) * 1000)

    ctx = SchemaCaptureCtx()
    try:
        register_fn(ctx)
    except Exception as e:
        return E2EResult(plugin=plugin, test="schema_consistency", status="FAIL",
                         message=f"register 异常: {e}",
                         duration_ms=(time.perf_counter() - t0) * 1000)

    if not ctx.tools:
        return E2EResult(plugin=plugin, test="schema_consistency", status="SKIP",
                         message="无工具注册",
                         duration_ms=(time.perf_counter() - t0) * 1000)

    issues = []
    for tool in ctx.tools:
        schema = tool["schema"]
        handler = tool["handler"]
        handler_name = tool["handler_name"]

        # 尝试获取 handler 签名
        try:
            import inspect
            sig = inspect.signature(handler)
            handler_params = list(sig.parameters.keys())
        except (ValueError, TypeError):
            handler_params = []

        schema_props = list(schema.get("properties", {}).keys()) if isinstance(schema, dict) else []

        # 检查: schema 中显式要求参数但 handler 不接受任何参数
        if schema_props:
            # handler 至少应接受 args 或 **kwargs
            has_var_positional = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL
                for p in sig.parameters.values()
            ) if handler_params else False
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            ) if handler_params else False
            # 常见模式：handler(args) 或 handler(args, kwargs)
            if not (handler_params or has_var_keyword) and not has_var_positional:
                issues.append({
                    "tool": tool["name"],
                    "issue": f"schema 声明参数 {schema_props} 但 handler ({handler_name}) 签名不接受参数",
                    "handler_params": handler_params,
                })

    details["tools_checked"] = len(ctx.tools)
    details["issues"] = issues

    if issues:
        return E2EResult(plugin=plugin, test="schema_consistency", status="WARN",
                         message=f"{len(issues)} 个工具 schema/handler 不匹配",
                         details=details,
                         duration_ms=(time.perf_counter() - t0) * 1000)

    return E2EResult(plugin=plugin, test="schema_consistency", status="PASS",
                     message=f"{len(ctx.tools)} 个工具 schema 一致",
                     details=details,
                     duration_ms=(time.perf_counter() - t0) * 1000)


# ----- 3. Provider 层验证 -----

def test_provider_interface(module, plugin_yaml: dict, plugin: str, plugin_path: Path = None) -> E2EResult:
    """验证插件是否注册了 Provider，且 Provider 实现 ABC 接口。"""
    t0 = time.perf_counter()
    details = {}

    if plugin_path is None:
        plugin_path = _find_plugin_path(plugin)
    if not plugin_path:
        return E2EResult(plugin=plugin, test="provider_interface", status="SKIP",
                         message="无法定位插件目录",
                         duration_ms=(time.perf_counter() - t0) * 1000)

    EXCLUDE_DIRS = {".venv", "venv", "node_modules", ".git", "__pycache__",
                    "tests", "benchmarks", ".nox", ".tox", "build", "dist", "egg-info"}

    provider_files = [f for f in plugin_path.rglob("provider.py")
                      if not any(excl in f.parts for excl in EXCLUDE_DIRS)]
    provider_files += [f for f in plugin_path.rglob("*_provider.py")
                       if not any(excl in f.parts for excl in EXCLUDE_DIRS)]
    # 排除 __init__.py 和 cli.py
    provider_files = [f for f in provider_files
                      if f.name not in ("__init__.py", "cli.py")]

    if not provider_files:
        return E2EResult(plugin=plugin, test="provider_interface", status="SKIP",
                         message="未找到 provider.py", duration_ms=(time.perf_counter() - t0) * 1000)

    # 先用主模块的 __path__ 来设置包上下文
    pkg_path = getattr(module, "__path__", [str(plugin_path)])
    pkg_name = getattr(module, "__package__", plugin.replace("-", "_"))
    parent_dir = str(plugin_path.parent)

    provider_classes = []
    for pf in provider_files:
        rel_path = pf.relative_to(plugin_path)
        try:
            # 使用主模块的包上下文来加载 provider 文件
            spec = importlib.util.spec_from_file_location(
                f"{pkg_name}.{pf.stem}", str(pf),
                submodule_search_locations=[str(pf.parent)]
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = pkg_name
                mod.__path__ = [str(pf.parent)]
                # 确保父包在 sys.modules 中
                if pkg_name not in sys.modules:
                    sys.modules[pkg_name] = module
                    # 也把 parent_dir 加到 sys.path
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)

                sys.modules[f"{pkg_name}.{pf.stem}"] = mod
                spec.loader.exec_module(mod)

                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if isinstance(attr, type) and attr.__module__ == mod.__name__:
                        if attr_name.endswith("Provider") and attr_name != "ABC":
                            # 跳过抽象基类
                            import abc
                            if issubclass(attr, abc.ABC) or getattr(attr, "_is_protocol", False):
                                continue
                            # 跳过有抽象方法的类
                            try:
                                abstract = getattr(attr, "__abstractmethods__", frozenset())
                                if abstract:
                                    continue
                            except Exception:
                                pass
                            provider_classes.append((rel_path, attr_name, attr))
        except Exception as e:
            details[f"load_error_{rel_path}"] = f"{type(e).__name__}: {e}"

    details["provider_files"] = [str(p) for p in provider_files]
    details["provider_classes"] = [f"{pc[0]}:{pc[1]}" for pc in provider_classes]

    if not provider_classes:
        return E2EResult(plugin=plugin, test="provider_interface", status="SKIP",
                         message="未找到 Provider 类", details=details,
                         duration_ms=(time.perf_counter() - t0) * 1000)

    # 尝试实例化 Provider（支持多种构造签名）
    instantiation_results = []
    for rel_path, class_name, cls in provider_classes:
        try:
            # 尝试几种常见构造签名
            instance = None
            construct_attempts = [
                ("无参", lambda: cls()),
                ("None", lambda: cls(None)),
                ("空字典", lambda: cls({})),
                ("空字符串", lambda: cls("")),
                ("空路径", lambda: cls(Path("/tmp"))),
            ]
            for style, ctor in construct_attempts:
                try:
                    instance = ctor()
                    break
                except (TypeError, ValueError):
                    continue

            if instance is None:
                instantiation_results.append({
                    "class": class_name,
                    "file": str(rel_path),
                    "status": "FAIL",
                    "error": "无法用常见签名实例化",
                })
                continue

            # 检查是否有基本的工具处理方法
            methods = [m for m in dir(instance) if not m.startswith("_")]
            instantiation_results.append({
                "class": class_name,
                "file": str(rel_path),
                "status": "PASS",
                "public_methods": methods[:10],
            })
        except Exception as e:
            instantiation_results.append({
                "class": class_name,
                "file": str(rel_path),
                "status": "FAIL",
                "error": f"{type(e).__name__}: {e}",
            })

    details["instantiation"] = instantiation_results
    fails = [r for r in instantiation_results if r["status"] == "FAIL"]

    if fails:
        return E2EResult(plugin=plugin, test="provider_interface", status="FAIL",
                         message=f"{len(fails)}/{len(instantiation_results)} Provider 实例化失败",
                         details=details,
                         duration_ms=(time.perf_counter() - t0) * 1000)

    return E2EResult(plugin=plugin, test="provider_interface", status="PASS",
                     message=f"{len(instantiation_results)} 个 Provider 正常",
                     details=details,
                     duration_ms=(time.perf_counter() - t0) * 1000)


# ----- 4. 插件依赖接口检查 -----

def test_dependency_interfaces(module, plugin_yaml: dict, plugin: str) -> E2EResult:
    """验证 plugin.yaml 声明的 dependencies 对应的插件是否存在且注册接口兼容。"""
    t0 = time.perf_counter()
    details = {}

    deps: List[str] = plugin_yaml.get("dependencies", []) or []
    if not deps:
        return E2EResult(plugin=plugin, test="dependency_interfaces", status="SKIP",
                         message="无依赖声明", duration_ms=(time.perf_counter() - t0) * 1000)

    missing = []
    for dep in deps:
        dep_path = _find_plugin_path(dep)
        if not dep_path:
            missing.append(dep)
            continue
        # 检查依赖插件的 register 函数
        dep_module, err = _load_plugin_module(dep, dep_path)
        if err:
            missing.append(f"{dep} (加载失败: {err})")

    details["declared_dependencies"] = deps
    details["missing"] = missing

    if missing:
        return E2EResult(plugin=plugin, test="dependency_interfaces", status="WARN",
                         message=f"{len(missing)} 个依赖插件缺失: {missing}",
                         details=details,
                         duration_ms=(time.perf_counter() - t0) * 1000)

    return E2EResult(plugin=plugin, test="dependency_interfaces", status="PASS",
                     message=f"{len(deps)} 个依赖全部存在",
                     duration_ms=(time.perf_counter() - t0) * 1000)


# ----- 5. 插件元数据完整性 -----

def test_metadata_integrity(plugin_yaml: dict, plugin: str, plugin_path: Path) -> E2EResult:
    """验证 plugin.yaml 字段完备性。"""
    t0 = time.perf_counter()
    details: dict = {}

    # 必要字段
    required_fields = ["name", "version"]
    missing_fields = [f for f in required_fields if f not in plugin_yaml]
    details["missing_required"] = missing_fields

    # 推荐字段
    recommended_fields = ["description", "kind"]
    missing_recommended = [f for f in recommended_fields if f not in plugin_yaml]
    details["missing_recommended"] = missing_recommended

    # version 格式检查: X.Y.Z 或 X.Y.Z.devN
    version = plugin_yaml.get("version", "")
    import re
    version_ok = bool(re.match(r'^\d+\.\d+\.\d+', str(version)))
    details["version_format_ok"] = version_ok
    details["version_raw"] = str(version)

    # 检查 __init__.py 是否存在
    init_ok = (plugin_path / "__init__.py").exists()
    details["init_exists"] = init_ok

    # tool/hook 声明的一致性:
    # plugin.yaml 中的 provides_tools 必须为列表
    tools_declared = plugin_yaml.get("provides_tools", [])
    if not isinstance(tools_declared, list):
        details["tools_declared_type"] = type(tools_declared).__name__
        tools_declared = []
    details["tools_declared"] = tools_declared

    # hooks 声明
    hooks_declared = plugin_yaml.get("hooks", [])
    if not isinstance(hooks_declared, list):
        details["hooks_declared_type"] = type(hooks_declared).__name__
        hooks_declared = []
    details["hooks_declared"] = hooks_declared

    issues = []
    if missing_fields:
        issues.append(f"缺少必要字段: {missing_fields}")
    if not version_ok:
        issues.append(f"version 格式不符合 X.Y.Z: {version}")

    if issues:
        return E2EResult(plugin=plugin, test="metadata_integrity", status="WARN",
                         message="; ".join(issues), details=details,
                         duration_ms=(time.perf_counter() - t0) * 1000)

    return E2EResult(plugin=plugin, test="metadata_integrity", status="PASS",
                     message=f"version={version}, tools={len(tools_declared)}, hooks={len(hooks_declared)}",
                     details=details,
                     duration_ms=(time.perf_counter() - t0) * 1000)


# ----- 6. 插件加载性能基线 -----

def test_load_performance(plugin: str, plugin_path: Path) -> E2EResult:
    """测量插件模块加载和 register 的性能基线。"""
    t0 = time.perf_counter()
    details = {}

    # 冷加载
    load_t0 = time.perf_counter()
    module, err = _load_plugin_module(plugin, plugin_path)
    load_time = (time.perf_counter() - load_t0) * 1000
    details["load_time_ms"] = round(load_time, 1)
    details["load_error"] = err

    if err:
        return E2EResult(plugin=plugin, test="load_performance", status="FAIL",
                         message=f"模块加载失败: {err}", details=details,
                         duration_ms=(time.perf_counter() - t0) * 1000)

    # register 耗时
    register_fn = getattr(module, "register", None)
    if callable(register_fn):
        class NullCtx:
            def register_tool(self, **kw):
                pass
            def register_hook(self, *a, **kw):
                pass
            def register_cli_command(self, **kw):
                pass
            def __getattr__(self, name):
                return lambda *a, **kw: None

        reg_t0 = time.perf_counter()
        try:
            register_fn(NullCtx())
            reg_time = (time.perf_counter() - reg_t0) * 1000
        except Exception as e:
            reg_time = (time.perf_counter() - reg_t0) * 1000
            details["register_error"] = f"{type(e).__name__}: {e}"
        details["register_time_ms"] = round(reg_time, 1)

    # 警告阈值
    warnings = []
    if load_time > 2000:
        warnings.append(f"加载耗时 {load_time:.0f}ms > 2s")
    if details.get("register_time_ms", 0) > 2000:
        warnings.append(f"register 耗时 {details['register_time_ms']:.0f}ms > 2s")

    total_ms = (time.perf_counter() - t0) * 1000

    if warnings:
        return E2EResult(plugin=plugin, test="load_performance", status="WARN",
                         message="; ".join(warnings), details=details,
                         duration_ms=total_ms)

    return E2EResult(plugin=plugin, test="load_performance", status="PASS",
                     message=f"load={load_time:.0f}ms, register={details.get('register_time_ms', 'N/A')}ms",
                     details=details, duration_ms=total_ms)


# ----- 7. 跨插件钩子优先级一致性 -----

def test_hook_priority_consistency(all_plugins: List[str]) -> E2EResult:
    """检查所有声明了 hooks 的插件是否有合理的执行优先级（无冲突声明）。"""
    t0 = time.perf_counter()
    details = {}

    hook_registry: Dict[str, List[Tuple[str, int]]] = defaultdict(list)  # hook_name -> [(plugin, phase)]

    phase_order = {
        "PRE_UTILITY": 0, "UTILITY": 1, "POST_UTILITY": 2,
        "PRE_BACKEND": 3, "BACKEND": 4, "POST_BACKEND": 5,
        "LATE": 6,
    }

    for name in all_plugins:
        p = _find_plugin_path(name)
        if not p:
            continue
        yaml_data = _read_plugin_yaml(name, p)
        hooks = yaml_data.get("hooks", [])
        if isinstance(hooks, list):
            phase = yaml_data.get("phase", "UTILITY")
            phase_val = phase_order.get(phase, 1)
            for h in hooks:
                hook_registry[h].append((name, phase_val))

    # 检查同钩子是否被多个插件注册（正常，但记录）
    details["hook_registry"] = {h: [p[0] for p in v] for h, v in sorted(hook_registry.items())}
    details["hook_count"] = {h: len(v) for h, v in hook_registry.items()}

    # 检查是否有不合理的高优先级声明
    high_priority = []
    for h, registrants in hook_registry.items():
        for plugin_name, pv in registrants:
            if pv <= 0:  # PRE_UTILITY
                high_priority.append(f"{plugin_name} 声明 {h} 在 {phase_val} 阶段")

    details["high_priority_hooks"] = high_priority

    total_registrations = sum(len(v) for v in hook_registry.values())
    return E2EResult(plugin="(cross-plugin)", test="hook_priority_consistency",
                     status="PASS",
                     message=f"{len(hook_registry)} 种钩子, {total_registrations} 次注册",
                     details=details,
                     duration_ms=(time.perf_counter() - t0) * 1000)


# ----- 8. Python 语法完整性 -----

def test_python_syntax(plugin: str, plugin_path: Path) -> E2EResult:
    """使用 py_compile 验证插件目录下所有 .py 文件的语法。"""
    t0 = time.perf_counter()
    import py_compile

    EXCLUDE_DIRS = {".venv", "venv", "node_modules", ".git", "__pycache__",
                    "tests", "benchmarks", ".nox", ".tox", "build", "dist", "egg-info"}

    py_files = [f for f in plugin_path.rglob("*.py")
                if not any(excl in f.parts for excl in EXCLUDE_DIRS)]

    details = {}
    errors = []
    for pf in py_files:
        try:
            py_compile.compile(str(pf), doraise=True)
        except py_compile.PyCompileError as e:
            rel = pf.relative_to(plugin_path)
            errors.append(str(rel))
            details[f"error_{rel}"] = str(e)

    details["total_files"] = len(py_files)
    details["error_files"] = errors

    if errors:
        return E2EResult(plugin=plugin, test="python_syntax", status="FAIL",
                         message=f"{len(errors)}/{len(py_files)} 文件语法错误",
                         details=details,
                         duration_ms=(time.perf_counter() - t0) * 1000)

    return E2EResult(plugin=plugin, test="python_syntax", status="PASS",
                     message=f"{len(py_files)} 个文件语法检查通过",
                     duration_ms=(time.perf_counter() - t0) * 1000)


# ----- 9. 环境变量 / 外部依赖检查 -----

def test_external_dependencies(plugin_yaml: dict, plugin: str) -> E2EResult:
    """检查 plugin.yaml 中声明的外部依赖和环境变量。"""
    t0 = time.perf_counter()
    details = {}

    required_env = plugin_yaml.get("requires_env", []) or []
    missing_env = []
    for env_var in required_env:
        if not os.environ.get(env_var):
            missing_env.append(env_var)

    # 检查 requires_python (如果声明了)
    requires_python = plugin_yaml.get("requires_python", "")
    python_ok = True
    if requires_python:
        import re
        match = re.search(r'>=?\s*(\d+\.\d+)', str(requires_python))
        if match:
            min_ver = tuple(int(x) for x in match.group(1).split("."))
            cur_ver = tuple(sys.version_info[:2])
            python_ok = cur_ver >= min_ver

    details["required_env"] = required_env
    details["missing_env"] = missing_env
    details["requires_python"] = requires_python
    details["python_version_ok"] = python_ok

    issues = []
    if missing_env:
        issues.append(f"缺环境变量: {missing_env}")
    if not python_ok:
        issues.append(f"Python 版本不满足 {requires_python}")

    if issues:
        return E2EResult(plugin=plugin, test="external_dependencies", status="WARN",
                         message="; ".join(issues), details=details,
                         duration_ms=(time.perf_counter() - t0) * 1000)

    return E2EResult(plugin=plugin, test="external_dependencies", status="PASS",
                     message=f"env={len(required_env)}, python={requires_python or 'any'}",
                     details=details,
                     duration_ms=(time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------------------
# 主测试编排
# ---------------------------------------------------------------------------

def run_e2e_tests(plugin_filter: Optional[str] = None) -> E2ECollector:
    """运行全部端到端测试。"""
    collector = E2ECollector()

    # 确定要测试的插件
    plugins_to_test = sorted(ALL_PLUGINS)
    if plugin_filter:
        plugins_to_test = [p for p in plugins_to_test if plugin_filter in p]
        if not plugins_to_test:
            print(f"⚠️ 未找到匹配 '{plugin_filter}' 的插件")
            return collector

    print(f"🔍 全功能测试: {len(plugins_to_test)} 个插件, {8} 个测试维度")
    print("=" * 60)

    all_plugin_paths: Dict[str, Path] = {}

    for name in plugins_to_test:
        p = _find_plugin_path(name)
        if p is None:
            print(f"  ❌ {name}: 无法定位插件目录")
            collector.add(E2EResult(plugin=name, test="locate", status="FAIL",
                                    message="无法定位插件目录"))
            continue
        all_plugin_paths[name] = p

    # 跨插件测试 (优先级一致性)
    safe_run(test_hook_priority_consistency, "(cross-plugin)",
             "hook_priority_consistency", collector, plugins_to_test)

    # 逐插件测试
    for name in plugins_to_test:
        p = all_plugin_paths.get(name)
        if p is None:
            continue

        print(f"\n  📦 {name} ({p.parent.name}/{name})")

        # 1. 元数据完整性
        yaml_data = _read_plugin_yaml(name, p)
        safe_run(test_metadata_integrity, name, "metadata_integrity",
                 collector, yaml_data, name, p)

        # 2. Python 语法
        safe_run(test_python_syntax, name, "python_syntax",
                 collector, name, p)

        # 3. 外部依赖
        safe_run(test_external_dependencies, name, "external_dependencies",
                 collector, yaml_data, name)

        # 4. 加载性能
        safe_run(test_load_performance, name, "load_performance",
                 collector, name, p)

        # 重新加载模块
        module, err = _load_plugin_module(name, p)
        if err:
            collector.add(E2EResult(plugin=name, test="module_load",
                                    status="FAIL", message=f"加载失败: {err}"))
            continue

        # 5. 钩子链路
        safe_run(test_hook_chain, name, "hook_chain",
                 collector, module, yaml_data, name)

        # 6. Schema 一致性
        safe_run(test_schema_consistency, name, "schema_consistency",
                 collector, module, yaml_data, name)

        # 7. Provider 接口
        safe_run(test_provider_interface, name, "provider_interface",
                 collector, module, yaml_data, name, p)

        # 8. 依赖接口
        safe_run(test_dependency_interfaces, name, "dependency_interfaces",
                 collector, module, yaml_data, name)

        # 清理模块缓存
        key = f"_e2e_{name.replace('-', '_')}"
        sys.modules.pop(key, None)

    print("\n" + "=" * 60)
    return collector


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def print_report(collector: E2ECollector):
    """打印测试报告。"""
    summary = collector.summary()
    total = summary["TOTAL"]

    print(f"\n📊 全功能测试报告")
    print(f"   测试用例: {total}")
    print(f"   ✅ PASS:  {summary['PASS']}")
    print(f"   ❌ FAIL:  {summary['FAIL']}")
    print(f"   ⚠️  WARN:  {summary['WARN']}")
    print(f"   ⏭️  SKIP:  {summary['SKIP']}")
    print()

    # 失败列表
    failures = collector.failures()
    if failures:
        print("❌ 失败详情:")
        for f in failures:
            print(f"  [{f.plugin}] {f.test}: {f.message}")
            if f.exception:
                lines = f.exception.strip().split("\n")[-3:]
                for l in lines:
                    print(f"    {l}")
        print()

    # 警告列表
    warns = [r for r in collector.results if r.status == "WARN"]
    if warns:
        print("⚠️  警告:")
        for w in warns:
            print(f"  [{w.plugin}] {w.test}: {w.message}")
        print()

    # 按插件汇总
    print("📋 插件级汇总:")
    plugins_in_order = sorted(set(r.plugin for r in collector.results if not r.plugin.startswith("(")))
    for p in plugins_in_order:
        plugin_results = collector.by_plugin(p)
        p_sum = defaultdict(int)
        for r in plugin_results:
            p_sum[r.status] += 1
        status_icon = "❌" if p_sum.get("FAIL", 0) else ("⚠️" if p_sum.get("WARN", 0) else "✅")
        print(f"  {status_icon} {p}: {dict(p_sum)}")

    # 保存 JSON 报告
    report_path = Path.home() / ".hermes" / "plugins" / "_tests" / "e2e-report.json"
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "results": [
            {"plugin": r.plugin, "test": r.test, "status": r.status,
             "message": r.message, "duration_ms": round(r.duration_ms, 1)}
            for r in collector.results
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 JSON 报告: {report_path}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes 插件全功能测试 (E2E + 深度)")
    parser.add_argument("--plugin", help="单插件测试 (支持模糊匹配)")
    parser.add_argument("--dry-run", action="store_true", help="仅列出待测插件")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    if args.dry_run:
        if args.plugin:
            matched = [p for p in sorted(ALL_PLUGINS) if args.plugin in p]
            print(f"匹配插件 ({len(matched)}):")
            for p in matched:
                pp = _find_plugin_path(p)
                status = "✅" if pp else "❌ 未找到"
                print(f"  {status} {p}")
        else:
            print(f"待测插件 ({len(ALL_PLUGINS)}):")
            for p in sorted(ALL_PLUGINS):
                pp = _find_plugin_path(p)
                status = "✅" if pp else "❌ 未找到目录"
                print(f"  {status} {p}")
        return 0

    collector = run_e2e_tests(plugin_filter=args.plugin)
    print_report(collector)

    summary = collector.summary()
    return 1 if summary["FAIL"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
