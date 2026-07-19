"""测试框架核心模块

提供 Hermes 插件测试所需的基础组件:
- MockPluginContext: 模拟 Hermes 的 PluginContext,记录插件注册行为
- PluginDiscovery: 扫描插件目录,发现待测试插件
- TestResult: 单个测试结果的数据结构
- ResultCollector: 收集并聚合所有测试结果
"""

from __future__ import annotations

import logging
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Mock 配置对象
# ---------------------------------------------------------------------------

class _MockConfig:
    """Mock 配置对象。

    支持 cfg_get 风格的链式访问: 任何缺失的键都返回另一个 MockConfig,
    最终在 cfg_get 取 default 时返回默认值。整体表现为"空配置"。
    """

    __slots__ = ()

    def get(self, key: Any, default: Any = None) -> Any:
        # 模拟空配置: 任何 get 都返回默认值
        return default

    def __getitem__(self, key: Any) -> "_MockConfig":
        # 链式访问不抛异常, 返回自身类型的实例
        return _MockConfig()

    def __contains__(self, key: Any) -> bool:
        return False

    def __bool__(self) -> bool:
        return False

    def __iter__(self):
        return iter(())

    def items(self):
        return []

    def keys(self):
        return []

    def values(self):
        return []


# ---------------------------------------------------------------------------
# MockPluginContext
# ---------------------------------------------------------------------------

class MockPluginContext:
    """模拟 Hermes 的 PluginContext。

    记录插件通过 register_* 系列方法注册的所有内容, 不真实接入 Hermes 内部。
    对未显式定义的属性访问(如 register_code_intelligence_provider)返回
    一个记录型可调用对象, 保证插件 register(ctx) 不会因缺少方法而崩溃。
    """

    # 已知的只读属性, __getattr__ 不应拦截这些名称
    _KNOWN_ATTRS = {
        "registered_tools", "registered_hooks", "registered_commands",
        "registered_cli_commands", "registered_skills", "registered_platforms",
        "registered_providers", "call_history", "_record",
        "logger", "_config", "manifest", "plugin_name",
        "event_loop", "db", "cache", "_llm",
    }

    def __init__(self, plugin_name: str = "mock_plugin") -> None:
        # 已注册工具: name -> handler 函数
        self.registered_tools: Dict[str, Any] = {}
        # 已注册钩子: hook_name -> [callback, ...]
        self.registered_hooks: Dict[str, List[Callable]] = {}
        # 已注册斜杠命令: name -> handler
        self.registered_commands: Dict[str, Any] = {}
        # 已注册 CLI 命令
        self.registered_cli_commands: Dict[str, Any] = {}
        # 已注册技能
        self.registered_skills: Dict[str, Any] = {}
        # 已注册平台适配器
        self.registered_platforms: Dict[str, Any] = {}
        # 已注册提供者(image_gen/video_gen/web_search/browser 等)
        self.registered_providers: Dict[str, Any] = {}
        # 全部方法调用历史
        self.call_history: List[Dict[str, Any]] = []
        # 日志器: 附加 NullHandler, 避免无 handler 警告
        self.logger: logging.Logger = logging.getLogger(f"mock_plugin.{plugin_name}")
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())
        # mock 配置对象
        self._config = _MockConfig()
        # 插件名(用于日志与上下文)
        self.plugin_name = plugin_name
        # manifest 占位
        self.manifest: Any = None
        # 插件可能访问的其他属性, 返回 None
        self.event_loop: Any = None
        self.db: Any = None
        self.cache: Any = None
        self._llm: Any = None

    # -- 内部记录 ----------------------------------------------------------

    def _record(self, method: str, **info: Any) -> None:
        """记录一次方法调用到 call_history。"""
        entry = {"method": method}
        entry.update(info)
        self.call_history.append(entry)

    # -- 工具注册 ----------------------------------------------------------
    # 兼容多种调用签名:
    #   1. 任务规格: register_tool(name, func, description=None, schema=None)
    #   2. 真实 PluginContext: register_tool(name, toolset, schema, handler, ...)
    #   3. 关键字风格: register_tool(name=..., handler=..., schema=..., toolset=...)

    def register_tool(
        self,
        name: str,
        func: Any = None,
        description: Any = None,
        schema: Any = None,
        **kwargs: Any,
    ) -> None:
        # 解析真正的 handler 函数
        handler: Optional[Callable] = None
        if callable(func):
            handler = func
        elif "handler" in kwargs:
            handler = kwargs.pop("handler")
        if handler is None:
            # 真实签名 (name, toolset, schema, handler) 时,
            # func=toolset(字符串), description=schema(dict), schema=handler(callable)
            for candidate in (description, schema):
                if callable(candidate):
                    handler = candidate
                    break
        # 归一化 description 与 schema
        desc = description if isinstance(description, str) else (kwargs.get("description", "") or kwargs.get("help", ""))
        sch = schema if isinstance(schema, (dict, type(None))) else kwargs.get("schema")
        # 记录
        self.registered_tools[name] = handler
        self._record(
            "register_tool",
            name=name,
            func=handler,
            description=desc,
            schema=sch,
            toolset=kwargs.get("toolset"),
        )

    # -- 钩子注册 ----------------------------------------------------------

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        self.registered_hooks.setdefault(hook_name, []).append(callback)
        self._record("register_hook", hook_name=hook_name, callback=callback)

    # -- 命令注册 ----------------------------------------------------------

    def register_command(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        **kwargs: Any,
    ) -> None:
        self.registered_commands[name] = {
            "handler": handler,
            "description": description,
            "args_hint": kwargs.get("args_hint", ""),
        }
        self._record(
            "register_command",
            name=name,
            handler=handler,
            description=description,
        )

    def register_cli_command(
        self,
        name: str,
        help: str = "",
        setup_fn: Any = None,
        handler_fn: Any = None,
        description: str = "",
        **kwargs: Any,
    ) -> None:
        self.registered_cli_commands[name] = {
            "help": help,
            "setup_fn": setup_fn,
            "handler_fn": handler_fn,
            "description": description,
        }
        self._record("register_cli_command", name=name, help=help)

    # -- 斜杠命令 / MCP 工具等 (接受任意参数, 仅记录) ----------------------

    def register_slash_command(self, *args: Any, **kwargs: Any) -> None:
        self._record("register_slash_command", args=args, kwargs=kwargs)

    def register_mcp_tool(self, *args: Any, **kwargs: Any) -> None:
        self._record("register_mcp_tool", args=args, kwargs=kwargs)

    def register_skill(self, name: str, path: Any = None, description: str = "", **kwargs: Any) -> None:
        self.registered_skills[name] = {"path": path, "description": description}
        self._record("register_skill", name=name, path=path, description=description)

    def register_platform(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.registered_platforms[name] = {"args": args, "kwargs": kwargs}
        self._record("register_platform", name=name)

    def register_context_engine(self, engine: Any) -> None:
        self.registered_providers["context_engine"] = engine
        self._record("register_context_engine")

    def register_image_gen_provider(self, provider: Any) -> None:
        self.registered_providers["image_gen"] = provider
        self._record("register_image_gen_provider")

    def register_video_gen_provider(self, provider: Any) -> None:
        self.registered_providers["video_gen"] = provider
        self._record("register_video_gen_provider")

    def register_web_search_provider(self, provider: Any) -> None:
        self.registered_providers["web_search"] = provider
        self._record("register_web_search_provider")

    def register_browser_provider(self, provider: Any) -> None:
        self.registered_providers["browser"] = provider
        self._record("register_browser_provider")

    # -- 消息注入 / 工具分发 (mock 实现, 不产生副作用) ----------------------

    def inject_context(self, hint: str) -> bool:
        self._record("inject_context", hint=hint)
        return True

    def inject_message(self, content: str, role: str = "user") -> bool:
        self._record("inject_message", content=content, role=role)
        return True

    def dispatch_tool(self, tool_name: str, args: dict, **kwargs: Any) -> str:
        self._record("dispatch_tool", tool_name=tool_name, args=args)
        return "{}"

    # -- 属性访问 ----------------------------------------------------------

    @property
    def hermes_home(self) -> Path:
        # Hermes 主目录
        return Path.home() / ".hermes"

    @property
    def config(self) -> _MockConfig:
        # mock 配置: 任何访问都返回默认值
        return self._config

    @property
    def llm(self) -> "_MockLlm":
        # mock LLM 门面, 避免插件访问 ctx.llm 时崩溃
        if self._llm is None:
            self._llm = _MockLlm()
        return self._llm

    def __getattr__(self, name: str) -> Any:
        # __getattr__ 仅在常规查找失败时触发。
        # 对未定义的 register_* 等方法返回记录型可调用对象,
        # 保证插件调用 ctx.<未知方法>(...) 时不会抛 AttributeError。
        if name.startswith("__") or name in self._KNOWN_ATTRS:
            raise AttributeError(name)

        def _recorder(*args: Any, **kwargs: Any) -> Any:
            self._record(name, args=args, kwargs=kwargs)
            return None

        return _recorder


class _MockLlm:
    """Mock LLM 门面: 任意调用都返回空结果。"""

    def chat(self, *args: Any, **kwargs: Any) -> str:
        return ""

    def complete(self, *args: Any, **kwargs: Any) -> str:
        return ""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)

        def _noop(*args: Any, **kwargs: Any) -> Any:
            return None

        return _noop


# ---------------------------------------------------------------------------
# PluginInfo / PluginDiscovery
# ---------------------------------------------------------------------------

@dataclass
class PluginInfo:
    """单个插件的发现信息。"""

    name: str
    path: Path
    manifest_path: Path
    init_path: Path
    # 插件类型: memory_provider / hook_only / tool_provider / standalone
    # 缺失或未知值默认为 standalone, 走原逻辑
    plugin_kind: str = "standalone"


class PluginDiscovery:
    """扫描 ~/.hermes/plugins/ 目录, 发现所有合法插件。

    排除规则:
    - _tests/ (本测试目录自身)
    - __pycache__
    - _disabled_skill_pool.py
    - 以 '.' 开头的隐藏目录
    - 无 plugin.yaml 的目录
    """

    # 排除的目录名集合
    EXCLUDED_NAMES = {"_tests", "__pycache__"}

    # plugin.yaml 中合法的 kind 值
    _VALID_KINDS = {"memory_provider", "hook_only", "tool_provider", "standalone"}

    def __init__(self, plugins_dir: Optional[Path] = None) -> None:
        if plugins_dir is None:
            plugins_dir = Path.home() / ".hermes" / "plugins"
        self.plugins_dir = Path(plugins_dir)

    def _is_excluded(self, name: str) -> bool:
        # 隐藏目录 / 已知排除项
        if name.startswith("."):
            return True
        if name in self.EXCLUDED_NAMES:
            return True
        return False

    def _read_plugin_kind(self, manifest_path: Path) -> str:
        # 从 plugin.yaml 读取 kind 字段, 缺失或未知值返回 standalone
        try:
            import yaml  # type: ignore
            text = manifest_path.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            if isinstance(data, dict):
                kind = data.get("kind")
                if isinstance(kind, str) and kind in self._VALID_KINDS:
                    return kind
        except Exception:
            # yaml 不可用或 manifest 解析失败时, 保持向后兼容
            pass
        return "standalone"

    def discover(self) -> List[PluginInfo]:
        """扫描并返回所有合法插件信息列表(按名称排序)。"""
        results: List[PluginInfo] = []
        if not self.plugins_dir.is_dir():
            return results

        for child in sorted(self.plugins_dir.iterdir()):
            if not child.is_dir():
                continue
            if self._is_excluded(child.name):
                continue
            manifest_path = child / "plugin.yaml"
            if not manifest_path.exists():
                # 兼容 .yml 后缀
                manifest_path = child / "plugin.yml"
            if not manifest_path.exists():
                continue
            init_path = child / "__init__.py"
            if not init_path.exists():
                # 缺少 __init__.py 仍可记录, 但测试时会被标记
                pass
            # 读取 plugin.yaml 的 kind 字段(如果存在), 区分插件类型
            plugin_kind = self._read_plugin_kind(manifest_path)
            results.append(
                PluginInfo(
                    name=child.name,
                    path=child,
                    manifest_path=manifest_path,
                    init_path=init_path,
                    plugin_kind=plugin_kind,
                )
            )
        return results


# ---------------------------------------------------------------------------
# TestResult / ResultCollector
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """单个测试结果。"""

    plugin_name: str
    test_name: str
    status: str  # PASS / FAIL / WARN / TIMEOUT / SKIP
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    exception: Optional[str] = None  # traceback 字符串


class ResultCollector:
    """收集所有测试结果, 提供聚合统计与按插件分组查询。"""

    def __init__(self) -> None:
        self._results: List[TestResult] = []

    def add(self, result: TestResult) -> None:
        self._results.append(result)

    @property
    def results(self) -> List[TestResult]:
        return list(self._results)

    # -- 聚合统计 ----------------------------------------------------------

    def summary(self) -> Dict[str, int]:
        """返回各状态计数字典。"""
        counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "TIMEOUT": 0, "SKIP": 0}
        for r in self._results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def total(self) -> int:
        return len(self._results)

    # -- 分组查询 ----------------------------------------------------------

    def by_plugin(self, plugin_name: str) -> List[TestResult]:
        return [r for r in self._results if r.plugin_name == plugin_name]

    def plugins(self) -> List[str]:
        # 保持发现顺序去重
        seen: List[str] = []
        for r in self._results:
            if r.plugin_name not in seen:
                seen.append(r.plugin_name)
        return seen

    def failures(self) -> List[TestResult]:
        return [r for r in self._results if r.status in ("FAIL", "TIMEOUT")]

    def warnings(self) -> List[TestResult]:
        return [r for r in self._results if r.status == "WARN"]

    def by_test(self, test_name: str) -> List[TestResult]:
        return [r for r in self._results if r.test_name == test_name]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def format_traceback(exc: BaseException) -> str:
    """格式化异常的完整 traceback 字符串。"""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def short_status(result: TestResult) -> str:
    """返回状态对应的 emoji 徽章。"""
    return {
        "PASS": "✅",
        "FAIL": "❌",
        "WARN": "⚠️",
        "TIMEOUT": "⏱️",
        "SKIP": "⏭️",
    }.get(result.status, "❓")
