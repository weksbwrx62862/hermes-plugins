"""
plugin-orchestrator v1.0.0 — 让插件从"各自为政"变为"有机协作"

核心能力：
  1. PluginContext — 跨插件共享上下文（shared_state + private_state + EventBus）
  2. 钩子优先级 — 通过 priority 控制执行顺序
  3. 跨插件管道 — 基于数据依赖的拓扑排序
  4. 会话生命周期管理 — 自动创建/销毁 PluginContext

完全向后兼容：
  - 不实现新接口的插件照常工作
  - 不感知 PluginContext 的回调忽略额外的 `plugin_context` kwarg
  - 不声明 pipeline 的插件保有原始注册顺序

工作原理：
  1. 启动时 monkey-patch PluginManager.invoke_hook()
  2. 拦截每个钩子调用，注入 plugin_context 参数
  3. 按优先级 + 管道依赖排序后执行
  4. on_session_start → 创建 PluginContext
  5. on_session_end → 销毁 PluginContext
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── 请求级 trace_id 日志注入 ─────────────────────────────────────────


class TraceIdFilter(logging.Filter):
    """将 PluginContext 的 trace_id 注入每条日志记录。

    通过 set_trace_id() 更新当前线程上下文的 trace_id，
    filter() 把它附加到每条 LogRecord 上（属性名 trace_id）。
    即使 trace_id 为空也允许日志输出（返回 True）。
    """

    def __init__(self):
        super().__init__()
        self._current_trace_id: str = ""

    def set_trace_id(self, trace_id: str) -> None:
        """更新当前要注入到日志记录中的 trace_id。"""
        self._current_trace_id = trace_id or ""

    def filter(self, record: logging.LogRecord) -> bool:
        # 始终设置 trace_id 字段（即使为空），并放行所有日志
        record.trace_id = self._current_trace_id
        return True


# 全局实例：供 orchestrator 在每次 pre_llm_call 时同步 trace_id
_trace_filter = TraceIdFilter()

# ── 延迟导入（避免循环依赖和启动顺序问题）─────────────────────────

_context = None
_pipeline = None
_GET_OR_CREATE = None
_REMOVE_CONTEXT = None
_GET_PIPELINE = None


def _lazy_imports():
    """延迟导入 context 和 pipeline 模块。

    使用 importlib 加载同级目录中的 .py 文件，
    兼容 Hermes 插件加载机制（不使用相对导入）。
    模块会被注册到 sys.modules 中，确保全局单例。
    """
    global _context, _pipeline, _GET_OR_CREATE, _REMOVE_CONTEXT, _GET_PIPELINE
    if _GET_OR_CREATE is not None:
        return

    import importlib.util, os, sys

    _plugin_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载 context.py (注册到 sys.modules 确保全局单例)
    _CTX_MOD_NAME = "plugin_orchestrator.context"
    if _CTX_MOD_NAME not in sys.modules:
        _ctx_spec = importlib.util.spec_from_file_location(
            _CTX_MOD_NAME,
            os.path.join(_plugin_dir, "context.py"),
        )
        _ctx_mod = importlib.util.module_from_spec(_ctx_spec)
        sys.modules[_CTX_MOD_NAME] = _ctx_mod
        _ctx_spec.loader.exec_module(_ctx_mod)
    else:
        _ctx_mod = sys.modules[_CTX_MOD_NAME]

    # 加载 pipeline.py
    _PIPE_MOD_NAME = "plugin_orchestrator.pipeline"
    if _PIPE_MOD_NAME not in sys.modules:
        _pipe_spec = importlib.util.spec_from_file_location(
            _PIPE_MOD_NAME,
            os.path.join(_plugin_dir, "pipeline.py"),
        )
        _pipe_mod = importlib.util.module_from_spec(_pipe_spec)
        sys.modules[_PIPE_MOD_NAME] = _pipe_mod
        _pipe_spec.loader.exec_module(_pipe_mod)
    else:
        _pipe_mod = sys.modules[_PIPE_MOD_NAME]

    _GET_OR_CREATE = _ctx_mod.get_or_create_context
    _REMOVE_CONTEXT = _ctx_mod.remove_context
    _GET_PIPELINE = _pipe_mod.get_pipeline_graph
    _context = _ctx_mod.PluginContext
    _pipeline = _pipe_mod.PipelineGraph


# ── Monkey-patch 状态 ───────────────────────────────────────────────

_patch_applied = False
_patch_lock = threading.Lock()
_original_invoke_hook = None
_plugin_name_map: Dict[str, Any] = {}  # callback_id → plugin_name


def _get_plugin_name_for_cb(cb: Callable) -> str:
    """推测回调函数属于哪个插件。"""
    cb_id = id(cb)
    if cb_id in _plugin_name_map:
        return _plugin_name_map[cb_id]

    # 通过 __module__ 推测插件名
    module = getattr(cb, "__module__", "") or ""
    
    # 策略 1: 如果模块名包含 "plugins."，提取插件名
    #   plugins.disk_cleanup.__init__ → disk_cleanup
    #   plugins.model_router.__init__ → model_router
    if "plugins." in module:
        parts = module.split("plugins.")[-1].split(".")
        if parts:
            name = parts[0]
            name = name.replace("-", "_")
            _plugin_name_map[cb_id] = name
            return name

    # 策略 2: 硬编码映射（快速查找）
    for part in module.split("."):
        if part.startswith("plugin") or part in (
            "model_router", "omnimem", "deepseek_cache_optimizer",
            "adaptive_multi_agent", "disk_cleanup", "codegraph",
            "dev_lifecycle", "skill_router", "prompt_optimizer",
            "gateway_restart", "log_translator", "taste_skill",
            "self_evolution", "understand_anything",
        ):
            _plugin_name_map[cb_id] = part
            return part

    # 策略 3: 取模块名的最后一段
    name = module.rsplit(".", 1)[-1] if module else "unknown"
    _plugin_name_map[cb_id] = name
    return name


# ── 优先级管理 ──────────────────────────────────────────────────────

_hook_priorities: Dict[str, Dict[str, int]] = {}  # hook_name → {plugin_name: priority}


def set_hook_priority(plugin_name: str, hook_name: str, priority: int) -> None:
    """设置插件在指定钩子上的执行优先级。越小越先执行。
    
    默认优先级 = 500。
    推荐范围:
      -900 ~ -100: 编排器/基础设施层（plugin-orchestrator, model-router）
      -99 ~ 99: 核心功能层（omnimem, ama）
      100 ~ 499: 业务插件层（skill-router, prompt-optimizer）
      500: 默认
      501 ~ 900: 后处理层（disk-cleanup, log-translator）
    """
    _hook_priorities.setdefault(hook_name, {})[plugin_name] = priority
    logger.debug("Hook priority: %s.%s = %d", plugin_name, hook_name, priority)


def get_hook_priority(plugin_name: str, hook_name: str) -> int:
    """获取插件在指定钩子上的优先级。"""
    return _hook_priorities.get(hook_name, {}).get(plugin_name, 500)


# ── 核心：带上下文的 invoke_hook ────────────────────────────────────

def _invoke_hook_with_context(
    self,  # PluginManager instance
    hook_name: str,
    **kwargs: Any,
) -> List[Any]:
    """增强版 invoke_hook：注入 PluginContext，按优先级+管道排序执行。"""

    _lazy_imports()

    # 如果 patch 还未安装，在此延迟重试（register() 可能早于模块就绪）
    if not _monkey_patch_installed:
        _install_monkey_patch()
        # 如果仍然未安装，回退到原始逻辑
        if not _monkey_patch_installed and _original_invoke_hook:
            return _original_invoke_hook(self, hook_name, **kwargs)

    callbacks = list(self._hooks.get(hook_name, []))

    if not callbacks:
        return []

    # ── 步骤 1: 解析 session_id ────────────────────────────────────
    session_id = kwargs.get("session_id", "") or ""
    agent = kwargs.get("agent")
    if not session_id and agent:
        session_id = getattr(agent, "session_id", "") or ""
    if not session_id:
        session_id = kwargs.get("session_key", "") or ""

    # ── 步骤 2: 获取/创建 PluginContext ─────────────────────────────
    plugin_context = None
    if session_id:
        plugin_context = _GET_OR_CREATE(session_id)

        # 更新元数据
        if agent:
            plugin_context.update_metadata(
                model=getattr(agent, "model", ""),
                provider=getattr(agent, "provider", ""),
                platform=getattr(agent, "platform", ""),
            )

        # 轮次管理
        if hook_name in ("pre_llm_call",):
            plugin_context.new_turn()

    # ── 步骤 3: 构建带插件名的回调列表 ────────────────────────────
    cb_entries: List[Tuple[str, Callable]] = []
    for cb in callbacks:
        name = _get_plugin_name_for_cb(cb)
        _plugin_name_map[id(cb)] = name
        cb_entries.append((name, cb))

    # ── 步骤 4: 按优先级排序 ───────────────────────────────────────
    def _sort_key(entry: Tuple[str, Callable]) -> int:
        name = entry[0]
        return get_hook_priority(name, hook_name)

    cb_entries.sort(key=_sort_key)

    # ── 步骤 5: 管道拓扑排序（覆盖优先级排序）─────────────────────
    try:
        pipeline_graph = _GET_PIPELINE()
        sorted_entries = pipeline_graph.topological_sort(cb_entries)
        # topological_sort returns (priority, name, cb)
        cb_entries = [(name, cb) for _, name, cb in sorted_entries]
    except Exception as exc:
        logger.debug("Pipeline sort skipped: %s", exc)

    # ── 步骤 6: 生成请求级 trace_id（仅 pre_llm_call）────────────
    # 每次 LLM 请求开始时刷新 trace_id，供本次请求所有日志关联
    if hook_name == "pre_llm_call" and plugin_context:
        plugin_context.new_trace_id()
        _trace_filter.set_trace_id(plugin_context.trace_id)

    # ── 步骤 7: 注入 plugin_context 并执行 ────────────────────────
    # transform 钩子的特殊语义：第一个非 None 返回值 = 替换内容
    _TRANSFORM_HOOKS = frozenset({
        "transform_tool_result", "transform_terminal_output",
        "transform_llm_output", "transform_request",
    })
    _SINGLE_RETURN_HOOKS = frozenset({
        "pre_gateway_dispatch",
        "pre_approval_request",
    }) | _TRANSFORM_HOOKS

    # ── 延迟导入熔断器和追踪（捕获异常避免启动失败） ────────────
    _cb_registry = None
    _trace_store = None
    _Span_cls = None
    try:
        from plugins.plugin_orchestrator.circuit_breaker import get_registry as _gbr
        from plugins.plugin_orchestrator.tracer import get_trace_store, Span
        _cb_registry = _gbr()
        _trace_store = get_trace_store()
        _Span_cls = Span
    except Exception:
        pass

    results: List[Any] = []
    for plugin_name, cb in cb_entries:
        # ── 熔断器检查（Hystrix 三态）──────────────────────────────
        if _cb_registry:
            breaker = _cb_registry.get_or_create(plugin_name, hook_name)
            if breaker.is_open():
                logger.debug(
                    "Breaker OPEN: skipping %s.%s",
                    plugin_name, hook_name,
                )
                continue
        else:
            breaker = None

        # ── 追踪 Span ──────────────────────────────────────────────
        span = None
        if _trace_store and _Span_cls:
            turn = plugin_context.turn_number if plugin_context else 0
            span = _Span_cls(hook_name, plugin_name, session_id, turn)

        try:
            # 注入 plugin_context 到 kwargs（如果回调接受的话）
            injected_kwargs = {**kwargs}
            injected_kwargs["plugin_context"] = plugin_context
            injected_kwargs["_plugin_name"] = plugin_name

            ret = cb(**injected_kwargs)

            # ── 成功记录 ──
            if breaker:
                breaker.on_success()
            if span:
                result_size = len(str(ret)) if ret is not None else 0
                span.end(success=True, result_size=result_size)
                _trace_store.record(span)

            if ret is None:
                continue

            # ── transform 钩子：首次非 None 返回即传播，后续忽略 ──
            if hook_name in _TRANSFORM_HOOKS:
                results.append(ret)
                logger.debug(
                    "%s: plugin '%s' returned replacement (first-wins), "
                    "skipping remaining %d callbacks",
                    hook_name, plugin_name, len(cb_entries) - cb_entries.index((plugin_name, cb)) - 1,
                )
                break  # 第一个返回值即生效

            # ── pre_gateway_dispatch / pre_approval：首个结果即决定 ──
            if hook_name in _SINGLE_RETURN_HOOKS:
                results.append(ret)
                break

            # ── 其他钩子：收集全部返回值 ──
            results.append(ret)

            # 如果返回 dict 且包含 context_merge，自动合并到共享状态
            if isinstance(ret, dict) and "context_merge" in ret:
                if plugin_context:
                    merge_data = ret["context_merge"]
                    if isinstance(merge_data, dict):
                        for k, v in merge_data.items():
                            plugin_context.shared_set(k, v)

            # 如果返回 dict 且包含 event，自动发布事件
            if isinstance(ret, dict) and "event" in ret:
                if plugin_context:
                    event_data = ret["event"]
                    if isinstance(event_data, dict):
                        event_type = event_data.get("type", "unknown")
                        event_payload = event_data.get("data", {})
                        plugin_context.event_bus.publish(
                            event_type,
                            source_plugin=plugin_name,
                            **event_payload,
                        )

        except Exception as exc:
            # 记录详细的异常信息，包含插件名和 hook 名
            # 提示：context_merge/event 输出会因异常而丢失，影响下游状态合并与事件发布
            logger.warning(
                "钩子异常: plugin=%s hook=%s error=%s — "
                "该插件的 context_merge/event 输出将丢失",
                plugin_name, hook_name, exc,
                exc_info=True,  # 包含堆栈
            )
            # ── 失败记录（触发熔断器）─
            if breaker:
                breaker.on_failure()
            if span:
                span.end(success=False, error=str(exc)[:200])
                _trace_store.record(span)

    # ── 步骤 8: 会话生命周期钩子 ──────────────────────────────────
    if hook_name == "on_session_end" and session_id:
        _REMOVE_CONTEXT(session_id)

    return results


# ── 注册入口 ────────────────────────────────────────────────────────

_monkey_patch_installed = False
_monkey_patch_lock = threading.Lock()


def _install_monkey_patch() -> None:
    """安全地 monkey-patch PluginManager.invoke_hook()。

    延迟到第一次实际调用时安装，避免启动顺序问题：
    - 插件 register() 可能早于 hermes_cli.plugins 被导入
    - 第一次 invoke_hook 调用时所有模块都已就绪
    """
    global _monkey_patch_installed, _original_invoke_hook

    if _monkey_patch_installed:
        return
    with _monkey_patch_lock:
        if _monkey_patch_installed:
            return

        try:
            import hermes_cli.plugins as _plugins_module
            pm_class = _plugins_module.PluginManager

            if hasattr(pm_class, "invoke_hook"):

                # 保存原始方法，用于回退
                _original_invoke_hook = pm_class.invoke_hook

                # 检查是否已经被另一个 orchestrator patch 了
                if getattr(pm_class.invoke_hook, "_orchestrator_patched", False):
                    logger.info("plugin-orchestrator: invoke_hook already patched (skip)")
                    _monkey_patch_installed = True
                    return

                # 包装方法
                def _patched_invoke_hook(self, hook_name, **kwargs):
                    # 注入 plugin_context 并排序执行
                    return _invoke_hook_with_context(self, hook_name, **kwargs)

                # 标记为 orchestrator patch
                _patched_invoke_hook._orchestrator_patched = True

                pm_class.invoke_hook = _patched_invoke_hook
                _monkey_patch_installed = True
                logger.info("plugin-orchestrator: monkey-patched PluginManager.invoke_hook()")
            else:
                logger.error("plugin-orchestrator: PluginManager.invoke_hook not found")
        except Exception as exc:
            # 如果 hermes_cli.plugins 还没加载，记录并延迟到首次调用
            if "'hermes_cli.plugins'" in str(exc) or "No module named" in str(exc):
                logger.debug("plugin-orchestrator: PluginManager not yet loaded, deferring patch")
                _monkey_patch_installed = False  # 允许重试
            else:
                logger.error("plugin-orchestrator: failed to install monkey-patch: %s", exc)


def _uninstall_monkey_patch() -> None:
    """恢复原始 invoke_hook。"""
    global _monkey_patch_installed, _original_invoke_hook
    if _original_invoke_hook:
        try:
            import hermes_cli.plugins as _plugins_module
            _plugins_module.PluginManager.invoke_hook = _original_invoke_hook
            _original_invoke_hook = None
            _monkey_patch_installed = False
            logger.info("plugin-orchestrator: restored original invoke_hook")
        except Exception:
            pass


# ── 插件注册 ────────────────────────────────────────────────────────


def register(ctx=None) -> None:
    """注册插件编排器。

    1. 安装 monkey-patch
    2. 为内置伙伴插件注册 pipeline 声明（如果它们还没注册的话）
    3. 设置默认优先级
    4. 注册 TraceIdFilter 到 root logger（注入请求级 trace_id）
    """

    _lazy_imports()

    # ── 安装 monkey-patch ──────────────────────────────────────────
    _install_monkey_patch()

    # ── 注册 TraceIdFilter 到 root logger ──────────────────────────
    # 不修改现有日志格式，仅注入 trace_id 属性，供需要时使用 %(trace_id)s
    root_logger = logging.getLogger()
    if _trace_filter not in root_logger.filters:
        root_logger.addFilter(_trace_filter)
        logger.debug("TraceIdFilter added to root logger")

    # ── 为已有插件注册 pipeline 声明 ───────────────────────────────
    # 这些声明可以在各插件的 plugin.yaml 中覆盖，但我们提供合理默认值
    pipeline = _GET_PIPELINE()

    default_pipelines = {
        "model_router": {
            "produces": ["model_selection", "model_quality", "routing_strategy", "budget_status", "provider_info", "session_metadata"],
            "needs": ["task_complexity", "ama_task_weight"],
        },
        "adaptive_multi_agent": {
            "produces": ["task_complexity", "ama_task_weight", "execution_mode", "ama_trajectory"],
            "needs": [],
        },
        "omnimem": {
            "produces": ["memory_context", "prefetch_result", "kg_triples"],
            "needs": ["session_metadata"],
        },
        "deepseek_cache_optimizer": {
            "produces": ["cache_diagnostics", "tool_result_compressed"],
            "needs": ["model_selection", "provider_info"],
        },
        "skill_router": {
            "produces": ["skill_injection"],
            "needs": ["task_complexity", "model_selection"],
        },
        "disk_cleanup": {
            "produces": ["files_created"],
            "needs": [],
        },
        "codegraph": {
            "produces": ["code_index"],
            "needs": [],
        },
    }

    for plugin_name, pl in default_pipelines.items():
        if pipeline.get_manifest(plugin_name) is None:
            pipeline.register(
                plugin_name,
                produces=pl.get("produces", []),
                needs=pl.get("needs", []),
            )

    # ── 设置默认优先级 ─────────────────────────────────────────────
    default_priorities = {
        # 基础设施层 (-900 ~ -100)
        "plugin_orchestrator": -900,
        "model_router": -800,
        "gateway_restart": -700,
        # 核心功能层 (-99 ~ 99)
        "omnimem": -50,
        "adaptive_multi_agent": -40,
        "deepseek_cache_optimizer": -30,
        "codegraph": -20,
        # 业务插件层 (100 ~ 499)
        "skill_router": 100,
        "prompt_optimizer": 150,
        "dev_lifecycle": 200,
        "taste_skill": 250,
        "understand_anything": 260,
        "self_evolution": 270,
        "repo_chinese_names": 280,
        # 后处理层 (501 ~ 900)
        "disk_cleanup": 800,
        "log_translator": 850,
    }

    for hook_name in (
        "pre_llm_call",
        "post_llm_call",
        "pre_tool_call",
        "post_tool_call",
        "transform_tool_result",
        "transform_llm_output",
        "transform_request",
        "pre_api_request",
        "post_api_request",
        "on_session_start",
        "on_session_end",
        "on_session_finalize",
        "on_session_reset",
        "pre_gateway_dispatch",
        "pre_approval_request",
        "post_approval_response",
    ):
        for plugin_name, priority in default_priorities.items():
            if get_hook_priority(plugin_name, hook_name) == 500:  # 还是默认
                set_hook_priority(plugin_name, hook_name, priority)

    logger.info(
        "plugin-orchestrator v1.1.0 registered — %d plugin pipelines, %d priority entries",
        len(default_pipelines),
        len(default_priorities) * 16,  # 16 hooks
    )


def unregister() -> None:
    """注销插件编排器（清理 monkey-patch）。"""
    _uninstall_monkey_patch()
