"""
PluginContext — 跨插件共享上下文

每个会话一个 PluginContext 实例，生命周期内可在所有插件的钩子回调中共享读写。

设计原则：
  - shared_state: 所有插件可见的公共状态
  - private_state: 按插件名隔离的私有命名空间
  - session_metadata: 会话级元数据（session_id, provider, model, platform）
  - 所有读写操作线程安全
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventBus:
    """插件间发布/订阅事件总线。

    任一插件可以 publish() 事件，其他插件通过 subscribe() 监听。
    支持通配符 '*' 匹配所有事件。
    线程安全。
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._event_history: List[Dict] = []  # 最近 100 条事件

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """订阅事件。event_type='*' 匹配所有事件。"""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """取消订阅。"""
        with self._lock:
            subs = self._subscribers.get(event_type, [])
            if callback in subs:
                subs.remove(callback)

    def publish(self, event_type: str, source_plugin: str, **data) -> None:
        """发布事件。通知所有匹配的订阅者。"""
        event = {
            "type": event_type,
            "source": source_plugin,
            "timestamp": time.time(),
            "data": data,
        }

        # 记录历史（最多 100 条）
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > 100:
                self._event_history = self._event_history[-100:]

        # 通知订阅者
        callbacks = []
        with self._lock:
            # 精确匹配
            if event_type in self._subscribers:
                callbacks.extend(self._subscribers[event_type])
            # 通配符匹配
            if "*" in self._subscribers:
                callbacks.extend(self._subscribers["*"])

        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass  # 一个订阅者崩溃不影响其他

    def history(self, event_type: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """获取最近的事件历史。"""
        with self._lock:
            if event_type is None:
                return list(self._event_history[-limit:])
            return [e for e in self._event_history[-limit * 2:] if e["type"] == event_type][-limit:]


class PluginContext:
    """跨插件共享上下文。

    每个 session_id 一个实例，在会话首轮创建并缓存，后续轮次复用。

    用法示例：
      # 插件 A
      ctx.shared.set("model_quality", 4)
      ctx.event_bus.publish("model_switched", source="model-router")

      # 插件 B
      quality = ctx.shared.get("model_quality")
      ctx.event_bus.subscribe("model_switched", on_model_switched)
    """

    def __init__(self, session_id: str = "", session_source: str = ""):
        self.shared: Dict[str, Any] = {}  # 公共状态，所有插件可见
        self.private: Dict[str, Dict[str, Any]] = {}  # 私有状态，按插件名隔离
        self.metadata: Dict[str, Any] = {
            "session_id": session_id,
            "source": session_source,
            "created_at": time.time(),
        }
        self.event_bus = EventBus()
        self.turn_number: int = 0
        self.trace_id: str = str(uuid.uuid4())  # 请求级追踪 ID
        self._lock = threading.Lock()

    # ── Shared state --------------------------------------------------

    def shared_get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.shared.get(key, default)

    def shared_set(self, key: str, value: Any) -> None:
        with self._lock:
            self.shared[key] = value

    def shared_pop(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.shared.pop(key, default)

    # ── Private state (per-plugin) ------------------------------------

    def plugin_get(self, plugin_name: str, key: str, default: Any = None) -> Any:
        with self._lock:
            ns = self.private.setdefault(plugin_name, {})
            return ns.get(key, default)

    def plugin_set(self, plugin_name: str, key: str, value: Any) -> None:
        with self._lock:
            ns = self.private.setdefault(plugin_name, {})
            ns[key] = value

    # ── Metadata ------------------------------------------------------

    def update_metadata(self, **kwargs) -> None:
        """更新会话元数据（model, provider, platform 等）。"""
        with self._lock:
            self.metadata.update(kwargs)

    # ── Turn management -----------------------------------------------

    def new_turn(self) -> int:
        """开始新的一轮。返回当前轮次号。"""
        with self._lock:
            self.turn_number += 1
            return self.turn_number

    # ── Trace ID management -------------------------------------------

    def new_trace_id(self) -> str:
        """生成新的 trace_id（每次 pre_llm_call 时调用）。返回新的 trace_id。"""
        with self._lock:
            self.trace_id = str(uuid.uuid4())
            return self.trace_id

    # ── Snapshot (for debugging) -------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.metadata.get("session_id", ""),
                "turn": self.turn_number,
                "shared_keys": list(self.shared.keys()),
                "private_plugins": list(self.private.keys()),
                "event_history_len": len(self.event_bus._event_history),
            }


# ── 全局上下文注册表 ──────────────────────────────────────────────────

_context_registry: Dict[str, PluginContext] = {}
_registry_lock = threading.Lock()


def get_context(session_id: str) -> Optional[PluginContext]:
    """获取指定会话的 PluginContext。"""
    with _registry_lock:
        return _context_registry.get(session_id)


def get_or_create_context(session_id: str, source: str = "") -> PluginContext:
    """获取或创建会话的 PluginContext。"""
    with _registry_lock:
        if session_id not in _context_registry:
            _context_registry[session_id] = PluginContext(
                session_id=session_id,
                session_source=source,
            )
            logger.debug("PluginContext created for session=%s", session_id[:8])
        return _context_registry[session_id]


def remove_context(session_id: str) -> None:
    """移除会话上下文（会话结束时调用）。"""
    with _registry_lock:
        _context_registry.pop(session_id, None)
        logger.debug("PluginContext removed for session=%s", session_id[:8])


def list_active_contexts() -> List[str]:
    """列出所有活跃的会话 ID。"""
    with _registry_lock:
        return list(_context_registry.keys())
