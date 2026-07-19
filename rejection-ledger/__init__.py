"""Rejection Ledger — 拒绝账本插件 (v1.0, 参考 OpenSquilla)。

安全机制：跟踪被拒绝的工具调用，多次拒绝后自动暂停自主运行。

核心能力：
  - 记录每次工具调用拒绝（用户拒绝、权限拒绝、沙箱拒绝）
  - 滑动窗口统计拒绝率
  - 连续拒绝超过阈值时自动暂停自主运行
  - 拒绝历史可查询，用于审计和调试

配置 (~/.hermes/config.yaml):
  plugins.rejection-ledger:
    enabled: true
    max_consecutive_rejections: 3   # 连续拒绝次数阈值
    window_size: 20                 # 滑动窗口大小
    auto_pause: true                # 超阈值自动暂停
    cooldown_seconds: 300           # 暂停后冷却时间
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("rejection-ledger")

# ─── 配置 ───

DEFAULT_MAX_CONSECUTIVE = 3
DEFAULT_WINDOW_SIZE = 20
DEFAULT_COOLDOWN = 300  # 5 分钟
LEDGER_FILE = Path.home() / ".hermes" / "rejection_ledger.json"


@dataclass
class RejectionEvent:
    """一次拒绝事件。"""
    timestamp: float
    tool_name: str
    reason: str  # "user_denied" | "permission_denied" | "sandbox_denied" | "auto_denied"
    session_id: str = ""
    detail: str = ""


class RejectionLedger:
    """拒绝账本：跟踪、统计、自动暂停。"""

    def __init__(
        self,
        max_consecutive: int = DEFAULT_MAX_CONSECUTIVE,
        window_size: int = DEFAULT_WINDOW_SIZE,
        auto_pause: bool = True,
        cooldown_seconds: int = DEFAULT_COOLDOWN,
    ):
        self.max_consecutive = max_consecutive
        self.window_size = window_size
        self.auto_pause = auto_pause
        self.cooldown_seconds = cooldown_seconds

        self._events: deque[RejectionEvent] = deque(maxlen=500)
        self._consecutive: Dict[str, int] = {}  # session_id → 连续拒绝数
        self._paused: Dict[str, float] = {}  # session_id → 暂停解除时间
        self._unavailable_tools: Dict[str, str] = {}  # tool_name → 不可用原因
        self._lock = threading.Lock()
        self._total_rejections = 0
        self._total_pauses = 0

        self._load()

    def record_rejection(
        self,
        tool_name: str,
        reason: str = "user_denied",
        session_id: str = "",
        detail: str = "",
    ) -> Dict:
        """记录一次拒绝，返回是否触发暂停。"""
        event = RejectionEvent(
            timestamp=time.time(),
            tool_name=tool_name,
            reason=reason,
            session_id=session_id,
            detail=detail,
        )

        with self._lock:
            self._events.append(event)
            self._total_rejections += 1

            # 更新连续拒绝计数
            key = session_id or "_global"
            self._consecutive[key] = self._consecutive.get(key, 0) + 1
            consecutive = self._consecutive[key]

            # 检查是否需要暂停
            should_pause = (
                self.auto_pause
                and consecutive >= self.max_consecutive
                and key not in self._paused
            )

            if should_pause:
                pause_until = time.time() + self.cooldown_seconds
                self._paused[key] = pause_until
                self._total_pauses += 1
                logger.warning(
                    "RejectionLedger: session %s paused for %ds after %d consecutive rejections",
                    key, self.cooldown_seconds, consecutive,
                )

        self._save()

        return {
            "recorded": True,
            "consecutive_rejections": consecutive,
            "paused": should_pause,
            "pause_until": pause_until if should_pause else self._paused.get(key),
            "threshold": self.max_consecutive,
        }

    def record_acceptance(self, session_id: str = "") -> None:
        """记录一次接受，重置连续拒绝计数。"""
        with self._lock:
            key = session_id or "_global"
            self._consecutive[key] = 0

    def is_paused(self, session_id: str = "") -> bool:
        """检查会话是否处于暂停状态。"""
        with self._lock:
            key = session_id or "_global"
            pause_until = self._paused.get(key)
            if pause_until is None:
                return False
            if time.time() >= pause_until:
                # 冷却期结束
                del self._paused[key]
                self._consecutive[key] = 0
                return False
            return True

    def get_pause_remaining(self, session_id: str = "") -> float:
        """返回暂停剩余秒数，0 表示未暂停。"""
        with self._lock:
            key = session_id or "_global"
            pause_until = self._paused.get(key)
            if pause_until is None:
                return 0
            remaining = pause_until - time.time()
            if remaining <= 0:
                del self._paused[key]
                self._consecutive[key] = 0
                return 0
            return remaining

    def mark_unavailable(self, tool_name: str, reason: str) -> None:
        """标记工具为不可用（Call-Storm + 拒绝等联合判定）。"""
        with self._lock:
            self._unavailable_tools[tool_name] = reason

    def is_unavailable(self, tool_name: str) -> bool:
        """检查工具是否被标记为不可用。"""
        with self._lock:
            return tool_name in self._unavailable_tools

    def get_stats(self) -> Dict:
        """返回账本统计信息。"""
        with self._lock:
            events = list(self._events)
            paused_sessions = {
                k: v for k, v in self._paused.items() if v > time.time()
            }

        # 按工具统计拒绝次数
        by_tool: Dict[str, int] = {}
        by_reason: Dict[str, int] = {}
        for e in events:
            by_tool[e.tool_name] = by_tool.get(e.tool_name, 0) + 1
            by_reason[e.reason] = by_reason.get(e.reason, 0) + 1

        return {
            "total_rejections": self._total_rejections,
            "total_pauses": self._total_pauses,
            "current_paused_sessions": len(paused_sessions),
            "rejections_by_tool": dict(sorted(by_tool.items(), key=lambda x: -x[1])[:10]),
            "rejections_by_reason": by_reason,
            "recent_events": [asdict(e) for e in events[-10:]],
            "config": {
                "max_consecutive": self.max_consecutive,
                "window_size": self.window_size,
                "auto_pause": self.auto_pause,
                "cooldown_seconds": self.cooldown_seconds,
            },
        }

    def get_recent(self, limit: int = 20) -> List[Dict]:
        """返回最近的拒绝事件。"""
        with self._lock:
            return [asdict(e) for e in list(self._events)[-limit:]]

    def _load(self):
        """从磁盘加载账本。"""
        if not LEDGER_FILE.exists():
            return
        try:
            with open(LEDGER_FILE) as f:
                data = json.load(f)
            self._total_rejections = data.get("total_rejections", 0)
            self._total_pauses = data.get("total_pauses", 0)
            for e in data.get("recent_events", []):
                self._events.append(RejectionEvent(**e))
        except Exception as e:
            logger.debug("RejectionLedger: failed to load: %s", e)

    def _save(self):
        """保存账本到磁盘。"""
        try:
            LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "total_rejections": self._total_rejections,
                "total_pauses": self._total_pauses,
                "recent_events": [asdict(e) for e in list(self._events)[-50:]],
                "saved_at": time.time(),
            }
            with open(LEDGER_FILE, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug("RejectionLedger: failed to save: %s", e)


# ─── 全局实例 ───

_ledger: Optional[RejectionLedger] = None
_ledger_lock = threading.Lock()


def get_ledger() -> RejectionLedger:
    global _ledger
    if _ledger is None:
        with _ledger_lock:
            if _ledger is None:
                _ledger = RejectionLedger()
    return _ledger


# ─── 事件驱动：budget_warning 事件回调设置的预算降级标记 ───
# 说明：orchestrator 的 event_bus 是 per-session 的，register() 时无 session_id，
# 因此事件订阅可能在 register() 阶段无法生效；此变量作为兜底，由事件回调设置。
_budget_degraded = False


# ─── Hook 处理 ───

def _on_tool_result(**kwargs) -> Optional[Dict]:
    """post_tool_call hook: 检测拒绝并记录。"""
    tool_name = kwargs.get("tool_name", "")
    result = kwargs.get("result", "")
    session_id = kwargs.get("session_id", "")

    if not tool_name or not result:
        return None

    # 检测拒绝模式
    result_lower = result.lower() if isinstance(result, str) else ""
    rejection_signals = [
        "denied", "rejected", "permission denied", "not allowed",
        "refused", "forbidden", "unauthorized", "blocked",
        "拒绝", "不允许", "无权限", "已阻止",
    ]

    is_rejection = any(s in result_lower for s in rejection_signals)
    if not is_rejection:
        # 记录接受，重置连续计数
        ledger = get_ledger()
        ledger.record_acceptance(session_id)
        # 正常情况，发布正常状态
        return {
            "context_merge": {
                "rejection_count": ledger._total_rejections,
                "consecutive_rejections": 0,
                "is_paused": False,
            },
        }

    # 记录拒绝
    reason = "user_denied"
    if "permission" in result_lower or "无权限" in result_lower:
        reason = "permission_denied"
    elif "sandbox" in result_lower:
        reason = "sandbox_denied"

    ledger = get_ledger()
    info = ledger.record_rejection(
        tool_name=tool_name,
        reason=reason,
        session_id=session_id,
        detail=result[:200],
    )

    consecutive = info["consecutive_rejections"]
    paused = info["paused"]

    # 预算降级时阈值降低：连续拒绝 2 次即暂停（正常是 3 次）
    pause_threshold = 2 if _budget_degraded else 3
    if consecutive >= pause_threshold:
        info["paused"] = True
        paused = True
        if _budget_degraded:
            logger.info(
                "预算降级模式下，连续拒绝 %d 次即触发暂停（阈值=%d）",
                consecutive, pause_threshold,
            )

    # ── P2 链路协作：读取 cache-optimizer 的 Call-Storm 信号 ──
    # 联合判断：Call-Storm + 拒绝 = 工具不可用
    plugin_context = kwargs.get("plugin_context")
    call_storm_tool = ""
    if plugin_context:
        call_storm_tool = plugin_context.shared_get("call_storm_tool", "")

    tool_unavailable = ""
    unavailable_reason = ""
    if call_storm_tool and call_storm_tool == tool_name:
        logger.warning(
            "工具 %s 同时出现 Call-Storm 和拒绝，标记为不可用",
            tool_name,
        )
        ledger.mark_unavailable(tool_name, "Call-Storm + 拒绝")
        tool_unavailable = tool_name
        unavailable_reason = "Call-Storm + 拒绝"

    if paused:
        return {
            "context": (
                f"⚠️ 安全暂停: 工具 {tool_name} 已被连续拒绝 "
                f"{consecutive} 次，"
                f"自主运行已暂停 {ledger.cooldown_seconds} 秒。"
                f"如需继续，请手动确认。"
            ),
            "context_merge": {
                "rejection_count": ledger._total_rejections,
                "consecutive_rejections": consecutive,
                "is_paused": True,
                "last_rejected_tool": tool_name,
                "tool_unavailable": tool_unavailable,
                "unavailable_reason": unavailable_reason,
            },
        }

    # 拒绝但未触发暂停，发布拒绝状态
    return {
        "context_merge": {
            "rejection_count": ledger._total_rejections,
            "consecutive_rejections": consecutive,
            "is_paused": False,
            "last_rejected_tool": tool_name,
            "tool_unavailable": tool_unavailable,
            "unavailable_reason": unavailable_reason,
        },
    }


# ─── 事前拦截：pre_tool_call 钩子 ───

def _get_tool_info(tool_name: str, session_id: str = "") -> Optional[Dict]:
    """查询工具的连续拒绝信息。

    由于 ledger 按 session_id 维度存储连续拒绝数（非 tool_name 维度），
    这里通过扫描最近事件统计该工具的连续拒绝次数：
    从最新事件向前扫描，遇到相同工具的拒绝事件则计数 +1，
    遇到不同工具则停止（即只统计"最近连续相同工具拒绝"）。
    """
    ledger = get_ledger()
    with ledger._lock:
        consecutive = 0
        for event in reversed(ledger._events):
            if event.tool_name == tool_name:
                consecutive += 1
            else:
                break
    paused = ledger.is_paused(session_id)
    return {
        "consecutive_rejections": consecutive,
        "paused": paused,
    }


def _pre_tool_call(**kwargs) -> Optional[Dict]:
    """工具调用前拦截：检查工具是否连续被拒绝，如果是则跳过调用。"""
    tool_name = kwargs.get("tool_name") or kwargs.get("name") or ""
    if not tool_name:
        return None

    session_id = kwargs.get("session_id", "")

    # 查询该工具的连续拒绝次数
    info = _get_tool_info(tool_name, session_id)
    if not info:
        return None

    consecutive = info.get("consecutive_rejections", 0)
    # 预算降级时阈值降低：连续拒绝 2 次即拦截（正常是 3 次）
    pause_threshold = 2 if _budget_degraded else 3

    if consecutive >= pause_threshold:
        logger.warning(
            "工具 %s 连续拒绝 %d 次（阈值 %d），拦截调用",
            tool_name, consecutive, pause_threshold,
        )
        return {
            "action": "skip",
            "reason": f"工具 {tool_name} 连续拒绝 {consecutive} 次，建议更换方案",
        }

    return None


# ─── 工具函数 ───

def handle_rejection_ledger_stats(args: Dict, **kwargs) -> str:
    """查询拒绝账本统计。"""
    ledger = get_ledger()
    stats = ledger.get_stats()
    return json.dumps(stats, indent=2, ensure_ascii=False)


def handle_rejection_ledger_recent(args: Dict, **kwargs) -> str:
    """查询最近的拒绝事件。"""
    ledger = get_ledger()
    limit = args.get("limit", 20)
    events = ledger.get_recent(limit)
    return json.dumps(events, indent=2, ensure_ascii=False)


def handle_rejection_ledger_check(args: Dict, **kwargs) -> str:
    """检查会话是否被暂停。"""
    ledger = get_ledger()
    session_id = args.get("session_id", "")
    paused = ledger.is_paused(session_id)
    remaining = ledger.get_pause_remaining(session_id)
    return json.dumps({
        "session_id": session_id,
        "paused": paused,
        "remaining_seconds": round(remaining, 1),
    })


# ─── 插件注册 ───

def register(ctx) -> None:
    "注册拒绝账本插件。"
    if ctx is None:
        logger.warning("RejectionLedger: register called with None context, skipping")
        return

    # 加载配置（带 ctx.config 防御）
    try:
        if not hasattr(ctx, 'config') or ctx.config is None:
            config = {}
        else:
            config = ctx.config.get("plugins", {}).get("rejection-ledger", {})
    except Exception:
        config = {}

    if config.get("enabled") is False:
        logger.info("RejectionLedger: disabled by config")
        return

    # 初始化全局 ledger
    global _ledger
    _ledger = RejectionLedger(
        max_consecutive=config.get("max_consecutive_rejections", DEFAULT_MAX_CONSECUTIVE),
        window_size=config.get("window_size", DEFAULT_WINDOW_SIZE),
        auto_pause=config.get("auto_pause", True),
        cooldown_seconds=config.get("cooldown_seconds", DEFAULT_COOLDOWN),
    )

    # 注册 hook
    ctx.register_hook("post_tool_call", _on_tool_result)
    ctx.register_hook("pre_tool_call", _pre_tool_call)

    # 注册工具
    ctx.register_tool(
        name="rejection_ledger_stats",
        toolset="rejection-ledger",
        schema={
            "type": "object",
            "properties": {},
        },
        handler=handle_rejection_ledger_stats,
        description="查询拒绝账本统计信息（总拒绝次数、暂停次数、按工具/原因分布）",
    )
    ctx.register_tool(
        name="rejection_ledger_recent",
        toolset="rejection-ledger",
        schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "返回数量", "default": 20},
            },
        },
        handler=handle_rejection_ledger_recent,
        description="查询最近的拒绝事件列表",
    )
    ctx.register_tool(
        name="rejection_ledger_check",
        toolset="rejection-ledger",
        schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "会话 ID"},
            },
        },
        handler=handle_rejection_ledger_check,
        description="检查指定会话是否因连续拒绝被暂停",
    )

    logger.info(
        "RejectionLedger v1.0 registered: post_tool_call + pre_tool_call hooks + 3 tools | "
        "max_consecutive=%d, auto_pause=%s, cooldown=%ds",
        _ledger.max_consecutive, _ledger.auto_pause, _ledger.cooldown_seconds,
    )

    # ── 订阅 budget_warning 事件，预算紧张时更激进暂停 ──
    # 注意：orchestrator 的 PluginContext.event_bus 是 per-session 的，
    # register() 阶段无 session_id，订阅可能无法立即生效。
    # 此处尝试获取已加载的 PluginContext 并订阅；失败时静默降级。
    try:
        import sys
        orch_ctx = None
        if "plugin_orchestrator.context" in sys.modules:
            from plugin_orchestrator.context import get_context
            # get_context 需要 session_id 参数；register() 阶段无 session，
            # 尝试无参数调用会抛 TypeError，由外层 except 捕获降级。
            orch_ctx = get_context()  # type: ignore[call-arg]

        if orch_ctx and hasattr(orch_ctx, 'event_bus'):
            def _on_budget_warning(event):
                """budget_warning 事件回调：预算紧张时降低拒绝阈值。"""
                global _budget_degraded
                _budget_degraded = True
                logger.info("收到 budget_warning 事件，拒绝阈值已降低")
                # 预算紧张时，连续拒绝 2 次即暂停（正常是 3 次）

            orch_ctx.event_bus.subscribe("budget_warning", _on_budget_warning)
            logger.info("rejection-ledger 已订阅 budget_warning 事件")
    except Exception as e:
        logger.debug("订阅 budget_warning 事件失败: %s", e)
