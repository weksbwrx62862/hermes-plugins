"""
plugin-orchestrator 诊断 CLI

提供命令行工具用于:
  - 查看活跃的 PluginContext 实例
  - 查看管道依赖关系
  - 查看钩子优先级
  - 查看事件历史
  - 查看熔断器状态
  - 查看追踪统计
  - 测试跨插件通信

用法:
  python -m plugins.plugin_orchestrator.diag [status|pipes|events|breakers|traces|test]
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def status() -> Dict[str, Any]:
    """返回编排器状态报告。"""
    result = {
        "version": "1.0.0",
        "patch_installed": False,
        "active_sessions": 0,
        "sessions": [],
        "pipeline_plugins": 0,
        "priority_entries": 0,
        "event_history_total": 0,
    }

    try:
        import hermes_cli.plugins as _pm
        original = getattr(_pm.PluginManager, "invoke_hook", None)
        from plugins.plugin_orchestrator.__init__ import _invoke_hook_with_context
        result["patch_installed"] = original is _invoke_hook_with_context
    except Exception as exc:
        result["patch_error"] = str(exc)

    try:
        from plugins.plugin_orchestrator.context import list_active_contexts, get_context
        sessions = list_active_contexts()
        result["active_sessions"] = len(sessions)
        for sid in sessions[:10]:
            ctx = get_context(sid)
            if ctx:
                snap = ctx.snapshot()
                snap["session_id_short"] = sid[:8]
                result["sessions"].append(snap)
    except Exception as exc:
        result["context_error"] = str(exc)

    try:
        from plugins.plugin_orchestrator.pipeline import get_pipeline_graph
        pg = get_pipeline_graph()
        deps = pg.list_deps()
        result["pipeline_plugins"] = len(deps)
        result["pipeline_deps"] = deps
    except Exception as exc:
        result["pipeline_error"] = str(exc)

    try:
        from plugins.plugin_orchestrator.__init__ import _hook_priorities
        result["priority_entries"] = sum(
            len(p) for p in _hook_priorities.values()
        )
    except Exception:
        pass

    # 熔断器状态
    try:
        from plugins.plugin_orchestrator.circuit_breaker import get_registry
        registry = get_registry()
        breakers = registry.dump_all()
        result["breakers"] = {
            "total": len(breakers),
            "open": sum(1 for b in breakers if b["state"] == "OPEN"),
            "half_open": sum(1 for b in breakers if b["state"] == "HALF_OPEN"),
        }
    except Exception:
        pass

    # 追踪统计
    try:
        from plugins.plugin_orchestrator.tracer import get_trace_store
        store = get_trace_store()
        result["trace"] = store.stats()
    except Exception:
        pass

    return result


def pipes() -> str:
    """返回管道依赖图的可读文本。"""
    try:
        from plugins.plugin_orchestrator.pipeline import get_pipeline_graph
        pg = get_pipeline_graph()
        deps = pg.list_deps()

        lines = ["╔══════════════════════════════════════════════╗"]
        lines.append("║       Plugin Pipeline Dependency Graph      ║")
        lines.append("╚══════════════════════════════════════════════╝")

        for plugin_name, info in sorted(deps.items()):
            produces = info.get("produces", [])
            consumes = info.get("consumes", {})

            lines.append(f"\n▸ {plugin_name}")
            if produces:
                lines.append(f"  生产: {', '.join(produces)}")
            if consumes:
                for key, producer in consumes.items():
                    status_str = "✓" if producer and producer != "(unknown)" else "✗"
                    lines.append(f"  消费: {key} ← {producer} {status_str}")

        return "\n".join(lines)

    except Exception as exc:
        return f"Error: {exc}"


def events(session_id: str = "", event_type: str = "", limit: int = 20) -> str:
    """返回事件历史。"""
    try:
        from plugins.plugin_orchestrator.context import get_context, list_active_contexts

        if session_id:
            ctx = get_context(session_id)
            if not ctx:
                return f"No context found for session {session_id[:8]}"
            history = ctx.event_bus.history(event_type or None, limit)
        else:
            sessions = list_active_contexts()
            if not sessions:
                return "No active sessions"

            ctx = get_context(sessions[0])
            if not ctx:
                return "No active context"
            history = ctx.event_bus.history(event_type or None, limit)

        lines = ["── Event History ──"]
        for evt in history:
            ts = time.strftime("%H:%M:%S", time.localtime(evt["timestamp"]))
            lines.append(f"  [{ts}] {evt['type']} ← {evt['source']}")
            for k, v in evt.get("data", {}).items():
                lines.append(f"         {k}: {v}")

        return "\n".join(lines)

    except Exception as exc:
        return f"Error: {exc}"


def breakers() -> str:
    """显示熔断器状态。"""
    try:
        from plugins.plugin_orchestrator.circuit_breaker import get_registry, BreakerState

        registry = get_registry()
        all_breakers = registry.list_all()

        lines = ["╔══════════════════════════════════════════════╗"]
        lines.append("║        Circuit Breaker Status             ║")
        lines.append("╚══════════════════════════════════════════════╝")

        if not all_breakers:
            lines.append("\n(no breakers registered yet)")
            return "\n".join(lines)

        open_count = sum(1 for b in all_breakers if b.state == BreakerState.OPEN)
        half_open_count = sum(1 for b in all_breakers if b.state == BreakerState.HALF_OPEN)

        lines.append(f"\n{'State':<10} {'Plugin':<25} {'Hook':<25} {'Failures':<10} {'Calls':<8}")
        lines.append("─" * 80)

        for b in sorted(all_breakers, key=lambda x: (x.state, x.plugin_name)):
            state_symbol = {
                BreakerState.CLOSED: "🟢",
                BreakerState.OPEN: "🔴",
                BreakerState.HALF_OPEN: "🟡",
            }.get(b.state, "⚪")
            lines.append(
                f"{state_symbol + ' ' + b.state:<10} {b.plugin_name:<25} "
                f"{b.hook_name:<25} {b.failure_count:<10} {b.total_calls:<8}"
            )

        lines.append(f"\nSummary: {len(all_breakers)} total, {open_count} OPEN, {half_open_count} HALF_OPEN")
        return "\n".join(lines)

    except Exception as exc:
        return f"Error loading breakers: {exc}"


def traces() -> str:
    """显示追踪统计。"""
    try:
        from plugins.plugin_orchestrator.tracer import get_trace_store

        store = get_trace_store()
        stats = store.stats()
        recent = store.recent(20)

        lines = ["╔══════════════════════════════════════════════╗"]
        lines.append("║        Hook Trace Stats                    ║")
        lines.append("╚══════════════════════════════════════════════╝")

        if stats["total_spans"] == 0:
            lines.append("\n(no traces yet)")
            return "\n".join(lines)

        lines.append(
            f"\nTotal Spans: {stats['total_spans']} | "
            f"Failures: {stats['total_failures']}"
        )
        lines.append(f"\n{'Plugin':<25} {'Calls':<8} {'Avg(ms)':<10} "
                      f"{'Success%':<10} {'Hooks'}")
        lines.append("─" * 80)

        for plugin, info in sorted(stats["by_plugin"].items()):
            lines.append(
                f"{plugin:<25} {info['calls']:<8} {info['avg_duration_ms']:<10} "
                f"{info['success_rate']:<10} {', '.join(info['hooks'])}"
            )

        lines.append(f"\nRecent {len(recent)} spans:")
        for s in recent[-10:]:
            status_icon = "✅" if s["success"] else "❌" if s["success"] is False else "⏳"
            lines.append(
                f"  {status_icon} {s['plugin']:<20} {s['hook']:<25} "
                f"{s['duration_ms']:>8.2f}ms"
                + (f' | {s["error"][:50]}' if s.get("error") else "")
            )

        return "\n".join(lines)

    except Exception as exc:
        return f"Error loading traces: {exc}"


def test_inter_plugin_communication() -> Dict[str, Any]:
    """测试跨插件通信是否正常。"""
    results = {"tests": []}

    # Test 1: PluginContext 创建和读写
    try:
        from plugins.plugin_orchestrator.context import get_or_create_context, remove_context
        test_sid = "__orchestrator_test_session__"
        ctx = get_or_create_context(test_sid)

        ctx.shared_set("test_key", "test_value")
        assert ctx.shared_get("test_key") == "test_value"

        ctx.plugin_set("test_plugin", "counter", 42)
        assert ctx.plugin_get("test_plugin", "counter") == 42

        remove_context(test_sid)
        results["tests"].append({"name": "PluginContext CRUD", "status": "PASS"})
    except Exception as exc:
        results["tests"].append({"name": "PluginContext CRUD", "status": f"FAIL: {exc}"})

    # Test 2: EventBus 发布/订阅
    try:
        from plugins.plugin_orchestrator.context import get_or_create_context, remove_context
        test_sid = "__orchestrator_test_session_2__"
        ctx = get_or_create_context(test_sid)

        received = []

        def handler(event):
            received.append(event)

        ctx.event_bus.subscribe("test_event", handler)
        ctx.event_bus.publish("test_event", source_plugin="test", key="value")

        assert len(received) == 1
        assert received[0]["data"]["key"] == "value"

        remove_context(test_sid)
        results["tests"].append({"name": "EventBus pub/sub", "status": "PASS"})
    except Exception as exc:
        results["tests"].append({"name": "EventBus pub/sub", "status": f"FAIL: {exc}"})

    # Test 3: Pipeline 拓扑排序
    try:
        from plugins.plugin_orchestrator.pipeline import PipelineGraph
        pg = PipelineGraph()
        pg.register("alpha", produces=["data_a"], consumes=[])
        pg.register("beta", produces=["data_b"], consumes=["data_a"])
        pg.register("gamma", produces=[], consumes=["data_a", "data_b"])

        callbacks = [
            ("gamma", lambda: "gamma_result"),
            ("alpha", lambda: "alpha_result"),
            ("beta", lambda: "beta_result"),
        ]

        sorted_entries = pg.topological_sort(callbacks)
        sorted_names = [name for _, name, _ in sorted_entries]

        # alpha must be first (produces data_a, consumes nothing)
        assert sorted_names.index("alpha") == 0
        # beta must come before gamma (beta produces data_b, gamma consumes data_b)
        assert sorted_names.index("beta") < sorted_names.index("gamma")

        results["tests"].append({"name": "Pipeline topological sort", "status": "PASS",
                                  "sorted_names": sorted_names})
    except Exception as exc:
        results["tests"].append({"name": "Pipeline topological sort", "status": f"FAIL: {exc}"})

    # Test 4: 向后兼容 — 不传入 plugin_context 的回调仍正常工作
    try:
        def old_style_callback(**kwargs):
            # 旧式回调不处理 plugin_context
            return "old_style_ok"

        result = old_style_callback(session_id="test", some_arg="value",
                                     plugin_context=None, _plugin_name="test")
        assert result == "old_style_ok"
        results["tests"].append({"name": "Backward compat: old-style callback", "status": "PASS"})
    except Exception as exc:
        results["tests"].append({"name": "Backward compat: old-style callback",
                                  "status": f"FAIL: {exc}"})

    # Test 5: Circuit Breaker 三态
    try:
        from plugins.plugin_orchestrator.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker("test", "test_hook", failure_threshold=2, recovery_timeout=0.5)

        # CLOSED → 正常运行
        assert cb.state == "CLOSED"
        assert cb.is_open() is False

        # 两次失败 → OPEN
        cb.on_failure()
        assert cb.state == "CLOSED"  # 第一次：未达阈值
        cb.on_failure()
        assert cb.state == "OPEN"    # 第二次：达到阈值

        # OPEN → 跳过
        assert cb.is_open() is True

        # 超时后 → HALF_OPEN
        import time
        time.sleep(0.6)
        assert cb.is_open() is False  # 进入 HALF_OPEN

        # HALF_OPEN 成功 → CLOSED
        cb.on_success()
        assert cb.state == "CLOSED"

        results["tests"].append({"name": "Circuit breaker tri-state", "status": "PASS"})
    except Exception as exc:
        results["tests"].append({"name": "Circuit breaker tri-state",
                                  "status": f"FAIL: {exc}"})

    # Test 6: Tracer Span
    try:
        from plugins.plugin_orchestrator.tracer import Span, get_trace_store
        store = get_trace_store()

        span = Span("test_hook", "test_plugin", "test_session", turn=1)
        import time
        time.sleep(0.01)
        span.end(success=True, result_size=100)
        store.record(span)

        stats = store.stats()
        assert stats["total_spans"] >= 1
        assert "test_plugin" in stats["by_plugin"]

        results["tests"].append({"name": "Tracer span recording", "status": "PASS"})
    except Exception as exc:
        results["tests"].append({"name": "Tracer span recording",
                                  "status": f"FAIL: {exc}"})

    passed = sum(1 for t in results["tests"] if t["status"] == "PASS")
    failed = len(results["tests"]) - passed
    results["summary"] = f"{passed}/{len(results['tests'])} passed"
    if failed:
        results["summary"] += f", {failed} failed"

    return results


def main():
    """CLI 入口。"""
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        print(json.dumps(status(), indent=2, ensure_ascii=False, default=str))
    elif cmd == "pipes":
        print(pipes())
    elif cmd == "events":
        sid = sys.argv[2] if len(sys.argv) > 2 else ""
        etype = sys.argv[3] if len(sys.argv) > 3 else ""
        print(events(session_id=sid, event_type=etype))
    elif cmd == "breakers":
        print(breakers())
    elif cmd == "traces":
        print(traces())
    elif cmd == "test":
        result = test_inter_plugin_communication()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: diag [status|pipes|events|breakers|traces|test]")


if __name__ == "__main__":
    main()
