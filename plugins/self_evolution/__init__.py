"""
自进化插件 — Hermes 标准工具版

8阶段 Loop | 3层评测 | 10维 AND 门控

提供 7 个标准 Hermes 工具：
  self_evo_status    - 查看进化状态
  self_evo_scan      - 扫描候选技能
  self_evo_approve   - 批准进化请求
  self_evo_reject    - 拒绝进化请求
  self_evo_execute   - 执行已批准的进化
  self_evo_evolve    - 直接进化指定技能
  self_evo_rollback  - 回滚技能到备份
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes_plugins.self_evolution")

# ── 模块级插件上下文，用于部署成功后发布事件 ──
_plugin_context = None

# ── 确保 self_evolution 包可导入 ──
_SELF_EVO_DIR = str(Path(__file__).parent)
_PARENT_DIR = str(Path(__file__).parent.parent)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _SELF_EVO_DIR not in sys.path:
    sys.path.insert(0, _SELF_EVO_DIR)


# ═══════════════════════════════════════════════════════════════════
# API 凭据解析
# ═══════════════════════════════════════════════════════════════════

import os

def _resolve_api_credentials(model: str) -> tuple[Optional[str], Optional[str]]:
    """根据模型名称自动选择 API key 和 base_url。"""
    m = model.lower()
    if "mimo" in m:
        key = os.getenv("MIMO_API_KEY") or os.getenv("MIMO_API_KEY_4")
        url = os.getenv("MIMO_BASE_URL") or os.getenv("MIMO_BASE_URL_4")
        return key, url
    elif "deepseek" in m:
        key = os.getenv("DEEPSEEK_API_KEY")
        url = "https://api.deepseek.com/v1" if key else None
        return key, url
    else:
        key = os.getenv("OPENAI_API_KEY")
        return key, None


# ═══════════════════════════════════════════════════════════════════
# 工具处理器
# ═══════════════════════════════════════════════════════════════════

def _tool_status(args: Dict[str, Any], **_kw) -> str:
    """查看所有 skill 的进化状态。"""
    try:
        from self_evolution.triggers.auto_trigger import AutoEvolutionManager
        aem = AutoEvolutionManager()
        states = aem.list_all()

        result = []
        for s in states:
            result.append({
                "skill": s.skill_name,
                "status": s.status,
                "usage_count": s.usage_count,
                "evolution_count": s.evolution_count,
                "evolved_score": round(s.evolved_score, 3) if s.evolved_score else 0,
            })

        pending = [s for s in states if s.status == "pending"]
        approved = [s for s in states if s.status == "approved"]

        return json.dumps({
            "total": len(states),
            "pending": len(pending),
            "approved": len(approved),
            "skills": result[:50],  # 限制返回数量
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _tool_scan(args: Dict[str, Any], **_kw) -> str:
    """扫描所有 skill，筛选达到进化阈值的候选。"""
    try:
        from self_evolution.triggers.auto_trigger import AutoEvolutionManager
        aem = AutoEvolutionManager()
        candidates = aem.scan_candidates()

        if not candidates:
            return json.dumps({
                "status": "ok",
                "candidates": 0,
                "message": "没有需要进化的 skill",
            }, ensure_ascii=False)

        result = []
        for c in candidates:
            result.append({
                "skill": c.skill_name,
                "usage_count": c.usage_count,
                "current_score": round(c.evolved_score, 3) if c.evolved_score else 0,
                "reason": getattr(c, "reason", ""),
            })

        return json.dumps({
            "status": "ok",
            "candidates": len(candidates),
            "skills": result,
            "message": f"发现 {len(candidates)} 个候选 skill，使用 self_evo_approve 批准",
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _tool_approve(args: Dict[str, Any], **_kw) -> str:
    """批准指定 skill 的进化请求。"""
    name = args.get("name", "")
    if not name:
        return json.dumps({"error": "name is required"})

    try:
        from self_evolution.triggers.auto_trigger import AutoEvolutionManager
        aem = AutoEvolutionManager()
        if aem.approve(name):
            return json.dumps({
                "status": "ok",
                "message": f"已批准 {name} 的进化请求，使用 self_evo_execute 执行",
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "status": "error",
                "message": f"{name} 不在待审批状态",
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _tool_reject(args: Dict[str, Any], **_kw) -> str:
    """拒绝指定 skill 的进化请求。"""
    name = args.get("name", "")
    if not name:
        return json.dumps({"error": "name is required"})

    try:
        from self_evolution.triggers.auto_trigger import AutoEvolutionManager
        aem = AutoEvolutionManager()
        if aem.reject(name):
            return json.dumps({
                "status": "ok",
                "message": f"已拒绝 {name}，进入冷却期",
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "status": "error",
                "message": f"{name} 不在待审批状态",
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _tool_execute(args: Dict[str, Any], **_kw) -> str:
    """执行所有已批准的进化任务。"""
    try:
        from self_evolution.triggers.auto_trigger import AutoEvolutionManager
        from self_evolution.core.evolution_manager import EvolutionManager
        from self_evolution.core.evolution_provider import EvolutionPhase

        aem = AutoEvolutionManager()
        approved = aem.get_approved()

        if not approved:
            return json.dumps({
                "status": "ok",
                "message": "没有已批准的进化任务",
                "executed": 0,
            }, ensure_ascii=False)

        # 初始化进化管理器（使用 create_manager 注册默认 provider）
        manager = create_manager()
        target_dir = str(Path.home() / ".hermes")
        iterations = args.get("iterations", 5)
        fast = args.get("fast", True)
        auto_deploy = args.get("auto_deploy", True)

        # 解析 API key / base_url（按模型名称自动选择）
        model_name = args.get("model", "deepseek-v4-pro")
        api_key, base_url = _resolve_api_credentials(model_name)

        manager.initialize_all(
            target_dir,
            model=model_name,
            iterations=iterations,
            use_llm_eval=not fast,
            api_key=api_key,
            base_url=base_url,
        )

        # 快照进化前的 harness 状态
        try:
            from self_evolution.core.harness_versioning import HarnessVersioner
            versioner = HarnessVersioner()
            pre_snapshot = versioner.snapshot(
                trigger="pre_evolution",
                description=f"进化前快照: {len(approved)} 个技能待进化",
            )
        except Exception:
            pre_snapshot = None

        # 初始化回归门限
        try:
            from self_evolution.core.regression_gate import RegressionGate
            regression_gate = RegressionGate()
        except Exception:
            regression_gate = None

        results = []
        for state in approved:
            aem.mark_executing(state.skill_name)
            try:
                result = manager.evolve_skill(
                    state.skill_name,
                    phase=EvolutionPhase.SKILL,
                    iterations=iterations,
                    auto_deploy=auto_deploy,
                )
                if result.error:
                    # 记录失败轨迹
                    try:
                        from self_evolution.core.failure_tracker import FailureTracker
                        FailureTracker().record(
                            error_text=result.error,
                            skill_name=state.skill_name,
                            context=f"evolution pipeline: {result.phase.name}",
                        )
                    except Exception:
                        pass
                    results.append({
                        "skill": state.skill_name,
                        "status": "failed",
                        "error": result.error,
                    })
                    aem._states[state.skill_name].status = "pending"
                else:
                    # 回归门限评估
                    gate_result = None
                    if regression_gate and result.holdout_score > 0:
                        try:
                            gate_result = regression_gate.evaluate_from_evolution_result(
                                skill_name=state.skill_name,
                                baseline_val=result.baseline_score,
                                baseline_holdout=result.baseline_score,  # approx
                                evolved_val=result.evolved_score,
                                evolved_holdout=result.holdout_score,
                                constraint_passed=result.constraint_passed,
                            )
                        except Exception:
                            pass

                    entry = {
                        "skill": state.skill_name,
                        "status": "success",
                        "baseline": round(result.baseline_score, 3),
                        "evolved": round(result.evolved_score, 3),
                        "improvement": round(result.improvement, 3),
                        "holdout": round(result.holdout_score, 3),
                        "deployed": result.deployed,
                        "constraint_passed": result.constraint_passed,
                    }
                    if gate_result:
                        entry["regression_gate"] = gate_result.to_dict()
                    results.append(entry)
                    aem.mark_done(state.skill_name, result.evolved_score)

                    # 部署成功后发布 skill_updated 事件，通知 skill-router 刷新嵌入缓存
                    if result.deployed:
                        _publish_skill_updated(state.skill_name)
            except Exception as e:
                results.append({
                    "skill": state.skill_name,
                    "status": "error",
                    "error": str(e),
                })

        aem._save()

        # 进化后快照 + 版本对比
        post_info = {}
        try:
            if pre_snapshot:
                post_snapshot = versioner.snapshot(
                    trigger="post_evolution",
                    description=f"进化后快照: {len(results)} 个技能已处理",
                )
                diff = versioner.diff(pre_snapshot.version_id, post_snapshot.version_id)
                post_info = {
                    "harness_pre": pre_snapshot.version_id,
                    "harness_post": post_snapshot.version_id,
                    "harness_changes": {
                        "added": len(diff.added_files),
                        "removed": len(diff.removed_files),
                        "modified": len(diff.modified_files),
                    },
                }
        except Exception:
            pass

        success_count = sum(1 for r in results if r["status"] == "success")
        return json.dumps({
            "status": "ok",
            "executed": len(results),
            "success": success_count,
            "results": results,
            **post_info,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _tool_evolve(args: Dict[str, Any], **_kw) -> str:
    """直接进化指定 skill（跳过审批流程）。"""
    name = args.get("name", "")
    if not name:
        return json.dumps({"error": "name is required"})

    try:
        from self_evolution.core.evolution_manager import EvolutionManager
        from self_evolution.core.evolution_provider import EvolutionPhase

        manager = create_manager()
        target_dir = str(Path.home() / ".hermes")
        iterations = args.get("iterations", 5)
        fast = args.get("fast", True)
        auto_deploy = args.get("auto_deploy", True)

        model_name = args.get("model", "deepseek-v4-pro")
        api_key, base_url = _resolve_api_credentials(model_name)

        manager.initialize_all(
            target_dir,
            model=model_name,
            iterations=iterations,
            use_llm_eval=not fast,
            api_key=api_key,
            base_url=base_url,
        )

        result = manager.evolve_skill(
            name,
            phase=EvolutionPhase.SKILL,
            iterations=iterations,
            auto_deploy=auto_deploy,
        )

        if result.error:
            return json.dumps({
                "status": "error",
                "skill": name,
                "error": result.error,
            }, ensure_ascii=False)

        constraint_checks = {}
        if result.constraint_details and hasattr(result.constraint_details, 'checks'):
            constraint_checks = {
                k: v for k, v in result.constraint_details.checks.items()
            }

        # 部署成功后发布 skill_updated 事件，通知 skill-router 刷新嵌入缓存
        if result.deployed:
            _publish_skill_updated(name)

        return json.dumps({
            "status": "ok",
            "skill": name,
            "baseline": round(result.baseline_score, 3),
            "evolved": round(result.evolved_score, 3),
            "improvement": round(result.improvement, 3),
            "iterations_used": result.iterations_used,
            "holdout_score": round(result.holdout_score, 3) if result.holdout_score else None,
            "constraint_passed": result.constraint_passed,
            "constraint_checks": constraint_checks,
            "deployed": result.deployed,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _tool_rollback(args: Dict[str, Any], **_kw) -> str:
    """回滚指定 skill 到最近备份。"""
    name = args.get("name", "")
    if not name:
        return json.dumps({"error": "name is required"})

    try:
        from self_evolution.core.evolution_manager import EvolutionManager
        from self_evolution.core.evolution_provider import EvolutionPhase

        manager = create_manager()
        target_dir = str(Path.home() / ".hermes")

        manager.initialize_all(target_dir, model="deepseek-v4-pro", iterations=1)
        provider = manager.get_provider(EvolutionPhase.SKILL)
        if not provider:
            return json.dumps({"error": "no SKILL provider available"})

        provider.initialize(target_dir)
        skill_path = Path(target_dir) / "skills" / name / "SKILL.md"

        if provider.handle_rollback(str(skill_path)):
            return json.dumps({
                "status": "ok",
                "message": f"已回滚 {name} 到最近备份",
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "status": "error",
                "message": f"{name} 回滚失败（可能没有备份）",
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _tool_failure_report(args: Dict[str, Any], **_kw) -> str:
    """查看失败轨迹分析报告 — Self-Harness 弱点挖掘"""
    try:
        from self_evolution.core.failure_tracker import FailureTracker
        tracker = FailureTracker()
        days = args.get("days", 30)
        report = tracker.generate_weakness_report(days=days)
        return report
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _tool_failure_record(args: Dict[str, Any], **_kw) -> str:
    """记录一条失败轨迹（供 omni_record_action 等 hook 调用）"""
    try:
        from self_evolution.core.failure_tracker import FailureTracker
        tracker = FailureTracker()
        error_text = args.get("error_text", "")
        if not error_text:
            return json.dumps({"error": "error_text is required"})

        record = tracker.record(
            error_text=error_text,
            session_id=args.get("session_id", ""),
            skill_name=args.get("skill_name", ""),
            tool_name=args.get("tool_name", ""),
            context=args.get("context", ""),
            turn_index=args.get("turn_index", 0),
            retry_count=args.get("retry_count", 0),
            resolved=args.get("resolved", False),
            pattern=args.get("pattern"),
        )
        return json.dumps({
            "status": "ok",
            "pattern": record.pattern,
            "harness_surface": record.harness_surface,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _tool_harness_snapshot(args: Dict[str, Any], **_kw) -> str:
    """快照当前 harness 状态 — 版本化管理"""
    try:
        from self_evolution.core.harness_versioning import HarnessVersioner
        versioner = HarnessVersioner()
        snapshot = versioner.snapshot(
            skill_name=args.get("skill_name", ""),
            trigger=args.get("trigger", "manual"),
            description=args.get("description", ""),
        )
        return json.dumps({
            "status": "ok",
            "version_id": snapshot.version_id,
            "file_count": len(snapshot.files),
            "trigger": snapshot.trigger,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _tool_harness_log(args: Dict[str, Any], **_kw) -> str:
    """查看 harness 版本日志"""
    try:
        from self_evolution.core.harness_versioning import HarnessVersioner
        versioner = HarnessVersioner()
        limit = args.get("limit", 10)
        log = versioner.format_version_log(limit=limit)
        return log
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════
# 工具定义
# ═══════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "self_evo_status",
        "description": "查看所有技能的自进化状态（监控/pending/已批准/执行中/完成）。",
        "parameters": {"type": "object", "properties": {}},
        "handler": _tool_status,
    },
    {
        "name": "self_evo_scan",
        "description": "扫描所有技能，筛选达到进化阈值的候选。用于发现需要进化的技能。",
        "parameters": {"type": "object", "properties": {}},
        "handler": _tool_scan,
    },
    {
        "name": "self_evo_approve",
        "description": "批准指定技能的进化请求。扫描后处于 pending 状态的技能可被批准。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名称"},
            },
            "required": ["name"],
        },
        "handler": _tool_approve,
    },
    {
        "name": "self_evo_reject",
        "description": "拒绝指定技能的进化请求，该技能将进入冷却期。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名称"},
            },
            "required": ["name"],
        },
        "handler": _tool_reject,
    },
    {
        "name": "self_evo_execute",
        "description": "执行所有已批准的进化任务。8阶段优化 Loop，自动部署通过验证的版本。",
        "parameters": {
            "type": "object",
            "properties": {
                "iterations": {"type": "integer", "description": "迭代次数（默认 5）", "default": 5},
                "fast": {"type": "boolean", "description": "快速模式（默认 true）", "default": True},
                "auto_deploy": {"type": "boolean", "description": "自动部署（默认 true）", "default": True},
                "model": {"type": "string", "description": "评估模型（默认 deepseek-v4-pro）"},
            },
        },
        "handler": _tool_execute,
    },
    {
        "name": "self_evo_evolve",
        "description": "直接进化指定技能（跳过审批）。适合手动触发单个技能的进化。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名称"},
                "iterations": {"type": "integer", "description": "迭代次数（默认 5）", "default": 5},
                "fast": {"type": "boolean", "description": "快速模式（默认 true）", "default": True},
                "auto_deploy": {"type": "boolean", "description": "自动部署（默认 true）", "default": True},
                "model": {"type": "string", "description": "评估模型（默认 deepseek-v4-pro）"},
            },
            "required": ["name"],
        },
        "handler": _tool_evolve,
    },
    {
        "name": "self_evo_rollback",
        "description": "回滚指定技能到最近备份版本。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名称"},
            },
            "required": ["name"],
        },
        "handler": _tool_rollback,
    },
    {
        "name": "self_evo_failure_report",
        "description": "查看失败轨迹分析报告 — Self-Harness 弱点挖掘。按失败模式聚类，显示高频失败和改进建议。",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "分析最近N天（默认30）", "default": 30},
            },
        },
        "handler": _tool_failure_report,
    },
    {
        "name": "self_evo_failure_record",
        "description": "记录一条失败轨迹。自动分类失败模式并推断影响的harness表面。",
        "parameters": {
            "type": "object",
            "properties": {
                "error_text": {"type": "string", "description": "错误文本"},
                "session_id": {"type": "string", "description": "会话ID"},
                "skill_name": {"type": "string", "description": "涉及的技能"},
                "tool_name": {"type": "string", "description": "涉及的工具"},
                "context": {"type": "string", "description": "上下文"},
                "pattern": {"type": "string", "description": "失败模式（可选，自动分类）"},
                "resolved": {"type": "boolean", "description": "是否已解决"},
            },
            "required": ["error_text"],
        },
        "handler": _tool_failure_record,
    },
    {
        "name": "self_evo_harness_snapshot",
        "description": "快照当前 harness 状态（系统提示+工具描述+技能文件）— 版本化管理。",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "关联的技能名"},
                "trigger": {"type": "string", "description": "触发原因", "default": "manual"},
                "description": {"type": "string", "description": "描述"},
            },
        },
        "handler": _tool_harness_snapshot,
    },
    {
        "name": "self_evo_harness_log",
        "description": "查看 harness 版本日志 — 每次进化前后的版本对比。",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "显示条数（默认10）", "default": 10},
            },
        },
        "handler": _tool_harness_log,
    },
]


# ═══════════════════════════════════════════════════════════════════
# 技能更新事件发布
# ═══════════════════════════════════════════════════════════════════

def _publish_skill_updated(skill_name: str) -> None:
    """技能部署成功后发布 skill_updated 事件，通知 skill-router 刷新嵌入缓存。

    双通道通知：
      1. 通过 shared_state 写入 last_skill_updated（同步，无需 orchestrator）
      2. 通过 orchestrator EventBus 发布 skill_updated 事件（异步，需 orchestrator 启用）
    """
    global _plugin_context
    if not skill_name:
        return

    # 通道1: shared_state（同步，始终可用）
    if _plugin_context and hasattr(_plugin_context, "shared_set"):
        try:
            _plugin_context.shared_set("last_skill_updated", skill_name)
            logger.info("已写入 shared_state: last_skill_updated=%s", skill_name)
        except Exception as e:
            logger.debug("写入 shared_state 失败: %s", e)

    # 通道2: orchestrator EventBus（需 orchestrator 启用）
    try:
        from plugins.plugin_orchestrator.context import get_context
        orch_ctx = get_context()
        if orch_ctx and hasattr(orch_ctx, "publish"):
            orch_ctx.publish("skill_updated", {"skill_name": skill_name})
            logger.info("已发布 skill_updated 事件: %s", skill_name)
    except Exception:
        pass  # orchestrator 未启用时不影响正常工作


# ═══════════════════════════════════════════════════════════════════
# Hermes 插件注册
# ═══════════════════════════════════════════════════════════════════

def register(ctx) -> None:
    """Hermes 插件入口：注册所有工具。"""
    global _plugin_context
    _plugin_context = ctx

    registered = 0
    for tool_def in TOOLS:
        name = tool_def["name"]
        handler = tool_def["handler"]
        schema = tool_def.get("parameters", {})
        try:
            ctx.register_tool(
                name=name,
                handler=handler,
                schema=schema,
                toolset="self_evolution",
            )
            registered += 1
        except Exception as exc:
            logger.warning("SelfEvolution: failed to register tool %s: %s", name, exc)

    logger.info(
        "SelfEvolution v1.0 registered: %d tools",
        registered,
    )


def create_manager(optimizer_name: str = "diversify") -> "EvolutionManager":
    """创建并初始化 EvolutionManager，注册默认提供者。"""
    from self_evolution.core.evolution_manager import EvolutionManager
    from self_evolution.default_provider import DefaultEvolutionProvider
    manager = EvolutionManager()
    manager.add_provider(DefaultEvolutionProvider(optimizer_name=optimizer_name))
    return manager


__all__ = ["register", "TOOLS", "create_manager"]
