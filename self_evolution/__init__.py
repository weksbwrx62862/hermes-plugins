"""
自进化插件 — Hermes 标准工具版

8阶段 Loop | 3层评测 | 5维 AND 门控

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

# ── 确保 self_evolution 包可导入 ──
_SELF_EVO_DIR = str(Path(__file__).parent)
_PARENT_DIR = str(Path(__file__).parent.parent)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _SELF_EVO_DIR not in sys.path:
    sys.path.insert(0, _SELF_EVO_DIR)


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

        # 初始化进化管理器
        manager = EvolutionManager()
        target_dir = str(Path.home() / ".hermes")
        iterations = args.get("iterations", 5)
        fast = args.get("fast", True)
        auto_deploy = args.get("auto_deploy", True)

        manager.initialize_all(
            target_dir,
            model=args.get("model", "deepseek-v4-pro"),
            iterations=iterations,
            use_llm_eval=not fast,
        )

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
                    results.append({
                        "skill": state.skill_name,
                        "status": "failed",
                        "error": result.error,
                    })
                    aem._states[state.skill_name].status = "pending"
                else:
                    results.append({
                        "skill": state.skill_name,
                        "status": "success",
                        "baseline": round(result.baseline_score, 3),
                        "evolved": round(result.evolved_score, 3),
                        "improvement": round(result.improvement, 3),
                        "deployed": result.deployed,
                        "constraint_passed": result.constraint_passed,
                    })
                    aem.mark_done(state.skill_name, result.evolved_score)
            except Exception as e:
                results.append({
                    "skill": state.skill_name,
                    "status": "error",
                    "error": str(e),
                })

        aem._save()

        success_count = sum(1 for r in results if r["status"] == "success")
        return json.dumps({
            "status": "ok",
            "executed": len(results),
            "success": success_count,
            "results": results,
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

        manager = EvolutionManager()
        target_dir = str(Path.home() / ".hermes")
        iterations = args.get("iterations", 5)
        fast = args.get("fast", True)
        auto_deploy = args.get("auto_deploy", True)

        manager.initialize_all(
            target_dir,
            model=args.get("model", "deepseek-v4-pro"),
            iterations=iterations,
            use_llm_eval=not fast,
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

        manager = EvolutionManager()
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
]


# ═══════════════════════════════════════════════════════════════════
# Hermes 插件注册
# ═══════════════════════════════════════════════════════════════════

def register(ctx) -> None:
    """Hermes 插件入口：注册所有工具。"""
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
