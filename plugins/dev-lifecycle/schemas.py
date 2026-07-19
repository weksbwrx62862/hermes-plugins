"""dev-lifecycle 插件 — 工具 schema 定义。"""

from typing import Any, Dict

try:
    from .gates import get_role_boundary, RoleBoundary
except ImportError:
    from gates import get_role_boundary, RoleBoundary


DEV_WORKFLOW_SCHEMA = {
    "name": "dev_workflow",
    "description": (
        "软件开发生命周期导航工具。列出所有可用阶段和技能，或获取特定阶段的引导信息。\n"
        "当 agent 需要了解软件开发流程的当前阶段应该使用哪些技能时调用。\n"
        "三个阶段：\n"
        "  1. 构思 (ideate): grill-me, grill-with-docs, to-prd, to-issues, plan, writing-plans\n"
        "  2. 构建 (build): prototype, spike, improve-codebase-architecture, zoom-out\n"
        "     test-driven-development, subagent-driven-development, user-auth-system\n"
        "  3. 交付 (deliver): systematic-debugging, python-debugpy, node-inspect-debugger\n"
        "     requesting-code-review, triage, handoff\n"
        "操作类型：\n"
        "  - overview: 列出所有阶段和包含的技能\n"
        "  - stage: 获取特定阶段的详细引导（需指定 stage_name）\n"
        "  - skill: 获取特定技能的摘要和路径（需指定 skill_name）\n"
        "  - start: 启动新项目的生命周期跟踪（需指定 project_path）\n"
        "  - advance: 推进到下一个技能（需指定 skill_name）\n"
        "  - rollback: 回退到指定阶段（需指定 to_stage）\n"
        "  - resume: 恢复已有项目的生命周期（需指定 project_path）\n"
        "  - report: 生成当前项目进度报告"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["overview", "stage", "skill", "start", "advance", "rollback", "resume", "report", "kanban"],
                "description": (
                    "操作类型：\n"
                    "- overview: 列出所有阶段和包含的技能\n"
                    "- stage: 获取特定阶段的详细引导（需指定 stage_name）\n"
                    "- skill: 获取特定技能的摘要和路径（需指定 skill_name）\n"
                    "- start: 启动新项目的生命周期跟踪（需指定 project_path）\n"
                    "- advance: 推进到下一个技能（需指定 skill_name）\n"
                    "- rollback: 回退到指定阶段（需指定 to_stage）\n"
                    "- resume: 恢复已有项目的生命周期（需指定 project_path）\n"
                    "- report: 生成当前项目进度报告\n"
                    "- kanban: 生成看板视图（可指定 project_path）"
                ),
            },
            "stage_name": {
                "type": "string",
                "enum": ["ideate", "build", "deliver"],
                "description": "阶段名称（action=stage 时必填）",
            },
            "skill_name": {
                "type": "string",
                "description": "技能名称（action=skill/advance 时必填），如 test-driven-development",
            },
            "project_path": {
                "type": "string",
                "description": "项目绝对路径（action=start/resume 时必填）",
            },
            "to_stage": {
                "type": "string",
                "enum": ["ideate", "build", "deliver"],
                "description": "回退目标阶段（action=rollback 时必填）",
            },
        },
        "required": ["action"],
        "allOf": [
            {
                "if": {"properties": {"action": {"const": "stage"}}, "required": ["action"]},
                "then": {"required": ["stage_name"]},
            },
            {
                "if": {"properties": {"action": {"const": "skill"}}, "required": ["action"]},
                "then": {"required": ["skill_name"]},
            },
            {
                "if": {"properties": {"action": {"const": "start"}}, "required": ["action"]},
                "then": {"required": ["project_path"]},
            },
            {
                "if": {"properties": {"action": {"const": "advance"}}, "required": ["action"]},
                "then": {"required": ["skill_name"]},
            },
            {
                "if": {"properties": {"action": {"const": "rollback"}}, "required": ["action"]},
                "then": {"required": ["to_stage"]},
            },
            {
                "if": {"properties": {"action": {"const": "resume"}}, "required": ["action"]},
                "then": {"required": ["project_path"]},
            },
        ],
    },
}


# ── 角色边界辅助函数 ────────────────────────────────────────────


def enrich_skill_with_role_boundary(skill_info: Dict[str, Any]) -> Dict[str, Any]:
    """为技能信息字典补充 role_boundary 字段。

    根据 skill_info 中的 name 或 skill_name 字段查询 ROLE_BOUNDARIES 注册表，
    若找到对应边界则写入 role_boundary 字段；否则置为 None。

    Args:
        skill_info: 技能信息字典，需包含 "name" 或 "skill_name" 键。

    Returns:
        原始字典（已就地补充 role_boundary 字段），便于链式调用。
    """
    skill_name = skill_info.get("name") or skill_info.get("skill_name")
    skill_info["role_boundary"] = get_role_boundary(skill_name) if skill_name else None
    return skill_info


__all__ = ["DEV_WORKFLOW_SCHEMA", "enrich_skill_with_role_boundary", "get_role_boundary", "RoleBoundary"]
