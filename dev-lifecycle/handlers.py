"""dev-lifecycle 插件 — 工具处理器和生命周期技能注册表。

维护软件开发生命周期 11 个阶段 → 21 个技能的完整映射，
驱动 dev_workflow 工具的导航逻辑。
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    from .state import WorkflowManager
    from .gates import QualityGateManager
    from .context import ProjectDetector, SkillRecommender
    from .telemetry import TelemetryRecorder, TelemetryEvent
    from .constants import LIFECYCLE, AUX_SKILLS
except ImportError:
    from state import WorkflowManager
    from gates import QualityGateManager
    from context import ProjectDetector, SkillRecommender
    from telemetry import TelemetryRecorder, TelemetryEvent
    from constants import LIFECYCLE, AUX_SKILLS

logger = logging.getLogger("plugins.dev-lifecycle")

# 由 __init__.register() 注入
_plugin_ctx: Optional[object] = None

SKILLS_DIR = Path(os.path.expanduser("~/.hermes/skills/software-development"))

# 技能路径缓存
_skill_path_cache: Dict[str, Optional[Tuple[float, Path]]] = {}

# 技能元数据缓存（键为技能名，值为 (mtime, metadata_dict)）
_summary_cache: Dict[str, Tuple[float, dict]] = {}

_workflow_mgr: Optional[WorkflowManager] = None
_gate_mgr: Optional[QualityGateManager] = None
_project_detector: Optional[ProjectDetector] = None
_skill_recommender: Optional[SkillRecommender] = None
_telemetry: Optional[TelemetryRecorder] = None


def init_modules() -> None:
    """初始化所有子模块单例。"""
    global _workflow_mgr, _gate_mgr, _project_detector, _skill_recommender, _telemetry
    logger.info("初始化子模块单例")
    _workflow_mgr = WorkflowManager()
    logger.debug("WorkflowManager 已初始化")
    _gate_mgr = QualityGateManager()
    logger.debug("QualityGateManager 已初始化，已注册内置门禁")
    _project_detector = ProjectDetector()
    logger.debug("ProjectDetector 已初始化")
    _skill_recommender = SkillRecommender()
    logger.debug("SkillRecommender 已初始化")
    _telemetry = TelemetryRecorder()
    logger.debug("TelemetryRecorder 已初始化")


def _parse_frontmatter(content: str) -> Tuple[dict, str]:
    """解析 SKILL.md 的 YAML frontmatter，返回 (frontmatter_dict, body_text)。

    如果没有 frontmatter 或解析失败，返回 ({}, 原始内容)。
    """
    m = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not m:
        return {}, content
    fm_text = m.group(1)
    body = content[m.end():]
    try:
        fm_dict = yaml.safe_load(fm_text)
        if not isinstance(fm_dict, dict):
            fm_dict = {}
    except yaml.YAMLError:
        fm_dict = {}
    return fm_dict, body


def _strip_frontmatter(content: str) -> str:
    """移除 SKILL.md 顶部的 YAML frontmatter（向后兼容）。"""
    _, body = _parse_frontmatter(content)
    return body


def _skill_path(name: str) -> Optional[Path]:
    """给定技能名，返回其 SKILL.md 路径（带 mtime 校验缓存）。"""
    if name in _skill_path_cache:
        cached_mtime, cached_path = _skill_path_cache[name]
        if cached_path is not None:
            try:
                current_mtime = cached_path.stat().st_mtime
                if current_mtime == cached_mtime:
                    logger.debug("技能路径缓存命中: %s", name)
                    return cached_path
            except OSError:
                pass
        else:
            logger.debug("技能路径缓存命中（不存在）: %s", name)
            return None

    logger.debug("技能路径缓存未命中: %s", name)
    candidate = SKILLS_DIR / name / "SKILL.md"
    if candidate.exists():
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            mtime = 0.0
        _skill_path_cache[name] = (mtime, candidate)
        return candidate
    try:
        for p in SKILLS_DIR.rglob(f"{name}/SKILL.md"):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            _skill_path_cache[name] = (mtime, p)
            return p
    except OSError as e:
        logger.warning("递归搜索技能路径失败: %s — %s", name, e)
    _skill_path_cache[name] = (0.0, None)
    return None


def _extract_section_lines(body: str, heading: str) -> List[str]:
    """从 Markdown 正文中提取指定 ## 标题下的条目列表。

    匹配 `## <heading>` 到下一个同级或更高级标题之间的内容，
    提取每行非空非标题文本（去除列表标记）。
    """
    pattern = re.compile(
        rf'^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s|\Z)',
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(body)
    if not m:
        return []
    section = m.group(1)
    items: List[str] = []
    for line in section.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cleaned = re.sub(r'^[-*]\s+', '', stripped)
        items.append(cleaned)
    return items


def _read_skill_meta(name: str) -> dict:
    """读取技能 SKILL.md 的完整元数据（带 mtime 缓存）。

    返回 dict 包含：name, summary, prerequisites, outputs, when_to_use。
    """
    p = _skill_path(name)
    if not p:
        return {
            "name": name,
            "summary": f"❌ 技能 '{name}' 未找到",
            "prerequisites": [],
            "outputs": [],
            "when_to_use": [],
        }
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0

    if name in _summary_cache:
        cached_mtime, cached_meta = _summary_cache[name]
        if cached_mtime == mtime:
            logger.debug("技能元数据缓存命中: %s", name)
            return cached_meta

    try:
        content = p.read_text()
    except (IOError, OSError) as e:
        logger.error("读取技能文件失败: %s — %s", p, e)
        return {
            "name": name,
            "summary": f"❌ 读取技能文件失败: {e}",
            "prerequisites": [],
            "outputs": [],
            "when_to_use": [],
        }

    fm_dict, body = _parse_frontmatter(content)

    # 提取 summary：前 8 行非标题行
    lines = [l.strip() for l in body.strip().split("\n") if l.strip() and not l.startswith("#")]
    summary = "\n".join(lines[:8])

    # 提取 prerequisites
    prerequisites: List[str] = []
    if "prerequisites" in fm_dict:
        val = fm_dict["prerequisites"]
        if isinstance(val, list):
            prerequisites = [str(v) for v in val]
        elif isinstance(val, str):
            prerequisites = [val]
    if not prerequisites:
        prerequisites = _extract_section_lines(body, "Prerequisites")

    # 提取 outputs
    outputs: List[str] = []
    if "outputs" in fm_dict:
        val = fm_dict["outputs"]
        if isinstance(val, list):
            outputs = [str(v) for v in val]
        elif isinstance(val, str):
            outputs = [val]
    if not outputs:
        outputs = _extract_section_lines(body, "Outputs")

    # 提取 when_to_use
    when_to_use: List[str] = []
    for key in ("when_to_use", "whenToUse"):
        if key in fm_dict:
            val = fm_dict[key]
            if isinstance(val, list):
                when_to_use = [str(v) for v in val]
            elif isinstance(val, str):
                when_to_use = [val]
            break
    if not when_to_use:
        when_to_use = _extract_section_lines(body, "When to Use")

    meta = {
        "name": name,
        "summary": summary,
        "prerequisites": prerequisites,
        "outputs": outputs,
        "when_to_use": when_to_use,
    }
    _summary_cache[name] = (mtime, meta)
    return meta


def _read_summary(name: str) -> str:
    """读取技能 SKILL.md 的前几段作为摘要（向后兼容，内部调用 _read_skill_meta）。"""
    meta = _read_skill_meta(name)
    return meta["summary"]


def warmup_skill_cache() -> None:
    """预热技能路径缓存，遍历所有已知技能名。"""
    logger.info("开始预热技能路径缓存")
    for info in LIFECYCLE.values():
        for skill_name, _ in info["flow"]:
            _skill_path(skill_name)
    for skill_name in AUX_SKILLS:
        _skill_path(skill_name)
    logger.info("技能路径缓存预热完成，共 %d 条", len(_skill_path_cache))


# ── 处理器 ────────────────────────────────────────────────────────


def handle_dev_workflow(args: Dict[str, Any], **kwargs) -> str:
    """dev_workflow 工具的主处理器。"""
    try:
        action = args.get("action", "overview")
        logger.info("dev_workflow 调用: action=%s", action)

        if action == "overview":
            return _handle_overview()
        elif action == "stage":
            stage = args.get("stage_name", "")
            if stage not in LIFECYCLE:
                return json.dumps({
                    "error": f"未知阶段 '{stage}'。可用: {list(LIFECYCLE.keys())}",
                    "hint": "使用 action='overview' 查看所有阶段",
                }, ensure_ascii=False)
            return _handle_stage(stage)
        elif action == "skill":
            name = args.get("skill_name", "")
            return _handle_skill(name)
        elif action == "start":
            project_path = args.get("project_path", "")
            return _handle_start(project_path)
        elif action == "advance":
            skill_name = args.get("skill_name", "")
            return _handle_advance(skill_name)
        elif action == "rollback":
            to_stage = args.get("to_stage", "")
            return _handle_rollback(to_stage)
        elif action == "resume":
            project_path = args.get("project_path", "")
            return _handle_resume(project_path)
        elif action == "report":
            return _handle_report()
        else:
            return json.dumps({"error": f"未知 action: {action}"})
    except Exception as e:
        logger.error("dev_workflow 处理异常: %s", e, exc_info=True)
        return json.dumps({
            "error": "dev_workflow 内部错误",
            "detail": str(e),
        }, ensure_ascii=False)


def _handle_overview() -> str:
    """返回完整生命周期概览。"""
    stages = []
    for key, info in sorted(LIFECYCLE.items(), key=lambda x: x[1]["order"]):
        stages.append({
            "stage": key,
            "emoji": info["emoji"],
            "name": info["name_cn"],
            "description": info["description"],
            "skill_count": len(info["flow"]),
            "skills": [s[0] for s in info["flow"]],
        })
    return json.dumps({
        "lifecycle": "软件开发生命周期 · 3 阶段 · 21+ 技能",
        "stages": stages,
        "aux_skills": _discover_aux_skills(),
        "usage": "action='stage' + stage_name 查看阶段详情; action='skill' + skill_name 查看技能摘要",
    }, ensure_ascii=False, indent=2)


def _discover_aux_skills() -> List[str]:
    """动态发现辅助技能，优先通过 skill-router，回退到硬编码列表。"""
    if _plugin_ctx is not None:
        try:
            result = _plugin_ctx.dispatch_tool("skill_search", {"query": "hermes agent skill authoring debugging tui commands"})
            if result:
                data = json.loads(result) if isinstance(result, str) else result
                if isinstance(data, dict) and "skills" in data:
                    discovered = [s.get("name", "") for s in data["skills"] if s.get("name")]
                    logger.info("skill-router 动态发现 %d 个辅助技能: %s", len(discovered), discovered)
                    return discovered
        except Exception as e:
            logger.warning("skill-router 动态发现失败，回退到硬编码列表: %s", e)
    fallback = list(AUX_SKILLS.keys())
    logger.debug("使用硬编码辅助技能列表: %s", fallback)
    return fallback


def _handle_stage(stage: str) -> str:
    """返回特定阶段的详细引导。"""
    info = LIFECYCLE[stage]
    skills_detail = []
    for skill_name, purpose in info["flow"]:
        p = _skill_path(skill_name)
        skills_detail.append({
            "name": skill_name,
            "purpose": purpose,
            "path": str(p) if p else "NOT FOUND",
            "exists": p is not None,
        })
    return json.dumps({
        "stage": stage,
        "emoji": info["emoji"],
        "name": info["name_cn"],
        "description": info["description"],
        "flow": skills_detail,
        "action_items": [
            f"加载具体技能: skill_view(name='{s[0]}')" for s in info["flow"]
        ],
    }, ensure_ascii=False, indent=2)


def _handle_skill(name: str) -> str:
    """返回特定技能的摘要。"""
    meta = _read_skill_meta(name)
    p = _skill_path(name)
    exists = p is not None

    stage = None
    for sk, info in LIFECYCLE.items():
        for sn, _ in info["flow"]:
            if sn == name:
                stage = sk
                break
        if stage:
            break

    result = {
        "skill": name,
        "exists": exists,
        "path": str(p) if p else None,
        "stage": stage,
        "summary": meta["summary"],
        "prerequisites": meta["prerequisites"],
        "outputs": meta["outputs"],
        "when_to_use": meta["when_to_use"],
    }

    if not exists:
        result["hint"] = "技能文件未找到，检查 ~/.hermes/skills/software-development/"

    return json.dumps(result, ensure_ascii=False, indent=2)


def _handle_start(project_path: str) -> str:
    """启动新工作流。"""
    if not project_path:
        logger.warning("action=start 缺少 project_path 参数")
        return json.dumps({"error": "project_path 不能为空"}, ensure_ascii=False)

    logger.info("启动新工作流: project_path=%s", project_path)

    if _workflow_mgr is None:
        logger.debug("工作流管理器未初始化，执行 init_modules()")
        init_modules()

    try:
        from .config import load_config
    except ImportError:
        from config import load_config
    config = load_config()
    stages_for_workflow = {}
    for stage_key, stage_val in config.stages.items():
        skills_list = [s[0] for s in stage_val.get("flow", [])]
        stages_for_workflow[stage_key] = {"skills": skills_list}
    logger.debug("工作流阶段配置: %s", {k: v["skills"] for k, v in stages_for_workflow.items()})

    state = _workflow_mgr.start(project_path, {"stages": stages_for_workflow})
    logger.info("工作流已创建: project=%s, stage=%s, 技能数=%d", state.project_path, state.current_stage, len(state.skills_status))

    project_ctx = None
    if _project_detector:
        project_ctx = _project_detector.detect(project_path)
        logger.info("项目上下文探测: type=%s, languages=%s, frameworks=%s", project_ctx.project_type, project_ctx.languages, project_ctx.frameworks)

    recommended = None
    if project_ctx and _skill_recommender:
        ideate_skills = config.stages.get("ideate", {}).get("flow", [])
        recommended = _skill_recommender.recommend(project_ctx, "ideate", ideate_skills)
        if recommended:
            logger.info("ideate 阶段推荐技能（按优先级）: %s", [s[0] for s in recommended])

    return json.dumps({
        "status": "started",
        "project_path": state.project_path,
        "current_stage": state.current_stage,
        "skills_status": state.skills_status,
        "project_context": {
            "type": project_ctx.project_type if project_ctx else "unknown",
            "languages": project_ctx.languages if project_ctx else [],
            "frameworks": project_ctx.frameworks if project_ctx else [],
        } if project_ctx else None,
        "recommended_skills": [(s[0], s[1]) for s in recommended] if recommended else None,
    }, ensure_ascii=False, indent=2)


def _handle_advance(skill_name: str) -> str:
    """推进技能状态。"""
    if not skill_name:
        logger.warning("action=advance 缺少 skill_name 参数")
        return json.dumps({"error": "skill_name 不能为空"}, ensure_ascii=False)

    if _workflow_mgr is None:
        logger.warning("action=advance 时工作流管理器未初始化")
        return json.dumps({"error": "没有活跃的工作流，请先使用 action='start'"}, ensure_ascii=False)

    active = _workflow_mgr.list_active()
    if not active:
        logger.warning("action=advance 时没有活跃的工作流")
        return json.dumps({"error": "没有活跃的工作流"}, ensure_ascii=False)

    state = active[0]
    logger.info("推进技能: skill=%s, 当前阶段=%s, 项目=%s", skill_name, state.current_stage, state.project_path)

    workflow_id = _get_workflow_id(state.project_path)
    if workflow_id is None:
        logger.error("无法获取工作流 ID: project_path=%s", state.project_path)
        return json.dumps({"error": "无法获取工作流 ID"}, ensure_ascii=False)

    result = _workflow_mgr.advance(workflow_id, skill_name)
    logger.info("技能推进结果: completed=%s, current_stage=%s, next_skill=%s, can_advance=%s",
                skill_name, result.get("current_stage"), result.get("next_skill"), result.get("can_advance"))

    if _telemetry:
        project_ctx = _project_detector.detect(state.project_path) if _project_detector else None
        _telemetry.record(TelemetryEvent(
            skill_name=skill_name,
            stage=state.current_stage,
            project_type=project_ctx.project_type if project_ctx else "unknown",
        ))
        logger.debug("遥测已记录: skill=%s, stage=%s", skill_name, state.current_stage)

    gate_result = None
    if result.get("can_advance") and _gate_mgr:
        from_stage = state.current_stage
        to_stage = result.get("current_stage", from_stage)
        if from_stage != to_stage:
            gate_result = _gate_mgr.check(from_stage, to_stage, {})
            if gate_result.passed:
                logger.info("质量门禁通过: %s → %s", from_stage, to_stage)
            else:
                logger.warning("质量门禁未通过: %s → %s, 失败项=%s, 建议=%s",
                               from_stage, to_stage, gate_result.failures, gate_result.suggestions)

    response = {
        "skill_completed": skill_name,
        "current_stage": result.get("current_stage"),
        "next_skill": result.get("next_skill"),
        "can_advance": result.get("can_advance"),
    }
    if gate_result and not gate_result.passed:
        response["gate_check"] = {
            "passed": False,
            "failures": gate_result.failures,
            "suggestions": gate_result.suggestions,
        }

    ama_suggestion = None
    if result.get("can_advance") and result.get("current_stage") == "build" and _plugin_ctx is not None:
        try:
            ama_result = _plugin_ctx.dispatch_tool("ama_assess", {"task_description": "build 阶段并行任务评估"})
            if ama_result:
                ama_data = json.loads(ama_result) if isinstance(ama_result, str) else ama_result
                if isinstance(ama_data, dict) and ama_data.get("recommended_mode"):
                    ama_suggestion = {
                        "mode": ama_data["recommended_mode"],
                        "hint": "建议使用 subagent-driven-development 技能进行并行执行",
                    }
                    logger.info("AMA 建议: mode=%s", ama_data["recommended_mode"])
        except Exception as e:
            logger.debug("AMA 评估失败（非致命）: %s", e)

    if ama_suggestion:
        response["ama_suggestion"] = ama_suggestion

    return json.dumps(response, ensure_ascii=False, indent=2)


def _handle_rollback(to_stage: str) -> str:
    """回退到指定阶段。"""
    if not to_stage:
        logger.warning("action=rollback 缺少 to_stage 参数")
        return json.dumps({"error": "to_stage 不能为空"}, ensure_ascii=False)

    if to_stage not in LIFECYCLE:
        logger.warning("action=rollback 阶段无效: %s", to_stage)
        return json.dumps({"error": f"无效阶段: {to_stage}，可用: {list(LIFECYCLE.keys())}"}, ensure_ascii=False)

    if _workflow_mgr is None:
        logger.warning("action=rollback 时工作流管理器未初始化")
        return json.dumps({"error": "没有活跃的工作流"}, ensure_ascii=False)

    active = _workflow_mgr.list_active()
    if not active:
        logger.warning("action=rollback 时没有活跃的工作流")
        return json.dumps({"error": "没有活跃的工作流"}, ensure_ascii=False)

    state = active[0]
    logger.info("回退工作流: 从 %s 回退到 %s, 项目=%s", state.current_stage, to_stage, state.project_path)

    workflow_id = _get_workflow_id(state.project_path)
    if workflow_id is None:
        logger.error("回退时无法获取工作流 ID: project_path=%s", state.project_path)
        return json.dumps({"error": "无法获取工作流 ID"}, ensure_ascii=False)

    try:
        new_state = _workflow_mgr.rollback(workflow_id, to_stage)
        logger.info("回退成功: 当前阶段=%s, 技能状态=%s", new_state.current_stage, new_state.skills_status)
    except ValueError as e:
        logger.error("回退失败: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    return json.dumps({
        "status": "rolled_back",
        "current_stage": new_state.current_stage,
        "skills_status": new_state.skills_status,
    }, ensure_ascii=False, indent=2)


def _handle_resume(project_path: str) -> str:
    """恢复已有工作流。"""
    if not project_path:
        logger.warning("action=resume 缺少 project_path 参数")
        return json.dumps({"error": "project_path 不能为空"}, ensure_ascii=False)

    logger.info("恢复工作流: project_path=%s", project_path)

    if _workflow_mgr is None:
        logger.debug("工作流管理器未初始化，执行 init_modules()")
        init_modules()

    state = _workflow_mgr.resume(project_path)
    if state is None:
        logger.info("项目 %s 没有未完成的工作流", project_path)
        return json.dumps({
            "status": "no_active_workflow",
            "hint": f"项目 {project_path} 没有未完成的工作流，使用 action='start' 创建",
        }, ensure_ascii=False)

    completed = sum(1 for s in state.skills_status.values() if s == "completed")
    total = len(state.skills_status)
    logger.info("工作流已恢复: 项目=%s, 阶段=%s, 进度=%d/%d", state.project_path, state.current_stage, completed, total)

    return json.dumps({
        "status": "resumed",
        "project_path": state.project_path,
        "current_stage": state.current_stage,
        "progress": f"{completed}/{total}",
        "skills_status": state.skills_status,
    }, ensure_ascii=False, indent=2)


def _handle_report() -> str:
    """生成使用报告。"""
    logger.info("生成遥测使用报告")

    if _telemetry is None:
        logger.debug("遥测记录器未初始化，执行 init_modules()")
        init_modules()

    report = _telemetry.report()
    logger.info("遥测报告: 总事件=%d, 技能数=%d", report.get("total_events", 0), len(report.get("skill_usage", {})))

    low_usage_skills = []
    for skill_name, count in report.get("skill_usage", {}).items():
        if count <= 1:
            skip_rate = report.get("skip_rate", {}).get(skill_name, 0)
            if skip_rate > 0.5:
                low_usage_skills.append(skill_name)

    if low_usage_skills:
        logger.info("检测到 %d 个低效技能（使用≤1次+跳过率>50%%）: %s", len(low_usage_skills), low_usage_skills)

    if low_usage_skills and _plugin_ctx is not None:
        try:
            _plugin_ctx.dispatch_tool("self_evo_scan", {
                "target_skills": low_usage_skills,
                "reason": "低使用率+高跳过率，建议优化",
            })
            logger.info("已向 self-evolution 提交 %d 个低效技能的优化建议", len(low_usage_skills))
        except Exception as e:
            logger.debug("self-evolution 提交失败（非致命）: %s", e)

    return json.dumps(report, ensure_ascii=False, indent=2)


def _get_workflow_id(project_path: str) -> Optional[int]:
    """根据 project_path 获取活跃工作流 ID。"""
    import sqlite3
    try:
        from .state import DB_PATH
    except ImportError:
        from state import DB_PATH
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM workflows WHERE project_path = ? AND is_active = 1 ORDER BY updated_at DESC LIMIT 1",
            (project_path,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            logger.debug("查询到活跃工作流 ID=%d, project_path=%s", row[0], project_path)
        else:
            logger.debug("未查询到活跃工作流, project_path=%s", project_path)
        return row[0] if row else None
    except Exception as e:
        logger.error("查询工作流 ID 异常: project_path=%s, 错误=%s", project_path, e)
        return None


def handle_on_session_start(**kwargs) -> None:
    """Session 启动时注入生命周期上下文提示，并检测未完成工作流。"""
    if _plugin_ctx is None:
        logger.debug("on_session_start: _plugin_ctx 为空，跳过")
        return

    hint = (
        "[dev-lifecycle] 可使用 dev_workflow 工具导航软件开发生命周期："
        "action='overview' 查看全貌，action='stage' 查看阶段，action='skill' 查看技能详情。"
    )

    if _workflow_mgr is None:
        logger.debug("on_session_start: 工作流管理器未初始化，执行 init_modules()")
        init_modules()

    try:
        active = _workflow_mgr.list_active()
        if active:
            state = active[0]
            completed = sum(1 for s in state.skills_status.values() if s == "completed")
            total = len(state.skills_status)
            hint += (
                f" 检测到未完成工作流：项目 {state.project_path}，"
                f"当前阶段 {state.current_stage}，进度 {completed}/{total}。"
                f"使用 action='resume' 恢复。"
            )
            logger.info("on_session_start: 检测到未完成工作流 — 项目=%s, 阶段=%s, 进度=%d/%d",
                        state.project_path, state.current_stage, completed, total)
        else:
            logger.debug("on_session_start: 没有未完成的工作流")
    except Exception as e:
        logger.warning("on_session_start: 检测工作流状态异常: %s", e)

    try:
        _plugin_ctx.inject_context(hint)
        logger.info("on_session_start: 已注入生命周期上下文提示")
    except Exception as e:
        logger.warning("注入生命周期上下文提示失败: %s", e)
