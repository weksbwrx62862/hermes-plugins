"""dev-lifecycle 插件 — 配置管理模块。

支持从 lifecycle.yaml 加载自定义配置，并与默认配置合并。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .constants import LIFECYCLE
except ImportError:
    from constants import LIFECYCLE

logger = logging.getLogger("plugins.dev-lifecycle")

CONFIG_PATH = Path(os.path.expanduser("~/.hermes/plugins/dev-lifecycle/lifecycle.yaml"))

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    logger.warning("PyYAML 未安装，配置文件加载将回退到默认配置")


@dataclass
class LifecycleConfig:
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    gates: Dict[str, List[dict]] = field(default_factory=dict)
    custom_skills: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)


def _default_config() -> LifecycleConfig:
    """从 handlers.py 的 LIFECYCLE 生成默认配置。"""
    return LifecycleConfig(stages=dict(LIFECYCLE))


def merge_lifecycle(default: dict, custom: dict) -> dict:
    """合并生命周期定义。

    - 阶段定义：自定义阶段覆盖默认，默认阶段补充
    - 技能列表：自定义 flow 完全替换默认 flow（如果指定），否则保留默认
    """
    merged = dict(default)

    for stage_key, stage_val in custom.items():
        if stage_key not in merged:
            merged[stage_key] = stage_val
            continue

        merged_stage = dict(merged[stage_key])

        for field_key, field_val in stage_val.items():
            if field_key == "flow":
                merged_stage["flow"] = field_val
            else:
                merged_stage[field_key] = field_val

        merged[stage_key] = merged_stage

    return merged


def load_config() -> LifecycleConfig:
    """加载配置。

    尝试读取 lifecycle.yaml，不存在则返回默认配置，
    存在则解析 YAML 并与默认配置合并（自定义优先，默认补充）。
    """
    default = _default_config()

    if not CONFIG_PATH.exists():
        logger.debug("配置文件不存在，使用默认配置: %s", CONFIG_PATH)
        return default

    if not _YAML_AVAILABLE:
        logger.warning("PyYAML 不可用，回退到默认配置")
        return default

    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("读取配置文件失败: %s — %s", CONFIG_PATH, e)
        return default

    if not isinstance(raw, dict):
        logger.warning("配置文件格式不正确，回退到默认配置")
        return default

    stages = raw.get("stages", {})
    gates = raw.get("gates", {})
    custom_skills = raw.get("custom_skills", {})

    if stages:
        merged_stages = merge_lifecycle(default.stages, stages)
    else:
        merged_stages = default.stages

    parsed_custom_skills: Dict[str, List[Tuple[str, str]]] = {}
    for stage_name, skill_list in custom_skills.items():
        if isinstance(skill_list, list):
            parsed_custom_skills[stage_name] = [
                (item[0], item[1]) if isinstance(item, (list, tuple)) and len(item) >= 2
                else (str(item), "")
                for item in skill_list
            ]

    return LifecycleConfig(
        stages=merged_stages,
        gates=gates,
        custom_skills=parsed_custom_skills,
    )
