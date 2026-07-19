"""
Harness 版本化管理器 (HarnessVersioner)

灵感来源: Self-Harness 论文 — "把 harness 当成版本化资产"
核心思想: 每次进化前快照整个 harness 状态，可对比、可回滚、可审计

Harness 包含:
  1. 系统提示词 (system prompt sections)
  2. 工具描述 (tool descriptions in schemas)
  3. 错误恢复策略 (error recovery in handlers)
  4. 编排逻辑 (orchestration in plugins)
  5. Skill 文件 (SKILL.md)

版本化存储:
  ~/.hermes/self_evolution/harness_versions/
    ├── {version_id}/
    │   ├── manifest.json     — 元数据 (时间、触发原因、变更摘要)
    │   ├── system_prompt.md  — 系统提示快照
    │   ├── skills/           — 所有 skill 文件快照
    │   ├── tool_schemas/     — 工具 schema 快照
    │   └── diff.md           — 与上一版本的 diff
    └── current -> {latest_version_id}
"""

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HarnessSnapshot:
    """一次 harness 快照"""
    version_id: str
    timestamp: float
    trigger: str              # 触发原因 (pre_evolution, manual, scheduled)
    skill_name: str           # 被进化的 skill
    files: dict[str, str] = field(default_factory=dict)  # file_path → content_hash
    description: str = ""


@dataclass
class HarnessDiff:
    """两个版本之间的差异"""
    from_version: str
    to_version: str
    added_files: list[str] = field(default_factory=list)
    removed_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    content_diffs: dict[str, str] = field(default_factory=dict)  # file → diff_text


class HarnessVersioner:
    """
    Harness 版本化管理器

    操作:
      - snapshot(): 快照当前 harness 状态
      - diff(): 对比两个版本
      - rollback(): 回滚到指定版本
      - list_versions(): 列出所有版本
      - get_current(): 获取当前版本
    """

    VERSIONS_DIR = Path.home() / ".hermes" / "self_evolution" / "harness_versions"
    HERMES_DIR = Path.home() / ".hermes"

    def __init__(self):
        self.VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def snapshot(
        self,
        skill_name: str = "",
        trigger: str = "manual",
        description: str = "",
    ) -> HarnessSnapshot:
        """
        快照当前 harness 状态

        收集:
          1. ~/.hermes/skills/ 下所有 SKILL.md
          2. ~/.hermes/plugins/ 下的 plugin.yaml 和 schema 定义
          3. 系统提示相关配置
        """
        version_id = f"v_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"
        snapshot_dir = self.VERSIONS_DIR / version_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        files = {}

        # 1. 快照所有 SKILL.md
        skills_dir = snapshot_dir / "skills"
        skills_dir.mkdir(exist_ok=True)
        skills_source = self.HERMES_DIR / "skills"
        if skills_source.exists():
            for skill_file in skills_source.rglob("SKILL.md"):
                rel = skill_file.relative_to(skills_source)
                dest = skills_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(skill_file, dest)
                content_hash = hashlib.md5(skill_file.read_bytes()).hexdigest()
                files[f"skills/{rel}"] = content_hash

        # 2. 快照工具 schema（从 plugins 目录）
        schemas_dir = snapshot_dir / "tool_schemas"
        schemas_dir.mkdir(exist_ok=True)
        plugins_source = self.HERMES_DIR / "plugins"
        if plugins_source.exists():
            for schema_file in plugins_source.rglob("schemas.py"):
                rel = schema_file.relative_to(plugins_source)
                dest = schemas_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(schema_file, dest)
                content_hash = hashlib.md5(schema_file.read_bytes()).hexdigest()
                files[f"schemas/{rel}"] = content_hash

        # 3. 快照自进化相关配置
        config_file = self.HERMES_DIR / "config.yaml"
        if config_file.exists():
            dest = snapshot_dir / "config.yaml"
            shutil.copy2(config_file, dest)
            content_hash = hashlib.md5(config_file.read_bytes()).hexdigest()
            files["config.yaml"] = content_hash

        # 4. 写入 manifest
        manifest = {
            "version_id": version_id,
            "timestamp": time.time(),
            "trigger": trigger,
            "skill_name": skill_name,
            "description": description,
            "files": files,
            "file_count": len(files),
        }
        with open(snapshot_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        # 5. 更新 current 链接
        current_link = self.VERSIONS_DIR / "current"
        if current_link.is_symlink() or current_link.exists():
            current_link.unlink()
        current_link.symlink_to(snapshot_dir)

        snapshot = HarnessSnapshot(
            version_id=version_id,
            timestamp=time.time(),
            trigger=trigger,
            skill_name=skill_name,
            files=files,
            description=description,
        )

        logger.info(
            f"[harness_versioner] snapshot {version_id}: "
            f"{len(files)} files, trigger={trigger}"
        )
        return snapshot

    def list_versions(self, limit: int = 20) -> list[dict]:
        """列出所有版本"""
        versions = []
        for d in sorted(self.VERSIONS_DIR.iterdir(), reverse=True):
            if d.is_dir() and d.name.startswith("v_"):
                manifest_file = d / "manifest.json"
                if manifest_file.exists():
                    with open(manifest_file) as f:
                        manifest = json.load(f)
                    versions.append(manifest)
                    if len(versions) >= limit:
                        break
        return versions

    def get_current(self) -> Optional[dict]:
        """获取当前版本"""
        current_link = self.VERSIONS_DIR / "current"
        if current_link.is_symlink():
            manifest_file = current_link / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file) as f:
                    return json.load(f)
        return None

    def get_version(self, version_id: str) -> Optional[dict]:
        """获取指定版本"""
        manifest_file = self.VERSIONS_DIR / version_id / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                return json.load(f)
        return None

    def diff(self, from_version: str, to_version: str) -> HarnessDiff:
        """对比两个版本的差异"""
        from_dir = self.VERSIONS_DIR / from_version
        to_dir = self.VERSIONS_DIR / to_version

        from_manifest = self._load_manifest(from_dir)
        to_manifest = self._load_manifest(to_dir)

        from_files = set(from_manifest.get("files", {}).keys()) if from_manifest else set()
        to_files = set(to_manifest.get("files", {}).keys()) if to_manifest else set()

        added = sorted(to_files - from_files)
        removed = sorted(from_files - to_files)
        common = from_files & to_files

        modified = []
        content_diffs = {}
        from_files_dict = (from_manifest or {}).get("files", {})
        to_files_dict = (to_manifest or {}).get("files", {})
        for f in common:
            from_hash = from_files_dict[f]
            to_hash = to_files_dict[f]
            if from_hash != to_hash:
                modified.append(f)
                # 生成简单的 diff 描述
                content_diffs[f] = f"hash: {from_hash[:8]} → {to_hash[:8]}"

        return HarnessDiff(
            from_version=from_version,
            to_version=to_version,
            added_files=added,
            removed_files=removed,
            modified_files=modified,
            content_diffs=content_diffs,
        )

    def rollback(self, version_id: str) -> bool:
        """回滚到指定版本（仅回滚 skill 文件）"""
        version_dir = self.VERSIONS_DIR / version_id
        if not version_dir.exists():
            logger.error(f"[harness_versioner] version {version_id} not found")
            return False

        skills_dir = version_dir / "skills"
        if not skills_dir.exists():
            logger.error(f"[harness_versioner] no skills in version {version_id}")
            return False

        target_skills = self.HERMES_DIR / "skills"
        restored = 0
        for skill_file in skills_dir.rglob("SKILL.md"):
            rel = skill_file.relative_to(skills_dir)
            dest = target_skills / rel
            if dest.exists():
                # 备份当前版本
                backup = dest.with_suffix(f".bak.{int(time.time())}")
                shutil.copy2(dest, backup)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(skill_file, dest)
            restored += 1

        logger.info(f"[harness_versioner] rolled back to {version_id}, restored {restored} skills")
        return True

    def format_version_log(self, limit: int = 10) -> str:
        """格式化版本日志"""
        versions = self.list_versions(limit)
        if not versions:
            return "无 harness 版本记录"

        parts = ["## Harness 版本日志\n"]
        for v in versions:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(v["timestamp"]))
            parts.append(
                f"- **{v['version_id']}** ({ts}) "
                f"| {v['trigger']} | {v.get('skill_name', '-')} "
                f"| {v.get('file_count', 0)} files "
                f"| {v.get('description', '')}"
            )

        return "\n".join(parts)

    def _load_manifest(self, version_dir: Path) -> Optional[dict]:
        manifest_file = version_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                return json.load(f)
        return None
