"""
版本快照管理器 (SnapshotManager)

借鉴 Skill-insight 的快照版本管理设计：
  - 主版本：static/dynamic 批量优化触发（v1→v2）
  - 小版本：feedback 单用户反馈触发（v1.1）
  - Accept 后才晋升为主版本
  - 每次快附带 meta.json 记录来源、模式、原因

输出目录：
  <skill_dir>/snapshots/
  ├── v1/
  │   ├── SKILL.md          # 主版本快照
  │   └── meta.json         # 元数据
  ├── v1.1/
  │   ├── SKILL.md          # 小版本快照
  │   └── meta.json
  └── v2/
      ├── SKILL.md
      └── meta.json
"""

import json
import logging
import shutil
import difflib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class SnapshotMeta:
    """快照元数据。"""
    version: str                     # "v1", "v1.1", "v2"
    created_at: str                  # ISO 格式时间戳
    trigger: str                     # "auto" | "user"
    mode: str                        # "static" | "dynamic" | "feedback" | "hybrid"
    reason: str                      # 触发原因描述
    baseline_score: float = 0.0      # 优化前分数
    evolved_score: float = 0.0       # 优化后分数
    improvement: float = 0.0         # 改进幅度
    constraint_passed: bool = True   # 门控是否通过
    iterations: int = 0              # 优化迭代次数
    skill_name: str = ""             # 技能名称
    parent_version: str = ""         # 父版本（用于回退链）
    accepted: bool = False           # 是否已接受（小版本→主版本）


@dataclass
class DiffResult:
    """Diff 比较结果。"""
    old_version: str
    new_version: str
    unified_diff: str                # unified diff 文本
    added_lines: int = 0
    removed_lines: int = 0
    changed_sections: list[str] = field(default_factory=list)


class SnapshotManager:
    """
    版本快照管理器。

    核心功能：
      1. save_snapshot() — 保存当前版本快照
      2. diff_versions() — 比较两个版本的差异
      3. accept_version() — 接受小版本，晋升为主版本
      4. revert_to() — 回退到指定版本
      5. list_versions() — 列出所有版本
      6. get_latest() — 获取最新版本
    """

    SNAPSHOTS_DIR = "snapshots"

    def __init__(self, skill_dir: str | Path):
        self._skill_dir = Path(skill_dir)
        self._snapshots_dir = self._skill_dir / self.SNAPSHOTS_DIR
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

    # ── 保存快照 ──────────────────────────────────────────

    def save_snapshot(
        self,
        skill_text: str,
        version: str,
        trigger: str = "auto",
        mode: str = "static",
        reason: str = "",
        baseline_score: float = 0.0,
        evolved_score: float = 0.0,
        improvement: float = 0.0,
        constraint_passed: bool = True,
        iterations: int = 0,
        skill_name: str = "",
    ) -> Path:
        """
        保存一个版本快照。

        Args:
            skill_text: SKILL.md 内容
            version: 版本号（"v1", "v1.1", "v2"）
            trigger: "auto"（系统自动）或 "user"（用户触发）
            mode: "static" | "dynamic" | "feedback" | "hybrid"
            reason: 触发原因
            baseline_score: 优化前分数
            evolved_score: 优化后分数
            improvement: 改进幅度
            constraint_passed: 门控是否通过
            iterations: 优化迭代次数
            skill_name: 技能名称

        Returns:
            快照目录路径
        """
        snapshot_dir = self._snapshots_dir / version
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # 保存 SKILL.md
        skill_path = snapshot_dir / "SKILL.md"
        skill_path.write_text(skill_text, encoding="utf-8")

        # 确定父版本
        parent_version = self._get_parent_version(version)

        # 保存 meta.json
        meta = SnapshotMeta(
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            trigger=trigger,
            mode=mode,
            reason=reason,
            baseline_score=baseline_score,
            evolved_score=evolved_score,
            improvement=improvement,
            constraint_passed=constraint_passed,
            iterations=iterations,
            skill_name=skill_name,
            parent_version=parent_version,
            accepted=(trigger == "user"),  # 用户触发的默认已接受
        )

        meta_path = snapshot_dir / "meta.json"
        meta_path.write_text(
            json.dumps(asdict(meta), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            f"[snapshot] saved {version} for {skill_name or 'unknown'} "
            f"(trigger={trigger}, mode={mode}, improvement={improvement:+.3f})"
        )
        return snapshot_dir

    # ── Diff 比较 ─────────────────────────────────────────

    def diff_versions(self, old_version: str, new_version: str) -> DiffResult:
        """
        比较两个版本的差异。

        Args:
            old_version: 旧版本号
            new_version: 新版本号

        Returns:
            DiffResult 对象
        """
        old_path = self._snapshots_dir / old_version / "SKILL.md"
        new_path = self._snapshots_dir / new_version / "SKILL.md"

        if not old_path.exists():
            raise FileNotFoundError(f"Version {old_version} not found: {old_path}")
        if not new_path.exists():
            raise FileNotFoundError(f"Version {new_version} not found: {new_path}")

        old_lines = old_path.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = new_path.read_text(encoding="utf-8").splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"{old_version}/SKILL.md",
            tofile=f"{new_version}/SKILL.md",
            lineterm="",
        ))

        added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

        # 识别变更的章节
        changed_sections = self._extract_changed_sections(diff)

        return DiffResult(
            old_version=old_version,
            new_version=new_version,
            unified_diff="\n".join(diff),
            added_lines=added,
            removed_lines=removed,
            changed_sections=changed_sections,
        )

    def diff_current(self, version: str) -> DiffResult:
        """
        比较当前 SKILL.md 与指定版本的差异。

        Args:
            version: 要比较的版本号

        Returns:
            DiffResult 对象
        """
        current_path = self._skill_dir / "SKILL.md"
        if not current_path.exists():
            raise FileNotFoundError(f"Current SKILL.md not found: {current_path}")

        snapshot_path = self._snapshots_dir / version / "SKILL.md"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Version {version} not found: {snapshot_path}")

        old_lines = snapshot_path.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = current_path.read_text(encoding="utf-8").splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"{version}/SKILL.md",
            tofile="current/SKILL.md",
            lineterm="",
        ))

        added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
        changed_sections = self._extract_changed_sections(diff)

        return DiffResult(
            old_version=version,
            new_version="current",
            unified_diff="\n".join(diff),
            added_lines=added,
            removed_lines=removed,
            changed_sections=changed_sections,
        )

    # ── 版本管理 ─────────────────────────────────────────

    def accept_version(self, version: str) -> bool:
        """
        接受一个版本（标记为已接受）。

        对于小版本（如 v1.1），接受后会更新 meta.json 的 accepted 字段。
        对于主版本，直接标记已接受。

        Args:
            version: 版本号

        Returns:
            是否成功
        """
        meta = self._load_meta(version)
        if not meta:
            logger.warning(f"[snapshot] version {version} not found")
            return False

        meta.accepted = True
        self._save_meta(version, meta)

        logger.info(f"[snapshot] accepted {version}")
        return True

    def revert_to(self, target_version: str) -> bool:
        """
        回退到指定版本。

        会先保存当前版本为快照（作为回退前的备份），然后恢复目标版本。

        Args:
            target_version: 目标版本号

        Returns:
            是否成功
        """
        target_path = self._snapshots_dir / target_version / "SKILL.md"
        if not target_path.exists():
            logger.error(f"[snapshot] target version {target_version} not found")
            return False

        # 保存当前版本作为备份
        current_path = self._skill_dir / "SKILL.md"
        if current_path.exists():
            backup_version = self._generate_revert_backup_version()
            self.save_snapshot(
                skill_text=current_path.read_text(encoding="utf-8"),
                version=backup_version,
                trigger="auto",
                mode="revert",
                reason=f"Backup before reverting to {target_version}",
            )

        # 恢复目标版本
        target_text = target_path.read_text(encoding="utf-8")
        current_path.write_text(target_text, encoding="utf-8")

        logger.info(f"[snapshot] reverted to {target_version}")
        return True

    def list_versions(self) -> list[dict]:
        """
        列出所有版本及其元数据。

        Returns:
            版本列表，每个元素包含 version, meta 信息
        """
        versions = []
        for snapshot_dir in sorted(self._snapshots_dir.iterdir()):
            if not snapshot_dir.is_dir():
                continue
            meta = self._load_meta(snapshot_dir.name)
            if meta:
                versions.append(asdict(meta))
            else:
                versions.append({
                    "version": snapshot_dir.name,
                    "created_at": "unknown",
                    "trigger": "unknown",
                    "mode": "unknown",
                    "reason": "",
                    "accepted": False,
                })
        return versions

    def get_latest(self) -> Optional[str]:
        """
        获取最新版本号。

        Returns:
            最新版本号，无版本时返回 None
        """
        versions = self.list_versions()
        if not versions:
            return None
        return versions[-1]["version"]

    def get_next_version(self, mode: str = "static") -> str:
        """
        生成下一个版本号。

        主版本（static/dynamic/hybrid）：v1 → v2 → v3
        小版本（feedback）：v1 → v1.1 → v1.2

        Args:
            mode: 优化模式

        Returns:
            下一个版本号
        """
        versions = self.list_versions()
        if not versions:
            return "v1"

        # 找到最新的主版本号
        latest_major = 0
        latest_minor = 0
        for v in versions:
            ver = v["version"]
            if ver.startswith("v"):
                parts = ver[1:].split(".")
                if len(parts) == 1:
                    major = int(parts[0])
                    if major > latest_major:
                        latest_major = major
                        latest_minor = 0
                elif len(parts) == 2:
                    major, minor = int(parts[0]), int(parts[1])
                    if major > latest_major or (major == latest_major and minor > latest_minor):
                        latest_major = major
                        latest_minor = minor

        # 根据模式生成版本号
        if mode == "feedback":
            # 小版本递增
            return f"v{latest_major}.{latest_minor + 1}"
        else:
            # 主版本递增
            return f"v{latest_major + 1}"

    # ── 内部方法 ─────────────────────────────────────────

    def _get_parent_version(self, version: str) -> str:
        """获取父版本号。"""
        if "." in version:
            # 小版本：父版本是主版本
            return version.split(".")[0]
        else:
            # 主版本：父版本是上一个主版本
            major = int(version[1:])
            if major > 1:
                return f"v{major - 1}"
            return ""

    def _load_meta(self, version: str) -> Optional[SnapshotMeta]:
        """加载版本元数据。"""
        meta_path = self._snapshots_dir / version / "meta.json"
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            return SnapshotMeta(**data)
        except Exception as e:
            logger.warning(f"[snapshot] failed to load meta for {version}: {e}")
            return None

    def _save_meta(self, version: str, meta: SnapshotMeta) -> None:
        """保存版本元数据。"""
        meta_path = self._snapshots_dir / version / "meta.json"
        meta_path.write_text(
            json.dumps(asdict(meta), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _extract_changed_sections(self, diff_lines: list[str]) -> list[str]:
        """从 diff 中提取变更的章节标题。"""
        sections = []
        for line in diff_lines:
            if line.startswith("+") and not line.startswith("+++"):
                # 检查是否是章节标题
                stripped = line[1:].strip()
                if stripped.startswith("#"):
                    sections.append(stripped)
        return list(set(sections))

    def _generate_revert_backup_version(self) -> str:
        """生成回退备份的版本号。"""
        versions = self.list_versions()
        revert_count = sum(
            1 for v in versions
            if v.get("mode") == "revert"
        )
        return f"v0.revert{revert_count + 1}"


def format_diff_report(diff: DiffResult, max_lines: int = 50) -> str:
    """
    格式化 diff 报告为可读文本。

    Args:
        diff: DiffResult 对象
        max_lines: 最大显示行数

    Returns:
        格式化的 diff 报告
    """
    lines = [
        f"=== Diff: {diff.old_version} → {diff.new_version} ===",
        f"Added: +{diff.added_lines} lines, Removed: -{diff.removed_lines} lines",
    ]

    if diff.changed_sections:
        lines.append(f"Changed sections: {', '.join(diff.changed_sections)}")

    lines.append("")
    lines.append("--- Unified Diff ---")

    diff_lines = diff.unified_diff.splitlines()
    if len(diff_lines) > max_lines:
        lines.extend(diff_lines[:max_lines])
        lines.append(f"... ({len(diff_lines) - max_lines} more lines)")
    else:
        lines.extend(diff_lines)

    return "\n".join(lines)
