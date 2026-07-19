"""dev-lifecycle 插件 — 项目上下文感知模块。

探测项目语言、框架、测试等特征，并根据项目类型调整技能推荐顺序。"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("plugins.dev-lifecycle")


@dataclass
class ProjectContext:
    """项目上下文信息。"""

    project_path: str
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    has_tests: bool = False
    project_type: str = "unknown"


# 语言标识文件映射
_LANG_MARKERS: dict[str, list[str]] = {
    "python": ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"],
    "node": ["package.json"],
    "go": ["go.mod"],
    "rust": ["Cargo.toml"],
}

# 框架探测规则: (框架名, 探测文件/模式, 是否递归搜索)
_FRAMEWORK_RULES: list[tuple[str, str, bool]] = [
    ("django", "settings.py", True),
    ("flask", "app.py", False),
    ("fastapi", "main.py", False),
    ("react", ".jsx", True),
    ("react", ".tsx", True),
    ("vue", ".vue", True),
    ("express", "server.js", False),
]

# 测试目录/文件标记
_TEST_MARKERS_DIRS: list[str] = ["tests", "test", "__tests__"]
_TEST_MARKERS_FILES: list[str] = ["pytest.ini", "jest.config.js", "jest.config.ts", "jest.config.mjs", "jest.config.cjs"]


class ProjectDetector:
    """项目特征探测器。"""

    def detect(self, project_path: str) -> ProjectContext:
        """探测项目特征，返回 ProjectContext。"""
        ctx = ProjectContext(project_path=project_path)

        root = Path(project_path)
        if not root.exists():
            logger.debug("项目路径不存在: %s", project_path)
            return ctx

        ctx.languages = self._detect_languages(root)
        ctx.project_type = self._resolve_project_type(ctx.languages)
        ctx.frameworks = self._detect_frameworks(root)
        ctx.has_tests = self._detect_tests(root)

        logger.info(
            "项目探测完成: path=%s, type=%s, languages=%s, frameworks=%s, has_tests=%s",
            project_path, ctx.project_type, ctx.languages, ctx.frameworks, ctx.has_tests,
        )
        return ctx

    def _detect_languages(self, root: Path) -> list[str]:
        """根据标识文件探测语言。"""
        languages: list[str] = []
        for lang, markers in _LANG_MARKERS.items():
            for marker in markers:
                if (root / marker).exists():
                    languages.append(lang)
                    break
        return languages

    def _resolve_project_type(self, languages: list[str]) -> str:
        """根据探测到的语言确定项目类型。"""
        priority = ["python", "node", "go", "rust"]
        for lang in priority:
            if lang in languages:
                return lang
        return "unknown"

    def _detect_frameworks(self, root: Path) -> list[str]:
        """探测项目使用的框架。"""
        frameworks: list[str] = []
        seen: set[str] = set()

        for fw_name, pattern, recursive in _FRAMEWORK_RULES:
            if fw_name in seen:
                continue
            found = self._exists_pattern(root, pattern, recursive)
            if found:
                frameworks.append(fw_name)
                seen.add(fw_name)

        return frameworks

    def _detect_tests(self, root: Path) -> bool:
        """探测项目是否包含测试。"""
        for dir_name in _TEST_MARKERS_DIRS:
            if (root / dir_name).is_dir():
                return True

        for file_name in _TEST_MARKERS_FILES:
            if (root / file_name).exists():
                return True

        # Go 测试文件: *_test.go
        try:
            for _ in root.glob("*_test.go"):
                return True
        except OSError:
            pass

        return False

    @staticmethod
    def _exists_pattern(root: Path, pattern: str, recursive: bool) -> bool:
        """检查路径下是否存在匹配的文件。"""
        try:
            if recursive:
                for _ in root.rglob(pattern):
                    return True
            else:
                if (root / pattern).exists():
                    return True
        except OSError:
            pass
        return False


class SkillRecommender:
    """根据项目上下文调整技能推荐顺序。"""

    def recommend(
        self,
        project_context: ProjectContext,
        stage: str,
        available_skills: List[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """根据项目类型和阶段调整技能推荐顺序。

        排序规则:
          - Python 项目 + build 阶段: python-debugpy 排到最前
          - Node 项目 + build 阶段: node-inspect-debugger 排到最前
          - Python 项目 + deliver 阶段: python-debugpy 排到最前
          - 其他情况保持原顺序
        """
        skills = list(available_skills)

        if project_context.project_type == "python" and stage in ("build", "deliver"):
            return self._promote_skill(skills, "python-debugpy")

        if project_context.project_type == "node" and stage == "build":
            return self._promote_skill(skills, "node-inspect-debugger")

        return skills

    @staticmethod
    def _promote_skill(
        skills: List[Tuple[str, str]], target_name: str
    ) -> List[Tuple[str, str]]:
        """将指定技能提升到列表最前。"""
        promoted: list[tuple[str, str]] = []
        rest: list[tuple[str, str]] = []

        for skill in skills:
            if skill[0] == target_name:
                promoted.append(skill)
            else:
                rest.append(skill)

        return promoted + rest
