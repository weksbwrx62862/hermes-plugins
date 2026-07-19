"""
反思存储 — 跨进化运行的失败经验积累 (Reflexion 模式)。

每次进化失败后生成结构化反思并持久化。
下次进化同一技能时，加载历史反思注入变异提示词。

数据结构:
  {
    "skill_name": {
      "reflections": [
        {
          "timestamp": 1717000000,
          "failure_type": "skill_defect",
          "weak_dimension": "clarity",       # AND门失败的维度
          "feedback_summary": "触发条件过宽",
          "mutation_strategy": "simplify",   # 使用的变异策略
          "score_before": 0.65,
          "score_after": 0.62,              # 变异后反而更差
          "lesson": "简化不应删除错误处理步骤"
        }
      ],
      "pattern_stats": {
        "clarity_failures": 3,
        "completeness_failures": 1,
        "most_effective_strategy": "structural"
      }
    }
  }
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_STORE_PATH = Path.home() / ".hermes" / "evolution_reflections.json"
_MAX_REFLECTIONS_PER_SKILL = 30  # 每个技能最多保留 N 条反思


@dataclass
class EvolutionReflection:
    """单条进化反思"""
    timestamp: float
    failure_type: str           # skill_defect | optimization | execution_lapse
    weak_dimension: str         # accuracy | clarity | completeness | efficiency | safety
    feedback_summary: str       # 反馈摘要 (<=200 chars)
    mutation_strategy: str      # 使用的变异策略
    score_before: float         # 变异前分数
    score_after: float          # 变异后分数
    lesson: str                 # 经验教训 (<=200 chars)


class ReflectionStore:
    """反思存储管理器"""

    def __init__(self, path: Path = _STORE_PATH):
        self._path = path
        self._data: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        """加载反思数据"""
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = raw
                logger.debug("Loaded reflections for %d skills", len(self._data))
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Failed to load reflections: %s", e)
            self._data = {}

    def _save(self):
        """持久化反思数据"""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except OSError as e:
            logger.debug("Failed to save reflections: %s", e)

    def add_reflection(self, skill_name: str, reflection: EvolutionReflection):
        """添加一条反思"""
        if skill_name not in self._data:
            self._data[skill_name] = {"reflections": [], "pattern_stats": {}}

        entry = self._data[skill_name]
        entry["reflections"].append(asdict(reflection))

        # 限制数量 (保留最新的)
        if len(entry["reflections"]) > _MAX_REFLECTIONS_PER_SKILL:
            entry["reflections"] = entry["reflections"][-_MAX_REFLECTIONS_PER_SKILL:]

        # 更新模式统计
        self._update_pattern_stats(skill_name)

        self._save()
        logger.info(
            "Added reflection for %s: %s/%s (score %.3f→%.3f)",
            skill_name, reflection.failure_type, reflection.weak_dimension,
            reflection.score_before, reflection.score_after,
        )

    def _update_pattern_stats(self, skill_name: str):
        """更新模式统计"""
        reflections = self._data[skill_name]["reflections"]
        if not reflections:
            return

        # 统计各维度失败次数
        dim_counter = Counter(r.get("weak_dimension", "unknown") for r in reflections)
        strategy_effective = Counter()

        for r in reflections:
            # 有效策略: 变异后分数提升了
            if r.get("score_after", 0) > r.get("score_before", 0):
                strategy_effective[r.get("mutation_strategy", "unknown")] += 1

        stats = {}
        for dim, count in dim_counter.items():
            stats[f"{dim}_failures"] = count

        if strategy_effective:
            stats["most_effective_strategy"] = strategy_effective.most_common(1)[0][0]
        else:
            stats["most_effective_strategy"] = "unknown"

        self._data[skill_name]["pattern_stats"] = stats

    def get_reflections(self, skill_name: str, limit: int = 10) -> List[Dict]:
        """获取某个技能的最近 N 条反思"""
        entry = self._data.get(skill_name, {})
        reflections = entry.get("reflections", [])
        return reflections[-limit:]

    def get_pattern_stats(self, skill_name: str) -> Dict:
        """获取某个技能的模式统计"""
        entry = self._data.get(skill_name, {})
        return entry.get("pattern_stats", {})

    def get_lessons_prompt(self, skill_name: str, limit: int = 5) -> str:
        """
        生成反思经验提示词片段，注入到变异提示词中。

        Returns: 格式化的反思文本，如果没有历史则返回空字符串。
        """
        reflections = self.get_reflections(skill_name, limit=limit)
        if not reflections:
            return ""

        stats = self.get_pattern_stats(skill_name)

        lines = ["## 历史进化经验（来自反思存储）\n"]

        # 模式统计
        if stats:
            dim_failures = {k: v for k, v in stats.items() if k.endswith("_failures")}
            if dim_failures:
                sorted_dims = sorted(dim_failures.items(), key=lambda x: x[1], reverse=True)
                top_dims = [f"{d.replace('_failures', '')}({c}次)" for d, c in sorted_dims[:3]]
                lines.append(f"历史薄弱维度: {', '.join(top_dims)}")

            best_strategy = stats.get("most_effective_strategy", "unknown")
            if best_strategy != "unknown":
                lines.append(f"最有效变异策略: {best_strategy}")
            lines.append("")

        # 最近反思
        lines.append("最近失败经验:")
        for i, r in enumerate(reflections[-limit:], 1):
            lesson = r.get("lesson", "")
            dim = r.get("weak_dimension", "?")
            strategy = r.get("mutation_strategy", "?")
            score_change = ""
            before = r.get("score_before", 0)
            after = r.get("score_after", 0)
            if after > before:
                score_change = f" (+{after - before:.3f}✓)"
            elif after < before:
                score_change = f" ({after - before:.3f}✗)"
            lines.append(f"  {i}. [{dim}] {lesson} (策略={strategy}{score_change})")

        return "\n".join(lines)

    def get_all_skills(self) -> List[str]:
        """列出所有有反思记录的技能"""
        return list(self._data.keys())

    def clear(self, skill_name: Optional[str] = None):
        """清除反思数据"""
        if skill_name:
            self._data.pop(skill_name, None)
        else:
            self._data.clear()
        self._save()


# 全局单例
_store: Optional[ReflectionStore] = None


def get_reflection_store() -> ReflectionStore:
    """获取反思存储单例"""
    global _store
    if _store is None:
        _store = ReflectionStore()
    return _store
