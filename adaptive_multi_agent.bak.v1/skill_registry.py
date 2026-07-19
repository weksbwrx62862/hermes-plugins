"""AMA Agent Skill Registry — 借鉴 MULTICA 的技能注册与匹配。

子代理执行后自动提取技能标签、记录成功率/耗时，
下次类似任务优先匹配历史表现最好的子代理配置。

设计哲学（借鉴 MULTICA）：
- Agent 是可提升的——每次执行都是学习机会
- 技能标签自动提取，无需手动标注
- 技能图谱：发现技能之间的关联关系
- 冷启动友好：无历史数据时回退默认配置
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .subagent import AgentMode, SubagentConfig, SubagentResult, SubagentStatus


logger = logging.getLogger("ama.skills")

# ── 技能标签提取器 ──────────────────────────────────────

# 按领域分类的技能关键词
SKILL_LIBRARY: Dict[str, Dict[str, List[str]]] = {
    "programming": {
        "label": "编程",
        "keywords": [
            "代码", "python", "javascript", "rust", "go", "typescript", "java",
            "react", "api", "函数", "class", "算法", "数据库", "sql",
            "code", "function", "implement", "debug", "refactor",
            "测试", "test", "单元测试", "集成测试",
        ],
    },
    "data_analysis": {
        "label": "数据分析",
        "keywords": [
            "分析", "数据", "统计", "可视化", "图表", "趋势", "指标",
            "analyze", "data", "statistics", "visualize", "chart", "trend", "metric",
            "pandas", "numpy", "matplotlib", "excel", "csv",
        ],
    },
    "research": {
        "label": "研究调研",
        "keywords": [
            "研究", "调研", "搜索", "对比", "评测", "综述", "论文",
            "research", "investigate", "search", "compare", "review", "survey",
            "文献", "资料", "背景",
        ],
    },
    "writing": {
        "label": "写作",
        "keywords": [
            "写", "文章", "文档", "报告", "方案", "文案", "说明",
            "write", "article", "document", "report", "proposal", "copy",
            "创作", "翻译", "总结", "摘要",
        ],
    },
    "debugging": {
        "label": "调试",
        "keywords": [
            "调试", "修复", "bug", "错误", "异常", "崩溃", "问题",
            "debug", "fix", "error", "exception", "crash", "troubleshoot",
            "排查", "定位",
        ],
    },
    "architecture": {
        "label": "架构设计",
        "keywords": [
            "架构", "设计", "系统", "框架", "模块", "重构", "优化",
            "architecture", "design", "system", "framework", "refactor", "optimize",
            "技术选型", "方案设计",
        ],
    },
    "devops": {
        "label": "运维部署",
        "keywords": [
            "部署", "发布", "ci", "cd", "docker", "kubernetes", "监控",
            "deploy", "release", "pipeline", "monitor", "config",
            "devops", "infrastructure",
        ],
    },
    "security": {
        "label": "安全",
        "keywords": [
            "安全", "漏洞", "加密", "认证", "授权", "审计",
            "security", "vulnerability", "encrypt", "auth", "permission",
        ],
    },
}


@dataclass
class SkillTag:
    """一个技能标签。"""
    domain: str            # 领域名（如 "programming"）
    label: str             # 中文标签（如 "编程"）
    confidence: float      # 置信度 0-1
    extracted_from: str    # 来源："task" / "result" / "error"


@dataclass
class SkillRecord:
    """一条技能执行记录。"""
    skill_domain: str
    skill_label: str
    mode: str              # 使用的模式
    subagent_config_name: str  # 子代理配置名
    success: bool
    elapsed_seconds: float
    token_usage: int
    task_summary: str      # 任务摘要（前100字符）
    timestamp: float = field(default_factory=time.time)
    trace_id: str = ""


@dataclass
class SkillStats:
    """技能统计。"""
    skill_domain: str
    total_trials: int = 0
    successes: int = 0
    total_time: float = 0.0
    total_tokens: int = 0
    mode_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.total_trials, 1)

    @property
    def avg_time(self) -> float:
        return self.total_time / max(self.total_trials, 1)

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / max(self.total_trials, 1)

    def best_mode(self) -> Optional[str]:
        """返回成功率最高的模式。"""
        if not self.mode_distribution:
            return None
        # 简化：选使用次数最多的模式
        return max(self.mode_distribution, key=self.mode_distribution.get)


class SkillRegistry:
    """子代理技能注册表。

    借鉴 MULTICA 的 Agent 技能提升机制：
    - 自动提取任务技能标签
    - 记录每次执行的成功率和性能
    - 为新任务推荐最佳配置
    """

    PERSIST_PATH = Path.home() / ".hermes" / "ama" / "skill_registry.json"

    def __init__(self):
        self._records: List[SkillRecord] = []
        self._stats: Dict[str, SkillStats] = {}
        self._lock = threading.Lock()
        self._load()

    # ── 技能标签提取 ──────────────────────────────────

    @staticmethod
    def extract_tags(text: str, max_tags: int = 5) -> List[SkillTag]:
        """从文本中提取技能标签。

        使用关键词匹配，按置信度 = 命中关键词数 / 领域关键词总数 计算。
        """
        if not text:
            return []

        text_lower = text.lower()
        tags = []

        for domain, info in SKILL_LIBRARY.items():
            hit_count = sum(
                1 for kw in info["keywords"]
                if kw.lower() in text_lower
            )
            if hit_count > 0:
                confidence = min(hit_count / max(len(info["keywords"]), 1), 1.0)
                tags.append(SkillTag(
                    domain=domain,
                    label=info["label"],
                    confidence=confidence,
                    extracted_from="task",
                ))

        # 按置信度排序，取 top N
        tags.sort(key=lambda t: t.confidence, reverse=True)
        return tags[:max_tags]

    # ── 记录执行结果 ──────────────────────────────────

    def record(
        self,
        goal: str,
        mode: AgentMode,
        result: SubagentResult,
        subagent_config_name: str = "default",
    ):
        """记录一次子代理执行的技能信息。"""
        tags = self.extract_tags(goal)

        with self._lock:
            for tag in tags:
                record = SkillRecord(
                    skill_domain=tag.domain,
                    skill_label=tag.label,
                    mode=mode.value,
                    subagent_config_name=subagent_config_name,
                    success=result.status == SubagentStatus.COMPLETED,
                    elapsed_seconds=result.elapsed_seconds,
                    token_usage=result.token_usage,
                    task_summary=goal[:100],
                    trace_id=result.trace_id,
                )
                self._records.append(record)

                # 更新统计
                stats = self._stats.get(tag.domain)
                if not stats:
                    stats = SkillStats(skill_domain=tag.domain)
                    self._stats[tag.domain] = stats

                stats.total_trials += 1
                if record.success:
                    stats.successes += 1
                stats.total_time += result.elapsed_seconds
                stats.total_tokens += result.token_usage
                stats.mode_distribution[mode.value] += 1
                stats.last_used = record.timestamp

            # 限制记录数
            if len(self._records) > 1000:
                self._records = self._records[-500:]

            self._save()

    # ── 查询 ──────────────────────────────────────────

    def get_stats(self, domain: Optional[str] = None) -> Dict[str, SkillStats]:
        """获取技能统计。"""
        with self._lock:
            if domain:
                return {domain: self._stats[domain]} if domain in self._stats else {}
            return dict(self._stats)

    def recommend_config(
        self, task: str, task_type: str
    ) -> Optional[Dict[str, Any]]:
        """为任务推荐最佳子代理配置和模式。

        Returns:
            {"mode": "orchestrator_subagent", "config_name": "default", "confidence": 0.85}
        """
        tags = self.extract_tags(task)

        if not tags:
            return None

        with self._lock:
            best_config = None
            best_score = 0.0

            for tag in tags:
                stats = self._stats.get(tag.domain)
                if not stats or stats.total_trials < 2:
                    continue

                # 综合评分 = 成功率 * 置信度 * log(试验次数)
                score = (
                    stats.success_rate
                    * tag.confidence
                    * min(stats.total_trials / 10.0, 1.0)
                )

                if score > best_score:
                    best_mode = stats.best_mode()
                    best_config = {
                        "mode": best_mode or task_type,
                        "skill_domain": tag.domain,
                        "skill_label": tag.label,
                        "confidence": round(score, 3),
                        "trials": stats.total_trials,
                        "success_rate": round(stats.success_rate, 3),
                        "avg_time": round(stats.avg_time, 1),
                    }
                    best_score = score

        return best_config

    def skill_summary(self) -> Dict:
        """技能全景摘要。"""
        with self._lock:
            domains = []
            for domain, stats in sorted(
                self._stats.items(),
                key=lambda x: x[1].total_trials,
                reverse=True,
            ):
                domains.append({
                    "domain": domain,
                    "label": SKILL_LIBRARY.get(domain, {}).get("label", domain),
                    "trials": stats.total_trials,
                    "successes": stats.successes,
                    "success_rate": round(stats.success_rate, 3),
                    "avg_time": round(stats.avg_time, 1),
                    "avg_tokens": round(stats.avg_tokens, 0),
                    "best_mode": stats.best_mode(),
                    "last_used": stats.last_used,
                })

            return {
                "total_records": len(self._records),
                "total_domains": len(self._stats),
                "domains": domains,
            }

    # ── 持久化 ────────────────────────────────────────

    def _save(self):
        """保存到 JSON 文件。"""
        try:
            self.PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "records": [
                    {
                        "skill_domain": r.skill_domain,
                        "skill_label": r.skill_label,
                        "mode": r.mode,
                        "success": r.success,
                        "elapsed_seconds": r.elapsed_seconds,
                        "token_usage": r.token_usage,
                        "task_summary": r.task_summary,
                        "timestamp": r.timestamp,
                        "trace_id": r.trace_id,
                    }
                    for r in self._records[-200:]  # 只保留最近 200 条
                ],
                "stats": {
                    domain: {
                        "total_trials": s.total_trials,
                        "successes": s.successes,
                        "total_time": s.total_time,
                        "total_tokens": s.total_tokens,
                        "mode_distribution": dict(s.mode_distribution),
                        "last_used": s.last_used,
                    }
                    for domain, s in self._stats.items()
                },
            }

            with open(self.PERSIST_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.debug("[skill_registry] 保存失败: %s", e)

    def _load(self):
        """从 JSON 文件加载。"""
        if not self.PERSIST_PATH.exists():
            return

        try:
            with open(self.PERSIST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            for r in data.get("records", []):
                self._records.append(SkillRecord(
                    skill_domain=r["skill_domain"],
                    skill_label=r["skill_label"],
                    mode=r["mode"],
                    success=r["success"],
                    elapsed_seconds=r["elapsed_seconds"],
                    token_usage=r["token_usage"],
                    task_summary=r.get("task_summary", ""),
                    timestamp=r.get("timestamp", 0),
                    trace_id=r.get("trace_id", ""),
                ))

            for domain, s in data.get("stats", {}).items():
                self._stats[domain] = SkillStats(
                    skill_domain=domain,
                    total_trials=s["total_trials"],
                    successes=s["successes"],
                    total_time=s["total_time"],
                    total_tokens=s["total_tokens"],
                    mode_distribution=defaultdict(int, s.get("mode_distribution", {})),
                    last_used=s.get("last_used", 0),
                )

            logger.info(
                "[skill_registry] 已加载 %d 条记录, %d 个技能域",
                len(self._records), len(self._stats),
            )
        except Exception as e:
            logger.warning("[skill_registry] 加载失败: %s", e)

    def clear(self):
        """清空注册表。"""
        with self._lock:
            self._records.clear()
            self._stats.clear()
            if self.PERSIST_PATH.exists():
                self.PERSIST_PATH.unlink()


# 全局单例
_skill_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    global _skill_registry
    if _skill_registry is None:
        _skill_registry = SkillRegistry()
    return _skill_registry
