"""
失败轨迹结构化记录器 (FailureTracker)

灵感来源: Self-Harness 论文的 "Weakness Mining" 阶段
核心思想: 不是随机优化，而是从 agent 真实失败中提取模式，驱动有针对性的改进

失败模式分类 (按 Self-Harness 论文 + Hermes 实际):
  - tool_timeout:     工具调用超时
  - tool_error:       工具返回错误
  - understanding:    任务理解偏差（返工、多次澄清）
  - missing_artifact: 忘记创建/写入产物
  - tool_loop:        重复调用同一失败工具
  - env_loss:         跨 shell 环境状态丢失
  - premature_done:   未验证就声称完成
  - model_degrade:    模型降级后质量下降
  - context_overflow: 上下文过长丢失早期信息
  - other:            未分类

数据流:
  omni_record_action → FailureTracker.record() → 聚类分析 → 驱动 harness 修改
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FailurePattern(str, Enum):
    """失败模式枚举"""
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_ERROR = "tool_error"
    UNDERSTANDING = "understanding"
    MISSING_ARTIFACT = "missing_artifact"
    TOOL_LOOP = "tool_loop"
    ENV_LOSS = "env_loss"
    PREMATURE_DONE = "premature_done"
    MODEL_DEGRADE = "model_degrade"
    CONTEXT_OVERFLOW = "context_overflow"
    OTHER = "other"


# 自动分类规则: (关键词/模式 → FailurePattern)
_AUTO_CLASSIFY_RULES = [
    # tool_timeout
    (["timeout", "timed out", "超时", "deadline exceeded"], FailurePattern.TOOL_TIMEOUT),
    # tool_error
    (["error", "failed", "Traceback", "Error code", "错误", "exit_code"], FailurePattern.TOOL_ERROR),
    # tool_loop
    (["same_tool_failure_warning", "repeated_exact_failure", "loop"], FailurePattern.TOOL_LOOP),
    # premature_done
    (["premature", "未验证", "incomplete", "not verified"], FailurePattern.PREMATURE_DONE),
    # env_loss
    (["not found", "No such file", "command not found", "环境变量"], FailurePattern.ENV_LOSS),
    # model_degrade
    (["fallback", "降级", "model switch", "provider unavailable"], FailurePattern.MODEL_DEGRADE),
    # context_overflow
    (["context length", "token limit", "too long", "truncated", "省略"], FailurePattern.CONTEXT_OVERFLOW),
]


@dataclass
class FailureRecord:
    """一条失败记录"""
    timestamp: float
    session_id: str
    skill_name: str
    pattern: str              # FailurePattern.value
    tool_name: str            # 涉及的工具
    error_summary: str        # 错误摘要 (前 200 字)
    context: str              # 上下文 (前 500 字)
    turn_index: int = 0       # 第几轮出错
    retry_count: int = 0      # 重试次数
    resolved: bool = False    # 是否最终解决
    harness_surface: str = "" # 影响的 harness 表面 (prompt/tool_desc/error_recovery/orchestration)


def auto_classify(error_text: str) -> FailurePattern:
    """根据错误文本自动分类失败模式"""
    text_lower = error_text.lower()
    for keywords, pattern in _AUTO_CLASSIFY_RULES:
        for kw in keywords:
            if kw.lower() in text_lower:
                return pattern
    return FailurePattern.OTHER


def infer_harness_surface(pattern: FailurePattern) -> str:
    """推断失败模式影响的 harness 表面"""
    mapping = {
        FailurePattern.TOOL_TIMEOUT: "error_recovery",
        FailurePattern.TOOL_ERROR: "error_recovery",
        FailurePattern.UNDERSTANDING: "prompt",
        FailurePattern.MISSING_ARTIFACT: "prompt",
        FailurePattern.TOOL_LOOP: "orchestration",
        FailurePattern.ENV_LOSS: "orchestration",
        FailurePattern.PREMATURE_DONE: "prompt",
        FailurePattern.MODEL_DEGRADE: "orchestration",
        FailurePattern.CONTEXT_OVERFLOW: "orchestration",
        FailurePattern.OTHER: "unknown",
    }
    return mapping.get(pattern, "unknown")


class FailureTracker:
    """
    失败轨迹记录器

    存储: ~/.hermes/self_evolution/failures.db
    """

    DB_PATH = Path.home() / ".hermes" / "self_evolution" / "failures.db"

    def __init__(self):
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self.DB_PATH)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    session_id TEXT DEFAULT '',
                    skill_name TEXT DEFAULT '',
                    pattern TEXT NOT NULL,
                    tool_name TEXT DEFAULT '',
                    error_summary TEXT DEFAULT '',
                    context TEXT DEFAULT '',
                    turn_index INTEGER DEFAULT 0,
                    retry_count INTEGER DEFAULT 0,
                    resolved INTEGER DEFAULT 0,
                    harness_surface TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_failures_pattern
                ON failures(pattern)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_failures_skill
                ON failures(skill_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_failures_time
                ON failures(timestamp)
            """)
            conn.commit()

    def record(
        self,
        error_text: str,
        session_id: str = "",
        skill_name: str = "",
        tool_name: str = "",
        context: str = "",
        turn_index: int = 0,
        retry_count: int = 0,
        resolved: bool = False,
        pattern: Optional[str] = None,
    ) -> FailureRecord:
        """记录一条失败"""
        # 自动分类
        if pattern:
            try:
                fp = FailurePattern(pattern)
            except ValueError:
                fp = auto_classify(error_text)
        else:
            fp = auto_classify(error_text)

        surface = infer_harness_surface(fp)

        record = FailureRecord(
            timestamp=time.time(),
            session_id=session_id,
            skill_name=skill_name,
            pattern=fp.value,
            tool_name=tool_name,
            error_summary=error_text[:200],
            context=context[:500],
            turn_index=turn_index,
            retry_count=retry_count,
            resolved=resolved,
            harness_surface=surface,
        )

        with sqlite3.connect(str(self.DB_PATH)) as conn:
            conn.execute("""
                INSERT INTO failures
                (timestamp, session_id, skill_name, pattern, tool_name,
                 error_summary, context, turn_index, retry_count, resolved, harness_surface)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp, record.session_id, record.skill_name,
                record.pattern, record.tool_name, record.error_summary,
                record.context, record.turn_index, record.retry_count,
                int(record.resolved), record.harness_surface,
            ))
            conn.commit()

        logger.info(f"[failure_tracker] recorded: {fp.value} tool={tool_name}")
        return record

    def get_weakness_summary(
        self,
        days: int = 30,
        min_count: int = 2,
    ) -> list[dict]:
        """
        弱点聚类分析 — Self-Harness 的 "Weakness Mining"

        按 (pattern, harness_surface) 聚类，返回高频失败模式
        """
        cutoff = time.time() - days * 86400

        with sqlite3.connect(str(self.DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    pattern,
                    harness_surface,
                    COUNT(*) as count,
                    COUNT(DISTINCT tool_name) as tool_count,
                    COUNT(DISTINCT skill_name) as skill_count,
                    SUM(CASE WHEN resolved THEN 1 ELSE 0 END) as resolved_count,
                    AVG(retry_count) as avg_retries,
                    MIN(timestamp) as first_seen,
                    MAX(timestamp) as last_seen,
                    GROUP_CONCAT(DISTINCT tool_name) as tools_involved
                FROM failures
                WHERE timestamp > ?
                GROUP BY pattern, harness_surface
                HAVING COUNT(*) >= ?
                ORDER BY count DESC
            """, (cutoff, min_count)).fetchall()

        return [dict(r) for r in rows]

    def get_pattern_details(
        self,
        pattern: str,
        days: int = 30,
        limit: int = 20,
    ) -> list[dict]:
        """获取某个失败模式的详细记录"""
        cutoff = time.time() - days * 86400

        with sqlite3.connect(str(self.DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT *
                FROM failures
                WHERE pattern = ? AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (pattern, cutoff, limit)).fetchall()

        return [dict(r) for r in rows]

    def get_skill_weaknesses(self, skill_name: str, days: int = 30) -> list[dict]:
        """获取某个 skill 的失败模式"""
        cutoff = time.time() - days * 86400

        with sqlite3.connect(str(self.DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT pattern, harness_surface, COUNT(*) as count,
                       GROUP_CONCAT(DISTINCT tool_name) as tools
                FROM failures
                WHERE skill_name = ? AND timestamp > ?
                GROUP BY pattern, harness_surface
                ORDER BY count DESC
            """, (skill_name, cutoff)).fetchall()

        return [dict(r) for r in rows]

    def get_total_count(self, days: int = 30) -> int:
        """获取总失败数"""
        cutoff = time.time() - days * 86400
        with sqlite3.connect(str(self.DB_PATH)) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM failures WHERE timestamp > ?", (cutoff,)
            ).fetchone()
            return row[0] if row else 0

    def generate_weakness_report(self, days: int = 30) -> str:
        """生成弱点报告 — 用于驱动 harness 修改"""
        summary = self.get_weakness_summary(days=days)
        total = self.get_total_count(days=days)

        if not summary:
            return f"最近 {days} 天无失败记录 (共 {total} 条)"

        parts = [
            f"## 失败轨迹分析报告 (最近 {days} 天)",
            f"总失败数: {total}",
            "",
            "| 失败模式 | 次数 | 影响表面 | 涉及工具 | 解决率 |",
            "|---------|------|---------|---------|--------|",
        ]

        for s in summary:
            resolved_rate = f"{s['resolved_count']/s['count']*100:.0f}%" if s['count'] > 0 else "0%"
            parts.append(
                f"| {s['pattern']:20} | {s['count']:4} | {s['harness_surface']:15} | "
                f"{s['tool_count']} 个 | {resolved_rate} |"
            )

        # 建议
        parts.append("\n### 改进建议")
        for s in summary[:3]:
            pattern = s['pattern']
            surface = s['harness_surface']
            parts.append(f"\n**{pattern}** (影响: {surface})")

            if surface == "prompt":
                parts.append("  → 在系统提示中增加针对此失败模式的指导")
            elif surface == "error_recovery":
                parts.append("  → 优化错误恢复策略，增加重试/降级逻辑")
            elif surface == "orchestration":
                parts.append("  → 调整编排逻辑，增加检查点或限制")
            elif surface == "tool_desc":
                parts.append("  → 改进工具描述，增加错误处理说明")

        return "\n".join(parts)
