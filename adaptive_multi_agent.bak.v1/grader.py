"""Agent 评估评分器 — LLM-as-Judge 自动评分

参考 Anthropic Agent Eval 设计：
- 使用 LLM 对执行轨迹进行多维度评分
- 支持不同评分维度（完整性、正确性、效率）
- 为持续改进提供反馈
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 评分提示模板
GRADING_PROMPT = """你是一个 Agent 执行评估专家。请对以下 Agent 执行轨迹进行多维度评分。

## 评分维度（每维度 1-5 分）

| 维度 | 1分 | 3分 | 5分 |
|------|-----|-----|-----|
| completeness | 严重遗漏，大部分未完成 | 基本完成，有小部分遗漏 | 完全覆盖所有要求 |
| correctness | 有明显错误 | 基本正确，有小瑕疵 | 完全正确，无错误 |
| efficiency | 浪费大量资源/步骤 | 效率一般 | 高效执行，步骤精简 |
| tool_usage | 工具使用混乱 | 工具使用基本合理 | 工具选择精准，使用恰当 |
| error_handling | 完全没有错误处理 | 有基本错误处理 | 完善的错误处理和恢复机制 |

## 评估任务

任务描述：{task}

使用的执行模式：{mode}
复杂度评分：{complexity_score}

## 执行轨迹

{trajectory_summary}

## 最终结果

{final_result}

## 输出格式（严格 JSON）

```json
{{
    "completeness": 3,
    "correctness": 4,
    "efficiency": 3,
    "tool_usage": 4,
    "error_handling": 2,
    "overall_score": 3.2,
    "feedback": "详细的反馈说明",
    "strengths": ["优点1", "优点2"],
    "improvements": ["改进建议1", "改进建议2"]
}}
```

请严格按照上述 JSON 格式输出，不要添加其他内容。"""


class AgentGrader:
    """Agent 执行评估器

    使用 LLM-as-Judge 方法对执行轨迹进行多维度评分。
    """

    def __init__(self):
        self._logger = logging.getLogger("ama.grader")

    def grade(
        self,
        task: str,
        trajectory: Dict,
        ctx=None,
    ) -> Dict[str, Any]:
        """对执行轨迹进行评分

        Args:
            task: 原始任务描述
            trajectory: 轨迹记录（来自 TrajectoryRecorder）
            ctx: 插件上下文（用于调用 LLM）

        Returns:
            评分结果字典
        """
        # 构建轨迹摘要
        trajectory_summary = self._build_trajectory_summary(trajectory)
        final_result = trajectory.get("final_result", "无")
        mode = trajectory.get("mode", "未知")
        complexity_score = trajectory.get("complexity_score", 0)

        # 构建评分提示
        prompt = GRADING_PROMPT.format(
            task=task,
            mode=mode,
            complexity_score=complexity_score,
            trajectory_summary=trajectory_summary,
            final_result=final_result[:2000] if final_result else "无",
        )

        # 调用 LLM 进行评分
        try:
            if ctx:
                llm_response = ctx.dispatch_tool("generate_text", {
                    "prompt": prompt,
                    "max_tokens": 1000,
                })
            else:
                # 降级：使用规则评分
                return self._rule_based_grade(trajectory)

            # 解析 LLM 响应
            grade_result = self._parse_grade_response(llm_response)
            self._logger.info(
                "[Grader] LLM 评分完成: overall=%.1f | completeness=%d | correctness=%d",
                grade_result.get("overall_score", 0),
                grade_result.get("completeness", 0),
                grade_result.get("correctness", 0),
            )
            return grade_result

        except Exception as e:
            self._logger.warning("[Grader] LLM 评分失败，降级为规则评分: %s", e)
            return self._rule_based_grade(trajectory)

    def _build_trajectory_summary(self, trajectory: Dict) -> str:
        """构建轨迹摘要"""
        steps = trajectory.get("steps", [])
        if not steps:
            return "无执行步骤"

        lines = []
        for i, step in enumerate(steps, 1):
            step_type = step.get("step_type", "unknown")
            desc = step.get("description", "")
            success = "✅" if step.get("success") else "❌"
            duration = step.get("duration_ms", 0)

            line = f"Step {i} [{step_type}] {success} {desc}"
            if duration > 0:
                line += f" ({duration:.0f}ms)"

            # 添加工具调用信息
            tool_calls = step.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    tc_success = "✅" if tc.get("success") else "❌"
                    line += f"\n  └─ 工具: {tc.get('tool_name', 'unknown')} {tc_success}"

            lines.append(line)

        return "\n".join(lines)

    def _parse_grade_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 评分响应"""
        # 尝试提取 JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return self._validate_grade(data)
            except json.JSONDecodeError:
                pass

        # 尝试直接解析
        try:
            data = json.loads(response)
            return self._validate_grade(data)
        except json.JSONDecodeError:
            pass

        # 解析失败，返回默认值
        return {
            "completeness": 3,
            "correctness": 3,
            "efficiency": 3,
            "tool_usage": 3,
            "error_handling": 3,
            "overall_score": 3.0,
            "feedback": "评分解析失败，使用默认值",
            "strengths": [],
            "improvements": [],
        }

    def _validate_grade(self, data: Dict) -> Dict[str, Any]:
        """验证和标准化评分"""
        dimensions = ["completeness", "correctness", "efficiency", "tool_usage", "error_handling"]

        for dim in dimensions:
            if dim in data:
                data[dim] = max(1, min(5, int(data[dim])))
            else:
                data[dim] = 3

        # 计算总体分数
        if "overall_score" not in data:
            data["overall_score"] = sum(data[d] for d in dimensions) / len(dimensions)

        # 确保必要字段存在
        data.setdefault("feedback", "")
        data.setdefault("strengths", [])
        data.setdefault("improvements", [])

        return data

    def _rule_based_grade(self, trajectory: Dict) -> Dict[str, Any]:
        """规则评分（LLM 不可用时的降级方案）"""
        success = trajectory.get("success", False)
        steps = trajectory.get("steps", [])
        error = trajectory.get("error")
        duration_ms = trajectory.get("total_duration_ms", 0)

        # 基础分
        base = 4.0 if success else 2.0

        # 步骤效率
        if len(steps) <= 3:
            efficiency = 5
        elif len(steps) <= 6:
            efficiency = 4
        elif len(steps) <= 10:
            efficiency = 3
        else:
            efficiency = 2

        # 错误处理
        has_retry = any(s.get("step_type") == "retry" for s in steps)
        error_handling = 4 if has_retry else (3 if success else 1)

        return {
            "completeness": 4 if success else 2,
            "correctness": 4 if success else 2,
            "efficiency": efficiency,
            "tool_usage": 3,
            "error_handling": error_handling,
            "overall_score": base,
            "feedback": f"规则评分: {'成功' if success else '失败'}" + (f"，错误: {error}" if error else ""),
            "strengths": ["执行完成"] if success else [],
            "improvements": ["需要改进错误处理"] if not success else [],
        }


# 全局实例
_grader: Optional[AgentGrader] = None


def get_grader() -> AgentGrader:
    """获取全局评估器"""
    global _grader
    if _grader is None:
        _grader = AgentGrader()
    return _grader
