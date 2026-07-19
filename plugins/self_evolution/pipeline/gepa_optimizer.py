"""
GEPA 风格优化器 v2

改进点（基于 SkillOpt 对标分析）：
1. 接入 ReflectionStore 历史反思（跨轮次经验复用）
2. 改进变异 prompt（失败模式分组 + 成功案例 + 结构化编辑指导）
3. 收集成功轨迹（不只看失败，也学做得好的部分）
4. 反思类型区分（skill_defect vs execution_lapse）
"""

from __future__ import annotations

import logging
import time
from typing import Optional
from collections import Counter

from self_evolution.core.evolution_provider import EvalDataset, FitnessScore
from self_evolution.core.fitness import evaluate_skill, _get_active_model
from self_evolution.pipeline.base_optimizer import SkillOptimizerBase, OptimizeResult
from self_evolution.core.reflection_store import (
    get_reflection_store, EvolutionReflection,
)

logger = logging.getLogger(__name__)


# ── v2 变异 prompt：结构化分析 + 成功案例 + 历史反思 ──────────

_MUTATE_PROMPT_V2 = """你是 AI agent 技能优化专家。基于评估反馈改进下面的技能文件。

当前技能文本：
---
{current_text}
---

失败分析（共 {fail_count} 个失败案例，按模式分组）：
---
{failure_analysis}
---

成功案例（做得好的部分，改进时保持）：
---
{success_analysis}
---

{reflections_prompt}

改进要求：
1. **只改有问题的部分**，保持成功案例覆盖的能力不变
2. **具体化模糊指令**：如果反馈说"步骤不具体"，补充具体命令/示例
3. **处理边缘情况**：如果多个失败指向同一模式，添加防护措施
4. **保持结构稳定**：不大幅重组步骤顺序，除非反馈明确要求
5. **避免重复失败策略**：如果有历史反思，不要重蹈覆辙

只返回改进后的完整技能文本，不要任何解释。"""


def _group_failures(feedbacks: list[dict]) -> str:
    """将失败案例按 failure_cause 分组，提炼共性模式。"""
    if not feedbacks:
        return "无失败案例"

    # 按 failure_cause 分组
    by_cause: dict[str, list[dict]] = {}
    for fb in feedbacks:
        cause = fb.get("failure_cause", "unknown")
        by_cause.setdefault(cause, []).append(fb)

    lines = []
    for cause, items in by_cause.items():
        lines.append(f"\n## {cause} ({len(items)} 个)")
        # 提取共性 feedback
        feedback_texts = [i["feedback"][:150] for i in items]
        # 用 Counter 找高频关键词
        all_text = " ".join(feedback_texts).lower()
        keywords = ["步骤", "缺失", "模糊", "错误", "遗漏", "冗余", "不清晰",
                     "missing", "unclear", "wrong", "step", "error"]
        freq = {kw: all_text.count(kw) for kw in keywords if all_text.count(kw) > 0}
        if freq:
            lines.append(f"  高频词: {dict(sorted(freq.items(), key=lambda x: -x[1])[:5])}")

        # 展示前 3 个典型 feedback
        for i, item in enumerate(items[:3]):
            lines.append(f"  [{i+1}] 分数={item['score']:.2f} | {item['feedback'][:120]}")

    return "\n".join(lines)


def _summarize_successes(fitness_list: list[FitnessScore], examples: list) -> str:
    """总结成功案例的共同特征。"""
    successes = [
        (fs, ex) for fs, ex in zip(fitness_list, examples)
        if fs.composite >= 0.7
    ]
    if not successes:
        return "无成功案例"

    lines = [f"共 {len(successes)} 个成功案例（分数 ≥ 0.7）："]
    for fs, ex in successes[:3]:
        task_preview = ex.task_input[:80] if hasattr(ex, 'task_input') else str(ex)[:80]
        lines.append(f"  ✓ 分数={fs.composite:.2f} | 任务: {task_preview}")
        if fs.feedback:
            lines.append(f"    优点: {fs.feedback[:100]}")

    return "\n".join(lines)


class GEPAOptimizer(SkillOptimizerBase):
    """GEPA 风格优化器 v2：feedback → mutate → eval → select 单候选循环。"""

    name = "gepa"

    def __init__(self, client=None, model: str = None):
        self._client = client
        self._model = model

    def optimize(
        self,
        skill_text: str,
        dataset: EvalDataset,
        iterations: int = 5,
        temperature: float = 0.7,
        skill_name: str = "",
    ) -> OptimizeResult:
        from openai import OpenAI
        client = self._client or OpenAI()
        model = self._model or _get_active_model()

        current_text = skill_text
        current_score = self._score(skill_text, dataset, client, model)
        best_text = current_text
        best_score = current_score
        audit_report = {"iterations": [], "optimizer": "gepa-v2"}
        no_improve_streak = 0

        # 加载历史反思
        reflections_prompt = ""
        if skill_name:
            reflections_prompt = get_reflection_store().get_lessons_prompt(skill_name)

        logger.info(f"[GEPA-v2] baseline score: {current_score:.3f}")

        iterations_used = 0
        for i in range(iterations):
            # 收集反馈（失败+成功）
            failure_feedbacks, success_summary, fitness_list = self._collect_feedback(
                current_text, dataset, client, model
            )

            # 变异（带结构化分析）
            mutated = self._mutate(
                current_text, failure_feedbacks, success_summary,
                reflections_prompt, client, model, temperature
            )

            new_score = self._score(mutated, dataset, client, model)
            improvement = new_score - current_score

            if new_score > best_score:
                best_text = mutated
                best_score = new_score

            accepted = new_score > current_score
            if accepted:
                current_text = mutated
                current_score = new_score
                no_improve_streak = 0
            else:
                no_improve_streak += 1

            # 保存反思
            if skill_name and not accepted and failure_feedbacks:
                _store = get_reflection_store()
                for fb in failure_feedbacks[:2]:  # 只存 top 2
                    _store.add_reflection(skill_name, EvolutionReflection(
                        timestamp=time.time(),
                        failure_type=fb.get("failure_cause", "skill_defect"),
                        weak_dimension=self._detect_weak_dim(fb.get("feedback", "")),
                        feedback_summary=fb.get("feedback", "")[:200],
                        mutation_strategy="gepa-v2",
                        score_before=current_score,
                        score_after=new_score,
                        lesson=f"iter{i+1}: {fb.get('feedback', '')[:150]}",
                    ))

            iter_report = {
                "iteration": i + 1,
                "score": new_score,
                "best_score": best_score,
                "accepted": accepted,
                "improvement": improvement,
                "failures": len(failure_feedbacks),
            }
            audit_report["iterations"].append(iter_report)

            iterations_used = i + 1
            logger.info(
                f"[GEPA-v2] iter {i+1}: {current_score:.3f} → {new_score:.3f} "
                f"(Δ={improvement:+.3f}) accepted={accepted} "
                f"failures={len(failure_feedbacks)} streak={no_improve_streak}"
            )

            # 连续 2 轮无改善 → 提前停止
            if no_improve_streak >= 2:
                logger.info(f"[GEPA-v2] early stop after {i+1} iterations")
                break

        return OptimizeResult(
            evolved_text=best_text,
            best_score=best_score,
            iterations_used=iterations_used,
            audit_report=audit_report,
        )

    def score(self, skill_text: str, dataset: EvalDataset) -> float:
        from openai import OpenAI
        client = self._client or OpenAI()
        model = self._model or _get_active_model()
        return self._score(skill_text, dataset, client, model)

    def _score(self, skill_text: str, dataset: EvalDataset, client, model) -> float:
        if not dataset.val:
            return 0.0
        total = 0.0
        for ex in dataset.val:
            fs = evaluate_skill(skill_text, ex, client, model)
            total += fs.composite
        return total / len(dataset.val)

    def _collect_feedback(self, text, dataset, client, model):
        """收集失败+成功反馈，返回 (failures, success_summary, fitness_list)。"""
        failure_feedbacks = []
        fitness_list = []
        examples = dataset.val[:5]

        for ex in examples:
            fs = evaluate_skill(text, ex, client, model)
            fitness_list.append(fs)

            if fs.composite < 0.7:
                failure_feedbacks.append({
                    "task": str(ex.task_input)[:80] if hasattr(ex, 'task_input') else "",
                    "score": fs.composite,
                    "feedback": fs.feedback or "无反馈",
                    "failure_cause": fs.failure_cause or "unknown",
                })

        success_summary = _summarize_successes(fitness_list, examples)
        return failure_feedbacks, success_summary, fitness_list

    def _mutate(self, text, failure_feedbacks, success_summary,
                reflections_prompt, client, model, temperature):
        """基于结构化分析变异技能文本。"""
        if not failure_feedbacks:
            # 没有失败，做小幅度优化
            failure_analysis = "无失败案例，技能表现良好。请做微小优化：提高清晰度、补充边缘情况处理。"
        else:
            failure_analysis = _group_failures(failure_feedbacks)

        prompt = _MUTATE_PROMPT_V2.format(
            current_text=text,
            fail_count=len(failure_feedbacks),
            failure_analysis=failure_analysis,
            success_analysis=success_summary,
            reflections_prompt=reflections_prompt or "",
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4000,
            )
            result = resp.choices[0].message.content or ""
            # 去除可能的 markdown 代码块包裹
            result = result.strip()
            if result.startswith("```"):
                lines = result.split("\n")
                # 去掉首尾的 ``` 行
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                result = "\n".join(lines)
            return result if result else text
        except Exception as exc:
            logger.warning(f"[GEPA-v2] 变异失败: {exc}")
            return text

    @staticmethod
    def _detect_weak_dim(feedback: str) -> str:
        """从反馈文本中检测最弱维度。"""
        fb_lower = feedback.lower()
        dim_keywords = {
            "accuracy": ["准确", "正确", "error", "错误", "wrong", "incorrect"],
            "clarity": ["清晰", "模糊", "unclear", "ambiguous", "confusing", "混淆"],
            "completeness": ["完整", "缺失", "缺少", "missing", "incomplete", "遗漏"],
            "efficiency": ["效率", "冗余", "slow", "redundant", "冗长"],
            "safety": ["安全", "危险", "unsafe", "risk", "harmful"],
        }
        scores = {}
        for dim, keywords in dim_keywords.items():
            scores[dim] = sum(1 for kw in keywords if kw in fb_lower)
        if not scores or max(scores.values()) == 0:
            return "unknown"
        return max(scores, key=scores.get)
