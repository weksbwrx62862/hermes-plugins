"""
GEPA 风格优化器

Gradient-free Evolutionary Prompt Adaptation：
feedback → mutate → eval → select 循环，每次只保留最优候选。
"""

from __future__ import annotations

import logging
from typing import Optional

from self_evolution.core.evolution_provider import EvalDataset
from self_evolution.core.fitness import evaluate_skill, _get_active_model
from self_evolution.pipeline.base_optimizer import SkillOptimizerBase, OptimizeResult

logger = logging.getLogger(__name__)


class GEPAOptimizer(SkillOptimizerBase):
    """GEPA 风格优化器：feedback → mutate → eval → select 单候选循环。"""

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
    ) -> OptimizeResult:
        from openai import OpenAI
        client = self._client or OpenAI()
        model = self._model or _get_active_model()

        current_text = skill_text
        current_score = self.score(skill_text, dataset)
        best_text = current_text
        best_score = current_score
        audit_report = {"iterations": [], "optimizer": "gepa"}

        for i in range(iterations):
            feedback = self._collect_feedback(current_text, dataset, client, model)
            mutated = self._mutate(current_text, feedback, client, model, temperature)
            new_score = self.score(mutated, dataset)

            if new_score > best_score:
                best_text = mutated
                best_score = new_score
                current_text = mutated
                current_score = new_score

            audit_report["iterations"].append({
                "iteration": i + 1,
                "score": new_score,
                "best_score": best_score,
                "accepted": new_score > current_score,
            })

        return OptimizeResult(
            evolved_text=best_text,
            best_score=best_score,
            iterations_used=iterations,
            audit_report=audit_report,
        )

    def score(self, skill_text: str, dataset: EvalDataset) -> float:
        from openai import OpenAI
        client = self._client or OpenAI()
        model = self._model or _get_active_model()
        if not dataset.val:
            return 0.0
        total = 0.0
        for ex in dataset.val:
            fs = evaluate_skill(skill_text, ex, client, model)
            total += fs.composite
        return total / len(dataset.val)

    def _collect_feedback(self, text, dataset, client, model):
        """从验证集收集失败反馈。"""
        feedbacks = []
        for ex in dataset.val[:5]:
            fs = evaluate_skill(text, ex, client, model)
            if fs.composite < 0.7:
                feedbacks.append(
                    f"任务: {ex.task_input[:80]} | 分数: {fs.composite:.2f} | 问题: {fs.failure_cause}"
                )
        return "\n".join(feedbacks) if feedbacks else "无明显问题"

    def _mutate(self, text, feedback, client, model, temperature):
        """基于反馈变异技能文本。"""
        prompt = f"""基于以下反馈改进技能描述。只修改有问题的部分，保持其余不变。

当前技能：
---
{text}
---

失败反馈：
{feedback}

改进后的技能（完整输出）："""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"[GEPA] 变异失败: {exc}")
            return text
