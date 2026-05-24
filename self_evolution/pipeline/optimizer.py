"""
8阶段管线 — 阶段 6: SkillOptimizer 优化循环

v2 改进：
  1. 策略多样化探索：每次迭代生成 K=4 个不同执行策略
  2. 对比更新：成功 vs 失败轨迹对比，提炼改进点
  3. 独立 Auditor：9 项机械检查，拦截有害更新
  4. 反思驱动变异：只有技能缺陷才触发变异

核心思路（GEPA 教学模拟）：
  1. 在 train set 上评估，收集 feedback
  2. LLM 基于 feedback 针对性改写 skill
  3. 在 val set 上评估新版本
  4. 更好则保留，否则回退
  5. 重复 N 轮，连续 2 次无改善则提前停止
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI

from self_evolution.core.evolution_provider import EvalDataset, FitnessScore
from self_evolution.core.fitness import evaluate_skill, quick_fitness, reflect_on_failure, ReflectionResult

logger = logging.getLogger(__name__)

# ── 变异提示词 ─────────────────────────────────────────────────

_MUTATE_PROMPT = """你是 AI agent 技能优化专家。基于评估反馈改进下面的技能文件。

当前技能文本：
---
{current_text}
---

评估反馈（来自实际使用测试）：
---
{feedback}
---

请基于反馈重写技能文本，解决发现的问题。注意：
- 保持相同的总体目的和结构
- 提高指令清晰度
- 处理反馈中提到的边缘情况
- 改进分步流程
- 如果反馈指出缺少步骤，添加步骤
- 如果反馈指出措辞不清，改写措辞

只返回改进后的技能文本，不要任何解释。"""


# ── 策略多样化变异提示词 ─────────────────────────────────────

_DIVERSIFY_PROMPT = """你是 AI agent 技能优化专家。基于评估反馈，生成 {k} 个不同的改进策略。

当前技能文本：
---
{current_text}
---

评估反馈：
---
{feedback}
---

请生成 {k} 个不同的改进方案，每个方案采用不同的策略：
1. 方案A（保守）：最小改动，只修复最关键的问题
2. 方案B（结构）：重新组织步骤顺序和逻辑
3. 方案C（补充）：添加缺失的步骤和边缘情况处理
4. 方案D（简化）：精简冗余内容，提高清晰度

返回 JSON 数组：
```json
[
  {{"strategy": "conservative", "text": "改进后的技能文本..."}},
  {{"strategy": "structural", "text": "改进后的技能文本..."}},
  {{"strategy": "supplement", "text": "改进后的技能文本..."}},
  {{"strategy": "simplify", "text": "改进后的技能文本..."}}
]
```
只返回 JSON，不要解释。"""


# ── 对比更新提示词 ─────────────────────────────────────────────

_CONTRAST_PROMPT = """你是技能优化专家。通过对比成功和失败的执行轨迹，提炼改进点。

当前技能文本：
---
{current_text}
---

成功轨迹：
---
{success_trace}
---

失败轨迹：
---
{failure_trace}
---

请分析成功和失败的关键差异，提炼出具体的改进建议。只返回改进建议，不要修改技能文本。"""


# ── Auditor 检查项 ─────────────────────────────────────────────

_AUDITOR_PROMPT = """你是技能质量审计员。请对下面的技能文本进行 9 项机械检查。

技能文本：
---
{skill_text}
---

检查项：
1. **格式完整性**：是否有标题、步骤、示例等基本结构
2. **一致性**：术语、命名、格式是否前后一致
3. **可执行性**：步骤是否具体可执行，没有模糊描述
4. **完整性**：是否覆盖了关键步骤和边缘情况
5. **清晰度**：指令是否清晰无歧义
6. **错误处理**：是否包含错误处理和故障排除
7. **示例质量**：示例是否充分、有代表性
8. **安全性**：是否有危险操作警告
9. **可维护性**：是否易于理解和修改

返回 JSON：
```json
{{
  "passed": true,
  "score": 0.85,
  "checks": {{
    "format": {{"passed": true, "note": ""}},
    "consistency": {{"passed": true, "note": ""}},
    "executability": {{"passed": false, "note": "步骤3缺少具体命令"}},
    "completeness": {{"passed": true, "note": ""}},
    "clarity": {{"passed": true, "note": ""}},
    "error_handling": {{"passed": false, "note": "缺少错误处理"}},
    "examples": {{"passed": true, "note": ""}},
    "safety": {{"passed": true, "note": ""}},
    "maintainability": {{"passed": true, "note": ""}}
  }},
  "critical_failures": ["error_handling"],
  "recommendations": ["添加错误处理步骤", "步骤3需要更具体的命令"]
}}
```
只返回 JSON，不要解释。"""


@dataclass
class AuditResult:
    """审计结果"""
    passed: bool
    score: float
    checks: dict
    critical_failures: list
    recommendations: list


@dataclass
class DiversifyResult:
    """多样化变异结果"""
    strategy: str
    text: str
    score: float = 0.0


class SkillOptimizerBase(ABC):
    """技能优化器基类 — 模块化优化器框架。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """优化器名称。"""
        ...

    @abstractmethod
    def optimize(
        self,
        skill_text: str,
        dataset: EvalDataset,
        iterations: int = 5,
    ) -> tuple[str, float, int, dict]:
        """
        优化循环。

        Returns: (evolved_text, best_score, iterations_used, audit_report)
        """
        ...

    def score(self, skill_text: str, examples: list) -> float:
        """在给定数据集上计算平均分数。默认实现调用 evaluate_skill。"""
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            try:
                fitness = evaluate_skill(skill_text, ex, self.client, self.model)
                scores.append(fitness.composite)
            except Exception:
                pass
        return sum(scores) / len(scores) if scores else 0.0

    def audit(self, skill_text: str) -> "AuditResult":
        """审计技能质量。默认实现返回通过。"""
        return AuditResult(
            passed=True, score=1.0, checks={},
            critical_failures=[], recommendations=[],
        )


class SkillOptimizer(SkillOptimizerBase):
    """
    技能优化器 — 阶段 6 核心。

    v2 改进：
      1. 策略多样化：每次生成 K=4 个候选
      2. 对比更新：成功/失败轨迹对比
      3. 独立 Auditor：9 项机械检查
      4. 反思驱动：只有技能缺陷才变异

    两种模式：
      - use_llm=True: 完整 LLM-as-judge（慢但准）
      - use_llm=False: 快速启发式（快但粗糙，用于迭代加速）
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        use_llm: bool = True,
        k_candidates: int = 4,
    ):
        self.client = client
        self.model = model
        self.use_llm = use_llm
        self.k_candidates = k_candidates

    @property
    def name(self) -> str:
        return "diversify-evolver"

    def optimize(
        self,
        skill_text: str,
        dataset: EvalDataset,
        iterations: int = 5,
    ) -> tuple[str, float, int, dict]:
        """
        优化循环 v2。

        Returns: (evolved_text, best_score, iterations_used, audit_report)
        """

        current_text = skill_text
        current_score = self._score_on_split(current_text, dataset.val)
        best_text = current_text
        best_score = current_score
        no_improve_streak = 0
        audit_report = {"iterations": [], "audits": []}
        temperature_penalty = 0.0

        logger.info(f"[optimizer] baseline val score: {current_score:.3f}")

        for iteration in range(1, iterations + 1):
            iter_report = {"iteration": iteration}

            # 步骤 1: 评估 + 反思
            train_score, feedbacks, reflections, fitness_scores = self._eval_with_reflection(
                current_text, dataset.train
            )

            # silent-bypass 检测
            if self._check_silent_bypass(fitness_scores):
                audit_report["silent_bypass"] = True
                audit_report["evolution_priority"] = "low"
                logger.warning("[optimizer] 检测到 silent-bypass: 技能从未被实际遵循")

            # 过滤：只保留需要变异的反馈
            mutate_feedbacks = [
                fb for fb, ref in zip(feedbacks, reflections)
                if ref.should_mutate
            ]

            if not mutate_feedbacks:
                logger.info(f"[optimizer] iter {iteration}: no skill defects, skipping")
                no_improve_streak += 1
                iter_report["status"] = "no_defects"
                audit_report["iterations"].append(iter_report)
                if no_improve_streak >= 2:
                    break
                continue

            combined_feedback = self._merge_feedback(mutate_feedbacks)
            iter_report["feedback_count"] = len(mutate_feedbacks)

            # 步骤 2: 策略多样化变异（生成 K 个候选）
            adjusted_temp = max(0.1, 0.7 - temperature_penalty)
            candidates = self._diversify(current_text, combined_feedback, temperature=adjusted_temp)
            iter_report["candidates"] = len(candidates)

            # 步骤 3: 评估所有候选
            for cand in candidates:
                cand.score = self._score_on_split(cand.text, dataset.val)

            # 步骤 4: 选择最佳候选
            best_candidate = max(candidates, key=lambda c: c.score)
            iter_report["best_strategy"] = best_candidate.strategy
            iter_report["best_score"] = best_candidate.score

            # overfit 检测
            if self._check_overfit(train_score, best_candidate.score):
                temperature_penalty = 0.2
                iter_report["overfit"] = True
                logger.warning(
                    f"[optimizer] iter {iteration}: 检测到过拟合 "
                    f"(train={train_score:.3f}, val={best_candidate.score:.3f})"
                )

            # 步骤 5: Auditor 检查
            audit = self._audit(best_candidate.text)
            iter_report["audit_passed"] = audit.passed
            iter_report["audit_score"] = audit.score

            if not audit.passed:
                logger.warning(f"[optimizer] iter {iteration}: auditor rejected")
                iter_report["status"] = "audit_rejected"
                audit_report["iterations"].append(iter_report)
                no_improve_streak += 1
                if no_improve_streak >= 2:
                    break
                continue

            # 步骤 6: 择优
            if best_candidate.score > best_score:
                improvement = best_candidate.score - best_score
                best_text = best_candidate.text
                best_score = best_candidate.score
                current_text = best_candidate.text
                no_improve_streak = 0
                iter_report["status"] = "improved"
                iter_report["improvement"] = improvement
                logger.info(
                    f"[optimizer] iter {iteration}: "
                    f"{current_score:.3f} → {best_candidate.score:.3f} (+{improvement:.3f}) "
                    f"strategy={best_candidate.strategy}"
                )
            else:
                no_improve_streak += 1
                iter_report["status"] = "no_improvement"
                logger.info(
                    f"[optimizer] iter {iteration}: "
                    f"no improvement ({best_candidate.score:.3f} ≤ {best_score:.3f})"
                )

            current_score = best_candidate.score
            audit_report["iterations"].append(iter_report)

            # 连续无改善 → 提前停止
            if no_improve_streak >= 2:
                logger.info(f"[optimizer] stopping early after {iteration} iterations")
                break

        logger.info(f"[optimizer] done: best score {best_score:.3f}")
        return best_text, best_score, iteration, audit_report

    # ── 内部实现 ─────────────────────────────────────────

    def _score_on_split(self, skill_text: str, examples: list) -> float:
        """在给定数据集上计算平均分数。"""
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            try:
                if self.use_llm:
                    fitness = evaluate_skill(skill_text, ex, self.client, self.model)
                else:
                    fitness = quick_fitness(skill_text, ex)
                scores.append(fitness.composite)
            except Exception as exc:
                logger.debug(f"eval error: {exc}")
        return sum(scores) / len(scores) if scores else 0.0

    def _eval_with_reflection(
        self, skill_text: str, examples: list
    ) -> tuple[float, list[str], list[ReflectionResult], list]:
        """评估 + 反思。返回 (平均分, 反馈列表, 反思列表, 适应度列表)。"""
        scores = []
        feedbacks = []
        reflections = []
        fitness_scores = []

        logger.info(f"[optimizer._eval_with_reflection] 开始评估 {len(examples)} 个样本")

        for idx, ex in enumerate(examples):
            try:
                if self.use_llm:
                    fitness = evaluate_skill(skill_text, ex, self.client, self.model)
                else:
                    fitness = quick_fitness(skill_text, ex)
                scores.append(fitness.composite)
                fitness_scores.append(fitness)

                if fitness.composite < 0.5:
                    logger.info(
                        f"[optimizer._eval_with_reflection] 样本{idx} 分数低 ({fitness.composite:.3f}) | "
                        f"failure_cause={fitness.failure_cause} → 触发反思"
                    )
                    agent_output = ""
                    reflection = reflect_on_failure(
                        skill_text, ex, agent_output, self.client, self.model
                    )
                    if reflection.reflection_type in ("skill_defect", "optimization"):
                        reflection.should_mutate = True
                    elif reflection.reflection_type == "execution_lapse":
                        reflection.should_mutate = False
                    elif reflection.reflection_type == "discovery":
                        reflection.should_mutate = False
                        logger.info(f"[optimizer] 发现: {reflection.feedback[:100]}")
                    reflections.append(reflection)
                    feedbacks.append(fitness.feedback)
                else:
                    ref_type = "optimization" if fitness.composite < 0.8 else "discovery"
                    if ref_type == "discovery":
                        logger.info(f"[optimizer] 发现: {fitness.feedback[:100]}")
                    reflections.append(ReflectionResult(
                        reflection_type=ref_type,
                        should_mutate=ref_type in ("skill_defect", "optimization"),
                        feedback=fitness.feedback,
                        confidence=0.8,
                    ))
                    feedbacks.append(fitness.feedback)
            except Exception as exc:
                logger.warning(f"[optimizer._eval_with_reflection] 样本{idx} 评估异常: {exc}")

        avg = sum(scores) / len(scores) if scores else 0.0
        mutate_count = sum(1 for r in reflections if r.should_mutate)
        logger.info(
            f"[optimizer._eval_with_reflection] 完成 | avg={avg:.3f} | "
            f"total={len(scores)} | mutate_needed={mutate_count} | "
            f"reflection_types={[r.reflection_type for r in reflections]}"
        )
        return avg, feedbacks, reflections, fitness_scores

    def _check_silent_bypass(self, fitness_scores: list) -> bool:
        """检测 silent-bypass：技能从未被实际遵循。"""
        if not fitness_scores:
            return True
        procedure_scores = [f.procedure_following for f in fitness_scores]
        return all(p < 0.1 for p in procedure_scores)

    def _check_overfit(self, train_score: float, val_score: float, threshold: float = 0.15) -> bool:
        """检测过拟合：train 远优于 val。"""
        return (train_score - val_score) > threshold

    def _eval_with_feedback(
        self, skill_text: str, examples: list
    ) -> tuple[float, list[str]]:
        """评估并收集所有 feedback。"""
        scores = []
        feedbacks = []
        for ex in examples:
            try:
                if self.use_llm:
                    fitness = evaluate_skill(skill_text, ex, self.client, self.model)
                else:
                    fitness = quick_fitness(skill_text, ex)
                scores.append(fitness.composite)
                if fitness.feedback:
                    feedbacks.append(fitness.feedback)
            except Exception:
                pass

        avg = sum(scores) / len(scores) if scores else 0.0
        return avg, feedbacks

    def _diversify(self, current_text: str, feedback: str, temperature: float = 0.7) -> list[DiversifyResult]:
        """策略多样化变异 — 生成 K 个不同候选。"""
        logger.info(f"[optimizer._diversify] 生成 {self.k_candidates} 个候选 | temperature={temperature:.2f}")

        prompt = _DIVERSIFY_PROMPT.format(
            current_text=current_text,
            feedback=feedback[:2000],
            k=self.k_candidates,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        import json
        import re
        text = response.choices[0].message.content or "[]"

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group(1))
                except json.JSONDecodeError:
                    items = []
            else:
                items = []

        candidates = []
        for item in items:
            if isinstance(item, dict) and "text" in item:
                candidates.append(DiversifyResult(
                    strategy=item.get("strategy", "unknown"),
                    text=item["text"],
                ))

        if not candidates:
            logger.warning("[optimizer._diversify] LLM 未返回有效候选，使用 fallback（原文本）")
            candidates.append(DiversifyResult(
                strategy="fallback",
                text=current_text,
            ))

        strategies = [c.strategy for c in candidates]
        logger.info(f"[optimizer._diversify] 生成 {len(candidates)} 个候选 | strategies={strategies}")
        return candidates

    def _audit(self, skill_text: str) -> AuditResult:
        """独立 Auditor — 9 项机械检查。"""
        logger.info("[optimizer._audit] 开始 9 项机械检查")

        prompt = _AUDITOR_PROMPT.format(skill_text=skill_text[:3000])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        import json
        import re
        text = response.choices[0].message.content or "{}"

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(1))
                except json.JSONDecodeError:
                    result = {}
            else:
                result = {}

        audit = AuditResult(
            passed=result.get("passed", False),
            score=result.get("score", 0),
            checks=result.get("checks", {}),
            critical_failures=result.get("critical_failures", []),
            recommendations=result.get("recommendations", []),
        )

        logger.info(
            f"[optimizer._audit] 审计完成 | passed={audit.passed} | score={audit.score:.2f} | "
            f"critical_failures={audit.critical_failures} | recommendations={audit.recommendations[:3]}"
        )
        return audit

    def _contrast_update(
        self, current_text: str, success_trace: str, failure_trace: str
    ) -> str:
        """对比更新 — 通过成功/失败轨迹对比提炼改进点。"""
        logger.info("[optimizer._contrast_update] 开始对比更新")

        prompt = _CONTRAST_PROMPT.format(
            current_text=current_text[:3000],
            success_trace=success_trace[:2000],
            failure_trace=failure_trace[:2000],
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        result = response.choices[0].message.content or ""
        logger.info(f"[optimizer._contrast_update] 对比更新完成 | result_len={len(result)}")
        return result

    def _merge_feedback(self, feedbacks: list[str], max_len: int = 2000) -> str:
        """合并多条 feedback，用词袋相似度去重（>95% 相似视为重复）。"""
        if len(feedbacks) <= 1:
            return feedbacks[0] if feedbacks else ""

        logger.info(f"[optimizer._merge_feedback] 合并 {len(feedbacks)} 条反馈")

        import numpy as np

        def _embed(text: str, dim: int = 384) -> np.ndarray:
            vec = np.zeros(dim, dtype=np.float32)
            tl = text.lower()
            for n in [3, 4, 5]:
                for i in range(len(tl) - n + 1):
                    h = hash(tl[i:i+n]) % dim
                    vec[h] += 1.0
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec

        vecs = [_embed(fb) for fb in feedbacks]
        unique = [feedbacks[0]]
        unique_vecs = [vecs[0]]

        for i in range(1, len(feedbacks)):
            max_sim = max(np.dot(vecs[i], uv) for uv in unique_vecs)
            if max_sim < 0.95:
                unique.append(feedbacks[i])
                unique_vecs.append(vecs[i])

        merged = "\n\n---\n\n".join(unique)
        if len(merged) > max_len:
            merged = merged[:max_len] + "..."

        logger.info(f"[optimizer._merge_feedback] 去重后 {len(unique)}/{len(feedbacks)} 条 | merged_len={len(merged)}")
        return merged

    def _mutate(self, current_text: str, feedback: str,
                iteration: int = 1, total_iterations: int = 5) -> str:
        """基于 feedback 让 LLM 重写 skill，temperature 退火（0.8→0.3）。"""
        if total_iterations > 1:
            temperature = 0.8 - 0.5 * (iteration - 1) / (total_iterations - 1)
        else:
            temperature = 0.5
        temperature = round(max(0.3, temperature), 2)

        prompt = _MUTATE_PROMPT.format(
            current_text=current_text,
            feedback=feedback[:2000],
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        logger.debug(f"[optimizer] mutate iter={iteration}/{total_iterations} temp={temperature}")
        return response.choices[0].message.content or current_text


class GEPAOptimizer(SkillOptimizerBase):
    """GEPA 风格优化器 — 简单的 feedback→mutate→eval→select 循环。"""

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini", use_llm: bool = True):
        self.client = client
        self.model = model
        self.use_llm = use_llm

    @property
    def name(self) -> str:
        return "gepa-evolver"

    def optimize(
        self,
        skill_text: str,
        dataset: EvalDataset,
        iterations: int = 5,
    ) -> tuple[str, float, int, dict]:
        current_text = skill_text
        current_score = self.score(current_text, dataset.val)
        best_text = current_text
        best_score = current_score
        no_improve_streak = 0
        audit_report = {"iterations": [], "optimizer": "gepa"}

        logger.info(f"[GEPA] 开始优化 | baseline={current_score:.3f} | iterations={iterations}")

        for iteration in range(1, iterations + 1):
            iter_report = {"iteration": iteration}

            train_score, feedbacks = self._eval_with_feedback(current_text, dataset.train)
            logger.info(f"[GEPA] iter {iteration}: train_score={train_score:.3f} | feedbacks={len(feedbacks)}")

            if not feedbacks:
                no_improve_streak += 1
                logger.info(f"[GEPA] iter {iteration}: 无反馈，跳过变异 | streak={no_improve_streak}")
                if no_improve_streak >= 2:
                    break
                continue

            combined_feedback = "\n\n---\n\n".join(feedbacks[:3])

            temperature = max(0.3, 0.8 - 0.5 * (iteration - 1) / max(iterations - 1, 1))
            mutated_text = self._mutate(current_text, combined_feedback, temperature)

            new_score = self.score(mutated_text, dataset.val)

            if new_score > best_score:
                improvement = new_score - current_score
                best_text = mutated_text
                best_score = new_score
                current_text = mutated_text
                no_improve_streak = 0
                iter_report["status"] = "improved"
                iter_report["improvement"] = improvement
                logger.info(
                    f"[GEPA] iter {iteration}: ✓ 改进 | "
                    f"{current_score:.3f} → {new_score:.3f} (+{improvement:.3f})"
                )
            else:
                no_improve_streak += 1
                iter_report["status"] = "no_improvement"
                logger.info(
                    f"[GEPA] iter {iteration}: ✗ 无改进 | {new_score:.3f} ≤ {best_score:.3f} | streak={no_improve_streak}"
                )

            current_score = new_score
            audit_report["iterations"].append(iter_report)

            if no_improve_streak >= 2:
                logger.info(f"[GEPA] 连续 {no_improve_streak} 次无改进，提前停止")
                break

        logger.info(f"[GEPA] 优化完成 | best_score={best_score:.3f} | iterations_used={iteration}")
        return best_text, best_score, iteration, audit_report

    def _eval_with_feedback(self, skill_text, examples):
        scores = []
        feedbacks = []
        for ex in examples:
            try:
                if self.use_llm:
                    fitness = evaluate_skill(skill_text, ex, self.client, self.model)
                else:
                    fitness = quick_fitness(skill_text, ex)
                scores.append(fitness.composite)
                if fitness.feedback:
                    feedbacks.append(fitness.feedback)
            except Exception:
                pass
        avg = sum(scores) / len(scores) if scores else 0.0
        return avg, feedbacks

    def _mutate(self, current_text, feedback, temperature):
        prompt = _MUTATE_PROMPT.format(current_text=current_text, feedback=feedback[:2000])
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or current_text
