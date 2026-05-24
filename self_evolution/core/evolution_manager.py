"""
自进化协调器 (EvolutionManager)

职责：
  1. 管理提供者注册（同一 Phase 最多一个）
  2. 编排 8 阶段管线
  3. 错误隔离（一个提供者失败不影响其他）
  4. 提供 CLI 入口 evolve_skill()

设计参考：s24 的 MemoryManager 协调器模式
"""

import logging
import time
from pathlib import Path
from typing import Optional

from .evolution_provider import (
    EvolutionProvider, EvolutionPhase, PipelineStage,
    EvalDataset, EvolutionResult, ConstraintResult, FitnessScore,
)

logger = logging.getLogger(__name__)


class EvolutionManager:
    """
    协调器：管理自进化提供者 + 编排管线。

    关键设计：
      - 同一 Phase 只有一个提供者激活
      - 内置失败不影响其他 Phase
      - 约束门控是硬门控 — 不过就拒绝
    """

    def __init__(self):
        self._providers: dict[EvolutionPhase, EvolutionProvider] = {}
        self._target_dir: Optional[str] = None

    # ── 注册 ───────────────────────────────────────────────

    def add_provider(self, provider: EvolutionProvider) -> None:
        """注册一个进化提供者。同一 Phase 冲突时拒绝。"""
        phase = provider.get_phase()
        if phase in self._providers:
            existing = self._providers[phase].name
            logger.warning(
                f"[evolution] Phase {phase.name} already has provider '{existing}', "
                f"rejecting '{provider.name}'"
            )
            return
        self._providers[phase] = provider
        logger.info(f"[evolution] registered provider '{provider.name}' for {phase.name}")

    def initialize_all(self, target_dir: str, **kwargs) -> None:
        """初始化所有提供者。"""
        self._target_dir = target_dir
        for phase, provider in self._providers.items():
            try:
                if provider.is_available():
                    provider.initialize(target_dir, **kwargs)
                    logger.info(f"[evolution] initialized {provider.name}")
                else:
                    logger.warning(f"[evolution] {provider.name} not available, skipping")
            except Exception as exc:
                logger.error(f"[evolution] failed to init {provider.name}: {exc}")

    # ── 路由 ───────────────────────────────────────────────

    def get_provider(self, phase: EvolutionPhase) -> Optional[EvolutionProvider]:
        return self._providers.get(phase)

    def list_providers(self) -> list[str]:
        return [f"{phase.name}: {p.name}" for phase, p in self._providers.items()]

    # ── 8 阶段管线 ─────────────────────────────────────────

    def evolve_skill(
        self,
        skill_name: str,
        phase: EvolutionPhase = EvolutionPhase.SKILL,
        iterations: int = 5,
        auto_deploy: bool = False,
    ) -> EvolutionResult:
        """
        8 阶段完整管线入口。

        阶段 1-2: 准备
        阶段 3-5: 3 层评测（s26）
        阶段 6-7: 优化验证（s27）
        阶段 8: 部署
        """

        provider = self.get_provider(phase)
        if not provider:
            return EvolutionResult(
                target_name=skill_name,
                phase=phase,
                baseline_score=0,
                evolved_score=0,
                improvement=0,
                iterations_used=0,
                constraint_passed=False,
                constraint_details=ConstraintResult(),
                holdout_score=0,
                evolved_text="",
                error=f"No provider for phase {phase.name}",
            )

        t_start = time.time()

        # ── SurrogateVerifier 初始化（静默降级）──────────
        surrogate = None
        try:
            from self_evolution.core.surrogate_verifier import SurrogateVerifier
            surrogate = SurrogateVerifier()
        except Exception as exc:
            logger.debug(f"[evolve:{skill_name}] SurrogateVerifier 初始化失败，静默降级: {exc}")

        # ── 阶段 1: DETECT ──────────────────────────────
        logger.info(f"[evolve:{skill_name}] STAGE 1/8 DETECT — checking trigger")
        if not self._should_evolve(provider, skill_name):
            return EvolutionResult(
                target_name=skill_name, phase=phase,
                baseline_score=0, evolved_score=0, improvement=0,
                iterations_used=0, constraint_passed=False,
                constraint_details=ConstraintResult(),
                holdout_score=0, evolved_text="",
                error="Trigger condition not met",
            )

        # ── 阶段 2: SELECT ──────────────────────────────
        logger.info(f"[evolve:{skill_name}] STAGE 2/8 SELECT — loading target")
        skill_text, skill_path = self._load_target(skill_name, phase)
        if not skill_text:
            return EvolutionResult(
                target_name=skill_name, phase=phase,
                baseline_score=0, evolved_score=0, improvement=0,
                iterations_used=0, constraint_passed=False,
                constraint_details=ConstraintResult(),
                holdout_score=0, evolved_text="",
                error=f"Skill not found: {skill_name}",
            )

        # ── 阶段 3: BUILD — Layer 1: 评估数据集 ────────
        logger.info(f"[evolve:{skill_name}] STAGE 3/8 BUILD — generating eval dataset")
        try:
            dataset = provider.build_dataset(skill_text, num_cases=15)
            logger.info(
                f"[evolve:{skill_name}] dataset: "
                f"{len(dataset.train)}t/{len(dataset.val)}v/{len(dataset.holdout)}h"
            )
        except Exception as exc:
            return self._error_result(skill_name, phase, f"Build dataset failed: {exc}")

        # ── 阶段 4: BASELINE — Layer 2: 适应度打分 ──────
        logger.info(f"[evolve:{skill_name}] STAGE 4/8 BASELINE — evaluating current version")
        try:
            baseline_score = self._score_on_dataset(provider, skill_text, dataset.val)
            logger.info(f"[evolve:{skill_name}] baseline score: {baseline_score:.3f}")
        except Exception as exc:
            return self._error_result(skill_name, phase, f"Baseline eval failed: {exc}")

        # ── 阶段 5: CONSTRAINTS — Layer 3: 8维 AND 门控 ─
        logger.info(f"[evolve:{skill_name}] STAGE 5/8 CONSTRAINTS — 8-dimension AND gate")
        try:
            constraints = provider.validate_constraints(skill_text)
            if not constraints.passed:
                logger.warning(
                    f"[evolve:{skill_name}] baseline constraint FAILED: {constraints.failures}"
                )
                return EvolutionResult(
                    target_name=skill_name, phase=phase,
                    baseline_score=baseline_score, evolved_score=baseline_score,
                    improvement=0, iterations_used=0,
                    constraint_passed=False, constraint_details=constraints,
                    holdout_score=baseline_score, evolved_text=skill_text,
                    error=f"Baseline failed constraints: {', '.join(constraints.failures)}",
                )
            logger.info(f"[evolve:{skill_name}] 8-dimension AND gate: ALL PASSED ✓")
        except Exception as exc:
            return self._error_result(skill_name, phase, f"Constraint check failed: {exc}")

        # ── 阶段 6: OPTIMIZE — feedback→mutate→eval→select ─
        logger.info(f"[evolve:{skill_name}] STAGE 6/8 OPTIMIZE — {iterations} iterations")
        try:
            evolved_text, evolved_score, iters_used, audit_report = provider.optimize(
                skill_text, dataset, iterations=iterations
            )
            train_score = audit_report.get("last_train_score", evolved_score) if audit_report else evolved_score
            improvement = evolved_score - baseline_score
            logger.info(
                f"[evolve:{skill_name}] optimization done: "
                f"{baseline_score:.3f} → {evolved_score:.3f} "
                f"({'+' if improvement >= 0 else ''}{improvement:.3f}), "
                f"{iters_used}/{iterations} iterations"
            )
        except Exception as exc:
            return self._error_result(skill_name, phase, f"Optimization failed: {exc}")

        # ── 阶段 7: VALIDATE — holdout + 跨模型 + 约束 ──
        logger.info(f"[evolve:{skill_name}] STAGE 7/8 VALIDATE — final checks")
        try:
            # 7.1 holdout 最终考试
            holdout_score = self._score_on_dataset(provider, evolved_text, dataset.holdout)
            if holdout_score < baseline_score:
                logger.warning(
                    f"[evolve:{skill_name}] holdout score ({holdout_score:.3f}) "
                    f"< baseline ({baseline_score:.3f}), not deploying"
                )
                holdout_fail = ConstraintResult(
                    passed=False, checks={},
                    failures=[f"holdout_score({holdout_score:.3f}) < baseline({baseline_score:.3f})"]
                )
                self._archive_reject(skill_name, evolved_text, holdout_fail,
                                     baseline_score, evolved_score, iters_used,
                                     audit_report=audit_report)
                return EvolutionResult(
                    target_name=skill_name, phase=phase,
                    baseline_score=baseline_score, evolved_score=evolved_score,
                    improvement=improvement, iterations_used=iters_used,
                    constraint_passed=False, constraint_details=holdout_fail,
                    holdout_score=holdout_score, evolved_text=evolved_text,
                    error="Holdout score regression",
                )
            logger.info(
                f"[evolve:{skill_name}] holdout score: {holdout_score:.3f} "
                f"(baseline: {baseline_score:.3f}) ✓"
            )

            # 7.1.1 Surrogate 校准：用 GT holdout 分数校准
            if surrogate and dataset.holdout:
                try:
                    surrogate.calibrate(evolved_text, dataset.holdout[0], holdout_score)
                except Exception as exc:
                    logger.debug(f"[evolve:{skill_name}] surrogate 校准失败: {exc}")

            # 7.2 跨模型验证（可选）
            cross_model_score = 0.0
            baseline_cross_model = 0.0
            if dataset.holdout:
                try:
                    from self_evolution.core.fitness import evaluate_cross_model
                    cross_model_score = evaluate_cross_model(
                        evolved_text, dataset.holdout[0],
                        client=None, model=None,
                    )
                    if cross_model_score > 0:
                        baseline_cross_model = evaluate_cross_model(
                            skill_text, dataset.holdout[0],
                            client=None, model=None,
                        )
                except Exception as exc:
                    logger.debug(f"[evolve:{skill_name}] cross-model eval skipped: {exc}")
                    cross_model_score = 0.0
                    baseline_cross_model = 0.0

            if cross_model_score > 0:
                logger.info(
                    f"[evolve:{skill_name}] cross-model score: {cross_model_score:.3f} "
                    f"(baseline: {baseline_cross_model:.3f})"
                )

            # 7.3 约束门控（含跨模型稳定性维度）
            evolved_constraints = provider.validate_constraints(
                evolved_text,
                baseline=skill_text,
                cross_model_score=cross_model_score if cross_model_score > 0 else None,
                baseline_cross_model=baseline_cross_model if baseline_cross_model > 0 else None,
                train_score=baseline_score,
                val_score=evolved_score,
                audit_report=audit_report,
            )
            if not evolved_constraints.passed:
                logger.warning(
                    f"[evolve:{skill_name}] evolved constraint FAILED: "
                    f"{evolved_constraints.failures}"
                )
                self._archive_reject(skill_name, evolved_text, evolved_constraints,
                                     baseline_score, evolved_score, iters_used,
                                     audit_report=audit_report)
                return EvolutionResult(
                    target_name=skill_name, phase=phase,
                    baseline_score=baseline_score, evolved_score=evolved_score,
                    improvement=improvement, iterations_used=iters_used,
                    constraint_passed=False, constraint_details=evolved_constraints,
                    holdout_score=holdout_score, evolved_text=evolved_text,
                    cross_model_score=cross_model_score,
                    error=f"Evolved failed constraints: {', '.join(evolved_constraints.failures)}",
                )
        except Exception as exc:
            return self._error_result(skill_name, phase, f"Validation failed: {exc}")

        # ── 评测保真度报告 ──────────────────────────────
        if surrogate:
            if audit_report is None:
                audit_report = {}
            audit_report["verifier_fidelity"] = surrogate.stats.fidelity
            audit_report["verifier_gap"] = surrogate.stats.last_gap
            audit_report["verifier_needs_retrain"] = surrogate.needs_retrain()

        # ── 阶段 8: DEPLOY ─────────────────────────────
        logger.info(f"[evolve:{skill_name}] STAGE 8/8 DEPLOY")
        deployed = False
        if auto_deploy and improvement > 0:
            try:
                deployed = provider.deploy(str(skill_path), evolved_text)
                if deployed:
                    logger.info(f"[evolve:{skill_name}] ✓ deployed (improvement: {improvement:+.3f})")
                else:
                    logger.warning(f"[evolve:{skill_name}] deploy returned False")
            except Exception as exc:
                logger.error(f"[evolve:{skill_name}] deploy failed: {exc}")
        elif improvement <= 0:
            logger.info(f"[evolve:{skill_name}] no improvement, skipping deploy")

        elapsed = time.time() - t_start
        logger.info(f"[evolve:{skill_name}] pipeline completed in {elapsed:.1f}s")

        # ── Surrogate 重训练检查 ─────────────────────────
        if surrogate and surrogate.needs_retrain():
            try:
                surrogate.retrain()
            except Exception as exc:
                logger.debug(f"[evolve:{skill_name}] surrogate 重训练失败: {exc}")

        return EvolutionResult(
            target_name=skill_name,
            phase=phase,
            baseline_score=baseline_score,
            evolved_score=evolved_score,
            improvement=improvement,
            iterations_used=iters_used,
            constraint_passed=evolved_constraints.passed,
            constraint_details=evolved_constraints,
            holdout_score=holdout_score,
            evolved_text=evolved_text,
            deployed=deployed,
            cross_model_score=cross_model_score,
            audit_report=audit_report,
        )

    # ── 内部辅助 ──────────────────────────────────────────

    def _should_evolve(self, provider: EvolutionProvider, skill_name: str) -> bool:
        """检查是否应该触发进化（阶段 1）。"""
        context = {"skill_name": skill_name, "auto": False}
        return provider.detect_trigger(context) or True  # CLI 模式默认允许

    def _load_target(self, skill_name: str, phase: EvolutionPhase) -> tuple[Optional[str], Optional[Path]]:
        """加载目标 skill/tool/prompt 文本（阶段 2）。"""
        if self._target_dir is None:
            return None, None

        targets_dir = Path(self._target_dir)
        match phase:
            case EvolutionPhase.SKILL:
                skill_path = targets_dir / "skills" / skill_name / "SKILL.md"
                if skill_path.exists():
                    return skill_path.read_text(), skill_path
                # 也找 ~/.hermes/skills/
                alt_path = Path.home() / ".hermes" / "skills" / skill_name / "SKILL.md"
                if alt_path.exists():
                    return alt_path.read_text(), alt_path
            case _:
                pass
        return None, None

    def _score_on_dataset(
        self, provider: EvolutionProvider, skill_text: str, examples: list
    ) -> float:
        """在数据集上计算平均适应度分数。"""
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            try:
                fitness = provider.evaluate(skill_text, ex)
                scores.append(fitness.composite)
            except Exception as exc:
                logger.debug(f"eval error on example: {exc}")
        return sum(scores) / len(scores) if scores else 0.0

    def _error_result(self, skill_name: str, phase: EvolutionPhase, error: str) -> EvolutionResult:
        logger.error(f"[evolve:{skill_name}] ERROR: {error}")
        return EvolutionResult(
            target_name=skill_name, phase=phase,
            baseline_score=0, evolved_score=0, improvement=0,
            iterations_used=0, constraint_passed=False,
            constraint_details=ConstraintResult(),
            holdout_score=0, evolved_text="",
            error=error,
        )

    # ── 回滚归档 ──────────────────────────────────────────

    REJECT_DIR = Path.home() / ".hermes" / ".evolution_rejects"

    def _archive_reject(
        self, skill_name: str, evolved_text: str,
        constraints: ConstraintResult,
        baseline_score: float, evolved_score: float, iterations: int,
        audit_report: dict = None,
    ) -> Path:
        """保存被门控拒绝的版本到归档目录（用于审计回滚）。"""
        import time as _time
        self.REJECT_DIR.mkdir(parents=True, exist_ok=True)
        ts = _time.strftime("%Y%m%d_%H%M%S")
        path = self.REJECT_DIR / f"{skill_name}_{ts}.md"

        report_lines = [
            f"Evolution REJECT — {skill_name}",
            f"Time: {ts}",
            f"Baseline: {baseline_score:.3f} → Evolved: {evolved_score:.3f}",
            f"Iterations: {iterations}",
            f"Failures: {', '.join(constraints.failures)}",
        ]

        if audit_report:
            if "verifier_fidelity" in audit_report:
                report_lines.append(f"评测保真度: {audit_report['verifier_fidelity']:.3f}")
                report_lines.append(f"Surrogate-GT Gap: {audit_report['verifier_gap']:.3f}")
                if audit_report.get("verifier_needs_retrain"):
                    report_lines.append("⚠️ Surrogate 需要重训练")

        header = "<!--\n" + "\n".join(report_lines) + "\n-->\n"
        path.write_text(header + evolved_text)
        logger.info(f"[evolve:{skill_name}] rejected version archived: {path}")
        return path

    @classmethod
    def get_reject_history(cls, skill_name: str = None) -> list[dict]:
        """查询 reject 历史。可选按 skill_name 过滤。"""
        if not cls.REJECT_DIR.exists():
            return []
        results = []
        import re as _re
        for f in sorted(cls.REJECT_DIR.glob("*.md"), reverse=True):
            if skill_name and not f.name.startswith(skill_name + "_"):
                continue
            text = f.read_text()
            m = _re.search(r'<!--\n(.*?)\n-->', text, _re.DOTALL)
            if m:
                meta_str = m.group(1)
                entry = {"file": str(f.name), "skill_name": skill_name or ""}
                for line in meta_str.strip().split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        entry[k.strip().lower().replace(" ", "_")] = v.strip()
                results.append(entry)
        return results

    def shutdown_all(self) -> None:
        for provider in self._providers.values():
            try:
                provider.shutdown()
            except Exception:
                pass
        self._providers.clear()
