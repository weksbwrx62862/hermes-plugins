"""
默认自进化提供者 (DefaultEvolutionProvider)

完整实现 EvolutionProvider 接口，覆盖 Phase 1 (Skill 进化)。

依赖：
  - openai (LLM 调用)
  - dataset_builder (Layer 1: 数据集)
  - fitness (Layer 2: 适应度)
  - constraints (Layer 3: 5维AND门控)
  - optimizer (阶段6: 优化循环)
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from self_evolution.core.evolution_provider import (
    EvolutionProvider, EvolutionPhase,
    EvalDataset, EvalExample, FitnessScore, ConstraintResult,
)
from self_evolution.core.dataset_builder import (
    SyntheticDatasetBuilder, SessionDatasetBuilder,
    BoundaryProbeBuilder, GoldenDatasetLoader,
)
from self_evolution.core.fitness import evaluate_skill, quick_fitness
from self_evolution.core.constraints import ConstraintValidator
from self_evolution.core.surrogate_verifier import SurrogateVerifier
from self_evolution.pipeline.base_optimizer import SkillOptimizerBase, OptimizeResult
from self_evolution.pipeline.optimizer import SkillOptimizer

logger = logging.getLogger(__name__)


class DefaultEvolutionProvider(EvolutionProvider):
    """
    默认自进化实现 — Phase 1 (Skill 文本优化)。

    配置参数：
      - model: 用于数据集生成、打分、变异的模型
      - iterations: 优化迭代次数 (默认 5)
      - use_llm_eval: 是否使用完整 LLM-as-judge（True 慢但准，False 快但粗）
      - auto_deploy: 是否自动部署通过验证的版本
      - optimizer_name: 优化策略名称 (默认 "diversify")
    """

    _optimizer_registry: dict[str, type[SkillOptimizerBase]] = {}

    def __init__(self, optimizer_name: str = "diversify"):
        self._available = True
        self._client: Optional[OpenAI] = None
        self._model = "gpt-4o-mini"
        self._target_dir = ""
        self._iterations = 5
        self._use_llm_eval = True
        self.optimizer_name = optimizer_name

        # 子组件
        self._dataset_builder: Optional[SyntheticDatasetBuilder] = None
        self._constraints: Optional[ConstraintValidator] = None
        self._optimizer: Optional[SkillOptimizer] = None
        self._surrogate_verifier: Optional[SurrogateVerifier] = None

    @classmethod
    def add_optimizer(cls, name: str, optimizer_cls: type[SkillOptimizerBase]):
        """注册自定义优化器。"""
        cls._optimizer_registry[name] = optimizer_cls

    def _get_optimizer(self) -> SkillOptimizerBase:
        """获取当前优化器实例。"""
        builtin = {
            "gepa": "self_evolution.pipeline.gepa_optimizer.GEPAOptimizer",
            "diversify": "self_evolution.pipeline.diversify_optimizer.DiversifyOptimizer",
        }
        name = self.optimizer_name

        if name in self._optimizer_registry:
            logger.info(f"[default-evolver._get_optimizer] 使用注册优化器: {name}")
            return self._optimizer_registry[name](client=self._client, model=self._model)

        if name in builtin:
            logger.info(f"[default-evolver._get_optimizer] 使用内置优化器: {name}")
            module_path, cls_name = builtin[name].rsplit(".", 1)
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            return cls(client=self._client, model=self._model)

        logger.warning(f"未知优化器 '{name}'，使用 diversify")
        from self_evolution.pipeline.diversify_optimizer import DiversifyOptimizer
        return DiversifyOptimizer(client=self._client, model=self._model)

    # ── 必须实现 ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "default-evolver"

    def is_available(self) -> bool:
        try:
            import openai
            return self._available
        except ImportError:
            logger.warning("[default-evolver] openai not installed")
            return False

    def initialize(self, target_dir: str, **kwargs) -> None:
        self._target_dir = target_dir
        self._model = kwargs.get("model", "gpt-4o-mini")
        self._iterations = kwargs.get("iterations", 5)
        self._use_llm_eval = kwargs.get("use_llm_eval", True)

        # 初始化 OpenAI client
        api_key = kwargs.get("api_key")
        base_url = kwargs.get("base_url")
        self._client = OpenAI(api_key=api_key, base_url=base_url) if api_key else OpenAI()

        # 初始化子组件
        self._dataset_builder = SyntheticDatasetBuilder(self._client, self._model)
        self._constraints = ConstraintValidator(self._client)
        self._optimizer = SkillOptimizer(
            self._client, self._model, use_llm=self._use_llm_eval
        )
        self._surrogate_verifier = SurrogateVerifier(
            client=self._client,
            model=self._model,
        )

        logger.info(f"[default-evolver] initialized: model={self._model}, "
                    f"iterations={self._iterations}, llm_eval={self._use_llm_eval}")

    def get_phase(self) -> EvolutionPhase:
        return EvolutionPhase.SKILL

    def build_dataset(self, skill_text: str, num_cases: int = 15) -> EvalDataset:
        if not self._dataset_builder:
            raise RuntimeError("Provider not initialized")

        logger.info(f"[build_dataset] 开始构建混合数据集 | num_cases={num_cases}")

        dataset = self._dataset_builder.generate(skill_text, num_cases)

        try:
            boundary_builder = BoundaryProbeBuilder(self._client, self._model)
            boundary_examples = boundary_builder.generate(skill_text, num_cases=10)
            dataset.train.extend(boundary_examples)
            logger.info(f"[build_dataset] 边界探测: +{len(boundary_examples)} → train")
        except Exception:
            logger.debug("[build_dataset] 边界探测生成失败，跳过", exc_info=True)

        try:
            session_builder = SessionDatasetBuilder(self._client, self._model)
            skill_name = skill_text.split('\n')[0][:50]
            session_examples = session_builder.generate(skill_name, max_sessions=10)
            dataset.val.extend(session_examples)
            logger.info(f"[build_dataset] SessionDB: +{len(session_examples)} → val")
        except Exception:
            logger.debug("[build_dataset] SessionDB 加载失败，跳过", exc_info=True)

        try:
            golden_loader = GoldenDatasetLoader()
            skill_name = self._extract_skill_name(skill_text)
            golden_examples = golden_loader.load(skill_name)
            dataset.holdout.extend(golden_examples)
            logger.info(f"[build_dataset] Golden: +{len(golden_examples)} → holdout")
        except Exception:
            logger.debug("[build_dataset] Golden 测试集加载失败，跳过", exc_info=True)

        logger.info(f"[build_dataset] 数据源统计: {dataset.source_summary}")
        return dataset

    @staticmethod
    def _extract_skill_name(skill_text: str) -> str:
        """从技能文本中提取技能名称。"""
        for line in skill_text.split('\n')[:5]:
            line = line.strip()
            if line.startswith('name:'):
                return line[5:].strip().strip('"').strip("'")
            if line.startswith('# '):
                return line[2:].strip()
        return skill_text.split('\n')[0][:50]

    def evaluate(self, skill_text: str, example: EvalExample) -> FitnessScore:
        if not self._client:
            raise RuntimeError("Provider not initialized")
        if self._use_llm_eval:
            logger.info(f"[evaluate] LLM-as-judge 模式 | task={example.task_input[:60]}...")
            return evaluate_skill(skill_text, example, self._client, self._model)
        logger.info(f"[evaluate] quick_fitness 模式 | task={example.task_input[:60]}...")
        return quick_fitness(skill_text, example)

    def validate_constraints(
        self, skill_text: str, baseline: Optional[str] = None, **kwargs
    ) -> ConstraintResult:
        if not self._constraints:
            raise RuntimeError("Provider not initialized")
        return self._constraints.validate_all(skill_text, baseline, **kwargs)

    # ── 可选实现 ──────────────────────────────────────────

    def detect_trigger(self, context: dict) -> bool:
        # CLI 模式：始终允许
        if context.get("mode") == "cli":
            return True
        # 自动模式：检查使用频率
        return super().detect_trigger(context)

    def optimize(
        self,
        skill_text: str,
        dataset: EvalDataset,
        iterations: int = 5,
        use_bayesian: bool = False,
        bayesian_init_points: int = 5,
        bayesian_patience: int = 10,
    ) -> tuple[str, float, int, dict]:
        """
        优化技能。
        
        参数:
            skill_text: 原始技能文本
            dataset: 评估数据集
            iterations: 优化迭代次数
            use_bayesian: 是否使用贝叶斯优化（默认 False，使用传统优化器）
            bayesian_init_points: 贝叶斯优化初始探索点数
            bayesian_patience: 贝叶斯优化早停耐心值
        """
        optimizer = self._get_optimizer()
        
        # 如果启用贝叶斯优化，使用贝叶斯优化器
        if use_bayesian:
            try:
                from self_evolution.pipeline.bayesian_optimizer import BayesianSkillOptimizer, create_skill_eval_fn
                
                logger.info("[optimize] 使用贝叶斯优化 | init_points=%d | patience=%d", 
                           bayesian_init_points, bayesian_patience)
                
                # 创建评估函数
                def eval_fn(**params):
                    """贝叶斯优化评估函数"""
                    # 使用 LLM 生成优化版本
                    temperature = params.get('temperature', 0.7)
                    # 这里可以扩展更多参数
                    return optimizer.score(skill_text, dataset.examples)
                
                # 创建贝叶斯优化器
                bayesian_optimizer = BayesianSkillOptimizer(
                    eval_fn=eval_fn,
                    verbose=0,  # 静默模式
                )
                
                # 定义参数空间
                pbounds = {
                    'temperature': (0.1, 1.0),
                }
                
                # 执行贝叶斯优化
                best_params, best_score, iterations_used = bayesian_optimizer.optimize_with_early_stopping(
                    pbounds=pbounds,
                    init_points=bayesian_init_points,
                    n_iter=iterations,
                    patience=bayesian_patience,
                )
                
                logger.info("[optimize] 贝叶斯优化完成 | best_params=%s | best_score=%.3f | iterations=%d",
                           best_params, best_score, iterations_used)
                
                # 使用最优参数生成最终版本
                # 这里可以根据 best_params 调整技能文本
                evolved_text = skill_text  # 简化示例
                
                audit_report = {
                    'bayesian_optimization': True,
                    'best_params': best_params,
                    'iterations_used': iterations_used,
                }
                
                return (evolved_text, best_score, iterations_used, audit_report)
                
            except Exception as e:
                logger.warning("[optimize] 贝叶斯优化失败，降级到传统优化: %s", e)
                # 降级到传统优化
        
        # 传统优化
        logger.info(f"[optimize] 开始优化 | optimizer={optimizer.name} | iterations={iterations} | dataset={dataset.source_summary}")
        result = optimizer.optimize(skill_text, dataset, iterations)
        logger.info(
            f"[optimize] 优化完成 | score={result.best_score:.3f} | "
            f"iterations_used={result.iterations_used} | audit_keys={list(result.audit_report.keys())}"
        )
        return (result.evolved_text, result.best_score, result.iterations_used, result.audit_report)

    def deploy(self, target_path: str, evolved_text: str) -> bool:
        """备份原文件 → 写入进化版。"""
        try:
            path = Path(target_path)

            # 阶段 8.1: 备份
            backup_dir = path.parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{path.stem}_{timestamp}.bak"
            shutil.copy2(path, backup_path)
            logger.info(f"[deploy] backed up to {backup_path}")

            # 阶段 8.2: 写入
            path.write_text(evolved_text, encoding="utf-8")
            logger.info(f"[deploy] wrote evolved skill to {path}")

            return True
        except Exception as exc:
            logger.error(f"[deploy] failed: {exc}")
            return False

    def handle_rollback(self, target_path: str) -> bool:
        """回滚到最近备份。"""
        try:
            path = Path(target_path)
            backup_dir = path.parent / "backups"
            if not backup_dir.exists():
                logger.warning("[rollback] no backup directory")
                return False

            backups = sorted(
                backup_dir.glob(f"{path.stem}_*.bak"),
                reverse=True,
            )
            if not backups:
                logger.warning(f"[rollback] no backups found for {target_path}")
                return False

            latest = backups[0]
            shutil.copy2(latest, path)
            logger.info(f"[rollback] restored from {latest}")
            return True
        except Exception as exc:
            logger.error(f"[rollback] failed: {exc}")
            return False

    def shutdown(self) -> None:
        self._client = None
        self._dataset_builder = None
        self._constraints = None
        self._optimizer = None
        self._surrogate_verifier = None
