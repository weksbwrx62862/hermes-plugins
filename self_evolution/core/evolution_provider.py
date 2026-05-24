"""
自进化插件核心接口定义 (EvolutionProvider ABC)

设计原则：
  - 7个必须实现的方法 + 5个可选方法
  - 遵循 s24 插件架构模式：接口→协调器→配置加载
  - 同一时间只有一个 EvolutionProvider 激活
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ── 数据模型 ───────────────────────────────────────────────

class EvolutionPhase(Enum):
    """四层进化目标"""
    SKILL = auto()           # Phase 1: skill 文件文本优化（最高价值/最低风险）
    TOOL_DESC = auto()       # Phase 2: 工具描述优化（中等价值/低风险）
    PROMPT_SECTION = auto()  # Phase 3: 系统提示词优化（高价值/高风险）
    TOOL_CODE = auto()       # Phase 4: 工具代码进化（高价值/最高风险）


class PipelineStage(Enum):
    """8阶段管线"""
    DETECT = 1       # 检测触发条件（自动/手动）
    SELECT = 2       # 选择优化目标 skill/tool/prompt
    BUILD = 3        # 生成评估数据集（s26 Layer 1）
    BASELINE = 4     # 评估基准版本（s26 Layer 2）
    CONSTRAINTS = 5  # 约束门控检查（s26 Layer 3）
    OPTIMIZE = 6     # 优化循环：feedback→mutate→evaluate→select
    VALIDATE = 7     # 最终验证：holdout + 约束
    DEPLOY = 8       # 备份→写入→生效


@dataclass
class EvalExample:
    """评估用例"""
    task_input: str
    expected_behavior: str  # rubric 描述，不是精确匹配
    source: str = "synthetic"  # synthetic | session | golden | boundary


@dataclass
class EvalDataset:
    """评估数据集 (60/20/20 分割)"""
    train: list[EvalExample] = field(default_factory=list)
    val: list[EvalExample] = field(default_factory=list)
    holdout: list[EvalExample] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.holdout)

    @property
    def source_summary(self) -> dict[str, int]:
        """返回各数据源的用例数量。"""
        counts: dict[str, int] = {}
        for ex in self.train + self.val + self.holdout:
            counts[ex.source] = counts.get(ex.source, 0) + 1
        return counts


@dataclass
class FitnessScore:
    """适应度分数 — 多维打分"""
    correctness: float = 0.0          # 正确性 40%
    procedure_following: float = 0.0  # 步骤遵循度 30%
    conciseness: float = 0.0          # 简洁性 20%
    skill_clarity: float = 0.0        # 技能清晰度 10%
    feedback: str = ""                # 驱动优化的改进建议
    failure_cause: str = "none"       # skill_defect | execution_lapse | none

    @property
    def composite(self) -> float:
        """加权综合分: 0.4*c + 0.3*p + 0.2*con + 0.1*clarity"""
        return (0.4 * self.correctness + 0.3 * self.procedure_following
                + 0.2 * self.conciseness + 0.1 * self.skill_clarity)


@dataclass
class ConstraintResult:
    """约束门控结果 — 8维 AND 逻辑"""
    passed: bool = True
    checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class EvolutionResult:
    """一次自进化的完整结果"""
    target_name: str
    phase: EvolutionPhase
    baseline_score: float
    evolved_score: float
    improvement: float                 # evolved - baseline
    iterations_used: int
    constraint_passed: bool
    constraint_details: ConstraintResult
    holdout_score: float
    evolved_text: str
    deployed: bool = False
    error: Optional[str] = None
    cross_model_score: float = 0.0
    audit_report: dict = field(default_factory=dict)


# ── EvolutionProvider ABC 接口 ─────────────────────────────

class EvolutionProvider(ABC):
    """
    自进化提供者接口。

    必须实现 (7):
      name, is_available, initialize, get_phase,
      build_dataset, evaluate, validate_constraints

    可选实现 (5):
      detect_trigger, optimize, deploy, handle_rollback, shutdown
    """

    # ── 必须实现 ──────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """短标识符，如 'default-evolver'、'gepa-evolver'"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检查是否可用（依赖、配置）。不做网络请求。"""
        ...

    @abstractmethod
    def initialize(self, target_dir: str, **kwargs) -> None:
        """
        初始化进化器。
        Args:
            target_dir: hermes-agent 仓库根目录
            **kwargs: 可选配置（model, iterations, etc.）
        """
        ...

    @abstractmethod
    def get_phase(self) -> EvolutionPhase:
        """返回当前提供者负责的进化阶段"""
        ...

    @abstractmethod
    def build_dataset(self, skill_text: str, num_cases: int = 15) -> EvalDataset:
        """
        生成评估数据集 (3层评测 - Layer 1)。
        LLM 读 skill 文本 → 生成 (task, rubric) 测试用例。
        """
        ...

    @abstractmethod
    def evaluate(self, skill_text: str, example: EvalExample) -> FitnessScore:
        """
        适应度打分 (3层评测 - Layer 2)。
        两步: agent执行 → LLM-as-judge打分。
        """
        ...

    @abstractmethod
    def validate_constraints(
        self,
        skill_text: str,
        baseline: Optional[str] = None,
        **kwargs,
    ) -> ConstraintResult:
        """
        约束门控检查 (3层评测 - Layer 3)。
        8维 AND 门控：全部通过才算通过。

        kwargs 可包含:
          cross_model_score: 跨模型验证分数
          baseline_cross_model: 基线跨模型分数
          train_score: 训练集分数
          val_score: 验证集分数
          audit_report: 审计报告
        """
        ...

    # ── 可选实现 ──────────────────────────────────────────

    def detect_trigger(self, context: dict) -> bool:
        """检测是否触发自进化（Phase 5 自动触发）。"""
        return False

    def optimize(
        self,
        skill_text: str,
        dataset: EvalDataset,
        iterations: int = 5,
    ) -> tuple[str, float, int, dict]:
        """
        优化循环：feedback → mutate → evaluate → select。
        重复 N 次，连续 improvement=0 则提前停止。
        Returns: (evolved_text, best_score, iterations_used, audit_report)
        """
        return skill_text, 0.0, 0, {}

    def deploy(self, target_path: str, evolved_text: str) -> bool:
        """备份原文件 → 写入进化版。"""
        return False

    def handle_rollback(self, target_path: str) -> bool:
        """回滚到最近备份。"""
        return False

    def shutdown(self) -> None:
        """清理资源。"""
        pass
