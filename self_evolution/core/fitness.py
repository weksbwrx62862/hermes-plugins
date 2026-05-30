"""
3层评测 — Layer 2: 适应度函数 (FitnessScore + evaluate_skill)

两步流程：
  1. Agent 执行：用 skill 做 task_input 任务，得到 agent_output
  2. LLM-as-judge 打分：skill + task + rubric + agent_output → 多维分数 + feedback

快速启发式模式：
  keyword overlap 代理评分（加速优化循环，最终评估切回 LLM）

v2 改进：
  - 区分四种反思类型（技能缺陷 vs 执行失误）
  - 失败后先判断"是手册写错了，还是我没按手册做"
  - 只有技能缺陷才触发变异，执行失误不改技能
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from openai import OpenAI

from self_evolution.core.evolution_provider import FitnessScore, EvalExample

logger = logging.getLogger(__name__)

# ── 模型读取：自动获取当前活跃模型 ──────────────────────────

def _get_active_model(default: str = "deepseek-v4-pro") -> str:
    """从 Hermes config.yaml 读取当前活跃模型，找不到则用 default。"""
    config_path = Path.home() / ".hermes" / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            router = cfg.get("model_router", {})
            strategy = router.get("strategy", "smartest")
            providers = cfg.get("providers", {})
            for name, pdata in providers.items():
                if not isinstance(pdata, dict):
                    continue
                models = pdata.get("models", [])
                if models:
                    return models[0]
            pools = router.get("pools", {})
            if strategy in pools:
                pool_models = pools[strategy]
                if pool_models:
                    return pool_models[0].get("model", default)
        except Exception:
            pass
    return default


# ── 四种反思类型 ──────────────────────────────────────────────

@dataclass
class ReflectionResult:
    """反思结果"""
    reflection_type: str  # discovery | optimization | skill_defect | execution_lapse
    should_mutate: bool   # 是否需要修改技能
    feedback: str         # 具体反馈
    confidence: float     # 0.0-1.0，判断置信度


# ── 反思提示词 ─────────────────────────────────────────────────

_REFLECTION_PROMPT = """你是技能反思专家。Agent 使用下面的技能执行任务后，请判断失败原因属于哪种类型。

技能文件：
---
{skill_text}
---

用户任务：
{task_input}

期望行为：
{expected_behavior}

Agent 的输出：
---
{agent_output}
---

请判断属于以下四种反思类型之一：

1. **discovery** — 任务成功，但发现了新技能的苗头（记下来）
2. **optimization** — 任务成功，但效率不高（优化已有技能）
3. **skill_defect** — 任务失败，是技能本身有缺陷（需要改技能）
4. **execution_lapse** — 任务失败，但技能没问题，是执行环节出了岔子（不改技能）

关键判断标准：
- 如果 Agent 输出与技能描述一致但结果错误 → skill_defect（技能写错了）
- 如果 Agent 输出明显偏离技能描述 → execution_lapse（没按手册做）
- 如果部分正确部分错误 → 分析哪部分是技能问题，哪部分是执行问题

返回 JSON：
```json
{{
  "reflection_type": "skill_defect",
  "should_mutate": true,
  "feedback": "具体改进建议...",
  "confidence": 0.9
}}
```
只返回 JSON，不要解释。"""


# ── LLM-as-judge 提示词（v2 四维评估）─────────────────────────

_JUDGE_PROMPT = """你是技能评估专家。评估 agent 使用下面技能执行任务的表现。

技能文件：
---
{skill_text}
---

用户任务：
{task_input}

期望行为 (rubric)：
{expected_behavior}

Agent 的输出：
---
{agent_output}
---

评估四个维度并返回 JSON：

1. **correctness** (0.0-1.0，权重 0.4): 输出是否达到了 rubric 的核心要求？
2. **procedure_following** (0.0-1.0，权重 0.3): 是否按照技能描述的步骤执行？
3. **conciseness** (0.0-1.0，权重 0.2): 输出是否简洁？
4. **skill_clarity** (0.0-1.0，权重 0.1): 技能描述是否清晰、完整、无歧义？

5. **feedback** (string): **对技能文本本身的改进建议**。
6. **failure_cause** (string): 如果失败，是 "skill_defect"（技能写错）还是 "execution_lapse"（没按手册做）还是 "none"（成功）。

返回 JSON：
```json
{{
  "correctness": 0.8,
  "procedure_following": 0.7,
  "conciseness": 0.9,
  "skill_clarity": 0.8,
  "feedback": "技能缺少错误处理说明...",
  "failure_cause": "skill_defect"
}}
```
只返回 JSON，不要解释。"""


def evaluate_skill(
    skill_text: str,
    example: EvalExample,
    client: OpenAI,
    model: str = None,
) -> FitnessScore:
    """LLM-as-judge 评估：agent 执行 → 打分（含反思类型判断）

    model=None 时自动从 Hermes config 读取当前活跃模型。"""
    if model is None:
        model = _get_active_model()

    logger.info(f"[evaluate_skill] 开始评估 | model={model} | task={example.task_input[:60]}... | source={example.source}")

    # 步骤 1: Agent 执行任务
    agent_output = _simulate_agent(skill_text, example.task_input, client, model)
    logger.info(f"[evaluate_skill] Agent 执行完成 | output_len={len(agent_output)}")

    # 步骤 2: LLM-as-judge 打分
    prompt = _JUDGE_PROMPT.format(
        skill_text=skill_text[:3000],
        task_input=example.task_input,
        expected_behavior=example.expected_behavior,
        agent_output=agent_output[:3000],
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    raw_content = response.choices[0].message.content or "{}"
    result = _parse_json(raw_content)

    if not result:
        logger.warning(f"[evaluate_skill] JSON 解析失败，raw={raw_content[:200]}")

    fitness = FitnessScore(
        correctness=result.get("correctness", 0),
        procedure_following=result.get("procedure_following", 0),
        conciseness=result.get("conciseness", 0),
        skill_clarity=result.get("skill_clarity", 0),
        feedback=result.get("feedback", ""),
        failure_cause=result.get("failure_cause", "none"),
    )

    logger.info(
        f"[evaluate_skill] 评估完成 | composite={fitness.composite:.3f} | "
        f"c={fitness.correctness:.2f} p={fitness.procedure_following:.2f} "
        f"con={fitness.conciseness:.2f} clarity={fitness.skill_clarity:.2f} | "
        f"failure_cause={fitness.failure_cause}"
    )

    return fitness


def reflect_on_failure(
    skill_text: str,
    example: EvalExample,
    agent_output: str,
    client: OpenAI,
    model: str = None,
) -> ReflectionResult:
    """失败反思 — 判断是技能缺陷还是执行失误。"""
    if model is None:
        model = _get_active_model()

    logger.info(f"[reflect_on_failure] 开始反思 | model={model} | task={example.task_input[:60]}...")

    prompt = _REFLECTION_PROMPT.format(
        skill_text=skill_text[:3000],
        task_input=example.task_input,
        expected_behavior=example.expected_behavior,
        agent_output=agent_output[:3000],
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    raw_content = response.choices[0].message.content or "{}"
    result = _parse_json(raw_content)

    if not result:
        logger.warning(f"[reflect_on_failure] JSON 解析失败，raw={raw_content[:200]}")

    reflection = ReflectionResult(
        reflection_type=result.get("reflection_type", "skill_defect"),
        should_mutate=result.get("should_mutate", True),
        feedback=result.get("feedback", ""),
        confidence=result.get("confidence", 0.5),
    )

    logger.info(
        f"[reflect_on_failure] 反思完成 | type={reflection.reflection_type} | "
        f"should_mutate={reflection.should_mutate} | confidence={reflection.confidence:.2f} | "
        f"feedback={reflection.feedback[:80]}..."
    )

    return reflection


def quick_fitness(
    skill_text: str,
    example: EvalExample,
) -> FitnessScore:
    """
    快速启发式评分（替代 LLM-as-judge）。

    用 keyword overlap 做代理评分。
    不精确但够快——优化循环中用，最终评估切回 LLM-as-judge。

    公式: score = 0.3 + 0.7 * (overlap_ratio)
    """
    expected_words = set(example.expected_behavior.lower().split())
    skill_words = set(skill_text.lower().split())

    if not expected_words:
        logger.info("[quick_fitness] expected_words 为空，返回默认分数 0.5")
        return FitnessScore(correctness=0.5, procedure_following=0.5,
                            conciseness=0.8, skill_clarity=0.7, feedback="")

    overlap = len(expected_words & skill_words) / len(expected_words)
    base_score = 0.3 + 0.7 * overlap

    fitness = FitnessScore(
        correctness=min(1.0, base_score),
        procedure_following=min(1.0, base_score * 0.9),
        conciseness=0.8,
        skill_clarity=0.7,
        feedback="",
    )

    logger.info(f"[quick_fitness] overlap={overlap:.3f} | composite={fitness.composite:.3f}")
    return fitness


def _simulate_agent(skill_text: str, task_input: str, client: OpenAI, model: str) -> str:
    """模拟 Agent 执行任务。"""
    prompt = f"""你是一个 AI Agent。请根据下面的技能说明执行用户任务。

技能说明：
---
{skill_text[:3000]}
---

用户任务：
{task_input}

请按照技能说明的步骤执行任务，输出结果。"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content or ""


def _parse_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return {}


# ── 跨模型泛化评测 ──────────────────────────────────────────

def evaluate_cross_model(
    skill_text: str,
    example: EvalExample,
    client: OpenAI,
    model: str,
    cross_model: str = None,
    cross_client: OpenAI = None,
) -> float:
    """
    跨模型验证：在第二个模型上评估技能效果。

    返回跨模型的 composite 分数。如果 cross_model 未指定，
    尝试从 Hermes config 读取备用模型。
    """
    if cross_model is None:
        cross_model = _get_fallback_model()
    if cross_client is None:
        cross_client = client

    logger.info(f"[evaluate_cross_model] 跨模型验证 | cross_model={cross_model} | task={example.task_input[:60]}...")

    try:
        fitness = evaluate_skill(skill_text, example, cross_client, cross_model)
        logger.info(f"[evaluate_cross_model] 跨模型验证完成 | score={fitness.composite:.3f}")
        return fitness.composite
    except Exception as exc:
        logger.warning(f"[cross-model] 评估失败: {exc}")
        return -1.0


def _get_fallback_model() -> str:
    """从 Hermes config 获取备用模型（与主模型不同的模型）。"""
    config_path = Path.home() / ".hermes" / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            providers = cfg.get("providers", {})
            active_model = _get_active_model()
            for name, pdata in providers.items():
                if not isinstance(pdata, dict):
                    continue
                models = pdata.get("models", [])
                for m in models:
                    if m != active_model:
                        logger.info(f"[_get_fallback_model] 选中备用模型: {m} (active={active_model})")
                        return m
        except Exception:
            pass
    logger.info(f"[_get_fallback_model] 未找到备用模型，使用默认 gpt-4o-mini")
    return "gpt-4o-mini"
