"""
3层评测 — Layer 1: 混合数据源评估数据集构建

数据来源：
  - Synthetic: LLM 合成生成
  - SessionDB: 从 Hermes 对话历史挖掘
  - Boundary: 边界探测任务生成（Skills-Coach 风格）
  - Golden: 人工手写测试集

分割策略：
  - golden → holdout（最高保真度，不参与训练）
  - session → val（真实数据，验证用）
  - synthetic + boundary → train（训练用）
  - 无 golden/session 时回退到 60/20/20 分割
"""

import json
import logging
import random
import re
from pathlib import Path
from openai import OpenAI

from self_evolution.core.evolution_provider import EvalExample, EvalDataset

logger = logging.getLogger(__name__)

_BUILD_PROMPT = """读取下面的 AI 技能文件，生成 {num_cases} 个测试用例。

每个用例是两部分：
1. task_input: 一个用户可能提出、需要用到这个技能的任务描述
2. expected_behavior: 一段 rubric 描述，说明 agent 处理这个任务时应该怎么做（不是精确文本，是语义描述）

要求：
- task_input 要多样化，覆盖技能的不同方面
- expected_behavior 描述"应该怎么做"而非"必须输出什么"
- 用例之间不要重复

技能文件：
---
{skill_text}
---

返回 JSON 数组格式：
```json
[
  {{
    "task_input": "...",
    "expected_behavior": "..."
  }}
]
```
只返回 JSON，不要解释。"""

_RUBRIC_PROMPT = """根据下面的对话上下文，为用户任务生成一段 rubric 描述。

用户任务：{task_input}

对话上下文：
---
{context}
---

rubric 要求：
- 描述 agent 处理这个任务时"应该怎么做"
- 是语义描述，不是精确匹配
- 简洁明了，3-5 句话

只返回 rubric 文本，不要解释。"""

_BOUNDARY_PROMPT = """你是技能边界探测专家。分析下面的技能，生成 {num_cases} 个挑战其能力边界的测试用例。

边界探测方向：
1. 技能未明确覆盖的边缘情况
2. 技能步骤之间的模糊地带
3. 输入超出技能预期范围的情况
4. 多步骤组合可能失败的场景
5. 技能描述中"如果...则..."条件分支的未覆盖路径

技能文件：
---
{skill_text}
---

返回 JSON 数组格式：
```json
[
  {{
    "task_input": "...",
    "expected_behavior": "...",
    "boundary_type": "edge_case|ambiguity|out_of_scope|combination|uncovered_branch"
  }}
]
```
只返回 JSON。"""


class SyntheticDatasetBuilder:
    """评估数据集构建器 — 3层评测 Layer 1"""

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def generate(self, skill_text: str, num_cases: int = 15) -> EvalDataset:
        """从 skill 文本生成评估数据集。"""
        logger.info(f"[SyntheticDatasetBuilder] 开始生成 {num_cases} 个合成用例 | model={self.model}")

        prompt = _BUILD_PROMPT.format(num_cases=num_cases, skill_text=skill_text)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        raw = response.choices[0].message.content
        examples = self._parse_examples(raw)

        logger.info(f"[SyntheticDatasetBuilder] 解析到 {len(examples)} 个用例")

        for ex in examples:
            ex.source = "synthetic"

        random.shuffle(examples)
        n = len(examples)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        dataset = EvalDataset(
            train=examples[:train_end],
            val=examples[train_end:val_end],
            holdout=examples[val_end:],
        )

        logger.info(
            f"[SyntheticDatasetBuilder] 数据集生成完成 | "
            f"train={len(dataset.train)} val={len(dataset.val)} holdout={len(dataset.holdout)}"
        )
        return dataset

    def _parse_examples(self, raw: str) -> list[EvalExample]:
        """从 LLM 输出解析测试用例。"""
        json_match = re.search(r'\[[\s\S]*\]', raw)
        if not json_match:
            return []

        try:
            items = json.loads(json_match.group())
            return [
                EvalExample(
                    task_input=item["task_input"],
                    expected_behavior=item["expected_behavior"],
                )
                for item in items
                if "task_input" in item and "expected_behavior" in item
            ]
        except (json.JSONDecodeError, KeyError):
            return []


class SessionDatasetBuilder:
    """从 Hermes 对话历史挖掘评估用例。"""

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def generate(self, skill_name: str, max_sessions: int = 20) -> list[EvalExample]:
        """从对话历史中挖掘与指定技能相关的 (task, rubric) 对。"""
        logger.info(f"[SessionDatasetBuilder] 开始扫描 | skill={skill_name} | max={max_sessions}")

        sessions = self._scan_sessions(skill_name, max_sessions)
        if not sessions:
            logger.info("[SessionDatasetBuilder] 未找到相关会话")
            return []

        logger.info(f"[SessionDatasetBuilder] 找到 {len(sessions)} 个相关会话，开始提取 rubric")

        examples: list[EvalExample] = []
        for session in sessions:
            task_input = session["task_input"]
            context = session["context"]
            rubric = self._extract_rubric(task_input, context)
            if rubric:
                examples.append(EvalExample(
                    task_input=task_input,
                    expected_behavior=rubric,
                    source="session",
                ))

        logger.info(f"[SessionDatasetBuilder] 提取完成 | {len(examples)}/{len(sessions)} 个有效用例")
        return examples

    def _scan_sessions(self, skill_name: str, max_sessions: int) -> list[dict]:
        """扫描会话文件，提取与技能相关的对话。"""
        sessions_dir = Path.home() / ".hermes" / "sessions"
        if not sessions_dir.exists():
            return []

        skill_lower = skill_name.lower()
        results: list[dict] = []

        session_files = sorted(sessions_dir.glob("*.jsonl"), reverse=True)
        for sf in session_files:
            if len(results) >= max_sessions:
                break

            try:
                lines = sf.read_text(encoding="utf-8").strip().split("\n")
            except (OSError, UnicodeDecodeError):
                continue

            messages: list[dict] = []
            for line in lines:
                try:
                    msg = json.loads(line)
                    if "role" in msg and "content" in msg:
                        messages.append(msg)
                except (json.JSONDecodeError, KeyError):
                    continue

            for i, msg in enumerate(messages):
                if msg["role"] != "user":
                    continue

                content = msg["content"]
                if skill_lower not in content.lower():
                    continue

                context_parts: list[str] = []
                start = max(0, i - 2)
                end = min(len(messages), i + 4)
                for j in range(start, end):
                    role = messages[j]["role"]
                    text = messages[j]["content"][:300]
                    context_parts.append(f"[{role}]: {text}")

                results.append({
                    "task_input": content[:200],
                    "context": "\n".join(context_parts),
                })

                if len(results) >= max_sessions:
                    break

        return results

    def _extract_rubric(self, task_input: str, context: str) -> str:
        """用 LLM 从对话上下文生成 rubric。"""
        prompt = _RUBRIC_PROMPT.format(task_input=task_input, context=context[:2000])

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            logger.debug("[SessionDatasetBuilder] rubric 生成失败", exc_info=True)
            return ""


class BoundaryProbeBuilder:
    """边界探测任务生成器（Skills-Coach 风格）。"""

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def generate(self, skill_text: str, num_cases: int = 10) -> list[EvalExample]:
        """生成挑战技能边界的测试用例。"""
        logger.info(f"[BoundaryProbeBuilder] 开始生成 {num_cases} 个边界探测用例 | model={self.model}")

        prompt = _BOUNDARY_PROMPT.format(num_cases=num_cases, skill_text=skill_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
            )
        except Exception:
            logger.debug("[BoundaryProbeBuilder] LLM 调用失败", exc_info=True)
            return []

        raw = response.choices[0].message.content
        json_match = re.search(r'\[[\s\S]*\]', raw)
        if not json_match:
            logger.warning("[BoundaryProbeBuilder] LLM 输出中未找到 JSON 数组")
            return []

        try:
            items = json.loads(json_match.group())
            examples: list[EvalExample] = []
            for item in items:
                if "task_input" not in item or "expected_behavior" not in item:
                    continue
                examples.append(EvalExample(
                    task_input=item["task_input"],
                    expected_behavior=item["expected_behavior"],
                    source="boundary",
                ))
            logger.info(f"[BoundaryProbeBuilder] 生成 {len(examples)} 个边界探测用例")
            return examples
        except (json.JSONDecodeError, KeyError):
            logger.warning("[BoundaryProbeBuilder] JSON 解析失败")
            return []


class GoldenDatasetLoader:
    """人工测试集加载器。"""

    @staticmethod
    def load(skill_name: str) -> list[EvalExample]:
        """加载人工编写的测试用例。"""
        path = Path.home() / ".hermes" / "skills" / skill_name / "golden_tests.json"
        if not path.exists():
            logger.info(f"[GoldenDatasetLoader] 未找到 golden 测试集 | path={path}")
            return []
        try:
            data = json.loads(path.read_text())
            examples = [
                EvalExample(
                    task_input=item["task_input"],
                    expected_behavior=item["expected_behavior"],
                    source="golden",
                )
                for item in data
                if "task_input" in item and "expected_behavior" in item
            ]
            logger.info(f"[GoldenDatasetLoader] 加载 {len(examples)} 个 golden 用例 | path={path}")
            return examples
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"[GoldenDatasetLoader] golden 测试集 JSON 解析失败 | path={path}")
            return []
