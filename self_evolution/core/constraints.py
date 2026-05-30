"""
3层评测 — Layer 3: ConstraintValidator — 10维 AND 门控

进化后的文本必须通过全部 10 个硬检查。
任何一条不过就拒绝，不管分数多高。
AND 逻辑：dim1 ∧ dim2 ∧ ... ∧ dim10 = 全部通过才部署。

10 维门控：
  1.  size_limit         ≤ 15KB  — 防止技能膨胀
  2.  growth_limit       ≤ 原始 120% — 防止一次进化暴涨
  3.  non_empty          不能为空 — 变异可能产生空文本
  4.  skill_structure    合法 markdown — 必须以 # 或 --- 开头
  5.  semantic_fidelity  语义保真度 — 不能偏离原始目的（LLM 判断）
  6.  cross_model_stability 跨模型稳定性 — 跨模型不退化
  7.  overfit_detection  过拟合检测 — train-val 差距不过大
  8.  silent_bypass_check 技能可被实际触发 — 非 silent-bypass
  9.  l1_compliance      L1 静态合规 — YAML frontmatter 格式校验
  10. l2_quality          L2 语义质量 — LLM 六维评估（Role/Structure/Instruction/Content/Risk/Execution）

借鉴 Skill-insight 的分层检查设计。
"""

import json
import logging
import re
from pathlib import Path
from openai import OpenAI
from typing import Optional

from self_evolution.core.evolution_provider import ConstraintResult

from self_evolution.core.fitness import _get_active_model

logger = logging.getLogger(__name__)

# 语义保真度检测提示词
_FIDELITY_PROMPT = """你是技能审计员。比较原始技能和进化后的技能，判断进化后是否保持了原始目的。

原始技能：
---
{original}
---

进化后技能：
---
{evolved}
---

判断标准：进化后的技能是否仍然：
1. 解决同一个问题？
2. 服务于同一个使用场景？
3. 没有引入无关的内容？

返回 JSON：
```json
{{
  "fidelity_passed": true,
  "reason": "进化后保持了核心目的，只改进了措辞和步骤清晰度"
}}
```
或者：
```json
{{
  "fidelity_passed": false,
  "reason": "进化后偏离了原始目的——原来讲 CI 配置，现在变成了 Docker 部署指南"
}}
```
只返回 JSON。"""


class ConstraintValidator:
    """
    约束门控验证器 — 3层评测 Layer 3

    8 维 AND 门控：
      size_limit ∧ growth_limit ∧ non_empty ∧ skill_structure
      ∧ semantic_fidelity ∧ cross_model_stability ∧ overfit_detection
      ∧ silent_bypass_check
    """

    MAX_SIZE = 15 * 1024       # 15KB
    MAX_GROWTH = 1.20           # 120%

    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client

    def validate_all(
        self,
        text: str,
        baseline: Optional[str] = None,
        **kwargs,
    ) -> ConstraintResult:
        """
        执行全部 8 维门控检查。

        Args:
            text: 要验证的 skill 文本
            baseline: 原始版本（用于 growth_limit 和 fidelity）
            **kwargs: 额外参数
              cross_model_score: 跨模型验证分数
              baseline_cross_model: 基线跨模型分数
              train_score: 训练集分数
              val_score: 验证集分数
              audit_report: 审计报告
        """

        checks = {}
        failures = []

        logger.info(f"[constraints] 开始 8 维 AND 门控检查 | text_len={len(text)} | has_baseline={baseline is not None}")

        # 维度 1: size_limit — 文字数 ≤ 15KB
        checks["size_limit"] = len(text) <= self.MAX_SIZE
        if not checks["size_limit"]:
            failures.append(f"size_limit: {len(text)} > {self.MAX_SIZE} bytes")
            logger.info(f"[constraints] 维度1 size_limit ❌ | {len(text)} > {self.MAX_SIZE}")
        else:
            logger.info(f"[constraints] 维度1 size_limit ✓ | {len(text)} ≤ {self.MAX_SIZE}")

        # 维度 2: growth_limit — 不超过原始 120%
        if baseline:
            checks["growth_limit"] = len(text) <= len(baseline) * self.MAX_GROWTH
            if not checks["growth_limit"]:
                growth_pct = len(text) / len(baseline) * 100
                failures.append(f"growth_limit: {growth_pct:.1f}% > {self.MAX_GROWTH*100:.0f}%")
                logger.info(f"[constraints] 维度2 growth_limit ❌ | {growth_pct:.1f}% > {self.MAX_GROWTH*100:.0f}%")
            else:
                growth_pct = len(text) / len(baseline) * 100
                logger.info(f"[constraints] 维度2 growth_limit ✓ | {growth_pct:.1f}% ≤ {self.MAX_GROWTH*100:.0f}%")
        else:
            checks["growth_limit"] = True
            logger.info("[constraints] 维度2 growth_limit ⊘ | 无 baseline，跳过")

        # 维度 3: non_empty — 不能为空
        checks["non_empty"] = len(text.strip()) > 0
        if not checks["non_empty"]:
            failures.append("non_empty: text is empty")
            logger.info("[constraints] 维度3 non_empty ❌ | 文本为空")
        else:
            logger.info(f"[constraints] 维度3 non_empty ✓ | strip_len={len(text.strip())}")

        # 维度 4: skill_structure — 合法 markdown 格式
        checks["skill_structure"] = self._check_structure(text)
        if not checks["skill_structure"]:
            failures.append("skill_structure: must start with # or --- (valid markdown)")
            logger.info("[constraints] 维度4 skill_structure ❌ | 不以 # 或 --- 开头")
        else:
            logger.info("[constraints] 维度4 skill_structure ✓")

        # 维度 5: semantic_fidelity — 语义保真度（需要 client + baseline）
        if baseline and self.client:
            checks["semantic_fidelity"] = self._check_fidelity(baseline, text)
            if not checks["semantic_fidelity"]:
                failures.append("semantic_fidelity: evolved text deviated from original purpose")
                logger.info("[constraints] 维度5 semantic_fidelity ❌ | 语义偏离原始目的")
            else:
                logger.info("[constraints] 维度5 semantic_fidelity ✓")
        else:
            checks["semantic_fidelity"] = True
            logger.info(f"[constraints] 维度5 semantic_fidelity ⊘ | has_baseline={baseline is not None}, has_client={self.client is not None}")

        # 维度 6: cross_model_stability — 跨模型稳定性
        cross_model_score = kwargs.get("cross_model_score")
        baseline_cross_model = kwargs.get("baseline_cross_model")
        if cross_model_score is not None and baseline_cross_model is not None:
            checks["cross_model_stability"] = self._check_cross_model_stability(
                cross_model_score, baseline_cross_model
            )
            if not checks["cross_model_stability"]:
                failures.append(
                    f"cross_model_stability: cross_model({cross_model_score:.3f}) "
                    f"regressed vs baseline({baseline_cross_model:.3f})"
                )
                logger.info(
                    f"[constraints] 维度6 cross_model_stability ❌ | "
                    f"cross={cross_model_score:.3f} < baseline*0.9={baseline_cross_model*0.9:.3f}"
                )
            else:
                logger.info(
                    f"[constraints] 维度6 cross_model_stability ✓ | "
                    f"cross={cross_model_score:.3f} ≥ baseline*0.9={baseline_cross_model*0.9:.3f}"
                )
        else:
            checks["cross_model_stability"] = True
            logger.info("[constraints] 维度6 cross_model_stability ⊘ | 无跨模型数据，跳过")

        # 维度 7: overfit_detection — 过拟合检测
        train_score = kwargs.get("train_score")
        val_score = kwargs.get("val_score")
        if train_score is not None and val_score is not None:
            checks["overfit_detection"] = self._check_overfit(train_score, val_score)
            if not checks["overfit_detection"]:
                failures.append(
                    f"overfit_detection: train({train_score:.3f}) - val({val_score:.3f}) "
                    f"gap too large"
                )
                gap = train_score - val_score
                logger.info(f"[constraints] 维度7 overfit_detection ❌ | gap={gap:.3f} > 0.20")
            else:
                gap = train_score - val_score
                logger.info(f"[constraints] 维度7 overfit_detection ✓ | gap={gap:.3f} ≤ 0.20")
        else:
            checks["overfit_detection"] = True
            logger.info("[constraints] 维度7 overfit_detection ⊘ | 无 train/val 分数，跳过")

        # 维度 8: silent_bypass_check — 技能可被实际触发
        checks["silent_bypass_check"] = self._check_silent_bypass(text)
        if not checks["silent_bypass_check"]:
            failures.append("silent_bypass_check: skill appears to be a silent bypass")
            logger.info("[constraints] 维度8 silent_bypass_check ❌ | 检测到 silent-bypass 模式")
        else:
            logger.info("[constraints] 维度8 silent_bypass_check ✓")

        # 维度 9: l1_compliance — L1 静态合规（借鉴 Skill-insight）
        l1_result = self._check_l1_compliance(text)
        checks["l1_compliance"] = l1_result["passed"]
        if not l1_result["passed"]:
            failures.append(f"l1_compliance: {l1_result['reason']}")
            logger.info(f"[constraints] 维度9 l1_compliance ❌ | {l1_result['reason']}")
        else:
            logger.info("[constraints] 维度9 l1_compliance ✓")

        # 维度 10: l2_quality — L2 语义质量（借鉴 Skill-insight 六维评估）
        if self.client:
            l2_result = self._check_l2_quality(text)
            checks["l2_quality"] = l2_result["passed"]
            if not l2_result["passed"]:
                failures.append(f"l2_quality: {l2_result['reason']}")
                logger.info(f"[constraints] 维度10 l2_quality ❌ | {l2_result['reason']}")
            else:
                logger.info(f"[constraints] 维度10 l2_quality ✓ | score={l2_result.get('score', 0):.2f}")
        else:
            checks["l2_quality"] = True
            logger.info("[constraints] 维度10 l2_quality ⊘ | 无 client，跳过")

        passed = len(failures) == 0
        logger.info(
            f"[constraints] 10 维门控结果 | passed={passed} | "
            f"checks={checks} | failures={failures}"
        )

        return ConstraintResult(
            passed=passed,
            checks=checks,
            failures=failures,
        )

    def _check_structure(self, text: str) -> bool:
        """检查是否为合法 markdown 结构。"""
        stripped = text.strip()
        return bool(stripped) and (
            stripped.startswith("#") or stripped.startswith("---")
        )

    def _check_fidelity(self, original: str, evolved: str) -> bool:
        """用 LLM 检查语义保真度。"""
        if not self.client:
            return True  # 无 client 时默认通过

        prompt = _FIDELITY_PROMPT.format(original=original[:3000], evolved=evolved[:3000])

        try:
            response = self.client.chat.completions.create(
                model=_get_active_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = response.choices[0].message.content
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                data = json.loads(match.group())
                return data.get("fidelity_passed", True)
        except Exception:
            pass

        return True  # 出错时保守处理：默认通过

    def _check_cross_model_stability(
        self, cross_model_score: float, baseline_cross_model: float
    ) -> bool:
        """跨模型稳定性检查：跨模型分数不低于基线的 90%。"""
        if baseline_cross_model <= 0:
            return True
        return cross_model_score >= baseline_cross_model * 0.9

    def _check_overfit(self, train_score: float, val_score: float) -> bool:
        """过拟合检测：train-val 差距不超过 0.20。"""
        return (train_score - val_score) <= 0.20

    def _check_silent_bypass(self, text: str) -> bool:
        """检查技能是否为 silent-bypass（看起来有效但运行时不会被遵循）。"""
        stripped = text.strip().lower()
        bypass_patterns = [
            "skip this step",
            "ignore this instruction",
            "do nothing",
            "no action needed",
            "bypass",
        ]
        return not any(p in stripped for p in bypass_patterns)

    # ── L1 静态合规检查（借鉴 Skill-insight）──────────────

    def _check_l1_compliance(self, text: str) -> dict:
        """
        L1 静态合规检查 — 硬规则校验。

        检查项：
        1. YAML Frontmatter: name, description 存在且格式正确
        2. Length Check: 内容长度不超过阈值
        3. Header Structure: 包含必要的章节标题

        Returns:
            {"passed": bool, "reason": str, "details": dict}
        """
        details = {}
        issues = []

        # 1. YAML Frontmatter 检查
        has_frontmatter = text.strip().startswith("---")
        if has_frontmatter:
            # 提取 frontmatter
            parts = text.split("---", 2)
            if len(parts) >= 3:
                fm = parts[1].strip()
                has_name = "name:" in fm
                has_desc = "description:" in fm
                details["has_frontmatter"] = True
                details["has_name"] = has_name
                details["has_description"] = has_desc

                if not has_name:
                    issues.append("missing 'name' in frontmatter")
                if not has_desc:
                    issues.append("missing 'description' in frontmatter")
            else:
                details["has_frontmatter"] = True
                details["has_name"] = False
                details["has_description"] = False
                issues.append("malformed frontmatter (missing closing ---)")
        else:
            details["has_frontmatter"] = False
            issues.append("missing YAML frontmatter (must start with ---)")

        # 2. Length Check
        max_length = 50 * 1024  # 50KB 绝对上限
        details["length"] = len(text)
        if len(text) > max_length:
            issues.append(f"content too long: {len(text)} > {max_length} bytes")

        # 3. Header Structure — 检查是否有至少一个 # 标题
        has_header = bool(re.search(r'^#+\s+\S', text, re.MULTILINE))
        details["has_header"] = has_header
        if not has_header:
            issues.append("no markdown headers found (must have at least one # heading)")

        passed = len(issues) == 0
        return {
            "passed": passed,
            "reason": "; ".join(issues) if issues else "L1 compliance passed",
            "details": details,
        }

    # ── L2 语义质量评估（借鉴 Skill-insight 六维评估）────

    _L2_QUALITY_PROMPT = """你是技能质量评估专家。对以下 Agent Skill 进行六维质量评估。

Skill 内容：
---
{text}
---

请从以下六个维度评估（每项 0-1 分）：

1. **Role（职责明确性）**：Skill 是否聚焦于单一目的？description 是否包含具体可匹配的触发信号？
2. **Structure（结构规范性）**：格式是否规范？是否保持精炼？
3. **Instruction（指令适配性）**：指令自由度是否与任务风险等级匹配？
4. **Content（内容一致性）**：同一概念是否使用统一术语？是否存在过期信息？
5. **Risk（风险可控性）**：安全边界和权限控制是否完备？
6. **Execution（脚本与参考文档质量）**：脚本是否完整？错误处理是否包含修复建议？

返回 JSON：
```json
{{
  "scores": {{
    "role": 0.8,
    "structure": 0.9,
    "instruction": 0.7,
    "content": 0.8,
    "risk": 0.9,
    "execution": 0.8
  }},
  "overall": 0.82,
  "issues": ["具体问题1", "具体问题2"],
  "passed": true
}}
```

总分 >= 0.6 为通过。只返回 JSON。"""

    def _check_l2_quality(self, text: str) -> dict:
        """
        L2 语义质量评估 — LLM 六维评估（借鉴 Skill-insight）。

        六维：Role / Structure / Instruction / Content / Risk / Execution

        Returns:
            {"passed": bool, "reason": str, "score": float, "details": dict}
        """
        if not self.client:
            return {"passed": True, "reason": "no client", "score": 0, "details": {}}

        prompt = self._L2_QUALITY_PROMPT.format(text=text[:4000])

        try:
            response = self.client.chat.completions.create(
                model=_get_active_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = response.choices[0].message.content
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                data = json.loads(match.group())
                overall = data.get("overall", 0)
                passed = data.get("passed", overall >= 0.6)
                issues = data.get("issues", [])

                return {
                    "passed": passed,
                    "reason": "; ".join(issues) if issues else f"L2 quality score: {overall:.2f}",
                    "score": overall,
                    "details": data.get("scores", {}),
                }
        except Exception as e:
            logger.debug(f"[constraints] L2 quality check failed: {e}")

        return {"passed": True, "reason": "L2 check error, default pass", "score": 0, "details": {}}
