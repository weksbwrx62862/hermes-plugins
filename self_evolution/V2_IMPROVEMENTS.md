# 自进化插件 v2 — 借鉴清华 EmbodiSkill + SkillEvolver

## 改进点

### 1. 四种反思类型（借鉴 EmbodiSkill）

失败后先判断"是手册写错了，还是我没按手册做"：

| 类型 | 场景 | 处理 |
|------|------|------|
| **discovery** | 成功 + 发现新技能苗头 | 记下来 |
| **optimization** | 成功 + 效率不高 | 优化技能 |
| **skill_defect** | 失败 + 技能有缺陷 | **改技能** |
| **execution_lapse** | 失败 + 技能没问题 | **不改技能** |

**关键创新**：只有 `skill_defect` 才触发变异，避免"执行失误"污染技能。

### 2. 策略多样化探索（借鉴 SkillEvolver）

每次迭代生成 **K=4** 个不同策略的候选：

| 策略 | 描述 |
|------|------|
| conservative | 最小改动，只修复最关键问题 |
| structural | 重新组织步骤顺序和逻辑 |
| supplement | 添加缺失步骤和边缘情况 |
| simplify | 精简冗余内容，提高清晰度 |

### 3. 独立 Auditor（借鉴 SkillEvolver）

9 项机械检查，拦截有害更新：

| 检查项 | 说明 |
|--------|------|
| 格式完整性 | 标题、步骤、示例等基本结构 |
| 一致性 | 术语、命名、格式前后一致 |
| 可执行性 | 步骤具体可执行 |
| 完整性 | 覆盖关键步骤和边缘情况 |
| 清晰度 | 指令清晰无歧义 |
| 错误处理 | 包含错误处理和故障排除 |
| 示例质量 | 示例充分有代表性 |
| 安全性 | 危险操作警告 |
| 可维护性 | 易于理解和修改 |

### 4. 对比更新（借鉴 SkillEvolver）

成功 vs 失败轨迹对比，提炼改进点。

## 新增文件

```
~/.hermes/plugins/self_evolution/
├── core/
│   ├── fitness_v2.py          # 增强版评测（含反思类型）
│   └── orchestrator_v2.py     # 增强版编排器
└── pipeline/
    └── optimizer_v2.py        # 增强版优化器
```

## 使用方式

```python
# 使用 v2 优化器
from self_evolution.pipeline.optimizer_v2 import SkillOptimizerV2
from openai import OpenAI

client = OpenAI(api_key="...", base_url="...")
optimizer = SkillOptimizerV2(client=client, model="deepseek-v4-pro", k_candidates=4)

evolved_text, best_score, iterations, audit_report = optimizer.optimize(
    skill_text=skill_text,
    dataset=dataset,
    iterations=5,
)
```

## 实验结果（预期）

| 指标 | v1 | v2 (预期) | 改进 |
|------|-----|-----------|------|
| 有害更新拦截率 | 0% | ~17% | +17% |
| 技能质量 | 基准 | +13.3% | 参考论文 |
| 迭代效率 | 基准 | +20% | 策略多样化 |

## 参考论文

- EmbodiSkill: https://arxiv.org/abs/2605.10332
- SkillEvolver: https://arxiv.org/abs/2605.10500
