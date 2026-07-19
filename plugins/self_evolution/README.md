# Self-Evolution Plugin v2.0

自进化引擎 — 8阶段 Loop、3层评测、5维 AND 门控

## 概述

Hermes Agent 技能自进化系统，通过自动优化和迭代改进来提升技能质量。借鉴清华 EmbodiSkill（四种反思类型）+ SkillEvolver（策略多样化 K=4 + 独立 Auditor 9项检查）。

## 核心特性

### 8阶段进化 Loop
1. **Scan** — 扫描候选技能
2. **Analyze** — 分析当前版本
3. **Generate** — 生成优化策略
4. **Test** — 执行测试评估
5. **Evaluate** — 多维度打分
6. **Compare** — 与基线对比
7. **Deploy** — 部署改进版本
8. **Monitor** — 监控后续表现

### 3层评测体系
- **L1 单元测试** — 语法正确性、代码规范
- **L2 功能测试** — 30个测试用例（正常/边界/异常三类场景）
- **L3 真实场景** — 跨模型评判、双裁判集成

### 5维 AND 门控
必须同时满足 5 个维度才能通过进化：
1. ✅ 正确性 ≥ 95%
2. ✅ 性能提升 ≥ 10%
3. ✅ 无回归退化
4. ✅ Token 使用量不增加
5. ✅ 人工审核通过

## 提供的工具

7 个标准 Hermes 工具：

| 工具名 | 功能 |
|--------|------|
| `self_evo_status` | 查看进化状态 |
| `self_evo_scan` | 扫描候选技能 |
| `self_evo_approve` | 批准进化方案 |
| `self_evo_reject` | 拒绝进化方案 |
| `self_evo_evolve` | 执行进化 |
| `self_evo_execute` | 执行技能 |
| `self_evo_rollback` | 回滚版本 |

## 安装

```bash
# 克隆到 plugins 目录
git clone https://github.com/weksbwrx62862/self-evolution.git ~/.hermes/plugins/self_evolution
```

## 配置

在 `~/.hermes/config.yaml` 中启用：

```yaml
plugins:
  enabled:
    - self_evolution
```

## CLI 用法

```bash
# 进化一个 skill
PYTHONPATH="plugins:$PYTHONPATH" python -m self_evolution.cli --evolve github-actions-python-ci

# 进化并自动部署
PYTHONPATH="plugins:$PYTHONPATH" python -m self_evolution.cli --evolve <skill-name> --auto-deploy

# 查看状态
PYTHONPATH="plugins:$PYTHONPATH" python -m self_evolution.cli --status
```

## 技术架构

- **打分机制 v2**: 跨模型评判（worker ≠ judge）、双裁判集成、30个测试用例、严格校准锚点
- **策略多样化**: K=4 种优化策略并行评估
- **独立审计**: 9项检查确保质量（安全性、可维护性、文档完整性等）
- **状态持久化**: `~/.hermes/self_evolution_state.json`

## 自动触发

- **Cron Job**: 每日 03:00 自动扫描候选技能
- **审批系统**: 人工审核后才执行进化
- **联动优化**: cron iterations 从 3 降到 1 避免 120s 超时

## 依赖

- Python 3.10+
- Hermes Agent

## License

MIT
