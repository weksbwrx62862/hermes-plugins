# Adaptive Multi-Agent (AMA) Plugin

自适应多智能体调度插件 — 6种协作模式，智能选择最佳执行策略

## 概述

根据任务复杂度自动选择最佳的 Anthropic 多智能体协作模式，通过 Hermes `delegate_task` 委派子智能体执行，支持智能模式切换、性能持久化和历史学习。

## 六种协作模式

### 1. Generator-Verifier（生成-验证）
- **适用**: 代码生成、文档撰写、数据转换
- **机制**: 生成器产出，验证器检查，不通过则重试
- **优势**: 质量保证高

### 2. Orchestrator-Subagent（协调-子代理）
- **适用**: 复杂任务分解、并行执行
- **机制**: 协调器分解任务，多个子代理并行执行
- **优势**: 高效利用资源

### 3. Agent Teams（团队协作）
- **适用**: 需要多角色协作的任务
- **机制**: 多个专业代理组成团队，分工协作
- **优势**: 专业化分工

### 4. Message Bus（事件驱动）
- **适用**: 需要松耦合的复杂工作流
- **机制**: 发布-订阅模式，异步事件驱动
- **优势**: 灵活性高

### 5. Shared State（共享状态）
- **适用**: 需要共享上下文的任务
- **机制**: 所有代理共享状态空间，协同更新
- **优势**: 信息一致性好

### 6. Parallel Fusion（并行融合）
- **适用**: 多视角分析、综合研判、跨域引用类任务
- **机制**: 参考模型并行生成多样化回答，迭代优化，裁决模型综合所有回答生成最终结果
- **优势**: 答案更全面、偏差更小

## 提供的工具

| 工具名 | 功能 |
|--------|------|
| `ama_execute` | 执行入口，自动选择模式 |
| `ama_assess` | 评估任务复杂度并推荐模式（不执行） |
| `ama_switch_mode` | 手动切换当前会话模式 |
| `ama_stats` | 查询执行统计 |
| `ama_cancel` | 取消正在执行的任务 |
| `ama_clarify` | 需求澄清与智能评分 |
| `ama_diagnose` | 诊断内部状态 |
| `ama_trajectories` | 查询执行轨迹记录 |
| `ama_grade` | 对指定轨迹进行 LLM-as-Judge 评分 |
| `ama_skills` | 查询子代理技能注册表统计 |
| `ama_workflow` | 列出或检查工作流模板 |
| `ama_resume` | 列出或恢复中断的任务 |

## 使用示例

```python
# 评估任务复杂度
ama_assess(task="重构认证模块", context="需要支持 OAuth2")

# 执行任务（自动选择模式）
ama_execute(
    task="实现用户注册登录系统",
    context="使用 JWT + bcrypt",
    human_input_mode="NEVER"
)

# 查看统计
ama_stats(period="week", detail=True)

# 手动切换模式
ama_switch_mode(mode="orchestrator_subagent")

# 诊断状态
ama_diagnose(include_ts_params=True)

# 查询最近执行轨迹
ama_trajectories(limit=10, success_only=True)

# 对指定轨迹进行评分
ama_grade(trajectory_id="tr-xxx")

# 查看技能注册表
ama_skills(domain="programming")

# 列出工作流模板
ama_workflow(action="list")

# 查看具体工作流
ama_workflow(action="info", workflow_id="software_dev")

# 恢复中断任务
ama_resume(action="list")
ama_resume(action="resume", trace_id="trace-xxx")
```

## 智能模式选择

插件使用 **Thompson Sampling** 算法学习最优模式：

1. 初始: 基于任务特征的启发式规则
2. 学习: 根据历史成功率调整 Beta 分布参数
3. 收敛: 自动偏向高成功率模式

### 任务特征分析
- 任务描述长度
- 并行度需求
- 依赖关系复杂度
- 预估执行时间
- 显式验证需求
- 多角色 / 多视角信号
- 跨域引用需求

## 熔断机制

当某个模式连续失败时：
1. 第 1 次失败: 记录
2. 第 2 次失败: 降低权重
3. 第 3 次连续失败: **触发熔断**，临时禁用该模式
4. 冷却期后自动恢复

## 配置

```yaml
plugins:
  enabled:
    - adaptive_multi_agent

ama:
  max_concurrent_children: 3  # 最大并发子代理数
  default_timeout: 300        # 默认超时(秒)
  enable_learning: true       # 启用 Thompson Sampling 学习
  allow_mode_switch: true     # 是否允许失败时自动切换模式
  circuit_breaker:
    threshold: 3              # 熔断阈值
    cooldown: 3600            # 冷却期(秒)
  llm_refine_enabled: true    # 规则评分在模糊区间时启用 LLM 二次评估
  llm_refine_range: [4.0, 6.0]  # 触发 LLM 二次评估的分数区间
  use_dag: true               # 高复杂度编排任务是否启用 DAG 编排
  dag_max_workers: 4          # DAG 最大并行工作线程
```

### 可观测性联动

AMA 与 plugin-orchestrator / model-router 插件保持双向联动：

- `error_category`: 每次失败都会写入统一的错误分类（timeout / validation / llm_error / subagent_failure / cancelled / unknown），`ama_diagnose` 会按分类聚合展示最近失败。
- `model_quality`: 当 Router 报告的当前模型质量 `<=2` 时，AMA 会从候选模式中排除重型模式（`agent_teams`、`shared_state`、`message_bus`、`parallel_fusion`），降低失败概率。
- 执行结果会回流 Router，用于更新模型反馈（`record_model_feedback`）。

## 安装

```bash
git clone https://github.com/weksbwrx62862/adaptive-multi-agent.git ~/.hermes/plugins/adaptive_multi_agent
```

## 运行测试

```bash
cd ~/.hermes/plugins/adaptive_multi_agent
pytest tests/
```

## 限制

- `ama_execute` 需要 parent_agent 上下文，插件工具接口暂不可用
- 子代理无法使用 `clarify`, `memory`, `send_message` 等工具
- 嵌套委托默认关闭（`max_spawn_depth=1`）

## 依赖

- Python 3.10+
- Hermes Agent

## License

MIT
