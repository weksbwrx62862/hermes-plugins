# Adaptive Multi-Agent (AMA) 架构文档

## 1. 整体架构

AMA 采用分层、可扩展的插件架构，核心流程为：

```text
Hermes 工具调用
    │
    ▼
┌─────────────────┐
│  handlers.py    │  ← 工具处理器入口，负责参数校验、引擎单例管理
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  AdaptiveMultiAgentEngine │  ← 外观（Facade），编排评估、选型、执行、持久化
│      (engine.py)        │
└────────┬────────────────┘
         │
    ┌────┴────┬────────────┬──────────────┐
    ▼         ▼            ▼              ▼
┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│assessor│ │ selector │ │ executor │ │ diagnostics  │
│        │ │          │ │          │ │              │
└────────┘ └──────────┘ └────┬─────┘ └──────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  ┌────────────┐      ┌───────────────┐   ┌─────────────┐
  │  subagent  │      │dag_orchestrator│   │    graph    │
  │            │      │               │   │             │
  └────────────┘      └───────────────┘   └─────────────┘
        │
        ▼
  ┌─────────────┐   ┌───────────┐   ┌───────────────┐
  │  rolecards  │   │ workflows │   │ skill_registry│
  │             │   │           │   │               │
  └─────────────┘   └───────────┘   └───────────────┘
```

### 1.1 各层职责

| 模块 | 文件 | 职责 |
|------|------|------|
| 工具入口 | `handlers.py` | 解析工具参数、注入 PluginContext、调用引擎、格式化返回 |
| 引擎外观 | `engine.py` | 初始化子模块、维护生命周期、编排 `execute/assess/diagnose` 流程 |
| 任务评估 | `assessor.py` | 规则引擎评分、LLM 二次精修、需求澄清 |
| 模式选择 | `selector.py` | Thompson Sampling、规则过滤、成本感知、熔断器可用性判断 |
| 模式执行 | `executor.py` | 六种模式的具体调度、子代理调用、共享记忆、模式切换 |
| 内部诊断 | `diagnostics.py` | TS 参数、性能历史、熔断器状态、Mermaid 流程图 |
| 子代理抽象 | `subagent.py` | `AgentMode`、`SubagentConfig`、熔断器、重试策略、状态机、终止条件 |
| DAG 编排 | `dag_orchestrator.py` | 有向无环图任务依赖管理、拓扑排序、并行执行 |
| 状态图 | `graph.py` | 声明式图节点/边定义，供模式插件扩展 |
| 角色卡片 | `rolecards.py` | 标准化角色定义与角色上下文注入 |
| 工作流 | `workflows.py` | 预定义任务类型工作流模板 |
| 技能注册表 | `skill_registry.py` | 自动提取技能标签、记录成功率、推荐配置 |
| 持久化 | `persistence.py` | 统一 SQLite 访问、表结构、迁移、轨迹/性能存储 |
| 检查点 | `checkpoint.py` | 执行快照保存、中断恢复、可恢复任务列表 |
| 轨迹记录 | `trajectory.py` | 完整执行轨迹采集与持久化 |
| 评分器 | `grader.py` | LLM-as-Judge 多维度评分 |
| 错误分类 | `errors.py` | 统一错误类别枚举 |
| Schema | `schemas.py` | 12 个 AMA 工具的 JSON Schema 定义 |
| 包入口 | `__init__.py` | 注册工具、注入上下文、历史数据迁移 |

## 2. 数据流

```text
用户请求
    │
    ▼
[1] 任务评估（assessor）
    - 规则引擎提取特征、计算复杂度分数
    - 若分数落在模糊区间，触发 LLM 二次精修
    - 输出：complexity_score / task_type / features / recommended_mode
    │
    ▼
[2] 工作流匹配（workflows.match_workflow）
    - 按 task_type 匹配预定义工作流
    - 若匹配成功且未强制指定模式，使用工作流默认模式
    │
    ▼
[3] 模式选择（selector）
    - 规则过滤候选模式
    - Router→AMA 联动：低模型质量时排除重型模式
    - 成本感知降级
    - Thompson Sampling 采样最终模式
    │
    ▼
[4] 模式执行（executor）
    - 根据模式调用对应执行函数
    - 自动 DAG：orchestrator 模式在高复杂度时启用 DAG 编排
    - 子代理通过 Hermes delegate_task 执行
    - 失败时触发模式切换（若启用 allow_mode_switch）
    │
    ▼
[5] 持久化（persistence）
    - 写入 ama_performance（性能统计）
    - 写入 ama_executions（执行记录，含 error_category、status、trace_id 等）
    - 写入 ama_trajectories（完整轨迹）
    - 保存 ama_state_snapshots（检查点）
    │
    ▼
[6] 学习更新
    - selector.record_performance 更新历史表现
    - selector._ts_update 更新 Beta 后验分布
    - skill_registry 记录技能标签与成功率
    - 执行结果回流 model-router
```

## 3. 模块边界与职责

### 3.1 handlers.py（工具层）

- 仅负责参数提取、空值校验、结果序列化
- 通过全局单例 `_engine` / `_clarifier` 访问业务逻辑
- `handle_post_tool_call` 监控 `ama_execute` 的 token / 时间 / 重试 / 错误类别，生成告警
- `handle_on_session_start/end` 清理会话模式覆盖与过期结果缓存

### 3.2 engine.py（编排层）

- 初始化 `assessor`、`selector`、`executor`、熔断器、注册表、技能注册表等
- `execute()` 是完整执行入口，包含评估、选型、执行、切换、持久化、反馈闭环
- `assess()` / `diagnose()` / `generate_mermaid_diagram()` 为只读查询接口
- 不直接实现评估/选择/执行细节，全部委托给子模块

### 3.3 assessor.py（评估层）

- `TaskComplexityAssessor.assess()` 基于关键词与规则输出复杂度分数和任务类型
- 特征维度：`needs_parallelism`、`has_roles`、`is_event_driven`、`needs_collaboration`、`requires_shared_knowledge`、`multi_perspective`、`cross_reference`、`reasoning_depth` 等
- `RequirementClarifier` 通过多轮 LLM 提问澄清需求并返回评分结果

### 3.4 selector.py（选择层）

- `ModeSelectionEngine.select_mode()` 综合规则、历史、成本、模型质量选择模式
- Thompson Sampling 参数 `(alpha, beta)` 按 `(task_type, mode)` 维护
- `_apply_rules()` 根据复杂度和特征生成候选模式
- `record_performance()` 更新历史统计与 Beta 后验

### 3.5 executor.py（执行层）

- `ModeExecutor.execute_mode()` 根据模式分发到六种内置实现或插件图
- 包含 `SharedMemory`、`QualityScorer`、`SwitchContext` 等辅助类
- 六种模式：`_run_generator_verifier`、`_run_orchestrator_subagent`、`_run_agent_teams`、`_run_message_bus`、`_run_shared_state`、`_run_parallel_fusion`
- `_execute_with_dag()` 为 orchestrator 提供增强版 DAG 执行
- `try_switch_mode()` 在失败时基于错误类别智能切换模式

### 3.6 persistence.py（持久化层）

- 单例 `AMAPersistence`，统一访问 `~/.hermes/ama_state.db`
- WAL 模式、事务上下文管理器
- 对外提供 facade 函数：`load_performance`、`save_execution_transaction`、`query_trajectories` 等

## 4. 持久化层

### 4.1 数据库位置

```text
~/.hermes/ama_state.db
```

### 4.2 表结构

| 表名 | 用途 | 关键字段 |
|------|------|----------|
| `ama_performance` | 按任务类型和模式聚合的成功率、token、耗时 | `task_type`, `mode`, `trials`, `successes`, `avg_tokens`, `avg_time` |
| `ama_executions` | 每次执行的详细记录 | `session_id`, `task`, `task_type`, `complexity_score`, `mode_used`, `original_mode`, `success`, `token_usage`, `time_taken`, `switched_modes`, `switch_reason`, `trace_id`, `status`, `error_category`, `retries_attempted`, `timeout_seconds`, `created_at`, `updated_at` |
| `ama_state_snapshots` | 检查点快照 | `trace_id`, `round`, `state_json` |
| `ama_memory` | 任务-结果记忆 | `task_hash`, `task`, `result`, `task_type`, `mode`, `success` |
| `ama_trajectories` | 完整执行轨迹 | `trajectory_id`, `task`, `context`, `mode`, `complexity_score`, `steps_json`, `final_result`, `success`, `error`, `grade_score`, `grade_feedback` |

### 4.3 迁移策略

- `_ensure_schema()` 启动时执行 `CREATE TABLE IF NOT EXISTS`
- `_MIGRATE_SQL` 列表通过 `ALTER TABLE ... ADD COLUMN` 添加新列，重复执行时忽略 `OperationalError`
- 旧版独立 `~/.hermes/ama_trajectories/trajectories.db` 通过 `migrate_legacy_trajectories()` 迁移到统一数据库，迁移标记 `.migrated` 防止重复迁移

## 5. 测试结构

测试目录：`tests/`

| 测试文件 | 覆盖范围 |
|----------|----------|
| `conftest.py` | 公共 fixture：临时数据库、mock PluginContext、PYTHONPATH 注入 |
| `test_handlers.py` | 工具处理器集成测试（assess/stats/workflow/cancel/diagnose/execute/resume） |
| `test_assessor.py` | 任务复杂度评估与澄清逻辑 |
| `test_selector.py` | 模式选择、Thompson Sampling、历史更新 |
| `test_subagent.py` | 子代理配置、熔断器、重试策略、状态机 |
| `test_persistence.py` | SQLite 持久化读写、统计查询、迁移 |
| `test_rolecards.py` | 角色卡片注入与上下文生成 |
| `test_dag_orchestrator.py` | DAG 拓扑排序、并行执行、错误隔离 |

运行测试：

```bash
cd ~/.hermes/plugins/adaptive_multi_agent
pytest tests/
```

## 6. 扩展点

### 6.1 新增协作模式

1. 在 `subagent.py` 的 `AgentMode` 枚举中新增模式值
2. 在 `_MODE_PRESETS` 中新增 `SubagentConfig`
3. 在 `selector.py` 的 `_apply_rules()` 中加入选型规则
4. 在 `executor.py` 中新增 `_run_<mode>()` 方法并在 `execute_mode()` 中分发
5. 在 `schemas.py` 的 `force_mode` / `mode` enum 中加入新值
6. 在 `diagnostics.py` 的 `mode_flows` 中补充 Mermaid 流程图

### 6.2 新增工具

1. 在 `schemas.py` 的 `AMA_TOOL_SCHEMAS` 中定义工具 schema
2. 在 `handlers.py` 中实现 `handle_<tool_name>()`
3. 在 `__init__.py` 的 `_TOOLS` 元组中注册工具名、schema、handler、emoji
4. 在 `plugin.yaml` 的 `provides_tools` 列表中追加工具名
5. 在 `README.md` 工具表格和示例中补充说明

### 6.3 新增工作流

1. 在 `workflows.py` 的 `WORKFLOW_LIBRARY` 中新增 `WorkflowTemplate`
2. 定义各 `WorkflowStage`：id、name、role_id、goal_template、success_criteria、next_stage、fallback_stage
3. 将工作流匹配的 `task_types` 加入 `WorkflowTemplate.task_types`
4. 通过 `ama_workflow(action="info", workflow_id="xxx")` 验证

## 7. 关键设计决策

- **外观模式**：`AdaptiveMultiAgentEngine` 作为统一入口，降低工具层与子模块的耦合
- **策略模式**：模式选择、模式执行均可独立扩展
- **插件协议**：`ModePlugin` 协议允许通过 `SubagentRegistry.register_plugin()` 注册自定义模式图
- **统一持久化**：所有状态集中到一个 SQLite 文件，便于备份和迁移
- **双向联动**：与 model-router 共享 `model_quality` 和 `task_weight`，实现跨插件协同
- **可观测性**：每条执行记录都包含 `error_category`、`status`、`trace_id`，支持诊断和告警
