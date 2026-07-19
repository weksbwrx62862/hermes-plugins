# skill-router 插件详细分析 Spec

## Why

skill-router 是 Hermes 的统一技能路由插件（v3.1），承担“把用户查询映射到合适技能”的核心职责。随着后端从轻量 MiniLM 扩展到 SkillRouter（Qwen3-0.6B）双后端，并引入域分组、增量嵌入缓存、反馈学习等机制，其复杂性显著上升。为了后续安全迭代、故障排查和性能优化，需要对当前实现做一次系统性的结构、数据流和风险分析。

## What Changes

- 在 `.trae/specs/analyze-skill-router-plugin/` 下创建分析型规格文档（spec.md、tasks.md、checklist.md）。
- 对 `__init__.py`、`skillrouter_backend.py`、`plugin.yaml`、`scripts/evaluate.py` 进行只读逆向分析，不修改任何源代码。
- 输出：模块职责说明、依赖清单、核心数据流、降级路径、风险与改进建议、验证计划。
- **BREAKING**：无代码变更，因此不会引入运行时行为变化；但分析结论可能驱动后续重构规格。

## Impact

- Affected specs：skill-router v3.1 全功能域（lightweight 后端、skillrouter 后端、混合检索、反馈学习、缓存、域分类、pre_llm_call 钩子）。
- Affected code：
  - `/home/xxh/.hermes/plugins/skill-router/__init__.py`
  - `/home/xxh/.hermes/plugins/skill-router/skillrouter_backend.py`
  - `/home/xxh/.hermes/plugins/skill-router/plugin.yaml`
  - `/home/xxh/.hermes/plugins/skill-router/scripts/evaluate.py`
  - 外部依赖：`sentence_transformers`、`transformers`、`torch`、`numpy`、`jieba`、`sqlite3`、`yaml`。

## ADDED Requirements

### Requirement: 插件结构与元数据分析

系统 SHALL 提供 skill-router 插件的目录结构、文件职责和 `plugin.yaml` 元数据说明。

#### Scenario: 成功读取元数据

- **WHEN** 分析 `plugin.yaml` 时
- **THEN** 应能列出插件名、版本、kind、提供的工具（`skill_search`、`skill_feedback`）和钩子（`pre_llm_call`），并说明与 Hermes 的兼容版本要求。

### Requirement: 核心模块职责分析

系统 SHALL 逐模块说明 `__init__.py` 中以下组件的职责、状态管理和线程安全策略：

- `CacheManager`、`QueryCache`：TTL / LRU 缓存。
- `FeedbackStore`：JSONL 持久化反馈与指数衰减加权。
- `BM25Searcher`：jieba / 字符级分词、IDF 预计算。
- `HybridSearcher`：向量-BM25 分数归一化融合、可选 CrossEncoder 重排序。
- `search_skills`：端到端检索主流程（含域分组、双后端、降级）。
- `register` / `_pre_llm_call_hook`：Hermes 插件注册与上下文注入。

#### Scenario: 模块边界清晰

- **WHEN** 阅读分析文档
- **THEN** 能明确每个类的输入、输出、副作用和关键配置项，且不遗漏全局缓存实例（如 `_MODEL_CACHE`、`_SKILLROUTER_BACKEND`、`_sr_embeddings_cache`）。

### Requirement: SkillRouter 后端分析

系统 SHALL 说明 `SkillRouterBackend` 的工作方式：

- Qwen3-0.6B 嵌入模型：`last_token_pool` + L2 归一化、查询前缀、批量 CPU 推理。
- Qwen3-0.6B 重排序器：chat template、yes/no logit 差分打分。
- 延迟加载、线程锁、异常传播与降级触发条件。

#### Scenario: 后端切换可理解

- **WHEN** 配置 `backend: skillrouter` 时
- **THEN** 文档应说明模型路径、加载时机、内存占用（~2.4GB）和推理延迟（~5s），以及与 lightweight 后端的差异。

### Requirement: 数据流与降级策略分析

系统 SHALL 绘制并说明一次 `search_skills` 调用的完整数据流，并列出所有降级路径：

1. SkillRouter 后端初始化失败 → lightweight。
2. 嵌入模型不可用 → 纯 BM25。
3. 向量编码超时 → 纯 BM25。
4. Reranker 不可用 → 跳过重排序。
5. 融合/重排序异常 → 返回空列表。

#### Scenario: 异常路径可追溯

- **WHEN** 线上出现“推荐为空”或“推荐质量差”时
- **THEN** 维护人员能根据文档快速定位当前流程处于哪个降级层级及其触发原因。

### Requirement: 域分组与检索空间裁剪分析

系统 SHALL 说明 `_classify_domain`、`_DOMAIN_MAP`、`_DOMAIN_KEYWORDS`、子域映射和域聚合器如何限制候选技能集合，以及其对 top_k、置信度和最终注入的影响。

#### Scenario: 域过滤逻辑可审计

- **WHEN** 查询命中 `finance`、`lark`、`finance-dianjin` 等域时
- **THEN** 文档应说明候选集如何被截断、聚合器如何被注入、以及 `uncategorized` 兜底机制。

### Requirement: 依赖与运行环境清单

系统 SHALL 列出所有运行时依赖、模型文件路径、SQLite 数据库位置、缓存文件路径，以及它们对 HOME 目录的隐含要求。

#### Scenario: 新环境可部署

- **WHEN** 在新机器上启用 skill-router 时
- **THEN** 能依据清单准备 Python 包、模型权重和数据库文件，并预判磁盘/内存开销。

### Requirement: 风险登记与改进建议

系统 SHALL 识别当前实现中的设计风险、并发风险、性能风险和可维护性风险，并按优先级给出改进建议（不强制在本次实现）。

#### Scenario: 风险可量化

- **WHEN** 评审分析文档
- **THEN** 每条风险应包含：位置、触发条件、影响面、建议措施、是否阻塞上线。

### Requirement: 评估与验证计划

系统 SHALL 说明 `scripts/evaluate.py` 的评估维度（纯向量 / BM25 / 混合、Top-1 / Top-3、置信度分布、错误对分析），并给出可执行的本地验证步骤。

#### Scenario: 变更后可复测

- **WHEN** 后续修改 skill-router 代码时
- **THEN** 能按文档运行评估脚本并判断准确率是否退化。

## MODIFIED Requirements

无。本次为只读分析，不修改现有功能规格。

## REMOVED Requirements

无。本次不删除任何功能。
