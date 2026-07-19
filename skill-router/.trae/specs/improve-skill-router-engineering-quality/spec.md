# skill-router P2 工程质量全面提升 Spec

## Why

P0 与 P1 优化完成后，skill-router 在性能和域配置灵活性上已有明显改善，但代码组织层面仍存在以下工程债务：

1. `__init__.py` 超过 1700 行，检索、配置、域过滤、反馈、hook 逻辑混杂，维护成本高。
2. `scripts/evaluate.py` 维护了一套独立的 `BM25Evaluator`，与 `__init__.py` 中的 `BM25Searcher` 实现重复，未来调整分词/IDF 时容易分叉。
3. 模型缓存、后端缓存、查询缓存、嵌入缓存等全局状态分散且依赖隐式全局变量，单测难以注入，也存在重复加载风险。
4. `HybridSearcher` 使用 min-max 归一化，导致混合检索 Top-1 置信度全部落在 high 区间（100% >= 0.4），分数可解释性差。
5. 核心路径（HybridSearcher、重排序、pre_llm_call 置信度过滤、反馈学习）缺乏单元测试覆盖。
6. 类型注解不完整，静态检查未纳入常规验证。

本次优化一次性解决以上 P2 问题，在保持对外接口不变的前提下提升可维护性、可测试性和可解释性。

## What Changes

### 1. 模块化拆分

将 `__init__.py` 中的内聚逻辑抽取为以下子模块（均放在插件根目录）：

- `bm25_searcher.py`：`BM25Searcher` 类及分词逻辑。
- `domain_config.py`：`_DEFAULT_DOMAIN_CONFIG`、配置加载/合并/校验、域分类与过滤函数。
- `hybrid_searcher.py`：`HybridSearcher` 类及可选分数校准逻辑。
- `state.py`：`PluginState` 类，集中管理所有全局缓存与锁。

`__init__.py` 保留为插件入口：

- 通过动态导入助手 `__init__.py` 加载上述子模块（避免插件加载器对包语义的依赖）。
- 保留 `register()`、`search_skills()`、`_pre_llm_call_hook()` 等 Hermes 需要的入口函数/Hook。
- 外部行为不变。

### 2. `evaluate.py` 复用 `BM25Searcher`

- 删除 `scripts/evaluate.py` 中的 `BM25Evaluator` 类。
- `evaluate.py` 通过动态导入使用 `bm25_searcher.py` 中的 `BM25Searcher`。
- 确保离线评估的 BM25 实现与线上完全一致。

### 3. 全局状态收拢到 `PluginState`

- 新建 `PluginState` 类，集中持有：
  - `_model_cache` 与 `_model_lock`
  - `_skillrouter_backend` 与 `_backend_lock`
  - `_query_cache`
  - `_sr_embeddings_cache` 与锁
  - `_embedding_cache`、`_bm25_cache`、`_config_cache`、`_skill_index_cache`
- 模块级全局实例 `plugin_state = PluginState()`，所有函数通过该实例访问缓存，便于单测替换。

### 4. 混合检索分数校准

- 在 `HybridSearcher` 中增加可配置的分数校准策略，通过 `hybrid_calibration` 配置项控制：
  - `minmax`：默认，保持现有行为。
  - `sigmoid`：对 BM25 分数做 sigmoid 压缩，使融合分数分布更稳定。
  - `zscore`：使用全局均值/标准差标准化。
- 默认保持 `minmax`，不破坏现有置信度阈值。

### 5. 补齐单元测试

新增/扩展以下测试文件：

- `tests/test_hybrid_searcher.py`：测试融合排序、分数校准策略、空输入、单分支缺失。
- `tests/test_state.py`：测试 `PluginState` 缓存命中/失效、锁的基本行为。
- `tests/test_pre_llm_call.py`：测试置信度过滤（low/medium/high）、core/pool 注入、域推荐。
- `tests/test_feedback.py`：测试反馈读取/写入/评分调整。

### 6. 类型注解与静态检查

- 为公共函数和关键类补充类型注解。
- 运行 `ruff check .` 与 `mypy .`，修复关键告警（未使用变量、缺失返回类型、危险默认参数等）。
- 不强制一次性消灭所有告警，优先修复可能导致运行时错误的项。

## Impact

- Affected specs：skill-router v3.1 全部内部实现，但对外接口不变。
- Affected code：
  - `/home/xxh/.hermes/plugins/skill-router/__init__.py`（大幅精简）
  - 新增 `/home/xxh/.hermes/plugins/skill-router/bm25_searcher.py`
  - 新增 `/home/xxh/.hermes/plugins/skill-router/domain_config.py`
  - 新增 `/home/xxh/.hermes/plugins/skill-router/hybrid_searcher.py`
  - 新增 `/home/xxh/.hermes/plugins/skill-router/state.py`
  - 修改 `/home/xxh/.hermes/plugins/skill-router/scripts/evaluate.py`
  - 新增/扩展测试文件
  - 修改 `/home/xxh/.hermes/plugins/skill-router/README.md`
- **BREAKING**：
  - 内部全局常量与部分内部函数位置变化；外部代码不应直接依赖 skill-router 内部符号。
  - `hybrid_calibration` 默认 `minmax`，显式启用 `sigmoid`/`zscore` 会改变分数分布，属于可选行为变更。

## ADDED Requirements

### Requirement: 模块化拆分

系统 SHALL 将 `BM25Searcher`、域配置与过滤、`HybridSearcher`、全局状态分别拆分为独立模块，且 `__init__.py` 仍能被 Hermes 插件加载器正常加载。

#### Scenario: 代码维护

- **WHEN** 开发者修改 BM25 分词逻辑时
- **THEN** 只需编辑 `bm25_searcher.py`，无需在 1700 行的 `__init__.py` 中定位。

### Requirement: evaluate.py 复用 BM25Searcher

系统 SHALL 删除 `scripts/evaluate.py` 中的 `BM25Evaluator`，改为使用 `bm25_searcher.py` 中的 `BM25Searcher`。

#### Scenario: 评估一致性

- **WHEN** 运行 `python scripts/evaluate.py` 时
- **THEN** 离线 BM25 实现与线上 `BM25Searcher` 完全一致。

### Requirement: 全局状态收拢

系统 SHALL 提供 `PluginState` 类集中管理所有全局缓存与锁，并通过单一模块级实例 `plugin_state` 访问。

#### Scenario: 单测注入

- **WHEN** 测试需要替换模型缓存时
- **THEN** 可以直接替换 `plugin_state._model_cache`，无需修改全局变量。

### Requirement: 混合检索分数校准

系统 SHALL 支持通过配置 `hybrid_calibration` 选择分数校准策略（`minmax`/`sigmoid`/`zscore`），默认 `minmax`。

#### Scenario: 改善分数分布

- **WHEN** 配置 `hybrid_calibration: sigmoid` 时
- **THEN** 混合检索 Top-1 分数分布应不再全部落在 `>=0.4`，且 Top-1/Top-3 准确率不显著下降。

### Requirement: 核心路径单元测试

系统 SHALL 为 `HybridSearcher`、分数校准、`PluginState`、`pre_llm_call` 置信度过滤、反馈学习提供单元测试。

#### Scenario: 回归防护

- **WHEN** 运行 `pytest tests/ -v` 时
- **THEN** 上述新增测试与已有测试全部通过。

### Requirement: 类型注解与静态检查

系统 SHALL 为公共函数补充类型注解，并运行 `ruff check .` 与 `mypy .`，修复关键告警。

#### Scenario: 持续集成

- **WHEN** 提交代码后
- **THEN** `ruff check .` 与 `mypy .` 不应报告运行时风险类错误。

## MODIFIED Requirements

### Requirement: 插件入口结构

- **原实现**：所有逻辑集中在 `__init__.py`。
- **修改后**：`__init__.py` 作为入口 facade，逻辑拆分到子模块。
- **接口不变**：`register()`、`search_skills()`、`_pre_llm_call_hook()` 签名与行为不变。

### Requirement: 混合检索分数计算

- **原实现**：强制 min-max 归一化。
- **修改后**：支持可选校准策略，默认 min-max。
- **配置项**：`hybrid_calibration`。

## REMOVED Requirements

无。本次不删除用户可见功能。
