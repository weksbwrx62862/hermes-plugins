# skill-router P0 性能与一致性优化 Spec

## Why

根据 `analyze-skill-router-plugin` 分析报告，`skill-router` 的 BM25 实现存在两项关键问题：

1. `BM25Searcher.search()` 每次查询都遍历全部文档重新统计 TF，时间复杂度随技能池线性增长，是明显的性能瓶颈。
2. `BM25Searcher` 的 IDF 公式未取对数，与标准 BM25 不一致，导致分数不可解释；同时 `scripts/evaluate.py` 中文档文本截断为 `body[:500]` / `body[:200]`，与线上使用完整 body 的逻辑不一致，离线评估结果无法准确反映线上效果。

本次优化聚焦以上 P0 问题，在保持对外接口不变的前提下提升检索性能与可解释性。

## What Changes

- 在 `__init__.py` 的 `BM25Searcher` 中预计算倒排索引（`term -> [(name, tf), ...]`），将 `search()` 的时间复杂度从 `O(n_docs * avg_dl)` 降至 `O(n_query_terms * avg_posting_len)`。
- 将 `BM25Searcher` 的 IDF 公式改为标准 BM25 对数形式：`log((N - df + 0.5) / (df + 0.5) + 1)`。
- 同步修改 `scripts/evaluate.py` 中的 `BM25Evaluator`：
  - 使用完整 `description` 与 `body`（不再截断），与线上 `BM25Searcher` 保持一致。
  - 向量评估同样使用完整 body。
- 新增 `tests/test_bm25_searcher.py` 与 `tests/test_evaluate_consistency.py`，覆盖 BM25 检索正确性、倒排索引等价性与评估脚本文本构造一致性。
- 不修改 `search_skills`、`register`、`_pre_llm_call_hook` 等外部接口；不修改 `plugin.yaml` 与默认配置。

## Impact

- Affected specs：skill-router v3.1 的 lightweight 后端检索、降级路径、离线评估。
- Affected code：
  - `/home/xxh/.hermes/plugins/skill-router/__init__.py`（`BM25Searcher`）
  - `/home/xxh/.hermes/plugins/skill-router/scripts/evaluate.py`（`BM25Evaluator`、向量评估文本构造）
  - 新增 `/home/xxh/.hermes/plugins/skill-router/tests/test_bm25_searcher.py`
  - 新增 `/home/xxh/.hermes/plugins/skill-router/tests/test_evaluate_consistency.py`
- **BREAKING**：BM25 IDF 公式的修改会改变 BM25 绝对分数，但仅影响内部排序与融合权重；由于 `HybridSearcher` 会对 BM25 分数重新归一化，对最终用户体验的影响可控。将在测试中验证排序一致性不显著退化。

## ADDED Requirements

### Requirement: BM25 倒排索引性能优化

系统 SHALL 在 `BM25Searcher.__init__()` 中预计算倒排索引，使得 `search()` 只需要遍历查询词项对应的 posting 列表即可计算得分。

#### Scenario: 大技能池下的检索延迟

- **WHEN** 技能池包含大量技能（例如 1000+）时
- **THEN** `BM25Searcher.search()` 的延迟不应随文档总量线性增长，而应主要取决于查询词项的 posting 列表长度。

### Requirement: 标准 BM25 IDF

系统 SHALL 使用标准 BM25 IDF 公式：

```python
idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
```

替换当前未取对数的实现。

#### Scenario: 分数可解释

- **WHEN** 调试或评估 BM25 分数时
- **THEN** 分数应与标准 BM25 实现处于同一数量级，便于与文献/其他工具对比。

### Requirement: 评估脚本与线上逻辑一致

系统 SHALL 保证 `scripts/evaluate.py` 构造文档文本的方式与线上 `BM25Searcher` / `search_skills` 一致，即使用完整 `description` 与 `body_text`，不再截断。

#### Scenario: 离线评估反映线上效果

- **WHEN** 运行 `python scripts/evaluate.py` 时
- **THEN** 其使用的技能文本应与线上推理完全一致，使得 Top-1/Top-3 准确率、置信度分布等指标可用于判断线上回归。

### Requirement: 回归测试

系统 SHALL 提供单元测试，验证：

- 倒排索引版本与原始暴力遍历版本的排序结果一致（允许 IDF 公式变更带来的微小差异，但 Top-K 顺序应高度一致）。
- `BM25Searcher` 对空查询、空技能池、不存在的词项能正确返回空列表。
- `evaluate.py` 加载技能时不再截断 body。

#### Scenario: 变更安全

- **WHEN** 提交优化代码后
- **THEN** 运行 `pytest tests/` 应全部通过，且 `scripts/evaluate.py` 能正常输出评估结果。

## MODIFIED Requirements

### Requirement: BM25Searcher 内部实现

- **原实现**：遍历所有文档统计 TF。
- **修改后**：基于预计算倒排索引累加得分。
- **接口不变**：`__init__(skills)` 与 `search(query, top_k=10)` 的签名、返回值格式不变。

## REMOVED Requirements

无。
