# Checklist

- [x] `BM25Searcher` 已实现倒排索引
  - [x] `_inverted_index` 在 `__init__()` 中正确构建
  - [x] `search()` 基于 posting 列表累加得分，不再遍历全部文档
  - [x] 返回格式仍为 `List[Tuple[str, float]]`

- [x] `BM25Searcher` 已使用标准 BM25 IDF
  - [x] 公式为 `math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)`
  - [x] 空技能池、空查询、无命中词返回 `[]`

- [x] `scripts/evaluate.py` 已与线上逻辑一致
  - [x] `load_skills()` 返回完整 `description` 与 `body`
  - [x] `BM25Evaluator` 使用完整 body
  - [x] 向量评估使用完整 body
  - [x] `BM25Evaluator` 的 IDF 公式与 `BM25Searcher` 一致

- [x] 单元测试已创建并通过
  - [x] `tests/test_bm25_searcher.py` 存在并覆盖主要场景
  - [x] `tests/test_evaluate_consistency.py` 存在并覆盖一致性场景
  - [x] `pytest tests/ -v` 全部通过（8 passed）

- [x] 集成验证已完成
  - [x] `python scripts/evaluate.py` 在本地资源存在时已正常启动并进入评估流程（完整运行因耗时较长被用户跳过）
  - [x] `search_skills` 的 BM25 降级路径未被破坏（BM25Searcher 接口与返回格式保持不变）
  - [x] 未修改外部接口（`register`、`search_skills` 签名、`plugin.yaml`）

- [x] 文档与清单已更新
  - [x] `tasks.md` 所有任务已勾选
  - [x] 本 checklist 所有检查项已勾选
  - [x] 最终变更摘要已输出
