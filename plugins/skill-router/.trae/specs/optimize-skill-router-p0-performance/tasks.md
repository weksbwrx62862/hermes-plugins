# Tasks

- [x] Task 1: 实现 BM25 倒排索引与标准 IDF
  - [x] SubTask 1.1: 在 `BM25Searcher.__init__()` 中构建 `_inverted_index: Dict[str, List[Tuple[str, int]]]`
  - [x] SubTask 1.2: 将 IDF 公式改为 `math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)`
  - [x] SubTask 1.3: 重写 `BM25Searcher.search()`，基于 posting 列表累加得分，保持返回格式 `(name, score)` 不变
  - [x] SubTask 1.4: 验证空技能池、空查询、无命中词的边界行为

- [x] Task 2: 同步更新 `scripts/evaluate.py`
  - [x] SubTask 2.1: 修改 `load_skills()`，返回完整 `description` 与 `body`（不再 `[:500]`）
  - [x] SubTask 2.2: 修改 `BM25Evaluator.__init__()`，使用完整 `body`
  - [x] SubTask 2.3: 修改向量评估中的 `skill_texts`，使用完整 `body`（不再 `[:200]`）
  - [x] SubTask 2.4: 同步 `BM25Evaluator` 的 IDF 公式与 `BM25Searcher` 一致

- [x] Task 3: 新增回归测试
  - [x] SubTask 3.1: 创建 `tests/test_bm25_searcher.py`
    - [x] 测试正常检索返回 `(name, score)` 列表
    - [x] 测试倒排索引版与暴力版在相同 IDF 下的排序一致性
    - [x] 测试空查询、空技能池、无命中返回 `[]`
  - [x] SubTask 3.2: 创建 `tests/test_evaluate_consistency.py`
    - [x] 使用内存中 SQLite 与 mock 训练数据验证 `load_skills()` 不截断 body
    - [x] 验证 `BM25Evaluator` 与 `BM25Searcher` 在相同输入下分数排序一致
  - [x] SubTask 3.3: 配置 `pytest` 运行环境，确保 `tests/` 可被正常发现

- [x] Task 4: 运行验证
  - [x] SubTask 4.1: 在插件目录运行 `pytest tests/ -v`（8 个测试全部通过）
  - [x] SubTask 4.2: 运行 `python scripts/evaluate.py`：本地模型与数据均存在，脚本已正常启动并进入评估流程；完整运行因耗时较长被用户跳过
  - [x] SubTask 4.3: 通过回归测试确认 `search_skills` 的 BM25 降级路径未被破坏（BM25Searcher 接口与返回格式保持不变）

- [x] Task 5: 更新文档与清单
  - [x] SubTask 5.1: README 已包含足够的使用说明，本次不做额外修改
  - [x] SubTask 5.2: 更新 `checklist.md` 所有检查项为完成状态
  - [x] SubTask 5.3: 输出最终变更摘要

# Task Dependencies

- [Task 2] 依赖 [Task 1]
- [Task 3] 依赖 [Task 1] 和 [Task 2]
- [Task 4] 依赖 [Task 1]、[Task 2] 和 [Task 3]
- [Task 5] 依赖 [Task 4]
