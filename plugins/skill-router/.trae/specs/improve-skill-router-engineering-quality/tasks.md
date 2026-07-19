# Tasks

- [x] Task 1: 创建子模块并迁移代码
  - [x] SubTask 1.1: 创建 `bm25_searcher.py`，迁移 `BM25Searcher` 及分词逻辑
  - [x] SubTask 1.2: 创建 `domain_config.py`，迁移 `_DEFAULT_DOMAIN_CONFIG`、加载/合并/校验、域分类过滤函数
  - [x] SubTask 1.3: 创建 `hybrid_searcher.py`，迁移 `HybridSearcher` 并加入分数校准
  - [x] SubTask 1.4: 创建 `state.py`，实现 `PluginState` 并迁移全局缓存与锁
  - [x] SubTask 1.5: 精简 `__init__.py`，通过动态导入使用子模块，保留 `register`/`search_skills`/`_pre_llm_call_hook`

- [x] Task 2: `evaluate.py` 复用 `BM25Searcher`
  - [x] SubTask 2.1: 删除 `BM25Evaluator`
  - [x] SubTask 2.2: 在 `evaluate.py` 中动态导入 `bm25_searcher.py` 的 `BM25Searcher`
  - [x] SubTask 2.3: 调整 BM25 评估函数，使用 `BM25Searcher.search`

- [x] Task 3: 混合检索分数校准
  - [x] SubTask 3.1: 在 `_DEFAULT_CONFIG` 中新增 `hybrid_calibration`（默认 `minmax`）
  - [x] SubTask 3.2: 在 `HybridSearcher` 中实现 `minmax`/`sigmoid`/`zscore` 校准
  - [x] SubTask 3.3: `search_skills` 读取配置并传入 `HybridSearcher`

- [x] Task 4: 补齐单元测试
  - [x] SubTask 4.1: 创建 `tests/test_hybrid_searcher.py`
  - [x] SubTask 4.2: 创建 `tests/test_state.py`
  - [x] SubTask 4.3: 创建 `tests/test_pre_llm_call.py`
  - [x] SubTask 4.4: 创建 `tests/test_feedback.py`

- [x] Task 5: 类型注解与静态检查
  - [x] SubTask 5.1: 为公共函数/类补充类型注解
  - [x] SubTask 5.2: 运行 `ruff check .` 并修复关键告警（已通过）
  - [x] SubTask 5.3: 运行 `mypy .` 并修复关键告警（无错误）

- [x] Task 6: 集成验证
  - [x] SubTask 6.1: 运行 `pytest tests/ -v`（54 passed, 1 warning）
  - [x] SubTask 6.2: 运行 `python scripts/evaluate.py`（默认 minmax），指标与优化前一致（Top-1 59.4%，Top-3 80.0%）
  - [x] SubTask 6.3: 运行 `python scripts/compare_calibrations.py`，`sigmoid`/`zscore` 分数分布与 minmax 明显不同

- [x] Task 7: 文档与清单
  - [x] SubTask 7.1: 更新 `README.md`，说明新增模块、配置项、静态检查命令
  - [x] SubTask 7.2: 更新 `tasks.md` 与 `checklist.md` 完成状态
  - [x] SubTask 7.3: 输出最终变更摘要

# Task Dependencies

- [Task 2] 依赖 [Task 1]
- [Task 3] 依赖 [Task 1]
- [Task 4] 依赖 [Task 1]、[Task 2]、[Task 3]
- [Task 5] 依赖 [Task 1]
- [Task 6] 依赖 [Task 1]、[Task 2]、[Task 3]、[Task 4]、[Task 5]
- [Task 7] 依赖 [Task 6]
