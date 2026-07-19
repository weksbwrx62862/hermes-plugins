# Checklist

- [x] 模块化拆分已完成
  - [x] `bm25_searcher.py` 存在且包含 `BM25Searcher`
  - [x] `domain_config.py` 存在且包含配置与过滤逻辑
  - [x] `hybrid_searcher.py` 存在且包含 `HybridSearcher` 与校准逻辑
  - [x] `state.py` 存在且包含 `PluginState`
  - [x] `__init__.py` 能正常被 Hermes 加载，`register`/`search_skills`/`_pre_llm_call_hook` 行为不变

- [x] `evaluate.py` 已复用 `BM25Searcher`
  - [x] `BM25Evaluator` 已删除
  - [x] 评估脚本的 BM25 结果与线上 `BM25Searcher` 一致

- [x] 混合检索分数校准已实现
  - [x] `hybrid_calibration` 配置项存在（默认 `minmax`）
  - [x] `sigmoid`/`zscore` 策略已实现
  - [x] 默认策略下 `evaluate.py` 指标与优化前一致

- [x] 单元测试已补齐
  - [x] `tests/test_hybrid_searcher.py` 存在并通过
  - [x] `tests/test_state.py` 存在并通过
  - [x] `tests/test_pre_llm_call.py` 存在并通过
  - [x] `tests/test_feedback.py` 存在并通过

- [x] 类型注解与静态检查已完成
  - [x] 关键公共函数已补充类型注解
  - [x] `ruff check .` 无关键运行时风险告警
  - [x] `mypy .` 无关键类型错误

- [x] 集成验证已通过
  - [x] `pytest tests/ -v` 全部通过（54 passed, 1 warning）
  - [x] `python scripts/evaluate.py` 默认策略下运行正常（Top-1 59.4%，Top-3 80.0%）
  - [x] 校准策略切换后可观察到分数分布变化

- [x] 文档与清单已更新
  - [x] `README.md` 已更新
  - [x] `tasks.md` 所有任务已勾选
  - [x] 本 checklist 所有检查项已勾选
  - [x] 最终变更摘要已输出
