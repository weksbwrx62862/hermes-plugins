# Tasks

- [x] Task 1: 定义默认域配置 Schema 与数据结构
  - [x] SubTask 1.1: 在 `__init__.py` 中定义 `_DEFAULT_DOMAIN_CONFIG`，完整包含当前 `_DOMAIN_MAP`、`_DOMAIN_KEYWORDS`、`_DOMAIN_AGGREGATORS`、`_DOMAIN_TOP_K`、`_SUB_DOMAIN_MAP`、lark/finance 特殊规则
  - [x] SubTask 1.2: 设计 `subdomains` 结构，将原 `_SUB_DOMAIN_MAP` 的 prefixes 与 `search_skills` 内联 sub_keywords 合并为 `{subdomain: {keywords, prefixes}}`
  - [x] SubTask 1.3: 将 `lark` 前缀规则、`finance` 名称关键词与强制包含项抽象为 `special_rules`

- [x] Task 2: 实现配置加载与合并
  - [x] SubTask 2.1: 实现 `_load_domain_config()`，从 `plugins.skill-router.domain_config` 读取用户配置
  - [x] SubTask 2.2: 实现默认配置与用户配置的深度合并（顶层键覆盖，列表替换）
  - [x] SubTask 2.3: 确保配置通过现有 `_config_cache` 缓存，支持 300s 热更新

- [x] Task 3: 重构域过滤函数
  - [x] SubTask 3.1: 修改 `_classify_domain(query, domain_config)`，使用配置中的 `keywords`
  - [x] SubTask 3.2: 修改 `_get_domain_aggregator(domain, domain_config)`，使用配置中的 `aggregators`
  - [x] SubTask 3.3: 提取 `_filter_skills_by_domain(query, domain, skills, top_k, domain_config)`，替代 `search_skills` 中的内联过滤逻辑
  - [x] SubTask 3.4: 在 `_filter_skills_by_domain` 中实现 `special_rules`（lark 前缀、finance 名称关键词、强制包含）和 `subdomains` 过滤
  - [x] SubTask 3.5: 在 `search_skills` 中调用 `_load_domain_config()` 并传入过滤函数；移除对旧全局常量的直接引用
  - [x] SubTask 3.6: 删除或私有化旧全局常量

- [x] Task 4: 实现配置校验
  - [x] SubTask 4.1: 实现 `_validate_domain_config(domain_config, skills)`
  - [x] SubTask 4.2: 对不存在的聚合器技能名输出 `warning`
  - [x] SubTask 4.3: 对无匹配技能名的子域前缀输出 `warning`
  - [x] SubTask 4.4: 在 `search_skills` 加载技能索引后调用校验函数

- [x] Task 5: 新增回归测试
  - [x] SubTask 5.1: 创建 `tests/test_domain_config.py`
  - [x] SubTask 5.2: 测试默认配置等价于旧常量
  - [x] SubTask 5.3: 测试自定义 `keywords` 改变 `_classify_domain` 结果
  - [x] SubTask 5.4: 测试自定义 `map` 改变候选集
  - [x] SubTask 5.5: 测试 `lark`/`finance` 特殊规则可配置
  - [x] SubTask 5.6: 测试子域 `keywords + prefixes` 过滤
  - [x] SubTask 5.7: 测试未知聚合器触发 warning

- [x] Task 6: 运行验证
  - [x] SubTask 6.1: 运行 `pytest tests/test_domain_config.py -v`（16 passed）
  - [x] SubTask 6.2: 运行 `pytest tests/ -v`（24 passed, 1 warning）
  - [x] SubTask 6.3: 运行 `python scripts/evaluate.py`，默认配置下结果与优化前一致，可用于后续调整关键词后对比高频错误对

- [x] Task 7: 更新文档与清单
  - [x] SubTask 7.1: 在 `README.md` 中补充 `domain_config` 配置示例
  - [x] SubTask 7.2: 更新 `tasks.md` 与 `checklist.md` 完成状态
  - [x] SubTask 7.3: 输出最终变更摘要

# Task Dependencies

- [Task 2] 依赖 [Task 1]
- [Task 3] 依赖 [Task 1] 和 [Task 2]
- [Task 4] 依赖 [Task 1]
- [Task 5] 依赖 [Task 1]、[Task 3] 和 [Task 4]
- [Task 6] 依赖 [Task 3]、[Task 4] 和 [Task 5]
- [Task 7] 依赖 [Task 6]
