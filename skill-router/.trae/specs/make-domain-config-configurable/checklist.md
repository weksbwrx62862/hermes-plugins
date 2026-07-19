# Checklist

- [x] `_DEFAULT_DOMAIN_CONFIG` 已定义并覆盖全部旧常量
  - [x] 包含 `keywords`、`map`、`top_k`、`aggregators`、`subdomains`、`special_rules`
  - [x] `subdomains` 结构同时包含子域分类 `keywords` 和过滤 `prefixes`
  - [x] `special_rules` 覆盖 `lark` 前缀、`finance` 名称关键词与强制包含

- [x] 配置加载与合并已实现
  - [x] `_load_domain_config()` 从 `plugins.skill-router.domain_config` 读取
  - [x] 未提供配置时返回完整默认值
  - [x] 复用 `_config_cache` 300s TTL

- [x] 域过滤函数已参数化
  - [x] `_classify_domain(query, domain_config)` 不依赖旧全局常量
  - [x] `_get_domain_aggregator(domain, domain_config)` 不依赖旧全局常量
  - [x] `_filter_skills_by_domain(query, domain, skills, top_k, domain_config)` 已提取
  - [x] `search_skills` 中调用新的过滤函数并传入 `domain_config`

- [x] 配置校验已实现
  - [x] `_validate_domain_config(domain_config, skills)` 存在
  - [x] 未知聚合器技能名触发 `warning`
  - [x] 无匹配技能的子域前缀触发 `warning`

- [x] 回归测试已创建并通过
  - [x] `tests/test_domain_config.py` 存在
  - [x] 默认配置等价于旧常量
  - [x] 自定义 keywords/map 生效
  - [x] 特殊规则可配置
  - [x] 子域过滤生效
  - [x] 未知聚合器触发 warning

- [x] 集成验证已完成
  - [x] `pytest tests/test_domain_config.py -v` 全部通过（16 passed）
  - [x] `pytest tests/ -v` 全部通过（24 passed）
  - [x] `python scripts/evaluate.py` 可运行，默认配置下评估指标与优化前一致

- [x] 文档与清单已更新
  - [x] `README.md` 已补充 `domain_config` 配置示例
  - [x] `tasks.md` 所有任务已勾选
  - [x] 本 checklist 所有检查项已勾选
  - [x] 最终变更摘要已输出
