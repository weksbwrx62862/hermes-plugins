# skill-router 域分组配置化 Spec

## Why

离线评估显示，当前 skill-router 的高频错误主要集中在域内误召回：

- `finance` 被错判为 `stock-quote-analysis`（14 次）
- `crypto-analysis` 与 `crypto` 混淆（8 次）
- `portfolio-manager` 与资产配置类技能互相误召回
- `codebase-inspection` 与 `improve-codebase-architecture` 混淆

这些错误的共同根因是：`_DOMAIN_MAP`、`_DOMAIN_KEYWORDS`、`_SUB_DOMAIN_MAP`、`_DOMAIN_AGGREGATORS` 以及 `lark`/`finance` 的特殊过滤逻辑全部硬编码在 `__init__.py` 中。业务新增域、调整关键词或修正候选集时，必须修改代码并重新部署。本次优化将域分组相关映射全部抽取到 `~/.hermes/config.yaml`，实现热更新（通过现有 300s 配置缓存）和按业务定制，降低高频错误对的回归成本。

## What Changes

- 在 `__init__.py` 中新增 `_DEFAULT_DOMAIN_CONFIG` 字典，包含当前所有硬编码域映射的默认值：
  - `keywords`：域关键词映射
  - `map`：域 → category 列表映射
  - `top_k`：域内候选截断上限（默认 3）
  - `aggregators`：域聚合器入口技能
  - `subdomains`：子域关键词 + 前缀过滤（替换 `_SUB_DOMAIN_MAP` 与内联 sub_keywords）
  - `special_rules`：`lark` 前缀过滤、`finance` 名称关键词兜底与强制包含项
- 新增 `_load_domain_config()`，从 `plugins.skill-router.domain_config` 读取用户配置并与默认值深度合并（顶层键覆盖，列表替换）。
- 新增 `_validate_domain_config(domain_config, skills)`，对聚合器技能名、子域前缀做存在性校验，缺失时输出 `warning` 日志。
- 将 `_classify_domain`、`_get_domain_aggregator`、`_filter_skills_by_domain` 改为接收 `domain_config`，不再依赖全局常量。
- 删除（或私有化）原有全局常量：`_DOMAIN_MAP`、`_DOMAIN_KEYWORDS`、`_DOMAIN_AGGREGATORS`、`_DOMAIN_TOP_K`、`_SUB_DOMAIN_MAP`、`_init_sub_domain_map`。
- 新增 `tests/test_domain_config.py`，覆盖：
  - 默认配置与旧常量语义一致
  - 自定义 keywords/map 生效
  - 子域前缀过滤生效
  - 未知聚合器触发 warning
- 不修改 `plugin.yaml`、不修改 `search_skills` / `register` / `_pre_llm_call_hook` 的外部签名；`search_skills` 内部仅把 `domain_config` 传入过滤函数。

## Impact

- Affected specs：skill-router v3.1 的域分类、候选集裁剪、`pre_llm_call` 的推荐域列表。
- Affected code：
  - `/home/xxh/.hermes/plugins/skill-router/__init__.py`（域映射常量、过滤函数、配置加载）
  - 新增 `/home/xxh/.hermes/plugins/skill-router/tests/test_domain_config.py`
- **BREAKING**：用户若已在 `~/.hermes/config.yaml` 中自定义了 `plugins.skill-router` 的任意字段，新增 `domain_config` 不会冲突；但如果未来版本继续依赖旧全局常量，必须改为读取配置。本次实现会移除这些常量，因此任何外部直接导入使用的行为会失效（skill-router 内部已统一处理）。

## ADDED Requirements

### Requirement: 域配置Schema

系统 SHALL 支持在 `~/.hermes/config.yaml` 的 `plugins.skill-router.domain_config` 下配置域分组映射，Schema 如下：

```yaml
plugins:
  skill-router:
    domain_config:
      top_k: 3                       # 可选，默认 3
      keywords:                      # 可选，默认使用当前硬编码值
        finance: ["股票", "基金", "投资", ...]
      map:                           # 可选，默认使用当前硬编码值
        finance: ["finance"]
      aggregators:                   # 可选
        finance: "finance-assistant"
      subdomains:                    # 可选
        finance-dianjin:
          corporate-banker:
            keywords: ["尽调", "贷前", ...]
            prefixes: ["credit-due-diligence", ...]
      special_rules:                 # 可选
        lark:
          name_prefix: "lark-"
        finance:
          name_keywords: ["stock", "fund", ...]
          always_include: ["finance-assistant"]
```

#### Scenario: 业务新增域

- **WHEN** 运维人员在配置文件中新增一个域的 `keywords`、`map`、`aggregators`
- **THEN** `search_skills` 应能在不修改代码的情况下识别该域并限制候选集。

### Requirement: 配置默认值兼容

系统 SHALL 当用户未提供 `domain_config` 或只提供部分键时，使用与当前硬编码逻辑完全等价的默认值，确保不配置时行为不变。

#### Scenario: 无自定义配置

- **WHEN** `~/.hermes/config.yaml` 中不存在 `plugins.skill-router.domain_config`
- **THEN** `_load_domain_config()` 返回的默认值应产生与优化前完全相同的域分类和过滤结果。

### Requirement: 配置热更新

系统 SHALL 复用现有的 `_config_cache`（TTL 300s），配置修改后 300s 内自动生效；显式触发 `skill_updated` 事件时应立即使配置缓存失效。

#### Scenario: 调整关键词后生效

- **WHEN** 用户修改 `domain_config.keywords.finance` 并保存配置文件
- **THEN** 最多 300s 后，新的 finance 关键词应被用于域分类。

### Requirement: 域过滤函数参数化

系统 SHALL 将 `_classify_domain`、`_get_domain_aggregator`、`_filter_skills_by_domain` 改为接收 `domain_config` 参数，不再直接读取全局常量。

#### Scenario: 单元测试可注入配置

- **WHEN** 测试传入自定义 `domain_config`
- **THEN** 域分类与过滤结果应仅由传入配置决定，不受默认常量影响。

### Requirement: 配置校验

系统 SHALL 在 `search_skills` 加载技能索引后调用 `_validate_domain_config(domain_config, skills)`，对以下情况输出 `warning`：

- `aggregators` 中指定的技能名不存在于当前索引。
- `subdomains.*.prefixes` 中没有任何一个前缀能匹配到当前索引中的技能名。

#### Scenario: 配置错误可发现

- **WHEN** 用户配置的聚合器技能名拼写错误
- **THEN** 日志中应出现 `warning`，提示该聚合器不存在，而不是静默注入失败。

### Requirement: 回归测试

系统 SHALL 提供单元测试覆盖：

- 默认配置等价于旧常量。
- 自定义 keywords 改变域分类结果。
- 自定义 map 改变候选集。
- `lark` 前缀规则、`finance` 名称关键词规则可配置。
- 子域 keywords + prefixes 过滤生效。
- 未知聚合器触发 warning。

#### Scenario: 变更安全

- **WHEN** 运行 `pytest tests/test_domain_config.py -v`
- **THEN** 所有测试通过，且 `search_skills` 在默认配置下的行为与优化前一致。

## MODIFIED Requirements

### Requirement: 域过滤内部实现

- **原实现**：依赖全局常量 `_DOMAIN_MAP`、`_DOMAIN_KEYWORDS`、`_DOMAIN_AGGREGATORS`、`_DOMAIN_TOP_K`、`_SUB_DOMAIN_MAP` 和内联特殊逻辑。
- **修改后**：通过 `_load_domain_config()` 加载配置，过滤函数接收 `domain_config`。
- **接口不变**：`search_skills`、`register`、`_pre_llm_call_hook` 的对外签名不变。

## REMOVED Requirements

无。本次不删除用户可见功能，仅移除内部全局常量。
