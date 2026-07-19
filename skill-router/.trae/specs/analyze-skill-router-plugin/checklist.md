# Checklist

- [x] 插件元数据与目录结构已完整记录
  - [x] `plugin.yaml` 中的 name、version、kind、provides_tools、provides_hooks、hermes_compat 已说明
  - [x] 各文件（`__init__.py`、`skillrouter_backend.py`、`README.md`、`scripts/evaluate.py`）职责已划分
  - [x] 外部 Python 依赖与模型文件路径已列出

- [x] `__init__.py` 核心模块分析已完成
  - [x] `CacheManager` / `QueryCache` 的 TTL / LRU 策略与线程安全已说明
  - [x] `FeedbackStore` 的 JSONL 持久化、衰减加权与全局实例已说明
  - [x] `BM25Searcher` 的分词、IDF 预计算与搜索流程已说明
  - [x] `HybridSearcher` 的归一化融合、CrossEncoder 重排序与类级缓存已说明
  - [x] `search_skills` 的主流程、双后端分支、域过滤与降级路径已说明
  - [x] `register` 与 `_pre_llm_call_hook` 的工具注册、事件订阅、上下文注入已说明

- [x] `skillrouter_backend.py` 分析已完成
  - [x] 嵌入模型的 last_token_pool + L2 归一化、查询前缀、批量推理已说明
  - [x] 重排序器的 chat template、yes/no logit 差分已说明
  - [x] 延迟加载、双锁线程安全、异常传播已说明
  - [x] 与 `__init__.py` 中 `_sr_embeddings_cache` 的协作已说明

- [x] 数据流与降级策略已梳理
  - [x] 一次 `search_skills` 的完整调用链已描述
  - [x] SkillRouter 初始化失败、嵌入模型不可用、编码超时、Reranker 不可用、融合异常五条降级路径已列出
  - [x] `pre_llm_call` 上下文注入的 core/pool 分级逻辑已描述

- [x] 域分组与检索空间裁剪已分析
  - [x] `_classify_domain`、`_DOMAIN_MAP`、`_DOMAIN_KEYWORDS` 的映射关系已说明
  - [x] `finance-dianjin` 子域、`lark` 前缀过滤、域聚合器注入逻辑已说明
  - [x] `_DOMAIN_TOP_K` 截断与 `uncategorized` 兜底机制已说明

- [x] 依赖与运行环境清单已整理
  - [x] Python 第三方包依赖已列出
  - [x] 模型权重路径、SQLite 数据库路径、缓存文件路径已列出
  - [x] 运行时内存/磁盘开销与 HOME 目录依赖已说明

- [x] 风险登记与改进建议已输出
  - [x] 并发与线程安全风险已识别
  - [x] 性能与资源风险已识别
  - [x] 可维护性风险已识别
  - [x] 改进建议已按优先级排序

- [x] 评估与验证计划已制定
  - [x] `scripts/evaluate.py` 的评估维度已说明
  - [x] 本地可执行的验证命令已给出
  - [x] 准确率、置信度分布、错误对分析的判读方法已说明

- [x] 文档一致性与完整性已复核
  - [x] 所有 ADDED Requirement 在 checklist 中均有对应条目
  - [x] 无代码变更，不会引入运行时回归
  - [x] 最终摘要已准备就绪
