# Tasks

- [x] Task 1: 梳理插件元数据与目录结构
  - [x] SubTask 1.1: 读取并解读 `plugin.yaml`（name、version、kind、provides_tools、provides_hooks、hermes_compat）
  - [x] SubTask 1.2: 列出目录中所有文件职责（`__init__.py`、`skillrouter_backend.py`、`README.md`、`scripts/evaluate.py`、`.gitignore`）
  - [x] SubTask 1.3: 输出外部 Python 依赖清单（`sentence_transformers`、`transformers`、`torch`、`numpy`、`jieba`、`yaml` 等）

- [x] Task 2: 分析 `__init__.py` 核心模块
  - [x] SubTask 2.1: 缓存系统分析（`CacheManager`、`QueryCache`、全局缓存实例）
  - [x] SubTask 2.2: 反馈系统分析（`FeedbackStore`、JSONL 持久化、衰减加权、全局实例）
  - [x] SubTask 2.3: BM25 检索分析（`BM25Searcher`、分词策略、IDF 预计算）
  - [x] SubTask 2.4: 混合检索分析（`HybridSearcher`、分数归一化、CrossEncoder 重排序）
  - [x] SubTask 2.5: 端到端主流程分析（`search_skills`、双后端分支、域过滤、降级路径）
  - [x] SubTask 2.6: Hermes 接口分析（`register`、工具注册、事件订阅、`_pre_llm_call_hook`）

- [x] Task 3: 分析 `skillrouter_backend.py`
  - [x] SubTask 3.1: 嵌入模型加载与推理（`encode_texts`、`encode_query`、last_token_pool、L2 归一化）
  - [x] SubTask 3.2: 重排序模型加载与推理（`rerank`、chat template、yes/no logit 差分）
  - [x] SubTask 3.3: 延迟加载与线程安全（双锁、`_loaded` 状态、异常处理）
  - [x] SubTask 3.4: 与 `__init__.py` 的交互点（增量嵌入缓存 `_sr_embeddings_cache`、重排序调用）

- [x] Task 4: 分析 `scripts/evaluate.py`
  - [x] SubTask 4.1: 评估脚本入口与依赖（模型路径、数据库路径、`training_data.json`）
  - [x] SubTask 4.2: 三种检索模式对比逻辑（纯向量、纯 BM25、混合）
  - [x] SubTask 4.3: 置信度分布与错误对分析方法
  - [x] SubTask 4.4: 输出可执行的本地验证命令

- [x] Task 5: 绘制数据流与降级策略
  - [x] SubTask 5.1: 描述一次 `search_skills` 的完整调用链
  - [x] SubTask 5.2: 列出所有降级路径及触发条件
  - [x] SubTask 5.3: 描述 `register` 与 `pre_llm_call` 如何影响 LLM 上下文注入

- [x] Task 6: 整理风险清单与改进建议
  - [x] SubTask 6.1: 并发与线程安全风险（全局锁粒度、缓存一致性）
  - [x] SubTask 6.2: 性能与资源风险（模型加载阻塞、大模型内存、CPU 推理延迟）
  - [x] SubTask 6.3: 可维护性风险（硬编码域映射、全局状态、单测缺失）
  - [x] SubTask 6.4: 输出优先级排序的改进建议表

- [x] Task 7: 编写验证/审计清单并输出摘要
  - [x] SubTask 7.1: 将分析结论同步到 `checklist.md`
  - [x] SubTask 7.2: 复核所有 Requirement 是否被覆盖
  - [x] SubTask 7.3: 输出最终分析摘要给用户

# Task Dependencies

- [Task 2] 依赖 [Task 1]
- [Task 3] 依赖 [Task 1]
- [Task 4] 依赖 [Task 1]
- [Task 5] 依赖 [Task 2] 和 [Task 3]
- [Task 6] 依赖 [Task 2]、[Task 3] 和 [Task 5]
- [Task 7] 依赖 [Task 2]、[Task 3]、[Task 4] 和 [Task 6]
