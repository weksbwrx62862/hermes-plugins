# skill-router 插件详细分析报告

> 分析范围：`/home/xxh/.hermes/plugins/skill-router`
> 版本：v3.1.0
> 分析性质：只读逆向分析，不修改源码

---

## 1. 插件元数据与目录结构

### 1.1 目录结构

```
skill-router/
├── __init__.py              # 主插件：路由逻辑、缓存、反馈、pre_llm_call 钩子
├── skillrouter_backend.py   # SkillRouter（Qwen3-0.6B）推理后端
├── plugin.yaml              # 插件元数据
├── README.md                # 用户文档
├── .gitignore
└── scripts/
    └── evaluate.py          # 离线评估脚本
```

### 1.2 plugin.yaml 解读

| 字段 | 值 | 说明 |
|------|-----|------|
| `name` | `skill-router` | 插件标识 |
| `version` | `3.1.0` | 当前版本 |
| `description` | 统一技能路由 v3.1 | 双后端 + 完整 body 嵌入 + core/pool 分层 |
| `kind` | `backend` | 后端插件 |
| `author` | `hermes-agent` | |
| `hermes_compat` | `>=2.0.0` | 兼容 Hermes 版本 |
| `provides_tools` | `skill_search`、`skill_feedback` | 暴露给 LLM/调用方的工具 |
| `provides_hooks` | `pre_llm_call` | 在 LLM 调用前注入技能上下文 |

### 1.3 文件职责

- `__init__.py`：路由主逻辑。包含缓存、反馈、BM25、混合检索、域分组、双后端调度、Hermes 注册与 `pre_llm_call` 钩子。
- `skillrouter_backend.py`：SkillRouter 后端封装。提供基于 Qwen3-0.6B 的嵌入与重排序，延迟加载、线程安全。
- `README.md`：面向使用者的配置说明、降级策略、模型存放路径。
- `scripts/evaluate.py`：离线评估脚本，对比纯向量 / 纯 BM25 / 混合检索的 Top-1/Top-3 准确率。

---

## 2. 外部依赖与运行环境

### 2.1 Python 第三方依赖

| 包 | 用途 | 是否必需 |
|----|------|----------|
| `sentence_transformers` | lightweight 后端 SentenceTransformer 与 CrossEncoder | 后端为 lightweight 时必需 |
| `transformers` | SkillRouter 后端加载 Qwen3-0.6B | 后端为 skillrouter 时必需 |
| `torch` | 两个后端的张量运算 | 必需 |
| `numpy` | 向量点积、堆叠 | 必需 |
| `jieba` | BM25 中文分词 | 可选，缺失时降级为字符级分词 |
| `yaml` | 读取 `~/.hermes/config.yaml` | 配置存在时必需 |

### 2.2 模型与数据文件路径

| 资源 | 默认路径 | 说明 |
|------|----------|------|
| lightweight 嵌入模型 | `~/.hermes/skills/devops/skill-router-scalable/fine-tuned-model-v7` | ~80MB，MiniLM v7 微调 |
| lightweight 重排序模型 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 自动下载，~80MB |
| SkillRouter 嵌入模型 | `~/models/skillrouter/SkillRouter-Embedding-0.6B` | ~1.2GB |
| SkillRouter 重排序模型 | `~/models/skillrouter/SkillRouter-Reranker-0.6B` | ~1.2GB |
| SQLite 技能索引 | `~/.hermes/skill_index.db` | `skills` 表含 `name/category/description/body/mtime` |
| 反馈记录 | `~/.hermes/skill_router_feedback.jsonl` | 追加写，启动时加载 |
| 嵌入 mtime 记录 | `~/.hermes/skill_embedding_mtimes.json` | 用于增量嵌入更新 |
| pool 配置 | `~/.hermes/skills/.pool_config.json` | 可扩展 core_skills |
| 评估训练数据 | `~/.hermes/skills/devops/skill-router-scalable/training_data.json` | `evaluate.py` 依赖 |

### 2.3 运行环境要求

- HOME 目录必须可写，因为所有配置、数据库、缓存、反馈文件均放在 `~/.hermes` 下。
- 后端为 `skillrouter` 时，首次推理会加载约 2.4GB 模型到 CPU 内存，建议使用 8GB+ 内存。
- 所有推理均使用 CPU + `torch.float32`，未做量化或 GPU 支持。

---

## 3. `__init__.py` 核心模块分析

### 3.1 缓存系统

#### `CacheManager`

- 单值 TTL 缓存，线程安全（`threading.Lock`）。
- `get()`：值不存在或已过期返回 `None`。
- `set(value)`：写入值并刷新时间戳。
- `invalidate()`：清空。

全局实例：

- `_config_cache`：配置缓存，TTL 300s。
- `_skill_index_cache`：技能索引缓存，TTL 300s。
- `_embedding_cache`：lightweight 嵌入矩阵缓存，TTL 300s；与索引缓存联动失效。
- `_bm25_cache`：BM25 搜索器缓存，TTL 300s；与索引缓存联动失效。

#### `QueryCache`

- 基于 `OrderedDict` 的 LRU + TTL 缓存。
- 默认 `max_size=1000`，`ttl=300`。
- 命中时移动到队尾；超容量时淘汰队首最久未使用项。
- 全局实例 `_query_cache` 在 `register()` 中初始化，与配置绑定。

**注意**：`QueryCache` 缓存的是最终推荐结果（含反馈调整后的分数）。如果反馈写入后未过期 TTL，后续相同查询会返回旧分数。

### 3.2 反馈系统

#### `FeedbackStore`

- 单例，启动时从 `~/.hermes/skill_router_feedback.jsonl` 加载历史。
- 记录类型：`success`（+0.05）、`skip`（-0.02）。
- 写入采用追加 JSONL，带锁保护 `_records` 与文件写入。
- 读取时按技能聚合，24h 内权重 1.0，超过后按半衰期 12h 指数衰减。
- 衰减公式：`weight = 0.5 ^ ((age_hours - 24) / 12)`，当 `age_hours > 24`。

**注意**：反馈调整直接加到最终分数上，可能使 low 分数越过 medium/high 阈值，或使 BM25 降级路径中的分数失真。

### 3.3 BM25 关键词检索

#### `BM25Searcher`

- 初始化时预计算：文档分词、词项 DF、IDF、平均文档长度。
- 分词优先使用 `jieba`，缺失时降级为字符级分词。
- 文档文本 = `name + description + body_text`（完整 body）。
- IDF 公式：

```python
idf = max(0.0, ((n_docs - df + 0.5) / (df + 0.5) + 1.0))
```

**注意**：标准 BM25 IDF 通常取对数，此处未取对数，导致 IDF 数值范围与常规实现不同，但不影响内部排序（因为同一查询下对所有文档是单调变换）。

- `search()` 每次遍历所有文档，统计查询词项的 TF 并计算 BM25 得分。当技能池很大时，每次查询的时间复杂度为 `O(n_docs * avg_dl)`，未使用倒排索引。

### 3.4 混合检索与重排序

#### `HybridSearcher`

- 输入：`vector_results`、`bm25_results`、可选 `skill_names`。
- 对两组分数分别做 min-max 归一化到 `[0, 1]`。
- 加权融合：`score = vector_weight * v_score + bm25_weight * b_score`。
- 当 `skill_names` 提供时，所有未出现在向量/BM25 结果中的技能也会以 `(0, 0)` 参与融合，最终分数为 0。

**注意**：若向量与 BM25 结果集合不相交，`all_names` 会包含很多 0 分技能，但排序后自然被截断，不影响 top_k。

#### CrossEncoder 重排序

- `HybridSearcher._load_reranker()` 类级单例，线程安全。
- 默认模型 `cross-encoder/ms-marco-MiniLM-L-6-v2`，CPU 运行。
- 当 `backend == "skillrouter"` 时跳过加载，由 `SkillRouterBackend.rerank()` 处理。
- `_rerank_results()` 构造 `(query, "name: description")` 对，用重排序分数替换原分数。

### 3.5 端到端主流程 `search_skills`

#### 主要步骤

1. **配置加载**：`_load_config()`，TTL 缓存 300s。
2. **查询缓存**：按 `query::top_k={top_k}` 缓存命中即返回。
3. **后端选择**：若 `backend == skillrouter` 则尝试初始化 `_get_skillrouter_backend()`；失败则回退 lightweight。
4. **索引加载**：`_load_skill_index()` 从 SQLite 读取技能（含 mtime）。
5. **core_skills 获取**：合并配置与 `~/.hermes/skills/.pool_config.json`。
6. **域分类**：`_classify_domain(query)` 基于关键词匹配。
7. **候选集裁剪**：若命中域，按 `_DOMAIN_MAP` + 特殊规则（lark 前缀、finance 关键词、finance-dianjin 子域）得到 `domain_skill_names`；超过 `_DOMAIN_TOP_K * 4` 时截断；不足 `top_k` 时补充 `uncategorized`。
8. **向量检索**：
   - `skillrouter` 后端：使用 `_sr_embeddings_cache` 做增量编码，然后 `np.dot` 计算相似度。
   - `lightweight` 后端：使用 `_encode_with_timeout()` 编码查询，超时则降级；再与 `_get_skill_embeddings()` 预计算的全量技能嵌入做点积。
9. **BM25 检索**：`_get_bm25_searcher().search(query, ...)`。
10. **降级路径**：若向量不可用或超时，直接用 BM25 结果生成最终列表（含反馈调整、置信度、域聚合器兜底）。
11. **混合融合**：`HybridSearcher.search()`。
12. **重排序**：
    - `skillrouter` 后端调用 `sr_backend.rerank()`。
    - `lightweight` 后端调用 `HybridSearcher._rerank_results()`。
13. **后处理**：合并反馈调整、计算置信度、排序、截断 `top_k`、写入查询缓存。

#### 域聚合器注入

- 命中 `finance` / `finance-dianjin` 时，若 `finance-assistant` 存在且未在结果中，则以 `max_score + 0.1` 作为 core 技能注入。
- 命中 `wechat-article` 时，聚合器为 `wechat-article-main`。

### 3.6 Hermes 插件接口

#### `register(ctx)`

- 防御性检查 `ctx` 与 `register_hook`。
- 若 `enabled=false` 直接返回。
- 初始化 `_query_cache`。
- **不再在 register 阶段预热模型**，避免阻塞启动。
- 优先通过 `ctx.register_tool()` 注册 `skill_search`、`skill_feedback`；否则回退 `tools.registry.register()`。
- 注册 `pre_llm_call` 钩子。
- 订阅 `plugins.plugin_orchestrator.context` 的 `skill_updated` 事件，用于清除 `_sr_embeddings_cache`、全局嵌入缓存和查询缓存。

#### `_pre_llm_call_hook(**kwargs)`

- 跳过空消息或 `/` 开头的命令。
- 读取 `plugin_context.shared_get` 中的 `routing_strategy`、`model_selection`、`last_skill_updated`。
- 若 `model_selection` 包含 `pro`，则 `top_k = min(top_k * 2, 10)`。
- 调用 `search_skills(user_message, top_k)`。
- **Core 技能**：仅列出名称，常驻 system prompt。
- **Pool 技能**：
  - `low` 置信度：不注入 pool 技能（但 core 仍注入）。
  - `medium/high`：注入名称 + 描述前 50 字符。
- 返回 `context` 字符串和 `context_merge` 字典（`skill_domains`、`skill_count`）。

---

## 4. `skillrouter_backend.py` 分析

### 4.1 `SkillRouterBackend` 结构

- 接收 `emb_model_path`、`rank_model_path`，延迟加载。
- 内部状态：`_emb_model`、`_emb_tokenizer`、`_rank_model`、`_rank_tokenizer`、`_loaded`。
- 两把锁：`_emb_lock`、`_rank_lock`，分别保护嵌入模型和重排序模型加载。

### 4.2 嵌入推理

#### `encode_texts(texts)`

- batch_size=8，max_length=4096，padding + truncation。
- 输出 `last_hidden_state`，用 `_last_token_pool` 取最后一个有效 token。
- L2 归一化后转 `numpy`。

#### `encode_query(query)`

- 在查询前拼接固定 instruction prefix：

```text
Instruct: Given a task description, retrieve the most relevant skill document that would help an agent complete the task
Query: {query}
```

- 同样做 last_token_pool + L2 归一化。

### 4.3 重排序推理

#### `rerank(query, candidates, skills, top_k)`

- 对每个候选构造文档：`{name} | {description[:500]} | {body_text[:2000]}`。
- 使用 Qwen3 chat template，system prompt 要求逐步思考并只回答 yes/no，`enable_thinking=True`。
- 取 `outputs.logits[0, -1, :]` 中 `yes` 与 `no` token 的 logit 差作为分数。
- 按差分排序，返回 top_k。

**注意**：重排序对每个候选单独调用一次模型前向传播，当候选数较多时（如 `top_k * 2`），延迟会线性增长。

### 4.4 与 `__init__.py` 的协作

- `__init__.py` 维护 `_sr_embeddings_cache: Dict[str, Tuple[embedding, mtime]]`，实现增量更新。
- `SkillRouterBackend` 本身不感知 mtime，只提供 `encode_texts`、`encode_query`、`rerank` 三个原子能力。
- 当 `SkillRouterBackend` 初始化失败时，`search_skills` 回退到 lightweight；若 lightweight 也失败，则走纯 BM25。

---

## 5. `scripts/evaluate.py` 分析

### 5.1 作用

离线评估当前 lightweight 后端在标注数据集上的表现，输出：

- 纯向量检索 Top-1 / Top-3 准确率。
- 纯 BM25 检索 Top-1 / Top-3 准确率。
- 混合检索（0.7 向量 + 0.3 BM25）Top-1 / Top-3 准确率。
- 置信度分布（high / medium / low）。
- 错误案例与高频错误对。
- 混合检索相对纯向量的提升/退化数量。

### 5.2 输入

- `MODEL_PATH`：MiniLM v7 路径。
- `DB_PATH`：`~/.hermes/skill_index.db`。
- `TRAINING_DATA`：`training_data.json`，每条包含 `query`、`positive`（正例技能名），可选 `negatives`。

### 5.3 实现细节

- `BM25Evaluator` 复现了 `__init__.py` 中的 BM25 算法，但文档文本截断为 `body[:500]`，与在线版本（完整 body）不一致。
- 向量评估时，技能文本截断为 `body[:200]`，也与在线版本不同。
- 混合检索的融合逻辑与 `HybridSearcher.search()` 一致：min-max 归一化后加权。

### 5.4 可执行命令

```bash
export CUDA_VISIBLE_DEVICES=""
cd /home/xxh/.hermes/plugins/skill-router
python scripts/evaluate.py
```

**前提**：模型文件、SQLite 数据库、`training_data.json` 均已存在。

---

## 6. 数据流与降级策略

### 6.1 一次 `search_skills` 的完整数据流

```
用户查询
   │
   ▼
加载配置 ──► 查询缓存命中？───是───► 直接返回
   │           否
   ▼
选择后端（lightweight / skillrouter，失败则降级）
   │
   ▼
加载技能索引（SQLite）
   │
   ▼
域分类 ──► 得到 domain_skill_names（可选）
   │
   ├─► 向量检索（带域过滤、超时保护）
   │
   ├─► BM25 检索（带域过滤）
   │
   ▼
是否降级？───是───► BM25 结果 + 反馈调整 + 域聚合器 ──► 返回
   │           否
   ▼
混合融合（向量 + BM25）
   │
   ▼
重排序（CrossEncoder / SkillRouter reranker，失败则跳过）
   │
   ▼
反馈调整 + 置信度分级 + 排序截断
   │
   ▼
写入查询缓存 ──► 返回结果
```

### 6.2 降级路径

| 编号 | 触发条件 | 降级行为 | 日志级别 |
|------|----------|----------|----------|
| 1 | `backend=skillrouter` 但 `_get_skillrouter_backend()` 初始化失败 | 回退到 lightweight 后端 | `warning` |
| 2 | lightweight 嵌入模型不存在或加载失败，或 `_get_skill_embeddings()` 返回空 | 仅使用 BM25 检索 | `info` |
| 3 | `_encode_with_timeout()` 超过 `encode_timeout`（默认 10s） | 仅使用 BM25 检索 | `warning` |
| 4 | Reranker 模型加载失败或 `_rerank_results()` / `rerank()` 抛出异常 | 跳过重排序，返回融合结果 | `warning` / `error` |
| 5 | 混合融合或后处理整体异常 | 返回空列表 `[]` | `error` |

---

## 7. 域分组与检索空间裁剪

### 7.1 域分类

`_classify_domain(query)` 对 `_DOMAIN_KEYWORDS` 中的关键词在查询中做命中计数，返回得分最高域；无命中返回 `None`。

当前支持 13 个域：`code`、`ml`、`creative`、`productivity`、`research`、`finance`、`lark`、`wechat-article`、`finance-dianjin`、`insurance-agent`、`crypto`、`other`。

### 7.2 候选集裁剪逻辑

1. 根据 `_DOMAIN_MAP` 拿到域对应的 category 集合。
2. 从全部技能中筛选 `category` 属于该集合的技能。
3. 特殊规则：
   - `lark`：保留 `name.startswith("lark-")` 的技能。
   - `finance`：额外保留名称包含 `stock/fund/finance/portfolio/market/crypto/macro/analysis/technical/news` 的技能，并强制加入 `finance-assistant`。
   - `finance-dianjin`：通过子域关键词进一步缩小到指定前缀集合。
4. 若候选数超过 `_DOMAIN_TOP_K * 4 = 12`，按名称排序截断到 12。
5. 若候选数不足 `top_k`，补充 `category == "uncategorized"` 的技能。

### 7.3 域聚合器

| 域 | 聚合器 | 作用 |
|----|--------|------|
| `finance` | `finance-assistant` | 统一金融助手入口 |
| `finance-dianjin` | `finance-assistant` | 同上 |
| `wechat-article` | `wechat-article-main` | 公众号一条龙入口 |

聚合器以 `max_score + 0.1` 强制注入结果，确保用户进入域时优先看到入口技能。

---

## 8. 风险登记与改进建议

| 优先级 | 位置 | 风险描述 | 触发条件 | 影响面 | 建议措施 |
|--------|------|----------|----------|--------|----------|
| P0 | `BM25Searcher.search()` | 每次查询都遍历所有文档重新统计 TF，无倒排索引 | 技能池扩大 | 检索延迟随文档量线性增长 | 预计算倒排索引；或在初始化时同时保存词项→文档映射 |
| P0 | `HybridSearcher._normalize_scores()` / `BM25Searcher` | IDF 公式缺少对数，BM25 分数与常规实现不可比；归一化依赖每次调用的 min-max | 任何查询 | 分数不可解释，跨查询不稳定 | 采用标准 BM25 IDF；或对融合分数做全局校准 |
| P1 | `_encode_with_timeout()` | 超时后工作线程仍在后台运行，可能继续占用 CPU/内存 | 模型加载慢或推理慢 | 资源泄漏、后续查询竞争 | 使用 `torch.jit`、进程池或设置真正可中断的工作进程 |
| P1 | `_sr_embeddings_cache` | 读取与更新 `_sr_embeddings_cache` 的锁粒度较大，且部分读取在锁外 | 并发查询 | 可能读到半更新状态（虽然 Python dict 操作原子，但逻辑上仍应统一） | 将缓存封装为独立类，统一加锁策略 |
| P1 | `FeedbackStore` | 反馈调整直接叠加到最终分数，无上界 | 大量 success 反馈 | 分数被人为推高，置信度失真 | 对反馈调整做 clipping 或归一化；按时间窗口衰减 |
| P1 | 域分类 | 基于关键词硬编码，无法处理新域或语义相近但关键词缺失的查询 | 新增业务域 | 漏召回、候选集裁剪错误 | 引入轻量分类模型或允许配置化关键词 |
| P2 | `SkillRouterBackend.rerank()` | 逐候选前向传播，延迟线性增长 | 候选数 > 10 | 端到端延迟可能 > 5s | 支持批量 rerank 或限制候选数 |
| P2 | 全局状态 | `_MODEL_CACHE`、`_SKILLROUTER_BACKEND`、`_query_cache` 等均为模块级全局变量 | 多实例/测试并发 | 状态难以隔离，单测困难 | 引入显式的 PluginState 类，支持依赖注入 |
| P2 | 单测缺失 | 插件目录无测试文件 | 任何代码变更 | 回归风险高 | 为核心类（BM25Searcher、HybridSearcher、FeedbackStore）补充单元测试 |
| P2 | `evaluate.py` | 评估时技能文本截断为 `body[:500]` 和 `body[:200]`，与在线完整 body 不一致 | 离线评估 | 评估结果不能准确反映线上效果 | 统一评估与在线的文本构造逻辑；使用完整 body 或相同截断策略 |
| P3 | 查询缓存 | 缓存结果包含反馈调整分数，反馈更新后仍可能返回旧结果 | 反馈写入后 300s 内 | 路由行为滞后 | 反馈写入时主动失效相关查询缓存；或把反馈调整移出缓存 |
| P3 | `register()` 事件订阅 | `skill_updated` 事件订阅失败时静默 `pass` | orchestrator 未启用 | 无影响，但隐藏了集成问题 | 记录 debug 日志说明订阅失败原因 |

---

## 9. 验证与审计计划

### 9.1 本地评估

```bash
# 进入插件目录
export CUDA_VISIBLE_DEVICES=""
cd /home/xxh/.hermes/plugins/skill-router
python scripts/evaluate.py
```

### 9.2 运行时快速验证

在 Hermes 加载 skill-router 后，可通过以下方式验证：

1. **工具注册**：调用 `skill_search` 工具，检查返回 JSON 是否包含 `name/category/description/score/tier/confidence`。
2. **降级验证**：临时移走 lightweight 模型目录，观察日志是否出现“嵌入模型或向量不可用，降级为纯 BM25 检索”。
3. **缓存验证**：连续两次相同查询，第二次应出现“查询缓存命中”日志。
4. **反馈验证**：调用 `skill_feedback` 后，再次查询同一技能，观察分数变化。
5. **域分类验证**：分别用“飞书审批”、“信贷尽调”、“公众号排版”查询，观察日志中的域分类与候选数。

### 9.3 审计清单

审计时应确认：

- 模型文件是否存在且可读。
- `~/.hermes/skill_index.db` 是否存在，且 `skills` 表包含 `mtime` 列。
- `~/.hermes/config.yaml` 中 `plugins.skill-router` 配置是否符合预期。
- 日志中是否出现非预期降级或异常。
- 查询缓存命中率与推荐延迟是否可接受。

---

## 10. 总结

skill-router v3.1 是一个功能完整的统一技能路由插件，覆盖了双后端、混合检索、域分组、增量缓存、反馈学习等能力。其设计在资源受限环境下做了大量降级与缓存优化，但也引入了全局状态复杂、硬编码域映射、BM25 性能与可解释性不足等风险。

本次分析未修改任何源代码，仅输出结构、数据流、依赖、风险与验证计划，为后续重构或性能优化提供基础依据。
