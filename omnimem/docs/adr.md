# OmniMem 架构决策记录 (ADR)

> 本文档记录 OmniMem 项目中的关键架构决策及其背景、理由和影响。
> 遵循 [ADR (Architecture Decision Record)](https://adr.github.io/) 规范。

---

# ADR-001: 五层记忆架构 (L0-L4)

## Status
Accepted

## Context

AI Agent 在运行过程中会产生大量不同生命周期和抽象程度的信息：
- 实时感知信号（如对话流、环境变化）需要即时处理但无需持久化
- 当前会话上下文需要高频访问、快速响应
- 结构化知识（事实、偏好、技能）需要长期可靠存储和高效检索
- 从经验中提炼的深层洞察需要周期性升华和关联
- 通过反复训练内化的"本能"级记忆需要模型参数级支持

传统单层或双层记忆系统无法区分这些不同层次的需求，导致：
- 短期工作记忆与长期存储混用，互相干扰
- 无感知机制，完全依赖被动接收信息
- 缺乏从原始数据到高层知识的升华路径

## Decision

采用**五层认知记忆架构**，模拟人类认知心理学模型：

| 层级 | 名称 | 类比人类记忆 | 核心职责 |
|:---:|:---|:---|:---|
| L0 | 感知层 (Perception) | 感觉记忆 | 主动监控、信号检测、意图预测 |
| L1 | 工作记忆 (Working Memory) | 短期记忆 | CoreBlock 常驻上下文 + Attachment 压缩状态 |
| L2 | 结构化记忆 (Structured) | 长期记忆 | Wing/Room 导航 + Drawer/Closet 双存储 + 三级索引 |
| L3 | 深层记忆 (Deep) | 语义记忆 | Consolidation 升华 + KnowledgeGraph 知识图谱 |
| L4 | 内化记忆 (Internalized) | 程序性记忆 | KVCache 高频缓存 + LoRA 模型微调 |

各层之间通过明确的升级/降级接口通信：
- L0 -> L1：感知信号经筛选进入工作记忆
- L1 -> L2：重要信息经结构化后持久化
- L2 -> L3：周期性 Consolidation 提炼为心智模型
- L3 -> L4：高频模式经训练成为参数级记忆

## Consequences

### 正面影响
- **职责清晰**：每层有明确的数据特征和访问模式，便于独立测试和优化
- **可扩展性**：新增层级不影响已有层级，符合开闭原则
- **认知对齐**：架构与人类记忆心理学模型高度对应，直觉性好
- **渐进式复杂度**：简单场景可用 L0-L2，高级场景按需启用 L3/L4

### 负面影响
- **复杂度高**：五层架构增加了理解和维护成本，新开发者学习曲线陡峭
- **跨层协调开销**：层间数据流转需要额外的编排逻辑
- **配置维度多**：每层都有独立配置项，组合空间大

---

# ADR-002: SQLite 作为主存储

## Status
Accepted

## Context

OmniMem 需要一个主存储方案来持久化记忆元数据和内容，要求：
- **嵌入式**：无需独立部署数据库服务，降低运维成本
- **零配置**：开箱即用，无需安装额外服务
- **事务支持**：保证派生数据（索引、图谱节点）的一致性
- **高性能**：支持高并发读写场景
- **WAL 模式**：支持读写并发，避免锁竞争

备选方案对比：

| 方案 | 嵌入式 | 零配置 | 事务 | 并发写 | 分布式 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| SQLite | 是 | 是 | 是 | WAL 有限 | 否 |
| PostgreSQL | 否 | 否 | 是 | 是 | 是 |
| DuckDB | 是 | 是 | 是 | 列存优化 | 否 |
| 文件系统 (JSON) | 是 | 是 | 否 | 文件锁 | 否 |

## Decision

采用 **SQLite 作为主存储引擎**，配合以下优化策略：

1. **WAL (Write-Ahead Logging) 模式**：允许读写并发操作
2. **PRAGMA 优化**：
   - `journal_mode=WAL` — 写前日志
   - `synchronous=NORMAL` — 平衡安全性与性能
   - `cache_size=-64000` — 64MB 页缓存
   - `mmap_size=268435456` — 256MB 内存映射
   - `busy_timeout=5000` — 5 秒锁等待超时
3. **连接池管理**：通过 `DrawerClosetStore` 统一管理连接生命周期

## Consequences

### 正面影响
- **部署极简**：单个 `.db` 文件即可运行，无需任何外部依赖
- **可靠性高**：ACID 事务保证数据一致性，WAL 模式防止崩溃损坏
- **迁移方便**：直接复制数据库文件即可备份/迁移
- **生态成熟**：Python 标准库支持，调试工具丰富（DB Browser for SQLite）

### 负面影响
- **不支持分布式水平扩展**：单机限制，无法像 PostgreSQL 那样分片
- **并发写入瓶颈**：WAL 模式虽改善读并发，但写入仍需串行化
- **大数据量性能下降**：百万级以上记录时查询性能需要额外索引优化

---

# ADR-003: Facade 模式分组子系统

## Status
Accepted

## Context

OmniMem 项目包含 30+ 子系统模块，分布在以下目录中：
- `memory/` — 存储相关（wing_room, drawer_closet, index, meta_store, markdown_store）
- `retrieval/` — 检索相关（hybrid_engine, vector, bm25, rrf, reranker）
- `governance/` — 治理相关（conflict, decay, forgetting, privacy, provenance, sync, audit, rbac, feedback, encryption, vector_clock）
- `deep/` — 深层处理（consolidation, knowledge_graph, reflect）
- `internalize/` — 内化处理（kv_cache, lora_train）
- `compression/` — 压缩相关（collapse, line_compress, llm_summary, micro, priority）
- `perception/` — 感知引擎
- `context/` — 上下文管理
- `core/` — 核心组件（block, attachment, soul, budget, saga, background）

如果 Provider 直接依赖所有子系统，将导致：
- **耦合度过高**：Provider 成为上帝对象，修改任何子系统都可能影响 Provider
- **调用链混乱**：外部调用者不知道该找哪个子系统
- **测试困难**：无法单独 Mock 某个子系统组

## Decision

采用 **Facade（外观）模式**，将 30+ 子系统按关注点分组为 5 个 Facade：

```
OmniMemProvider
    ├── StorageFacade      ← memory/ + compression/
    │     ├── DrawerClosetStore   (双存储)
    │     ├── WingRoomManager     (宫殿导航)
    │     ├── ThreeLevelIndex     (三级索引)
    │     └── CompressionPipeline (压缩管线)
    ├── RetrievalFacade     ← retrieval/ + context/
    │     ├── HybridRetriever     (混合检索)
    │     ├── VectorRetriever     (向量检索)
    │     ├── BM25Retriever       (关键词检索)
    │     ├── RRFFuser            (融合排序)
    │     └── ContextManager      (上下文管理)
    ├── GovernanceFacade    ← governance/
    │     ├── ConflictResolver    (冲突仲裁)
    │     ├── ForgettingCurve     (遗忘曲线)
    │     ├── TemporalDecay       (时间衰减)
    │     ├── PrivacyManager      (隐私分级)
    │     ├── ProvenanceTracker   (溯源追踪)
    │     ├── SyncEngine          (多实例同步)
    │     ├── RBACController      (权限控制)
    │     └── AuditLogger         (审计日志)
    ├── DeepMemoryFacade    ← deep/ + perception/
    │     ├── ReflectEngine       (反思引擎)
    │     ├── ConsolidationEngine (知识升华)
    │     ├── KnowledgeGraph      (知识图谱)
    │     └── PerceptionEngine    (感知引擎)
    └── SyncFacade          ← sync_facade.py
          └── 同步协调入口
```

每个 Facade 对外暴露精简的公共 API，内部封装子系统的协作细节。

## Consequences

### 正面影响
- **高内聚低耦合**：相关子系统聚集在同一 Facade 下，Facade 间边界清晰
- **简化调用链**：外部只需与 5 个 Facade 交互，无需了解内部 30+ 模块
- **易于替换**：替换某个子系统的实现只需修改对应 Facade 内部
- **独立测试**：每个 Facade 可独立进行单元测试和集成测试
- **并行开发**：不同团队可负责不同 Facade，减少冲突

### 负面影响
- **增加间接层**：每次调用多经过一层转发，可能有微小性能开销
- **Facade 自身可能膨胀**：如果分组不合理，某个 Facade 可能再次变成上帝对象
- **跨 Facade 协作**：某些操作（如 memorize 后触发反思）需要多个 Facade 配合

---

# ADR-004: ChromaDB 作为默认向量库

## Status
Accepted

## Context

OmniMem 的混合检索引擎需要一个嵌入式向量数据库来存储和检索记忆的语义嵌入向量。需求包括：
- **嵌入式部署**：无需独立服务，随应用启动
- **持久化**：重启后数据不丢失
- **过滤支持**：支持 metadata 过滤（按隐私级别、记忆类型等）
- **易于集成**：Python 原生支持，API 简洁

备选方案对比：

| 方案 | 嵌入式 | 持久化 | Metadata 过滤 | Python API | 大数据量性能 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ChromaDB | 是 | 是 | 是 | 优秀 | 中等 |
| Qdrant | 否（需服务） | 是 | 是 | 好 | 优秀 |
| FAISS | 是 | 需自行实现 | 有限 | 一般 | 优秀 |
| Milvus | 否（需服务） | 是 | 是 | 好 | 优秀 |
| pgvector | 否（需 PG） | 是 | 是 | 好 | 良好 |

## Decision

采用 **ChromaDB 作为默认向量后端**，同时设计 `VectorStore` 抽象接口以支持切换：

```python
# 向量库抽象接口
class VectorStore(ABC):
    @abstractmethod
    async def upsert(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict]): ...
    @abstractmethod
    async def query(self, query_embedding: list[float], n_results: int, where: dict | None = None) -> QueryResult: ...
    @abstractmethod
    async def delete(self, ids: list[str]): ...

# ChromaDB 实现（默认）
class ChromaVectorStore(VectorStore): ...

# 可扩展的其他实现
class QdrantVectorStore(VectorStore): ...   # 高性能场景
class FAISSVectorStore(VectorStore): ...    # 纯内存场景
```

通过 `vector_factory.py` 工厂类根据配置动态实例化。

## Consequences

### 正面影响
- **零配置启动**：ChromaDB 内嵌运行，无需安装额外服务
- **开发体验好**：API 设计简洁直观，文档完善
- **持久化内置**：默认使用本地持久化目录，重启不丢数据
- **Metadata 过滤原生支持**：完美匹配隐私分级、记忆类型等过滤需求
- **可替换性强**：抽象接口使得未来切换到 Qdrant/pgvector 成本极低

### 负面影响
- **大数据量性能有限**：十万级以上记录时延迟增加明显
- **内存占用较高**：全量加载索引到内存，大规模场景需要注意
- **并发能力有限**：不支持真正的分布式并发写入
- **版本兼容性**：ChromaDB 版本迭代较快，API 可能有 breaking change

---

# ADR-005: 混合检索策略 (Vec + BM25 + RRF)

## Status
Accepted

## Context

单一检索策略存在固有缺陷：

**纯向量检索的问题**：
- 语义匹配能力强，但对专有名词、ID、精确关键词召回差
- 例如："用户 ID 是 abc123" 这类记忆，语义相似度低但关键词命中率高

**纯 BM25 关键词检索的问题**：
- 关键词匹配精确，但无法理解同义词和语义相近的表达
- 例如：搜索"编程偏好"无法匹配到"coding preference"

实际场景中，AI Agent 的查询既包含语义模糊的开放式问题，也包含需要精确匹配的事实型问题。单一策略无法兼顾。

## Decision

采用 **混合检索 + RRF (Reciprocal Rank Fusion) 融合排序**策略：

```
用户查询
    │
    ├─────────────────┬──────────────────┐
    ▼                 ▼                  ▼
 向量检索           BM25 检索        （可选：图谱检索）
(语义相似度)      (关键词匹配)     (实体关系推理)
    │                 │                  │
    ▼                 ▼                  ▼
 Top-K 结果        Top-K 结果       Top-K 结果
    │                 │                  │
    └─────────────────┴──────────────────┘
                    │
                    ▼
              RRF 融合排序
              score(d) = Σ 1/(k + rank_i(d))
                    │
                    ▼
             （可选）Cross-Encoder 重排序
                    │
                    ▼
              最终结果列表
```

**RRF 公式**：`score(d) = Σ 1/(k + rank_i(d))`，其中 k 通常取 60

**关键参数**：
- `vector_weight` / `bm25_weight`：两种来源的基础权重
- `rrf_k`：RRF 平滑参数，默认 60
- `top_k`：每种来源返回的候选数量
- `enable_reranker`：是否启用 Cross-Encoder 二次重排序

## Consequences

### 正面影响
- **召回率显著提升**：互补覆盖语义匹配和关键词匹配的优势场景
- **鲁棒性强**：即使某一检索源失效（如 BM25 索引未建好），另一源仍能提供结果
- **可调优**：通过权重参数可根据实际数据分布调优融合效果
- **可扩展**：新增检索源（如知识图谱检索）只需加入 RRF 融合

### 负面影响
- **计算开销增加约 2 倍**：需要同时执行向量和 BM25 两次检索
- **延迟叠加**：总延迟 ≈ max(向量检索耗时, BM25 检索耗时) + RRF 融合耗时
- **参数调优复杂**：权重、top_k、rrf_k 等参数的组合空间较大
- **结果可解释性降低**：融合后的排序难以解释为何某条排在某位

---

# ADR-006: 艾宾浩斯遗忘曲线

## Status
Accepted

## Context

随着 AI Agent 运行时间增长，记忆数量将持续累积，导致：
- **存储膨胀**：无限制增长占用磁盘空间
- **检索效率下降**：候选集增大导致检索变慢
- **噪声比例升高**：过时/低质量记忆干扰准确检索
- **Token 浪费**：无关历史记忆被注入上下文窗口

传统的 TTL（Time-To-Live）过期策略过于粗暴——不管记忆是否被频繁使用，到期即删除。而人类的记忆遗忘遵循艾宾浩斯曲线：**复习次数越多，遗忘越慢**。

## Decision

采用 **四阶段遗忘曲线 + recall_count 加速** 的记忆生命周期管理：

```
活跃期 (active)
  │  新创建的记忆，正常参与检索
  │  默认保留周期：7 天
  ▼
巩固期 (consolidating)
  │  访问频率降低但仍有一定价值
  │  进入此阶段条件：超过 active_ttl 未被 recall
  │  此阶段检索权重降为 0.7x
  ▼
归档期 (archived)
  │  长期未访问，降级存储
  │  进入此阶段条件：超过 consolidating_ttl 且 recall_count < 阈值
  │  不参与常规检索，仅在全量扫描时可见
  ▼
遗忘期 (forgotten)
  │  标记删除或物理删除
  │  进入此阶段条件：超过 archived_ttl
  │  可配置 soft_delete / hard_delete
  ▼
  [回收]
```

**recall_count 加速机制**：
- 每次 recall 操作递增计数器
- 高 recall_count 的记忆自动延长当前阶段的停留时间
- 类似"间隔重复"效应：越常回忆的记忆越不容易被遗忘

**配置项**：
```yaml
forgetting:
  active_ttl_days: 7          # 活跃期天数
  consolidating_ttl_days: 30  # 巩固期天数
  archived_ttl_days: 90       # 归档期天数
  recall_boost_factor: 1.5    # 回忆加速因子
  soft_delete: true           # 软删除模式
  auto_forget: true           # 自动执行遗忘扫描
  forget_scan_interval: 3600  # 扫描间隔（秒）
```

## Consequences

### 正面影响
- **自动化生命周期管理**：无需手动清理，系统自动维护健康的数据量
- **智能保留有价值记忆**：recall_count 机制确保常用记忆不被误删
- **存储可控**：长期运行的 Agent 不会因记忆无限增长而导致 OOM 或磁盘满
- **检索质量提升**：过时低质记忆逐步淡出，提高信噪比

### 负面影响
- **可能遗忘有用信息**：偶尔使用的长尾记忆可能在未被充分 recall 前就被归档
- **需要定期 review**：建议定期检查 archived 阶段的记忆，确认无误删
- **阶段转换时机难以精准设定**：TTL 参数需要根据实际使用模式调优
- **增加后台开销**：遗忘扫描是周期性任务，消耗一定 CPU/IO 资源

---

# ADR-007: 多入口模式 (Hermes + SDK + REST + MCP)

## Status
Accepted

## Context

OmniMem 的潜在使用者涵盖多种角色和场景：

1. **Hermes AI 框架用户**：作为 Hermes 插件 (`plugin.yaml`) 使用，通过 `provider.py` / `tool_router.py` 接入
2. **独立 Python 开发者**：希望在自有项目中以 SDK 方式集成记忆功能
3. **非 Python 服务**：需要通过 HTTP API 与 OmniMem 交互的后端服务
4. **MCP (Model Context Protocol) 生态**：希望以 MCP Server 形式接入 Claude Desktop / Cursor 等 MCP 客户端
5. **LangChain 用户**：希望使用 LangChain 的 Memory 抽象层

如果只支持一种入口方式，将大幅限制适用范围。

## Decision

采用 **多入口并存** 架构，提供四种主要入口 + 一种桥接适配器：

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Hermes    │  │    SDK      │  │   REST API  │  │  MCP Server │
│   Plugin    │  │ (Python)    │  │  (HTTP)     │  │  (stdio)    │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │                │
       ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OmniMem Core Engine                         │
│  (Provider → Facades → Subsystems → Storage)                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                     ┌──────────────────┐
                     │ LangChain Bridge │
                     │ (langchain_memory.py)│
                     └──────────────────┘
```

**各入口详情**：

| 入口 | 入口文件 | 使用方式 | 适用场景 |
|:---|:---|:---|:---|
| Hermes 插件 | `plugin.yaml` + `provider.py` | 放入 `plugins/memory/omnimem/` | Hermes 框架用户 |
| OmniMemSDK | `sdk.py` | `pip install omnimem[sdk]` | Python 项目集成 |
| REST API | `rest_api.py` | `omnimem-api --port 8765` | 跨语言/微服务 |
| MCP Server | `mcp_server.py` | `omnimem-mcp` | Claude Desktop / Cursor |
| LangChain | `langchain_memory.py` | `OmniMemChatMemory()` | LangChain 应用 |

## Consequences

### 正面影响
- **灵活适配多种场景**：从插件到独立服务，覆盖几乎所有使用模式
- **生态兼容性好**：MCP 和 LangChain 桥接扩大了潜在用户群
- **核心代码复用**：所有入口共享同一套 Core Engine，避免重复实现
- **独立演进**：每个入口可以独立迭代而不影响其他入口

### 负面影响
- **维护成本随入口数线性增长**：每个入口都需要独立的文档、测试、版本兼容
- **API 一致性挑战**：需要确保不同入口的行为语义一致
- **发布流程复杂**：pip 包需要包含所有入口的依赖声明
- **测试矩阵庞大**：每个入口 x 每个核心功能的组合测试

---

# ADR-008: L4 内化层插件化

## Status
Accepted

## Context

L4 内化记忆层包含两个重量级组件：

1. **KVCache Manager**：将高频访问的记忆预填充到模型的 KV Cache 中，实现毫秒级响应
2. **LoRA Trainer**：基于用户交互模式训练 LoRA 适配器，将行为模式"烧录"进模型参数

这两个组件的共同问题是**重度依赖**：
- `peft` (Parameter-Efficient Fine-Tuning)：LoRA 训练框架
- `transformers` + `torch`：模型推理和训练（通常 > 2GB 安装体积）
- `accelerate`：分布式训练支持

对于只需要 L0-L3 功能的用户来说，这些依赖是不必要的负担：
- 安装体积从 ~50MB 暴增到 > 2GB
- 引入 CUDA/cuDNN 等系统级依赖
- 在 CPU-only 环境下无法使用 LoRA 功能

## Decision

采用 **插件化架构**，将 L4 组件设计为可选插件：

```python
# 插件注册表
class PluginRegistry:
    _plugins: dict[str, type[InternalizationPlugin]] = {}

    @classmethod
    def register(cls, name: str, plugin_class: type[InternalizationPlugin]):
        cls._plugins[name] = plugin_class

    @classmethod
    def get(cls, name: str) -> InternalizationPlugin | None:
        plugin_class = cls._plugins.get(name)
        if plugin_class is None:
            return None
        return plugin_class()

# 插件接口
class InternalizationPlugin(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def initialize(self, config: dict) -> None: ...

    @abstractmethod
    def internalize(self, memories: list[Memory]) -> InternalizationResult: ...

    @property
    @abstractmethod
    def dependencies(self) -> list[str]:
        """返回所需的 pip 包名列表"""

    @property
    @abstractmethod
    def available(self) -> bool:
        """检查依赖是否已安装"""
```

**具体实现**：

| 插件 | 文件 | 依赖 | 触发条件 |
|:---|:---|:---|:---|:---|
| KVCachePlugin | `internalize/kv_cache.py` | torch | `kv_cache_threshold` > 0 |
| LoRAPlugin | `internalize/lora_train.py` | peft, transformers, torch | 显式调用 `lora_train` 工具 |

**安装选项**：
```bash
pip install omnimem          # 核心（L0-L3），不含 L4
pip install omnimem[lora]    # 含 LoRA 支持
pip install omnimem[all]     # 含全部可选依赖
```

## Consequences

### 正面影响
- **核心包保持轻量**：默认安装不含 PyTorch 等重型依赖
- **按需启用**：用户只在需要时才安装 L4 依赖
- **可扩展**：第三方可实现自定义 InternalizationPlugin
- **优雅降级**：依赖缺失时 L4 功能静默禁用，不影响 L0-L3

### 负面影响
- **插件发现机制复杂**：需要在运行时检测依赖是否可用
- **错误定位困难**：L4 功能不可用时用户可能不清楚原因
- **接口稳定性压力**：InternalizationPlugin 接口一旦发布需要保持向后兼容
- **测试分叉**：有/无 L4 依赖时需要不同的测试矩阵

---

## 决策索引

| 编号 | 标题 | 状态 | 影响范围 |
|:---:|:---|:---|:---:|
| ADR-001 | 五层记忆架构 (L0-L4) | Accepted | 全局架构 |
| ADR-002 | SQLite 作为主存储 | Accepted | 存储层 |
| ADR-003 | Facade 模式分组子系统 | Accepted | 整体组织 |
| ADR-004 | ChromaDB 作为默认向量库 | Accepted | 检索层 |
| ADR-005 | 混合检索策略 (Vec+BM25+RRF) | Accepted | 检索层 |
| ADR-006 | 艾宾浩斯遗忘曲线 | Accepted | 治理层 |
| ADR-007 | 多入口模式 (Hermes+SDK+REST+MCP) | Accepted | 接口层 |
| ADR-008 | L4 内化层插件化 | Accepted | L4 层 |
