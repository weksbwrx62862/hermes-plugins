# OmniMem 架构设计文档

> 本文档描述 OmniMem 五层认知记忆引擎的完整架构设计，包括系统总览、数据流、核心组件和扩展机制。

---

## 1. 项目概述

**OmniMem** 是一个为 AI Agent 设计的五层认知记忆引擎，模拟人类从感知到内化的完整记忆生命周期。它提供结构化存储、混合检索、智能治理、深层反思和安全防护等核心能力，支持 Hermes 插件、Python SDK、REST API 和 MCP Server 等多种接入方式。

### 设计定位

- **不是简单的向量数据库封装**：OmniMem 是完整的记忆管理系统，涵盖感知、存储、检索、治理、反思、内化全链路
- **不是纯 RAG 方案**：OmniMem 的记忆层级远超传统 RAG 的 document-store 模型
- **目标是成为 AI Agent 的"海马体"**：为 Agent 提供类似人类大脑的记忆管理基础设施

---

## 2. 五层记忆架构

### 2.1 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        L4 内化记忆 (Internalized)                    │
│   KVCacheManager (高频缓存) │ LoRATrainer (模型微调分身) [可选]      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ 训练获得 / 参数级
┌───────────────────────────────▼─────────────────────────────────────┐
│                        L3 深层记忆 (Deep)                           │
│   ReflectEngine (四步反思) │ ConsolidationEngine (知识升华)          │
│   KnowledgeGraph (时序三元组图谱)                                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ 周期性升华 / 图结构
┌───────────────────────────────▼─────────────────────────────────────┐
│                      L2 结构化记忆 (Structured)                     │
│   WingRoomManager (宫殿导航) │ DrawerClosetStore (双存储)            │
│   ThreeLevelIndex (三级索引) │ MetaStore (元数据)                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ 持久化 / 三级索引
┌───────────────────────────────▼─────────────────────────────────────┐
│                       L1 工作记忆 (Working Memory)                  │
│   CoreBlock (常驻上下文块) │ CompactAttachment (压缩状态附件)        │
│   BudgetManager (Token 预算控制)                                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ 高频访问 / 内存驻留
┌───────────────────────────────▼─────────────────────────────────────┐
│                        L0 感知层 (Perception)                       │
│   PerceptionEngine (主动监控 / 信号检测 / 意图预测)                  │
└─────────────────────────────────────────────────────────────────────┘
                                实时流 / 无需持久化
```

### 2.2 各层详细说明

#### L0 感知层 (Perception Layer)

| 属性 | 说明 |
|:---|:---|
| **类比人类记忆** | 感觉记忆（Sensory Memory） |
| **核心组件** | `PerceptionEngine` |
| **数据特征** | 实时流数据，无需持久化 |
| **生命周期** | 毫秒级，瞬时 |
| **主要职责** | 主动监控对话流、检测重要信号、预测用户意图 |

**工作流程**：
1. 监听 Agent 对话流和环境事件
2. 通过信号检测器识别值得记录的信息片段
3. 意图预测判断是否需要触发 memorize
4. 过滤系统注入内容，防止自我污染

#### L1 工作记忆 (Working Memory)

| 属性 | 说明 |
|:---|:---|
| **类比人类记忆** | 短期记忆（Short-term Memory） |
| **核心组件** | `CoreBlock` + `CompactAttachment` + `BudgetManager` |
| **数据特征** | 高频访问，内存驻留 |
| **容量限制** | 可配置 Token 预算（默认 4000） |
| **主要职责** | 维护当前会话上下文、管理 Token 预算 |

**关键概念**：

- **CoreBlock**：常驻上下文块，包含身份信息、当前任务摘要等核心信息
- **CompactAttachment**：压缩后的历史状态附件，通过 `omni_detail` 按需展开
- **BudgetManager**：自适应 Token 预算控制，根据查询复杂度动态调整

#### L2 结构化记忆 (Structured Memory)

| 属性 | 说明 |
|:---|:---|
| **类比人类记忆** | 长期记忆（Long-term Memory）的海马体索引区 |
| **核心组件** | `WingRoomManager` + `DrawerClosetStore` + `ThreeLevelIndex` + `MetaStore` |
| **数据特征** | 持久化到 SQLite + ChromaDB，三级索引加速 |
| **生命周期** | 受遗忘曲线管理 |
| **主要职责** | 可靠持久化、高效检索、空间组织 |

**宫殿记忆法组织结构**：

```
OmniMem Palace (宫殿)
├── Wing: personal (个人)
│   ├── Room: preferences (偏好)
│   │   ├── Drawer: ui_preferences (UI 偏好)
│   │   └── Closet: tech_preferences (技术偏好)
│   └── Room: skills (技能)
├── Wing: project (项目)
│   ├── Room: architecture (架构)
│   └── Room: decisions (决策)
└── Wing: shared (共享)
    └── Room: team_knowledge (团队知识)
```

**双存储模型**：
- **Drawer**：高频访问记忆，SQLite 行存储，支持精确查询
- **Closet**：低频大体积记忆，Markdown 文件存储，适合长文本

**三级索引**：
1. **向量索引**（ChromaDB）：语义相似度检索
2. **关键词索引**（BM25）：精确关键词匹配
3. **元数据索引**（SQLite）：按类型/隐私/时间/来源过滤

#### L3 深层记忆 (Deep Memory)

| 属性 | 说明 |
|:---|:---|
| **类比人类记忆** | 语义记忆（Semantic Memory）+ 情景记忆（Episodic Memory） |
| **核心组件** | `ReflectEngine` + `ConsolidationEngine` + `KnowledgeGraph` |
| **数据特征** | 从原始事实提炼的抽象知识，图结构 |
| **触发条件** | 周期性或手动触发（fact_threshold 达标后自动） |
| **主要职责** | 知识升华、模式发现、心智模型构建 |

**Consolidation 四阶段升华**：

```
Fact (原始事实)
    │ "用户使用 FastAPI 而非 Flask"
    ▼
Observation (观察)
    │ "用户在多个项目中都选择了 FastAPI"
    ▼
Pattern (模式)
    │ "用户倾向于选择异步优先的 Python Web 框架"
    ▼
Mental Model (心智模型)
    │ "用户是异步编程倡导者，偏好现代 Python 技术栈"
```

**ReflectEngine 四步反思循环**：

```
1. Retrieve (检索) → 从 L2 获取相关记忆
2. Analyze (分析) → 结合 Disposition 性格参数深度推理
3. Synthesize (综合) → 生成新的洞察和关联
4. Store (存储) → 将反思结果写回 L2/L3
```

**KnowledgeGraph 时序三元组**：
- 传统三元组：(主体, 谓词, 客体)
- OmniMem 扩展：(主体, 谓词, 客体, 时间戳, 置信度, 来源)
- 支持时间衰减的边权重和置信度传播

#### L4 内化记忆 (Internalized Memory)

| 属性 | 说明 |
|:---|:---|
| **类比人类记忆** | 程序性记忆（Procedural Memory）— "肌肉记忆" |
| **核心组件** | `KVCacheManager` + `LoRATrainer` [插件] |
| **数据特征** | 模型参数级，训练获得 |
| **依赖要求** | PyTorch + PEFT + Transformers（可选安装） |
| **主要职责** | 毫秒级高频响应、行为模式内化 |

**KVCache 预填充**：
- 监控记忆访问频率
- 当某记忆 recall_count 超过阈值时，预填充到 KV Cache
- 效果：后续对该记忆的引用实现零延迟

**LoRA 分身系统 (Shade)**：
- 基于用户交互模式训练低秩适配器
- 每个 Shade 代表用户的一个"侧面"（如工作模式 vs 休闲模式）
- 推理时动态加载对应 Shade 的 LoRA 权重

---

## 3. 系统总览图

```
                         ┌─────────────────────────────────────────┐
                         │           入口层 (Entry Points)          │
                         │                                         │
                         │  HermesPlugin │ SDK │ RESTAPI │ MCPServer│
                         └────────────────────┬────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          OmniMemProvider                                 │
│                     (主入口 / 请求路由 / 生命周期管理)                      │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ ToolRouter│→│memorize  │→│  recall  │→│ reflect  │→│ govern   │  │
│  │(工具路由) │  │ handler  │  │ handler  │  │ handler  │  │ handler  │  │
│  └──────────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
└─────────────────────────┼──────────┼──────────┼──────────┼─────────────┘
                          │          │          │          │
       ┌──────────────────┼──────────┼──────────┼──────────┤
       │                  │          │          │          │
       ▼                  ▼          ▼          ▼          ▼
┌──────────────┐  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│StorageFacade │  │RetrievalFacade│ │DeepMemoryFacade│ │GovernanceFacade│
│              │  │              │ │              │ │              │
│ memory/      │  │ retrieval/   │ │ deep/        │ │ governance/  │
│ compression/ │  │ context/     │ │ perception/  │ │              │
└──────┬───────┘  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                 │                │                │
       ▼                 ▼                ▼                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           存储层 (Storage Layer)                      │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │   SQLite    │  │  ChromaDB   │  │ Markdown Files│  │Knowledge  │  │
│  │ (主存储)    │  │(向量索引)   │  │ (Closet 存储) │  │ Graph     │  │
│  │ Drawer/Closet│  │ VectorStore │  │              │  │ (L3 图谱) │  │
│  │ Meta/Index  │  │             │  │              │  │           │  │
│  └─────────────┘  └─────────────┘  └──────────────┘  └───────────┘  │
└──────────────────────────────────────────────────────────────────────┘

       ┌──────────────────────────────────────────────────────────┐
       │                 横切面 (Cross-Cutting Concerns)           │
       │                                                          │
       │  SecurityValidator │ SagaCoordinator │ BackgroundExecutor│
       │  SoulSystem        │ AsyncLLMClient  │ FeedbackCollector │
       └──────────────────────────────────────────────────────────┘
```

---

## 4. 核心数据流

### 4.1 写入流 (Memorize Flow)

```
Agent 调用 omni_memorize({"content": "..."})
         │
         ▼
┌─────────────────┐
│  memorize handler │  ← handlers/memorize.py
│  1. 参数校验      │
│  2. 安全扫描      │  ← SecurityValidator (14 种检测)
│  3. 反递归检查    │  ← 拒绝系统注入内容
│  4. 内容预处理    │  ← 清洗/归一化
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PerceptionEngine │  ← perception/engine.py (L0)
│  5. 信号增强      │  ← 补充隐含信息（时间戳、来源等）
│  6. 意图分类      │  ← 自动推断 memory_type
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  StorageFacade   │  ← facades/storage.py
│  7. 空间分配      │  ← WingRoomManager 分配位置
│  8. 双存储写入    │  ← DrawerClosetStore 写入 SQLite
│  9. 三级索引更新  │  ← ThreeLevelIndex 同步更新
│                    │     - 向量嵌入 → ChromaDB
│                    │     - BM25 词表 → SQLite FTS
│                    │     - 元数据 → SQLite
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SagaCoordinator  │  ← core/saga.py
│ 10. 异步派生写入  │  ← 确保索引/图谱最终一致性
│ 11. 补偿事务注册  │  ← 失败时可回滚
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GovernanceFacade │  ← facades/governance.py
│ 12. 隐私标记      │  ← PrivacyManager 设置默认级别
│ 13. 溯源记录      │  ← ProvenanceTracker 记录来源
│ 14. 审计日志      │  ← AuditLogger 记录操作
└────────┬────────┘
         │
         ▼
    返回 memory_id + 确认
```

### 4.2 检索流 (Recall Flow)

```
Agent 调用 omni_recall({"query": "...", "mode": "rag"})
         │
         ▼
┌─────────────────┐
│   recall handler │  ← handlers/recall.py
│  1. 查询扩展      │  ← 同义词扩展（config/synonyms.json）
│  2. 模式选择      │  ← rag / llm
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RetrievalFacade  │  ← facades/retrieval.py
│                  │
│  ┌──────────────┴──────────────┐
│  │     HybridRetriever         │  ← retrieval/engine.py
│  │                             │
│  │  ┌──────────┐ ┌──────────┐  │
│  │  │VectorRet. │ │BM25Ret.  │  │
│  │  │(语义相似) │ │(关键词)  │  │
│  │  └─────┬────┘ └─────┬────┘  │
│  │        │            │       │
│  │        └─────┬──────┘       │
│  │              ▼              │
│  │        ┌──────────┐        │
│  │        │ RRFFuser  │        │  ← retrieval/rrf.py
│  │        │融合排序   │        │     score = Σ 1/(k+rank)
│  │        └─────┬────┘        │
│  │              │             │
│  │       (可选) ▼             │
│  │        ┌──────────┐        │
│  │        │Reranker  │        │  ← retrieval/reranker.py
│  │        │Cross-Enc │        │     二次精排
│  │        └──────────┘        │
│  └──────────────┬──────────────┘
│                  │
└──────────────────┤
                   ▼
┌─────────────────────────────────┐
│      ContextManager             │  ← context/manager.py
│  1. 精炼压缩     │  ← 压缩为 <=60 字摘要
│  2. 语义去重      │  ← DedupEngine 去除重复
│  3. 预算控制      │  ← BudgetManager 截断超预算结果
│  4. 懒加载标记    │  ← 完整内容需 omni_detail 拉取
└────────┬──────────┘
         │
         ▼
┌─────────────────┐
│ GovernanceFacade │
│ 5. 隐私过滤      │  ← 按当前权限过滤 secret 级别
│ 6. 时间衰减加权  │  ← TemporalDecay 调整排序
│ 7. 遗忘状态排除  │  ← 排除 archived/forgotten 状态
└────────┬────────┘
         │
         ▼
    返回精炼后的上下文列表
```

### 4.3 治理流 (Govern Flow)

```
Agent 调用 omni_govern({"action": "scan_conflicts"})
         │
         ▼
┌─────────────────┐
│   govern handler │  ← handlers/govern.py
│  1. 动作解析      │  ← scan_conflicts / set_privacy / archive ...
│  2. 权限校验      │  ← RBACController 检查操作权限
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GovernanceFacade │  ← facades/governance.py
│                  │
│  根据 action 路由到子处理器：               │
│                                          │
│  ┌──────────────────────────────────┐    │
│  │ action: scan_conflicts           │    │
│  │   → ConflictResolver             │    │
│  │     1. 语义聚类候选对            │    │
│  │     2. 矛盾检测（LLM 辅助）      │    │
│  │     3. 按策略解决（latest/conf） │    │
│  │     4. 生成冲突报告              │    │
│  ├──────────────────────────────────┤    │
│  │ action: set_privacy             │    │
│  │   → PrivacyManager              │    │
│  │     1. 验证目标级别有效性        │    │
│  │     2. 更新隐私标记              │    │
│  │     3. 加密 secret 数据          │    │  ← cryptography/Fernet
│  ├──────────────────────────────────┤    │
│  │ action: run_forgetting          │    │
│  │   → ForgettingCurve             │    │
│  │     1. 扫描所有记忆              │    │
│  │     2. 应用四阶段转换规则        │    │
│  │     3. recall_count 加速调整     │    │
│  │     4. 执行软删除/硬删除         │    │
│  ├──────────────────────────────────┤    │
│  │ action: audit_log               │    │
│  │   → AuditLogger                 │    │
│  │     1. 记录操作详情              │    │
│  │     2. 关联操作者和时间戳        │    │
│  └──────────────────────────────────┘    │
│                                          │
└────────┬─────────────────────────────────┘
         │
         ▼
    返回治理操作结果
```

### 4.4 反思流 (Reflect Flow)

```
Agent 调用 omni_reflect({"query": "用户技术栈", "disposition": {...}})
         │
         ▼
┌─────────────────┐
│  DeepMemoryFacade│  ← facades/deep_memory.py
│                  │
│  ┌──────────────────────────────┐    │
│  │ Step 1: ReflectEngine        │    │  ← deep/reflect.py
│  │   - 从 L2 检索相关记忆        │    │
│  │   - 加载 Disposition 性格参数  │    │
│  │   - LLM 深度分析              │    │
│  │     * skepticism: 怀疑程度     │    │
│  │     * literalness: 字面程度    │    │
│  │     * empathy: 共情程度       │    │
│  └──────────────┬───────────────┘    │
│                 │                    │
│  ┌──────────────▼───────────────┐    │
│  │ Step 2: ConsolidationEngine  │    │  ← deep/consolidation.py
│  │   - Fact → Observation       │    │
│  │   - Observation → Pattern     │    │
│  │   - Pattern → Mental Model    │    │
│  │   - 每阶段需要 fact_threshold  │    │
│  │     数量的源事实支撑          │    │
│  └──────────────┬───────────────┘    │
│                 │                    │
│  ┌──────────────▼───────────────┐    │
│  │ Step 3: KnowledgeGraph       │    │  ← deep/knowledge_graph.py
│  │   - 提取实体和关系            │    │
│  │   - 创建/更新时序三元组        │    │
│  │   - 传播置信度和时间衰减      │    │
│  └──────────────┬───────────────┘    │
│                 │                    │
│  ┌──────────────▼───────────────┐    │
│  │ Step 4: 回写结果             │    │
│  │   - 新洞察写回 L2            │    │
│  │   - 心智模型存入 L3          │    │
│  │   - 图谱节点/边持久化         │    │
│  └──────────────────────────────┘    │
│                                          │
└────────┬─────────────────────────────────┘
         │
         ▼
    返回反思洞察报告
```

---

## 5. Facade 分组设计

### 5.1 StorageFacade — 存储外观

**职责边界**：所有数据的持久化读写操作

**组成子系统**：

| 子系统模块 | 文件路径 | 核心职责 |
|:---|:---|:---|
| WingRoomManager | `memory/wing_room.py` | 宫殿空间导航与分配 |
| DrawerClosetStore | `memory/drawer_closet.py` | Drawer(Closet) 双存储读写 |
| ThreeLevelIndex | `memory/index.py` | 向量/BM25/元数据三级索引维护 |
| MetaStore | `memory/meta_store.py` | 记忆元数据 CRUD |
| MarkdownStore | `memory/markdown_store.py` | Closet 大文本 Markdown 存储 |
| CollapseCompressor | `compression/collapse.py` | 记忆折叠压缩 |
| LineCompressor | `compression/line_compress.py` | 行级压缩 |
| LLMSummaryCompressor | `compression/llm_summary.py` | LLM 摘要生成 |
| MicroCompressor | `compression/micro.py` | 微压缩 |
| PriorityCompressor | `compression/priority.py` | 优先级压缩策略 |

**公共接口**：
```python
class StorageFacade:
    async def store(self, memory: MemoryRecord) -> str: ...      # 存储记忆
    async def retrieve(self, memory_id: str) -> MemoryRecord: ... # 按 ID 获取
    async def update(self, memory_id: str, **kwargs): ...         # 更新记忆
    async def delete(self, memory_id: str): ...                   # 删除记忆
    async def list_by_wing(self, wing: str) -> list[MemoryRecord]: ...  # 按空间列举
    async def compact(self, memories: list[MemoryRecord]) -> CompressedResult: ...  # 压缩
```

### 5.2 RetrievalFacade — 检索外观

**职责边界**：所有数据查询和检索操作

**组成子系统**：

| 子系统模块 | 文件路径 | 核心职责 |
|:---|:---|:---|
| HybridRetriever | `retrieval/engine.py` | 混合检索编排 |
| VectorRetriever | `retrieval/vector.py` | ChromaDB 向量相似度检索 |
| BM25Retriever | `retrieval/bm25.py` | BM25 关键词检索 |
| RRFFuser | `retrieval/rrf.py` | 倒数排序融合 |
| CrossEncoderReranker | `retrieval/reranker.py` | 二次重排序 |
| ContextManager | `context/manager.py` | 精炼/去重/预算控制 |
| VectorFactory | `retrieval/vector_factory.py` | 向量库实例工厂 |
| VectorStore (ABC) | `retrieval/vector_store.py` | 向量库抽象接口 |

**公共接口**：
```python
class RetrievalFacade:
    async def search(self, query: str, mode: str = "rag", **kwargs) -> SearchResult: ...
    async def hybrid_search(self, query: str, top_k: int = 10) -> FusedResult: ...
    async def vector_search(self, query_embedding: list[float], top_k: int) -> VectorResult: ...
    async def bm25_search(self, query: str, top_k: int) -> BM25Result: ...
    async def refine_context(self, results: list[MemoryRecord], budget: int) -> RefineResult: ...
```

### 5.3 GovernanceFacade — 治理外观

**职责边界**：所有横切面治理策略

**组成子系统**：

| 子系统模块 | 文件路径 | 核心职责 |
|:---|:---|:---|
| ConflictResolver | `governance/conflict.py` | 多源冲突检测与仲裁 |
| TemporalDecay | `governance/decay.py` | 基于时间的权重衰减 |
| ForgettingCurve | `governance/forgetting.py` | 四阶段遗忘曲线管理 |
| PrivacyManager | `governance/privacy.py` | 四级隐私分级与加密 |
| ProvenanceTracker | `governance/provenance.py` | 来源追踪与变更历史 |
| SyncEngine | `governance/sync.py` | 多实例同步协调 |
| RBACController | `governance/rbac.py` | 基于角色的访问控制 |
| AuditLogger | `governance/audit_log.py` | 操作审计日志 |
| GovernanceAuditor | `governance/auditor.py` | 定期巡检与健康报告 |
| FeedbackCollector | `governance/feedback.py` | 用户反馈收集与分析 |
| EncryptionService | `governance/encryption.py` | Fernet 加密/解密 |
| VectorClock | `governance/vector_clock.py` | 分布式向量时钟 |

**公共接口**：
```python
class GovernanceFacade:
    async def scan_conflicts(self) -> ConflictReport: ...
    async def resolve_conflict(self, conflict_id: str, strategy: str): ...
    async def set_privacy(self, memory_id: str, level: PrivacyLevel): ...
    async def run_forgetting_curve(self) -> ForgetResult: ...
    async def track_provenance(self, memory_id: str, event: ProvenanceEvent): ...
    async def sync_with_remote(self) -> SyncResult: ...
    async def check_permission(self, user: str, action: str, target: str) -> bool: ...
    async def log_audit(self, action: str, details: dict): ...
    async def run_health_check(self) -> HealthReport: ...
```

### 5.4 DeepMemoryFacade — 深层记忆外观

**职责边界**：L3 深层处理和 L0 感知

**组成子系统**：

| 子系统模块 | 文件路径 | 核心职责 |
|:---|:---|:---|
| ReflectEngine | `deep/reflect.py` | 四步反思循环 |
| ConsolidationEngine | `deep/consolidation.py` | Fact→Observation→Pattern→Model 升华 |
| KnowledgeGraph | `deep/knowledge_graph.py` | 时序三元组知识图谱 |
| PerceptionEngine | `perception/engine.py` | 信号检测与意图预测 |

**公共接口**：
```python
class DeepMemoryFacade:
    async def reflect(self, query: str, disposition: Disposition) -> ReflectResult: ...
    async def consolidate(self, domain: str) -> ConsolidationResult: ...
    async def query_graph(self, cypher_query: str) -> GraphResult: ...
    async def perceive(self, signal: RawSignal) -> PerceptionResult: ...
```

### 5.5 SyncFacade — 同步外观

**职责边界**：多实例间数据同步协调

**组成子系统**：

| 子系统模块 | 文件路径 | 核心职责 |
|:---|:---|:---|
| SyncCoordinator | `facades/sync_facade.py` | 同步策略选择与执行 |

**公共接口**：
```python
class SyncFacade:
    async def push_changes(self) -> SyncResult: ...
    async def pull_changes(self) -> SyncResult: ...
    async def resolve_sync_conflicts(self) -> list[Conflict]: ...
    async def get_sync_status(self) -> SyncStatus: ...
```

---

## 6. 检索引擎设计

### 6.1 混合检索管线

```
                         用户 Query
                            │
                            ▼
                   ┌─────────────────┐
                   │  Query Processor │
                   │  1. 文本清洗      │
                   │  2. 同义词扩展    │  ← synonyms.json
                   │  3. 中文分词增强  │  ← zh_words.json
                   └────────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │ Vector Ret. │ │ BM25 Ret.  │ │ Graph Ret. │
     │            │ │            │ │ (可选)     │
     │ Embedding  │ │ Tokenize   │ │ Entity     │
     │   ↓        │ │   ↓        │ │   ↓        │
     │ ChromaDB   │ │ SQLite FTS │ │ KG Match   │
     │ Top-K=20   │ │ Top-K=20   │ │ Top-K=10   │
     └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                   ┌───────────────┐
                   │  RRFFuser     │
                   │  k=60         │
                   │  score(d)=    │
                   │  Σ1/(k+rank) │
                   └───────┬───────┘
                           │
                   (可选启用)
                           ▼
                   ┌───────────────┐
                   │ Cross-Encoder │
                   │ Reranker      │
                   │ 精排 Top-10   │
                   └───────┬───────┘
                           ▼
                   ┌───────────────┐
                   │ContextManager │
                   │ 1. 精炼压缩   │  ≤60字摘要
                   │ 2. 语义去重   │
                   │ 3. 预算截断   │
                   │ 4. 懒加载标记 │
                   └───────┬───────┘
                           ▼
                      最终结果
```

### 6.2 RRF 融合算法详解

**Reciprocal Rank Fusion (RRF)** 是一种无需训练的多列表排序融合算法：

$$score(d) = \sum_{i=1}^{n} \frac{1}{k + rank_i(d)}$$

其中：
- $d$ : 候选文档
- $n$ : 检索源数量（默认 2：向量 + BM25）
- $k$ : 平滑常数（默认 60）
- $rank_i(d)$ : 文档 d 在第 i 个检索源中的排名

**示例计算**：

假设查询 "用户数据库偏好"，两个检索源返回：

| 排名 | 向量检索结果 | BM25 检索结果 |
|:---:|:---|:---|
| 1 | "PostgreSQL 主库" (score=0.92) | "PostgreSQL 主库" |
| 2 | "MySQL 备库" (score=0.85) | "用户喜欢 SQL" |
| 3 | "MongoDB 缓存" (score=0.78) | "数据库选型讨论" |
| 4 | "Redis 会话" (score=0.70) | "NoSQL 尝试" |

RRF 融合得分（k=60）：

| 文档 | 向量 rank | BM25 rank | RRF score |
|:---|:---:|:---:|:---:|
| PostgreSQL 主库 | 1 | 1 | 1/61 + 1/61 = **0.0328** |
| MySQL 备库 | 2 | - | 1/62 = **0.0161** |
| 用户喜欢 SQL | - | 2 | 1/62 = **0.0161** |
| MongoDB 缓存 | 3 | - | 1/63 = **0.0157** |
| 数据库选型讨论 | - | 3 | 1/63 = **0.0157** |

最终排序：PostgreSQL > MySQL > 用户喜欢 SQL > MongoDB > 数据库选型讨论

### 6.3 检索性能优化策略

| 优化点 | 技术 | 效果 |
|:---|:---|:---|
| 向量预热 | 启动时预加载前 500 条到内存 | 首次检索延迟降低 80% |
| BM25 缓存 | 会话启动时批量重建索引（2000 条/秒） | 后续 BM25 查询 < 5ms |
| 并行检索 | 向量和 BM25 同时执行（asyncio.gather） | 总延迟 = max(两者) |
| 结果缓存 | 相同查询短时间内返回缓存结果 | 重复查询零延迟 |
| 异步预取 | BackgroundTaskExecutor 预测下一轮查询并预取 | 感知零延迟 |

---

## 7. 治理引擎设计

### 7.1 冲突仲裁 (ConflictResolver)

```
候选记忆池
    │
    ▼
┌─────────────────┐
│  语义聚类        │  ← Embedding + 聚类算法
│  将可能矛盾的    │
│  记忆分组        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  矛盾检测        │  ← LLM 辅助语义矛盾识别
│  逐对比较组内    │
│  记忆的一致性    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  冲突解决        │  ← 三种策略可选
│                  │
│  Strategy: latest   → 保留最新版本
│  Strategy: confidence → 保留高置信度
│  Strategy: manual    → 标记待人工审核
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  冲突报告生成    │
│  - 冲突对列表    │
│  - 解决结果      │
│  - 影响范围      │
└─────────────────┘
```

### 7.2 遗忘曲线 (ForgettingCurve)

```
时间轴 →

  active         consolidating     archived        forgotten
  [============][==================][===============][========]
  0             7 天               37 天            127 天

  检索权重:      1.0x               0.7x            0.0x
  参与常规检索:  是                 是              否
  recall_count   正常计数           加速延缓过渡     不再计数

  ┌── recall_count 加速效应 ──┐
  │                           │
  │  recall_count=0  → 正常速度过渡              │
  │  recall_count=5  → 延长 50% 当前阶段         │
  │  recall_count=10 → 延长 100% 当前阶段        │
  │  recall_count=20 → 延长 200% 当前阶段        │
  │                           │
  └───────────────────────────┘
```

### 7.3 隐私分级 (PrivacyManager)

```
隐私等级（由低到高）：

  public ────────── 全局可见，任何实例可读
    │
  team ──────────── 团队内共享，同 group_id 可读
    │
  personal ──────── 仅创建者可读
    │
  secret ────────── 仅创建者可读 + Fernet AES-256-GCM 加密存储

加密流程：
  secret 级别记忆
    │
    ▼
  明文内容 + metadata
    │
    ▼
  Fernet(key).encrypt(content)  ← cryptography 库
    │
    ▼
  密文写入 SQLite
    │
    ▼
  读取时自动解密（key 来自环境变量 OMNIMEM_ENCRYPTION_KEY）
```

### 7.4 溯源追踪 (ProvenanceTracker)

每条记忆维护完整的溯源链：

```python
@dataclass
class ProvenanceRecord:
    memory_id: str
    created_at: datetime
    created_by: str              # 创建者（agent_id / user_id）
    source_type: SourceType      # agent_memorize / user_input / perception / consolidation
    source_reference: str        # 来源引用（如 Turn ID / 感知信号 ID）
    change_history: list[ChangeEvent]  # 变更历史
    # ChangeEvent: {timestamp, operator, field, old_value, new_value, reason}
```

### 7.5 RBAC 权限控制 (RBACController)

```
角色 (Role) → 权限 (Permission) → 操作 (Action)

Roles:
  admin      → 全部权限
  editor     → memorize / recall / reflect / govern(set_privacy, archive)
  viewer     → recall / detail only
  system     → perception / auto_memorize / forgetting (内部角色)

Permissions:
  memorize:write    → omni_memorize
  memorize:read     → omni_detail
  recall:execute    → omni_recall
  reflect:execute   → omni_reflect
  govern:conflict   → govern scan_conflicts / resolve
  govern:privacy    → govern set_privacy
  govern:lifecycle  → govern archive / delete
  audit:read        → 查看审计日志
```

### 7.6 审计日志 (AuditLogger)

```python
@dataclass
class AuditEntry:
    timestamp: datetime
    actor: str                 # 操作者
    action: str                # 操作类型
    target: str                # 目标资源 (memory_id / *)
    details: dict              # 操作详情
    ip_address: str | None     # 来源 IP（REST API 场景）
    session_id: str            # 会话 ID
    result: str                # success / denied / error
    error_message: str | None  # 错误信息（如有）
```

---

## 8. 安全体系设计

### 8.1 SecurityValidator 14 种检测模式

`SecurityValidator` 在每个写入路径上执行安全扫描，防止恶意输入污染记忆系统：

| 编号 | 检测模式 | 检测目标 | 处理方式 |
|:---:|:---|:---|:---|
| 1 | **反递归检测** | 检测是否为系统自身注入的内容 | 拒绝存储 |
| 2 | **Prompt 注入检测** | 检测试图覆盖系统指令的模式串 | 剥离/拒绝 |
| 3 | **Unicode 绕过检测** | 检测全角字符/零宽字符/同形字攻击 | 归一化 |
| 4 | **编码逃逸检测** | 检测 Base64/Hex/URL 编码的隐藏指令 | 解码后重新扫描 |
| 5 | **指令越权检测** | 检测试图修改配置/权限的操作指令 | 拒绝 |
| 6 | **数据泄露诱导检测** | 检索查询中是否有诱导输出敏感信息的模式 | 警告/过滤 |
| 7 | **SQL 注入检测** | 检测 SQL 注入模式（防御性） | 转义/拒绝 |
| 8 | **XSS payload 检测** | 检测 HTML/JS 注入（跨场景防护） | 转义 |
| 9 | **路径遍历检测** | 检测文件路径操作（Closet 存储场景） | 拒绝 |
| 10 | **OS 命令注入检测** | 检测 shell 命令模式 | 拒绝 |
| 11 | **长度爆炸检测** | 检测异常长的输入（DoS 防护） | 截断/拒绝 |
| 12 | **频率限制检测** | 检测单位时间内异常高的写入频率 | 限流/告警 |
| 13 | **语义一致性检测** | LLM 辅助检测输入是否包含矛盾/虚假信息 | 标记/降权 |
| 14 | **来源可信度验证** | 验证声明来源的真实性（防伪造） | 标记不可信 |

### 8.2 安全防线架构

```
外部输入
    │
    ▼
┌─────────────────────────────────────────┐
│ 第一道防线：输入净化 (Input Sanitization) │
│  - Unicode 归一化 (NFKC)                │
│  - 控制字符剥离                          │
│  - 长度限制                              │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 第二道防线：模式匹配 (Pattern Matching)  │
│  - 14 种检测模式逐一扫描                 │
│  - 已知攻击签名匹配                      │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 第三道防线：语义分析 (Semantic Analysis) │
│  - LLM 辅助意图判断                     │
│  - 上下文一致性验证                      │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 第四道防线：运行时隔离 (Runtime Guard)   │
│  - Saga 事务补偿                         │
│  - 操作审计日志                          │
│  - RBAC 权限检查                         │
└─────────────────────────────────────────┘
```

---

## 9. 扩展点

### 9.1 向量库抽象 (VectorStore)

```python
# retrieval/vector_store.py
class VectorStore(ABC):
    """向量存储抽象接口，支持切换不同后端"""

    @abstractmethod
    async def initialize(self, persist_dir: str, collection_name: str) -> None: ...

    @abstractmethod
    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        documents: list[str] | None = None,
    ) -> None: ...

    @abstractmethod
    async def query(
        self,
        query_embeddings: list[list[float]] | None = None,
        query_texts: list[str] | None = None,
        n_results: int = 10,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[dict]: ...

    @abstractmethod
    async def delete(self, ids: list[str] | None = None) -> None: ...

    @abstractmethod
    async def count(self) -> int: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
```

**已实现的后端**：

| 后端 | 类名 | 适用场景 |
|:---|:---|:---|
| ChromaDB | `ChromaVectorStore` | 默认，嵌入式，开发/中小规模 |
| Qdrant | `QdrantVectorStore` | 大规模生产环境，高性能 |
| FAISS | `FAISSVectorStore` | 纯内存，极速检索 |

### 9.2 LLM 后端抽象 (LLMBackend)

```python
# utils/llm_backend.py
class LLMBackend(ABC):
    """LLM 后端抽象接口"""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
```

**已实现的后端**：

| 后端 | 类名 | 说明 |
|:---|:---|:---|
| OpenAI | `OpenAIBackend` | GPT-4o / text-embedding-3-small |
| Anthropic | `AnthropicBackend` | Claude 系列 |
| Ollama | `OllamaBackend` | 本地模型 (Qwen/Llama) |
| OpenAI Compatible | `OpenAICompatibleBackend` | 任何兼容 OpenAI API 格式的服务 |

### 9.3 L4 插件系统 (InternalizationPlugin)

详见 [ADR-008](adr.md#adr-008-l4-内化层插件化)。核心接口：

```python
# internalize/plugin.py
class InternalizationPlugin(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def dependencies(self) -> list[str]: ...

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    async def initialize(self, config: dict) -> None: ...

    @abstractmethod
    async def internalize(self, memories: list[MemoryRecord]) -> InternalizationResult: ...

    @abstractmethod
    async def recall_internalized(self, query: str) -> InternalizedResult | None: ...
```

### 9.4 压缩管线 (CompressionPipeline)

```python
# compression/pipeline.py
class CompressionPipeline:
    """可组合的压缩管线"""

    def __init__(self, strategies: list[CompressionStrategy]):
        self.strategies = strategies

    async def compress(
        self,
        memories: list[MemoryRecord],
        budget: int,
        preserve_keys: list[str] | None = None,
    ) -> CompressedResult:
        """
        压缩流程：
        1. MicroCompressor   — 微压缩（去除冗余词）
        2. LineCompressor    — 行级压缩（合并相似行）
        3. CollapseCompressor — 折叠压缩（合并同类记忆）
        4. PriorityCompressor — 优先级压缩（按重要性裁剪）
        5. LLMSummaryCompressor — LLM 摘要（最终精炼）
        """
        ...
```

**可插拔的压缩策略**：每种策略可独立启用/禁用/替换。

---

## 10. 配置体系

### 10.1 配置项分组

```yaml
# ===== 核心配置 =====
save_interval: 15                  # 自动保存间隔（轮数）
auto_memorize: true                # 自动记忆开关
default_privacy: personal          # 默认隐私级别

# ===== 检索配置 =====
retrieval_mode: rag                # 默认检索模式: rag / llm
vector_backend: chromadb           # 向量数据库后端
max_prefetch_tokens: 300           # 最大预取 token 数
enable_reranker: false             # Cross-Encoder 重排序开关

# ===== 工作记忆配置 =====
budget_tokens: 4000                # 工作记忆 Token 预算
core_block_max_items: 5            # CoreBlock 最大条目数
attachment_compression_ratio: 0.3  # Attachment 压缩率

# ===== L2 存储配置 =====
drawer_table: memories             # Drawer 表名
closet_dir: ./closet              # Closet 目录
index_build_batch: 500             # 索引构建批次大小

# ===== L3 深层配置 =====
fact_threshold: 10                 # Consolidation 触发阈值（事实数）
reflect_model: default             # 反思使用的 LLM 模型
graph_persist_path: ./knowledge_graph.json  # 图谱持久化路径

# ===== L4 内化配置 =====
kv_cache_threshold: 10             # KV Cache 预填充阈值
kv_cache_max: 100                  # KV Cache 最大条目数
lora_base_model: Qwen2.5-7B        # LoRA 基座模型
lora_rank: 16                      # LoRA 秩
lora_alpha: 32                     # LoRA alpha

# ===== 治理配置 =====
conflict_strategy: latest          # 冲突解决策略
forgetting_active_ttl_days: 7      # 遗忘曲线 - 活跃期
forgetting_consolidating_ttl_days: 30  # 遗忘曲线 - 巩固期
forgetting_archived_ttl_days: 90   # 遗忘曲线 - 归档期
sync_mode: none                    # 同步模式: none / file_lock / changelog
sync_interval: 30                  # 同步间隔（秒）

# ===== 安全配置 =====
security_scan_level: standard      # 安全扫描级别: minimal / standard / strict
encryption_enabled: false          # 加密开关
audit_log_retention_days: 90       # 审计日志保留天数

# ===== LLM 配置 =====
llm_provider: openai               # LLM 提供商
llm_api_key: ${OPENAI_API_KEY}     # API Key（支持环境变量）
llm_model: gpt-4o-mini             # 默认模型
embedding_model: text-embedding-3-small  # 嵌入模型
llm_temperature: 0.7               # 生成温度
llm_max_tokens: 2048               # 最大生成 token
```

### 10.2 配置加载优先级

```
最高优先级
    │
    ▼  代码显式设置 (provider._config.set("key", value))
    │
    ▼  环境变量 (OMNIMEM_*)
    │
    ▼  配置文件 (~/.omnimem/config.yaml 或项目 config.yaml)
    │
    ▼  默认值 (OmniMemConfig.defaults)
    │
最低优先级
```

### 10.3 热重载机制

- 每 10 轮自动检查配置文件变更
- 支持手动触发：`provider._config.reload()`
- 部分配置项支持运行时修改（如 budget_tokens、retrieval_mode）
- 敏感配置项（如 encryption_key）修改后需重启生效

---

## 附录 A: 相关文档

| 文档 | 路径 | 说明 |
|:---|:---|:---|
| 架构决策记录 | [docs/adr.md](adr.md) | 8 个关键架构决策的详细记录 |
| 快速入门 | [docs/quickstart.md](quickstart.md) | 最快上手指南 |
| API 参考 | [docs/api_reference.md](api_reference.md) | 完整 API 文档 |
| 配置参考 | [docs/config_reference.md](config_reference.md) | 所有配置项说明 |
| 贡献指南 | [CONTRIBUTING.md](../CONTRIBUTING.md) | 开发规范与流程 |
| 变更日志 | [CHANGELOG.md](../CHANGELOG.md) | 版本历史 |

## 附录 B: 术语表

| 术语 | 英文 | 定义 |
|:---|:---|:---|
| 感知层 | Perception Layer | L0，实时信号捕获层 |
| 工作记忆 | Working Memory | L1，短期高频访问层 |
| 结构化记忆 | Structured Memory | L2，长期持久化层 |
| 深层记忆 | Deep Memory | L3，知识升华层 |
| 内化记忆 | Internalized Memory | L4，模型参数层 |
| 宫殿记忆法 | Method of Loci / Memory Palace | L2 空间组织隐喻 |
| 抽屉 | Drawer | L2 高频存储单元 |
| 壁橱 | Closet | L2 低频大文本存储单元 |
| 侧翼 | Wing | L2 顶层空间分类 |
| 房间 | Room | L2 二级空间分类 |
| 常驻块 | CoreBlock | L1 核心上下文容器 |
| 压缩附件 | CompactAttachment | L1 历史状态压缩体 |
| 反思引擎 | Reflect Engine | L3 四步反思循环 |
| 升华引擎 | Consolidation Engine | L3 知识提炼管线 |
| 遗忘曲线 | Forgetting Curve | 基于艾宾浩斯的生命周期管理 |
| 倒数排序融合 | RRF | 多路检索结果融合算法 |
| 外观模式 | Facade Pattern | 子系统分组封装模式 |
| 编排器 | Provider | 系统主入口和生命周期管理者 |
| 分身 | Shade | L4 LoRA 微调的角色实例 |
