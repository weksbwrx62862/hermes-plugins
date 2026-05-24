<div align="center">

# OmniMem

**五层认知记忆引擎 -- 为 AI Agent 提供从感知到内化的完整记忆管理**

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-227%20passed-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-improving-yellow)]()

</div>

## 简介

OmniMem 是一个为 AI Agent 设计的**五层混合认知记忆系统**，模拟人类从感知到内化的完整记忆生命周期。它提供结构化存储、混合检索（向量 + BM25 + RRF）、艾宾浩斯遗忘曲线自动归档、深层反思引擎、完整治理体系和安全防护机制。支持 Hermes 插件、Python SDK、REST API 和 MCP Server 等多种接入方式，可作为 Agent 的"海马体"基础设施。

## 特性

- **五层记忆架构** -- L0 感知层 -> L1 工作记忆 -> L2 结构化记忆 -> L3 深层记忆 -> L4 内化记忆
- **混合检索引擎** -- 向量语义检索 + BM25 关键词检索 + RRF 融合排序 + 可选 Cross-Encoder 重排序
- **艾宾浩斯遗忘曲线** -- 四阶段自动归档（active -> consolidating -> archived -> forgotten）+ recall_count 加速延缓
- **反思引擎** -- 四步反思循环（Retrieve -> Analyze -> Synthesize -> Store）+ Disposition 性格系统
- **知识升华管线** -- ConsolidationEngine 四阶段提炼：Fact -> Observation -> Pattern -> Mental Model
- **时序知识图谱** -- 扩展三元组（主体, 谓词, 客体, 时间戳, 置信度, 来源）
- **治理引擎** -- 冲突仲裁 / 时间衰减 / 隐私分级（四级） / 溯源追踪 / RBAC / 审计日志
- **安全体系** -- SecurityValidator 14 种注入检测模式 + Fernet AES-256-GCM 加密
- **LoRA 分身系统** -- Shade 角色微调分身，将行为模式"烧录"进模型参数 [可选]
- **多入口支持** -- Hermes 插件 / OmniMemSDK / REST API / MCP Server / LangChain Bridge
- **Facade 架构** -- 5 个 Facade 分组 30+ 子系统，高内聚低耦合
- **Saga 事务协调** -- 派生数据最终一致性保证，失败自动补偿

## 架构概览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           入口层 (Entry Points)                           │
│                                                                           │
│    HermesPlugin  │  OmniMemSDK  │  REST API  │  MCP Server  │  LangChain   │
└────────────────────────────┬──────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         OmniMemProvider (主入口)                          │
│                                                                           │
│  ToolRouter --> memorize / recall / reflect / govern / compact / detail  │
└──────┬───────────────┬───────────────┬───────────────┬───────────────────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│StorageFacade │ │RetrievalFacade│ │DeepMemoryFacade│ │GovernanceFacade│
│              │ │              │ │              │ │              │
│ memory/      │ │ retrieval/   │ │ deep/        │ │ governance/  │
│ compression/ │ │ context/     │ │ perception/  │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                            存储层 (Storage)                              │
│                                                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐     │
│  │  SQLite   │  │ ChromaDB │  │ Markdown Files│  │ Knowledge Graph  │     │
│  │ 主存储    │  │ 向量索引  │  │ Closet 存储   │  │ L3 时序三元组    │     │
│  └──────────┘  └──────────┘  └──────────────┘  └──────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                       五层记忆架构 (L0 - L4)                              │
│                                                                           │
│  L4: Internalized  KVCache(高频缓存) + LoRA(模型微调) [可选插件]         │
│  L3: Deep          Reflect(反思) + Consolidation(升华) + KnowledgeGraph  │
│  L2: Structured    WingRoom(宫殿导航) + DrawerCloset(双存储) + 三级索引   │
│  L1: Working       CoreBlock(常驻上下文) + Attachment(压缩状态)           │
│  L0: Perception    PerceptionEngine(信号检测 + 意图预测)                  │
└──────────────────────────────────────────────────────────────────────────┘
```

## 工具接口

OmniMem 向 Agent 暴露 7 个工具接口：

| 工具 | 功能 | 典型使用场景 |
|:---|:---|:---|
| `omni_memorize` | 存储记忆 | 用户纠正、偏好声明、重要决策 |
| `omni_recall` | 检索记忆 | 回答需要历史上下文的问题 |
| `omni_reflect` | L3 深层反思 | 从原始事实提炼心智模型和洞察 |
| `omni_govern` | 治理操作 | 冲突解决、隐私设置、归档、审计 |
| `omni_compact` | 压缩准备 | 上下文过长时触发压缩 |
| `omni_detail` | 记忆详情 | 按需拉取记忆完整内容 |
| `memory` | 兼容内置 memory 工具 | 无缝替换现有记忆系统，零迁移成本 |

## 快速开始

### 环境要求

- Python 3.10+

### 独立 SDK 模式

```bash
pip install omnimem
```

```python
from omnimem.sdk import OmniMemSDK

# 初始化
sdk = OmniMemSDK()

# 存储记忆
sdk.memorize("用户偏好深色主题", memory_type="preference")

# 检索记忆
result = sdk.recall("主题偏好")
print(result)

# 关闭
sdk.close()
```

### MCP 服务器模式

```bash
pip install omnimem[mcp]
omnimem-mcp
```

### REST API 模式

```bash
pip install omnimem[rest]
omnimem-api --port 8765
```

```bash
curl -X POST http://localhost:8765/api/memorize \
  -H "Content-Type: application/json" \
  -d '{"content": "用户偏好深色主题", "memory_type": "preference"}'
```

### Hermes 插件模式

将本仓库放入 `plugins/memory/omnimem/` 目录，在 `config.yaml` 中配置：

```yaml
memory:
  provider: omnimem
```

## 安装

```bash
# 最小安装（核心依赖，L0-L3）
pip install omnimem

# 完整安装（含加密/嵌入模型/LoRA/MCP/LangChain）
pip install omnimem[all]

# 仅 SDK 模式
pip install omnimem[sdk]

# 仅 MCP 服务器
pip install omnimem[mcp]

# 含 LoRA 支持（L4 内化层）
pip install omnimem[lora]

# 开发模式（含测试/lint/类型检查依赖）
pip install -e ".[dev]"
```

## 记忆类型

| 类型 | 说明 | 示例 |
|:---:|:---|:---|
| `fact` | 客观事实 | "项目使用 Docker 部署" |
| `preference` | 用户偏好 | "用户喜欢暗色主题" |
| `correction` | 纠正信息 | "之前说的不对，实际是..." |
| `skill` | 技能/能力 | "用户擅长 React 开发" |
| `procedural` | 流程/步骤 | "部署流程：1. git pull 2. ..." |
| `event` | 事件记录 | "[Turn 15] 用户切换了分支" |

## 项目结构

```
omnimem/
├── compression/            # 数据压缩与优化
│   ├── collapse.py         # 记忆折叠压缩
│   ├── line_compress.py    # 行级压缩
│   ├── llm_summary.py      # LLM 摘要生成
│   ├── micro.py            # 微压缩
│   ├── pipeline.py         # 压缩管线编排
│   └── priority.py         # 优先级压缩策略
├── context/                # 上下文管理
│   └── manager.py          # ContextManager：精炼/去重/预算控制
├── core/                   # 核心组件
│   ├── block.py            # CoreBlock：常驻上下文块
│   ├── attachment.py       # CompactAttachment：压缩状态附件
│   ├── soul.py             # SoulSystem：身份与性格管理
│   ├── budget.py           # BudgetManager：Token 预算控制
│   ├── saga.py             # SagaCoordinator：分布式事务协调
│   ├── background.py       # BackgroundTaskExecutor：后台任务
│   ├── store_service.py    # MemoryStoreService：存储服务层
│   ├── async_provider.py   # AsyncOmniMemProvider：异步包装器
│   ├── dedup.py            # DedupEngine：语义去重
│   ├── memory_monitor.py   # MemoryMonitor：内存监控
│   ├── import_export.py    # 导入导出工具
│   └── tool_router.py      # ToolRouter：工具路由分发
├── deep/                   # 深层记忆处理（L3）
│   ├── consolidation.py    # ConsolidationEngine：知识升华
│   ├── knowledge_graph.py  # KnowledgeGraph：时序知识图谱
│   └── reflect.py          # ReflectEngine：反思引擎
├── facades/                # Facade 外观层
│   ├── storage.py          # StorageFacade：存储外观
│   ├── retrieval.py        # RetrievalFacade：检索外观
│   ├── governance.py       # GovernanceFacade：治理外观
│   ├── deep_memory.py      # DeepMemoryFacade：深层记忆外观
│   └── sync_facade.py      # SyncFacade：同步外观
├── governance/             # 治理引擎
│   ├── conflict.py         # ConflictResolver：冲突仲裁
│   ├── decay.py            # TemporalDecay：时间衰减
│   ├── forgetting.py        # ForgettingCurve：遗忘曲线
│   ├── privacy.py          # PrivacyManager：隐私分级与加密
│   ├── provenance.py       # ProvenanceTracker：溯源追踪
│   ├── sync.py             # SyncEngine：多实例同步
│   ├── rbac.py             # RBACController：权限控制
│   ├── audit_log.py        # AuditLogger：审计日志
│   ├── auditor.py          # GovernanceAuditor：巡检器
│   ├── feedback.py         # FeedbackCollector：反馈收集
│   ├── encryption.py       # EncryptionService：加密服务
│   └── vector_clock.py     # VectorClock：分布式向量时钟
├── handlers/               # API 处理器
│   ├── memorize.py         # 记忆存储处理
│   ├── recall.py           # 检索处理
│   ├── govern.py           # 治理处理
│   ├── schemas.py          # 工具模式定义
│   └── compat_handler.py   # 兼容层工具
├── internalize/            # 内化记忆（L4，可选插件）
│   ├── kv_cache.py         # KVCacheManager：高频缓存
│   ├── lora_train.py       # LoRATrainer：LoRA 训练器
│   └── plugin.py           # InternalizationPlugin：插件接口
├── memory/                 # 记忆存储（L2）
│   ├── wing_room.py        # WingRoomManager：宫殿导航
│   ├── drawer_closet.py    # DrawerClosetStore：双存储
│   ├── index.py            # ThreeLevelIndex：三级索引
│   ├── markdown_store.py   # MarkdownStore：Markdown 存储
│   ├── meta_store.py       # MetaStore：元数据存储
│   └── types.py            # 类型定义
├── perception/             # 感知层（L0）
│   └── engine.py           # PerceptionEngine：感知引擎
├── retrieval/              # 检索引擎
│   ├── engine.py           # HybridRetriever：混合检索器
│   ├── vector.py           # VectorRetriever：向量检索
│   ├── bm25.py             # BM25Retriever：关键词检索
│   ├── rrf.py              # RRF：倒数排序融合
│   ├── reranker.py         # CrossEncoderReranker：重排序
│   ├── vector_factory.py   # VectorFactory：向量库工厂
│   └── vector_store.py     # VectorStore：抽象接口
├── utils/                  # 工具函数
│   ├── llm_client.py       # AsyncLLMClient：异步 LLM 客户端
│   ├── llm_backend.py      # LLMBackend：LLM 后端抽象
│   └── security.py         # SecurityValidator：安全验证
├── config.py               # OmniMemConfig：配置管理（支持热重载）
├── provider.py             # OmniMemProvider：主入口（Hermes 模式）
├── sdk.py                  # OmniMemSDK：独立 SDK 入口
├── rest_api.py             # REST API 服务器
├── mcp_server.py           # MCP Server 入口
├── langchain_memory.py     # LangChain Memory 桥接
├── plugin.yaml             # Hermes 插件配置
└── __init__.py             # 包初始化
```

## 文档

| 文档 | 说明 |
|:---|:---|
| [快速入门](docs/quickstart.md) | 最快上手指南 |
| [API 参考](docs/api_reference.md) | 完整 API 接口文档 |
| [配置参考](docs/config_reference.md) | 所有配置项详细说明 |
| [架构设计](docs/architecture.md) | 系统架构、数据流、组件设计 |
| [架构决策记录](docs/adr.md) | 8 个关键架构决策及理由 |

## 开发

### 本地开发环境

```bash
# 克隆仓库
git clone https://github.com/weksbwrx62862/omnimem.git
cd omnimem

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 代码检查
ruff check omnimem
mypy omnimem
```

### 代码规范

- **类型注解**: 所有公共函数必须有完整的类型注解
- **Docstring**: 使用 Google 风格的 docstring
- **代码格式**: 使用 ruff 进行格式化和 lint
- **提交规范**: 使用 Conventional Commits（`feat:` / `fix:` / `docs:` 等）

## 技术栈

| 类别 | 技术 | 用途 |
|:---|:---|:---|
| 语言 | Python 3.10+ | 核心开发语言 |
| 主存储 | SQLite (WAL 模式) | 记忆元数据和内容持久化 |
| 向量数据库 | ChromaDB (默认) / Qdrant / FAISS | 语义嵌入向量的存储与检索 |
| 关键词检索 | rank-bm25 | BM25 关键词匹配 |
| Token 计数 | tiktoken | 精确 Token 统计与预算控制 |
| 配置解析 | PyYAML | YAML 配置文件解析 |
| 嵌入模型 | sentence-transformers | 语义嵌入生成（可选） |
| 加密 | cryptography (Fernet) | 隐私数据 AES-256-GCM 加密（可选） |
| LoRA 微调 | PEFT + Transformers + PyTorch | L4 内化层训练（可选） |
| 测试 | pytest | 单元测试与集成测试 |
| 代码质量 | ruff + mypy | linting 与静态类型检查 |

## 路线图

- [x] 五层记忆架构 (L0-L4)
- [x] 混合检索引擎（向量 + BM25 + RRF）
- [x] 完整治理引擎（冲突仲裁/遗忘曲线/隐私/RBAC/审计）
- [x] Saga 事务协调
- [x] 多实例同步（file_lock / changelog）
- [x] 内置 memory 工具兼容
- [x] Facade 模式重构（5 个 Facade 分组子系统）
- [x] MCP Server 支持
- [x] REST API 支持
- [x] OmniMemSDK 独立入口
- [ ] 分布式部署支持
- [ ] Web 管理界面
- [ ] 记忆可视化（知识图谱渲染）
- [ ] 自动 LoRA 训练流水线
- [ ] pgvector / Milvus 向量后端
- [ ] 多模态记忆（图像/音频）

## 常见问题

<details>
<summary><b>Q: ChromaDB 启动失败怎么办？</b></summary>

确保已安装正确版本的 ChromaDB：
```bash
pip install "chromadb>=0.4.0,<0.7.0"
```
如果问题持续，尝试清理 ChromaDB 缓存目录后重启。
</details>

<details>
<summary><b>Q: 如何查看记忆数据？</b></summary>

记忆数据存储在 SQLite 数据库中，可通过 SDK 或 REST API 查询：
```python
from omnimem.sdk import OmniMemSDK
sdk = OmniMemSDK()
results = sdk.recall("查询关键词", mode="rag")
```
</details>

<details>
<summary><b>Q: 支持哪些向量数据库？</b></summary>

当前默认支持 ChromaDB。通过 `VectorStore` 抽象接口可切换到 Qdrant 或 FAISS。在 `config.yaml` 中设置 `vector_backend` 即可。
</details>

<details>
<summary><b>Q: 如何配置加密存储？</b></summary>

安装加密依赖并设置环境变量：
```bash
pip install omnimem[all]
# 或单独安装
pip install cryptography
export OMNIMEM_ENCRYPTION_KEY="your-secure-32-byte-key"
```
然后在配置中启用 `encryption_enabled: true`。
</details>

<details>
<summary><b>Q: 记忆合并策略有哪些？</b></summary>

冲突仲裁支持三种策略：`latest`（保留最新）、`confidence`（保留高置信度）、`manual`（标记待人工审核）。通过 `conflict_strategy` 配置项设置。
</details>

## Contributing

欢迎提交 Issue 和 Pull Request！详情请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

快速开始：

```bash
git clone https://github.com/weksbwrx62862/omnimem.git
cd omnimem
pip install -e ".[dev]"
pre-commit install   # 安装 pre-commit 钩子
pytest tests/ -v     # 运行测试
```

## 致谢

OmniMem 的设计与实现受益于众多优秀的开源项目、学术研究和技术社区：

### 直接依赖

- [ChromaDB](https://github.com/chroma-core/chroma) -- 向量数据库存储
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) -- BM25 关键词检索
- [tiktoken](https://github.com/openai/tiktoken) -- Token 计数
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) -- 语义嵌入
- [cryptography](https://github.com/pyca/cryptography) -- 隐私数据加密
- [PEFT](https://github.com/huggingface/peft) / [Transformers](https://github.com/huggingface/transformers) / [PyTorch](https://github.com/pytorch/pytorch) -- LoRA 微调与模型推理

### 架构灵感

- **Hindsight** -- Reflect 工具循环、Consolidation 四阶段升华、Disposition 性格系统
- **MemPalace** -- Wing/Room/Hall 三级空间组织结构
- **MemOS / ActMemory** -- KV Cache 预填充机制、知识图谱时序三元组
- **ReMe** -- 6 字段结构化摘要设计、会话监听
- **memU** -- 主动感知引擎、意图预测
- **Anthropic managed-agents** -- 存储层/上下文管理层/上下文窗口三层分离

### 相关开源项目

[MemGPT](https://github.com/cpacker/MemGPT) · [mem0](https://github.com/mem0ai/mem0) · [Letta](https://github.com/letta-ai/letta) · [LangChain](https://github.com/langchain-ai/langchain) · [Zep](https://github.com/getzep/zep) · [CoALA](https://github.com/lingo-mit/coala) · [Generative Agents](https://github.com/joonspk-research/generative_agents)

完整的致谢列表见 [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md)。

## License

[MIT](LICENSE)

## 安全

OmniMem 支持可选的加密存储（Fernet AES-256-GCM），保护敏感记忆数据。SecurityValidator 提供 14 种注入检测模式防止自我污染。

如发现安全漏洞，请通过以下方式报告（**不要**创建公开 Issue）：

- **邮箱**: security@omnimem.dev
- **响应时间**: 72 小时内确认，30 天内修复

详见 [SECURITY.md](SECURITY.md)。

---

<div align="center">

**OmniMem** -- 让 AI 智能体拥有持久、可进化、可治理的记忆

[Back to Top](#-omnimem)

</div>
