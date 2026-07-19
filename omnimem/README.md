# OmniMem — 五层认知记忆系统

[![Tests](https://img.shields.io/badge/tests-410+-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

> 为 AI Agent 设计的五层认知记忆引擎，模拟人类从感知到内化的完整记忆生命周期。

---

## 架构概览

OmniMem 采用五层认知架构 + 治理引擎横切面 + Facade 分组设计：

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

       ┌──────────────────────────────────────────────────────────┐
       │                 治理引擎 (横切面)                          │
       │  ConflictResolver │ ForgettingCurve │ PrivacyManager     │
       │  ProvenanceTracker│ RBACController  │ AuditLogger        │
       │  SecurityValidator│ SagaCoordinator │ TemporalDecay      │
       └──────────────────────────────────────────────────────────┘
```

**Facade 分组**：StorageFacade / RetrievalFacade / GovernanceFacade / DeepMemoryFacade

---

## 核心特性

| 特性 | 说明 |
|:---|:---|
| **6 通道混合检索** | 向量 + BM25 + 图谱 + 同义词扩展 + Store 补充 + 全量 fallback，RRF 融合排序 |
| **FSRS 遗忘曲线** | 4 阶段生命周期 (active → consolidating → archived → forgotten)，recall_count 加速延缓 |
| **语义去重** | 精确内容去重 + 高相似度语义合并，避免记忆膨胀 |
| **冲突仲裁** | 语义聚类 → LLM 矛盾检测 → 三策略解决 (latest/confidence/manual) |
| **隐私分级** | public / team / personal / secret 四级，secret 级 Fernet AES-256-GCM 加密 |
| **Saga 事务** | 异步派生写入 + 补偿事务注册，确保索引/图谱最终一致性 |
| **知识图谱** | 时序三元组 (主体, 谓词, 客体, 时间戳, 置信度, 来源)，支持时间衰减和置信度传播 |
| **安全防线** | SecurityValidator 14 种检测模式 (反递归 / Prompt 注入 / Unicode 绕过 / 编码逃逸等) |
| **L4 内化** | KV Cache 预填充 + LoRA 分身系统 (Shade)，毫秒级高频响应 |

---

## 快速开始

### 安装

```bash
# 克隆到 plugins 目录
git clone https://github.com/weksbwrx62862/omnimem.git ~/.hermes/plugins/omnimem

# 安装依赖
cd ~/.hermes/plugins/omnimem
pip install -r requirements.txt

# MCP 服务器模式（可选）
pip install omnimem[mcp]
```

### 三种接入方式

#### 1. Python SDK

```python
from omnimem.sdk import OmniMemSDK

# 初始化（默认存储目录 ~/.omnimem）
sdk = OmniMemSDK()

# 存储记忆
result = sdk.memorize("用户偏好深色主题", memory_type="preference", confidence=4)

# 检索记忆
result = sdk.recall("主题偏好", mode="rag")

# 治理操作
sdk.govern(action="forgetting_status")

# 反思
sdk.reflect(query="用户的学习模式", disposition={"skepticism": 3, "empathy": 4})

# 关闭
sdk.close()
```

#### 2. Hermes 插件

在 `~/.hermes/config.yaml` 中启用：

```yaml
plugins:
  enabled:
    - omnimem
```

Hermes 框架会自动注册 7 个工具接口，Agent 可直接调用 `omni_memorize`、`omni_recall` 等。

#### 3. MCP 服务器

```bash
# 启动 MCP 服务器（stdio 模式）
python -m omnimem.mcp_server

# 指定存储目录
python -m omnimem.mcp_server --storage-dir /path/to/data
```

MCP 客户端可通过标准协议调用 `omni_memorize`、`omni_recall`、`omni_reflect`、`omni_govern` 四个工具。

#### 4. REST API

```bash
# 启动 REST API 服务
python -m omnimem.rest_api 8765
```

端点：`/api/memorize` `/api/recall` `/api/reflect` `/api/govern` `/api/compact` `/api/detail` `/api/export` `/api/import` `/api/health`

---

## API 参考

OmniMem 向 Agent 暴露 7 个工具接口：

| 工具 | 说明 | 核心参数 |
|:---|:---|:---|
| **omni_memorize** | 主动存储记忆 | `content`, `memory_type`, `confidence`, `scope`, `privacy` |
| **omni_recall** | 检索相关记忆 | `query`, `mode`(rag/llm), `max_tokens` |
| **omni_compact** | 手动触发上下文压缩 | `budget` |
| **omni_reflect** | 深层反思，生成洞察 | `query`, `disposition`(skepticism/literalness/empathy) |
| **omni_govern** | 治理操作 | `action`, `target`, `params` |
| **omni_detail** | 按 ID 获取记忆详情 | `action`(get/list/events), `memory_id` |
| **memory** | 兼容内置 memory 工具 | `action`(add/replace/remove), `target`, `content` |

详细参数和返回值请参阅 [docs/api_reference.md](docs/api_reference.md)。

---

## 配置

所有配置项及说明请参阅 [docs/config_reference.md](docs/config_reference.md)。

关键配置项：

```yaml
# 检索配置
retrieval_mode: rag                # rag / llm
vector_backend: chromadb           # chromadb / qdrant / faiss
enable_reranker: false

# 工作记忆
budget_tokens: 4000                # Token 预算

# 治理配置
conflict_strategy: latest          # latest / confidence / manual
forgetting_active_ttl_days: 7      # 遗忘曲线活跃期

# LLM 配置
llm_provider: openai               # openai / anthropic / ollama / openai_compatible
llm_model: gpt-4o-mini
embedding_model: text-embedding-3-small
```

配置加载优先级：代码显式设置 > 环境变量 (`OMNIMEM_*`) > 配置文件 > 默认值

---

## 测试

```bash
# 运行全部测试
cd ~/.hermes/plugins/omnimem
pytest tests/ -v

# 生成覆盖率报告
pytest tests/ --cov=omnimem --cov-report=html
```

410+ 测试用例，覆盖五层架构、Facade、Handler、治理引擎、检索管线、安全防线等全部模块。

---

## 项目结构

```
omnimem/
├── sdk.py                      # 独立 SDK 入口
├── mcp_server.py               # MCP 服务器
├── rest_api.py                 # REST API 服务
├── config.py                   # 配置管理
├── core/
│   ├── tool_router.py          # 工具路由
│   ├── saga.py                 # Saga 事务协调
│   ├── dedup.py                # 语义去重
│   ├── import_export.py        # 导入导出
│   └── engram_bridge.py        # Plur 共享记忆桥接
├── handlers/
│   ├── memorize.py             # 写入处理器
│   ├── recall.py               # 检索处理器
│   └── govern.py               # 治理处理器
├── facades/
│   ├── storage.py              # StorageFacade
│   ├── retrieval.py            # RetrievalFacade
│   ├── governance.py           # GovernanceFacade
│   ├── deep_memory.py          # DeepMemoryFacade
│   └── sync_facade.py          # SyncFacade
├── memory/                     # L2 存储子系统
│   ├── wing_room.py            # 宫殿空间导航
│   ├── drawer_closet.py        # 双存储模型
│   ├── index.py                # 三级索引
│   ├── meta_store.py           # 元数据管理
│   └── markdown_store.py       # Closet Markdown 存储
├── retrieval/                  # 检索子系统
│   ├── engine.py               # 混合检索编排
│   ├── vector.py               # 向量检索 (ChromaDB)
│   ├── bm25.py                 # BM25 关键词检索
│   ├── rrf.py                  # RRF 融合
│   ├── reranker.py             # Cross-Encoder 重排序
│   ├── vector_store.py         # 向量库抽象接口
│   └── vector_factory.py       # 向量库实例工厂
├── context/                    # 上下文管理
│   └── manager.py              # 精炼/去重/预算控制
├── compression/                # 压缩管线
│   ├── pipeline.py             # 可组合压缩管线
│   ├── collapse.py             # 折叠压缩
│   ├── line_compress.py        # 行级压缩
│   ├── llm_summary.py          # LLM 摘要
│   ├── micro.py                # 微压缩
│   └── priority.py             # 优先级压缩
├── governance/                 # 治理引擎
│   ├── conflict.py             # 冲突仲裁
│   ├── forgetting.py           # 遗忘曲线
│   ├── privacy.py              # 隐私分级
│   ├── provenance.py           # 溯源追踪
│   ├── rbac.py                 # RBAC 权限控制
│   ├── audit_log.py            # 审计日志
│   ├── decay.py                # 时间衰减
│   ├── sync.py                 # 多实例同步
│   ├── encryption.py           # Fernet 加密
│   ├── vector_clock.py         # 分布式向量时钟
│   ├── auditor.py              # 定期巡检
│   └── feedback.py             # 反馈收集
├── deep/                       # L3 深层记忆
│   ├── reflect.py              # 四步反思循环
│   ├── consolidation.py        # 知识升华
│   └── knowledge_graph.py      # 时序三元组图谱
├── perception/                 # L0 感知层
│   └── engine.py               # 信号检测与意图预测
├── internalize/                # L4 内化层
│   └── plugin.py               # 内化插件接口
├── security/                   # 安全防线
│   └── validator.py            # SecurityValidator (14 种检测)
├── utils/
│   └── llm_backend.py          # LLM 后端抽象
├── docs/
│   ├── architecture.md         # 架构设计文档
│   ├── api_reference.md        # API 参考文档
│   ├── config_reference.md     # 配置参考
│   ├── adr.md                  # 架构决策记录
│   └── quickstart.md           # 快速入门
├── examples/
│   ├── sdk_basic.py            # SDK 基本使用示例
│   ├── hermes_integration.py   # Hermes 插件集成示例
│   ├── mcp_server_example.py   # MCP 服务器使用示例
│   └── engram_bridge_example.py # EngramBridge 示例
└── tests/                      # 测试套件 (410+ 用例)
```

---

## 相关资源

- **架构设计**: [docs/architecture.md](docs/architecture.md)
- **API 参考**: [docs/api_reference.md](docs/api_reference.md)
- **配置参考**: [docs/config_reference.md](docs/config_reference.md)
- **架构决策记录**: [docs/adr.md](docs/adr.md)
- **FSRS 算法**: https://github.com/open-spaced-repetition/fsrs4anki

---

## License

MIT License - 详见 [LICENSE](LICENSE)
