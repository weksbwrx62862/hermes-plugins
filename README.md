<p align="center">
  <h1 align="center">Hermes Plugins</h1>
  <p align="center">Hermes Agent 框架官方插件仓库 — 智能体能力扩展的模块化解决方案</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Hermes-%3E%3D2.0.0-orange.svg" alt="Hermes">
  <img src="https://img.shields.io/badge/Git%20LFS-Enabled-blueviolet.svg" alt="Git LFS">
</p>

---

## 简介

Hermes Plugins 是一个插件 monorepo，为 Hermes Agent 框架提供统一的插件管理、分发和运行基础设施。每个插件以 `plugin.yaml` 声明元数据、依赖、hooks 和 tools，被 Hermes 运行时动态加载，为智能体注入记忆、路由、协作、演进等核心能力。

插件设计遵循"即插即用"（Plug-and-Play）原则 —— 安装依赖、复制目录、启用即可生效。

---

## 插件矩阵

| 插件 | 版本 | 类型 | 核心能力 |
|------|------|------|----------|
| [omnimem](#omnimem) | 1.0.0 | backend | 五层混合记忆系统（感知→工作→结构化→深层→内化），含治理引擎 |
| [dev-lifecycle](#dev-lifecycle) | 2.0.0 | backend | 软件开发全生命周期工作流（grill→PRD→plan→prototype→TDD→review→handoff） |
| [adaptive_multi_agent](#adaptive_multi_agent) | 1.0.0 | backend | 自适应多智能体协作调度（Generator-Verifier / Orchestrator-Subagent / Agent Teams 等 5 种模式） |
| [model-router](#model-router) | 2.2.0 | backend | 多模型智能路由（时间感知 + 成本监控 + 影子流量 + 路由矩阵配置化） |
| [self_evolution](#self_evolution) | 2.0.0 | standalone | 自进化引擎（8 阶段 Loop、3 层评测、5 维 AND 门控，技能自动优化） |
| [skill-router](#skill-router) | 2.0.0 | backend | 统一技能路由（微调中文嵌入 + core/pool 分层 + pre_llm_call 自动注入） |
| [deepseek-cache-optimizer](#deepseek-cache-optimizer) | 1.1.0 | backend | DeepSeek/MiMo prefix-cache 优化（工具排序 + 前缀保护 + storm 检测） |
| [memory](#memory) | — | backend | 记忆插件骨架（待扩展） |
| [skill_pool.bak](#skill_poolbak) | 1.0.0 | standalone | 技能池向量索引 + 语义搜索（备份存档） |

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Hermes Runtime                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Plugin   │  │ Hook     │  │ Tool     │  │ Lifecycle  │ │
│  │ Loader   │  │ Manager  │  │ Registry │  │ Manager    │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘ │
│       │              │              │              │         │
└───────┼──────────────┼──────────────┼──────────────┼─────────┘
        │              │              │              │
   ┌────▼──────────────▼──────────────▼──────────────▼────┐
   │                   plugin.yaml                         │
   │  name / version / hooks / tools / dependencies       │
   └──────────────────────┬───────────────────────────────┘
                          │
   ┌──────────────────────▼───────────────────────────────┐
   │                Plugin Implementation                  │
   │  Python modules / configs / models / tests            │
   └──────────────────────────────────────────────────────┘
```

---

## 快速开始

### 前置条件

- Python 3.10+
- [Hermes Agent](https://github.com/weksbwrx62862/hermes) >= 2.0.0
- Git LFS（用于克隆包含模型文件的插件）

### 安装

```bash
# 克隆仓库（需要 Git LFS 来拉取模型文件）
git clone https://github.com/weksbwrx62862/hermes-plugins.git
cd hermes-plugins

# 为需要模型文件的插件安装依赖
pip install -e omnimem[all]
```

### 启用插件

在 Hermes 配置中启用需要的插件：

```yaml
# hermes_config.yaml
plugins:
  - name: omnimem
    path: ./hermes-plugins/omnimem
  - name: dev-lifecycle
    path: ./hermes-plugins/dev-lifecycle
  - name: model-router
    path: ./hermes-plugins/model-router
```

---

## 核心功能详解

### omnimem

**五层混合记忆系统**，为 AI Agent 提供类人记忆能力。

```python
from omnimem import MemorySystem

ms = MemorySystem()
ms.memorize("用户偏好深色模式", tags=["preference", "ui"])
results = ms.recall("用户喜欢什么主题？")
```

**层级体系**：
- **L1 感知记忆**：会话级短期缓存，高精度瞬时召回
- **L2 工作记忆**：当前任务上下文，自动衰减
- **L3 结构化记忆**：向量索引 + BM25 混合检索
- **L4 深层记忆**：知识图谱推理 + 实体关联
- **L5 内化记忆**：LoRA 微调，模型级知识注入

**治理引擎**：冲突检测、衰减曲线、遗忘策略、隐私加密、溯源审计。

---

### dev-lifecycle

**软件开发生命周期技能包 v2**，覆盖从需求到交付的全链路工作流。

```
grill ──→ PRD ──→ plan ──→ prototype ──→ TDD ──→ debug ──→ review ──→ triage ──→ handoff
```

**v2 新特性**：
- 工作流状态机：`start` / `advance` / `rollback` / `resume` / `report`
- 质量门禁（Quality Gates）：每个阶段自动检查准入条件
- 项目上下文感知：自动理解项目结构和约定
- 遥测统计：追踪工作流耗时、吞吐量、阻塞点

---

### adaptive_multi_agent

**自适应多智能体调度**，根据任务复杂度自动选择最佳协作模式。

**5 种协作模式**：
| 模式 | 适用场景 | 复杂度 |
|------|----------|--------|
| Generator-Verifier | 代码生成 + 验证 | 低 |
| Orchestrator-Subagent | 任务分解 + 委派 | 中 |
| Agent Teams | 多角色协作 | 中高 |
| Message Bus | 事件驱动通信 | 高 |
| Shared State | 共享状态协作 | 极高 |

**核心能力**：智能模式切换、性能持久化、历史学习。

---

### model-router

**多模型智能路由 v2.2**，在成本和性能间自动平衡。

```python
# 时间感知路由 + 成本监控
route = router.select(task="文档翻译", budget=0.01)
# → 返回最佳模型和参数配置
```

**v2.2 新特性**：
- 时间感知路由（按时间段自动切换模型池）
- 成本监控面板（实时追踪 token 消耗和费用）
- 影子流量（新模型静默评估，不影响生产）
- 路由矩阵配置化（YAML 声明式路由规则）

---

### self_evolution

**自进化引擎 v2**，让技能自动发现改进空间并自我优化。

```
扫描技能池 ──→ 评估质量 ──→ 生成变体 ──→ 适应度筛选 ──→ 门控验证 ──→ 部署
```

- **8 阶段进化 Loop**：扫描 → 评估 → 生成 → 训练 → 验证 → 门控 → 部署 → 监控
- **3 层评测**：基础功能 / 边界条件 / 对抗测试
- **5 维 AND 门控**：正确性、效率、安全性、稳定性、兼容性

---

### skill-router

**统一技能路由 v2.0**，微调中文嵌入模型实现高精度技能匹配。

- **core/pool 分层**：核心技能常驻，pool 技能按需加载
- **pre_llm_call 自动注入**：在 LLM 调用前自动注入相关技能上下文
- **语义搜索 + 反馈学习**：用户反馈持续优化匹配精度

---

### deepseek-cache-optimizer

**DeepSeek/MiMo prefix-cache 优化器**，最大化缓存命中率。

- **Pillar 1**：工具排序 + 前缀保护压缩
- **Pillar 2**：call-storm 检测 + 失败信号升级
- **Pillar 3**：轮末自动压缩

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 记忆系统 | ChromaDB, BM25, FAISS, Sentence-Transformers |
| 知识图谱 | NetworkX, 自定义图引擎 |
| LLM 集成 | OpenAI SDK, httpx |
| 进化引擎 | 遗传算法 (GEPA), 适应度函数 |
| 缓存优化 | DeepSeek API prefix-cache |
| 嵌入检索 | 微调中文 BGE 嵌入 |
| 数据格式 | YAML (plugin.yaml), JSON |
| 测试框架 | pytest, pytest-asyncio |
| 代码质量 | Ruff, MyPy, pre-commit |

---

## 项目结构

```
hermes-plugins/
├── .gitignore
├── .gitattributes
├── README.md
├── __init__.py
├── adaptive_multi_agent/       # 自适应多智能体调度
│   ├── plugin.yaml
│   ├── engine.py
│   ├── graph.py
│   ├── handlers.py
│   ├── persistence.py
│   ├── schemas.py
│   └── subagent.py
├── deepseek-cache-optimizer/   # DeepSeek 缓存优化
│   ├── plugin.yaml
│   ├── __init__.py
│   └── README.md
├── dev-lifecycle/              # 开发生命周期技能包
│   ├── plugin.yaml
│   ├── config.py
│   ├── constants.py
│   ├── context.py
│   ├── gates.py
│   ├── handlers.py
│   ├── schemas.py
│   ├── state.py
│   ├── telemetry.py
│   └── tests/
├── memory/                     # 记忆插件（骨架）
│   └── __init__.py
├── model-router/               # 多模型智能路由
│   ├── plugin.yaml
│   ├── __init__.py
│   └── cost_monitor.py
├── omnimem/                    # 五层混合记忆系统
│   ├── plugin.yaml
│   ├── pyproject.toml
│   ├── core/                   # 核心引擎
│   ├── deep/                   # 深度记忆
│   ├── governance/             # 治理模块
│   ├── memory/                 # 记忆存储
│   ├── models/                 # 嵌入/重排模型 (LFS)
│   ├── retrieval/              # 检索引擎
│   ├── compression/            # 压缩管道
│   └── tests/                  # 测试套件
├── self_evolution/             # 自进化引擎
│   ├── plugin.yaml
│   ├── core/                   # 核心引擎
│   ├── pipeline/               # 优化管道
│   └── triggers/               # 触发机制
├── skill-router/               # 统一技能路由
│   ├── plugin.yaml
│   ├── __init__.py
│   └── scripts/
└── skill_pool.bak/             # 技能池（备份）
    ├── plugin.yaml
    └── __init__.py
```

---

## 开发指南

### 创建新插件

1. 在根目录下创建插件目录 `my-plugin/`
2. 添加 `plugin.yaml` 声明元数据、hooks、tools
3. 实现插件逻辑（`__init__.py` 作为入口）
4. 参考现有插件（推荐 `model-router` 作为最简参考）

### 插件规范

每个插件必须包含：
- `plugin.yaml`：插件声明文件（name, version, hooks, tools, dependencies）
- `__init__.py`：插件入口模块
- 符合 Hermes Plugin API 的工具和钩子实现

### 模型文件管理

`omnimem/models/` 目录下的嵌入和重排模型通过 Git LFS 管理。如需添加新模型：

```bash
git lfs track "*.safetensors" "*.bin" "*.msgpack"
git add models/my-model/
git commit -m "feat(models): add new embedding model"
```

---

## License

MIT License

---

## Security

如需报告安全漏洞，请参阅 [SECURITY.md](omnimem/SECURITY.md) 或通过 GitHub Security Advisories 私密报告。

---

<p align="center">
  <sub>Built with ❤️ for the Hermes Agent ecosystem</sub>
</p>