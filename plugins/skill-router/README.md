# Skill Router Plugin v3.1

统一技能路由插件，支持双后端切换。

## 架构

```
用户查询 → 域分组过滤 → 向量检索 + BM25 → 融合排序 → (可选)重排序 → Top-K 结果
```

### 双后端

| 后端 | 模型 | 大小 | 速度 | 适用场景 |
|------|------|------|------|----------|
| **lightweight** (默认) | MiniLM 微调 v7 | ~80MB | <1s | 日常使用，资源友好 |
| **skillrouter** | Qwen3-0.6B (×2) | ~2.4GB | ~5s | 大规模技能池，更高精度 |

### 核心设计

- **core 技能**：常驻 system prompt（高频技能名称列表）
- **pool 技能**：按需语义搜索 Top-K，带置信度过滤
- **完整 body 嵌入**：使用技能全文而非截断 500 字符（SkillRouter 论文核心发现）
- **增量更新**：基于 mtime 的嵌入缓存，技能变更时只重新计算变更部分
- **反馈学习**：成功/跳过反馈影响后续路由评分

## 配置

```yaml
# ~/.hermes/config.yaml
plugins.skill-router:
  enabled: true
  backend: "lightweight"          # 或 "skillrouter"
  top_k: 5
  core_skills:
    - hermes-agent
    - skill-creator
    - web-search-china
  # 混合检索分数校准策略：minmax（默认）/ sigmoid / zscore
  hybrid_calibration: minmax
  # SkillRouter 后端路径（默认即可）
  skillrouter_emb_path: "~/models/skillrouter/SkillRouter-Embedding-0.6B"
  skillrouter_rank_path: "~/models/skillrouter/SkillRouter-Reranker-0.6B"

  # 域分组配置（可选），用于控制查询命中特定域时的候选集裁剪
  domain_config:
    top_k: 3
    keywords:
      finance: ["股票", "基金", "投资", "理财"]
    map:
      finance: ["finance"]
    aggregators:
      finance: "finance-assistant"
    special_rules:
      lark:
        name_prefix: "lark-"
      finance:
        name_keywords: ["stock", "fund", "finance"]
        always_include: ["finance-assistant"]
```

## 模型存放

| 模型 | 路径 | 用途 |
|------|------|------|
| MiniLM v7 | `~/.hermes/skills/devops/skill-router-scalable/fine-tuned-model-v7` | lightweight 后端嵌入 |
| MiniLM reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (自动下载) | lightweight 后端重排序 |
| SkillRouter Emb | `~/models/skillrouter/SkillRouter-Embedding-0.6B` | skillrouter 后端嵌入 |
| SkillRouter Rank | `~/models/skillrouter/SkillRouter-Reranker-0.6B` | skillrouter 后端重排序 |
| bge-reranker-v2-m3 | `~/models/bge-reranker-v2-m3` (备用) | 未使用，已迁出插件目录 |

## 文件结构

```
skill-router/
├── __init__.py              # 插件入口（register / search_skills / pre_llm_call）
├── bm25_searcher.py         # BM25 关键词检索器
├── domain_config.py         # 域分组配置与过滤逻辑
├── hybrid_searcher.py       # 混合检索器（含分数校准策略）
├── state.py                 # PluginState 全局状态管理
├── skillrouter_backend.py   # SkillRouter Qwen3-0.6B 推理后端
├── plugin.yaml              # 插件元数据
├── README.md                # 本文档
├── .gitignore
├── scripts/
│   ├── evaluate.py          # 评估脚本（复用 BM25Searcher / HybridSearcher）
│   └── compare_calibrations.py  # 三种校准策略对比脚本
└── tests/                   # 单元测试
    ├── test_bm25_searcher.py
    ├── test_domain_config.py
    ├── test_evaluate_consistency.py
    ├── test_feedback.py
    ├── test_hybrid_searcher.py
    ├── test_pre_llm_call.py
    └── test_state.py
```

## 静态检查

项目使用 `ruff` 与 `mypy` 做代码静态检查。由于插件目录名含连字符，mypy 需要显式指定文件并开启 `--explicit-package-bases`：

```bash
# ruff（代码风格与未使用变量等）
ruff check .

# mypy（类型检查，需在插件根目录执行）
mypy --explicit-package-bases --ignore-missing-imports \
  __init__.py bm25_searcher.py domain_config.py hybrid_searcher.py \
  state.py skillrouter_backend.py scripts/evaluate.py
```

## 降级策略

1. SkillRouter 加载失败 → 自动降级到 lightweight
2. 嵌入模型不可用 → 降级到纯 BM25
3. 向量编码超时 → 降级到纯 BM25
4. Reranker 不可用 → 跳过重排序，返回原始排序
