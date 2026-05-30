# Skill Router Plugin v2.0

统一技能路由 — 微调中文嵌入 + core/pool 分层 + pre_llm_call 自动注入

## 概述

合并 skill-router + skill_pool 的优点，提供智能的技能路由能力。通过微调的中文嵌入模型实现语义检索，Top-3 准确率达 80.7%。

## 核心特性

### 微调中文嵌入模型
- **训练数据**: 1402 条技能描述对
- **模型**: fine-tuned-model-v3 (基于 hf-mirror.com)
- **损失函数**: MNR (Multiple Negatives Ranking) Loss
- **性能**: Top-3 准确率 80.7%

### core/pool 分层管理
- **Core 层**: 高频核心技能，常驻内存
- **Pool 层**: 长尾技能，按需加载
- **自动升降级**: 根据使用频率动态调整

### pre_llm_call 自动注入
- 在 LLM 调用前自动检索相关技能
- 无缝集成到 agent 循环
- 无需手动调用 skill_pool_search

### 查询缓存
- LRU 缓存热点查询
- 语义相似查询复用
- 缓存命中率监控

### 反馈循环
- 记录技能使用效果
- 动态调整权重
- 持续优化路由质量

## 数据库

- **skill_index.db**: 技能索引数据库
  - `skills` 表: 146 个技能
  - `skill_embeddings` 表: 嵌入向量
  - `bilingual_embeddings` 表: 双语嵌入

## 提供的工具

| 工具名 | 功能 |
|--------|------|
| `skill_pool_search` | 语义检索技能 |
| `skill_pool_list` | 列出所有技能 |
| `skill_pool_build` | 重建索引 |
| `skill_pool_auto_tune` | 自动调优 |
| `skill_pool_snapshot` | 获取快照 |
| `skill_pool_usage` | 使用统计 |
| `skill_pool_set_core` | 设置核心技能 |
| `skill_pool_set_pool` | 设置池技能 |

## 安装

```bash
git clone https://github.com/weksbwrx62862/skill-router.git ~/.hermes/plugins/skill-router
```

## 配置

```yaml
plugins:
  enabled:
    - skill-router

skill_router:
  model_path: ~/.hermes/skill-router/fine-tuned-model-v3
  top_k: 5
  cache_size: 1000
  enable_feedback: true
```

## CUDA 兼容性

⚠️ **重要**: GTX 1050 (CC 6.1) 不兼容 PyTorch CC 7.5+

```bash
# 必须禁用 CUDA
export CUDA_VISIBLE_DEVICES=""

# 或在代码中强制 CPU
device = "cpu"
```

## 模型训练

```bash
# 训练数据: ~/.hermes/skill-router/training_data.jsonl (302条)
# 继续微调会导致灾难性遗忘，必须从基础模型重新训练
```

## 性能基准

| 指标 | 数值 |
|------|------|
| 训练数据 | 1402 条 |
| Top-1 准确率 | ~60% |
| Top-3 准确率 | ~80.7% |
| 索引技能数 | 146 |
| 查询延迟 | <50ms |

## 依赖

- Python 3.10+
- PyTorch (CPU 模式)
- sentence-transformers
- SQLite3

## License

MIT
