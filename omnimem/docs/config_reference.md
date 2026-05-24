# OmniMem 配置参考文档

OmniMem 通过 `config.yaml` 文件管理所有配置项，支持运行时热重载（每 10 轮检测文件变更）。配置文件位于 `<storage_dir>/omnimem/config.yaml`。

---

## 目录

- [存储配置](#存储配置)
- [检索配置](#检索配置)
- [治理配置](#治理配置)
- [压缩配置](#压缩配置)
- [向量库配置](#向量库配置)
- [L3 深层记忆配置](#l3-深层记忆配置)
- [L4 内化记忆配置](#l4-内化记忆配置)
- [同步配置](#同步配置)
- [LLM 配置](#llm-配置)
- [系统提示配置](#系统提示配置)
- [配置热重载](#配置热重载)

---

## 存储配置

控制记忆的持久化行为。

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_interval` | integer | `15` | 每 N 轮对话自动执行一次存档检查点。较小的值更频繁保存但开销更大 |
| `auto_memorize` | boolean | `true` | 是否启用感知驱动的自动记忆写入。关闭后仅通过 `omni_memorize` 工具手动存储 |
| `default_privacy` | string | `"personal"` | 新记忆的默认隐私级别。可选值：`public`、`team`、`personal`、`secret` |
| `max_prefetch_tokens` | integer | `300` | prefetch（预取注入）阶段的最大 token 预算。控制每轮自动注入的记忆量 |

---

## 检索配置

控制记忆检索的行为和策略。

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `retrieval_mode` | string | `"rag"` | 默认检索模式。`rag`：快速向量+BM25 混合检索（毫秒级）；`llm`：深度推理+意图预测+同义词扩展（秒级） |
| `enable_reranker` | boolean | `false` | 是否启用 Cross-Encoder 重排序。需要安装 `sentence-transformers`（`pip install omnimem[embeddings]`） |
| `budget_tokens` | integer | `4000` | 工作记忆的 token 预算上限。影响 `omni_compact` 和上下文注入的总量 |
| `conflict_strategy` | string | `"latest"` | 冲突解决策略。`latest`：保留最新条目；`confidence`：保留置信度最高的条目；`manual`：需手动解决 |

---

## 治理配置

控制遗忘曲线、冲突检测和隐私治理。

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `forgetting_active_days` | integer | `7` | 活跃记忆保留天数。在此期间记忆处于 active 状态，可被正常检索 |
| `forgetting_consolidating_days` | integer | `30` | 巩固记忆保留天数。超过活跃期后进入 consolidating 状态，仍可检索但优先级降低 |
| `forgetting_archived_days` | integer | `90` | 归档记忆保留天数。超过巩固期后归档，不再参与常规检索，但可通过 `omni_govern(reactivate)` 恢复 |

### 遗忘曲线阶段

```
active → consolidating → archived → (自动清理)
  7天       30天           90天
```

---

## 压缩配置

控制上下文压缩管线的行为。

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_compression` | boolean | `false` | 是否启用 5 层压缩管线。在 `on_pre_compress` 钩子中执行：collapse → priority → micro → line_compress → llm_summary |

### 压缩管线阶段

```
collapse      — 块级折叠（合并重复段落）
priority      — 优先级过滤（移除低价值内容）
micro         — 微观压缩（缩写、去冗余）
line_compress — 行级压缩（精简每行表达）
llm_summary   — LLM 摘要（语义级浓缩）
```

---

## 向量库配置

控制向量存储后端的选择。

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vector_backend` | string | `"chromadb"` | 向量存储后端。可选值：`chromadb`、`qdrant`、`pgvector` |
| `vector_store_backend` | string | `"chromadb"` | 向量存储后端（别名，与 `vector_backend` 保持一致） |
| `qdrant_url` | string | `"localhost:6333"` | Qdrant 服务器地址（仅当 `vector_backend` 为 `qdrant` 时生效） |

### 后端对比

| 后端 | 部署方式 | 适用场景 |
|------|----------|----------|
| `chromadb` | 本地嵌入式 | 单实例开发/轻量部署，零外部依赖 |
| `qdrant` | 独立服务 | 大规模生产环境，支持分布式 |
| `pgvector` | PostgreSQL 扩展 | 已有 PostgreSQL 基础设施的场景 |

---

## L3 深层记忆配置

控制 Consolidation（事实→观察→心智模型升华）和知识图谱。

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fact_threshold` | integer | `10` | Consolidation 触发阈值。当待升华的记忆数量达到此值时自动执行 Consolidation 处理 |

### Consolidation 流程

```
事实(fact) → 观察(observation) → 心智模型(mental_model)
```

- 事实：原始记忆条目
- 观察：多条事实的综合归纳
- 心智模型：深层模式识别和抽象

---

## L4 内化记忆配置

控制 KV Cache（高频记忆预填充）和 LoRA（深层人格微调）。

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `kv_cache_threshold` | integer | `10` | KV Cache 自动预填充的访问次数阈值。记忆被访问达到此次数后自动预填充到 KV Cache |
| `kv_cache_max` | integer | `100` | KV Cache 最大条目数。超过时按 LRU 策略淘汰 |
| `lora_base_model` | string | `"Qwen2.5-7B"` | LoRA 微调的基座模型名称 |
| `lora_rank` | integer | `16` | LoRA 秩（rank），影响微调参数量。较大的值表达力更强但显存占用更多 |
| `lora_alpha` | integer | `32` | LoRA alpha 参数，控制微调强度。通常设为 rank 的 2 倍 |

### LoRA 依赖

LoRA 功能需要额外安装：

```bash
pip install omnimem[lora]
```

依赖包括：`peft>=0.8.0`、`transformers>=4.35.0`、`torch>=2.0.0`

### Shade（人格切片）

LoRA 支持多 shade 切换，每个 shade 代表一种人格/风格模式：

- 通过 `omni_govern(shade_switch)` 切换
- 通过 `omni_govern(shade_list)` 列出所有可用 shade
- 未知 shade 名称会自动创建

---

## 同步配置

控制多实例间的记忆同步。

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sync_mode` | string | `"none"` | 同步模式。`none`：禁用同步；`file_lock`：文件锁模式；`changelog`：变更日志模式 |
| `sync_interval` | integer | `30` | changelog 模式下的同步间隔（秒）。每 N 秒从其他实例拉取变更 |
| `sync_conflict_resolution` | string | `"latest_wins"` | 同步冲突解决策略。`latest_wins`：最新写入优先；`manual`：需手动解决 |

### 同步模式对比

| 模式 | 机制 | 适用场景 |
|------|------|----------|
| `none` | 无同步 | 单实例部署 |
| `file_lock` | 文件锁互斥 | 同机器多进程 |
| `changelog` | 变更日志 + 向量时钟 | 多机器分布式部署 |

---

## LLM 配置

控制 Reflect 和 Consolidation 使用的 LLM 客户端。

LLM 凭证通过环境变量加载，优先级从高到低：

1. 环境变量（`OPENAI_API_KEY`、`OPENAI_BASE_URL` 等）
2. Hermes 环境变量
3. Hermes 配置文件

| 环境变量 | 说明 |
|----------|------|
| `OPENAI_API_KEY` | LLM API Key |
| `OPENAI_BASE_URL` | LLM API Base URL |
| `HERMES_LLM_MODEL` | 默认模型名称 |

未配置 LLM 时，Reflect 和 Consolidation 功能将降级运行（返回缓存结果或跳过 LLM 调用）。

---

## 系统提示配置

控制系统提示注入的行为。

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `system_prompt_char_limit` | integer | `500` | 系统提示中核心记忆摘要的字符预算基数。实际预算会根据查询关键词数量动态扩展（每个关键词 +40 字符，上限 +300） |

---

## 配置热重载

OmniMem 支持运行时配置热重载：

- 每 10 轮对话自动检测 `config.yaml` 文件变更
- 文件修改时间变化时自动重新加载
- 无需重启服务即可生效

```python
from omnimem.config import OmniMemConfig
from pathlib import Path

config = OmniMemConfig(Path("~/.omnimem/omnimem"))

config.set("save_interval", 10)

config.save()

reloaded = config.reload(force=True)
```

---

## 完整配置示例

```yaml
save_interval: 15
retrieval_mode: rag
vector_backend: chromadb
vector_store_backend: chromadb
qdrant_url: localhost:6333
max_prefetch_tokens: 300
budget_tokens: 4000
fact_threshold: 10
enable_reranker: false
conflict_strategy: latest
default_privacy: personal
auto_memorize: true
kv_cache_threshold: 10
kv_cache_max: 100
lora_base_model: Qwen2.5-7B
lora_rank: 16
lora_alpha: 32
sync_mode: none
sync_interval: 30
sync_conflict_resolution: latest_wins
forgetting_active_days: 7
forgetting_consolidating_days: 30
forgetting_archived_days: 90
enable_compression: false
```

---

## 相关文档

- [快速入门](quickstart.md) — 5 分钟上手指南
- [API 参考](api_reference.md) — 7 种工具接口的详细参数说明
