# OmniMem 快速入门

5 分钟上手 OmniMem — 五层混合记忆系统。

---

## 安装

```bash
pip install omnimem
```

核心依赖会自动安装：`chromadb`、`rank-bm25`、`tiktoken`、`pyyaml`。

可选依赖：

```bash
pip install omnimem[crypto]
pip install omnimem[embeddings]
pip install omnimem[mcp]
pip install omnimem[lora]
pip install omnimem[all]
```

---

## 方式一：独立 SDK 模式

无需 Hermes 框架，直接创建 OmniMem 实例并调用记忆操作。

```python
from omnimem.sdk import OmniMemSDK

sdk = OmniMemSDK()

sdk.memorize("用户偏好深色主题", memory_type="preference")

sdk.memorize("项目使用 Python 3.12", memory_type="fact", confidence=5)

result = sdk.recall("主题偏好")
print(result)

reflection = sdk.reflect("用户的技术偏好")
print(reflection)

sdk.close()
```

使用上下文管理器自动关闭：

```python
from omnimem.sdk import OmniMemSDK

with OmniMemSDK() as sdk:
    sdk.memorize("部署环境是 Linux", memory_type="fact")
    result = sdk.recall("部署环境")
    print(result)
```

自定义存储目录和配置：

```python
from omnimem.sdk import OmniMemSDK

sdk = OmniMemSDK(
    storage_dir="~/.my_omnimem",
    config={
        "save_interval": 10,
        "retrieval_mode": "rag",
        "enable_compression": True,
    },
)

sdk.memorize("数据库使用 PostgreSQL 16", memory_type="fact", confidence=4)

sdk.close()
```

健康检查：

```python
from omnimem.sdk import OmniMemSDK

sdk = OmniMemSDK()
health = sdk.health_check()
print(health)
sdk.close()
```

导入导出：

```python
from omnimem.sdk import OmniMemSDK

sdk = OmniMemSDK()

sdk.export_memories("./backup.json", format="json")

sdk.import_memories("./backup.json", skip_duplicates=True)

sdk.close()
```

---

## 方式二：Hermes 插件模式

OmniMem 实现了 Hermes `MemoryProvider` ABC，通过 `plugin.yaml` 自动注册。

### 配置

在 Hermes 的 `config.yaml` 中设置：

```yaml
memory:
  provider: omnimem
```

### 插件注册

`plugin.yaml` 已预置：

```yaml
name: omnimem
version: 1.0.0
description: "OmniMem — 五层混合记忆系统"
pip_dependencies:
  - "chromadb>=0.4.0,<0.7.0"
  - "rank-bm25>=0.2.0,<0.3.0"
  - "tiktoken>=0.7.0"
  - "pyyaml>=6.0"
hooks:
  - on_session_end
  - on_pre_compress
  - on_memory_write
  - on_delegation
```

### 暴露的工具

Hermes 模式下，OmniMem 自动注册 7 个工具：

- `omni_memorize` — 主动存储记忆
- `omni_recall` — 主动检索记忆
- `omni_compact` — 压缩前准备
- `omni_reflect` — L3 深层反思
- `omni_govern` — 治理操作
- `omni_detail` — 按需拉取记忆细节
- `memory` — 兼容内置 memory 工具

### 钩子回调

| 钩子 | 说明 |
|------|------|
| `on_session_end` | 会话结束时执行 Consolidation + 治理归档 |
| `on_pre_compress` | 压缩前构建 Attachment + 紧急保存 |
| `on_memory_write` | 内置记忆写入时触发冲突检测 |
| `on_delegation` | 子 Agent 完成时记录过程记忆 |

---

## 方式三：MCP 服务器模式

通过 Model Context Protocol 暴露 OmniMem 的工具接口。

### 安装

```bash
pip install omnimem[mcp]
```

### 启动

```bash
omnimem-mcp
```

指定存储目录：

```bash
omnimem-mcp --storage-dir /path/to/storage
```

### MCP 客户端配置

在 MCP 客户端的配置中添加：

```json
{
  "mcpServers": {
    "omnimem": {
      "command": "omnimem-mcp",
      "args": ["--storage-dir", "/path/to/storage"]
    }
  }
}
```

### 可用工具

MCP 模式下暴露 4 个核心工具：

- `omni_memorize` — 存储记忆
- `omni_recall` — 检索记忆
- `omni_reflect` — 深层反思
- `omni_govern` — 治理操作

---

## 配置

OmniMem 通过 `config.yaml` 文件管理配置，支持热重载。配置文件位于 `<storage_dir>/omnimem/config.yaml`。

### 常用配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `save_interval` | `15` | 每 N 轮自动保存 |
| `retrieval_mode` | `"rag"` | 默认检索模式（rag/llm） |
| `vector_backend` | `"chromadb"` | 向量存储后端 |
| `enable_compression` | `false` | 启用压缩管线 |
| `budget_tokens` | `4000` | 工作记忆 token 预算 |
| `conflict_strategy` | `"latest"` | 冲突解决策略 |
| `default_privacy` | `"personal"` | 默认隐私级别 |
| `auto_memorize` | `true` | 自动记忆感知信号 |
| `forgetting_active_days` | `7` | 活跃记忆保留天数 |
| `forgetting_consolidating_days` | `30` | 巩固记忆保留天数 |
| `forgetting_archived_days` | `90` | 归档记忆保留天数 |

完整配置项参考：[配置参考文档](config_reference.md)

### SDK 模式配置

```python
from omnimem.sdk import OmniMemSDK

sdk = OmniMemSDK(config={
    "save_interval": 10,
    "retrieval_mode": "rag",
    "enable_compression": True,
    "budget_tokens": 3000,
    "forgetting_active_days": 14,
})
```

### YAML 配置文件

```yaml
save_interval: 10
retrieval_mode: rag
vector_backend: chromadb
enable_compression: true
budget_tokens: 4000
conflict_strategy: latest
default_privacy: personal
auto_memorize: true
forgetting_active_days: 7
forgetting_consolidating_days: 30
forgetting_archived_days: 90
kv_cache_threshold: 10
kv_cache_max: 100
sync_mode: none
sync_interval: 30
```

---

## 五层架构概览

```
L0 感知层    — 主动监控 + 信号检测 + 意图预测
L1 工作记忆  — CoreBlock(常驻上下文) + Attachment(压缩后状态)
L2 结构化记忆 — Wing/Room 宫殿导航 + Drawer/Closet 双存储
L3 深层记忆  — Consolidation(事实→观察→心智模型) + 知识图谱
L4 内化记忆  — KV Cache(高频) + LoRA(深层) [可选]
```

治理引擎（横切面）：冲突仲裁 + 时间衰减 + 遗忘曲线 + 隐私分级 + 溯源追踪

---

## 下一步

- [API 参考文档](api_reference.md) — 7 种工具接口的详细参数说明
- [配置参考文档](config_reference.md) — 30+ 配置项的完整说明
- [GitHub 仓库](https://github.com/weksbwrx62862/omnimem) — 源码和 Issue 反馈
