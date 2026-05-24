# OmniMem API 参考文档

OmniMem 向 Agent 暴露 7 个工具接口，覆盖记忆的存储、检索、压缩、反思、治理、详情查询和兼容操作。

---

## 目录

- [omni_memorize](#omni_memorize)
- [omni_recall](#omni_recall)
- [omni_compact](#omni_compact)
- [omni_reflect](#omni_reflect)
- [omni_govern](#omni_govern)
- [omni_detail](#omni_detail)
- [memory](#memory)

---

## omni_memorize

主动存储一条记忆。适用于重要事实、决策、纠正、用户偏好或任何值得在未来会话中回忆的信息。

### 参数

| 名称 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `content` | string | ✅ | — | 要存储的记忆内容 |
| `memory_type` | string | ❌ | `"fact"` | 记忆类型，可选值：`fact`、`preference`、`correction`、`skill`、`procedural`、`event` |
| `confidence` | integer | ❌ | `3` | 置信度等级（1=不确定，5=非常确定），范围 1-5 |
| `scope` | string | ❌ | `"personal"` | 记忆作用域，可选值：`personal`、`project`、`shared` |
| `privacy` | string | ❌ | `"personal"` | 隐私级别，可选值：`public`、`team`、`personal`、`secret` |

### 处理流程

1. 转义字符还原（`\n`/`\t`/`\r` → 实际控制字符）
2. 安全扫描（注入攻击检测）
3. 反递归防护（拒绝存储系统注入内容）
4. privacy → scope 推导（privacy 始终覆盖 scope，确保 wing 分类一致）
5. 精确内容去重（完全相同内容跳过）
6. 语义去重（高相似度合并更新）
7. 冲突检测（同 room/同 type 的矛盾检测）
8. 写入 L2 结构化记忆
9. 写入三层索引 + 检索索引（Saga 协调）
10. L3 知识图谱提取 + Consolidation 队列提交
11. L4 KV Cache 自动预填充检查

### 返回值

| status | 说明 |
|--------|------|
| `stored` | 成功存储 |
| `duplicate_skipped` | 精确或语义重复，跳过 |
| `conflict_rejected` | 冲突严重，拒绝存储 |
| `blocked` | 安全扫描拦截 |
| `rejected` | 反递归防护拦截 |

**成功返回示例：**

```json
{
  "status": "stored",
  "memory_id": "mem-abc123",
  "wing": "personal",
  "room": "preferences-ui",
  "type": "preference",
  "privacy": "personal",
  "kv_cached": false
}
```

**冲突警告返回示例：**

```json
{
  "status": "stored",
  "memory_id": "mem-def456",
  "wing": "personal",
  "room": "facts-python",
  "type": "fact",
  "privacy": "personal",
  "kv_cached": false,
  "conflict_warning": {
    "conflict_type": "contradiction",
    "conflicting_with": "mem-xyz789",
    "reason": "Contradicts existing memory about Python version"
  }
}
```

**重复跳过返回示例：**

```json
{
  "status": "duplicate_skipped",
  "reason": "Exact content already exists",
  "existing_id": "mem-abc123"
}
```

### SDK 调用示例

```python
from omnimem.sdk import OmniMemSDK

sdk = OmniMemSDK()

result = sdk.memorize(
    content="用户偏好深色主题",
    memory_type="preference",
    confidence=4,
    scope="personal",
    privacy="personal",
)

sdk.close()
```

---

## omni_recall

搜索 OmniMem 中的相关记忆。在回答关于过去上下文、用户偏好或决策的问题之前使用。支持语义搜索和关键词搜索。

### 参数

| 名称 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | ✅ | — | 搜索查询 |
| `mode` | string | ❌ | `"rag"` | 检索模式。`rag`：快速向量+BM25 混合检索（毫秒级）；`llm`：深度推理+意图预测（秒级） |
| `max_tokens` | integer | ❌ | `1500` | 返回结果的最大 token 数 |

### 检索流程

**rag 模式（默认）：**
1. HybridRetriever.search() 执行向量+BM25+RRF 融合检索
2. 图谱检索通道补充（知识图谱三元组）
3. 时间衰减 + 隐私过滤
4. 主存储验证（过滤索引残留）
5. 最低相关性过滤（关键词验证）
6. ContextManager 精炼

**llm 模式（深度检索）：**
1. 同 rag 模式基础流程
2. 额外 store 内容搜索补充通道（同义词扩展 + 关键词重叠过滤）
3. 图谱检索通道补充
4. 后续过滤和精炼同 rag

**无结果 fallback：** 向量+BM25 均无结果时，回退到 store 全量关键词匹配。

### 返回值

| status | 说明 |
|--------|------|
| `found` | 找到相关记忆 |
| `no_results` | 未找到任何相关记忆 |

**成功返回示例：**

```json
{
  "status": "found",
  "query": "主题偏好",
  "count": 2,
  "memories": [
    {
      "memory_id": "mem-abc123",
      "content": "用户偏好深色主题",
      "type": "preference",
      "score": 0.89,
      "summary": "偏好深色主题"
    }
  ],
  "hint": "Use omni_detail with a memory_id to fetch full content."
}
```

**无结果返回示例：**

```json
{
  "status": "no_results",
  "query": "不存在的主题",
  "message": "No relevant memories found."
}
```

### SDK 调用示例

```python
from omnimem.sdk import OmniMemSDK

sdk = OmniMemSDK()

result = sdk.recall("主题偏好", mode="rag", max_tokens=1500)

result_llm = sdk.recall("用户对编程语言的态度", mode="llm")

sdk.close()
```

---

## omni_compact

手动触发上下文压缩。OmniMem 的渐进式压缩引擎会在压缩前通过 `on_pre_compress` 钩子保存上下文。当注意到上下文变长时使用。

### 参数

| 名称 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `budget` | integer | ❌ | `4000` | 压缩后的目标 token 预算 |

### 返回值

```json
{
  "status": "ready",
  "budget": 4000,
  "message": "OmniMem will save context before compaction via on_pre_compress. Trigger compaction normally — OmniMem hooks handle the rest."
}
```

### SDK 调用示例

```python
from omnimem.sdk import OmniMemSDK

sdk = OmniMemSDK()

result = sdk.compact(budget=3000)

sdk.close()
```

---

## omni_reflect

对积累的记忆进行反思，生成更深层的洞察。将原始事实整合为观察和心智模型。当需要从过去的经验中综合模式时使用。

### 参数

| 名称 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | ✅ | — | 要反思的主题或问题 |
| `disposition` | object | ❌ | — | 反思性格修饰，调整反思输出的语气和侧重点 |

#### disposition 子参数

| 名称 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `skepticism` | integer | ❌ | `3` | 怀疑程度（1=信任，5=非常谨慎），范围 1-5 |
| `literalness` | integer | ❌ | `2` | 字面程度（1=推测性，5=精确/可验证），范围 1-5 |
| `empathy` | integer | ❌ | `4` | 共情程度（1=事实导向，5=感受导向），范围 1-5 |

### 返回值

```json
{
  "status": "reflected",
  "query": "用户的学习模式",
  "observation": "用户倾向于通过实践学习，偏好代码示例而非纯理论描述",
  "mental_model": "实践型学习者，需要可运行的代码示例来理解概念",
  "confidence": 0.82,
  "reflection_depth": 2,
  "disposition_used": {
    "skepticism": 3,
    "literalness": 2,
    "empathy": 4
  }
}
```

### SDK 调用示例

```python
from omnimem.sdk import OmniMemSDK

sdk = OmniMemSDK()

result = sdk.reflect(
    query="用户的学习模式",
    disposition={
        "skepticism": 3,
        "literalness": 2,
        "empathy": 4,
    },
)

sdk.close()
```

---

## omni_govern

管理记忆治理：解决冲突、设置隐私级别、触发遗忘/归档、查看记忆溯源等。

### 参数

| 名称 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `action` | string | ✅ | — | 治理操作，见下方 action 列表 |
| `target` | string | ❌ | — | 记忆 ID 或查询字符串 |
| `params` | object | ❌ | — | 附加参数，不同 action 需要不同参数 |

### action 列表

| action | 说明 | target | params |
|--------|------|--------|--------|
| `resolve_conflict` | 解决记忆冲突。无 target 时全局扫描，有 target 时检查指定记忆 | 可选，记忆 ID | — |
| `set_privacy` | 设置记忆隐私级别，同步更新 index/store/wing | 必填，记忆 ID | `{"level": "public\|team\|personal\|secret"}` |
| `archive` | 归档记忆（软删除，走遗忘曲线） | 必填，记忆 ID | — |
| `reactivate` | 重新激活已归档的记忆 | 必填，记忆 ID | — |
| `provenance` | 查询记忆溯源信息 | 必填，记忆 ID | — |
| `forgetting_status` | 查看遗忘曲线状态 | — | — |
| `lora_train` | L4：触发 LoRA 训练 | — | `{"shade": "default", "epochs": 3}` |
| `shade_switch` | L4：切换 LoRA shade（人格切片） | shade 名称 | `{"shade": "shade名"}` |
| `shade_list` | L4：列出所有可用 shade | — | — |
| `kv_cache_stats` | L4：查看 KV Cache 统计 | — | — |
| `consolidation_stats` | L3：查看 Consolidation 统计 | — | — |
| `sync_status` | 查看同步引擎状态 | — | — |
| `sync_instances` | 查看活跃同步实例 | — | — |
| `export_memories` | 导出记忆到文件 | — | `{"output_path": "路径", "format": "json\|markdown", "wing": "可选", "memory_type": "可选"}` |
| `import_memories` | 从文件导入记忆 | — | `{"input_path": "路径", "skip_duplicates": true, "resolve_conflicts": true}` |

### 返回值示例

**resolve_conflict（发现冲突）：**

```json
{
  "status": "conflicts_found",
  "action_taken": "archived_old_entries",
  "reason": "Found 2 conflicting pairs, archived 2 old entries",
  "conflicts": [
    {
      "memory_a": {"id": "mem-001", "content": "使用Python 3.10", "type": "fact"},
      "memory_b": {"id": "mem-002", "content": "不再使用Python 3.10", "type": "correction"},
      "overlap": 0.75,
      "negation_in": "b"
    }
  ],
  "archived": ["mem-001"]
}
```

**set_privacy：**

```json
{
  "status": "updated",
  "memory_id": "mem-abc123",
  "privacy": "team",
  "wing": "team"
}
```

**archive：**

```json
{
  "status": "archived",
  "memory_id": "mem-abc123"
}
```

**forgetting_status：**

```json
{
  "status": "ok",
  "forgetting": {
    "active_count": 42,
    "consolidating_count": 15,
    "archived_count": 8
  }
}
```

**export_memories：**

```json
{
  "status": "exported",
  "count": 128,
  "path": "/path/to/export.json"
}
```

### SDK 调用示例

```python
from omnimem.sdk import OmniMemSDK

sdk = OmniMemSDK()

sdk.govern(action="resolve_conflict")

sdk.govern(action="set_privacy", target="mem-abc123", params={"level": "team"})

sdk.govern(action="archive", target="mem-abc123")

sdk.govern(action="forgetting_status")

sdk.govern(action="export_memories", params={"output_path": "./backup.json", "format": "json"})

sdk.close()
```

---

## omni_detail

按 ID 获取特定记忆的完整详情。prefetch 只注入简洁摘要，此工具允许按需加载完整内容。

### 参数

| 名称 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `action` | string | ❌ | `"list"` | 操作类型：`get`（获取详情）、`list`（列出本 turn 注入的记忆）、`events`（查询会话事件日志） |
| `memory_id` | string | ❌ | — | 记忆 ID（action=`get` 时必填） |
| `from_turn` | integer | ❌ | `0` | 事件查询的起始 turn 编号 |
| `to_turn` | integer | ❌ | 当前 turn | 事件查询的结束 turn 编号 |
| `query` | string | ❌ | — | 事件查询的可选过滤关键词 |

### action 说明

| action | 说明 | 必填参数 |
|--------|------|----------|
| `get` | 按 ID 获取记忆完整详情 | `memory_id` |
| `list` | 列出本 turn 注入的所有记忆（含 ID，供 detail 查找） | — |
| `events` | 查询指定 turn 范围的会话事件日志 | `from_turn`、`to_turn`（可选） |

### 返回值示例

**action=list：**

```json
{
  "status": "ok",
  "count": 3,
  "memories": [
    {
      "memory_id": "mem-abc123",
      "content": "用户偏好深色主题",
      "type": "preference",
      "score": 0.89
    }
  ]
}
```

**action=get：**

```json
{
  "status": "found",
  "memory_id": "mem-abc123",
  "content": "用户偏好深色主题，在所有应用中都启用 dark mode",
  "type": "preference",
  "confidence": 4,
  "privacy": "personal",
  "wing": "personal",
  "room": "preferences-ui",
  "archived": false
}
```

**action=events：**

```json
{
  "status": "ok",
  "from_turn": 0,
  "to_turn": 10,
  "count": 5,
  "events": [
    {
      "turn": 3,
      "memory_id": "evt-001",
      "content": "[create] 用户偏好深色主题",
      "type": "event",
      "stored_at": "2025-01-15T10:30:00Z"
    }
  ]
}
```

### SDK 调用示例

```python
from omnimem.sdk import OmniMemSDK

sdk = OmniMemSDK()

detail = sdk.detail(memory_id="mem-abc123")

memories = sdk.detail_list()

events = sdk.detail_events(from_turn=0, to_turn=10)

sdk.close()
```

---

## memory

兼容内置 memory 工具的简化接口。将持久化信息保存到跨会话存活的记忆中。记忆会被注入到未来的 turn 中，因此保持简洁和聚焦。

### 参数

| 名称 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `action` | string | ✅ | — | 操作类型：`add`（新增）、`replace`（替换）、`remove`（删除） |
| `target` | string | ✅ | — | 目标分类：`memory`（环境笔记/经验教训）、`user`（用户画像/偏好/习惯） |
| `content` | string | ❌ | — | 条目内容，`add` 和 `replace` 时必填 |
| `old_text` | string | ❌ | — | 用于定位条目的子串，`replace` 和 `remove` 时必填 |

### action 说明

| action | 说明 | 必填参数 |
|--------|------|----------|
| `add` | 新增一条记忆 | `content` |
| `replace` | 替换已有记忆（先归档旧条目，再写入新内容） | `content`、`old_text` |
| `remove` | 删除已有记忆（软删除，走遗忘曲线归档） | `old_text` |

### target 说明

| target | 说明 | 对应 memory_type |
|--------|------|-----------------|
| `memory` | 环境笔记、约定、经验教训 | `fact` |
| `user` | 用户偏好、沟通风格、习惯 | `preference` |

### 返回值示例

**add：**

```json
{
  "status": "stored",
  "memory_id": "mem-xyz789",
  "wing": "personal",
  "room": "facts-env",
  "type": "fact",
  "privacy": "personal",
  "kv_cached": false,
  "compat_note": "Routed from builtin 'memory' tool to OmniMem"
}
```

**replace：**

```json
{
  "status": "stored",
  "memory_id": "mem-new001",
  "wing": "personal",
  "room": "preferences-style",
  "type": "preference",
  "privacy": "personal",
  "kv_cached": false,
  "replaced_id": "mem-old001",
  "compat_note": "Replaced via builtin compat layer"
}
```

**remove：**

```json
{
  "success": true,
  "action": "archived",
  "memory_id": "mem-old001",
  "message": "user entry archived (soft delete).",
  "compat_note": "Removed via builtin compat layer (uses forgetting curve)"
}
```

### SDK 调用示例

```python
from omnimem.sdk import OmniMemSDK

sdk = OmniMemSDK()

sdk.memorize("项目使用 Python 3.12", memory_type="fact")

sdk.memorize("用户喜欢简洁的代码风格", memory_type="preference")

sdk.close()
```

> **注意：** `memory` 工具是兼容层，内部路由到 `omni_memorize`。SDK 模式下推荐直接使用 `sdk.memorize()` 并指定 `memory_type` 参数。
