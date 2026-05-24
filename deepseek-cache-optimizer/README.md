# DeepSeek Cache Optimizer v1.1

参考 Reasonix 架构的三层优化插件。

## 三大支柱

### Pillar 1 — 缓存优先循环
| 机制 | 钩子 | 说明 |
|------|------|------|
| **工具排序** | pre_llm_call | tools 按 name 字典序排列，字节级稳定 |
| **前缀保护压缩** | pre_llm_call | 消息 >14 条时压缩中间历史，保留 system + 最近 4 轮 |

### Pillar 2 — 工具调用修复
| 机制 | 钩子 | 说明 |
|------|------|------|
| **Call-Storm 检测** | post_tool_call | 相同 (tool, args) 在 5 次窗口内重复 ≥3 次 → 告警 |
| **失败信号升级** | pre_llm_call | 连续 3 次工具失败 → 自动升级模型 |

升级映射：
- mimo-v2.5 → mimo-v2.5-pro
- deepseek-v4-flash → deepseek-v4-pro
- gpt-4o-mini → gpt-4o

### Pillar 3 — 成本控制
| 机制 | 钩子 | 说明 |
|------|------|------|
| **轮末自动压缩** | transform_tool_result | 工具结果 >3000 token → 截断保留头尾 60%+20% |

## 统计

- 缓存命中率：`~/.hermes/deepseek_cache_stats.json`
- 日志输出：每 5 次 API 请求或 30 秒保存

## 配置

自动启用，无需配置。常量可在 `__init__.py` 顶部调整：
- `TOOL_RESULT_CAP_TOKENS = 3000` — 压缩阈值
- `FAILURE_ESCALATION_THRESHOLD = 3` — 升级阈值
- `STORM_WINDOW = 5` / `STORM_THRESHOLD = 3` — 风暴检测
