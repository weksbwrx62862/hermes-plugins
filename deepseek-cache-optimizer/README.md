# DeepSeek Cache Optimizer v1.1

DeepSeek/MiMo prefix-cache 优化器 — 工具排序 + 前缀保护 + storm 检测 + 轮末压缩

## 概述

参考 Reasonix 架构设计的三层优化插件，最大化 DeepSeek/MiMo API 的 prefix-cache 命中率，降低 90% 成本。

## 核心特性

### 1. 工具排序优化 (Tool Sorting)
- **问题**: 工具调用顺序随机导致前缀不稳定
- **方案**: 字典序排序工具定义，保持前缀一致性
- **效果**: 提升缓存命中率

### 2. 前缀保护压缩 (Prefix Protection)
- **问题**: 上下文压缩可能破坏已缓存前缀
- **方案**: 识别不可变前缀，压缩仅作用于增量部分
- **效果**: 保护缓存投资

### 3. Storm 检测 (v1.1 新增)
- **问题**: 相同工具反复调用（call-storm）
- **方案**: 滑动窗口去重，注入反思轮
- **效果**: 避免无效调用

### 4. 轮末压缩 (v1.1 新增)
- **问题**: 工具结果过长占用上下文
- **方案**: 工具结果 >3000 token 自动压缩
- **效果**: 后续轮看摘要，需要时重新读取

### 5. 失败信号升级 (v1.1 新增)
- **问题**: 连续工具失败时应切换更强模型
- **方案**: 连续 3 次失败自动升级 v4-pro，带警告提示
- **效果**: 智能降级/升级

### 6. 缓存命中统计
- **post_api_request hook**: 记录每次调用的缓存状态
- **统计持久化**: `~/.hermes/deepseek_cache_stats.json`
- **实时监控**: 命中率、token 数、成本节省

## 缓存机制

DeepSeek 的缓存定价：
- **缓存命中**: $0.014/1M tokens (input)
- **缓存未命中**: $0.14/1M tokens (input)
- **价格差**: 10倍！

### 前缀结构

```
┌─────────────────────────────────────────┐
│ 不可变前缀 (IMMUTABLE PREFIX)            │ ← 会话期间固定
│   system + tool_specs + few_shots        │   缓存命中候选
├─────────────────────────────────────────┤
│ 仅追加日志 (APPEND-ONLY LOG)             │ ← 单调增长
│   [assistant₁][tool₁][assistant₂]...    │   保留前轮前缀
├─────────────────────────────────────────┤
│ 临时草稿 (VOLATILE SCRATCH)              │ ← 每轮重置
│   R1 思考、临时计划                       │   不发送给 API
└─────────────────────────────────────────┘
```

### 不变量
1. 前缀整个会话只算一次，hash 固定
2. 日志按追加顺序序列化，**永不改写**
3. 草稿先蒸馏再折叠进日志

## 当前效果

- **MiMo 缓存命中率**: 96-100%
- **累计缓存 token**: 7351 万+
- **累计请求**: 967 次

## 安装

```bash
git clone https://github.com/weksbwrx62862/deepseek-cache-optimizer.git ~/.hermes/plugins/deepseek-cache-optimizer
```

## 配置

```yaml
plugins:
  enabled:
    - deepseek-cache-optimizer
```

插件自动工作，无需额外配置。

## Hook 机制

| Hook | 触发时机 | 功能 |
|------|----------|------|
| `pre_tool_call` | 工具调用前 | 工具排序 |
| `post_tool_call` | 工具调用后 | 结果压缩 |
| `transform_tool_result` | 结果转换 | 轮末压缩 |
| `post_api_request` | API 请求后 | 统计记录 |

## 支持的 Provider

- ✅ DeepSeek
- ✅ MiMo
- ✅ OpenAI
- ✅ Anthropic

## 统计查看

```bash
# 查看缓存统计
cat ~/.hermes/deepseek_cache_stats.json

# 示例输出
{
  "total_requests": 967,
  "cache_hits": 931,
  "cache_hit_rate": 0.962,
  "cached_tokens": 73510000,
  "cost_saved_usd": 10.29
}
```

## 依赖

- Python 3.10+
- Hermes Agent

## License

MIT
