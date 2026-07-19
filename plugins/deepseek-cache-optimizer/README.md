# DeepSeek Cache Optimizer v2.0.0

DeepSeek/MiMo prefix-cache 优化器 — 工具排序 + Prompt 归一化 + 前缀保护 + TokenJuice 感知压缩 + Call-Storm 检测 + 失败升级 + Reasoning 裁剪 + Prefix 指纹监控 + Cache Miss 诊断

## 概述

参考 Reasonix 架构设计的三层优化插件，最大化 DeepSeek/MiMo API 的 prefix-cache 命中率，降低 90% 成本。

## 核心特性

### 1. 工具排序优化 (Tool Sorting)
- **问题**: 工具调用顺序随机导致前缀不稳定
- **方案**: 字典序排序工具定义，保持前缀一致性
- **效果**: 提升缓存命中率

### 2. Prompt 归一化 (Prompt Normalization)
- **问题**: 消息格式、空白或字段差异导致前缀变化
- **方案**: 统一消息结构，规范化文本与工具字段
- **效果**: 减少缓存未命中

### 3. 前缀保护压缩 (Prefix Protection)
- **问题**: 上下文压缩可能破坏已缓存前缀
- **方案**: 识别不可变前缀，压缩仅作用于增量部分
- **效果**: 保护缓存投资

### 4. TokenJuice 模式感知压缩 (TokenJuice-Aware Compression)
- **问题**: 静态压缩阈值无法适应不同场景
- **方案**: 基于 TokenJuice 模式动态调整压缩乘数
- **效果**: 成本感知的智能压缩

### 5. 自适应压缩阈值 (Adaptive Compression Threshold)
- **问题**: 工具结果长度差异大，统一阈值效果差
- **方案**: 根据结果长度与上下文动态调整压缩阈值
- **效果**: 保留关键信息，避免过度压缩

### 6. Call-Storm 检测
- **问题**: 相同工具反复调用（call-storm）
- **方案**: 滑动窗口去重，注入反思轮
- **效果**: 避免无效调用

### 7. 失败信号升级 (Failure Escalation)
- **问题**: 连续工具失败时应切换更强模型
- **方案**: 连续失败达到阈值后自动升级模型，带警告提示
- **效果**: 智能降级/升级

### 8. Reasoning 裁剪 (Reasoning Trimming)
- **问题**: Reasoning 内容过长且不稳定，污染前缀
- **方案**: 对非关键 reasoning 内容进行裁剪或折叠
- **效果**: 保持前缀稳定，降低 token 消耗

### 9. Prefix 指纹监控 (Prefix Fingerprint Monitoring)
- **问题**: 前缀细微变化难以察觉
- **方案**: 计算并监控前缀指纹，检测前缀漂移
- **效果**: 及时发现并修复缓存前缀破坏

### 10. Cache Miss 诊断 (Cache Miss Diagnosis)
- **问题**: 缓存未命中原因不明
- **方案**: 在 `post_api_request` 中分析未命中特征并记录
- **效果**: 定位前缀变化来源，持续优化

### 11. 缓存命中率反馈循环
- **post_api_request hook**: 记录每次调用的缓存命中与未命中 token
- **统计持久化**: `~/.hermes/deepseek_cache_stats.json`
- **实时监控**: 命中率、token 数、压缩次数、storm 抑制次数

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
| `transform_request` | 请求转换阶段 | 工具排序、消息归一化、Reasoning 裁剪、Prefix 指纹计算 |
| `pre_llm_call` | LLM 调用前 | 成本感知压缩乘数、失败升级提示、Prefix 指纹监控 |
| `post_tool_call` | 工具调用后 | Call-Storm 检测 |
| `transform_tool_result` | 工具结果转换 | 工具结果自适应压缩 |
| `post_api_request` | API 请求后 | 缓存命中率统计、Cache Miss 诊断 |

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
  "start_time": "2026-07-01T00:00:00",
  "last_save": "2026-07-05T12:34:56",
  "total_requests": 967,
  "total_hit_tokens": 73510000,
  "total_miss_tokens": 2940000,
  "total_tokens": 76450000,
  "total_reasoning_tokens": 1234000,
  "total_compactions": 45,
  "total_storm_suppressions": 12,
  "total_escalations": 3,
  "total_normalizations": 178,
  "by_model": {
    "deepseek-chat": {
      "requests": 900,
      "hit_tokens": 70000000,
      "miss_tokens": 2500000
    }
  }
}
```

## 依赖

- Python 3.10+
- Hermes Agent

## License

MIT
