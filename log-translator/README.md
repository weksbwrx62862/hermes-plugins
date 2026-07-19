# Hermes 日志翻译器插件 v1.1.0

把 Hermes 的英文日志/报错信息自动翻译成中文显示。

## v1.1.0 新特性

- ✅ **LRU 缓存**: 相同消息只翻译一次，缓存加速 **3.8x**
- ✅ **关键词预过滤**: 按首关键词分组，跳过不相关的规则组
- ✅ **翻译统计**: 追踪命中率、缓存命中、最活跃规则
- ✅ **去重**: 移除重复规则，65 条唯一规则
- ✅ **66 条规则覆盖**: 会话、工具、凭证、路由、限流、回退、压缩等

## 性能

```
1000 条日志翻译:
  有缓存: 4.1ms (243,672 msg/s)
  无缓存: 15.4ms (64,792 msg/s)
  缓存加速: 3.8x
```

## 使用方法

### CLI

```bash
# 测试翻译效果
python ~/.hermes/plugins/log-translator/__init__.py --test

# 查看翻译统计
python ~/.hermes/plugins/log-translator/__init__.py --stats

# 列出所有规则
python ~/.hermes/plugins/log-translator/__init__.py --list

# 安装翻译器
python ~/.hermes/plugins/log-translator/__init__.py --install --mode replace --cache-size 512
```

### Python API

```python
from log_translator import install_translator, get_translator

# 安装 (替换模式 + LRU 缓存)
translator = install_translator(mode="replace", cache_size=512)

# 查看统计
stats = translator.get_stats()
print(f"命中率: {stats['hit_rate_pct']}%, 缓存命中: {stats['cache_hits']}")

# 查看缓存
print(translator.get_cache_info())
```

### config.yaml 配置

```yaml
plugins:
  log-translator:
    enabled: true
    mode: "replace"        # replace=替换原文, append=追加中文注释
    cache_size: 512         # LRU 缓存大小 (0=禁用)
    enable_stats: true      # 启用翻译统计
```

## 翻译效果

| 原始日志 | 翻译后 |
|----------|--------|
| `tool terminal completed (2.5s, 1234 chars)` | `工具 terminal 完成 (耗时 2.5 秒, 1234 字符)` |
| `Error: Connection timeout` | `错误: Connection timeout` |
| `Model Router: strategy=auto` | `模型路由: 策略=auto` |
| `Plugin my-plugin loaded` | `插件 'my-plugin' 已加载` |

## 规则覆盖

| 类别 | 规则数 | 示例 |
|------|--------|------|
| 会话/对话 | 1 | conversation turn |
| 工具调用 | 2 | completed, failed |
| 凭证池 | 3 | exhausted, rotated, auth failure |
| Model Router | 12 | strategy, routing, keys, cooldown |
| 插件加载 | 2 | loaded, registered |
| Curator | 4 | snapshot, rollback, cron |
| 速率限制 | 2 | exceeded, bucket full |
| 模型回退 | 2 | fallback, exhausted |
| 上下文压缩 | 2 | window exceeded, compressing |
| Gateway | 3 | started, shutdown, session |
| Cron | 3 | scheduled, completed, failed |
| Delegation | 3 | delegating, subagent completed/failed |
| 文件操作 | 3 | read, written, not found |
| 通用错误 | 7 | Error, Failed, Warning, Timeout 等 |

## 架构

```
日志消息 → 关键词预过滤 → LRU 缓存查找 → 正则匹配 → 翻译输出
              ↓ 命中          ↓ 命中
         只尝试相关规则    直接返回缓存结果
```

## 许可证

MIT License
