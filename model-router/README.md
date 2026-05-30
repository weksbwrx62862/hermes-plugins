# Model Router Plugin v2.2

多模型智能路由 — 时间感知路由 + 成本监控 + 影子流量 + 路由矩阵配置化

## 概述

为 Hermes Agent 提供智能的模型选择和负载均衡。根据任务复杂度、时间段、成本预算自动选择最佳模型，支持多密钥轮换和限流保护。

## 核心特性

### 时间感知路由
- **非高峰期** (00:00-08:00 北京时间): 优先 MiMo（0.8x 系数优惠）
- **高峰期**: 优先 DeepSeek（稳定可靠）
- **自动切换**: 根据当前北京时间动态选择

### 4种路由策略

| 策略 | 说明 |
|------|------|
| `cheapest` | 省钱模式，优先使用低成本模型 |
| `fastest` | 极速模式，优先使用低延迟模型 |
| `smartest` | 最强模式，使用最高质量模型 |
| `auto` | 自适应模式，综合考虑时间/成本/复杂度 |

### 多密钥轮转
- MiMo 3 个 API Key 自动轮换
- 均衡负载，避免单 key 限流
- 失败自动切换下一个 key

### Provider 令牌桶限流
- 单 Provider 并发控制
- 自动退避和重试
- 限流状态实时监控

### 成本监控 (v2.2 新增)
- 记录每次调用的 token 数和成本
- 按模型/Provider 统计
- 成本预算预警

### 影子流量 (v2.2 新增)
- 1% 抽样对比不同模型质量
- 路由决策效果评估
- 数据驱动的策略优化

### 路由矩阵配置化 (v2.2 新增)
- JSON 热更新，无需重启
- 灵活的任务分类规则
- 动态调整路由权重

## 模型池

当前支持的模型：

**MiMo 系列** (小米自研)
- mimo-v2.5-pro: 旗舰版，1M 上下文
- mimo-v2.5: 标准版
- mimo-v2-pro: 专业版
- mimo-v2-omni: 快速版

**DeepSeek 系列**
- deepseek-v4-pro: 推理增强
- deepseek-v4-flash: 快速版

**NVIDIA NIM**
- 8 个可用模型（GLM-5.1 超时不可用）

## 安装

```bash
git clone https://github.com/weksbwrx62862/model-router.git ~/.hermes/plugins/model-router
```

## 配置

```yaml
plugins:
  enabled:
    - model-router

model_router:
  strategy: auto  # cheapest/fastest/smartest/auto
  mimo_keys:
    - ${MIMO_API_KEY}
    - ${MIMO_API_KEY_2}
    - ${MIMO_API_KEY_3}
```

## API

### model_route
根据任务描述选择最佳模型

### model_pool
查看当前模型池状态

### model_balance
切换路由策略

### rate_limit_status
查看限流状态

### time_info
查看当前时间段和路由策略

## 技术细节

- **任务分类器**: 用便宜模型做前置分类，识别任务复杂度
- **延迟统计**: 实时跟踪各模型响应时间
- **健康检查**: 定期探活，自动下线不可用模型
- **成本系数**: 非高峰期 MiMo 0.8x 优惠

## 依赖

- Python 3.10+
- Hermes Agent

## License

MIT
