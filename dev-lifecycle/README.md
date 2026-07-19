# Dev Lifecycle Plugin v2.0

软件开发生命周期技能包 — grill→PRD→plan→prototype→TDD→debug→review→triage→handoff 全链路工作流

## 概述

将 software-development 目录下 21 个技能组织为三段式生命周期，通过 `dev_workflow` 工具让 agent 导航开发流程，实现从需求到交付的完整覆盖。

## 三段式生命周期

### 1. 构思阶段 (Ideate)
- **grill-me** — 无情追问，深入挖掘需求
- **grill-with-docs** — 结合文档追问
- **to-prd** — 生成产品需求文档 (PRD)
- **to-issues** — 拆分为独立 Issue
- **plan** — 制定实现计划
- **writing-plans** — 编写详细计划

### 2. 构建阶段 (Build)
- **prototype** — 快速原型验证
- **spike** — 技术调研和验证
- **improve-codebase-architecture** — 代码架构优化
- **zoom-out** — 获取更广泛视角
- **test-driven-development** — TDD 测试驱动开发
- **subagent-driven-development** — 子代理驱动开发
- **user-auth-system** — 用户认证系统

### 3. 交付阶段 (Deliver)
- **systematic-debugging** — 系统化调试
- **python-debugpy** — Python 调试
- **node-inspect-debugger** — Node.js 调试
- **requesting-code-review** — 代码审查请求
- **triage** — 问题分类处理
- **handoff** — 项目交接

## 角色边界约束（借鉴 gstack）

dev-lifecycle 为关键技能定义角色边界，借鉴 gstack 的 "reviewer 只找 bug 不修 bug" 设计哲学。角色边界是**软约束**（提示性），在 `dev_workflow(action="skill")` 返回中包含，不强制阻断执行。

### 已定义边界的技能

| 技能 | 禁止行为 | 理由 |
|------|----------|------|
| requesting-code-review | 修复发现的 bug、直接修改被审查的代码 | reviewer 修 bug 会破坏客观判断 |
| systematic-debugging | 跳过假设验证步骤、盲目修改代码 | 跳过假设验证会导致误判根因 |
| test-driven-development | 先写实现再补测试、跳过红绿重构循环 | 先实现后测试会偏离 TDD 纪律 |

### 使用示例

```python
# 查询技能的角色边界
result = dev_workflow(action="skill", skill_name="requesting-code-review")
# 返回包含 role_boundary 字段：
# {
#   "role_boundary": {
#     "allowed": ["报告发现的 bug", "提供修复建议", "评估代码质量"],
#     "forbidden": ["修复发现的 bug", "直接修改被审查的代码"],
#     "rationale": "reviewer 修 bug 会破坏客观判断"
#   },
#   "hint": "角色边界约束（软约束，借鉴 gstack 设计，不强制阻断执行）：..."
# }
```

### 编程式访问

也可以通过 `schemas.enrich_skill_with_role_boundary` 或 `gates.get_role_boundary` 直接查询：

```python
from gates import get_role_boundary
b = get_role_boundary("systematic-debugging")
# b.allowed, b.forbidden, b.rationale
```

## 提供的工具

### dev_workflow

软件开发生命周期导航工具，支持以下操作：

| 操作 | 说明 |
|------|------|
| `overview` | 列出所有阶段和技能 |
| `stage` | 获取特定阶段引导 |
| `skill` | 获取技能摘要 |
| `start` | 启动新项目生命周期 |
| `advance` | 推进到下一个技能 |
| `rollback` | 回退到指定阶段 |
| `resume` | 恢复已有项目 |
| `report` | 生成进度报告 |

## 使用示例

```python
# 查看所有阶段
dev_workflow(action="overview")

# 启动新项目
dev_workflow(action="start", project_path="/path/to/project")

# 获取构建阶段引导
dev_workflow(action="stage", stage_name="build")

# 推进到 TDD
dev_workflow(action="advance", skill_name="test-driven-development")

# 生成进度报告
dev_workflow(action="report")
```

## 技能依赖图

```
ideate (构思)
  ├── grill-me / grill-with-docs
  ├── to-prd
  ├── to-issues
  └── plan / writing-plans
       ↓
build (构建)
  ├── prototype / spike
  ├── improve-codebase-architecture
  ├── test-driven-development
  └── subagent-driven-development
       ↓
deliver (交付)
  ├── systematic-debugging
  ├── requesting-code-review
  ├── triage
  └── handoff
```

## 安装

```bash
git clone https://github.com/weksbwrx62862/dev-lifecycle.git ~/.hermes/plugins/dev-lifecycle
```

## 配置

```yaml
plugins:
  enabled:
    - dev-lifecycle
```

## 特性

- **流程导航**: 自动建议下一步操作
- **状态持久化**: 记录项目当前阶段
- **回滚支持**: 可回退到任意阶段
- **进度报告**: 自动生成项目进度
- **技能集成**: 无缝调用 21 个技能

## 依赖

- Python 3.10+
- Hermes Agent

## License

MIT
