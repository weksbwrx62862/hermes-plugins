# Skill Router Plugin v1.1 — 技能智能路由

## 概述

使用微调嵌入模型预筛选相关技能，在每次 LLM 调用前自动注入相关技能推荐。

## 核心能力

- **微调嵌入模型**：使用 v3 模型（302 条训练数据）
- **pre_llm_call 钩子**：在每次 LLM 调用前自动预筛选技能
- **上下文注入**：将预筛选结果注入用户消息（不破坏系统提示缓存）
- **预计算嵌入**：首次加载时预计算所有技能嵌入，后续检索 <1s

## 安装

插件已安装在 `~/.hermes/plugins/skill-router/`

## 配置

在 `~/.hermes/config.yaml` 中添加：

```yaml
plugins:
  skill-router:
    enabled: true
    model_path: ~/.hermes/skills/devops/skill-router-scalable/fine-tuned-model-v3
    db_path: ~/.hermes/skill_index.db
    top_k: 5
```

## 使用方式

### 1. 自动注入（推荐）

启用插件后，每次用户消息都会自动触发预筛选：

```
用户: 帮我调试 Python 脚本

[自动注入的上下文]
[Skill Router: Related skills found for your query]
  - python-debugpy: Debug Python: pdb REPL + debugpy remote (DAP)...
  - jupyter-live-kernel: Iterative Python via live Jupyter kernel...
  - codebase-inspection: Inspectcodebases w/ pygount...
[Consider loading the most relevant skill with skill_view(name)]
```

### 2. 工具调用

LLM 也可以主动调用 `skill_search` 工具：

```json
{
  "name": "skill_search",
  "arguments": {
    "query": "帮我调试 Python 脚本",
    "top_k": 5
  }
}
```

## 架构

```
用户消息
    ↓
pre_llm_call 钩子
    ↓
skill_router.search_skills()
    ↓
预计算的技能嵌入 + 查询嵌入
    ↓
余弦相似度 Top-K
    ↓
注入上下文到用户消息
    ↓
LLM 看到相关技能推荐
```

## 性能

- **首次加载**：~30s（加载模型 + 预计算 146 个技能嵌入）
- **后续检索**：<1s（使用预计算的嵌入）
- **准确率**：Top-1: 59.3%，Top-3: 78.1%（302 条测试集）

## 文件结构

```
~/.hermes/plugins/skill-router/
├── __init__.py      # 插件主文件
├── plugin.yaml      # 插件配置
└── README.md        # 本文件

~/.hermes/skills/devops/skill-router-scalable/
├── training_data.json          # 302 条训练数据
├── fine-tuned-model-v3/        # v3 模型
└── scripts/
```

## 持续改进

### 收集真实查询

当用户查询未命中正确技能时，可以将查询加入训练数据：

```python
# 添加到 ~/.hermes/skills/devops/skill-router-scalable/training_data.json
{
  "query": "用户查询",
  "positive": "正确技能名",
  "negatives": ["相似但错误的技能1", "相似但错误的技能2"]
}
```

### 重新微调

```python
# 运行微调脚本
cd ~/.hermes/skills/devops/skill-router-scalable
python3 scripts/fine_tune_embedding.py --train
```

## 相关文件

- 技能索引数据库：`~/.hermes/skill_index.db`
- 微调模型：`~/.hermes/skills/devops/skill-router-scalable/fine-tuned-model-v3/`
- 训练数据：`~/.hermes/skills/devops/skill-router-scalable/training_data.json`
