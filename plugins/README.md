# GitHub 周榜 AI Top5 插件集合

将 GitHub 周榜 AI Top5 项目集成到 Hermes Agent 的插件集合。

## 包含插件

### 1. CodeGraph 代码知识图谱
- **功能**: 代码库索引、符号搜索、依赖分析、MCP 服务
- **安装**: `curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh | sh`
- **使用**: 
  ```bash
  codegraph init -i    # 初始化项目
  codegraph query "函数名"  # 搜索符号
  codegraph serve      # 启动 MCP 服务
  ```

### 2. Understand-Anything 代码理解仪表盘
- **功能**: 生成交互式知识图谱和可视化仪表盘
- **依赖**: Node.js >= 22, pnpm >= 10
- **安装**: 
  ```bash
  git clone https://github.com/Lum1104/Understand-Anything.git
  cd Understand-Anything
  pnpm install
  pnpm --filter @understand-anything/core build
  pnpm --filter @understand-anything/skill build
  ```
- **使用**: 在 Claude Code 中运行 `/understand` 命令

### 3. Taste-Skill AI前端设计约束
- **功能**: 提供专业的前端设计规范和约束
- **技能**: 
  - `taste-skill` - 主前端设计技能
  - `redesign-skill` - 重新设计约束
  - `output-skill` - 输出完整性约束
  - `soft-skill` - 高端 agency 风格
  - `minimalist-skill` - 极简编辑风格
  - `brutalist-skill` - 粗野主义风格

## 插件架构

每个插件遵循 Hermes Agent 插件开发规范：

```
plugins/<name>/
├── __init__.py          # register(ctx) 入口
├── plugin.yaml          # 元数据
└── provider.py          # 具体实现
```

## 安装方式

1. 将插件目录复制到 `~/.hermes/plugins/`
2. 重启 Hermes Agent 或运行 `hermes plugins reload`
3. 插件将自动注册并提供工具

## 使用示例

### CodeGraph
```python
# 初始化项目索引
codegraph.init("/path/to/project")

# 搜索符号
results = codegraph.query("/path/to/project", "calculateTotal")

# 获取文件结构
files = codegraph.files("/path/to/project")
```

### Understand-Anything
```python
# 分析代码库
graph = understand.analyze("/path/to/project")

# 启动仪表盘
understand.dashboard("/path/to/project")

# 搜索知识图谱
results = understand.search("/path/to/project", "authentication")
```

### Taste-Skill
```python
# 获取设计约束
design = taste.design(page_type="landing", vibe="minimalist")

# 获取重新设计约束
redesign = taste.redesign()

# 获取设计审计约束
audit = taste.audit()
```

## 开发说明

这些插件是 Hermes Agent 的扩展，遵循以下原则：

1. **配置驱动**: 通过 `plugin.yaml` 定义元数据
2. **提供者模式**: 通过 `provider.py` 实现具体功能
3. **工具注册**: 通过 `register(ctx)` 注册工具
4. **错误隔离**: 每个插件独立，互不影响

## 相关链接

- [CodeGraph GitHub](https://github.com/colbymchenry/codegraph)
- [Understand-Anything GitHub](https://github.com/Lum1104/Understand-Anything)
- [Taste-Skill GitHub](https://github.com/Leonxlnx/taste-skill)
- [Hermes Agent 插件开发文档](~/.hermes/skills/software-development/hermes-plugin-development/SKILL.md)
