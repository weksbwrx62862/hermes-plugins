# Contributing to OmniMem

感谢你对 OmniMem 的兴趣！我们欢迎各种形式的贡献，包括 bug 报告、功能请求、文档改进和代码提交。

## 开发环境搭建

### 前提条件

- Python 3.10+
- Git

### 克隆仓库

```bash
git clone https://github.com/yourusername/omnimem.git
cd omnimem
```

### 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 安装开发依赖

```bash
pip install -r requirements-dev.txt
```

### 安装 pre-commit 钩子

```bash
pre-commit install
```

## 代码规范

### 格式化

我们使用 **Ruff** 进行代码格式化和检查：

```bash
# 自动修复问题
ruff check . --fix

# 格式化代码
ruff format .
```

### 类型检查

我们使用 **MyPy** 进行静态类型检查：

```bash
mypy . --ignore-missing-imports
```

### 代码风格

- 遵循 PEP 8 规范
- 使用双引号字符串
- 行长度限制 100 字符
- 所有公共函数和类必须有文档字符串
- 优先使用类型注解

## 测试

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_core.py -v

# 生成覆盖率报告
pytest tests/ --cov=omnimem --cov-report=html
```

### 测试规范

- 所有新功能必须附带测试
- 测试文件命名：`test_<module>.py`
- 测试类命名：`Test<ClassName>`
- 测试方法命名：`test_<scenario>`
- 使用 `unittest.TestCase` 或 pytest 风格

## 提交更改

### 分支命名

- 功能分支：`feature/<feature-name>`
- Bug 修复：`fix/<bug-description>`
- 文档更新：`docs/<description>`
- 性能优化：`perf/<description>`

### Commit Message 规范

使用清晰的 commit message：

```
<type>: <subject>

<body>

<footer>
```

类型：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具相关

示例：
```
feat: add vector clock synchronization

Implement vector clock for distributed memory consistency.
Supports multi-instance conflict resolution.

Fixes #123
```

## Pull Request 流程

1. **Fork 仓库** 并克隆到本地
2. **创建分支** 从 `main` 分支：`git checkout -b feature/my-feature`
3. **提交更改** 并推送到你的 fork
4. **创建 Pull Request** 到主仓库的 `main` 分支
5. **等待审查** 并根据反馈修改

### PR 检查清单

- [ ] 代码遵循项目代码规范
- [ ] 新增/修改的功能已添加测试
- [ ] 所有测试通过
- [ ] 文档已更新
- [ ] CHANGELOG.md 已更新

## 报告 Bug

请使用 [Bug Report 模板](https://github.com/yourusername/omnimem/issues/new?template=bug_report.md) 创建 issue，并包含：

- 清晰的 bug 描述
- 复现步骤
- 预期行为 vs 实际行为
- 环境信息（Python 版本、操作系统）
- 最小可复现代码

## 请求功能

请使用 [Feature Request 模板](https://github.com/yourusername/omnimem/issues/new?template=feature_request.md) 创建 issue，并描述：

- 功能的使用场景
- 期望的解决方案
- 考虑过的替代方案

## 社区

- 在 [Discussions](https://github.com/yourusername/omnimem/discussions) 中提问和交流
- 尊重他人，保持友善

## 许可证

通过提交贡献，你同意你的代码将在 MIT 许可证下发布。
