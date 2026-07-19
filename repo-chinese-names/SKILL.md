# repo-chinese-names v1.1.0

GitHub 仓库中文名映射，让仓库列表显示中文名。

## v1.1.0 新特性

- **分类管理**: 3 大分类 (Hermes 插件/金融量化/工具平台)，22 个仓库
- **外部配置**: `~/.hermes/repo-names.json` 自定义映射
- **模糊匹配**: 大小写不兼容 + 部分名称匹配
- **CLI 管理**: add/remove/list/search/categories/lookup/unmapped
- **使用统计**: 追踪映射查询频率

## CLI 用法

```bash
# 列出所有映射
python __init__.py list

# 按分类筛选
python __init__.py list --category finance

# 添加映射
python __init__.py add my-repo "我的仓库" -c custom

# 搜索
python __init__.py search 路由

# 查询 (支持模糊匹配)
python __init__.py lookup Model-Router --fuzzy

# 列出分类
python __init__.py categories

# 找未映射仓库
python __init__.py unmapped repo1 repo2 repo3

# 显示使用统计
python __init__.py stats
```

## API

```python
from repo_chinese_names import (
    get_chinese_name,    # 获取中文名 (可选 fuzzy=True)
    get_category,        # 获取分类
    search_repos,        # 搜索仓库
    format_repo_list,    # 格式化列表 (可选 show_category=True)
    get_all_mappings,    # 所有映射
    get_unmapped_repos,  # 找未映射仓库
)
```

## 外部配置

文件: `~/.hermes/repo-names.json`

简单格式:
```json
{"my-repo": "我的仓库"}
```

分类格式:
```json
{"custom": {"label": "自定义", "repos": {"my-repo": "我的仓库"}}}
```
