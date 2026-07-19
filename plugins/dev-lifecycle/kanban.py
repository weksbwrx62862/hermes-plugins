"""
看板视图生成器 — 可视化开发进度

功能：
  - 任务状态可视化
  - 拖拽式操作支持
  - 进度统计
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class KanbanColumn:
    """看板列"""
    name: str
    title: str
    color: str
    tasks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class KanbanBoard:
    """看板"""
    title: str
    columns: List[KanbanColumn]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_dev_lifecycle_kanban(
    project_name: str = "Hermes 项目",
    stages: List[str] = None,
) -> KanbanBoard:
    """创建开发生命周期看板
    
    参数:
        project_name: 项目名称
        stages: 阶段列表
    
    返回:
        KanbanBoard 看板对象
    """
    if stages is None:
        stages = ["ideate", "build", "deliver"]
    
    columns = []
    
    # 为每个阶段创建列
    stage_config = {
        "ideate": {"title": "构思", "color": "#4CAF50"},
        "build": {"title": "构建", "color": "#2196F3"},
        "deliver": {"title": "交付", "color": "#FF9800"},
        "todo": {"title": "待办", "color": "#9E9E9E"},
        "in_progress": {"title": "进行中", "color": "#FF5722"},
        "review": {"title": "审查", "color": "#9C27B0"},
        "done": {"title": "完成", "color": "#4CAF50"},
    }
    
    for stage in stages:
        config = stage_config.get(stage, {"title": stage, "color": "#607D8B"})
        columns.append(KanbanColumn(
            name=stage,
            title=config["title"],
            color=config["color"],
            tasks=[],
        ))
    
    return KanbanBoard(
        title=f"{project_name} - 开发看板",
        columns=columns,
    )


def add_task_to_kanban(
    board: KanbanBoard,
    column_name: str,
    task_id: str,
    task_title: str,
    task_description: str = "",
    priority: str = "medium",
    assignee: str = "",
    labels: List[str] = None,
) -> None:
    """添加任务到看板
    
    参数:
        board: 看板对象
        column_name: 列名称
        task_id: 任务 ID
        task_title: 任务标题
        task_description: 任务描述
        priority: 优先级 (low, medium, high, critical)
        assignee: 负责人
        labels: 标签列表
    """
    # 查找列
    column = None
    for col in board.columns:
        if col.name == column_name:
            column = col
            break
    
    if column is None:
        raise ValueError(f"列 '{column_name}' 不存在")
    
    # 创建任务
    task = {
        "id": task_id,
        "title": task_title,
        "description": task_description,
        "priority": priority,
        "assignee": assignee,
        "labels": labels or [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    
    column.tasks.append(task)
    logger.info("添加任务: %s -> %s", task_id, column_name)


def move_task(
    board: KanbanBoard,
    task_id: str,
    from_column: str,
    to_column: str,
) -> None:
    """移动任务
    
    参数:
        board: 看板对象
        task_id: 任务 ID
        from_column: 源列名称
        to_column: 目标列名称
    """
    # 查找源列和目标列
    source_col = None
    target_col = None
    
    for col in board.columns:
        if col.name == from_column:
            source_col = col
        elif col.name == to_column:
            target_col = col
    
    if source_col is None:
        raise ValueError(f"源列 '{from_column}' 不存在")
    if target_col is None:
        raise ValueError(f"目标列 '{to_column}' 不存在")
    
    # 查找并移动任务
    task = None
    for t in source_col.tasks:
        if t["id"] == task_id:
            task = t
            break
    
    if task is None:
        raise ValueError(f"任务 '{task_id}' 在列 '{from_column}' 中不存在")
    
    # 移动任务
    source_col.tasks.remove(task)
    task["updated_at"] = datetime.now().isoformat()
    target_col.tasks.append(task)
    
    logger.info("移动任务: %s: %s -> %s", task_id, from_column, to_column)


def get_kanban_stats(board: KanbanBoard) -> Dict[str, Any]:
    """获取看板统计信息
    
    参数:
        board: 看板对象
    
    返回:
        统计信息字典
    """
    total_tasks = 0
    column_stats = {}
    
    for col in board.columns:
        task_count = len(col.tasks)
        total_tasks += task_count
        
        # 按优先级统计
        priority_stats = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for task in col.tasks:
            priority = task.get("priority", "medium")
            priority_stats[priority] = priority_stats.get(priority, 0) + 1
        
        column_stats[col.name] = {
            "title": col.title,
            "task_count": task_count,
            "priority_stats": priority_stats,
        }
    
    return {
        "board_title": board.title,
        "total_tasks": total_tasks,
        "column_count": len(board.columns),
        "column_stats": column_stats,
    }


def render_kanban_html(board: KanbanBoard) -> str:
    """渲染看板为 HTML
    
    参数:
        board: 看板对象
    
    返回:
        HTML 字符串
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{board.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell', 'Open Sans', 'Helvetica Neue', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .board {{
            display: flex;
            gap: 20px;
            overflow-x: auto;
            padding: 20px;
        }}
        .column {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            min-width: 300px;
            max-width: 350px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }}
        .column-header {{
            padding: 16px 20px;
            color: white;
            font-weight: 600;
            font-size: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .column-content {{
            padding: 12px;
            min-height: 200px;
        }}
        .task {{
            background: white;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            cursor: pointer;
            transition: all 0.2s;
        }}
        .task:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        .task-title {{
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }}
        .task-meta {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .label {{
            background: #e0e0e0;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            color: #666;
        }}
        .priority-low {{ border-left: 4px solid #4CAF50; }}
        .priority-medium {{ border-left: 4px solid #FF9800; }}
        .priority-high {{ border-left: 4px solid #FF5722; }}
        .priority-critical {{ border-left: 4px solid #f44336; }}
    </style>
</head>
<body>
    <h1 style="color: white; text-align: center; margin-bottom: 30px;">{board.title}</h1>
    <div class="board">
"""
    
    for column in board.columns:
        html += f"""
        <div class="column">
            <div class="column-header" style="background: {column.color};">
                <span>{column.title}</span>
                <span class="task-count">{len(column.tasks)}</span>
            </div>
            <div class="column-content">
"""
        
        for task in column.tasks:
            priority_class = f"priority-{task.get('priority', 'medium')}"
            labels_html = "".join(f'<span class="label">{label}</span>' for label in task.get('labels', []))
            
            html += f"""
                <div class="task {priority_class}">
                    <div class="task-title">{task.get('title', '')}</div>
                    <div class="task-meta">
                        {labels_html}
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    return html


# 示例用法
if __name__ == "__main__":
    # 创建看板
    board = create_dev_lifecycle_kanban("Hermes 项目")
    
    # 添加任务
    add_task_to_kanban(board, "ideate", "task-1", "需求分析", "分析用户需求", "high", ["需求"])
    add_task_to_kanban(board, "build", "task-2", "代码实现", "实现核心功能", "medium", ["开发"])
    add_task_to_kanban(board, "deliver", "task-3", "测试部署", "部署到生产环境", "low", ["运维"])
    
    # 获取统计
    stats = get_kanban_stats(board)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # 渲染 HTML
    html = render_kanban_html(board)
    with open("/tmp/kanban.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("看板已生成: /tmp/kanban.html")
