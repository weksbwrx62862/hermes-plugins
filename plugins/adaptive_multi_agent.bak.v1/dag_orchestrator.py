"""
DAG 任务编排器 — 支持有向无环图任务依赖

优势：
  - 并行执行无依赖任务
  - 自动拓扑排序
  - 依赖关系可视化
  - 故障隔离
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DAGNode:
    """DAG 节点"""
    name: str
    handler: Callable
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DAGResult:
    """DAG 执行结果"""
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)


class DAGOrchestrator:
    """DAG 任务编排器
    
    支持有向无环图的任务依赖管理，自动拓扑排序和并行执行。
    
    典型用法：
        dag = DAGOrchestrator()
        dag.add_node("research", research_handler)
        dag.add_node("design", design_handler, dependencies=["research"])
        dag.add_node("implement", implement_handler, dependencies=["design"])
        dag.add_node("test", test_handler, dependencies=["implement"])
        
        result = dag.execute(ctx, task="开发新功能")
    """
    
    def __init__(self, max_workers: int = 4):
        """
        参数:
            max_workers: 最大并行执行线程数
        """
        self._nodes: Dict[str, DAGNode] = {}
        self._max_workers = max_workers
    
    def add_node(
        self,
        name: str,
        handler: Callable,
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """添加节点到 DAG
        
        参数:
            name: 节点名称（必须唯一）
            handler: 执行函数 (ctx, **kwargs) -> result
            dependencies: 依赖的节点名称列表
            metadata: 节点元数据
        """
        if name in self._nodes:
            raise ValueError(f"节点 '{name}' 已存在")
        
        # 验证依赖是否存在（允许后添加的节点）
        dependencies = dependencies or []
        
        self._nodes[name] = DAGNode(
            name=name,
            handler=handler,
            dependencies=dependencies,
            metadata=metadata or {},
        )
        
        logger.debug("添加节点: %s, 依赖: %s", name, dependencies)
    
    def _validate(self) -> None:
        """验证 DAG 结构（无环、依赖存在）"""
        # 检查所有依赖是否存在
        for node in self._nodes.values():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    raise ValueError(f"节点 '{node.name}' 依赖的节点 '{dep}' 不存在")
        
        # 检查是否有环（使用 Kahn 算法）
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        for node in self._nodes.values():
            in_degree[node.name] = len(node.dependencies)
            for dep in node.dependencies:
                graph[dep].append(node.name)
        
        # BFS 检测环
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        visited = 0
        
        while queue:
            current = queue.popleft()
            visited += 1
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if visited != len(self._nodes):
            raise ValueError("DAG 中存在环")
        
        logger.info("DAG 验证通过: %d 个节点", len(self._nodes))
    
    def _topological_sort(self) -> List[List[str]]:
        """拓扑排序，返回并行组
        
        返回:
            每组内的节点可以并行执行，组间有依赖关系
        """
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        for node in self._nodes.values():
            in_degree[node.name] = len(node.dependencies)
            for dep in node.dependencies:
                graph[dep].append(node.name)
        
        # BFS 拓扑排序，按层级分组
        groups = []
        current_level = [name for name, degree in in_degree.items() if degree == 0]
        
        while current_level:
            groups.append(current_level)
            next_level = []
            
            for current in current_level:
                for neighbor in graph[current]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_level.append(neighbor)
            
            current_level = next_level
        
        return groups
    
    def execute(
        self,
        ctx,
        task: str,
        context: Optional[str] = None,
        **kwargs,
    ) -> DAGResult:
        """执行 DAG
        
        参数:
            ctx: 上下文对象
            task: 任务描述
            context: 额外上下文
            **kwargs: 其他参数
        
        返回:
            DAGResult 执行结果
        """
        logger.info("开始执行 DAG | task=%s | nodes=%d", task[:50], len(self._nodes))
        
        # 验证 DAG 结构
        try:
            self._validate()
        except ValueError as e:
            logger.error("DAG 验证失败: %s", e)
            return DAGResult(success=False, errors={"validation": str(e)})
        
        # 拓扑排序
        parallel_groups = self._topological_sort()
        logger.info("DAG 拓扑排序完成: %d 个并行组", len(parallel_groups))
        
        # 执行结果存储
        results = {}
        errors = {}
        execution_order = []
        
        # 按组执行
        for group_idx, group in enumerate(parallel_groups):
            logger.info("执行第 %d 组: %s", group_idx + 1, group)
            
            if len(group) == 1:
                # 单节点，直接执行
                node_name = group[0]
                try:
                    result = self._execute_node(ctx, node_name, task, context, results, **kwargs)
                    results[node_name] = result
                    execution_order.append(node_name)
                    logger.info("节点 %s 执行成功", node_name)
                except Exception as e:
                    errors[node_name] = str(e)
                    logger.error("节点 %s 执行失败: %s", node_name, e)
            else:
                # 多节点，并行执行
                with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                    futures = {}
                    for node_name in group:
                        future = executor.submit(
                            self._execute_node,
                            ctx, node_name, task, context, results, **kwargs
                        )
                        futures[future] = node_name
                    
                    for future in as_completed(futures):
                        node_name = futures[future]
                        try:
                            result = future.result()
                            results[node_name] = result
                            execution_order.append(node_name)
                            logger.info("节点 %s 执行成功", node_name)
                        except Exception as e:
                            errors[node_name] = str(e)
                            logger.error("节点 %s 执行失败: %s", node_name, e)
        
        # 检查是否所有节点都成功
        success = len(errors) == 0
        
        logger.info("DAG 执行完成: success=%s, results=%d, errors=%d",
                    success, len(results), len(errors))
        
        return DAGResult(
            success=success,
            results=results,
            errors=errors,
            execution_order=execution_order,
            parallel_groups=parallel_groups,
        )
    
    def _execute_node(
        self,
        ctx,
        node_name: str,
        task: str,
        context: Optional[str],
        previous_results: Dict[str, Any],
        **kwargs,
    ) -> Any:
        """执行单个节点
        
        参数:
            ctx: 上下文对象
            node_name: 节点名称
            task: 任务描述
            context: 额外上下文
            previous_results: 前序节点的结果
            **kwargs: 其他参数
        
        返回:
            节点执行结果
        """
        node = self._nodes[node_name]
        
        # 准备依赖结果
        dependency_results = {
            dep: previous_results.get(dep)
            for dep in node.dependencies
        }
        
        # 执行节点
        result = node.handler(
            ctx,
            task=task,
            context=context,
            dependency_results=dependency_results,
            **kwargs,
        )
        
        return result


# 示例：任务处理函数
def research_handler(ctx, task: str, context: Optional[str] = None, **kwargs) -> Dict:
    """研究阶段"""
    logger.info("执行研究: %s", task[:50])
    # 这里调用实际的研究逻辑
    return {"research_result": "研究完成", "task": task}


def design_handler(ctx, task: str, context: Optional[str] = None, **kwargs) -> Dict:
    """设计阶段"""
    logger.info("执行设计: %s", task[:50])
    research_result = kwargs.get("dependency_results", {}).get("research", {})
    return {"design_result": "设计完成", "research": research_result}


def implement_handler(ctx, task: str, context: Optional[str] = None, **kwargs) -> Dict:
    """实现阶段"""
    logger.info("执行实现: %s", task[:50])
    design_result = kwargs.get("dependency_results", {}).get("design", {})
    return {"implement_result": "实现完成", "design": design_result}


def test_handler(ctx, task: str, context: Optional[str] = None, **kwargs) -> Dict:
    """测试阶段"""
    logger.info("执行测试: %s", task[:50])
    implement_result = kwargs.get("dependency_results", {}).get("implement", {})
    return {"test_result": "测试完成", "implement": implement_result}
