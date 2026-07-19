from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

START = "__start__"
END = "__end__"


class ModeNode:
    """图节点，封装一个可执行的处理函数"""

    def __init__(self, name: str, handler: Callable):
        self.name = name
        self.handler = handler


@dataclass
class GraphState:
    """图执行状态，在节点间传递"""

    task: str = ""
    context: Optional[str] = None
    result: str = ""
    success: bool = False
    token_usage: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    _internal: Dict[str, Any] = field(default_factory=dict)


class CompiledModeGraph:
    """编译后的可执行图"""

    def __init__(
        self,
        mode_name: str,
        nodes: Dict[str, ModeNode],
        edges: Dict[str, List[Tuple[Optional[Callable], str]]],
        entry_node: str,
    ):
        self.mode_name = mode_name
        self.nodes = nodes
        self.edges = edges
        self.entry_node = entry_node
        self._logger = logging.getLogger(f"ama.graph.{mode_name}")

    def execute(
        self, ctx, task: str, context: Optional[str] = None, **kwargs
    ) -> Dict:
        """执行图，从 entry_node 开始沿边遍历直到 END"""
        state = GraphState(task=task, context=context)
        current_node = self.entry_node
        max_steps = 50
        step = 0

        while current_node != END and step < max_steps:
            if current_node not in self.nodes:
                self._logger.warning("图执行遇到未注册节点: %s，终止", current_node)
                break

            node = self.nodes[current_node]
            try:
                update = node.handler(ctx, state, **kwargs)
                if isinstance(update, dict):
                    for k, v in update.items():
                        if hasattr(state, k):
                            setattr(state, k, v)
                        else:
                            state._internal[k] = v
            except Exception as e:
                self._logger.error("节点 %s 执行失败: %s", current_node, e)
                state.success = False
                state._internal["error"] = str(e)
                state._internal["failed_node"] = current_node
                break

            step += 1
            current_node = self._route(current_node, state)

        return {
            "result": state.result,
            "success": state.success,
            "token_usage": state.token_usage,
            "metadata": state.metadata,
        }

    def _route(self, from_node: str, state: GraphState) -> str:
        """根据边定义路由到下一个节点"""
        if from_node not in self.edges:
            return END

        for condition_fn, target_node in self.edges[from_node]:
            if condition_fn is None:
                return target_node
            if condition_fn(state):
                return target_node

        return END


class ModeGraph:
    """声明式状态图定义器"""

    def __init__(self, mode_name: str):
        self.mode_name = mode_name
        self._nodes: Dict[str, ModeNode] = {}
        self._edges: Dict[str, List[Tuple[Optional[Callable], str]]] = {}
        self._entry_node: Optional[str] = None

    def add_node(self, name: str, handler: Callable) -> "ModeGraph":
        """添加节点"""
        self._nodes[name] = ModeNode(name, handler)
        if self._entry_node is None:
            self._entry_node = name
        return self

    def add_edge(self, from_node: str, to_node: str) -> "ModeGraph":
        """添加无条件边"""
        if from_node not in self._edges:
            self._edges[from_node] = []
        self._edges[from_node].append((None, to_node))
        return self

    def add_conditional_edges(
        self, from_node: str, condition: Callable, mapping: Dict[str, str]
    ) -> "ModeGraph":
        """添加条件边"""
        if from_node not in self._edges:
            self._edges[from_node] = []

        for cond_key, target_node in mapping.items():

            def make_checker(key):
                def checker(state):
                    result = condition(state)
                    return result == key

                return checker

            self._edges[from_node].append((make_checker(cond_key), target_node))

        return self

    def set_entry_point(self, node_name: str) -> "ModeGraph":
        """设置入口节点"""
        self._entry_node = node_name
        return self

    def compile(self) -> CompiledModeGraph:
        """编译为可执行图"""
        if not self._entry_node:
            raise ValueError(f"图 {self.mode_name} 未设置入口节点")
        return CompiledModeGraph(
            mode_name=self.mode_name,
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            entry_node=self._entry_node,
        )
