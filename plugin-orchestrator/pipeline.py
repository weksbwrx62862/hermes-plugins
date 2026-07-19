"""
Pipeline — 跨插件数据依赖管道（GitHub Actions 风格）

允许插件声明 "我需要 X 数据，由 Y 插件生产"。
编排器按依赖拓扑排序执行钩子回调，确保生产者先于消费者运行。

两种声明语法（等价的）：

  # 原始语法
  PipelineManifest("cache_opt", produces=["cache_hit"], consumes=["model_selection"])

  # GitHub Actions 风格（推荐）
  PipelineManifest("cache_opt", produces=["cache_hit"], needs=["model_selection"])

在 plugin.yaml 中声明（YAML）:

  pipeline:
    produces:
      - model_selection
    needs:
      - task_complexity

在钩子回调中:
  from plugins.plugin_orchestrator.context import get_context
  ctx = get_context(session_id)
  ctx.shared_set("model_selection", "deepseek-v4-pro")  # 生产
  quality = ctx.shared_get("task_complexity")             # 消费
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class PipelineManifest:
    """单个插件的管道声明。

    支持两种构造函数参数风格:
      PipelineManifest("alpha", produces=["a"], consumes=["b"])  # 原始
      PipelineManifest("alpha", produces=["a"], needs=["b"])     # GitHub Actions

    也支持 from_yaml() 从 dict 加载:
      PipelineManifest.from_yaml("plugin_name", {
          "produces": ["a"],
          "needs": ["b"],
      })
    """

    def __init__(
        self,
        plugin_name: str,
        produces: List[str] = None,
        consumes: List[str] = None,
        needs: List[str] = None,  # GitHub Actions 风格别名
    ):
        self.plugin_name = plugin_name
        self.produces: Set[str] = set(produces or [])
        # 'needs' 是 'consumes' 的别名
        combined_consumes = set(consumes or [])
        if needs:
            combined_consumes |= set(needs)
        self.consumes: Set[str] = combined_consumes

    @classmethod
    def from_yaml(cls, plugin_name: str, config: dict) -> "PipelineManifest":
        """从 YAML/dict 配置加载管道声明。

        支持结构:
          {"produces": ["a"], "consumes": ["b"]}
          {"produces": ["a"], "needs": ["b"]}
          {"uses": ["a"]}  # produces=uses, consumes=empty
          空字典 → 空声明
        """
        produces = config.get("produces", [])
        consumes = config.get("consumes", [])
        needs = config.get("needs", [])
        uses = config.get("uses", [])  # 仅生产（无消费）

        # 'uses' 是 'produces' 的别名
        all_produces = list(set(produces) | set(uses))
        return cls(
            plugin_name=plugin_name,
            produces=all_produces or None,
            consumes=consumes or None,
            needs=needs or None,
        )

    def __repr__(self):
        parts = []
        if self.produces:
            parts.append(f"+{self.produces}")
        if self.consumes:
            parts.append(f"needs={self.consumes}")
        return f"PipelineManifest({self.plugin_name}, {' '.join(parts)})"


class PipelineGraph:
    """管道依赖图。

    维护所有插件的数据生产和消费关系。
    为每个钩子点计算最优的插件执行顺序。
    """

    def __init__(self):
        self._manifests: Dict[str, PipelineManifest] = {}  # plugin_name → manifest
        self._producers: Dict[str, str] = {}  # data_key → plugin_name

    def register(
        self,
        plugin_name: str,
        produces: List[str] = None,
        consumes: List[str] = None,
        needs: List[str] = None,  # GitHub Actions 风格别名
    ) -> None:
        """注册一个插件的管道声明。

        支持两种语法（二选一）：
          produces/consumes — 原始语义
          produces/needs    — GitHub Actions 风格（推荐）

        示例:
          pg.register("model_router", produces=["model_selection"])
          pg.register("cache_opt", needs=["model_selection"])
        """
        manifest = PipelineManifest(plugin_name, produces, consumes, needs=needs)
        self._manifests[plugin_name] = manifest

        for key in manifest.produces:
            if key in self._producers and self._producers[key] != plugin_name:
                logger.warning(
                    "Pipeline: data_key '%s' produced by both '%s' and '%s'. Last registration wins.",
                    key, self._producers[key], plugin_name,
                )
            self._producers[key] = plugin_name

        logger.debug(
            "Pipeline registered: %s [+%s needs=%s]",
            plugin_name,
            manifest.produces,
            manifest.consumes,
        )

    def register_from_yaml(self, plugin_name: str, config: dict) -> None:
        """从 YAML/dict 配置注册管道声明。"""
        manifest = PipelineManifest.from_yaml(plugin_name, config)
        self._manifests[plugin_name] = manifest
        for key in manifest.produces:
            if key in self._producers and self._producers[key] != plugin_name:
                logger.warning(
                    "Pipeline: data_key '%s' produced by both '%s' and '%s'. Last registration wins.",
                    key, self._producers[key], plugin_name,
                )
            self._producers[key] = plugin_name
        logger.debug(
            "Pipeline registered (YAML): %s [+%s needs=%s]",
            plugin_name,
            manifest.produces,
            manifest.consumes,
        )

    def unregister(self, plugin_name: str) -> None:
        """注销一个插件。"""
        manifest = self._manifests.pop(plugin_name, None)
        if manifest:
            for key in manifest.produces:
                if self._producers.get(key) == plugin_name:
                    del self._producers[key]

    def get_producer(self, data_key: str) -> Optional[str]:
        """获取生产指定数据的插件名。"""
        return self._producers.get(data_key)

    def get_manifest(self, plugin_name: str) -> Optional[PipelineManifest]:
        return self._manifests.get(plugin_name)

    def topological_sort(
        self,
        hook_callbacks: List[Tuple[str, Callable]],
    ) -> List[Tuple[int, str, Callable]]:
        """按管道依赖对钩子回调进行拓扑排序。

        规则:
          1. 若 plugin-A.produces X 且 plugin-B.needs X（consumes），则 A 在 B 之前执行
          2. 若无可信依赖，保持原注册顺序
          3. 循环依赖: warn 并保持原顺序（不破坏系统）

        返回: [(priority, name, callback), ...]，按依赖顺序
        """
        if not hook_callbacks:
            return []

        plugin_names = [name for name, _ in hook_callbacks]
        default_order = {name: i for i, name in enumerate(plugin_names)}

        # 构建有向图
        graph: Dict[str, Set[str]] = defaultdict(set)  # plugin → 依赖它的插件集合
        in_degree: Dict[str, int] = defaultdict(int)
        added_edges: Set[tuple] = set()  # 跟踪已添加的 (producer, consumer) 边

        for name in plugin_names:
            manifest = self._manifests.get(name)
            if not manifest:
                continue
            for consumed_key in manifest.consumes:
                producer = self._producers.get(consumed_key)
                if producer and producer != name and producer in default_order:
                    edge = (producer, name)
                    if edge not in added_edges:
                        added_edges.add(edge)
                        graph[producer].add(name)
                        in_degree[name] += 1

        # Kahn 拓扑排序
        queue = [name for name in plugin_names if in_degree.get(name, 0) == 0]
        sorted_names: List[str] = []
        while queue:
            current = queue.pop(0)
            sorted_names.append(current)
            for dependent in graph.get(current, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # 检查循环依赖：未排序的插件被追加在末尾（保持原始注册顺序）
        unsorted = [name for name in plugin_names if name not in sorted_names]
        if unsorted:
            logger.warning(
                "Pipeline cycle detected involving %s. Falling back to registration order.",
                unsorted,
            )
            sorted_names.extend(unsorted)

        # 按新的顺序组装结果，保留原始 priority 信息
        name_to_cb = {name: cb for name, cb in hook_callbacks}
        # 生成 (priority=顺序索引, name, cb) 元组，让 invoke_hook 保持新顺序
        result = []
        for i, name in enumerate(sorted_names):
            if name in name_to_cb:
                result.append((i, name, name_to_cb[name]))
        return result

    def list_deps(self) -> Dict[str, dict]:
        """列出所有注册的依赖关系。"""
        result = {}
        for name, manifest in self._manifests.items():
            deps = {}
            for key in manifest.consumes:
                producer = self._producers.get(key, "(unknown)")
                deps[key] = producer
            result[name] = {
                "produces": list(manifest.produces),
                "consumes": deps,
            }
        return result

    def to_yaml_config(self) -> Dict[str, dict]:
        """将当前管道图导出为 YAML 兼容的 dict 配置。"""
        result = {}
        for name, manifest in self._manifests.items():
            entry = {}
            if manifest.produces:
                entry["produces"] = list(manifest.produces)
            if manifest.consumes:
                entry["needs"] = list(manifest.consumes)
            if entry:
                result[name] = entry
        return result


# ── 全局单例 ──────────────────────────────────────────────────────────

_pipeline_graph: Optional[PipelineGraph] = None


def get_pipeline_graph() -> PipelineGraph:
    global _pipeline_graph
    if _pipeline_graph is None:
        _pipeline_graph = PipelineGraph()
    return _pipeline_graph
