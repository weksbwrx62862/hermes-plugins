"""VectorClock — 向量时钟实现。

分布式多实例同步协议的基础组件：
  1. 为每条记忆附加逻辑时钟，判断因果关系
  2. 检测并发冲突（不可比较的向量时钟）
  3. 合并向量时钟（取各节点最大值）

与 Lamport 时间戳相比，向量时钟可以检测并发事件，
适用于 OmniMem 多实例共享记忆库时的冲突检测。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VectorClock:
    """向量时钟。

    每个实例维护一个独立的计数器，形成多维时钟向量。
    示例：
      vc_A = {"instance-1": 3, "instance-2": 1}
      vc_B = {"instance-1": 2, "instance-2": 2}
      → A 和 B 并发冲突（不可比较）
    """

    def __init__(self, clock: dict[str, int] | None = None):
        """初始化向量时钟。

        Args:
            clock: 初始时钟字典，如 {"node-a": 1, "node-b": 3}
        """
        self._clock: dict[str, int] = dict(clock) if clock else {}

    def increment(self, node_id: str) -> VectorClock:
        """递增指定节点的时钟。

        Returns:
            self（支持链式调用）
        """
        self._clock[node_id] = self._clock.get(node_id, 0) + 1
        return self

    def compare(self, other: VectorClock) -> int:
        """与另一个向量时钟比较。

        Returns:
            -1: self 发生在 other 之前（self < other）
             0: 并发冲突（不可比较）
             1: self 发生在 other 之后（self > other）
        """
        all_nodes = set(self._clock.keys()) | set(other._clock.keys())
        lt = False  # self < other
        gt = False  # self > other

        for node in all_nodes:
            a = self._clock.get(node, 0)
            b = other._clock.get(node, 0)
            if a < b:
                lt = True
            elif a > b:
                gt = True

        if lt and gt:
            return 0  # 并发冲突
        if lt:
            return -1
        if gt:
            return 1
        return 0  # 完全相等

    def merge(self, other: VectorClock) -> VectorClock:
        """合并两个向量时钟（取各节点最大值）。

        Returns:
            新的 VectorClock 实例
        """
        merged = {}
        for node in set(self._clock.keys()) | set(other._clock.keys()):
            merged[node] = max(self._clock.get(node, 0), other._clock.get(node, 0))
        return VectorClock(merged)

    def to_dict(self) -> dict[str, int]:
        """序列化为字典。"""
        return dict(self._clock)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VectorClock:
        """从字典反序列化。"""
        return cls({k: int(v) for k, v in d.items() if isinstance(v, (int, float))})

    @classmethod
    def from_json(cls, s: str) -> VectorClock:
        """从 JSON 字符串反序列化。"""
        try:
            return cls.from_dict(json.loads(s))
        except Exception:
            return cls()

    def to_json(self) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(self._clock)

    def save(self, path: Path) -> bool:
        """持久化向量时钟状态到文件。

        Args:
            path: 保存路径

        Returns:
            是否成功
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._clock, f, ensure_ascii=False)
            return True
        except Exception as e:
            logger.warning("VectorClock save failed: %s", e)
            return False

    @classmethod
    def load(cls, path: Path) -> VectorClock:
        """从文件加载向量时钟状态。

        Args:
            path: 保存路径

        Returns:
            VectorClock 实例（加载失败时返回空时钟）
        """
        try:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return cls({k: int(v) for k, v in data.items()})
        except Exception as e:
            logger.warning("VectorClock load failed: %s", e)
        return cls()

    @classmethod
    def recover_from_entries(
        cls,
        node_id: str,
        entries: list[dict[str, Any]],
    ) -> VectorClock:
        """从已有记忆条目中恢复所有实例的计数器。

        用于进程重启后恢复因果一致性。遍历所有含 VC 的记录，
        取各节点最大值作为当前时钟状态。

        Args:
            node_id: 本实例 ID（保留参数，恢复所有节点计数器）
            entries: 记忆条目列表（含 vc 字段）

        Args:
            node_id: 本实例的节点 ID
            entries: 所有记忆条目列表（含 vc 字段）

        Returns:
            恢复后的 VectorClock，各节点计数器为所有记录中的最大值
        """
        _ = node_id  # 保留参数用于未来按节点过滤恢复
        recovered: dict[str, int] = {}
        for entry in entries:
            vc_raw = entry.get("vc", {})
            if isinstance(vc_raw, str):
                try:
                    vc_raw = json.loads(vc_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(vc_raw, dict):
                continue
            for nid, count in vc_raw.items():
                try:
                    c = int(count)
                    recovered[nid] = max(recovered.get(nid, 0), c)
                except (ValueError, TypeError):
                    continue
        if recovered:
            logger.info(
                "VectorClock recovered from %d entries: max counters=%s",
                len(entries),
                {k: v for k, v in recovered.items() if v > 0},
            )
        return cls(recovered)

    def __repr__(self) -> str:
        return f"VectorClock({self._clock})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self._clock == other._clock

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._clock.items())))


def detect_conflict(
    local_vc: VectorClock,
    remote_vc: VectorClock,
) -> bool:
    """检测两个向量时钟是否表示并发冲突。

    Returns:
        True 表示存在并发冲突（需要合并解决）
    """
    return local_vc.compare(remote_vc) == 0 and local_vc != remote_vc


def merge_records(
    local: dict[str, Any],
    remote: dict[str, Any],
    memory_type: str = "fact",
) -> dict[str, Any]:
    """基于记忆类型执行结构化合并。

    合并策略：
      - preference: 合并新旧值（如"喜欢猫" + "喜欢狗" → "喜欢猫、狗"）
      - correction: 使用 VC 比较决定，远程更新则覆盖
      - fact: 使用 VC 比较决定，远程更新则覆盖

    Args:
        local: 本地记忆记录
        remote: 远程记忆记录
        memory_type: 记忆类型，决定合并策略

    Returns:
        合并后的记录
    """
    merged = dict(local)
    merged["vc"] = (
        VectorClock.from_dict(local.get("vc", {}))
        .merge(VectorClock.from_dict(remote.get("vc", {})))
        .to_dict()
    )

    local_vc = VectorClock.from_dict(local.get("vc", {}))
    remote_vc = VectorClock.from_dict(remote.get("vc", {}))

    if memory_type == "preference":
        # 偏好合并：保留两者内容，去重后拼接
        local_content = local.get("content", "")
        remote_content = remote.get("content", "")
        if remote_content and remote_content not in local_content:
            merged["content"] = f"{local_content}；{remote_content}"
    elif memory_type == "correction" or memory_type == "fact":
        # 使用 VC 比较决定是否覆盖
        cmp = remote_vc.compare(local_vc)
        if cmp == 1:
            # 远程版本更新 → 覆盖
            merged["content"] = remote.get("content", local.get("content", ""))
        elif cmp == 0:
            # 并发冲突 → 也接受远程（concurrent merge）
            merged["content"] = remote.get("content", local.get("content", ""))

    return merged
