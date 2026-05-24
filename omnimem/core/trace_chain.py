"""TraceChain — node_id 全链路溯源。

溯源链路：L3 Persona → L2 Scenario → L1 Atom → L0 Conversation

每个节点携带：
- node_id: 唯一标识
- layer: L0/L1/L2/L3
- parent_ids: 上层来源节点列表
- child_ids: 下层派生节点列表
- ref_path: 底层原文文件路径（L0）

设计原则：
- 任何摘要都可追溯到原始对话
- 链路是确定性的，不存在"不可逆"的摘要
- 支持双向遍历（上钻/下钻）
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)


class TraceChain:
    """全链路溯源 — 参考 TencentDB node_id 机制。

    使用独立 SQLite 文件 (trace_chain.db) 存储节点关系，
    与 meta_store.db 分离，避免事务冲突。
    """

    def __init__(self, data_dir: Path):
        self._chain_db = data_dir / "trace_chain.db"
        self._init_db()

    def _init_db(self) -> None:
        """初始化溯源链数据库。"""
        self._conn = sqlite3.connect(str(self._chain_db), timeout=5, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="trace_nodes",
            create_sql="""
                CREATE TABLE IF NOT EXISTS trace_nodes (
                    node_id TEXT PRIMARY KEY,
                    layer TEXT NOT NULL,
                    parent_ids_json TEXT DEFAULT '[]',
                    child_ids_json TEXT DEFAULT '[]',
                    ref_path TEXT DEFAULT '',
                    metadata_json TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                )
            """,
            migrations=[],
        )
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trace_layer ON trace_nodes(layer)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trace_created ON trace_nodes(created_at)
        """)
        self._conn.commit()

    def record_derivation(
        self,
        parent_node_ids: list[str],
        child_node_id: str,
        child_layer: str,
        ref_path: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """记录派生关系：parent → child。

        同时更新 parent 的 child_ids 列表。

        Args:
            parent_node_ids: 上层来源节点 ID 列表
            child_node_id: 当前节点 ID
            child_layer: 当前节点层级 (L0/L1/L2/L3)
            ref_path: 底层原文文件路径（L0 节点才有）
            metadata: 额外元数据
        """
        now = time.time()

        # 插入 child 节点
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO trace_nodes
                   (node_id, layer, parent_ids_json, child_ids_json, ref_path, metadata_json, created_at)
                   VALUES (?, ?, ?, '[]', ?, ?, ?)""",
                (
                    child_node_id,
                    child_layer,
                    json.dumps(parent_node_ids, ensure_ascii=False),
                    ref_path,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                ),
            )

            # 更新每个 parent 的 child_ids
            for parent_id in parent_node_ids:
                self._conn.execute(
                    """INSERT OR IGNORE INTO trace_nodes
                       (node_id, layer, parent_ids_json, child_ids_json, ref_path, metadata_json, created_at)
                       VALUES (?, 'L0', '[]', '[]', '', '{{}}', ?)""",
                    (parent_id, now),
                )
                row = self._conn.execute(
                    "SELECT child_ids_json FROM trace_nodes WHERE node_id = ?",
                    (parent_id,),
                ).fetchone()
                if row:
                    child_ids = json.loads(row[0] or "[]")
                    if child_node_id not in child_ids:
                        child_ids.append(child_node_id)
                        self._conn.execute(
                            "UPDATE trace_nodes SET child_ids_json = ? WHERE node_id = ?",
                            (json.dumps(child_ids, ensure_ascii=False), parent_id),
                        )

            self._conn.commit()
            logger.warning("TraceChain record_derivation: %s → %s (%s)", parent_node_ids, child_node_id, child_layer)
        except Exception as e:
            logger.warning("TraceChain record_derivation failed: %s", e)
            self._conn.rollback()

    def drill_down(
        self, node_id: str, max_depth: int = 10
    ) -> list[dict[str, Any]]:
        """下钻：从高层节点追溯到低层原文。

        递归遍历 parent_ids，返回完整溯源链。
        防循环：visited set + max_depth 限制。

        Args:
            node_id: 起始节点 ID
            max_depth: 最大递归深度（防止循环引用）

        Returns:
            溯源链节点列表，从高层到低层
        """
        result: list[dict[str, Any]] = []
        visited: set[str] = set()

        def _recurse(nid: str, depth: int) -> None:
            if depth > max_depth or nid in visited:
                return
            visited.add(nid)
            row = self._get_node(nid)
            if not row:
                return
            result.append(row)
            parent_ids = json.loads(row.get("parent_ids_json", "[]"))
            for parent_id in parent_ids:
                _recurse(parent_id, depth + 1)

        _recurse(node_id, 0)
        logger.warning("TraceChain drill_down: node=%s, depth=%d, result_count=%d", node_id, max_depth, len(result))
        return result

    def drill_up(
        self, node_id: str, max_depth: int = 10
    ) -> list[dict[str, Any]]:
        """上钻：从低层节点找到所有引用它的高层摘要。

        递归遍历 child_ids，返回所有上层节点。

        Args:
            node_id: 起始节点 ID
            max_depth: 最大递归深度

        Returns:
            上层节点列表，从低层到高层
        """
        result: list[dict[str, Any]] = []
        visited: set[str] = set()

        def _recurse(nid: str, depth: int) -> None:
            if depth > max_depth or nid in visited:
                return
            visited.add(nid)
            row = self._get_node(nid)
            if not row:
                return
            result.append(row)
            child_ids = json.loads(row.get("child_ids_json", "[]"))
            for child_id in child_ids:
                _recurse(child_id, depth + 1)

        _recurse(node_id, 0)
        logger.warning("TraceChain drill_up: node=%s, depth=%d, result_count=%d", node_id, max_depth, len(result))
        return result

    def get_ref_path(self, node_id: str) -> str | None:
        """获取底层原文文件路径。"""
        row = self._get_node(node_id)
        if row and row.get("ref_path"):
            return row["ref_path"]
        return None

    def recover_full_text(self, node_id: str) -> str | None:
        """按 node_id 恢复完整原文。

        对应 TencentDB 的 node_id 溯源：
        Agent 看着符号图谱推理，需核对细节时 grep node_id 即可恢复原文。
        """
        ref = self.get_ref_path(node_id)
        if ref:
            p = Path(ref)
            if p.exists():
                try:
                    return p.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning("TraceChain recover_full_text read failed: %s", e)
        return None

    def _get_node(self, node_id: str) -> dict[str, Any] | None:
        """从数据库获取节点详情。"""
        try:
            row = self._conn.execute(
                "SELECT node_id, layer, parent_ids_json, child_ids_json, ref_path, metadata_json, created_at "
                "FROM trace_nodes WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "node_id": row[0],
                "layer": row[1],
                "parent_ids_json": row[2],
                "child_ids_json": row[3],
                "ref_path": row[4],
                "metadata_json": row[5],
                "created_at": row[6],
            }
        except Exception as e:
            logger.warning("TraceChain _get_node failed: %s", e)
            return None

    def get_node_count(self) -> int:
        """返回节点总数。"""
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM trace_nodes").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    def close(self) -> None:
        """关闭数据库连接。"""
        try:
            self._conn.close()
        except Exception as e:
            logger.warning("TraceChain close failed: %s", e)
