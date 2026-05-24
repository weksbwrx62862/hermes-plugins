"""GovernanceAuditor — 后台治理巡检器。

P0方案六：长期运行的一致性保障机制。
定期检查并修复以下不一致：
  1. 幽灵索引：index/retriever 中有但 MetaStore 中已不存在的条目
  2. 漏索引：MetaStore 中有但 index/retriever 中缺失的条目
  3. 归档残留：已归档记忆在检索索引中的残留

设计原则：
  - 以 MetaStore 为唯一事实来源（Single Source of Truth）
  - 只读审计优先，修复操作需显式调用 repair() 或 _repair_from_metastore()
  - 利用现有 store/index/retriever 接口，不引入新存储
  - 轻量级实现，避免全量扫描导致的长耗时阻塞
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GovernanceAuditor:
    """治理巡检器：检测并修复 Store/Index/Retriever 之间的不一致。"""

    def __init__(
        self,
        store: Any,
        index: Any,
        retriever: Any,
        forgetting: Any,
    ):
        """初始化巡检器。

        Args:
            store: DrawerClosetStore 实例
            index: ThreeLevelIndex 实例
            retriever: HybridRetriever 实例
            forgetting: ForgettingCurve 实例
        """
        self._store = store
        self._index = index
        self._retriever = retriever
        self._forgetting = forgetting

    def run_full_audit(self, limit: int = 2000) -> dict[str, Any]:
        """执行全量一致性审计。

        ★ 以 MetaStore 为唯一事实来源（Single Source of Truth），
        检测 Index/Retriever 与 MetaStore 之间的不一致。

        Args:
            limit: 最多审计的条目数（防止大数据量时阻塞）

        Returns:
            审计结果字典，包含各类不一致条目列表
        """
        ghost_in_index: list[str] = []
        missing_in_index: list[str] = []
        ghost_in_retriever: list[str] = []
        missing_in_retriever: list[str] = []

        # ★ 1. 以 MetaStore 为基准获取有效记忆 ID
        meta_entries = self._store._meta_store.get_all(limit=limit)
        meta_ids: set[str] = {
            e.get("memory_id", "") for e in meta_entries if e.get("memory_id", "")
        }

        # 2. 获取 index 中的条目
        index_entries = self._index.search_all_for_retrieval(limit=limit)
        index_ids: set[str] = {
            e.get("memory_id", "") for e in index_entries if e.get("memory_id", "")
        }

        # 检测幽灵索引：index 有但 MetaStore 无
        for mid in index_ids:
            if mid not in meta_ids:
                ghost_in_index.append(mid)

        # 检测漏索引：MetaStore 有但 index 无
        for mid in meta_ids:
            if mid not in index_ids:
                missing_in_index.append(mid)

        # 3. 检测已归档但在 index 中残留的条目
        try:
            archived = self._forgetting.get_archived_ids(limit=limit)
            for mid in archived:
                if mid in index_ids:
                    ghost_in_index.append(mid)
                if self._retriever.bm25_document_count > 0:
                    entry = self._store.get(mid)
                    if entry is None and mid in index_ids:
                        ghost_in_retriever.append(mid)
        except Exception as e:
            logger.warning("Audit archived check failed: %s", e)

        # 4. ChromaDB 条目数一致性检查（以 MetaStore 为基准）
        chroma_count: int = 0
        chroma_degraded: bool = False
        try:
            chroma_count = self._retriever._vector.count()
            meta_count = len(meta_ids)
            threshold = max(meta_count // 20, 5)
            if abs(meta_count - chroma_count) > threshold:
                chroma_degraded = True
                logger.warning(
                    "Audit: ChromaDB 条目数偏差超阈值 — meta=%d, chroma=%d, 阈值=%d",
                    meta_count,
                    chroma_count,
                    threshold,
                )
        except Exception as e:
            logger.warning("Audit ChromaDB count check failed: %s", e)

        total_issues = (
            len(ghost_in_index)
            + len(missing_in_index)
            + len(ghost_in_retriever)
            + len(missing_in_retriever)
        )

        return {
            "ghost_in_index": list(set(ghost_in_index)),
            "missing_in_index": list(set(missing_in_index)),
            "ghost_in_retriever": list(set(ghost_in_retriever)),
            "missing_in_retriever": list(set(missing_in_retriever)),
            "total_issues": total_issues,
            "scanned_meta": len(meta_ids),
            "scanned_index": len(index_ids),
            "chroma_count": chroma_count,
            "chroma_degraded": chroma_degraded,
        }

    def repair(self, audit_result: dict[str, Any]) -> int:
        """根据审计结果自动修复不一致。

        修复策略：
          - 幽灵索引：从 index 中删除
          - 漏索引：从 store 读取后重新写入 index 和 retriever
          - 归档残留：从 index/retriever 中删除

        Args:
            audit_result: run_full_audit() 的返回值

        Returns:
            成功修复的条目数
        """
        fixed = 0

        # 修复幽灵索引
        for mid in audit_result.get("ghost_in_index", []):
            try:
                if self._index.delete(mid):
                    logger.warning("Auditor: removed ghost index %s", mid)
                    fixed += 1
            except Exception as e:
                logger.warning("Auditor: failed to remove ghost index %s: %s", mid, e)

        # 修复漏索引
        for mid in audit_result.get("missing_in_index", []):
            try:
                entry = self._store.get(mid)
                if not entry:
                    continue
                self._index.add(
                    memory_id=mid,
                    wing=entry.get("wing", ""),
                    hall=entry.get("hall", entry.get("type", "fact")),
                    room=entry.get("room", ""),
                    content=entry.get("content", ""),
                    summary=entry.get("summary", ""),
                    type=entry.get("type", "fact"),
                    confidence=entry.get("confidence", 3),
                    privacy=entry.get("privacy", "personal"),
                    scope=entry.get("privacy", "personal"),
                    stored_at=entry.get("stored_at", ""),
                    provenance="",
                )
                self._retriever.add(
                    entry.get("content", ""),
                    memory_id=mid,
                    metadata={
                        "memory_id": mid,
                        "type": entry.get("type", "fact"),
                        "confidence": entry.get("confidence", 3),
                        "scope": entry.get("privacy", "personal"),
                        "privacy": entry.get("privacy", "personal"),
                        "wing": entry.get("wing", ""),
                        "room": entry.get("room", ""),
                        "stored_at": entry.get("stored_at", ""),
                    },
                )
                logger.warning("Auditor: re-indexed missing entry %s", mid)
                fixed += 1
            except Exception as e:
                logger.warning("Auditor: failed to re-index %s: %s", mid, e)

        return fixed

    def quick_health_check(self) -> dict[str, Any]:
        """快速健康检查：以 MetaStore 为基准对比计数，不扫描全量 ID。

        Returns:
            健康状态摘要
        """
        meta_count = self._store._meta_store.count()
        index_count = len(self._index.search_all_for_retrieval(limit=5000))
        retriever_count = self._retriever.bm25_document_count

        return {
            "meta_count": meta_count,
            "index_count": index_count,
            "retriever_bm25_count": retriever_count,
            "healthy": abs(meta_count - index_count) <= max(meta_count // 20, 5),
        }

    def _repair_from_metastore(self, limit: int = 2000) -> int:
        """以 MetaStore 为唯一事实来源，修复 Index/Retriever 中的不一致数据。

        遍历 MetaStore 中的所有 memory_id，检查 Index 和 Retriever 中
        是否存在对应条目，缺失则补充。

        Args:
            limit: 最多处理的条目数

        Returns:
            成功修复的条目数
        """
        fixed = 0

        # ★ 以 MetaStore 为基准获取所有记忆 ID
        meta_entries = self._store._meta_store.get_all(limit=limit)
        meta_ids: set[str] = {
            e.get("memory_id", "") for e in meta_entries if e.get("memory_id", "")
        }

        # 获取 Index 中已有的 ID
        index_entries = self._index.search_all_for_retrieval(limit=limit)
        index_ids: set[str] = {
            e.get("memory_id", "") for e in index_entries if e.get("memory_id", "")
        }

        # 遍历 MetaStore，检查 Index 和 Retriever 中的缺失
        for mid in meta_ids:
            entry = self._store.get(mid)
            if not entry:
                continue

            # 检查 Index 缺失
            if mid not in index_ids:
                try:
                    self._index.add(
                        memory_id=mid,
                        wing=entry.get("wing", ""),
                        hall=entry.get("hall", entry.get("type", "fact")),
                        room=entry.get("room", ""),
                        content=entry.get("content", ""),
                        summary=entry.get("summary", ""),
                        type=entry.get("type", "fact"),
                        confidence=entry.get("confidence", 3),
                        privacy=entry.get("privacy", "personal"),
                        scope=entry.get("privacy", "personal"),
                        stored_at=entry.get("stored_at", ""),
                        provenance="",
                    )
                    logger.info("Auditor._repair_from_metastore: 补充 Index 条目 %s", mid)
                    fixed += 1
                except Exception as e:
                    logger.warning("Auditor._repair_from_metastore: 补充 Index 失败 %s: %s", mid, e)

            # 检查 Retriever 缺失（通过 add 补充）
            try:
                self._retriever.add(
                    entry.get("content", ""),
                    memory_id=mid,
                    metadata={
                        "memory_id": mid,
                        "type": entry.get("type", "fact"),
                        "confidence": entry.get("confidence", 3),
                        "scope": entry.get("privacy", "personal"),
                        "privacy": entry.get("privacy", "personal"),
                        "wing": entry.get("wing", ""),
                        "room": entry.get("room", ""),
                        "stored_at": entry.get("stored_at", ""),
                    },
                )
            except Exception as e:
                logger.warning("Auditor._repair_from_metastore: 补充 Retriever 失败 %s: %s", mid, e)

        # 清理 Index 中 MetaStore 已不存在的幽灵条目
        ghost_ids = index_ids - meta_ids
        for mid in ghost_ids:
            try:
                if self._index.delete(mid):
                    logger.info("Auditor._repair_from_metastore: 清理幽灵 Index 条目 %s", mid)
                    fixed += 1
            except Exception as e:
                logger.warning("Auditor._repair_from_metastore: 清理幽灵 Index 失败 %s: %s", mid, e)

        if fixed > 0:
            logger.info("Auditor._repair_from_metastore: 共修复 %d 条不一致", fixed)
        return fixed
