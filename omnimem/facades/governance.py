"""GovernanceFacade — 治理引擎横切面。

封装: ConflictResolver, TemporalDecay, ForgettingCurve, PrivacyManager,
      ProvenanceTracker, SyncEngine, VectorClock, GovernanceAuditor,
      TemporalKnowledgeGraph
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omnimem.governance.audit_log import AuditLogger
from omnimem.governance.auditor import GovernanceAuditor
from omnimem.governance.conflict import ConflictResolver
from omnimem.governance.decay import TemporalDecay
from omnimem.governance.forgetting import ForgettingCurve
from omnimem.governance.kms import KMSManager
from omnimem.governance.privacy import PrivacyManager
from omnimem.governance.provenance import ProvenanceTracker
from omnimem.governance.rbac import RBACManager
from omnimem.governance.sync import SyncConfig, SyncEngine
from omnimem.governance.temporal_kg import TemporalKnowledgeGraph
from omnimem.governance.vector_clock import VectorClock

logger = logging.getLogger(__name__)


class GovernanceFacade:
    """治理门面：冲突仲裁 + 时间衰减 + 遗忘 + 隐私 + 溯源 + 同步。"""

    def __init__(
        self,
        data_dir: Path,
        config: Any,
        session_id: str,
        storage_facade: Any,
        retriever: Any,
    ):
        gov_dir = data_dir / "governance"

        self._conflict_resolver = ConflictResolver(
            strategy=config.get("conflict_strategy", "latest")
        )
        self._temporal_decay = TemporalDecay()
        self._forgetting = ForgettingCurve(gov_dir, config)
        self._kms = KMSManager(gov_dir)
        self._privacy = PrivacyManager(
            default_level=config.get("default_privacy", "personal"),
            session_id=session_id,
            kms_manager=self._kms,
        )
        self._privacy.bind_store(storage_facade.store)
        storage_facade.store.bind_privacy_manager(self._privacy)

        self._provenance = ProvenanceTracker(data_dir=gov_dir)

        # 同步引擎
        self._sync_engine = SyncEngine(
            data_dir,
            SyncConfig(
                mode=config.get("sync_mode", "none"),
                instance_name=f"omnimem-{session_id[:8]}",
                sync_interval=config.get("sync_interval", 30),
                conflict_resolution=config.get("sync_conflict_resolution", "latest_wins"),
            ),
        )

        # 向量时钟 — 仅在启用同步时初始化，单机模式短路以减少无谓开销
        _sync_mode = config.get("sync_mode", "none")
        self._vc_db_path = gov_dir / "vector_clock.db"
        self._vc_json_path = gov_dir / "vector_clock.json"
        if _sync_mode != "none":
            if self._vc_db_path.exists():
                self._vector_clock = VectorClock.load_from_sqlite(self._vc_db_path)
            elif self._vc_json_path.exists():
                self._vector_clock = VectorClock.load(self._vc_json_path)
            else:
                self._vector_clock = VectorClock()
        else:
            self._vector_clock = None  # 单机模式短路
        self._instance_id = self._sync_engine._config.instance_id

        # 审计器
        self._auditor = GovernanceAuditor(
            store=storage_facade.store,
            index=storage_facade.index,
            retriever=retriever,
            forgetting=self._forgetting,
        )

        # 操作审计日志
        self._audit_logger = AuditLogger(gov_dir, max_rows=config.get("audit_log_max_rows", 100000))

        # RBAC 访问控制
        self._rbac = RBACManager(gov_dir)

        # 时序知识图谱
        self._temporal_kg = TemporalKnowledgeGraph(gov_dir, config)

    @property
    def conflict_resolver(self) -> ConflictResolver:
        return self._conflict_resolver

    @property
    def temporal_decay(self) -> TemporalDecay:
        return self._temporal_decay

    @property
    def forgetting(self) -> ForgettingCurve:
        return self._forgetting

    @property
    def privacy(self) -> PrivacyManager:
        return self._privacy

    @property
    def provenance(self) -> ProvenanceTracker:
        return self._provenance

    @property
    def sync_engine(self) -> SyncEngine:
        return self._sync_engine

    @property
    def auditor(self) -> GovernanceAuditor:
        return self._auditor

    @property
    def audit_logger(self) -> AuditLogger:
        return self._audit_logger

    @property
    def rbac(self) -> RBACManager:
        return self._rbac

    @property
    def kms(self) -> KMSManager:
        return self._kms

    @property
    def temporal_kg(self) -> TemporalKnowledgeGraph:
        return self._temporal_kg

    @property
    def vector_clock(self) -> "VectorClock | None":
        return self._vector_clock

    @property
    def instance_id(self) -> str:
        return self._instance_id

    def close(self) -> None:
        """关闭治理资源。"""
        # 持久化向量时钟状态（单机模式下 _vector_clock 为 None，跳过持久化）
        if self._vector_clock is not None:
            try:
                self._vector_clock.save_to_sqlite(self._vc_db_path)
            except Exception as e:
                logger.warning("VectorClock save on close failed: %s", e)
        self.forgetting.close()
        self.provenance.close()
        self.sync_engine.close()
        self._audit_logger.close()
        if self._temporal_kg:
            self._temporal_kg.close()

    def trust_feedback(self, memory_id: str, useful: bool) -> float:
        """信任评分反馈：Agent 使用记忆后反馈有用/无用。

        参考 Holographic: +0.05 (有用) / -0.10 (无用)。
        信任评分范围 0.0-1.0。

        Args:
            memory_id: 记忆 ID
            useful: True 表示有用，False 表示无用

        Returns:
            更新后的信任评分
        """
        entry = self._store.get(memory_id)
        if not entry:
            return 0.5
        current_trust = entry.get("trust", 0.5)
        delta = 0.05 if useful else -0.10
        new_trust = max(0.0, min(1.0, current_trust + delta))
        try:
            self._store.update(memory_id, {"trust": new_trust})
            self._audit_logger.record(
                action=("trust_helpful" if useful else "trust_unhelpful"),
                target=memory_id,
                detail=f"trust: {current_trust:.2f} -> {new_trust:.2f}",
            )
        except Exception as e:
            logger.warning("Trust feedback update failed for %s: %s", memory_id, e)
        return new_trust
