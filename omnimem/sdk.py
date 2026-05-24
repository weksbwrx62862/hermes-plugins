"""OmniMemSDK — 独立 SDK 模式，直接初始化子组件，不依赖 Hermes MemoryProvider。

提供轻量级 API，可直接创建 OmniMem 实例并调用记忆操作，
无需 Hermes 框架注册机制，也无需 MagicMock 注入。

用法:
    from omnimem.sdk import OmniMemSDK

    sdk = OmniMemSDK(storage_dir="~/.omnimem")
    sdk.memorize("用户喜欢Python", memory_type="preference")
    result = sdk.recall("用户喜欢什么")
    sdk.close()
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from omnimem.config import OmniMemConfig
from omnimem.core.dedup import SemanticDedupService
from omnimem.facades.governance import GovernanceFacade
from omnimem.facades.retrieval import RetrievalFacade
from omnimem.facades.storage import StorageFacade

logger = logging.getLogger(__name__)


class OmniMemSDK:
    """OmniMem 独立 SDK — 直接初始化子组件，不依赖 agent.memory_provider。"""

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if storage_dir is None:
            storage_dir = Path.home() / ".omnimem"
        self._data_dir = Path(storage_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = f"sdk-{uuid.uuid4().hex[:12]}"

        # 直接初始化子组件（不依赖 agent.memory_provider）
        self._config = OmniMemConfig(self._data_dir)
        if config:
            for key, value in config.items():
                self._config.set(key, value)

        # 初始化存储层
        self._storage = StorageFacade(self._data_dir, self._config)
        self._storage.init_l2()

        # 初始化检索层
        self._retrieval = RetrievalFacade(self._data_dir, self._config, self._storage)

        # 初始化治理层
        self._governance = GovernanceFacade(
            self._data_dir, self._config, self._session_id,
            self._storage, self._retrieval.retriever,
        )

        # 显式绑定子组件
        self._store = self._storage.store
        self._index = self._storage.index
        self._retriever = self._retrieval.retriever
        self._forgetting = self._governance.forgetting
        self._conflict_resolver = self._governance.conflict_resolver
        self._privacy = self._governance.privacy
        self._provenance = self._governance.provenance
        self._perception = self._retrieval.perception
        self._dedup = SemanticDedupService(self._store, self._retriever)

        logger.info("OmniMemSDK initialized: session=%s, data_dir=%s", self._session_id, self._data_dir)

    def memorize(self, content: str, memory_type: str = "fact", **kwargs: Any) -> dict:
        """存储记忆。"""
        from omnimem.handlers.memorize import handle_memorize

        args = {"content": content, "memory_type": memory_type, **kwargs}
        raw = handle_memorize(self._build_provider_proxy(), args)
        return json.loads(raw)

    def recall(self, query: str, mode: str = "rag", **kwargs: Any) -> dict:
        """检索记忆。"""
        from omnimem.handlers.recall import handle_recall

        args = {"query": query, "mode": mode, **kwargs}
        raw = handle_recall(self._build_provider_proxy(), args)
        return json.loads(raw)

    def reflect(self, query: str, **kwargs: Any) -> dict:
        """深层反思。"""
        from omnimem.core.tool_router import handle_reflect

        args = {"query": query, **kwargs}
        raw = handle_reflect(args, None, None)
        return json.loads(raw)

    def govern(self, action: str, **kwargs: Any) -> dict:
        """治理操作。"""
        from omnimem.handlers.govern import handle_govern

        args = {"action": action, **kwargs}
        raw = handle_govern(self._build_provider_proxy(), args)
        return json.loads(raw)

    def compact(self, **kwargs: Any) -> dict:
        """压缩前准备。"""
        from omnimem.core.tool_router import handle_compact

        args = {**kwargs}
        raw = handle_compact(args)
        return json.loads(raw)

    def detail(self, memory_id: str, **kwargs: Any) -> dict:
        """按需拉取记忆细节。"""
        from omnimem.core.tool_router import handle_detail

        args = {"action": "get", "memory_id": memory_id, **kwargs}
        raw = handle_detail(
            args,
            context_manager=self._retrieval.context_manager,
            store=self._store,
            forgetting=self._forgetting,
        )
        return json.loads(raw)

    def detail_list(self, **kwargs: Any) -> dict:
        """列出记忆。"""
        from omnimem.core.tool_router import handle_detail

        args = {"action": "list", **kwargs}
        raw = handle_detail(
            args,
            context_manager=self._retrieval.context_manager,
            store=self._store,
            forgetting=self._forgetting,
        )
        return json.loads(raw)

    def detail_events(self, from_turn: int = 0, to_turn: int | None = None, **kwargs: Any) -> dict:
        """获取事件记录。"""
        from omnimem.core.tool_router import handle_detail

        args: dict[str, Any] = {"action": "events", "from_turn": from_turn, **kwargs}
        if to_turn is not None:
            args["to_turn"] = to_turn
        raw = handle_detail(
            args,
            context_manager=self._retrieval.context_manager,
            store=self._store,
            forgetting=self._forgetting,
        )
        return json.loads(raw)

    def health_check(self) -> dict:
        """健康检查。"""
        result: dict[str, Any] = {
            "status": "healthy",
            "session_id": self._session_id,
            "data_dir": str(self._data_dir),
        }

        try:
            store_count = len(self._store.search(limit=1))
            result["store_accessible"] = True
        except Exception as e:
            result["store_accessible"] = False
            result["store_error"] = str(e)
            result["status"] = "unhealthy"

        try:
            health = self._governance.auditor.quick_health_check()
            result["audit"] = health
            if not health.get("healthy", True):
                result["status"] = "degraded"
        except Exception as e:
            result["audit_error"] = str(e)

        return result

    def export_memories(
        self, output_path: str, format: str = "json", **kwargs: Any
    ) -> dict:
        """导出记忆。"""
        from omnimem.core.import_export import MemoryExporter

        exporter = MemoryExporter(
            self._store,
            self._index,
            self._store._meta_store,
        )
        if format == "markdown":
            count = exporter.export_markdown(output_path, wing=kwargs.get("wing"))
        else:
            count = exporter.export_json(
                output_path,
                wing=kwargs.get("wing"),
                memory_type=kwargs.get("memory_type"),
            )
        return {"status": "exported", "count": count, "path": str(output_path)}

    def import_memories(self, input_path: str, **kwargs: Any) -> dict:
        """导入记忆。"""
        from omnimem.core.import_export import MemoryImporter

        importer = MemoryImporter(
            self._store,
            self._index,
            self._retriever,
            self._dedup,
            self._conflict_resolver,
            self._forgetting,
        )
        result = importer.import_json(
            input_path,
            skip_duplicates=kwargs.get("skip_duplicates", True),
            resolve_conflicts=kwargs.get("resolve_conflicts", True),
        )
        return {"status": "imported", **result}

    def _build_provider_proxy(self) -> Any:
        """构建轻量级 Provider 代理对象，供 handler 函数访问子组件。

        handler 函数（如 handle_memorize、handle_recall）需要通过 provider 对象
        访问 _store、_retriever 等属性。此代理避免创建完整的 OmniMemProvider 实例。
        """
        return _SDKProviderProxy(self)

    def _provider_style_memorize(self, args: dict[str, Any]) -> str:
        """Provider 风格记忆写入（接受 args 字典，返回 JSON 字符串）。

        供 AsyncOmniMemSDK 通过 asyncio.to_thread 调用。
        """
        from omnimem.handlers.memorize import handle_memorize

        return handle_memorize(self._build_provider_proxy(), args)

    def _provider_style_recall(self, args: dict[str, Any]) -> str:
        """Provider 风格检索（接受 args 字典，返回 JSON 字符串）。"""
        from omnimem.handlers.recall import handle_recall

        return handle_recall(self._build_provider_proxy(), args)

    def _provider_style_reflect(self, args: dict[str, Any]) -> str:
        """Provider 风格反思（接受 args 字典，返回 JSON 字符串）。"""
        from omnimem.core.tool_router import handle_reflect

        return handle_reflect(args, None, None)

    def _provider_style_govern(self, args: dict[str, Any]) -> str:
        """Provider 风格治理（接受 args 字典，返回 JSON 字符串）。"""
        from omnimem.handlers.govern import handle_govern

        return handle_govern(self._build_provider_proxy(), args)

    def close(self) -> None:
        """关闭 SDK，释放资源。"""
        try:
            self._storage.flush()
        except Exception as e:
            logger.warning("SDK close: storage flush failed: %s", e)
        try:
            self._retrieval.flush()
        except Exception as e:
            logger.warning("SDK close: retrieval flush failed: %s", e)
        try:
            self._governance.close()
        except Exception as e:
            logger.warning("SDK close: governance close failed: %s", e)
        try:
            self._storage.close()
        except Exception as e:
            logger.warning("SDK close: storage close failed: %s", e)
        logger.info("OmniMemSDK closed: session=%s", self._session_id)

    def __enter__(self) -> OmniMemSDK:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class _SDKProviderProxy:
    """轻量级 Provider 代理，将 handler 需要的属性委托到 SDK 子组件。

    handler 函数通过 provider._store、provider._retriever 等属性访问子组件，
    此代理将属性访问转发到 OmniMemSDK 实例，避免创建完整 OmniMemProvider。
    """

    def __init__(self, sdk: OmniMemSDK) -> None:
        self._sdk = sdk

    @property
    def _store(self):
        return self._sdk._store

    @property
    def _index(self):
        return self._sdk._index

    @property
    def _retriever(self):
        return self._sdk._retriever

    @property
    def _forgetting(self):
        return self._sdk._forgetting

    @property
    def _conflict_resolver(self):
        return self._sdk._conflict_resolver

    @property
    def _privacy(self):
        return self._sdk._privacy

    @property
    def _provenance(self):
        return self._sdk._provenance

    @property
    def _perception(self):
        return self._sdk._perception

    @property
    def _dedup(self):
        return self._sdk._dedup

    @property
    def _config(self):
        return self._sdk._config

    @property
    def _data_dir(self):
        return self._sdk._data_dir

    @property
    def _session_id(self):
        return self._sdk._session_id

    @property
    def _wing_room(self):
        return self._sdk._storage.wing_room

    @property
    def _context_manager(self):
        return self._sdk._retrieval.context_manager

    @property
    def _auditor(self):
        return self._sdk._governance.auditor

    @property
    def _storage(self):
        return self._sdk._storage

    @property
    def _retrieval(self):
        return self._sdk._retrieval

    @property
    def _governance(self):
        return self._sdk._governance
