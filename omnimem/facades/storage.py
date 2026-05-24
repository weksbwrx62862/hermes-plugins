"""StorageFacade — L1 工作记忆 + L2 结构化记忆。

封装: SoulSystem, CoreBlock, BudgetManager, CompactAttachment,
      WingRoomManager, DrawerClosetStore, ThreeLevelIndex, MarkdownStore
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omnimem.core.block import CoreBlock
from omnimem.core.budget import BudgetManager
from omnimem.core.soul import SoulSystem
from omnimem.memory.drawer_closet import DrawerClosetStore
from omnimem.memory.index import ThreeLevelIndex
from omnimem.memory.markdown_store import MarkdownStore
from omnimem.memory.wing_room import WingRoomManager


class StorageFacade:

    def __init__(self, data_dir: Path, config: Any):
        self._data_dir = data_dir
        self._config = config
        self._soul = SoulSystem(data_dir / "soul")
        self._core_block = CoreBlock(
            identity_block=self._soul.load_identity(),
            context_block="",
            plan_block="",
        )
        self._budget = BudgetManager(max_tokens=config.get("budget_tokens", 4000))
        self.attachments: list[Any] = []
        self._wing_room: Any = None
        self._store: Any = None
        self._index: Any = None
        self._md_store: Any = None

    def init_l2(self) -> None:
        self._wing_room = WingRoomManager(self._data_dir / "palace")
        self._store = DrawerClosetStore(
            self._data_dir / "palace",
            write_buffer_threshold=self._config.get("write_buffer_threshold", 20),
        )
        self._index = ThreeLevelIndex(self._data_dir / "index")
        self._md_store = MarkdownStore(self._data_dir / "palace")

    @property
    def soul(self) -> SoulSystem:
        return self._soul

    @property
    def core_block(self) -> CoreBlock:
        return self._core_block

    @property
    def budget(self) -> BudgetManager:
        return self._budget

    @property
    def wing_room(self) -> WingRoomManager:
        return self._wing_room

    @property
    def store(self) -> DrawerClosetStore:
        return self._store

    @property
    def index(self) -> ThreeLevelIndex:
        return self._index

    @property
    def md_store(self) -> MarkdownStore:
        return self._md_store

    def flush(self) -> None:
        if self._store:
            self._store.flush()
        if self._md_store:
            self._md_store.flush()

    def close(self) -> None:
        if self._index:
            self._index.close()
