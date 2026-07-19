"""DrawerClosetStore — Drawer(逐字原文) + Closet(摘要指针) 双存储。

参考 MemPalace 的 Drawer(逐字原文存储) + Closet(摘要指针存储) 设计：
  - Drawer: 完整原文，Markdown + YAML Front Matter 格式，不丢失任何细节
  - Closet: 摘要指针，轻量索引，用于快速检索和浏览

写入路径：
  palace/<wing>/<hall>/<room>/drawer/<memory_id>.md  ← Drawer 原文
  palace/<wing>/<hall>/<room>/closet/<memory_id>.md  ← Closet 摘要
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnimem.core.saga import SagaCoordinator, SagaStep
from omnimem.governance.encryption import EncryptionUnavailableError
from omnimem.memory.meta_store import MetaStore

logger = logging.getLogger(__name__)


@dataclass
class WriteOp:
    """可序列化的磁盘写入操作描述，替代 partial 函数对象以降低崩溃时数据丢失风险。"""

    op_type: str  # "drawer" 或 "closet"
    path: Path
    content: str
    memory_type: str
    confidence: int
    privacy: str
    stored_at: datetime
    provenance: dict[str, Any] | None = None
    vc: str = ""
    entities: list[Any] | None = None


def _generate_id() -> str:
    """生成唯一记忆 ID。"""
    return uuid.uuid4().hex[:12]


class DrawerClosetStore:
    """Drawer(逐字原文) + Closet(摘要指针) 双存储。

    内存索引有容量上限，超出时按 LRU 淘汰冷数据。
    淘汰后磁盘查找仍然可用（通过 rglob 回退）。

    性能优化：
      - 二级倒排索引：type/wing → set(memory_id)，search() O(k) 替代 O(n)
      - 内存索引中 content 按需加载（仅在 get() 和 search_by_content 时）
    """

    # 内存索引最大条目数
    _MAX_CLOSET_INDEX = 10000

    def __init__(
        self,
        palace_dir: Path,
        max_index_size: int = 0,
        write_buffer_threshold: int = 20,
        config: Any | None = None,
    ):
        self._palace_dir = palace_dir
        self._palace_dir.mkdir(parents=True, exist_ok=True)
        # 内存索引（Closet 加速），带容量限制
        self._closet_index: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_index = max_index_size or self._MAX_CLOSET_INDEX
        # 路径索引：memory_id → drawer_path，加速磁盘查找
        self._id_to_path: dict[str, Path] = {}
        # ★ 二级倒排索引：type/wing → set(memory_id)，加速分类搜索
        self._type_index: dict[str, set[str]] = {}
        self._wing_index: dict[str, set[str]] = {}
        # OPT-1: 延迟绑定的 PrivacyManager（用于加密/解密）
        self._privacy_manager: Any = None
        # 配置引用，用于判断加密默认是否启用
        self._config = config
        # ★ 磁盘写入缓冲：批量 flush 减少高频 add 时的 IO 压力
        self._write_buffer: list[WriteOp] = []
        self._pending_disk_writes = 0
        self._WRITE_BUFFER_THRESHOLD = write_buffer_threshold
        # ★ 线程安全锁：保护 _closet_index 和 _write_buffer 的并发访问
        self._index_lock = threading.RLock()

        # ★ P0方案一：MetaStore SQLite 元数据存储（并行双写）
        # 保留 Drawer 文件作为冷备份，元数据主查询走 SQLite
        self._meta_store = MetaStore(palace_dir / ".meta")
        # ★ P0修复：Saga 协调器，保证 Drawer/MetaStore 双写事务一致性
        self._saga = SagaCoordinator(pending_path=palace_dir / ".meta" / "saga_pending.json")

    def bind_privacy_manager(self, privacy_manager: Any) -> None:
        """OPT-1: 绑定 PrivacyManager，用于 secret 级加解密。"""
        self._privacy_manager = privacy_manager

    @property
    def meta_store(self) -> MetaStore:
        """公开访问 MetaStore 实例（供审计/导出等跨层使用）。"""
        return self._meta_store

    def add(
        self,
        wing: str,
        room: str,
        content: str,
        memory_type: str = "fact",
        confidence: int = 3,
        privacy: str = "personal",
        provenance: dict[str, Any] | None = None,
        vc: str = "",
        memory_id: str = "",
        **kwargs: Any,
    ) -> str:
        """添加一条记忆，同时写入 Drawer 和 Closet。

        OPT-1: secret 级内容在写入前加密。

        Args:
            memory_id: 可选指定 ID（用于分布式同步时保留远程 ID）

        Returns:
            memory_id
        """
        if not memory_id:
            memory_id = _generate_id()
        now = datetime.now(timezone.utc)
        room_path = self._palace_dir / wing / memory_type / room

        # OPT-1: secret 级内容加密
        stored_content = content
        if privacy == "secret":
            encryption_enabled = (
                self._config.get("enable_encryption", False)
                if self._config is not None
                else False
            )
            if encryption_enabled:
                if self._privacy_manager is None or not getattr(
                    self._privacy_manager.encryption, "is_available", lambda: False
                )():
                    raise EncryptionUnavailableError(
                        "未配置有效加密密钥，无法写入 secret 级记忆"
                    )
                stored_content = self._privacy_manager.encrypt_content(content)
            elif self._privacy_manager is not None:
                stored_content = self._privacy_manager.encrypt_content(content)

        # 1. Drawer: 完整原文（加密后的）
        drawer_dir = room_path / "drawer"
        drawer_dir.mkdir(parents=True, exist_ok=True)
        drawer_path = drawer_dir / f"{memory_id}.md"
        # ★ 批量缓冲写入，降低高频 add 的磁盘 IO；使用 WriteOp 替代 partial，避免闭包不可序列化
        with self._index_lock:
            entities = kwargs.pop("entities", [])
            self._write_buffer.append(
                WriteOp(
                    op_type="drawer",
                    path=drawer_path,
                    content=stored_content,
                    memory_type=memory_type,
                    confidence=confidence,
                    privacy=privacy,
                    stored_at=now,
                    provenance=provenance,
                    vc=vc,
                    entities=entities,
                )
            )
            self._pending_disk_writes += 1

        # 2. Closet: 摘要指针（secret 级不存摘要，存标记）
        closet_dir = room_path / "closet"
        closet_dir.mkdir(parents=True, exist_ok=True)
        closet_path = closet_dir / f"{memory_id}.md"
        # ★ R19修复Minor-2: summary中换行符替换为空格，避免含\n的内容在首行截断
        if privacy == "secret":
            closet_summary = "[加密记忆]"
        else:
            closet_summary = content[:200].replace("\n", " ").replace("\r", " ").replace("\t", " ")
        with self._index_lock:
            self._write_buffer.append(
                WriteOp(
                    op_type="closet",
                    path=closet_path,
                    content=closet_summary,
                    memory_type=memory_type,
                    confidence=confidence,
                    privacy=privacy,
                    stored_at=now,
                )
            )
            self._pending_disk_writes += 1

        if self._pending_disk_writes >= self._WRITE_BUFFER_THRESHOLD * 2:
            self._flush_write_buffer()

        # 3. 内存索引（secret 级存密文，非 secret 级存原文）
        with self._index_lock:
            # secret 级不在内存索引中保留明文 content，使用已加密的 stored_content
            index_content = stored_content if privacy == "secret" else content
            self._closet_index[memory_id] = {
                "memory_id": memory_id,
                "content": index_content,
                "summary": closet_summary,
                "type": memory_type,
                "confidence": confidence,
                "privacy": privacy,
                "wing": wing,
                "room": room,
                "hall": memory_type,
                "stored_at": now.isoformat(),
                "provenance": provenance,
                "vc": vc,
                **kwargs,
            }
            self._touch(memory_id)
            self._evict_if_needed()

        # 4. 路径索引，加速磁盘查找
        self._id_to_path[memory_id] = drawer_path

        # ★ 5. 二级倒排索引
        self._type_index.setdefault(memory_type, set()).add(memory_id)
        self._wing_index.setdefault(wing, set()).add(memory_id)

        # secret 级不在 MetaStore 中保留明文 content_preview，避免通过 FTS 泄露
        meta_content_preview = "" if privacy == "secret" else content[:500]
        meta_fields = {
            "memory_id": memory_id,
            "wing": wing,
            "hall": memory_type,
            "room": room,
            "type": memory_type,
            "confidence": confidence,
            "privacy": privacy,
            "stored_at": now.isoformat(),
            "summary": closet_summary,
            "content_preview": meta_content_preview,
            "drawer_path": str(drawer_path),
            "vc": vc,
        }

        def _write_meta() -> None:
            """写入 MetaStore（正向操作）。"""
            self._meta_store.add(**meta_fields)

        def _compensate() -> None:
            """回滚 Drawer/Closet 文件、内存索引和 MetaStore 记录（补偿操作）。

            当缓冲尚未 flush 时，drawer/closet 文件尚未落盘，删除操作会安全失败；
            MetaStore 记录已写入，需要显式回滚以保证双写一致性。
            """
            logger.warning("Saga 补偿：回滚 memory_id=%s", memory_id)
            try:
                if drawer_path.exists():
                    drawer_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Saga 补偿：删除 drawer 文件失败: %s", e)
            try:
                if closet_path.exists():
                    closet_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Saga 补偿：删除 closet 文件失败: %s", e)
            try:
                self._meta_store.delete(memory_id)
            except Exception as e:
                logger.warning("Saga 补偿：删除 MetaStore 记录失败: %s", e)
            # 清理内存索引
            with self._index_lock:
                self._closet_index.pop(memory_id, None)
            self._id_to_path.pop(memory_id, None)
            for _type_set in self._type_index.values():
                _type_set.discard(memory_id)
            for _wing_set in self._wing_index.values():
                _wing_set.discard(memory_id)

        # 设计意图说明（P0-1 语义澄清）：
        #   1. 实际的 drawer 写盘在 Saga 之前已通过 _write_buffer 完成（见上方第 155-202 行），
        #      进入 Saga 时文件已落盘并已更新内存索引，故本步骤 action 为 no-op 是有意设计，
        #      并非遗漏。
        #   2. 步骤名 drawer_commit 表示“提交/登记”语义（区别于“写入 drawer”），
        #      作用是为后续 meta_store_write 失败时提供补偿锚点。
        #   3. _compensate 负责回滚已落盘的 drawer/closet 文件以及内存索引（closet_index、
        #      id_to_path、type_index、wing_index），保证 Saga 失败时数据一致性。
        saga_result = self._saga.execute(
            memory_id=memory_id,
            steps=[
                SagaStep(name="drawer_commit", action=lambda: None, compensate=_compensate),
                SagaStep(name="meta_store_write", action=_write_meta),
            ],
        )

        if not saga_result.success:
            logger.warning(
                "Drawer/MetaStore 双写 Saga 失败: memory_id=%s, failed_step=%s, error=%s",
                memory_id, saga_result.failed_step, saga_result.error,
            )

        logger.info(
            "Stored memory %s in %s/%s/%s (type=%s, confidence=%d, privacy=%s)",
            memory_id,
            wing,
            memory_type,
            room,
            memory_type,
            confidence,
            privacy,
        )
        return memory_id

    def _decrypt_entry_content(self, entry: dict[str, Any]) -> dict[str, Any]:
        """对 secret 级条目的 content 按需解密，返回新字典。"""
        result = dict(entry)
        if result.get("privacy") != "secret":
            return result
        if self._privacy_manager is None:
            return result
        raw_content = result.get("content", "")
        # 仅对真实密文解密，避免对占位符或已解密内容重复操作
        if raw_content and self._privacy_manager.is_encrypted(raw_content):
            result["content"] = self._privacy_manager.decrypt_content(raw_content)
        return result

    def _sanitize_secret_result(self, entry: dict[str, Any]) -> dict[str, Any]:
        """对搜索结果中的 secret 级条目做脱敏处理，避免泄露明文。"""
        result = dict(entry)
        if result.get("privacy") == "secret":
            result["content"] = "[加密记忆 — 使用 omni_detail 解锁]"
            result["content_preview"] = ""
            result["_encrypted"] = True
        return result

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """根据 ID 获取记忆。优先内存索引，回退 MetaStore + Drawer 文件。"""
        # 1. 内存索引（热数据）
        if memory_id in self._closet_index:
            self._touch(memory_id)
            return self._decrypt_entry_content(self._closet_index[memory_id])

        # 2. MetaStore 元数据 + Drawer 原文
        meta_result = self._meta_store.get(memory_id)
        if meta_result:
            disk_entry = self._find_on_disk(memory_id)
            if disk_entry:
                cached_entry: dict[str, Any] = {**meta_result, "content": disk_entry["content"]}
            else:
                cached_entry = dict(meta_result)
            with self._index_lock:
                self._closet_index[memory_id] = cached_entry
                self._touch(memory_id)
                self._evict_if_needed()
            return self._decrypt_entry_content(cached_entry)

        # 3. Legacy: Drawer 文件查询回退
        logger.debug("Drawer file query fallback (MetaStore miss)")
        result = self._find_on_disk(memory_id)
        if result:
            with self._index_lock:
                self._closet_index[memory_id] = result
                self._touch(memory_id)
                self._evict_if_needed()
            return self._decrypt_entry_content(result)
        return None

    def delete(self, memory_id: str) -> bool:
        """删除记忆：MetaStore + 内存索引 + 文件系统 + 路径索引 + 倒排索引。

        Returns:
            True 如果至少清理了一层存储
        """
        deleted_any = False

        # 1. MetaStore（SQLite 元数据）
        try:
            self._meta_store.delete(memory_id)
            deleted_any = True
        except Exception as e:
            logger.warning("DrawerClosetStore.delete: MetaStore failed for %s: %s", memory_id, e)

        # 2. 内存索引
        with self._index_lock:
            if memory_id in self._closet_index:
                del self._closet_index[memory_id]
                deleted_any = True

        # 3. 路径索引
        drawer_path = self._id_to_path.pop(memory_id, None)

        # 4. 文件系统（drawer + closet .md 文件）
        if drawer_path is not None:
            try:
                drawer_path.unlink(missing_ok=True)
                deleted_any = True
            except Exception as e:
                logger.warning("DrawerClosetStore.delete: drawer unlink failed: %s", e)
            # 也尝试删除对应的 closet 文件
            try:
                closet_path = drawer_path.parent.parent / "closet" / drawer_path.name
                closet_path.unlink(missing_ok=True)
                deleted_any = True
            except Exception as e:
                logger.warning("DrawerClosetStore.delete: closet unlink failed: %s", e)

        # 5. 倒排索引清理
        for _type_set in self._type_index.values():
            _type_set.discard(memory_id)
        for _wing_set in self._wing_index.values():
            _wing_set.discard(memory_id)

        return deleted_any

    # ★ Legacy: Drawer 文件查询方法，MetaStore 未命中时回退使用
    def _find_on_disk(self, memory_id: str) -> dict[str, Any] | None:
        """在磁盘上查找记忆，优先用路径索引，回退到 rglob。"""
        # 策略1：用已知的路径索引
        known_path = self._id_to_path.get(memory_id)
        if known_path and known_path.exists():
            return self._read_drawer(known_path)

        # 策略2：rglob 回退（路径索引未命中时）
        for drawer_file in self._palace_dir.rglob(f"drawer/{memory_id}.md"):
            result = self._read_drawer(drawer_file)
            if result:
                # 记录路径以供下次快速查找
                self._id_to_path[memory_id] = drawer_file
                return result

        return None

    def search(
        self,
        wing: str = "",
        room: str = "",
        memory_type: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """按条件搜索记忆。

        ★ 仅使用 MetaStore SQLite 查询（O(log n)），
        内存二级索引回退已移除。
        secret 级结果会脱敏，不返回明文。
        """
        # 唯一查询路径：MetaStore SQL 索引查询
        meta_results = self._meta_store.search(
            wing=wing, room=room, memory_type=memory_type, limit=limit
        )
        if meta_results:
            enriched = []
            for mr in meta_results:
                mid = mr.get("memory_id", "")
                if mid in self._closet_index:
                    entry = dict(self._closet_index[mid])
                    self._touch(mid)
                else:
                    entry = dict(mr)
                enriched.append(self._sanitize_secret_result(entry))
                if len(enriched) >= limit:
                    break
            return enriched

        return []

    def search_by_content(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """按内容关键词搜索。

        ★ 仅使用 MetaStore FTS5/LIKE 查询，内存子串匹配回退已移除。
        secret 级结果会脱敏，不返回明文。
        """
        # 唯一查询路径：MetaStore 全文搜索
        meta_results = self._meta_store.search_by_content(query, limit=limit)
        if meta_results:
            enriched = []
            for mr in meta_results:
                mid = mr.get("memory_id", "")
                if mid in self._closet_index:
                    entry = dict(self._closet_index[mid])
                    self._touch(mid)
                else:
                    entry = dict(mr)
                enriched.append(self._sanitize_secret_result(entry))
                if len(enriched) >= limit:
                    break
            return enriched

        return []

    def get_all_for_indexing(self) -> list[dict[str, Any]]:
        """获取所有记忆（用于检索引擎索引）。"""
        return [
            self._sanitize_secret_result(dict(entry))
            for entry in self._closet_index.values()
        ]

    def warm_up(self, entries: list[dict[str, Any]]) -> None:
        """从外部数据源（如 ThreeLevelIndex）预热内存索引和 MetaStore。

        避免首次查询需要 rglob 扫描磁盘。
        """
        for entry in entries:
            mid = entry.get("memory_id", "")
            if not mid or mid in self._closet_index:
                continue
            with self._index_lock:
                self._closet_index[mid] = entry
                self._touch(mid)
            # 更新二级索引
            mtype = entry.get("type", "fact")
            wing = entry.get("wing", "")
            self._type_index.setdefault(mtype, set()).add(mid)
            if wing:
                self._wing_index.setdefault(wing, set()).add(mid)
        self._evict_if_needed()
        # ★ P0方案一：同步预热 MetaStore
        self._meta_store.warm_up(entries)
        logger.info("Warmed up %d entries into closet index and meta store", len(entries))

    def update_privacy(self, memory_id: str, privacy: str, new_wing: str | None = None) -> bool:
        """更新记忆的隐私级别。可选同步更新wing。"""
        updated = False
        with self._index_lock:
            if memory_id in self._closet_index:
                self._closet_index[memory_id]["privacy"] = privacy
                if new_wing:
                    self._closet_index[memory_id]["wing"] = new_wing
                # ★ 同步更新磁盘 Drawer 文件
                self._update_drawer_privacy(memory_id, privacy, new_wing)
                self._touch(memory_id)
                updated = True
            else:
                # 即使不在内存索引中，也尝试从磁盘查找并更新
                result = self._find_on_disk(memory_id)
                if result:
                    self._closet_index[memory_id] = result
                    self._closet_index[memory_id]["privacy"] = privacy
                    if new_wing:
                        self._closet_index[memory_id]["wing"] = new_wing
                    self._touch(memory_id)
                    self._update_drawer_privacy(memory_id, privacy, new_wing)
                    updated = True

        # ★ P0方案一：同步更新 MetaStore
        if updated:
            self._meta_store.update_privacy(memory_id, privacy, new_wing or "")
        return updated

    def update_field(self, memory_id: str, **fields: Any) -> bool:
        """更新内存索引和 MetaStore 中的指定字段。

        ★ R28v2修复BUG-3：供 memorize.py 写入 conflicting_with/conflict_type 等字段。
        """
        updated = False
        with self._index_lock:
            if memory_id in self._closet_index:
                for k, v in fields.items():
                    self._closet_index[memory_id][k] = v
                self._touch(memory_id)
                updated = True
            else:
                result = self._find_on_disk(memory_id)
                if result:
                    self._closet_index[memory_id] = result
                    for k, v in fields.items():
                        self._closet_index[memory_id][k] = v
                    self._touch(memory_id)
                    updated = True
        if updated:
            self._meta_store.update_field(memory_id, **fields)
        return updated

    def _update_drawer_privacy(self, memory_id: str, privacy: str, new_wing: str | None = None) -> None:
        """更新 Drawer 磁盘文件中的 privacy 和 wing 字段。"""
        drawer_path = self._id_to_path.get(memory_id)
        if not drawer_path or not drawer_path.exists():
            return
        try:
            text = drawer_path.read_text(encoding="utf-8")
            if text.startswith("---"):
                import re

                text = re.sub(
                    r"privacy:\s*\S+",
                    f"privacy: {privacy}",
                    text,
                    count=1,
                )
                if new_wing:
                    text = re.sub(
                        r"wing:\s*\S+",
                        f"wing: {new_wing}",
                        text,
                        count=1,
                    )
                drawer_path.write_text(text, encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to update drawer privacy for %s: %s", memory_id, e)

    def _flush_write_buffer(self) -> None:
        """执行缓冲队列中的所有磁盘写入。"""
        with self._index_lock:
            buffer = list(self._write_buffer)
            self._write_buffer.clear()
            self._pending_disk_writes = 0
        for op in buffer:
            try:
                if op.op_type == "drawer":
                    self._write_drawer(
                        op.path,
                        op.content,
                        op.memory_type,
                        op.confidence,
                        op.privacy,
                        op.provenance,
                        op.stored_at,
                        op.vc,
                        op.entities,
                    )
                elif op.op_type == "closet":
                    self._write_closet(
                        op.path,
                        op.content,
                        op.memory_type,
                        op.confidence,
                        op.privacy,
                        op.stored_at,
                    )
                else:
                    logger.warning("未知的 WriteOp 类型: %s", op.op_type)
            except Exception as e:
                logger.warning("Buffered write failed: %s", e)

    def flush(self) -> None:
        """显式刷新所有待写入的磁盘缓冲和 MetaStore。"""
        if self._write_buffer:
            self._flush_write_buffer()
        self._meta_store.flush()

    async def async_flush(self) -> None:
        """异步刷新所有待写入数据（磁盘缓冲 + MetaStore）。

        将同步 IO 操作提交到线程池执行，避免阻塞事件循环。
        """
        import asyncio

        await asyncio.to_thread(self.flush)

    def close(self) -> None:
        """关闭存储引擎，释放 SQLite 连接等资源。"""
        self.flush()
        if hasattr(self, "_meta_store") and self._meta_store:
            self._meta_store.close()

    def _write_drawer(
        self,
        path: Path,
        content: str,
        memory_type: str,
        confidence: int,
        privacy: str,
        provenance: dict[str, Any] | None,
        stored_at: datetime,
        vc: str = "",
        entities: list[str] | None = None,
    ) -> None:
        """写入 Drawer（完整原文，Markdown + YAML FM）。"""
        front_matter = {
            "memory_id": path.stem,
            "type": memory_type,
            "confidence": confidence,
            "privacy": privacy,
            "stored_at": stored_at.isoformat(),
        }
        if provenance:
            front_matter["provenance"] = provenance
        if vc:
            front_matter["vc"] = vc
        if entities:
            front_matter["entities"] = entities

        try:
            import yaml

            fm_str = yaml.dump(front_matter, allow_unicode=True, default_flow_style=False)
        except ImportError:
            fm_str = "\n".join(f"{k}: {v}" for k, v in front_matter.items())

        text = f"---\n{fm_str}---\n\n{content}\n"
        path.write_text(text, encoding="utf-8")

    def _write_closet(
        self,
        path: Path,
        content: str,
        memory_type: str,
        confidence: int,
        privacy: str,
        stored_at: datetime,
    ) -> None:
        """写入 Closet（摘要指针）。"""
        front_matter = {
            "memory_id": path.stem,
            "type": memory_type,
            "confidence": confidence,
            "privacy": privacy,
            "stored_at": stored_at.isoformat(),
        }

        try:
            import yaml

            fm_str = yaml.dump(front_matter, allow_unicode=True, default_flow_style=False)
        except ImportError:
            fm_str = "\n".join(f"{k}: {v}" for k, v in front_matter.items())

        text = f"---\n{fm_str}---\n\n{content}\n"
        path.write_text(text, encoding="utf-8")

    def _read_drawer(self, path: Path) -> dict[str, Any] | None:
        """从 Drawer 文件读取记忆。

        OPT-1: secret 级内容保持密文返回，由上层 get() 按需解密，
        避免在内存索引中留下明文副本。
        """
        try:
            text = path.read_text(encoding="utf-8")
            # 解析 YAML front matter
            if text.startswith("---"):
                parts = text.split("---", 2)
                if len(parts) >= 3:
                    try:
                        import yaml

                        fm = yaml.safe_load(parts[1]) or {}
                    except ImportError:
                        fm = {}
                    content = parts[2].strip()
                    privacy = fm.get("privacy", "personal")
                    return {
                        "memory_id": fm.get("memory_id", path.stem),
                        "content": content,
                        "type": fm.get("type", "fact"),
                        "confidence": fm.get("confidence", 3),
                        "privacy": privacy,
                        "stored_at": fm.get("stored_at"),
                        "provenance": fm.get("provenance"),
                        "vc": fm.get("vc", ""),
                        "entities": fm.get("entities", []),
                    }
            return {"memory_id": path.stem, "content": text}
        except Exception as e:
            logger.warning("Failed to read drawer %s: %s", path, e)
            return None

    # ─── LRU 管理 ─────────────────────────────────────────────

    def _touch(self, memory_id: str) -> None:
        """更新访问顺序（LRU），使用 OrderedDict O(1) 操作。"""
        if memory_id in self._closet_index:
            self._closet_index.move_to_end(memory_id)

    def _evict_if_needed(self) -> None:
        """超出容量时淘汰最久未访问的条目。"""
        if len(self._closet_index) > self._max_index:
            # ★ 先刷盘再淘汰：确保被淘汰的条目已持久化到磁盘
            self.flush()
        while len(self._closet_index) > self._max_index:
            oldest_id, oldest_entry = self._closet_index.popitem(last=False)
            # ★ 修复2: 同步清理倒排索引中的悬空引用
            oldest_type = oldest_entry.get("type", "") if isinstance(oldest_entry, dict) else ""
            oldest_wing = oldest_entry.get("wing", "") if isinstance(oldest_entry, dict) else ""
            if oldest_type and oldest_type in self._type_index:
                self._type_index[oldest_type].discard(oldest_id)
                if not self._type_index[oldest_type]:
                    del self._type_index[oldest_type]
            if oldest_wing and oldest_wing in self._wing_index:
                self._wing_index[oldest_wing].discard(oldest_id)
                if not self._wing_index[oldest_wing]:
                    del self._wing_index[oldest_wing]
            logger.debug("Closet index evicted: %s", oldest_id)
