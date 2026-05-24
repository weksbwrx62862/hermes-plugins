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
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from omnimem.memory.meta_store import MetaStore

logger = logging.getLogger(__name__)


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

    def __init__(self, palace_dir: Path, max_index_size: int = 0, write_buffer_threshold: int = 20):
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
        # ★ 磁盘写入缓冲：批量 flush 减少高频 add 时的 IO 压力
        self._write_buffer: list[Any] = []
        self._pending_disk_writes = 0
        self._WRITE_BUFFER_THRESHOLD = write_buffer_threshold

        # ★ P0方案一：MetaStore SQLite 元数据存储（并行双写）
        # 保留 Drawer 文件作为冷备份，元数据主查询走 SQLite
        self._meta_store = MetaStore(palace_dir / ".meta")

    def bind_privacy_manager(self, privacy_manager: Any) -> None:
        """OPT-1: 绑定 PrivacyManager，用于 secret 级加解密。"""
        self._privacy_manager = privacy_manager

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
        if privacy == "secret" and self._privacy_manager is not None:
            stored_content = self._privacy_manager.encrypt_content(content)

        # 1. Drawer: 完整原文（加密后的）
        drawer_dir = room_path / "drawer"
        drawer_dir.mkdir(parents=True, exist_ok=True)
        drawer_path = drawer_dir / f"{memory_id}.md"
        # ★ 批量缓冲写入，降低高频 add 的磁盘 IO
        self._write_buffer.append(
            lambda: self._write_drawer(
                drawer_path, stored_content, memory_type, confidence, privacy, provenance, now, vc
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
        self._write_buffer.append(
            lambda: self._write_closet(
                closet_path, closet_summary, memory_type, confidence, privacy, now
            )
        )
        self._pending_disk_writes += 1

        if self._pending_disk_writes >= self._WRITE_BUFFER_THRESHOLD * 2:
            self._flush_write_buffer()

        # 3. 内存索引（secret 级内存中存原文，磁盘存密文）
        self._closet_index[memory_id] = {
            "memory_id": memory_id,
            "content": content,  # 内存中保留原文
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

        # ★ P0方案一：并行写入 MetaStore（SQLite 元数据）
        self._meta_store.add(
            memory_id=memory_id,
            wing=wing,
            hall=memory_type,
            room=room,
            type=memory_type,
            confidence=confidence,
            privacy=privacy,
            stored_at=now.isoformat(),
            summary=closet_summary,
            content_preview=content[:500],
            drawer_path=str(drawer_path),
            vc=vc,
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

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """根据 ID 获取记忆。优先 MetaStore，回退内存索引/Drawer 文件。"""
        # 优先路径：MetaStore SQLite 查询
        meta_result = self._meta_store.get(memory_id)
        if meta_result:
            # 补充内存中的 content（若可用，更完整）
            if memory_id in self._closet_index:
                self._touch(memory_id)
                return dict(self._closet_index[memory_id])
            return meta_result

        # 内存索引
        if memory_id in self._closet_index:
            self._touch(memory_id)
            return dict(self._closet_index[memory_id])

        # ★ DEPRECATED: Drawer 文件查询回退，仅作兼容保留
        logger.debug("Drawer file query fallback is deprecated, use MetaStore")
        result = self._find_on_disk(memory_id)
        if result:
            self._closet_index[memory_id] = result
            self._touch(memory_id)
            self._evict_if_needed()
        return result

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

    # ★ DEPRECATED: Drawer 文件查询方法，仅作兼容保留
    def _find_on_disk(self, memory_id: str) -> dict[str, Any] | None:
        """★ DEPRECATED: 在磁盘上查找记忆，优先用路径索引，回退到 rglob。"""
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
                    enriched.append(dict(self._closet_index[mid]))
                else:
                    enriched.append(mr)
                if len(enriched) >= limit:
                    break
            return enriched

        return []

    def search_by_content(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """按内容关键词搜索。

        ★ 仅使用 MetaStore FTS5/LIKE 查询，内存子串匹配回退已移除。
        """
        # 唯一查询路径：MetaStore 全文搜索
        meta_results = self._meta_store.search_by_content(query, limit=limit)
        if meta_results:
            enriched = []
            for mr in meta_results:
                mid = mr.get("memory_id", "")
                if mid in self._closet_index:
                    enriched.append(dict(self._closet_index[mid]))
                else:
                    enriched.append(mr)
                if len(enriched) >= limit:
                    break
            return enriched

        return []

    def get_all_for_indexing(self) -> list[dict[str, Any]]:
        """获取所有记忆（用于检索引擎索引）。"""
        return [dict(entry) for entry in self._closet_index.values()]

    def warm_up(self, entries: list[dict[str, Any]]) -> None:
        """从外部数据源（如 ThreeLevelIndex）预热内存索引和 MetaStore。

        避免首次查询需要 rglob 扫描磁盘。
        """
        for entry in entries:
            mid = entry.get("memory_id", "")
            if not mid or mid in self._closet_index:
                continue
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

    def update_privacy(self, memory_id: str, privacy: str, new_wing: Optional[str] = None) -> bool:
        """更新记忆的隐私级别。可选同步更新wing。"""
        updated = False
        if memory_id in self._closet_index:
            self._closet_index[memory_id]["privacy"] = privacy
            if new_wing:
                self._closet_index[memory_id]["wing"] = new_wing
            # ★ 同步更新磁盘 Drawer 文件
            self._update_drawer_privacy(memory_id, privacy, new_wing)
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
        if memory_id in self._closet_index:
            for k, v in fields.items():
                self._closet_index[memory_id][k] = v
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

    def _update_drawer_privacy(self, memory_id: str, privacy: str, new_wing: Optional[str] = None) -> None:
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
        for fn in self._write_buffer:
            try:
                fn()
            except Exception as e:
                logger.warning("Buffered write failed: %s", e)
        self._write_buffer.clear()
        self._pending_disk_writes = 0

    def flush(self) -> None:
        """显式刷新所有待写入的磁盘缓冲和 MetaStore。"""
        if self._write_buffer:
            self._flush_write_buffer()

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
        summary: str,
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

        text = f"---\n{fm_str}---\n\n{summary}\n"
        path.write_text(text, encoding="utf-8")

    def _read_drawer(self, path: Path) -> dict[str, Any] | None:
        """从 Drawer 文件读取记忆。OPT-1: secret 级内容自动解密。"""
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
                    # OPT-1: 解密 secret 级内容
                    if privacy == "secret" and self._privacy_manager is not None:
                        content = self._privacy_manager.decrypt_content(content)
                    return {
                        "memory_id": fm.get("memory_id", path.stem),
                        "content": content,
                        "type": fm.get("type", "fact"),
                        "confidence": fm.get("confidence", 3),
                        "privacy": privacy,
                        "stored_at": fm.get("stored_at"),
                        "provenance": fm.get("provenance"),
                        "vc": fm.get("vc", ""),
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
            oldest_id, _ = self._closet_index.popitem(last=False)
            logger.debug("Closet index evicted: %s", oldest_id)
