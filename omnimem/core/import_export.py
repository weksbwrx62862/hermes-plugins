from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EXPORT_VERSION = "1.0"

logger = logging.getLogger(__name__)


class MemoryExporter:

    def __init__(self, store: Any, index: Any, meta_store: Any):
        self._store = store
        self._index = index
        self._meta = meta_store

    def export_json(
        self,
        output_path: str | Path,
        wing: str | None = None,
        memory_type: str | None = None,
    ) -> int:
        output_path = Path(output_path)
        entries = self._store.search(limit=10000)
        if wing:
            entries = [e for e in entries if e.get("wing") == wing]
        if memory_type:
            entries = [e for e in entries if e.get("type") == memory_type]

        records: list[dict[str, Any]] = []
        for entry in entries:
            mid = entry.get("memory_id", "")
            full = self._store.get(mid) or entry
            record: dict[str, Any] = {
                "memory_id": mid,
                "content": full.get("content", ""),
                "summary": full.get("summary", ""),
                "type": full.get("type", "fact"),
                "wing": full.get("wing", ""),
                "room": full.get("room", ""),
                "privacy": full.get("privacy", "personal"),
                "confidence": full.get("confidence", 3),
                "created_at": full.get("stored_at", ""),
                "access_count": 0,
            }
            records.append(record)

        payload: dict[str, Any] = {
            "version": _EXPORT_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "count": len(records),
            "memories": records,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return len(records)

    def export_markdown(
        self,
        output_dir: str | Path,
        wing: str | None = None,
    ) -> int:
        output_dir = Path(output_dir)
        entries = self._store.search(limit=10000)
        if wing:
            entries = [e for e in entries if e.get("wing") == wing]

        count = 0
        for entry in entries:
            mid = entry.get("memory_id", "")
            full = self._store.get(mid) or entry
            entry_wing = full.get("wing", "unknown")
            entry_room = full.get("room", "default")
            entry_type = full.get("type", "fact")

            file_dir = output_dir / entry_wing / entry_room
            file_dir.mkdir(parents=True, exist_ok=True)
            file_path = file_dir / f"{mid}.md"

            front_matter_lines = [
                f"memory_id: \"{mid}\"",
                f"type: \"{entry_type}\"",
                f"wing: \"{entry_wing}\"",
                f"room: \"{entry_room}\"",
                f"privacy: \"{full.get('privacy', 'personal')}\"",
                f"confidence: {full.get('confidence', 3)}",
                f"created_at: \"{full.get('stored_at', '')}\"",
            ]
            fm = "\n".join(front_matter_lines)
            content = full.get("content", "")
            text = f"---\n{fm}\n---\n\n{content}\n"
            file_path.write_text(text, encoding="utf-8")
            count += 1

        return count


class MemoryImporter:

    def __init__(
        self,
        store: Any,
        index: Any,
        retriever: Any,
        dedup_service: Any,
        conflict_resolver: Any,
        forgetting: Any,
    ):
        self._store = store
        self._index = index
        self._retriever = retriever
        self._dedup = dedup_service
        self._conflict = conflict_resolver
        self._forgetting = forgetting

    def import_json(
        self,
        input_path: str | Path,
        skip_duplicates: bool = True,
        resolve_conflicts: bool = True,
    ) -> dict[str, int]:
        input_path = Path(input_path)
        raw = input_path.read_text(encoding="utf-8")
        payload = json.loads(raw)

        records = payload.get("memories", [])
        total = len(records)
        imported = 0
        skipped = 0
        conflicts = 0

        for record in records:
            content = record.get("content", "")
            if not content:
                skipped += 1
                continue

            memory_type = record.get("type", "fact")
            existing_id = record.get("memory_id", "")
            privacy = record.get("privacy", "personal")
            confidence = record.get("confidence", 3)
            wing = record.get("wing", "")
            room = record.get("room", "")

            if skip_duplicates:
                dedup_result = self._dedup.semantic_dedup(content, memory_type)
                if dedup_result["action"] == "skip":
                    skipped += 1
                    continue

            if resolve_conflicts:
                candidates = self._dedup.search_candidates(content)
                conflict_result = self._conflict.check(
                    content,
                    existing_memories=[
                        {"content": m.get("content", ""), "memory_id": m.get("memory_id", "")}
                        for m in candidates[:10]
                    ],
                )
                if conflict_result.has_conflict:
                    resolution = self._conflict.resolve(content, conflict_result)
                    if resolution.action == "reject":
                        conflicts += 1
                        continue

            if existing_id and self._store.get(existing_id):
                new_id = uuid.uuid4().hex[:12]
            else:
                new_id = existing_id or uuid.uuid4().hex[:12]

            self._store.add(
                wing=wing or "personal",
                room=room or "imported",
                content=content,
                memory_type=memory_type,
                confidence=confidence,
                privacy=privacy,
                memory_id=new_id,
            )

            self._index.add(
                memory_id=new_id,
                wing=wing or "personal",
                hall=memory_type,
                room=room or "imported",
                content=content,
                summary=content[:200].replace("\n", " "),
                type=memory_type,
                confidence=confidence,
                privacy=privacy,
                stored_at=record.get("created_at", datetime.now(timezone.utc).isoformat()),
            )

            try:
                self._retriever.add(
                    content,
                    memory_id=new_id,
                    metadata={
                        "memory_id": new_id,
                        "type": memory_type,
                        "confidence": confidence,
                        "privacy": privacy,
                        "wing": wing or "personal",
                        "room": room or "imported",
                    },
                )
            except Exception as e:
                logger.warning("MemoryImporter retriever add failed: %s", e)

            self._forgetting.record_access(new_id)
            imported += 1

        self._store.flush()
        self._index.flush()

        return {
            "total": total,
            "imported": imported,
            "skipped": skipped,
            "conflicts": conflicts,
        }
