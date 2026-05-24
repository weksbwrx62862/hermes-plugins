"""MemoryExporter / MemoryImporter 导入导出模块单元测试。

覆盖: export_json (wing/type过滤) / export_markdown / import_json
       import_json (去重/冲突/ID冲突处理)
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from omnimem.core.import_export import MemoryExporter, MemoryImporter


class TestMemoryExporter(unittest.TestCase):
    """MemoryExporter 导出功能测试。"""

    def setUp(self) -> None:
        self.store = MagicMock()
        self.index = MagicMock()
        self.meta = MagicMock()
        self.exporter = MemoryExporter(self.store, self.index, self.meta)

    def test_export_json_basic(self) -> None:
        self.store.search.return_value = [
            {"memory_id": "m1", "type": "fact", "wing": "personal",
             "room": "test", "privacy": "personal", "confidence": 3,
             "stored_at": "2026-01-01T00:00:00Z"}
        ]
        self.store.get.return_value = None  # falls back to entry itself
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "export.json"
            count = self.exporter.export_json(out)
            self.assertEqual(count, 1)
            self.assertTrue(out.exists())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["count"], 1)
            self.assertEqual(data["memories"][0]["type"], "fact")

    def test_export_json_wing_filter(self) -> None:
        self.store.search.return_value = [
            {"memory_id": "m1", "wing": "personal", "type": "fact"},
            {"memory_id": "m2", "wing": "team", "type": "fact"},
            {"memory_id": "m3", "wing": "personal", "type": "preference"},
        ]
        self.store.get.side_effect = lambda mid: {"memory_id": mid, "wing": "personal" if mid != "m2" else "team", "type": "fact", "room": "r", "privacy": "personal", "confidence": 3, "stored_at": ""}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "export_wing.json"
            count = self.exporter.export_json(out, wing="personal")
            self.assertEqual(count, 2)

    def test_export_json_type_filter(self) -> None:
        self.store.search.return_value = [
            {"memory_id": "m1", "type": "fact", "wing": "personal"},
            {"memory_id": "m2", "type": "preference", "wing": "personal"},
        ]
        self.store.get.side_effect = lambda mid: {"memory_id": mid, "type": "preference" if mid == "m2" else "fact", "wing": "personal", "room": "r", "privacy": "personal", "confidence": 3, "stored_at": ""}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "export_type.json"
            count = self.exporter.export_json(out, memory_type="preference")
            self.assertEqual(count, 1)

    def test_export_markdown(self) -> None:
        self.store.search.return_value = [
            {"memory_id": "md1", "type": "fact", "wing": "personal",
             "room": "notes", "privacy": "personal", "confidence": 4,
             "stored_at": "2026-01-01T00:00:00Z"}
        ]
        self.store.get.return_value = {
            "memory_id": "md1", "type": "fact", "wing": "personal",
            "room": "notes", "privacy": "personal", "confidence": 4,
            "stored_at": "2026-01-01T00:00:00Z",
            "content": "# Hello Markdown\n\nTest content."
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "md_export"
            count = self.exporter.export_markdown(out_dir)
            self.assertEqual(count, 1)
            expected_path = out_dir / "personal" / "notes" / "md1.md"
            self.assertTrue(expected_path.exists())
            content = expected_path.read_text(encoding="utf-8")
            self.assertIn("memory_id:", content)
            self.assertIn("Hello Markdown", content)

    def test_export_markdown_wing_filter(self) -> None:
        self.store.search.return_value = [
            {"memory_id": "md_a", "wing": "public", "type": "fact"},
            {"memory_id": "md_b", "wing": "personal", "type": "fact"},
        ]
        self.store.get.side_effect = lambda mid: {
            "memory_id": mid, "wing": "public" if mid == "md_a" else "personal",
            "room": "r", "type": "fact", "privacy": "public", "confidence": 3,
            "stored_at": "", "content": "test"
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "md_wing"
            count = self.exporter.export_markdown(out_dir, wing="public")
            self.assertEqual(count, 1)


class TestMemoryImporter(unittest.TestCase):
    """MemoryImporter 导入功能测试。"""

    def setUp(self) -> None:
        self.store = MagicMock()
        self.store.add = MagicMock()
        self.store.flush = MagicMock()
        self.store.get = MagicMock(return_value=None)
        self.index = MagicMock()
        self.index.add = MagicMock()
        self.index.flush = MagicMock()
        self.retriever = MagicMock()
        self.dedup = MagicMock()
        self.conflict = MagicMock()
        self.forgetting = MagicMock()
        self.importer = MemoryImporter(
            self.store, self.index, self.retriever,
            self.dedup, self.conflict, self.forgetting
        )

    def test_import_basic(self) -> None:
        self.dedup.semantic_dedup.return_value = {"action": "create"}
        conflict_result = MagicMock()
        conflict_result.has_conflict = False
        self.conflict.check.return_value = conflict_result

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "import.json"
            inp.write_text(json.dumps({
                "version": "1.0",
                "exported_at": "2026-01-01T00:00:00Z",
                "count": 1,
                "memories": [{
                    "memory_id": "imp-1",
                    "content": "imported content",
                    "type": "fact",
                    "wing": "personal",
                    "room": "test",
                    "privacy": "personal",
                    "confidence": 4,
                    "created_at": "2026-01-01T00:00:00Z",
                }]
            }), encoding="utf-8")
            result = self.importer.import_json(inp)
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["imported"], 1)
            self.assertEqual(result["skipped"], 0)

    def test_import_skip_duplicates(self) -> None:
        self.dedup.semantic_dedup.return_value = {"action": "skip", "existing_id": "dup-1"}

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "import_dup.json"
            inp.write_text(json.dumps({
                "version": "1.0",
                "memories": [{
                    "memory_id": "dup-entry",
                    "content": "duplicate text",
                    "type": "fact",
                }]
            }), encoding="utf-8")
            result = self.importer.import_json(inp)
            self.assertEqual(result["skipped"], 1)
            self.assertEqual(result["imported"], 0)

    def test_import_skip_duplicates_disabled(self) -> None:
        self.dedup.semantic_dedup.return_value = {"action": "skip"}
        conflict_result = MagicMock()
        conflict_result.has_conflict = False
        self.conflict.check.return_value = conflict_result

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "import_nodup.json"
            inp.write_text(json.dumps({
                "version": "1.0",
                "memories": [{
                    "memory_id": "imp-2",
                    "content": "content here",
                    "type": "fact",
                }]
            }), encoding="utf-8")
            result = self.importer.import_json(inp, skip_duplicates=False)
            self.assertEqual(result["imported"], 1)

    def test_import_empty_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "import_empty.json"
            inp.write_text(json.dumps({
                "version": "1.0",
                "memories": [
                    {"memory_id": "e1", "content": "", "type": "fact"},
                    {"memory_id": "e2", "content": "valid", "type": "fact"},
                ]
            }), encoding="utf-8")
            self.dedup.semantic_dedup.return_value = {"action": "create"}
            conflict_result = MagicMock()
            conflict_result.has_conflict = False
            self.conflict.check.return_value = conflict_result
            result = self.importer.import_json(inp)
            self.assertEqual(result["skipped"], 1)
            self.assertEqual(result["imported"], 1)

    def test_import_conflict_reject(self) -> None:
        self.dedup.semantic_dedup.return_value = {"action": "create"}
        conflict_result = MagicMock()
        conflict_result.has_conflict = True
        self.conflict.check.return_value = conflict_result
        resolution = MagicMock()
        resolution.action = "reject"
        self.conflict.resolve.return_value = resolution

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "import_conflict.json"
            inp.write_text(json.dumps({
                "version": "1.0",
                "memories": [{
                    "memory_id": "c1",
                    "content": "conflicting content",
                    "type": "fact",
                }]
            }), encoding="utf-8")
            result = self.importer.import_json(inp)
            self.assertEqual(result["conflicts"], 1)
            self.assertEqual(result["imported"], 0)

    def test_import_existing_id_collision(self) -> None:
        """已存在的 memory_id 应生成新ID。"""
        self.dedup.semantic_dedup.return_value = {"action": "create"}
        conflict_result = MagicMock()
        conflict_result.has_conflict = False
        self.conflict.check.return_value = conflict_result
        self.store.get.return_value = {"memory_id": "collision"}  # 标记为已存在

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "import_collision.json"
            inp.write_text(json.dumps({
                "version": "1.0",
                "memories": [{
                    "memory_id": "collision",
                    "content": "new content",
                    "type": "fact",
                }]
            }), encoding="utf-8")
            result = self.importer.import_json(inp)
            self.assertEqual(result["imported"], 1)
            # store.add should have been called with a new ID (not "collision")
            call_args = self.store.add.call_args[1]
            self.assertNotEqual(call_args.get("memory_id"), "collision")
