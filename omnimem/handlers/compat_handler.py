from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from omnimem.utils.security import SecurityValidator


class CompatHandler:
    def __init__(
        self,
        memorize_fn: Callable[[dict[str, Any]], str],
        store: Any,
        forgetting: Any,
        extract_core_fact_fn: Callable[[str], str],
    ) -> None:
        self._memorize_fn = memorize_fn
        self._store = store
        self._forgetting = forgetting
        self._extract_core_fact = extract_core_fact_fn

    def handle(self, args: dict[str, Any]) -> str:
        action = args.get("action", "")
        target = args.get("target", "memory")
        content = args.get("content", "").strip()
        old_text = args.get("old_text", "").strip()

        if action not in ("add", "replace", "remove"):
            return json.dumps({"error": f"Unknown action '{action}'. Use: add, replace, remove"})

        if action in ("add", "replace") and not content:
            return json.dumps({"error": "Content is required for 'add' and 'replace'."})

        if action in ("replace", "remove") and not old_text:
            return json.dumps({"error": "old_text is required for 'replace' and 'remove'."})

        if content:
            scan_error = SecurityValidator.scan_threats(content)
            if scan_error:
                return json.dumps({"success": False, "error": scan_error})

        mem_type = "preference" if target == "user" else "fact"

        if content and len(content) > 100:
            refined = self._extract_core_fact(content)
            if refined and len(refined) < len(content):
                content = refined

        if action == "add":
            return self._compat_set(content, mem_type)
        elif action == "replace":
            return self._compat_get(content, old_text, mem_type, target)
        elif action == "remove":
            return self._compat_delete(old_text, mem_type, target)

        return json.dumps({"error": "Unreachable"})

    def _compat_set(self, content: str, mem_type: str) -> str:
        result = self._memorize_fn(
            {
                "content": content,
                "memory_type": mem_type,
                "confidence": 4,
                "scope": "personal",
                "privacy": "personal",
            }
        )
        parsed = json.loads(result)
        parsed["compat_note"] = "Routed from builtin 'memory' tool to OmniMem"
        return json.dumps(parsed)

    def _compat_get(self, content: str, old_text: str, mem_type: str, target: str) -> str:
        matches = self._store.search_by_content(old_text, limit=10)
        filtered = [m for m in matches if m.get("type") == mem_type]

        if not filtered:
            return json.dumps(
                {
                    "success": False,
                    "error": f"No matching {target} entry found for '{old_text[:50]}'.",
                }
            )

        if len(filtered) > 1:
            previews = [m.get("content", "")[:60] for m in filtered[:5]]
            return json.dumps(
                {
                    "success": False,
                    "error": "Multiple entries matched. Be more specific.",
                    "matches": previews,
                }
            )

        old_id = filtered[0]["memory_id"]
        self._forgetting.archive(old_id)

        result = self._memorize_fn(
            {
                "content": content,
                "memory_type": mem_type,
                "confidence": 4,
                "scope": "personal",
                "privacy": "personal",
            }
        )
        parsed = json.loads(result)
        parsed["replaced_id"] = old_id
        parsed["compat_note"] = "Replaced via builtin compat layer"
        return json.dumps(parsed)

    def _compat_delete(self, old_text: str, mem_type: str, target: str) -> str:
        matches = self._store.search_by_content(old_text, limit=10)
        filtered = [m for m in matches if m.get("type") == mem_type]

        if not filtered:
            return json.dumps({"success": False, "error": f"No matching {target} entry found."})

        if len(filtered) > 1:
            previews = [m.get("content", "")[:60] for m in filtered[:5]]
            return json.dumps(
                {
                    "success": False,
                    "error": "Multiple entries matched. Be more specific.",
                    "matches": previews,
                }
            )

        old_id = filtered[0]["memory_id"]
        self._forgetting.archive(old_id)

        return json.dumps(
            {
                "success": True,
                "action": "archived",
                "memory_id": old_id,
                "message": f"{target} entry archived (soft delete).",
                "compat_note": "Removed via builtin compat layer (uses forgetting curve)",
            }
        )
