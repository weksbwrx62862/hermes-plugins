import json
from pathlib import Path
from typing import Any


class RBACManager:
    def __init__(self, governance_dir: Path):
        self._rbac_path = governance_dir / "rbac.json"
        self._roles: dict[str, dict] = {}
        self._users: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        if self._rbac_path.exists():
            with open(self._rbac_path, encoding="utf-8") as f:
                data = json.load(f)
            self._roles = data.get("roles", {})
            self._users = data.get("users", {})
        else:
            self._roles = {
                "admin": {"permissions": ["read", "write", "delete", "govern", "export", "import", "audit"]},
                "editor": {"permissions": ["read", "write", "delete", "export"]},
                "viewer": {"permissions": ["read", "export"]},
                "auditor": {"permissions": ["read", "audit"]},
            }
            self._users = {"default": ["editor"]}
            self._save()

    def _save(self) -> None:
        self._rbac_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._rbac_path, "w", encoding="utf-8") as f:
            json.dump({"roles": self._roles, "users": self._users}, f, ensure_ascii=False, indent=2)

    def check_permission(self, user_id: str, permission: str) -> bool:
        role_names = self._users.get(user_id, self._users.get("default", []))
        for role_name in role_names:
            role = self._roles.get(role_name, {})
            if permission in role.get("permissions", []):
                return True
        return False

    def assign_role(self, user_id: str, role_name: str) -> None:
        if user_id not in self._users:
            self._users[user_id] = []
        if role_name not in self._users[user_id]:
            self._users[user_id].append(role_name)
        self._save()

    def revoke_role(self, user_id: str, role_name: str) -> None:
        if user_id in self._users and role_name in self._users[user_id]:
            self._users[user_id].remove(role_name)
        self._save()

    def add_role(self, role_name: str, permissions: list[str]) -> None:
        self._roles[role_name] = {"permissions": permissions}
        self._save()

    def get_user_permissions(self, user_id: str) -> list[str]:
        role_names = self._users.get(user_id, self._users.get("default", []))
        perms = set()
        for rn in role_names:
            perms.update(self._roles.get(rn, {}).get("permissions", []))
        return sorted(perms)
