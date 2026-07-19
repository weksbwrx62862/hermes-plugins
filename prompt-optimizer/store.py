"""prompt-optimizer 插件 — Prompt Garden 资产存储引擎。

SQLite-backed，支持版本历史、标签、全文搜索。
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("plugins.prompt-optimizer.store")

DB_PATH = Path.home() / ".hermes" / "prompt_garden.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """初始化数据库表（幂等）。"""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'user',
            description TEXT DEFAULT '',
            prompt TEXT NOT NULL,
            tags TEXT DEFAULT '',
            prompt_hash TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(name, mode)
        );

        CREATE TABLE IF NOT EXISTS prompt_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'user',
            version INTEGER NOT NULL,
            prompt TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_prompts_name ON prompts(name);
        CREATE INDEX IF NOT EXISTS idx_prompts_tags ON prompts(tags);
        CREATE INDEX IF NOT EXISTS idx_history_name ON prompt_history(name, mode);
    """)
    conn.close()
    logger.info("Prompt Garden 数据库已初始化: %s", DB_PATH)


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


class PromptStore:
    """Prompt Garden 资产管理器。"""

    def save(self, name: str, prompt: str, mode: str = "user",
             description: str = "", tags: str = "") -> Dict[str, Any]:
        """保存提示词，自动版本管理。"""
        conn = _connect()
        now = time.time()
        ph = _hash(prompt)

        existing = conn.execute(
            "SELECT version, prompt_hash, prompt FROM prompts WHERE name=? AND mode=?",
            (name, mode)
        ).fetchone()

        if existing:
            if existing["prompt_hash"] == ph:
                conn.close()
                return {"status": "unchanged", "name": name, "mode": mode,
                        "version": existing["version"],
                        "message": "内容未变化，跳过保存"}
            new_ver = existing["version"] + 1
            # 存历史（旧版本）
            conn.execute(
                "INSERT INTO prompt_history (name, mode, version, prompt, prompt_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, mode, existing["version"], existing["prompt"], existing["prompt_hash"], now)
            )
            conn.execute(
                "UPDATE prompts SET prompt=?, description=?, tags=?, prompt_hash=?, "
                "version=?, updated_at=? WHERE name=? AND mode=?",
                (prompt, description, tags, ph, new_ver, now, name, mode)
            )
        else:
            new_ver = 1
            conn.execute(
                "INSERT INTO prompts (name, mode, description, prompt, tags, prompt_hash, "
                "version, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, mode, description, prompt, tags, ph, new_ver, now, now)
            )

        conn.commit()
        conn.close()
        return {"status": "saved", "name": name, "mode": mode,
                "version": new_ver, "hash": ph}

    def list_all(self, tag: Optional[str] = None, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有保存的提示词。"""
        conn = _connect()
        sql = "SELECT name, mode, description, tags, version, prompt_hash, created_at, updated_at FROM prompts"
        params: list = []
        wheres = []
        if tag:
            wheres.append("tags LIKE ?")
            params.append(f"%{tag}%")
        if mode:
            wheres.append("mode=?")
            params.append(mode)
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        sql += " ORDER BY updated_at DESC"

        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get(self, name: str, mode: str = "user") -> Optional[Dict[str, Any]]:
        """获取指定提示词详情。"""
        conn = _connect()
        row = conn.execute(
            "SELECT * FROM prompts WHERE name=? AND mode=?", (name, mode)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def search(self, query: str) -> List[Dict[str, Any]]:
        """搜索提示词（名称 + 内容 + 描述 + 标签）。"""
        conn = _connect()
        like = f"%{query}%"
        rows = conn.execute(
            "SELECT name, mode, description, tags, version, created_at, updated_at FROM prompts "
            "WHERE name LIKE ? OR prompt LIKE ? OR description LIKE ? OR tags LIKE ? "
            "ORDER BY updated_at DESC LIMIT 20",
            (like, like, like, like)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete(self, name: str, mode: str = "user") -> bool:
        """删除提示词及其历史。"""
        conn = _connect()
        cur = conn.execute("DELETE FROM prompts WHERE name=? AND mode=?", (name, mode))
        conn.execute("DELETE FROM prompt_history WHERE name=? AND mode=?", (name, mode))
        conn.commit()
        deleted = cur.rowcount > 0
        conn.close()
        return deleted

    def history(self, name: str, mode: str = "user") -> List[Dict[str, Any]]:
        """获取版本历史。"""
        conn = _connect()
        # 当前版本
        current = conn.execute(
            "SELECT version, prompt, prompt_hash, updated_at FROM prompts WHERE name=? AND mode=?",
            (name, mode)
        ).fetchone()
        # 历史版本
        rows = conn.execute(
            "SELECT version, prompt, prompt_hash, created_at FROM prompt_history "
            "WHERE name=? AND mode=? ORDER BY version DESC",
            (name, mode)
        ).fetchall()
        conn.close()

        result = []
        if current:
            result.append({
                "version": current["version"], "prompt": current["prompt"],
                "hash": current["prompt_hash"], "timestamp": current["updated_at"],
                "is_current": True
            })
        for r in rows:
            result.append({
                "version": r["version"], "prompt": r["prompt"],
                "hash": r["prompt_hash"], "timestamp": r["created_at"],
                "is_current": False
            })
        return result

    def export_all(self) -> str:
        """导出全部为 JSON。"""
        conn = _connect()
        rows = conn.execute("SELECT * FROM prompts ORDER BY name").fetchall()
        conn.close()
        data = [dict(r) for r in rows]
        return json.dumps(data, ensure_ascii=False, indent=2)
