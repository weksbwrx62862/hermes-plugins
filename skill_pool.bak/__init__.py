"""
SkillPool 插件 — 技能池向量索引 + 语义搜索 + 快照管理

独立部署版本，不依赖 self_evolution。

核心思路：
  基础技能 (core): 常驻 system prompt，不超过 8-10 个
  池中技能 (pool): 按需从向量索引中语义搜索 top-K
  子智能体路由: delegate_task 时同步搜索匹配技能

技术方案：
  - sentence-transformers 本地 embedding（或 n-gram hash fallback）
  - numpy 内存数组 + JSON 元数据（轻量，无需外部 DB）
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np

logger = logging.getLogger("hermes_plugins.skill_pool")


# ═══════════════════════════════════════════════════════════════════
# SkillEntry / SkillPool
# ═══════════════════════════════════════════════════════════════════

class SkillEntry:
    """技能条目"""
    def __init__(self, name: str, description: str, category: str,
                 skill_path: str, tier: str = "pool"):
        self.name = name
        self.description = description
        self.category = category
        self.skill_path = skill_path
        self.tier = tier  # "core" | "pool"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "skill_path": self.skill_path,
            "tier": self.tier,
        }


class SkillPool:
    """
    技能池 — 向量索引 + 语义搜索

    索引文件：
      ~/.hermes/skills/.pool_index.json
      ~/.hermes/skills/.pool_mtimes.json
      ~/.hermes/skills/.pool_config.json
    """

    INDEX_FILE = Path.home() / ".hermes" / "skills" / ".pool_index.json"
    MTIME_FILE = Path.home() / ".hermes" / "skills" / ".pool_mtimes.json"
    CONFIG_FILE = Path.home() / ".hermes" / "skills" / ".pool_config.json"
    SKILLS_DIR = Path.home() / ".hermes" / "skills"

    DEFAULT_CORE = {
        "skill-creator",
        "hermes-agent",
        "web-search-china",
    }

    def __init__(self):
        self._entries: list[SkillEntry] = []
        self._embeddings: Optional[np.ndarray] = None
        self._core_skills: set[str] = set()
        self._loaded = False

    # ── 构建索引 ────────────────────────────────────────────

    def build_index(
        self,
        embed_fn=None,
        model_name: str = "all-MiniLM-L6-v2",
        incremental: bool = True,
    ) -> int:
        """扫描所有 SKILL.md → 生成 embedding → 保存索引。"""
        self._load_config()

        if incremental and self.INDEX_FILE.exists():
            changed = self._incremental_update(embed_fn, model_name)
            if changed >= 0:
                logger.info("[skillpool] incremental: %d skills changed", changed)
                self._save()
                return len(self._entries)

        self._entries = self._scan_skills()
        logger.info("[skillpool] scanned %d skills", len(self._entries))

        if not self._entries:
            return 0

        self._ensure_model_downloaded(model_name)

        texts = [e.description for e in self._entries]
        embeddings = self._compute_embeddings(texts, embed_fn, model_name)
        self._embeddings = np.array(embeddings, dtype=np.float32)
        logger.info("[skillpool] embeddings shape: %s", self._embeddings.shape)

        self._save()
        self._save_mtimes()
        return len(self._entries)

    def load_index(self) -> bool:
        """加载已有索引（JSON + base64）。"""
        old_pkl = self.INDEX_FILE.with_suffix(".pkl")
        if old_pkl.exists() and not self.INDEX_FILE.exists():
            logger.info("[skillpool] migrating from pickle to JSON index")
            self._migrate_from_pickle(old_pkl)

        if not self.INDEX_FILE.exists():
            return False
        try:
            import base64
            with open(self.INDEX_FILE) as f:
                data = json.load(f)
            self._entries = [
                SkillEntry(**e) if isinstance(e, dict) else e
                for e in data.get("entries", [])
            ]
            emb_raw = data.get("embeddings", "")
            if emb_raw:
                emb_bytes = base64.b64decode(emb_raw)
                self._embeddings = np.frombuffer(emb_bytes, dtype=np.float32).reshape(
                    data["shape"][0], data["shape"][1]
                )
            self._load_config()
            self._loaded = True
            return True
        except Exception as e:
            logger.warning("[skillpool] failed to load index: %s", e)
            return False

    # ── 搜索 ─────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        include_core: bool = True,
    ) -> list[SkillEntry]:
        """语义搜索 top-K 匹配技能。"""
        if self._embeddings is None:
            if not self.load_index():
                return []

        query_vec = self._quick_embed(query)

        similarities = np.dot(self._embeddings, query_vec) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )

        indices = np.argsort(similarities)[::-1]
        results = []
        for idx in indices:
            entry = self._entries[idx]
            if not include_core and entry.tier == "core":
                continue
            if similarities[idx] < 0.15:
                continue
            if len(results) >= k:
                break
            results.append(entry)

        return results

    def get_core_skills(self) -> list[SkillEntry]:
        return [e for e in self._entries if e.tier == "core"]

    def get_pool_skills(self) -> list[SkillEntry]:
        return [e for e in self._entries if e.tier == "pool"]

    # ── 分类配置 ────────────────────────────────────────────

    def set_core(self, skill_name: str) -> None:
        self._core_skills.add(skill_name)
        self._update_tiers()
        self._save_config()

    def set_pool(self, skill_name: str) -> None:
        self._core_skills.discard(skill_name)
        self._update_tiers()
        self._save_config()

    def auto_tune_core(
        self,
        top_n: int = 8,
        min_usage: int = 5,
        pinned: set[str] = None,
    ) -> int:
        """根据 .usage.json 实际使用数据自动调整常驻列表。"""
        pinned = pinned or {"skill-creator", "hermes-agent"}

        usage = self._read_usage()
        if not usage:
            return len(self._core_skills)

        scored = []
        for name, data in usage.items():
            activity = data.get("use_count", 0) + data.get("view_count", 0)
            if activity >= min_usage:
                scored.append((name, activity))

        scored.sort(key=lambda x: (-x[1], x[0]))

        new_core = set(pinned)
        for name, _ in scored:
            if len(new_core) >= top_n:
                break
            new_core.add(name)

        changed = new_core != self._core_skills
        self._core_skills = new_core
        self._update_tiers()
        self._save_config()

        if changed:
            logger.info("[skillpool] auto-tuned core: %d skills", len(new_core))
        return len(new_core)

    def get_auto_tune_status(self) -> dict:
        return {
            "core_skills": sorted(self._core_skills),
            "core_count": len(self._core_skills),
            "pinned": ["skill-creator", "hermes-agent"],
            "usage_stats": self._read_usage_summary(),
        }

    def _read_usage(self) -> dict:
        path = self.SKILLS_DIR / ".usage.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _read_usage_summary(self) -> list[dict]:
        usage = self._read_usage()
        scored = []
        for name, data in usage.items():
            activity = data.get("use_count", 0) + data.get("view_count", 0)
            scored.append({"name": name, "activity": activity,
                          "tier": self.get_tier(name)})
        scored.sort(key=lambda x: -x["activity"])
        return scored[:15]

    def get_tier(self, skill_name: str) -> str:
        return "core" if skill_name in self._core_skills else "pool"

    # ── Hermes 集成：写入快照 ─────────────────────────────

    def write_hermes_snapshot(self) -> int:
        """生成只含 core 技能的 Hermes 快照文件。"""
        snapshot_path = self.SKILLS_DIR / ".skills_prompt_snapshot.json"
        if not snapshot_path.exists():
            logger.warning("[skillpool] no existing snapshot to modify")
            return 0

        try:
            data = json.loads(snapshot_path.read_text())
        except (json.JSONDecodeError, OSError):
            return 0

        entries = data.get("skill_entries", data.get("entries", []))
        filtered = []
        removed = 0
        for entry in entries:
            name = entry.get("skill_name") or entry.get("frontmatter_name", "")
            if name in self._core_skills:
                filtered.append(entry)
            else:
                removed += 1

        data["skill_entries"] = filtered
        data["_filtered_by_skillpool"] = True
        data["_core_skills"] = sorted(self._core_skills)
        data["_filtered_count"] = len(filtered)
        data["_removed_count"] = removed

        snapshot_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("[skillpool] snapshot: %d core / %d removed (was %d)",
                    len(filtered), removed, len(entries))
        return len(filtered)

    # ── 摘要生成 ────────────────────────────────────────────

    def build_available_skills_block(
        self,
        query: Optional[str] = None,
        k_pool: int = 8,
    ) -> str:
        """生成 <available_skills> 块。"""
        lines = ["<available_skills>"]

        core = self.get_core_skills()
        if core:
            lines.append("  core:")
            for e in core:
                lines.append(f"    - {e.name}: {e.description}")

        if query:
            matched = self.search(query, k=k_pool, include_core=False)
            if matched:
                lines.append(f"  matched (top-{len(matched)}):")
                for e in matched:
                    lines.append(f"    - {e.name}: {e.description}")

        lines.append("</available_skills>")
        return "\n".join(lines)

    # ── 内部 ─────────────────────────────────────────────────

    def _scan_skills(self) -> list[SkillEntry]:
        entries = []
        for skill_md in sorted(self.SKILLS_DIR.rglob("SKILL.md")):
            try:
                rel = skill_md.relative_to(self.SKILLS_DIR)
                if rel.parts and rel.parts[0].startswith("."):
                    continue
                content = skill_md.read_text(encoding="utf-8")[:3000]
                fm, _ = self._parse_frontmatter(content)
                name = fm.get("name", skill_md.parent.name)
                desc = fm.get("description", "")
                category = str(rel.parent) if len(rel.parts) > 1 else "root"
                entries.append(SkillEntry(
                    name=name, description=desc, category=category,
                    skill_path=str(skill_md), tier=self._classify(name),
                ))
            except Exception:
                continue
        return entries

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict, str]:
        fm = {}
        body = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                for line in parts[1].strip().split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        fm[k.strip()] = v.strip().strip("\"'")
                body = parts[2]
        return fm, body

    def _classify(self, name: str) -> str:
        return "core" if name in self._core_skills else "pool"

    def _update_tiers(self):
        for e in self._entries:
            e.tier = self._classify(e.name)

    def _compute_embeddings(
        self, texts: list[str], embed_fn=None, model_name: str = "all-MiniLM-L6-v2"
    ) -> list[list[float]]:
        """计算文本 embedding（GFW 友好：15s 线程超时 fallback）。"""
        if embed_fn:
            return [embed_fn(t) for t in texts]

        import threading

        result = [None]
        error = [None]

        def _load_and_encode():
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name, trust_remote_code=True)
                result[0] = model.encode(texts, show_progress_bar=False).tolist()
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=_load_and_encode, daemon=True)
        t.start()
        t.join(timeout=15)

        if result[0] is not None:
            return result[0]

        if t.is_alive():
            logger.warning(
                "[skillpool] sentence-transformers timed out (GFW?), "
                "using n-gram fallback embedder"
            )

        if error[0]:
            logger.warning(
                "[skillpool] sentence-transformers error: %s, using fallback",
                error[0],
            )

        return [self._quick_embed(t).tolist() for t in texts]

    # ── 增量更新 ─────────────────────────────────────────────

    def _load_mtimes(self) -> dict[str, float]:
        if self.MTIME_FILE.exists():
            try:
                return json.loads(self.MTIME_FILE.read_text())
            except Exception:
                pass
        return {}

    def _save_mtimes(self) -> None:
        mtimes = {}
        for e in self._entries:
            p = Path(e.skill_path)
            if p.exists():
                mtimes[e.name] = p.stat().st_mtime
        with open(self.MTIME_FILE, "w") as f:
            json.dump(mtimes, f, indent=2)

    def _incremental_update(self, embed_fn, model_name: str) -> int:
        """增量更新：只对有变化的 SKILL.md 重新 embedding。返回变化数，-1=失败。"""
        try:
            if not self.load_index():
                return -1
        except Exception:
            return -1

        old_mtimes = self._load_mtimes()
        current_files = {}
        for skill_md in sorted(self.SKILLS_DIR.rglob("SKILL.md")):
            try:
                rel = skill_md.relative_to(self.SKILLS_DIR)
                if rel.parts and rel.parts[0].startswith("."):
                    continue
                fm, _ = self._parse_frontmatter(skill_md.read_text(encoding="utf-8")[:3000])
                name = fm.get("name", skill_md.parent.name)
                current_files[name] = skill_md
            except Exception:
                continue

        changed_names = set()
        for name, path in current_files.items():
            old_mtime = old_mtimes.get(name, 0)
            new_mtime = path.stat().st_mtime
            if abs(new_mtime - old_mtime) > 1.0:
                changed_names.add(name)

        deleted_names = set(old_mtimes.keys()) - set(current_files.keys())
        new_names = set(current_files.keys()) - {e.name for e in self._entries}

        if not changed_names and not deleted_names and not new_names:
            return 0

        # 删除旧条目
        if deleted_names:
            self._entries = [e for e in self._entries if e.name not in deleted_names]
            if self._embeddings is not None and len(self._entries) > 0:
                keep_idx = [i for i, e in enumerate(self._entries) if e.name not in deleted_names]
                self._embeddings = self._embeddings[keep_idx] if keep_idx else None

        # 重建变化和新增条目
        to_update = changed_names | new_names
        update_texts = []
        update_entries = []
        for name in to_update:
            path = current_files[name]
            try:
                content = path.read_text(encoding="utf-8")[:3000]
                fm, _ = self._parse_frontmatter(content)
                desc = fm.get("description", "")
                rel = path.relative_to(self.SKILLS_DIR)
                category = str(rel.parent) if len(rel.parts) > 1 else "root"
                entry = SkillEntry(
                    name=name, description=desc, category=category,
                    skill_path=str(path), tier=self._classify(name),
                )
                # 替换或追加
                replaced = False
                for i, e in enumerate(self._entries):
                    if e.name == name:
                        self._entries[i] = entry
                        replaced = True
                        break
                if not replaced:
                    self._entries.append(entry)
                update_texts.append(desc)
                update_entries.append(entry)
            except Exception:
                continue

        if update_texts:
            new_embs = self._compute_embeddings(update_texts, embed_fn, model_name)
            new_embs = np.array(new_embs, dtype=np.float32)

            if self._embeddings is not None and self._embeddings.shape[0] == len(self._entries) - len(update_entries):
                # 增量追加
                self._embeddings = np.vstack([self._embeddings, new_embs])
            else:
                # 全量重算
                all_texts = [e.description for e in self._entries]
                all_embs = self._compute_embeddings(all_texts, embed_fn, model_name)
                self._embeddings = np.array(all_embs, dtype=np.float32)

        self._save_mtimes()
        return len(changed_names) + len(deleted_names) + len(new_names)

    # ── 快速 embedding（fallback） ──────────────────────────

    @staticmethod
    def _quick_embed(text: str, dim: int = 384) -> np.ndarray:
        """n-gram hash 快速 embedding（无需模型，用于 fallback）。"""
        vec = np.zeros(dim, dtype=np.float32)
        text = text.lower()
        for n in (2, 3):
            for i in range(len(text) - n + 1):
                h = hash(text[i:i+n]) % dim
                vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ── 模型预下载 ─────────────────────────────────────────

    def _ensure_model_downloaded(self, model_name: str) -> None:
        """预下载 sentence-transformers 模型（GFW 友好：15s 超时）。"""
        import threading

        def _try_download():
            try:
                from sentence_transformers import SentenceTransformer
                SentenceTransformer(model_name, trust_remote_code=True)
            except Exception:
                pass

        t = threading.Thread(target=_try_download, daemon=True)
        t.start()
        t.join(timeout=15)

        if t.is_alive():
            logger.warning(
                "[skillpool] model download timed out, will use fallback"
            )

    # ── 序列化 ──────────────────────────────────────────────

    def _save(self) -> None:
        import base64
        data = {
            "version": 2,
            "entries": [e.to_dict() for e in self._entries],
            "shape": list(self._embeddings.shape) if self._embeddings is not None else [0, 0],
            "embeddings": base64.b64encode(self._embeddings.tobytes()).decode() if self._embeddings is not None else "",
        }
        self.INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.INDEX_FILE, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_config(self) -> None:
        if self.CONFIG_FILE.exists():
            try:
                cfg = json.loads(self.CONFIG_FILE.read_text())
                self._core_skills = set(cfg.get("core_skills", self.DEFAULT_CORE))
            except Exception:
                self._core_skills = set(self.DEFAULT_CORE)
        else:
            self._core_skills = set(self.DEFAULT_CORE)

    def _save_config(self) -> None:
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CONFIG_FILE, "w") as f:
            json.dump({"core_skills": sorted(self._core_skills)}, f, indent=2)

    def _migrate_from_pickle(self, old_pkl: Path) -> None:
        """从旧 pickle 格式迁移到 JSON。"""
        try:
            import pickle
            with open(old_pkl, "rb") as f:
                data = pickle.load(f)
            # 迁移后保存为 JSON
            if isinstance(data, dict):
                self._entries = data.get("entries", [])
                self._embeddings = data.get("embeddings")
            old_pkl.rename(old_pkl.with_suffix(".pkl.bak"))
            self._save()
            logger.info("[skillpool] migrated from pickle, backup: %s", old_pkl.with_suffix(".pkl.bak"))
        except Exception as e:
            logger.warning("[skillpool] pickle migration failed: %s", e)


# ═══════════════════════════════════════════════════════════════════
# 全局实例 & 工具注册
# ═══════════════════════════════════════════════════════════════════

_pool_instance: Optional[SkillPool] = None


def _get_pool() -> SkillPool:
    global _pool_instance
    if _pool_instance is None:
        _pool_instance = SkillPool()
    return _pool_instance


def _tool_build_index(args: Dict[str, Any], **kwargs) -> str:
    """构建/重建技能池索引。"""
    pool = _get_pool()
    incremental = args.get("incremental", True)
    count = pool.build_index(incremental=incremental)
    return json.dumps({
        "status": "ok",
        "indexed_skills": count,
        "incremental": incremental,
    }, ensure_ascii=False)


def _tool_search_skills(args: Dict[str, Any], **kwargs) -> str:
    """语义搜索技能。"""
    pool = _get_pool()
    query = args.get("query", "")
    k = args.get("k", 5)
    results = pool.search(query, k=k)
    return json.dumps({
        "query": query,
        "results": [
            {"name": e.name, "description": e.description, "tier": e.tier}
            for e in results
        ],
    }, ensure_ascii=False)


def _tool_list_skills(args: Dict[str, Any], **kwargs) -> str:
    """列出技能池中的所有技能。"""
    pool = _get_pool()
    if not pool.load_index():
        # 如果没有索引，直接扫描
        count = pool.build_index(incremental=False)

    tier_filter = args.get("tier", "all")  # core / pool / all
    if tier_filter == "core":
        skills = pool.get_core_skills()
    elif tier_filter == "pool":
        skills = pool.get_pool_skills()
    else:
        skills = pool._entries

    return json.dumps({
        "total": len(skills),
        "tier_filter": tier_filter,
        "skills": [
            {"name": e.name, "category": e.category, "tier": e.tier}
            for e in sorted(skills, key=lambda x: x.name)
        ],
    }, ensure_ascii=False)


def _tool_set_core(args: Dict[str, Any], **kwargs) -> str:
    """设置技能为常驻（core）。"""
    pool = _get_pool()
    name = args.get("name", "")
    if not name:
        return json.dumps({"error": "name is required"})
    pool.set_core(name)
    return json.dumps({"status": "ok", "name": name, "tier": "core"})


def _tool_set_pool(args: Dict[str, Any], **kwargs) -> str:
    """设置技能为按需（pool）。"""
    pool = _get_pool()
    name = args.get("name", "")
    if not name:
        return json.dumps({"error": "name is required"})
    pool.set_pool(name)
    return json.dumps({"status": "ok", "name": name, "tier": "pool"})


def _tool_auto_tune(args: Dict[str, Any], **kwargs) -> str:
    """根据使用数据自动调整常驻技能。"""
    pool = _get_pool()
    top_n = args.get("top_n", 8)
    count = pool.auto_tune_core(top_n=top_n)
    status = pool.get_auto_tune_status()
    return json.dumps({
        "status": "ok",
        "core_count": count,
        **status,
    }, ensure_ascii=False)


def _tool_snapshot(args: Dict[str, Any], **kwargs) -> str:
    """生成只含 core 技能的快照（影响下次 session 的技能列表）。"""
    pool = _get_pool()
    count = pool.write_hermes_snapshot()
    return json.dumps({
        "status": "ok",
        "core_skills_in_snapshot": count,
    }, ensure_ascii=False)


def _tool_skill_usage(args: Dict[str, Any], **kwargs) -> str:
    """查看技能使用统计。"""
    pool = _get_pool()
    return json.dumps(pool.get_auto_tune_status(), ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════
# Hermes 插件注册
# ═══════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "skill_pool_build",
        "description": "构建/重建技能池索引。扫描所有 SKILL.md 并生成向量索引。支持增量更新（仅处理变化的文件）。",
        "parameters": {
            "type": "object",
            "properties": {
                "incremental": {
                    "type": "boolean",
                    "description": "是否增量更新（默认 true）",
                    "default": True,
                },
            },
        },
        "handler": _tool_build_index,
    },
    {
        "name": "skill_pool_search",
        "description": "语义搜索技能池。输入自然语言查询，返回最匹配的技能列表。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "k": {"type": "integer", "description": "返回数量（默认 5）", "default": 5},
            },
            "required": ["query"],
        },
        "handler": _tool_search_skills,
    },
    {
        "name": "skill_pool_list",
        "description": "列出技能池中的所有技能。可按 tier（core/pool/all）筛选。",
        "parameters": {
            "type": "object",
            "properties": {
                "tier": {
                    "type": "string",
                    "enum": ["core", "pool", "all"],
                    "description": "筛选层级（默认 all）",
                    "default": "all",
                },
            },
        },
        "handler": _tool_list_skills,
    },
    {
        "name": "skill_pool_set_core",
        "description": "将技能设置为常驻（core），每次 session 都会自动注入 system prompt。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名称"},
            },
            "required": ["name"],
        },
        "handler": _tool_set_core,
    },
    {
        "name": "skill_pool_set_pool",
        "description": "将技能设置为按需（pool），只在语义搜索匹配时加载。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名称"},
            },
            "required": ["name"],
        },
        "handler": _tool_set_pool,
    },
    {
        "name": "skill_pool_auto_tune",
        "description": "根据使用数据自动调整常驻技能列表。使用频次高的技能自动提升为 core。",
        "parameters": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "常驻技能上限（默认 8）",
                    "default": 8,
                },
            },
        },
        "handler": _tool_auto_tune,
    },
    {
        "name": "skill_pool_snapshot",
        "description": "生成技能快照。只保留 core 技能在 system prompt 中，pool 技能需通过 skill_view 按需加载。",
        "parameters": {
            "type": "object",
            "properties": {},
        },
        "handler": _tool_snapshot,
    },
    {
        "name": "skill_pool_usage",
        "description": "查看技能使用统计和当前 core/pool 分类。",
        "parameters": {
            "type": "object",
            "properties": {},
        },
        "handler": _tool_skill_usage,
    },
]


def register(ctx) -> None:
    """Hermes 插件入口：注册所有工具。"""
    registered = 0
    for tool_def in TOOLS:
        name = tool_def["name"]
        handler = tool_def["handler"]
        description = tool_def.get("description", "")
        schema = tool_def.get("parameters", {})
        try:
            ctx.register_tool(
                name=name,
                handler=handler,
                schema=schema,
                toolset="skill_pool",
            )
            registered += 1
        except Exception as exc:
            logger.warning("SkillPool: failed to register tool %s: %s", name, exc)

    logger.info(
        "SkillPool v1.0 registered: %d tools, skills_dir=%s",
        registered, SkillPool.SKILLS_DIR,
    )


__all__ = ["SkillPool", "SkillEntry", "register", "TOOLS"]
