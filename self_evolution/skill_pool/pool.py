"""
技能池管理器 — 向量索引 + 语义搜索 + 按需路由

核心思路：
  基础技能 (core): 常驻 system prompt，不超过 8-10 个
  池中技能 (pool): 按需从向量索引中语义搜索 top-K
  子智能体路由: delegate_task 时同步搜索匹配技能

技术方案：
  - 用 sentence-transformers 做本地 embedding（无需 API 调用）
  - 或用 OpenAI compatible API 做 embedding
  - 向量存储：numpy 内存数组 + JSON 元数据（轻量，无需外部 DB）

工作流：
  1. build_index() → 扫描所有 SKILL.md → 生成 embedding → 存索引
  2. search(query, k=5) → embedding 查询 → 返回 top-K 匹配技能
  3. classify() → 判断技能是 core 还是 pool
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SkillEntry:
    """技能条目"""
    def __init__(self, name: str, description: str, category: str,
                 skill_path: str, tier: str = "pool"):
        self.name = name
        self.description = description
        self.category = category
        self.skill_path = skill_path
        self.tier = tier  # "core" | "pool"


class SkillPool:
    """
    技能池 — 向量索引 + 语义搜索

    索引文件：
      ~/.hermes/skills/.pool_index.pkl  (pickle 序列化)
      ~/.hermes/skills/.pool_config.json (core/pool 分类配置)
    """

    INDEX_FILE = Path.home() / ".hermes" / "skills" / ".pool_index.json"  # JSON（可检视）
    MTIME_FILE = Path.home() / ".hermes" / "skills" / ".pool_mtimes.json"
    CONFIG_FILE = Path.home() / ".hermes" / "skills" / ".pool_config.json"
    SKILLS_DIR = Path.home() / ".hermes" / "skills"

    # 默认基础技能（始终注入 system prompt）
    # 当 auto_tune 开启时，DEFAULT_CORE 只是初始种子，会被使用数据覆盖
    DEFAULT_CORE = {
        "skill-creator",
        "hermes-agent",
        "web-search-china",
    }

    def __init__(self):
        self._entries: list[SkillEntry] = []
        self._embeddings: Optional[np.ndarray] = None  # (N, dim)
        self._core_skills: set[str] = set()
        self._loaded = False

    # ── 构建索引 ────────────────────────────────────────────

    def build_index(
        self,
        embed_fn=None,  # callable(text) -> list[float]
        model_name: str = "all-MiniLM-L6-v2",  # 默认轻量模型
        incremental: bool = True,  # 是否增量更新
    ) -> int:
        """
        扫描所有 SKILL.md → 生成 embedding → 保存索引。

        Args:
            embed_fn: 自定义 embedding 函数。None 则用 sentence-transformers
            model_name: 本地模型名
            incremental: 是否仅更新变化的文件（默认 True）
        Returns: 索引条目数
        """
        # 1. 加载配置
        self._load_config()

        # 2. 尝试增量更新
        if incremental and self.INDEX_FILE.exists():
            changed = self._incremental_update(embed_fn, model_name)
            if changed >= 0:  # 增量成功
                logger.info(f"[skillpool] incremental: {changed} skills changed")
                self._save()
                return len(self._entries)

        # 3. 全量重建
        self._entries = self._scan_skills()
        logger.info(f"[skillpool] scanned {len(self._entries)} skills")

        if not self._entries:
            return 0

        # 4. 模型预下载（GFW 友好：首次运行前保证模型已下载）
        self._ensure_model_downloaded(model_name)

        # 5. 生成 embedding
        texts = [e.description for e in self._entries]
        embeddings = self._compute_embeddings(texts, embed_fn, model_name)
        self._embeddings = np.array(embeddings, dtype=np.float32)
        logger.info(f"[skillpool] embeddings shape: {self._embeddings.shape}")

        # 6. 保存 + 记录 mtime
        self._save()
        self._save_mtimes()
        return len(self._entries)

    def load_index(self) -> bool:
        """加载已有索引（JSON + base64）。"""
        # 兼容旧 pickle 格式
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
            logger.warning(f"[skillpool] failed to load index: {e}")
            return False

    # ── 搜索 ─────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        include_core: bool = True,
    ) -> list[SkillEntry]:
        """
        语义搜索 top-K 匹配技能。

        Args:
            query: 用户消息/任务描述
            k: 返回数量
            include_core: 是否包含基础技能
        Returns: 按相似度排序的技能列表
        """
        if self._embeddings is None:
            if not self.load_index():
                return []

        # 计算查询 embedding（简单版：词袋 + TF-IDF 近似）
        # 实际使用时传入 embed_fn
        query_vec = self._quick_embed(query)

        # 余弦相似度
        similarities = np.dot(self._embeddings, query_vec) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )

        # 排序取 top-K
        indices = np.argsort(similarities)[::-1]
        results = []
        for idx in indices:
            entry = self._entries[idx]
            if not include_core and entry.tier == "core":
                continue
            # 只返回匹配度 > 阈值的
            if similarities[idx] < 0.15:
                continue
            if len(results) >= k:
                break
            results.append(entry)

        return results

    def get_core_skills(self) -> list[SkillEntry]:
        """获取所有基础技能（始终注入）。"""
        return [e for e in self._entries if e.tier == "core"]

    def get_pool_skills(self) -> list[SkillEntry]:
        """获取所有池中技能。"""
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
        """
        根据 .usage.json 实际使用数据自动调整常驻列表。

        规则：
          1. 从 .usage.json 读取所有 skill 使用次数
          2. 按 (use_count + view_count) 降序排列
          3. 取 top_n，且使用次数 ≥ min_usage
          4. pinned 技能始终保留（即使使用次数低）
          5. 保存配置，下次启动生效

        Returns: 新的常驻技能数
        """
        pinned = pinned or {"skill-creator", "hermes-agent"}

        # 读真实使用数据
        usage = self._read_usage()
        if not usage:
            return len(self._core_skills)

        # 按活动量排序
        scored = []
        for name, data in usage.items():
            activity = data.get("use_count", 0) + data.get("view_count", 0)
            if activity >= min_usage:
                scored.append((name, activity))

        scored.sort(key=lambda x: (-x[1], x[0]))

        # 新常驻：top_n + pinned
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
            logger.info(f"[skillpool] auto-tuned core: {len(new_core)} skills "
                        f"(was {len(self._core_skills) if changed else len(new_core)}), "
                        f"added: {', '.join(sorted(new_core - self._core_skills))}")
        return len(new_core)

    def get_auto_tune_status(self) -> dict:
        """返回当前自动调整状态。"""
        return {
            "core_skills": sorted(self._core_skills),
            "core_count": len(self._core_skills),
            "pinned": ["skill-creator", "hermes-agent"],
            "usage_stats": self._read_usage_summary(),
        }

    def _read_usage(self) -> dict:
        """读 Hermes .usage.json。"""
        path = self.SKILLS_DIR / ".usage.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _read_usage_summary(self) -> list[dict]:
        """返回使用数据摘要（按活动量排序，top-15）。"""
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
        """
        生成只含 core 技能的 Hermes 快照文件。

        Hermes 启动时从 .skills_prompt_snapshot.json 读取技能列表。
        覆盖此文件后，下次 session 自动只看到 core 技能。
        pool 技能仍然存在，agent 可通过 skill_view() 按需加载。

        Returns: 写入的技能数
        """
        snapshot_path = self.SKILLS_DIR / ".skills_prompt_snapshot.json"
        if not snapshot_path.exists():
            logger.warning("[skillpool] no existing snapshot to modify")
            return 0

        try:
            data = json.loads(snapshot_path.read_text())
        except (json.JSONDecodeError, OSError):
            return 0

        # 过滤：只保留 core 技能
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
        logger.info(f"[skillpool] snapshot: {len(filtered)} core / {removed} removed "
                    f"(was {len(entries)})")
        return len(filtered)

    # ── 摘要生成 ────────────────────────────────────────────

    def build_available_skills_block(
        self,
        query: Optional[str] = None,
        k_pool: int = 8,
    ) -> str:
        """
        生成 <available_skills> 块。

        包含：
          - 所有 core 技能
          - top-K pool 技能（如果提供 query）
          - 如果无 query，仅 core

        格式与 Hermes system prompt 兼容。
        """
        lines = ["<available_skills>"]

        # Core 技能
        core = self.get_core_skills()
        if core:
            lines.append("  core:")
            for e in core:
                lines.append(f"    - {e.name}: {e.description}")

        # Pool 技能（按需）
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
        """扫描所有 SKILL.md 文件。"""
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
        """简单 YAML frontmatter 解析。"""
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

        # 使用线程超时：sentence-transformers 下载可能在子线程，signal 无效
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
                f"[skillpool] sentence-transformers timed out (GFW?), "
                f"using n-gram fallback embedder"
            )
            # 不 kill 线程，让它自然完成（daemon 会在进程退出时清理）

        if error[0]:
            logger.warning(
                f"[skillpool] sentence-transformers error: {error[0]}, "
                f"using fallback"
            )

        # fallback: n-gram hash
        return [self._quick_embed(t).tolist() for t in texts]

    # ── 增量更新 ─────────────────────────────────────────────

    def _load_mtimes(self) -> dict[str, float]:
        """加载 mtime 映射。"""
        if self.MTIME_FILE.exists():
            try:
                return json.loads(self.MTIME_FILE.read_text())
            except Exception:
                pass
        return {}

    def _save_mtimes(self) -> None:
        """保存当前技能的 mtime 映射。"""
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
        new_entries = self._scan_skills()
        new_by_name = {e.name: e for e in new_entries}

        changed = 0
        for e in new_entries:
            p = Path(e.skill_path)
            current_mtime = p.stat().st_mtime if p.exists() else 0
            old_mtime = old_mtimes.get(e.name, 0)

            if abs(current_mtime - old_mtime) > 1.0:  # 容忍1秒误差
                # 重新计算这个 skill 的 embedding
                vec = self._compute_embeddings([e.description], embed_fn, model_name)[0]
                # 找到旧 entries 中的位置
                for i, old_e in enumerate(self._entries):
                    if old_e.name == e.name:
                        self._entries[i] = e
                        self._embeddings[i] = np.array(vec, dtype=np.float32)
                        changed += 1
                        break
                else:
                    # 新 skill，追加
                    self._entries.append(e)
                    if self._embeddings is not None and len(self._embeddings) > 0:
                        self._embeddings = np.vstack([
                            self._embeddings,
                            np.array(vec, dtype=np.float32).reshape(1, -1)
                        ])
                    else:
                        self._embeddings = np.array([vec], dtype=np.float32)
                    changed += 1

        # 检测删除的条目
        deleted = [name for name in old_mtimes if name not in new_by_name]
        if deleted:
            keep_idx = [i for i, e in enumerate(self._entries) if e.name not in deleted]
            self._entries = [self._entries[i] for i in keep_idx]
            self._embeddings = self._embeddings[keep_idx]
            changed += len(deleted)

        if changed:
            self._save_mtimes()
        return changed

    def _ensure_model_downloaded(self, model_name: str) -> None:
        """确保 embedding 模型已下载（GFW 友好：15s 线程超时）。"""
        import threading

        result = [None]
        error = [None]

        def _download():
            try:
                from sentence_transformers import SentenceTransformer
                _ = SentenceTransformer(model_name, trust_remote_code=True)
                result[0] = True
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=_download, daemon=True)
        t.start()
        t.join(timeout=15)

        if result[0]:
            logger.info(f"[skillpool] model '{model_name}' ready")
        else:
            reason = "timed out" if t.is_alive() else str(error[0])
            logger.warning(f"[skillpool] model download skipped: {reason}, using fallback embedder")

    def _quick_embed(self, text: str, dim: int = 384) -> np.ndarray:
        """快速嵌入（词袋 + hash，无需模型）。"""
        # 简单实现：用字符 n-gram hash 做伪向量
        vec = np.zeros(dim, dtype=np.float32)
        text_lower = text.lower()
        for n in [3, 4, 5]:
            for i in range(len(text_lower) - n + 1):
                h = hash(text_lower[i:i+n]) % dim
                vec[h] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _save(self) -> None:
        """保存索引为 JSON + base64（可检视、版本兼容）。"""
        import base64
        self.INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        emb_bytes = self._embeddings.tobytes()
        data = {
            "version": 2,
            "entries": [
                {k: v for k, v in e.__dict__.items() if not k.startswith("_")}
                for e in self._entries
            ],
            "shape": list(self._embeddings.shape),
            "embeddings": base64.b64encode(emb_bytes).decode("ascii"),
        }
        with open(self.INDEX_FILE, "w") as f:
            json.dump(data, f)
        self._save_config()

    def _migrate_from_pickle(self, pkl_path: Path) -> None:
        """从旧 pickle 格式迁移到 JSON。"""
        try:
            with open(pkl_path, "rb") as f:
                old = pickle.load(f)
            self._entries = old["entries"]
            self._embeddings = old["embeddings"]
            self._save()
            pkl_path.rename(pkl_path.with_suffix(".pkl.bak"))
            logger.info("[skillpool] pickle → JSON migration complete")
        except Exception as exc:
            logger.warning(f"[skillpool] pickle migration failed: {exc}")

    def _save_config(self) -> None:
        with open(self.CONFIG_FILE, "w") as f:
            json.dump({"core_skills": sorted(self._core_skills)}, f, indent=2)

    def _load_config(self) -> None:
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE) as f:
                    data = json.load(f)
                self._core_skills = set(data.get("core_skills", []))
            except Exception:
                self._core_skills = self.DEFAULT_CORE.copy()
        else:
            self._core_skills = self.DEFAULT_CORE.copy()
            self._save_config()
