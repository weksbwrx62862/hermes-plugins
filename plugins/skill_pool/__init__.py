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
                 skill_path: str, tier: str = "pool", **kwargs):
        self.name = name
        self.description = description
        self.category = category
        self.skill_path = skill_path
        self.tier = tier  # "core" | "pool"
        # MUSE 启发: Skill 级统计（兼容旧索引无此字段）
        self.test_pass_count = kwargs.get("test_pass_count", 0)
        self.test_fail_count = kwargs.get("test_fail_count", 0)
        self.last_tested = kwargs.get("last_tested", None)
        self.usage_count = kwargs.get("usage_count", 0)
        self.last_used = kwargs.get("last_used", None)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "skill_path": self.skill_path,
            "tier": self.tier,
            "test_pass_count": self.test_pass_count,
            "test_fail_count": self.test_fail_count,
            "last_tested": self.last_tested,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
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
                # GFW 修复：使用镜像并优先本地缓存，避免 HuggingFace 联网超时
                if "HF_ENDPOINT" not in os.environ:
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                os.environ["HF_HUB_OFFLINE"] = "1"
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
            old_entries = self._entries
            self._entries = [e for e in old_entries if e.name not in deleted_names]
            if self._embeddings is not None:
                keep_idx = [i for i, e in enumerate(old_entries) if e.name not in deleted_names]
                if keep_idx:
                    self._embeddings = self._embeddings[keep_idx]
                    assert self._embeddings.shape[0] == len(self._entries)
                else:
                    self._embeddings = None

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
        """预下载 sentence-transformers 模型（GFW 友好：使用 hf-mirror）。"""
        import threading

        def _try_download():
            try:
                if "HF_ENDPOINT" not in os.environ:
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                # 不要设置 HF_HUB_OFFLINE，允许实际下载（首次需要）
                from sentence_transformers import SentenceTransformer
                SentenceTransformer(model_name, trust_remote_code=True)
                logger.info("[skillpool] model %s downloaded/loaded successfully", model_name)
            except Exception as e:
                logger.warning("[skillpool] model download failed: %s", e)

        t = threading.Thread(target=_try_download, daemon=True)
        t.start()
        t.join(timeout=60)  # 首次下载可能需要更长时间

        if t.is_alive():
            logger.warning(
                "[skillpool] model download timed out, will use fallback"
            )

    # ═══════════════════════════════════════════════════════════
    # MUSE 启发: 五阶段生命周期支持
    # ═══════════════════════════════════════════════════════════

    def evaluate_skill(self, name: str) -> dict:
        """跑 Skill 的 tests/ 目录中的所有测试，返回结果。

        Returns:
            {"passed": bool, "total": int, "passed_count": int,
             "failures": list[str], "output": str}
        """
        entry = self._find_entry(name)
        if not entry:
            return {"passed": False, "total": 0, "passed_count": 0,
                    "failures": ["Skill not found"], "output": ""}

        skill_dir = Path(entry.skill_path).parent
        test_dir = skill_dir / "tests"
        if not test_dir.exists() or not any(test_dir.iterdir()):
            return {"passed": True, "total": 0, "passed_count": 0,
                    "failures": [], "output": "No tests directory found — gate passes by default"}

        import subprocess
        results = {"total": 0, "passed_count": 0, "failures": [], "output": ""}
        outputs = []

        # Run all .py and .sh test files
        for test_file in sorted(test_dir.iterdir()):
            if not (test_file.suffix in (".py", ".sh")):
                continue
            results["total"] += 1
            try:
                if test_file.suffix == ".py":
                    proc = subprocess.run(
                        ["python3", str(test_file)],
                        cwd=str(skill_dir),
                        capture_output=True, text=True, timeout=30,
                    )
                else:
                    proc = subprocess.run(
                        ["bash", str(test_file)],
                        cwd=str(skill_dir),
                        capture_output=True, text=True, timeout=30,
                    )
                outputs.append(f"--- {test_file.name} ---\n{proc.stdout}")
                if proc.returncode == 0:
                    results["passed_count"] += 1
                else:
                    results["failures"].append(
                        f"{test_file.name}: exit={proc.returncode}\n{proc.stderr[:500]}"
                    )
            except subprocess.TimeoutExpired:
                results["failures"].append(f"{test_file.name}: timeout (30s)")
            except Exception as e:
                results["failures"].append(f"{test_file.name}: {e}")

        results["passed"] = len(results["failures"]) == 0
        results["output"] = "\n".join(outputs)

        # 更新统计
        entry.test_pass_count += results["passed_count"]
        entry.test_fail_count += len(results["failures"])
        entry.last_tested = time.strftime("%Y-%m-%d %H:%M:%S")

        return results

    def register_with_gate(self, name: str, description: str = "",
                           category: str = "", check_tests: bool = True) -> dict:
        """注册 Skill（带测试守门）。创建→评估→注册循环。

        Returns:
            {"status": "ok"|"failed"|"skipped", "tests": dict, "reason": str}
        """
        entry = self._find_entry(name)
        if not entry:
            return {"status": "failed", "reason": f"Skill '{name}' not found in index. Run build_index first."}

        # 1. 跑测试
        test_result = None
        if check_tests:
            test_result = self.evaluate_skill(name)
            if not test_result["passed"]:
                return {"status": "failed", "tests": test_result,
                        "reason": f"Tests failed ({test_result['passed_count']}/{test_result['total']} passed)"}

        # 2. 注册
        if entry.tier in ("core", "pool"):
            return {"status": "skipped", "tests": test_result,
                    "reason": f"Already registered as '{entry.tier}'"}

        entry.tier = "pool"
        self._save()
        self._save_config()
        return {"status": "ok", "tests": test_result,
                "reason": f"Registered as 'pool' (tests: {test_result['passed_count']}/{test_result['total']} passed)" if test_result else "Registered (no tests)"}

    def prune_skills(self, min_usage: int = 0, max_fail_rate: float = 0.5,
                     dry_run: bool = True) -> dict:
        """裁剪低质量/长期未使用的 Skill。

        Args:
            min_usage: 最少使用次数（低于此值考虑裁剪）
            max_fail_rate: 最大失败率（超过此值考虑裁剪）
            dry_run: True=仅报告，False=执行裁剪

        Returns:
            {"pruned": int, "candidates": list[dict], "reason": str}
        """
        candidates = []
        for entry in self._entries:
            if entry.tier == "core":
                continue  # 保护 core 技能
            reasons = []
            total_tests = entry.test_pass_count + entry.test_fail_count
            if total_tests > 0:
                fail_rate = entry.test_fail_count / total_tests
                if fail_rate > max_fail_rate:
                    reasons.append(f"high fail rate: {fail_rate:.1%}")
            if entry.usage_count < min_usage and entry.last_used:
                # 超过 30 天未使用
                try:
                    last_used_ts = time.mktime(time.strptime(entry.last_used, "%Y-%m-%d %H:%M:%S"))
                    if time.time() - last_used_ts > 30 * 86400:
                        reasons.append(f"unused >30 days (last: {entry.last_used})")
                except Exception:
                    pass
            if reasons:
                candidates.append({
                    "name": entry.name,
                    "reasons": reasons,
                    "fail_rate": (entry.test_fail_count / total_tests) if total_tests else 0,
                    "usage_count": entry.usage_count,
                })

        if not dry_run and candidates:
            prune_names = {c["name"] for c in candidates}
            self._entries = [e for e in self._entries if e.name not in prune_names]
            self._save()
            self._save_config()
            logger.info("[skillpool] pruned %d skills: %s", len(prune_names), prune_names)

        return {
            "pruned": 0 if dry_run else len(candidates),
            "candidates": candidates,
            "dry_run": dry_run,
        }

    def mark_used(self, name: str) -> None:
        """标记 Skill 被使用（更新 usage_count）。"""
        entry = self._find_entry(name)
        if entry:
            entry.usage_count += 1
            entry.last_used = time.strftime("%Y-%m-%d %H:%M:%S")

    def find_similar_skills(self, name: str, threshold: float = 0.85) -> list[dict]:
        """查找与指定 Skill 高度相似的其他 Skill（用于合并候选）。"""
        entry = self._find_entry(name)
        if not entry or self._embeddings is None:
            return []
        idx = next((i for i, e in enumerate(self._entries) if e.name == name), None)
        if idx is None:
            return []
        target_vec = self._embeddings[idx]
        norms = np.linalg.norm(self._embeddings, axis=1)
        target_norm = np.linalg.norm(target_vec)
        sims = np.dot(self._embeddings, target_vec) / (norms * target_norm + 1e-8)
        results = []
        for i, s in enumerate(sims):
            if i != idx and s >= threshold:
                results.append({
                    "name": self._entries[i].name,
                    "similarity": round(float(s), 4),
                })
        return sorted(results, key=lambda x: -x["similarity"])

    # ═══════════════════════════════════════════════════════════
    # MUSE 启发: .memory.md Skill 级记忆
    # ═══════════════════════════════════════════════════════════

    def read_memory(self, name: str) -> str:
        """读取 Skill 的 .memory.md 内容。"""
        entry = self._find_entry(name)
        if not entry:
            return ""
        memory_path = Path(entry.skill_path).parent / ".memory.md"
        if memory_path.exists():
            return memory_path.read_text(encoding="utf-8", errors="replace")
        return ""

    def append_memory(self, name: str, note: str) -> bool:
        """追加一条笔记到 Skill 的 .memory.md，同时写入聚合索引以便跨 Skill 搜索。"""
        entry = self._find_entry(name)
        if not entry:
            return False
        memory_path = Path(entry.skill_path).parent / ".memory.md"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry_text = f"\n## [{timestamp}]\n{note}\n"
        with open(memory_path, "a", encoding="utf-8") as f:
            f.write(entry_text)
        # 同步写入聚合索引（跨 Skill 搜索用）
        self._sync_memory_to_global_index(name, note, timestamp)
        return True

    def _sync_memory_to_global_index(self, name: str, note: str, timestamp: str) -> None:
        """同步 .memory.md 笔记到全局聚合索引。"""
        index_path = self.INDEX_FILE.parent / ".memories_index.json"
        try:
            if index_path.exists():
                idx = json.loads(index_path.read_text(encoding="utf-8"))
            else:
                idx = {"version": 1, "entries": []}
            idx["entries"].append({
                "skill": name,
                "timestamp": timestamp,
                "note": note[:500],
            })
            # 保持索引轻量：最多保留 1000 条
            if len(idx["entries"]) > 1000:
                idx["entries"] = idx["entries"][-1000:]
            index_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def search_memories(self, query: str, limit: int = 10) -> list[dict]:
        """模糊搜索所有 Skill 的 .memory.md 笔记。"""
        index_path = self.INDEX_FILE.parent / ".memories_index.json"
        if not index_path.exists():
            return []
        try:
            idx = json.loads(index_path.read_text(encoding="utf-8"))
            query_lower = query.lower()
            results = []
            for entry in idx.get("entries", []):
                if query_lower in entry.get("note", "").lower() or \
                   query_lower in entry.get("skill", "").lower():
                    results.append(entry)
            return results[:limit]
        except Exception:
            return []

    def _find_entry(self, name: str) -> Optional[SkillEntry]:
        """按名称查找条目。"""
        for e in self._entries:
            if e.name == name:
                return e
        return None

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
# MUSE 启发: 工具处理器
# ═══════════════════════════════════════════════════════════════════

def _tool_evaluate_skill(args: Dict[str, Any], **kwargs) -> str:
    name = args.get("name", "")
    try:
        pool = _get_pool()
        result = pool.evaluate_skill(name)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_register_gate(args: Dict[str, Any], **kwargs) -> str:
    name = args.get("name", "")
    check_tests = args.get("check_tests", True)
    try:
        pool = _get_pool()
        result = pool.register_with_gate(name, check_tests=check_tests)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_prune_skills(args: Dict[str, Any], **kwargs) -> str:
    min_usage = args.get("min_usage", 0)
    max_fail_rate = args.get("max_fail_rate", 0.5)
    dry_run = args.get("dry_run", True)
    try:
        pool = _get_pool()
        result = pool.prune_skills(min_usage=min_usage, max_fail_rate=max_fail_rate, dry_run=dry_run)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_memory_read(args: Dict[str, Any], **kwargs) -> str:
    name = args.get("name", "")
    try:
        pool = _get_pool()
        content = pool.read_memory(name)
        return json.dumps({"name": name, "memory": content}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_memory_append(args: Dict[str, Any], **kwargs) -> str:
    name = args.get("name", "")
    note = args.get("note", "")
    try:
        pool = _get_pool()
        ok = pool.append_memory(name, note)
        return json.dumps({"status": "ok" if ok else "failed", "name": name})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_find_similar(args: Dict[str, Any], **kwargs) -> str:
    name = args.get("name", "")
    threshold = args.get("threshold", 0.85)
    try:
        pool = _get_pool()
        results = pool.find_similar_skills(name, threshold)
        return json.dumps({"name": name, "similar": results}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_memory_search(args: Dict[str, Any], **kwargs) -> str:
    """跨 Skill 搜索 .memory.md 笔记内容。"""
    query = args.get("query", "")
    limit = args.get("limit", 10)
    try:
        pool = _get_pool()
        results = pool.search_memories(query, limit)
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


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
    try:
        pool = _get_pool()
        incremental = args.get("incremental", True)
        count = pool.build_index(incremental=incremental)
        return json.dumps({
            "status": "ok",
            "indexed_skills": count,
            "incremental": incremental,
        }, ensure_ascii=False)
    except Exception as e:
        logger.warning("[skillpool] build_index failed: %s", e)
        return json.dumps({"error": str(e)})


def _tool_search_skills(args: Dict[str, Any], **kwargs) -> str:
    """语义搜索技能。"""
    query = args.get("query")
    if not isinstance(query, str):
        return json.dumps({"error": "query is required and must be a string"})
    k = args.get("k", 5)
    try:
        pool = _get_pool()
        results = pool.search(query, k=k)
        return json.dumps({
            "query": query,
            "results": [
                {"name": e.name, "description": e.description, "tier": e.tier}
                for e in results
            ],
        }, ensure_ascii=False)
    except Exception as e:
        logger.warning("[skillpool] search failed: %s", e)
        return json.dumps({"error": str(e)})


def _tool_list_skills(args: Dict[str, Any], **kwargs) -> str:
    """列出技能池中的所有技能。"""
    try:
        pool = _get_pool()
        if not pool.load_index():
            # 如果没有索引，直接扫描
            pool.build_index(incremental=False)

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
    except Exception as e:
        logger.warning("[skillpool] list_skills failed: %s", e)
        return json.dumps({"error": str(e)})


def _tool_set_core(args: Dict[str, Any], **kwargs) -> str:
    """设置技能为常驻（core）。"""
    name = args.get("name", "")
    if not isinstance(name, str) or not name.strip():
        return json.dumps({"error": "name is required and must be a non-empty string"})
    try:
        pool = _get_pool()
        pool.set_core(name)
        return json.dumps({"status": "ok", "name": name, "tier": "core"})
    except Exception as e:
        logger.warning("[skillpool] set_core failed: %s", e)
        return json.dumps({"error": str(e)})


def _tool_set_pool(args: Dict[str, Any], **kwargs) -> str:
    """设置技能为按需（pool）。"""
    name = args.get("name", "")
    if not isinstance(name, str) or not name.strip():
        return json.dumps({"error": "name is required and must be a non-empty string"})
    try:
        pool = _get_pool()
        pool.set_pool(name)
        return json.dumps({"status": "ok", "name": name, "tier": "pool"})
    except Exception as e:
        logger.warning("[skillpool] set_pool failed: %s", e)
        return json.dumps({"error": str(e)})


def _tool_auto_tune(args: Dict[str, Any], **kwargs) -> str:
    """根据使用数据自动调整常驻技能。"""
    try:
        pool = _get_pool()
        top_n = args.get("top_n", 8)
        count = pool.auto_tune_core(top_n=top_n)
        status = pool.get_auto_tune_status()
        return json.dumps({
            "status": "ok",
            "core_count": count,
            **status,
        }, ensure_ascii=False)
    except Exception as e:
        logger.warning("[skillpool] auto_tune failed: %s", e)
        return json.dumps({"error": str(e)})


def _tool_snapshot(args: Dict[str, Any], **kwargs) -> str:
    """生成只含 core 技能的快照（影响下次 session 的技能列表）。"""
    try:
        pool = _get_pool()
        count = pool.write_hermes_snapshot()
        return json.dumps({
            "status": "ok",
            "core_skills_in_snapshot": count,
        }, ensure_ascii=False)
    except Exception as e:
        logger.warning("[skillpool] snapshot failed: %s", e)
        return json.dumps({"error": str(e)})


def _tool_skill_usage(args: Dict[str, Any], **kwargs) -> str:
    """查看技能使用统计。"""
    try:
        pool = _get_pool()
        return json.dumps(pool.get_auto_tune_status(), ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("[skillpool] skill_usage failed: %s", e)
        return json.dumps({"error": str(e)})


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
    # ── 新增 MUSE 启发工具 ──────────────────────────────────
    {
        "name": "skill_pool_evaluate",
        "description": "跑 Skill 的 tests/ 目录中的所有测试，返回通过/失败结果。MUSE 启发：测试驱动的质量守门。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名称"},
            },
            "required": ["name"],
        },
        "handler": _tool_evaluate_skill,
    },
    {
        "name": "skill_pool_register_gate",
        "description": "注册 Skill（带测试守门）。先跑 tests/ 目录测试，全部通过才注册。MUSE: create→evaluate→register 循环。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名称"},
                "check_tests": {"type": "boolean", "description": "是否跑测试守门（默认 true）", "default": True},
            },
            "required": ["name"],
        },
        "handler": _tool_register_gate,
    },
    {
        "name": "skill_pool_prune",
        "description": "裁剪低质量/长期未使用的 Skill。MUSE 启发：精炼阶段自动清理。dry_run=true 仅报告不执行。",
        "parameters": {
            "type": "object",
            "properties": {
                "min_usage": {"type": "integer", "description": "最少使用次数（默认 0）", "default": 0},
                "max_fail_rate": {"type": "number", "description": "最大失败率（默认 0.5，即 50%）", "default": 0.5},
                "dry_run": {"type": "boolean", "description": "仅报告不执行（默认 true）", "default": True},
            },
        },
        "handler": _tool_prune_skills,
    },
    {
        "name": "skill_pool_memory_read",
        "description": "读取 Skill 的 .memory.md（Skill 级记忆）。MUSE 启发：每 Skill 一本使用笔记。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名称"},
            },
            "required": ["name"],
        },
        "handler": _tool_memory_read,
    },
    {
        "name": "skill_pool_memory_append",
        "description": "追加一条笔记到 Skill 的 .memory.md。MUSE 启发：Skill 级经验累积。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名称"},
                "note": {"type": "string", "description": "笔记内容（支持 Markdown）"},
            },
            "required": ["name", "note"],
        },
        "handler": _tool_memory_append,
    },
    {
        "name": "skill_pool_find_similar",
        "description": "查找与指定 Skill 高度相似的其他 Skill（合并候选）。MUSE 启发：去重合并。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名称"},
                "threshold": {"type": "number", "description": "相似度阈值（默认 0.85）", "default": 0.85},
            },
            "required": ["name"],
        },
        "handler": _tool_find_similar,
    },
    {
        "name": "skill_pool_memory_search",
        "description": "模糊搜索所有 Skill 的 .memory.md 笔记内容。MUSE 启发：跨 Skill 经验检索。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "description": "结果条数上限（默认 10）", "default": 10},
            },
            "required": ["query"],
        },
        "handler": _tool_memory_search,
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
    if ctx is None:
        logger.warning("SkillPool: register ctx is None, skipping registration")
        return
    if not hasattr(ctx, "register_tool"):
        logger.warning("SkillPool: register ctx has no register_tool method, skipping registration")
        return
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
