"""Skill Router Plugin v3.1 — 统一技能路由

双后端架构：
  - lightweight: MiniLM 微调嵌入 + MiniLM CrossEncoder 重排序（默认，资源友好）
  - skillrouter: Qwen3-0.6B 编码器 + 重排序器（SkillRouter 论文方案，更高精度）

核心设计：
  - core 技能：常驻 system prompt（高频技能名称列表）
  - pool 技能：按需语义搜索 Top-K，带置信度过滤
  - 完整 body 嵌入：使用技能全文（非截断），论文核心发现
  - 增量更新：基于 mtime 的嵌入缓存
  - 反馈学习：成功/跳过反馈影响路由评分

配置 (~/.hermes/config.yaml):
  plugins.skill-router:
    enabled: true
    backend: lightweight  # 或 skillrouter
    top_k: 5
    core_skills:
      - hermes-agent
      - skill-creator
      - web-search-china
    hybrid_calibration: minmax  # 或 sigmoid / zscore
"""

import importlib.util
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── 动态导入助手 ──
# skill-router 目录名含连字符，无法作为 Python 包名导入，因此使用文件路径加载子模块。
def _load_local_module(name: str, rel_path: str) -> Any:
    """从插件目录动态加载本地子模块"""
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(plugin_dir, rel_path)
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块 {name}，spec 或 loader 为 None")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ── 加载子模块 ──
_bm25_searcher_mod = _load_local_module("bm25_searcher", "bm25_searcher.py")
_domain_config_mod = _load_local_module("domain_config", "domain_config.py")
_hybrid_searcher_mod = _load_local_module("hybrid_searcher", "hybrid_searcher.py")
_state_mod = _load_local_module("state", "state.py")

# 子模块符号别名，保持对外兼容
BM25Searcher = _bm25_searcher_mod.BM25Searcher
HybridSearcher = _hybrid_searcher_mod.HybridSearcher
CacheManager = _state_mod.CacheManager
QueryCache = _state_mod.QueryCache
PluginState = _state_mod.PluginState
plugin_state = _state_mod.plugin_state


# ── 默认配置 ──
_DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "model_path": "~/.hermes/skills/devops/skill-router-scalable/fine-tuned-model-v7",
    "db_path": "~/.hermes/skill_index.db",
    "top_k": 5,
    "core_skills": [
        "hermes-agent",
        "skill-creator",
        "web-search-china",
    ],
    "vector_weight": 0.7,
    "bm25_weight": 0.3,
    "use_reranker": True,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # 快速版(80MB, ~60ms)
    "backend": "lightweight",  # "lightweight" (MiniLM) or "skillrouter" (Qwen3-0.6B)
    "skillrouter_emb_path": "~/models/skillrouter/SkillRouter-Embedding-0.6B",
    "skillrouter_rank_path": "~/models/skillrouter/SkillRouter-Reranker-0.6B",
    "query_cache_ttl": 300,
    "query_cache_max_size": 1000,
    "encode_timeout": 10,
    "confidence_threshold_low": 0.3,
    "confidence_threshold_medium": 0.4,
    "hybrid_calibration": "minmax",  # minmax / sigmoid / zscore
}


def _load_config() -> Dict[str, Any]:
    """加载插件配置，带 300 秒 TTL 缓存"""
    cached = plugin_state.get_config_cache()
    if cached is not None:
        return cached

    try:
        import yaml
        config_path = os.path.expanduser("~/.hermes/config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            plugin_config = config.get("plugins", {}).get("skill-router", {})
            result = {**_DEFAULT_CONFIG, **plugin_config}
        else:
            result = _DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.debug("无法加载配置: %s", e)
        result = _DEFAULT_CONFIG.copy()

    plugin_state.set_config_cache(result)
    return result


# ── 注入配置加载器并暴露域配置符号 ──
_domain_config_mod.set_config_loader(_load_config)

_DEFAULT_DOMAIN_CONFIG = _domain_config_mod._DEFAULT_DOMAIN_CONFIG
_merge_domain_config = _domain_config_mod._merge_domain_config
_load_domain_config = _domain_config_mod._load_domain_config
_validate_domain_config = _domain_config_mod._validate_domain_config
_classify_domain = _domain_config_mod._classify_domain
_get_domain_aggregator = _domain_config_mod._get_domain_aggregator
_filter_skills_by_domain = _domain_config_mod._filter_skills_by_domain


# ── core/pool 分类 ──
_POOL_INDEX = Path.home() / ".hermes" / "skills" / ".pool_index.json"
_POOL_CONFIG = Path.home() / ".hermes" / "skills" / ".pool_config.json"


class FeedbackStore:
    """技能反馈存储，JSONL 持久化，线程安全

    记录技能使用反馈（成功/跳过），影响后续路由评分：
      - 成功使用: +0.05
      - 被跳过: -0.02
      - 超过 24 小时的反馈权重指数衰减（半衰期 12 小时）
      - 最终调整值会被限制在 [-0.5, 0.5]，避免反馈过度影响排序
    """

    FEEDBACK_FILE = Path.home() / ".hermes" / "skill_router_feedback.jsonl"
    SUCCESS_DELTA = 0.05
    SKIP_DELTA = -0.02
    DECAY_THRESHOLD_HOURS = 24.0
    HALF_LIFE_HOURS = 12.0
    MAX_ADJUSTMENT = 0.5

    def __init__(self):
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []
        self._load_history()

    def _load_history(self) -> None:
        """启动时加载历史反馈数据"""
        try:
            if self.FEEDBACK_FILE.exists():
                with open(self.FEEDBACK_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._records.append(json.loads(line))
        except Exception as e:
            logger.debug("加载反馈历史失败: %s", e)

    def record(self, skill_name: str, query: str, feedback_type: str) -> None:
        """记录一条反馈"""
        if feedback_type not in ("success", "skip"):
            return
        entry = {
            "skill_name": skill_name,
            "query": query,
            "feedback_type": feedback_type,
            "timestamp": time.time(),
        }
        with self._lock:
            self._records.append(entry)
            try:
                with open(self.FEEDBACK_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning("写入反馈记录失败: %s", e)

    @staticmethod
    def _normalize_score(total: float, max_abs: float = 0.5) -> float:
        """将反馈调整值限制在 [-max_abs, max_abs] 区间"""
        return max(-max_abs, min(max_abs, total))

    def get_adjustments(self, skill_name: str) -> float:
        """获取某技能的累计反馈调整分数"""
        now = time.time()
        total = 0.0
        with self._lock:
            for rec in self._records:
                if rec["skill_name"] != skill_name:
                    continue
                age_hours = (now - rec["timestamp"]) / 3600.0
                if rec["feedback_type"] == "success":
                    delta = self.SUCCESS_DELTA
                else:
                    delta = self.SKIP_DELTA
                weight = self._decay_weight(age_hours)
                total += delta * weight
        return self._normalize_score(total, self.MAX_ADJUSTMENT)

    def _decay_weight(self, age_hours: float) -> float:
        """衰减函数：24 小时内权重为 1.0，之后指数衰减"""
        if age_hours <= self.DECAY_THRESHOLD_HOURS:
            return 1.0
        return 0.5 ** ((age_hours - self.DECAY_THRESHOLD_HOURS) / self.HALF_LIFE_HOURS)


_feedback_store = FeedbackStore()


def _get_model_path() -> str:
    return os.path.expanduser(_load_config()["model_path"])


def _get_db_path() -> str:
    return os.path.expanduser(_load_config()["db_path"])


def _get_core_skills() -> Set[str]:
    """获取 core 技能列表"""
    config = _load_config()
    core: Set[str] = set(config.get("core_skills", []))

    if _POOL_CONFIG.exists():
        try:
            with open(_POOL_CONFIG) as f:
                pool_cfg = json.load(f)
            core.update(pool_cfg.get("core_skills", []))
        except Exception:
            pass

    return core


def _load_embedding_model() -> Optional[Any]:
    """加载嵌入模型（无 TTL，全局单例）"""
    with plugin_state.get_model_lock():
        cached = plugin_state.get_model_cache()
        if cached is not None:
            return cached
        model_path = _get_model_path()
        if not os.path.exists(model_path):
            logger.warning("技能路由模型未找到: %s", model_path)
            return None
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_path, device="cpu")
            plugin_state.set_model_cache(model)
            logger.info("已加载技能路由模型: %s", model_path)
            return model
        except Exception as e:
            logger.error("加载技能路由模型失败: %s", e)
            return None


def _get_skillrouter_backend(config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """获取 SkillRouter 后端（延迟加载，线程安全）

    Args:
        config: 配置字典，如果为 None 则自动加载

    Returns:
        SkillRouterBackend 实例，加载失败返回 None
    """
    cached = plugin_state.get_skillrouter_backend()
    if cached is not None:
        return cached

    if config is None:
        config = _load_config()

    backend_type = config.get("backend", "lightweight")
    if backend_type != "skillrouter":
        return None

    with plugin_state.get_backend_lock():
        cached = plugin_state.get_skillrouter_backend()
        if cached is not None:
            return cached

        try:
            from .skillrouter_backend import SkillRouterBackend

            emb_path = config.get("skillrouter_emb_path", "~/models/skillrouter/SkillRouter-Embedding-0.6B")
            rank_path = config.get("skillrouter_rank_path", "~/models/skillrouter/SkillRouter-Reranker-0.6B")

            logger.info("Initializing SkillRouter backend: emb=%s, rank=%s", emb_path, rank_path)
            backend = SkillRouterBackend(emb_path, rank_path)
            plugin_state.set_skillrouter_backend(backend)
            logger.info("SkillRouter backend initialized (models will be lazy-loaded on first query)")
            return backend

        except Exception as e:
            logger.error("Failed to initialize SkillRouter backend: %s", e)
            return None


_EMBEDDING_MTIMES_PATH = Path.home() / ".hermes" / "skill_embedding_mtimes.json"


def _load_embedding_mtimes() -> Dict[str, float]:
    """加载上次嵌入时各技能的 mtime 记录

    返回 {技能名: mtime} 字典，文件不存在或解析失败时返回空字典。
    """
    try:
        if _EMBEDDING_MTIMES_PATH.exists():
            with open(_EMBEDDING_MTIMES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.debug("加载嵌入 mtime 记录失败: %s", e)
    return {}


def _save_embedding_mtimes(mtimes: Dict[str, float]) -> None:
    """保存各技能的 mtime 记录到 JSON 文件"""
    try:
        with open(_EMBEDDING_MTIMES_PATH, "w", encoding="utf-8") as f:
            json.dump(mtimes, f, ensure_ascii=False)
    except Exception as e:
        logger.warning("保存嵌入 mtime 记录失败: %s", e)


def _load_skill_index() -> Dict[str, Dict[str, Any]]:
    """加载技能索引，带 300 秒 TTL 缓存

    索引缓存过期时，同步失效嵌入缓存，确保数据一致性。
    同时读取每个技能的 mtime（修改时间），用于增量嵌入更新。
    如果 skills 表没有 mtime 列，自动 ALTER TABLE 添加。
    """
    cached = plugin_state.get_skill_index_cache()
    if cached is not None:
        return cached

    plugin_state.invalidate_embedding_cache()
    plugin_state.invalidate_bm25_cache()

    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 检查 skills 表是否包含 mtime 列，不存在则添加
        cursor.execute("PRAGMA table_info(skills)")
        columns = {row[1] for row in cursor.fetchall()}
        if "mtime" not in columns:
            cursor.execute("ALTER TABLE skills ADD COLUMN mtime REAL DEFAULT 0.0")
            conn.commit()
            logger.info("已为 skills 表添加 mtime 列")

        cursor.execute("SELECT name, category, description, body, mtime FROM skills")
        skills: Dict[str, Dict[str, Any]] = {}
        for row in cursor.fetchall():
            skill_name, category, description, body, mtime = row
            skills[skill_name] = {
                "category": category or "",
                "description": description or "",
                "body_text": body or "",
                "mtime": mtime or 0.0,
            }
        conn.close()
        plugin_state.set_skill_index_cache(skills)
        logger.info("已从索引加载 %d 个技能", len(skills))
        return skills
    except Exception as e:
        logger.error("加载技能索引失败: %s", e)
        return {}


def _get_skill_embeddings() -> Tuple[List[str], Optional[Any]]:
    """获取技能嵌入向量，支持增量更新

    通过对比当前技能 mtime 与上次嵌入时的 mtime，实现增量更新：
      - 新增技能：追加嵌入
      - 修改技能（mtime 变化）：重新嵌入并替换
      - 删除技能：移除对应嵌入
      - 未变更技能：保留现有嵌入
    无任何变更时直接返回缓存。
    """
    cached = plugin_state.get_embedding_cache()
    skills = _load_skill_index()
    if not skills:
        return [], None

    model = _load_embedding_model()
    if not model:
        return [], None

    # 加载上次嵌入时的 mtime 记录
    prev_mtimes = _load_embedding_mtimes()

    # 计算当前各技能的 mtime
    curr_mtimes = {name: info["mtime"] for name, info in skills.items()}

    # 分类：新增、修改、删除、未变更
    curr_names = set(curr_mtimes.keys())
    prev_names = set(prev_mtimes.keys())

    added = curr_names - prev_names
    deleted = prev_names - curr_names
    modified = {
        name for name in curr_names & prev_names
        if curr_mtimes[name] != prev_mtimes[name]
    }
    changed = added | modified

    # 无变更且有缓存时直接返回
    if not changed and not deleted and cached is not None:
        return cached

    import numpy as np

    if cached is not None and (not changed and not deleted):
        # 无变更但无缓存（理论上不会走到这里）
        return cached

    if cached is not None:
        # 增量更新：基于现有嵌入矩阵进行修改
        old_names, old_embeddings = cached

        # 构建旧嵌入的名称→索引映射
        name_to_idx = {name: i for i, name in enumerate(old_names)}

        # 保留未变更技能的嵌入
        keep_names = [n for n in old_names if n not in deleted and n not in modified]
        keep_indices = [name_to_idx[n] for n in keep_names]
        keep_embeddings = old_embeddings[keep_indices] if keep_indices else np.empty((0, old_embeddings.shape[1]))

        # 对变更技能重新嵌入
        if changed:
            changed_texts: List[str] = []
            changed_names = sorted(changed)
            for name in changed_names:
                skill = skills[name]
                text = f"{name} | {skill['description']} | {skill.get('body_text', '')}"
                changed_texts.append(text)
            changed_embeddings = model.encode(
                changed_texts, convert_to_numpy=True,
                normalize_embeddings=True, batch_size=32,
            )
        else:
            changed_names = []
            changed_embeddings = np.empty((0, keep_embeddings.shape[1] if keep_embeddings.size else 0))

        # 合并：保留的 + 新增/修改的
        if keep_embeddings.size and changed_embeddings.size:
            new_embeddings = np.vstack([keep_embeddings, changed_embeddings])
        elif changed_embeddings.size:
            new_embeddings = changed_embeddings
        else:
            new_embeddings = keep_embeddings

        new_names = keep_names + changed_names

        plugin_state.set_embedding_cache((new_names, new_embeddings))

        # 保存新的 mtime 记录
        _save_embedding_mtimes(curr_mtimes)

        logger.info(
            "增量更新嵌入: 保留 %d, 变更 %d, 删除 %d",
            len(keep_names), len(changed), len(deleted),
        )
        return new_names, new_embeddings

    # 全量计算：无缓存或首次加载
    skill_names = list(skills.keys())
    skill_texts: List[str] = []
    for name in skill_names:
        skill = skills[name]
        text = f"{name} | {skill['description']} | {skill.get('body_text', '')}"
        skill_texts.append(text)

    embeddings = model.encode(
        skill_texts, convert_to_numpy=True,
        normalize_embeddings=True, batch_size=32,
    )
    plugin_state.set_embedding_cache((skill_names, embeddings))

    # 保存 mtime 记录
    _save_embedding_mtimes(curr_mtimes)

    logger.info("已预计算 %d 个技能的嵌入", len(skill_names))
    return skill_names, embeddings


def _get_bm25_searcher() -> Any:
    """获取 BM25Searcher 实例，带 TTL 缓存

    缓存与索引联动：索引刷新时 BM25 搜索器同步失效重建。
    """
    cached = plugin_state.get_bm25_cache()
    if cached is not None:
        return cached
    skills = _load_skill_index()
    searcher = BM25Searcher(skills)
    plugin_state.set_bm25_cache(searcher)
    logger.info("已构建 BM25Searcher（%d 个技能）", len(skills))
    return searcher


def _encode_with_timeout(model: Any, texts: List[str], timeout: float, **kwargs) -> Optional[Any]:
    """带超时保护的模型编码，超时返回 None

    使用子线程执行 model.encode，主线程通过 join(timeout) 等待。
    超时后放弃向量检索，降级为 BM25。
    """
    result_holder: List[Optional[Any]] = [None]
    error_holder: List[Optional[Exception]] = [None]

    def _worker() -> None:
        try:
            result_holder[0] = model.encode(texts, **kwargs)
        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        logger.warning("向量编码超时（%.1f 秒），降级为 BM25 检索", timeout)
        return None

    if error_holder[0] is not None:
        raise error_holder[0]

    return result_holder[0]


def _get_confidence(score: float, threshold_low: float, threshold_medium: float) -> str:
    """根据分数返回置信度等级

    score >= threshold_medium → "high"
    threshold_low <= score < threshold_medium → "medium"
    score < threshold_low → "low"
    """
    if score >= threshold_medium:
        return "high"
    elif score >= threshold_low:
        return "medium"
    return "low"


def search_skills(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """混合检索技能：融合向量语义检索与 BM25 关键词检索，带查询缓存

    降级策略：
      - 嵌入模型不可用时，降级为纯 BM25 检索
      - 向量编码超时时，降级为纯 BM25 检索
    """
    config = _load_config()
    if top_k is None:
        top_k = config.get("top_k", 5)

    cache_key = f"{query}::top_k={top_k}"
    cached = plugin_state.get_query_cache_item(cache_key)
    if cached is not None:
        logger.debug("查询缓存命中: %s", cache_key)
        return cached

    vector_weight = config.get("vector_weight", 0.7)
    bm25_weight = config.get("bm25_weight", 0.3)
    encode_timeout = config.get("encode_timeout", 10)
    threshold_low = config.get("confidence_threshold_low", 0.3)
    threshold_medium = config.get("confidence_threshold_medium", 0.4)

    # ── Determine backend ──
    backend_type = config.get("backend", "lightweight")
    sr_backend = None
    if backend_type == "skillrouter":
        sr_backend = _get_skillrouter_backend(config)
        if sr_backend is not None:
            logger.info("Using SkillRouter backend (Qwen3-0.6B)")
        else:
            logger.warning("SkillRouter backend requested but unavailable, falling back to lightweight")

    model = _load_embedding_model()
    skill_names, skill_embeddings = _get_skill_embeddings()
    skills = _load_skill_index()
    core_skills = _get_core_skills()

    if not skills:
        return []

    # ── P1: 域分组过滤 v2（Top-K 上限 + lark 前缀 + 聚合器 + 子域）──
    domain_config = _load_domain_config()
    _validate_domain_config(domain_config, skills)
    domain = _classify_domain(query, domain_config)
    domain_skill_names, domain_aggregator = _filter_skills_by_domain(
        query, domain, skills, top_k, domain_config
    )

    degraded = False

    # For SkillRouter backend, we use its own encoding; for lightweight, use the model
    if sr_backend is not None:
        # SkillRouter handles its own encoding
        degraded = False
    elif model is None or skill_embeddings is None:
        logger.info("嵌入模型或向量不可用，降级为纯 BM25 检索")
        degraded = True

    # 向量检索（带超时保护 + 域过滤）
    vector_results: List[Dict[str, Any]] = []
    if not degraded:
        try:
            import numpy as np

            if sr_backend is not None:
                # 使用 SkillRouter 后端增量嵌入缓存
                sr_lock = plugin_state.get_sr_embeddings_lock()
                with sr_lock:
                    changed_names: List[str] = []
                    changed_texts: List[str] = []
                    cached_names: List[str] = []
                    cached_embeddings: List[Any] = []

                    for name, info in skills.items():
                        skill_mtime = info.get("mtime", 0.0)
                        cached_emb = plugin_state.get_sr_embedding(name)
                        if cached_emb is not None and cached_emb[1] == skill_mtime:
                            # 缓存命中，复用已有嵌入
                            cached_names.append(name)
                            cached_embeddings.append(cached_emb[0])
                        else:
                            # 新增或 mtime 变化，需要重新编码
                            changed_names.append(name)
                            changed_texts.append(
                                f"{name} | {info.get('description', '')} | {info.get('body_text', '')}"
                            )

                    # 清理缓存中已删除的技能
                    removed = plugin_state.get_sr_embedding_names() - set(skills.keys())
                    for rm_name in removed:
                        plugin_state.remove_sr_embedding(rm_name)

                # 只对变化的技能重新编码
                if changed_texts:
                    new_embeddings = sr_backend.encode_texts(changed_texts)
                    # 更新缓存
                    with sr_lock:
                        for i, name in enumerate(changed_names):
                            plugin_state.set_sr_embedding(name, new_embeddings[i], skills[name].get("mtime", 0.0))
                else:
                    new_embeddings = np.empty((0,))

                # 合并缓存和新编码的结果
                sr_skill_names = cached_names + changed_names
                if cached_embeddings and new_embeddings.size:
                    doc_embeddings = np.vstack([
                        np.array(cached_embeddings),
                        new_embeddings,
                    ])
                elif new_embeddings.size:
                    doc_embeddings = new_embeddings
                elif cached_embeddings:
                    doc_embeddings = np.array(cached_embeddings)
                else:
                    doc_embeddings = np.empty((0,))

                if changed_texts:
                    logger.info(
                        "SkillRouter 增量编码: 缓存 %d, 新编码 %d",
                        len(cached_names), len(changed_names),
                    )

                # Encode query
                query_embedding = sr_backend.encode_query(query)

                # Compute similarities
                if doc_embeddings.size:  # type: ignore[attr-defined]
                    similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
                else:
                    similarities = np.array([])

                # Take top candidates with domain filtering
                fetch_k = top_k * 4 if domain_skill_names else top_k * 2
                top_indices = np.argsort(similarities)[::-1][:fetch_k]
                for idx in top_indices:
                    name = sr_skill_names[idx]
                    if domain_skill_names is not None and name not in domain_skill_names:
                        continue
                    vector_results.append({"name": name, "score": float(similarities[idx])})
                    if len(vector_results) >= top_k * 2:
                        break
            else:
                # Use lightweight model
                query_embedding = _encode_with_timeout(
                    model, [query], encode_timeout,
                    convert_to_numpy=True, normalize_embeddings=True,
                )
                if query_embedding is None:
                    degraded = True
                else:
                    similarities = np.dot(skill_embeddings, query_embedding.T).flatten()
                    # 取更多候选以便域过滤后仍有足够结果
                    fetch_k = top_k * 4 if domain_skill_names else top_k * 2
                    top_indices = np.argsort(similarities)[::-1][:fetch_k]
                    for idx in top_indices:
                        name = skill_names[idx]
                        # 域过滤：跳过不在当前域的技能
                        if domain_skill_names is not None and name not in domain_skill_names:
                            continue
                        vector_results.append({"name": name, "score": float(similarities[idx])})
                        if len(vector_results) >= top_k * 2:
                            break
        except Exception as e:
            logger.error("向量检索失败: %s，降级为 BM25 检索", e)
            degraded = True

    # BM25 检索（带域过滤）
    bm25_results: List[Tuple[str, float]] = []
    try:
        bm25_searcher: Any = _get_bm25_searcher()
        raw_bm25 = bm25_searcher.search(query, top_k=top_k * 4 if domain_skill_names else top_k * 2)
        # 域过滤
        if domain_skill_names is not None:
            bm25_results = [(n, s) for n, s in raw_bm25 if n in domain_skill_names][:top_k * 2]
        else:
            bm25_results = raw_bm25
    except Exception as e:
        logger.error("BM25 检索失败: %s", e)

    # 降级模式：BM25 结果直接作为最终结果
    if degraded:
        results = []
        for name, score in bm25_results[:top_k]:
            if name not in skills:
                continue
            skill = skills[name]
            tier = "core" if name in core_skills else "pool"
            feedback_adj = _feedback_store.get_adjustments(name)
            adjusted_score = score + feedback_adj
            confidence = _get_confidence(adjusted_score, threshold_low, threshold_medium)
            results.append({
                "name": name,
                "category": skill["category"],
                "description": skill["description"],
                "score": round(adjusted_score, 4),
                "tier": tier,
                "confidence": confidence,
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        # v2: 域聚合器优先注入（命中域时自动推荐路由入口技能）
        if domain_aggregator and domain_aggregator in skills:
            if not any(r["name"] == domain_aggregator for r in results):
                max_score = max((r["score"] for r in results), default=0.6)
                results.append({
                    "name": domain_aggregator,
                    "category": skills[domain_aggregator]["category"],
                    "description": skills[domain_aggregator]["description"],
                    "score": round(max_score + 0.1, 4),
                    "tier": "core",
                    "confidence": "high",
                })
        results = results[:top_k]

        if results:
            plugin_state.set_query_cache_item(cache_key, results)

        return results

    # 正常模式：混合融合
    try:
        use_reranker = config.get("use_reranker", True)
        reranker_model_name = config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        hybrid_calibration = config.get("hybrid_calibration", "minmax")

        hybrid = HybridSearcher(config={
            "vector_weight": vector_weight,
            "bm25_weight": bm25_weight,
            "use_reranker": use_reranker,
            "hybrid_calibration": hybrid_calibration,
        })

        # 加载 Reranker 模型（如果启用）
        # When SkillRouter backend is active, skip CrossEncoder loading
        reranker_model = None
        if use_reranker and sr_backend is None:
            reranker_model = hybrid._load_reranker(reranker_model_name, backend=backend_type)

        fused = hybrid.search(
            query=query,
            top_k=top_k * 2,  # 先检索更多候选
            vector_results=vector_results,
            bm25_results=bm25_results,
            skill_names=list(domain_skill_names) if domain_skill_names else skill_names,
        )

        # 使用 Reranker 重排序（如果启用且可用）
        if use_reranker:
            if sr_backend is not None:
                # Use SkillRouter reranker
                try:
                    fused = sr_backend.rerank(query=query, candidates=fused, skills=skills, top_k=top_k)
                    logger.info("SkillRouter reranker 重排序完成: %d 个结果", len(fused))
                except Exception as e:
                    logger.warning("SkillRouter reranker failed: %s, using original results", e)
            elif reranker_model is not None:
                # Use CrossEncoder reranker
                fused = hybrid._rerank_results(
                    query=query,
                    candidates=fused,
                    skills=skills,
                    top_k=top_k,
                    reranker_model=reranker_model
                )
                logger.info("Reranker 重排序完成: %d 个结果", len(fused))

        results = []
        for item in fused:
            name = item["name"]
            if name not in skills:
                continue
            skill = skills[name]
            tier = "core" if name in core_skills else "pool"
            feedback_adj = _feedback_store.get_adjustments(name)
            adjusted_score = item["score"] + feedback_adj
            confidence = _get_confidence(adjusted_score, threshold_low, threshold_medium)
            results.append({
                "name": name,
                "category": skill["category"],
                "description": skill["description"],
                "score": round(adjusted_score, 4),
                "tier": tier,
                "confidence": confidence,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        results = results[:top_k]

        if results:
            plugin_state.set_query_cache_item(cache_key, results)

        return results
    except Exception as e:
        logger.error("混合检索融合失败: %s", e)
        return []


def get_core_skills() -> List[Dict[str, Any]]:
    """获取所有 core 技能"""
    skills = _load_skill_index()
    core_names = _get_core_skills()
    return [
        {"name": name, "category": skills[name]["category"], "description": skills[name]["description"], "tier": "core"}
        for name in core_names if name in skills
    ]


def skill_search_tool(query: str, top_k: int = 5) -> str:
    """工具接口"""
    results = search_skills(query, top_k)
    if not results:
        return json.dumps({"success": False, "message": "No skills found", "results": []}, ensure_ascii=False)
    return json.dumps({"success": True, "query": query, "results": results, "total": len(results)}, ensure_ascii=False)


def skill_pool_snapshot() -> str:
    """获取技能池快照"""
    skills = _load_skill_index()
    core_names = _get_core_skills()

    core = [n for n in core_names if n in skills]
    pool = [n for n in skills if n not in core_names]

    return json.dumps({
        "core": core,
        "pool_count": len(pool),
        "total": len(skills),
    }, ensure_ascii=False)


# ── Hermes 插件接口 ──

def register(ctx: Any) -> None:
    # ctx 校验：容忍 None / 非法类型，避免插件加载器崩溃
    if ctx is None or not hasattr(ctx, "register_hook"):
        logger.warning("skill-router: ctx 无效或缺少 register_hook 方法，跳过注册")
        return

    try:
        config = _load_config()
        if not config.get("enabled", True):
            logger.info("技能路由插件已禁用")
            return

        # 初始化查询缓存
        plugin_state.create_query_cache(
            max_size=config.get("query_cache_max_size", 1000),
            ttl=config.get("query_cache_ttl", 300),
        )

        # 注意：不再在 register 阶段预热嵌入模型，改为首次查询时懒加载
        # 预热会导致 register() 阻塞 30 秒以上（ML 模型加载 + 全量编码）
        logger.info("skill-router: 嵌入模型将延迟到首次查询时加载")

        # 工具定义：优先使用 ctx.register_tool()（标准方式），与其它插件保持一致；
        # 若 ctx 不支持（旧版 Hermes），则回退到 tools.registry.register()。
        _skill_search_schema = {
            "name": "skill_search",
            "description": "Search for relevant skills using semantic embedding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of results", "default": 5}
                },
                "required": ["query"]
            }
        }

        def _skill_search_handler(args: Dict[str, Any], **kwargs) -> str:
            return skill_search_tool(
                query=args.get("query", ""),
                top_k=args.get("top_k", 5)
            )

        _skill_feedback_schema = {
            "name": "skill_feedback",
            "description": "报告技能使用结果，帮助改进路由准确性",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "技能名称"},
                    "query": {"type": "string", "description": "原始查询"},
                    "feedback_type": {"type": "string", "enum": ["success", "skip"], "description": "反馈类型：成功使用或跳过"},
                },
                "required": ["skill_name", "feedback_type"],
            },
        }

        def _skill_feedback_handler(args: Dict[str, Any], **kwargs) -> str:
            return _handle_skill_feedback(
                skill_name=args.get("skill_name", ""),
                query=args.get("query", ""),
                feedback_type=args.get("feedback_type", ""),
            )

        # 优先使用 ctx.register_tool() —— 这是 Hermes 标准接口，且测试框架据此判断注册是否成功
        if hasattr(ctx, "register_tool") and callable(getattr(ctx, "register_tool")):
            ctx.register_tool(
                name="skill_search",
                toolset="skills",
                schema=_skill_search_schema,
                handler=_skill_search_handler,
            )
            ctx.register_tool(
                name="skill_feedback",
                toolset="skills",
                schema=_skill_feedback_schema,
                handler=_skill_feedback_handler,
            )
            logger.debug("skill-router: 通过 ctx.register_tool() 注册工具")
        else:
            # 回退：旧版 Hermes 使用 tools.registry.register()
            try:
                from tools.registry import registry
            except ImportError:
                logger.warning("skill-router: ctx 不支持 register_tool 且 tools.registry 不可用，跳过工具注册")
                return

            registry.register(
                name="skill_search",
                toolset="skills",
                schema=_skill_search_schema,
                handler=_skill_search_handler,
                check_fn=lambda: _load_embedding_model() is not None,
            )
            registry.register(
                name="skill_feedback",
                toolset="skills",
                schema=_skill_feedback_schema,
                handler=_skill_feedback_handler,
            )
            logger.debug("skill-router: 通过 tools.registry.register() 注册工具")

        # 注册 pre_llm_call 钩子
        ctx.register_hook("pre_llm_call", _pre_llm_call_hook)

        # 订阅技能更新事件，刷新嵌入缓存（双向反馈：self_evolution → skill-router）
        try:
            from plugins.plugin_orchestrator.context import get_context
            orch_ctx = get_context()
            if orch_ctx and hasattr(orch_ctx, "subscribe"):
                def _on_skill_updated(event_data: Any) -> None:
                    skill_name = event_data.get("skill_name", "") if isinstance(event_data, dict) else ""
                    if skill_name and skill_name in plugin_state.get_sr_embedding_names():
                        plugin_state.remove_sr_embedding(skill_name)
                        # 同时失效全局嵌入缓存和查询缓存，确保下次查询重新编码
                        plugin_state.invalidate_embedding_cache()
                        plugin_state.clear_query_cache()
                        logger.info("收到 skill_updated 事件，已清除 '%s' 的嵌入缓存", skill_name)
                orch_ctx.subscribe("skill_updated", _on_skill_updated)
                logger.debug("skill-router: 已订阅 skill_updated 事件")
        except Exception:
            pass  # orchestrator 未启用时不影响 skill-router 正常工作

        logger.info("Skill router v3.1 已注册 (模块化 + 分数校准 + Core/Pool分离)")
    except Exception as e:
        logger.exception("skill-router: register 失败: %s", e)


def _handle_skill_feedback(skill_name: str, query: str, feedback_type: str) -> str:
    """处理 skill_feedback 工具调用"""
    _feedback_store.record(skill_name, query, feedback_type)
    return json.dumps({
        "success": True,
        "message": f"已记录技能 {skill_name} 的 {feedback_type} 反馈",
    }, ensure_ascii=False)


def _pre_llm_call_hook(**kwargs) -> Dict[str, Any]:
    """pre_llm_call 钩子：预筛选相关技能并注入上下文

    Core/Pool 分离注入：
      - Core 技能：始终注入（高频技能常驻 system prompt）
      - Pool 技能：语义检索后按置信度过滤注入
        - low (score < 0.3): 不注入 pool 技能（但 core 仍注入）
        - medium (0.3 <= score < 0.4): 注入但标注 [低置信度]
        - high (score >= 0.4): 正常注入
    """
    config = _load_config()
    domain_config = _load_domain_config()

    user_message = kwargs.get("user_message", "")
    if not user_message or len(user_message) < 5:
        return {}

    if user_message.strip().startswith("/"):
        return {}

    top_k = config.get("top_k", 5)

    # 从 plugin_context 读取任务复杂度信息，动态调整 top_k
    plugin_context = kwargs.get("plugin_context")
    if plugin_context:
        model = plugin_context.shared_get("model_selection", "")

        # 检查 shared_state 中的 last_skill_updated，清除过期嵌入缓存（兜底通道）
        last_updated = plugin_context.shared_get("last_skill_updated", "")
        if last_updated and last_updated in plugin_state.get_sr_embedding_names():
            plugin_state.remove_sr_embedding(last_updated)
            plugin_state.invalidate_embedding_cache()
            plugin_state.clear_query_cache()
            plugin_context.shared_set("last_skill_updated", "")  # 消费后清除
            logger.info("检测到 skill_updated (shared_state)，已清除 '%s' 的嵌入缓存", last_updated)

        # 如果是高复杂度任务（model 包含 pro 或 strategy 为 complex），增加技能推荐数
        if model and "pro" in model.lower():
            top_k = min(top_k * 2, 10)  # 高复杂度任务推荐更多技能

    results = search_skills(user_message, top_k)
    skills = _load_skill_index()
    core_skills = _get_core_skills()

    # ── Core 技能：仅列出名称（描述已在 system prompt 中） ──
    core_lines = []
    for name in core_skills:
        if name in skills:
            core_lines.append(f"  - {name}")

    # ── Pool 技能：按置信度过滤 ──
    pool_lines = []
    pool_confidence = "low"
    if results:
        pool_confidence = results[0].get("confidence", "low")
        for r in results:
            if r.get("tier") == "core":
                continue  # core 已单独注入，不重复
            desc = r["description"][:50] + "..." if len(r["description"]) > 50 else r["description"]
            pool_lines.append(f"  - {r['name']}: {desc}")

    # 没有任何技能可注入时跳过
    if not core_lines and not pool_lines:
        return {}
    # pool 置信度过低且无 core 技能时也跳过
    if not core_lines and pool_confidence == "low":
        return {}

    # ── 构造注入上下文（精简版） ──
    parts = ["[Skills]"]
    if core_lines:
        parts.append("Core: " + ", ".join(c.strip().lstrip("- ") for c in core_lines))
    if pool_lines and pool_confidence != "low":
        parts.extend(pool_lines)

    context = "\n".join(parts)

    # 收集推荐的技能域列表
    recommended_domains: List[str] = []
    domain = _classify_domain(user_message, domain_config)
    if domain:
        recommended_domains.append(domain)

    result: Dict[str, Any] = {"context": context}
    result["context_merge"] = {
        "skill_domains": recommended_domains,  # 当前推荐的技能域列表
        "skill_count": len(results),           # 推荐技能数量
    }
    return result
