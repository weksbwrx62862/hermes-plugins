"""
Skill Router Plugin v2.0 — 统一技能路由

合并 skill-router + skill_pool，取两者优点：
  - 微调中文嵌入模型（1402 条训练，80.7% Top-3）
  - core/pool 分层管理
  - pre_llm_call 钩子自动注入
  - 预计算嵌入，<1s 检索

核心设计：
  - core 技能：常驻 system prompt（8-10 个高频技能）
  - pool 技能：按需语义搜索 top-K
  - pre_llm_call：自动注入相关技能推荐

配置 (~/.hermes/config.yaml):
  plugins.skill-router:
    enabled: true
    model_path: ~/.hermes/skills/devops/skill-router-scalable/fine-tuned-model-v5
    top_k: 5
    core_skills:
      - hermes-agent
      - skill-creator
      - web-search-china
"""

import os
import json
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CacheManager:
    """统一 TTL 缓存管理器，线程安全

    封装带过期时间的缓存逻辑：
      - get(): 获取缓存值，过期或未设置时返回 None
      - set(value): 写入缓存并刷新时间戳
      - invalidate(): 主动使缓存失效
    """

    def __init__(self, ttl: float):
        self._ttl = ttl
        self._value = None
        self._timestamp: float = 0.0
        self._lock = threading.Lock()

    def get(self):
        """获取缓存值，过期或未设置时返回 None"""
        with self._lock:
            if self._value is None:
                return None
            if time.time() - self._timestamp >= self._ttl:
                return None
            return self._value

    def set(self, value):
        """设置缓存值并更新时间戳"""
        with self._lock:
            self._value = value
            self._timestamp = time.time()

    def invalidate(self):
        """使缓存失效"""
        with self._lock:
            self._value = None
            self._timestamp = 0.0


class QueryCache:
    """LRU + TTL 查询缓存，线程安全

    基于 OrderedDict 实现 LRU 淘汰策略：
      - get(): 命中时移动到末尾（最近使用），过期返回 None
      - set(): 写入缓存，超容量时淘汰最旧条目（头部）
      - clear(): 清空所有缓存
    """

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self._max_size = max_size
        self._ttl = ttl
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[List[Dict]]:
        """获取缓存值，过期返回 None，命中时更新 LRU 顺序"""
        with self._lock:
            if key not in self._cache:
                return None
            value, timestamp = self._cache[key]
            if time.time() - timestamp >= self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: List[Dict]):
        """写入缓存，超容量时淘汰最旧条目"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()


# ── 缓存实例 ──
_config_cache = CacheManager(ttl=300)       # 配置缓存：300 秒 TTL
_skill_index_cache = CacheManager(ttl=300)  # 索引缓存：300 秒 TTL
_embedding_cache = CacheManager(ttl=300)    # 嵌入缓存：与索引联动
_bm25_cache = CacheManager(ttl=300)        # BM25 搜索器缓存：与索引缓存联动


class FeedbackStore:
    """技能反馈存储，JSONL 持久化，线程安全

    记录技能使用反馈（成功/跳过），影响后续路由评分：
      - 成功使用: +0.05
      - 被跳过: -0.02
      - 超过 24 小时的反馈权重指数衰减（半衰期 12 小时）
    """

    FEEDBACK_FILE = Path.home() / ".hermes" / "skill_router_feedback.jsonl"
    SUCCESS_DELTA = 0.05
    SKIP_DELTA = -0.02
    DECAY_THRESHOLD_HOURS = 24.0
    HALF_LIFE_HOURS = 12.0

    def __init__(self):
        self._lock = threading.Lock()
        self._records: List[Dict] = []
        self._load_history()

    def _load_history(self):
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

    def record(self, skill_name: str, query: str, feedback_type: str):
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
        return total

    def _decay_weight(self, age_hours: float) -> float:
        """衰减函数：24 小时内权重为 1.0，之后指数衰减"""
        if age_hours <= self.DECAY_THRESHOLD_HOURS:
            return 1.0
        return 0.5 ** ((age_hours - self.DECAY_THRESHOLD_HOURS) / self.HALF_LIFE_HOURS)


_feedback_store = FeedbackStore()


class BM25Searcher:
    """基于 BM25 算法的关键词检索器

    使用 jieba 分词（不可用时降级为字符级分词），在初始化时预计算
    文档分词结果和 IDF 值，search 时仅计算查询与文档的 TF 加权得分。
    """

    _K1 = 1.5
    _B = 0.75

    def __init__(self, skills: Dict[str, Dict[str, Any]]):
        self._use_jieba = False
        try:
            import jieba
            jieba.setLogLevel(logging.WARNING)
            self._use_jieba = True
            logger.debug("BM25Searcher: 使用 jieba 分词")
        except ImportError:
            logger.debug("BM25Searcher: jieba 不可用，降级为字符级分词")

        self._doc_tokens: Dict[str, List[str]] = {}
        self._idf: Dict[str, float] = {}
        self._avg_dl: float = 0.0
        self._doc_len: Dict[str, int] = {}

        if not skills:
            return

        doc_freq: Dict[str, int] = {}
        total_dl = 0

        for name, info in skills.items():
            text = f"{name} {info.get('description', '')} {info.get('body_text', '')[:500]}"
            tokens = self._tokenize(text)
            self._doc_tokens[name] = tokens
            self._doc_len[name] = len(tokens)
            total_dl += len(tokens)

            seen = set()
            for t in tokens:
                if t not in seen:
                    doc_freq[t] = doc_freq.get(t, 0) + 1
                    seen.add(t)

        n_docs = len(skills)
        self._avg_dl = total_dl / n_docs if n_docs > 0 else 1.0

        for term, df in doc_freq.items():
            self._idf[term] = max(0.0, ((n_docs - df + 0.5) / (df + 0.5) + 1.0))

    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词"""
        if self._use_jieba:
            import jieba
            return [w for w in jieba.cut(text) if w.strip()]
        return [c for c in text if c.strip()]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """检索与查询最相关的技能，返回 (skill_name, score) 列表"""
        if not self._doc_tokens:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = {}

        for name, doc_tokens in self._doc_tokens.items():
            tf_map: Dict[str, int] = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            dl = self._doc_len[name]
            score = 0.0
            for qt in query_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                idf = self._idf.get(qt, 0.0)
                numerator = tf * (self._K1 + 1)
                denominator = tf + self._K1 * (1 - self._B + self._B * dl / self._avg_dl)
                score += idf * numerator / denominator

            if score > 0:
                scores[name] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class HybridSearcher:
    """混合检索器：融合向量检索与 BM25 关键词检索

    对向量分数和 BM25 分数分别归一化到 [0, 1] 后，按配置权重加权融合。
    支持可选的 Reranker 重排序（交叉编码器）。
    """
    
    # Reranker 缓存（类级别，线程安全）
    _reranker_model = None
    _reranker_lock = threading.Lock()
    
    @classmethod
    def _load_reranker(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """加载交叉编码器 Reranker（延迟加载，线程安全）"""
        if cls._reranker_model is not None:
            return cls._reranker_model
        
        with cls._reranker_lock:
            if cls._reranker_model is not None:
                return cls._reranker_model
            
            try:
                from sentence_transformers import CrossEncoder
                # 展开 ~ 路径
                resolved_path = os.path.expanduser(model_name)
                logger.info("加载 Reranker 模型: %s", resolved_path)
                cls._reranker_model = CrossEncoder(resolved_path, device="cpu")
                logger.info("Reranker 模型加载完成")
                return cls._reranker_model
            except Exception as e:
                logger.warning("Reranker 模型加载失败: %s，将跳过重排序", e)
                return None
    
    @staticmethod
    def _rerank_results(query: str, candidates: List[Dict], skills: Dict, top_k: int, reranker_model=None) -> List[Dict]:
        """使用交叉编码器对候选结果重排序
        
        参数:
            query: 查询文本
            candidates: 候选结果列表（包含 name 和 score）
            skills: 技能索引字典
            top_k: 返回数量
            reranker_model: 预加载的 Reranker 模型
        
        返回:
            重排序后的结果列表
        """
        if not candidates or len(candidates) <= 1:
            return candidates[:top_k]
        
        # 加载 Reranker
        if reranker_model is None:
            reranker_model = HybridSearcher._load_reranker()
        
        if reranker_model is None:
            logger.debug("Reranker 不可用，返回原始结果")
            return candidates[:top_k]
        
        try:
            # 构造 (query, doc) 对
            pairs = []
            for item in candidates:
                name = item["name"]
                skill = skills.get(name, {})
                # 使用技能名称 + 描述作为文档
                doc = f"{name}: {skill.get('description', '')}"
                pairs.append((query, doc))
            
            # 交叉编码器打分
            scores = reranker_model.predict(pairs, show_progress_bar=False)
            
            # 重新排序
            scored_candidates = []
            for i, item in enumerate(candidates):
                scored_candidates.append({
                    **item,
                    "rerank_score": float(scores[i]),
                    "original_score": item["score"],
                    "score": float(scores[i])  # 使用 Reranker 分数作为最终分数
                })
            
            scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            logger.debug("Reranker 重排序完成: %d 个候选，返回 top %d", len(candidates), top_k)
            return scored_candidates[:top_k]
            
        except Exception as e:
            logger.error("Reranker 重排序失败: %s，返回原始结果", e)
            return candidates[:top_k]

    def __init__(self, vector_weight: float = 0.7, bm25_weight: float = 0.3, use_reranker: bool = True):
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._use_reranker = use_reranker
        self._reranker_model = None

    @staticmethod
    def _normalize_scores(score_map: Dict[str, float]) -> Dict[str, float]:
        """将分数归一化到 [0, 1] 范围（min-max 归一化）"""
        if not score_map:
            return {}
        values = list(score_map.values())
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return {k: 1.0 for k in score_map}
        return {k: (v - min_v) / (max_v - min_v) for k, v in score_map.items()}

    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_results: List[Dict] = None,
        bm25_results: List[Tuple[str, float]] = None,
        skill_names: List[str] = None,
    ) -> List[Dict]:
        """融合向量与 BM25 检索结果

        参数:
            query: 查询文本（保留接口一致性，实际不参与计算）
            top_k: 返回结果数量
            vector_results: 向量检索结果列表，每项含 name 和 score
            bm25_results: BM25 检索结果列表，每项为 (name, score)
            skill_names: 全量技能名列表（用于兜底遍历）

        返回:
            融合后的结果列表，按加权分数降序排列
        """
        vector_results = vector_results or []
        bm25_results = bm25_results or []

        vector_score_map: Dict[str, float] = {r["name"]: r["score"] for r in vector_results}
        bm25_score_map: Dict[str, float] = dict(bm25_results)

        norm_vector = self._normalize_scores(vector_score_map)
        norm_bm25 = self._normalize_scores(bm25_score_map)

        all_names: Set[str] = set()
        if skill_names:
            all_names.update(skill_names)
        all_names.update(norm_vector.keys())
        all_names.update(norm_bm25.keys())

        fused: List[Dict] = []
        for name in all_names:
            v_score = norm_vector.get(name, 0.0)
            b_score = norm_bm25.get(name, 0.0)
            weighted = self._vector_weight * v_score + self._bm25_weight * b_score
            fused.append({"name": name, "score": weighted})

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]

# ── 查询缓存实例 ──
_query_cache: Optional[QueryCache] = None

# ── 模型缓存（无 TTL，加载一次常驻） ──
_MODEL_CACHE = None
_MODEL_CACHE_LOCK = threading.Lock()

# ── 默认配置 ──
_DEFAULT_CONFIG = {
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
    # "reranker_model": "~/.hermes/plugins/skill-router/models/bge-reranker-v2-m3",  # 中文精准版(2.2GB, ~1.2s)
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # 快速版(80MB, ~60ms)
    "query_cache_ttl": 300,
    "query_cache_max_size": 1000,
    "encode_timeout": 10,
    "confidence_threshold_low": 0.3,
    "confidence_threshold_medium": 0.4,
}

# ── 域分组映射（P1: 先选域再选技能）──
_DOMAIN_MAP = {
    "code": ["software-development", "devops", "github", "autonomous-ai-agents", "mcp", "dogfood"],
    "ml": ["mlops", "training", "inference", "models", "evaluation", "data-science"],
    "creative": ["creative", "media"],
    "productivity": ["productivity", "note-taking", "email", "apple"],
    "research": ["research", "red-teaming"],
    "finance": ["finance"],
    "other": ["gaming", "leisure", "smart-home", "social-media", "cloud"],
}

# 域关键词映射（用于快速域分类）
_DOMAIN_KEYWORDS = {
    "code": ["代码", "编程", "开发", "bug", "debug", "git", "github", "pr", "commit", "deploy", "docker", "kubernetes", "ci", "cd", "plugin", "插件", "脚本", "script", "python", "javascript", "typescript", "rust", "go", "java"],
    "ml": ["模型", "训练", "微调", "推理", "inference", "train", "finetune", "lora", "gpu", "cuda", "vllm", "gguf", "量化", "benchmark", "评估"],
    "creative": ["图", "画", "设计", "视频", "音乐", "动画", "ascii", "pixel", "漫画", "illustration", "diagram", "svg", "艺术", "art", "手绘", "sketch", "excalidraw"],
    "productivity": ["文件", "文档", "邮件", "日历", "笔记", "notion", "obsidian", "pdf", "表格", "sheet", "drive", "ppt", "幻灯片", "演示文稿", "演示"],
    "research": ["论文", "搜索", "研究", "arxiv", "知识", "wiki", "新闻", "news", "report"],
    "finance": ["股票", "基金", "加密", "货币", "投资", "理财", "行情", "价格", "收益", "分析", "portfolio", "stock", "fund", "crypto", "etf"],
}

def _classify_domain(query: str) -> Optional[str]:
    """根据查询文本快速分类到功能域

    使用关键词匹配，O(1) 复杂度。返回域名称或 None（无法分类）。
    """
    query_lower = query.lower()
    scores = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[domain] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


# ── core/pool 分类 ──
_POOL_INDEX = Path.home() / ".hermes" / "skills" / ".pool_index.json"
_POOL_CONFIG = Path.home() / ".hermes" / "skills" / ".pool_config.json"


def _load_config() -> Dict[str, Any]:
    """加载插件配置，带 30 秒 TTL 缓存"""
    cached = _config_cache.get()
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

    _config_cache.set(result)
    return result


def _get_model_path() -> str:
    return os.path.expanduser(_load_config()["model_path"])


def _get_db_path() -> str:
    return os.path.expanduser(_load_config()["db_path"])


def _get_core_skills() -> Set[str]:
    """获取 core 技能列表"""
    config = _load_config()
    core = set(config.get("core_skills", []))

    if _POOL_CONFIG.exists():
        try:
            with open(_POOL_CONFIG) as f:
                pool_cfg = json.load(f)
            core.update(pool_cfg.get("core_skills", []))
        except Exception:
            pass

    return core


def _load_embedding_model():
    """加载嵌入模型（无 TTL，全局单例）"""
    global _MODEL_CACHE
    with _MODEL_CACHE_LOCK:
        if _MODEL_CACHE is not None:
            return _MODEL_CACHE
        model_path = _get_model_path()
        if not os.path.exists(model_path):
            logger.warning("技能路由模型未找到: %s", model_path)
            return None
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_path, device="cpu")
            _MODEL_CACHE = model
            logger.info("已加载技能路由模型: %s", model_path)
            return model
        except Exception as e:
            logger.error("加载技能路由模型失败: %s", e)
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


def _save_embedding_mtimes(mtimes: Dict[str, float]):
    """保存各技能的 mtime 记录到 JSON 文件"""
    try:
        with open(_EMBEDDING_MTIMES_PATH, "w", encoding="utf-8") as f:
            json.dump(mtimes, f, ensure_ascii=False)
    except Exception as e:
        logger.warning("保存嵌入 mtime 记录失败: %s", e)


def _load_skill_index() -> Dict[str, Dict[str, Any]]:
    """加载技能索引，带 60 秒 TTL 缓存

    索引缓存过期时，同步失效嵌入缓存，确保数据一致性。
    同时读取每个技能的 mtime（修改时间），用于增量嵌入更新。
    如果 skills 表没有 mtime 列，自动 ALTER TABLE 添加。
    """
    cached = _skill_index_cache.get()
    if cached is not None:
        return cached

    _embedding_cache.invalidate()
    _bm25_cache.invalidate()

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
        skills = {}
        for row in cursor.fetchall():
            skill_name, category, description, body, mtime = row
            skills[skill_name] = {
                "category": category or "",
                "description": description or "",
                "body_text": body or "",
                "mtime": mtime or 0.0,
            }
        conn.close()
        _skill_index_cache.set(skills)
        logger.info("已从索引加载 %d 个技能", len(skills))
        return skills
    except Exception as e:
        logger.error("加载技能索引失败: %s", e)
        return {}


def _get_skill_embeddings():
    """获取技能嵌入向量，支持增量更新

    通过对比当前技能 mtime 与上次嵌入时的 mtime，实现增量更新：
      - 新增技能：追加嵌入
      - 修改技能（mtime 变化）：重新嵌入并替换
      - 删除技能：移除对应嵌入
      - 未变更技能：保留现有嵌入
    无任何变更时直接返回缓存。
    """
    cached = _embedding_cache.get()
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
            changed_texts = []
            changed_names = sorted(changed)
            for name in changed_names:
                skill = skills[name]
                text = f"{name} {skill['description']} {skill['body_text'][:500]}"
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

        _embedding_cache.set((new_names, new_embeddings))

        # 保存新的 mtime 记录
        _save_embedding_mtimes(curr_mtimes)

        logger.info(
            "增量更新嵌入: 保留 %d, 变更 %d, 删除 %d",
            len(keep_names), len(changed), len(deleted),
        )
        return new_names, new_embeddings

    # 全量计算：无缓存或首次加载
    skill_names = list(skills.keys())
    skill_texts = []
    for name in skill_names:
        skill = skills[name]
        text = f"{name} {skill['description']} {skill['body_text'][:500]}"
        skill_texts.append(text)

    embeddings = model.encode(
        skill_texts, convert_to_numpy=True,
        normalize_embeddings=True, batch_size=32,
    )
    _embedding_cache.set((skill_names, embeddings))

    # 保存 mtime 记录
    _save_embedding_mtimes(curr_mtimes)

    logger.info("已预计算 %d 个技能的嵌入", len(skill_names))
    return skill_names, embeddings


def _get_bm25_searcher() -> BM25Searcher:
    """获取 BM25Searcher 实例，带 TTL 缓存

    缓存与索引联动：索引刷新时 BM25 搜索器同步失效重建。
    """
    cached = _bm25_cache.get()
    if cached is not None:
        return cached
    skills = _load_skill_index()
    searcher = BM25Searcher(skills)
    _bm25_cache.set(searcher)
    logger.info("已构建 BM25Searcher（%d 个技能）", len(skills))
    return searcher


def _encode_with_timeout(model, texts, timeout: float, **kwargs) -> Optional[Any]:
    """带超时保护的模型编码，超时返回 None

    使用子线程执行 model.encode，主线程通过 join(timeout) 等待。
    超时后放弃向量检索，降级为 BM25。
    """
    result_holder = [None]
    error_holder = [None]

    def _worker():
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


def search_skills(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """混合检索技能：融合向量语义检索与 BM25 关键词检索，带查询缓存

    降级策略：
      - 嵌入模型不可用时，降级为纯 BM25 检索
      - 向量编码超时时，降级为纯 BM25 检索
    """
    config = _load_config()
    if top_k is None:
        top_k = config.get("top_k", 5)

    cache_key = f"{query}::top_k={top_k}"
    if _query_cache is not None:
        cached = _query_cache.get(cache_key)
        if cached is not None:
            logger.debug("查询缓存命中: %s", cache_key)
            return cached

    vector_weight = config.get("vector_weight", 0.7)
    bm25_weight = config.get("bm25_weight", 0.3)
    encode_timeout = config.get("encode_timeout", 10)
    threshold_low = config.get("confidence_threshold_low", 0.3)
    threshold_medium = config.get("confidence_threshold_medium", 0.4)

    model = _load_embedding_model()
    skill_names, skill_embeddings = _get_skill_embeddings()
    skills = _load_skill_index()
    core_skills = _get_core_skills()

    if not skills:
        return []

    # ── P1: 域分组过滤 ──
    # 先用关键词快速分类查询域，缩小搜索空间
    domain = _classify_domain(query)
    domain_skill_names = None
    if domain and domain in _DOMAIN_MAP:
        domain_categories = set(_DOMAIN_MAP[domain])
        # 域内技能（不含 uncategorized，避免污染）
        domain_only = {
            name for name, info in skills.items()
            if info.get("category") in domain_categories
        }
        # 特殊域：finance 类技能多在 uncategorized 中，按名称匹配
        if domain == "finance":
            finance_keywords = {"stock", "fund", "finance", "portfolio", "market", "crypto", "macro", "analysis", "technical", "news"}
            for name, info in skills.items():
                name_lower = name.lower()
                if any(kw in name_lower for kw in finance_keywords):
                    domain_only.add(name)
        # 如果域内技能太少（< top_k），加入 uncategorized 作为兜底
        if len(domain_only) >= top_k:
            domain_skill_names = domain_only
        else:
            domain_skill_names = {
                name for name, info in skills.items()
                if info.get("category") in domain_categories
                or info.get("category") == "uncategorized"
            }
        logger.debug("域分类: %s → %d/%d 个候选技能", domain, len(domain_skill_names), len(skills))

    degraded = False

    if model is None or skill_embeddings is None:
        logger.info("嵌入模型或向量不可用，降级为纯 BM25 检索")
        degraded = True

    # 向量检索（带超时保护 + 域过滤）
    vector_results: List[Dict] = []
    if not degraded:
        try:
            import numpy as np
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
        bm25_searcher = _get_bm25_searcher()
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
        results = results[:top_k]

        if _query_cache is not None and results:
            _query_cache.set(cache_key, results)

        return results

    # 正常模式：混合融合
    try:
        use_reranker = config.get("use_reranker", True)
        reranker_model_name = config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        hybrid = HybridSearcher(vector_weight=vector_weight, bm25_weight=bm25_weight, use_reranker=use_reranker)
        
        # 加载 Reranker 模型（如果启用）
        reranker_model = None
        if use_reranker:
            reranker_model = hybrid._load_reranker(reranker_model_name)
        
        fused = hybrid.search(
            query=query,
            top_k=top_k * 2,  # 先检索更多候选
            vector_results=vector_results,
            bm25_results=bm25_results,
            skill_names=list(domain_skill_names) if domain_skill_names else skill_names,
        )
        
        # 使用 Reranker 重排序（如果启用且可用）
        if use_reranker and reranker_model is not None:
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

        if _query_cache is not None and results:
            _query_cache.set(cache_key, results)

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

def register(ctx):
    global _query_cache
    from tools.registry import registry

    config = _load_config()
    if not config.get("enabled", True):
        logger.info("技能路由插件已禁用")
        return

    # 初始化查询缓存
    _query_cache = QueryCache(
        max_size=config.get("query_cache_max_size", 1000),
        ttl=config.get("query_cache_ttl", 300),
    )

    # 预热
    try:
        _get_skill_embeddings()
    except Exception as e:
        logger.warning("预计算技能嵌入失败: %s", e)

    # 注册工具
    registry.register(
        name="skill_search",
        toolset="skills",
        schema={
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
        },
        handler=lambda args, **kw: skill_search_tool(
            query=args.get("query", ""),
            top_k=args.get("top_k", 5)
        ),
        check_fn=lambda: _load_embedding_model() is not None,
    )

    # 注册 skill_feedback 工具
    registry.register(
        name="skill_feedback",
        toolset="skills",
        schema={
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
        },
        handler=lambda args, **kw: _handle_skill_feedback(
            skill_name=args.get("skill_name", ""),
            query=args.get("query", ""),
            feedback_type=args.get("feedback_type", ""),
        ),
    )

    # 注册 pre_llm_call 钩子
    ctx.register_hook("pre_llm_call", _pre_llm_call_hook)
    logger.info("Skill router v3.0 已注册 (域分组 + 中文Reranker + Core/Pool分离)")


def _handle_skill_feedback(skill_name: str, query: str, feedback_type: str) -> str:
    """处理 skill_feedback 工具调用"""
    _feedback_store.record(skill_name, query, feedback_type)
    return json.dumps({
        "success": True,
        "message": f"已记录技能 {skill_name} 的 {feedback_type} 反馈",
    }, ensure_ascii=False)


def _pre_llm_call_hook(**kwargs) -> Dict[str, str]:
    """pre_llm_call 钩子：预筛选相关技能并注入上下文

    Core/Pool 分离注入：
      - Core 技能：始终注入（高频技能常驻 system prompt）
      - Pool 技能：语义检索后按置信度过滤注入
        - low (score < 0.3): 不注入 pool 技能（但 core 仍注入）
        - medium (0.3 <= score < 0.4): 注入但标注 [低置信度]
        - high (score >= 0.4): 正常注入
    """
    config = _load_config()

    user_message = kwargs.get("user_message", "")
    if not user_message or len(user_message) < 5:
        return {}

    if user_message.strip().startswith("/"):
        return {}

    top_k = config.get("top_k", 5)
    results = search_skills(user_message, top_k)
    skills = _load_skill_index()
    core_skills = _get_core_skills()

    # ── Core 技能：始终注入 ──
    core_lines = []
    for name in core_skills:
        if name in skills:
            info = skills[name]
            desc = info["description"][:80] + "..." if len(info["description"]) > 80 else info["description"]
            core_lines.append(f"  - {name}[core]: {desc}")

    # ── Pool 技能：按置信度过滤 ──
    pool_lines = []
    pool_confidence = "low"
    if results:
        pool_confidence = results[0].get("confidence", "low")
        for r in results:
            if r.get("tier") == "core":
                continue  # core 已单独注入，不重复
            desc = r["description"][:80] + "..." if len(r["description"]) > 80 else r["description"]
            score_tag = f" (score={r['score']:.2f})" if pool_confidence == "medium" else ""
            pool_lines.append(f"  - {r['name']}{score_tag}: {desc}")

    # 没有任何技能可注入时跳过
    if not core_lines and not pool_lines:
        return {}
    # pool 置信度过低且无 core 技能时也跳过
    if not core_lines and pool_confidence == "low":
        return {}

    # ── 构造注入上下文 ──
    parts = ["[Skill Router: Related skills]"]

    if core_lines:
        parts.append("[Core skills — always available]")
        parts.extend(core_lines)

    if pool_lines and pool_confidence != "low":
        label = "[Recommended skills]"
        if pool_confidence == "medium":
            label = "[Recommended skills] [低置信度]"
        parts.append(label)
        parts.extend(pool_lines)

    parts.append("[Load with skill_view(name)]")
    context = "\n".join(parts)

    return {"context": context}
