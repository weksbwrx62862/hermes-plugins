"""KnowledgeGraph — SQLite 知识图谱。

参考 MemOS 的知识图谱设计 + MemPalace 的时序三元组：
  - 实体自动抽取：从记忆内容中提取实体和关系
  - 关系推理：基于已有三元组推断隐含关系
  - 时序有效性：valid_from/valid_to 时间过滤
  - 图谱检索通道：1-hop 扩展 + 关系网络发现

Phase 3 完整实现。
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)


# ─── 实体抽取模式 ────────────────────────────────────────────

# 中文实体模式：人名/地名/机构/技术术语
_ZH_ENTITY_PATTERNS = [
    r"[\u4e00-\u9fff]{2,4}(?=公司|团队|项目|系统|框架|平台|模块|服务|接口|数据库)",  # 组织/系统名
    r"(?<=用户|客户|同事|老板|领导|朋友)[\u4e00-\u9fff]{2,3}",  # 人名
]

# 英文实体模式
_EN_ENTITY_PATTERNS = [
    r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b",  # CamelCase
    r"\b[A-Z]{2,}\b",  # 缩写 API, SQL, etc.
    r"\b[a-z]+(?:-[a-z]+)+\b",  # kebab-case
    # 技术名词：不用 \b（中英混合时无效），但用前后断言防止子串匹配
    # 例：匹配 "Docker部署" 中的 "Docker"，但不匹配 "Pythonic" 中的 "Python"
    r"(?<![A-Za-z])(Python|Java|Go|Rust|TypeScript|React|Vue|Docker|K8s|Redis|MySQL|PostgreSQL|MongoDB|Neo4j|ChromaDB|SQLite)(?![A-Za-z])",
]

# 通用实体模式：从关系三元组中提取的实体
_GENERIC_ENTITY_PATTERNS = [
    # 中文关键词前面的名词（如"前端使用React"中的"前端"）
    r"(?<=[，。、\s])[\u4e00-\u9fff]{2,6}(?=使用|采用|选用|基于|依赖|运行)",
    # 中文关键词后面的英文技术名词
    r"(?:使用|采用|选用|基于|依赖)\s*([A-Z][A-Za-z0-9_.-]*)",
]

# 关系模式：从文本中提取 (主语, 关系, 宾语) 三元组
_RELATION_PATTERNS = [
    # 中文关系
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:使用|采用|选用|基于|依赖|运行在)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "uses",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:属于|归入|隶属于)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "belongs_to",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:导致|引起|造成|触发)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "causes",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:替代|取代|替换|升级为)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "replaces",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:连接|关联|对应|映射到)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "connects_to",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:优于|胜过|好于)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "better_than",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:包含|包括|由.*组成)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "contains",
    ),
    (r"([\u4e00-\u9fff]{2,8})\s*(?:在|于)\s*([\u4e00-\u9fff]{2,8})\s*(?:中|里|上)", "located_in"),
    # 英文关系
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:uses?|depends?\s+on|relies?\s+on)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "uses",
    ),
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:causes?|leads?\s+to|triggers?)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "causes",
    ),
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:replaces?|supersedes?)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "replaces",
    ),
    (
        r"(\b[A-Za-z][A-Za-z0-9_.-]*)\s+(?:contains?|includes?)\s+(\b[A-Za-z][A-Za-z0-9_.-]*)",
        "contains",
    ),
]

# 否定关系模式
_NEGATION_PATTERNS = [
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:不|并非|没有|无法|不能)\s*(?:使用|采用|依赖|支持)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "not_uses",
    ),
    (
        r"([\u4e00-\u9fff]{2,8})\s*(?:不同于|区别于|不是)\s*([\u4e00-\u9fffA-Za-z0-9_.-]+)",
        "differs_from",
    ),
]


# ─── 实体抽取函数 ────────────────────────────────────────────


def extract_entities(text: str) -> list[str]:
    """从文本中提取实体。

    ★ P1方案四：优先使用 jieba 分词 + 词性标注（若可用），
    与规则正则互补，提升通用命名实体覆盖率。
    """
    entities: set[str] = set()

    # 优先路径：jieba NER（人名nr/地名ns/机构名nt/其他专名nz）
    try:
        import jieba.posseg as pseg

        for word, flag in pseg.lcut(text):
            if flag in ("nr", "ns", "nt", "nz") and len(word) >= 2:
                entities.add(word)
    except ImportError:
        pass

    # 中文实体（规则正则，与 jieba 互补）
    for pattern in _ZH_ENTITY_PATTERNS:
        matches = re.findall(pattern, text)
        entities.update(matches)

    # 英文实体
    for pattern in _EN_ENTITY_PATTERNS:
        matches = re.findall(pattern, text)
        entities.update(matches)

    # 通用实体模式
    for pattern in _GENERIC_ENTITY_PATTERNS:
        matches = re.findall(pattern, text)
        entities.update(matches)

    # 从三元组中提取的实体（主语和宾语也是实体）
    triples = extract_triples(text)
    for subj, _, obj in triples:
        entities.add(subj)
        entities.add(obj)

    # ★ 裸中文人名检测（无需上下文标记）
    # 使用与 _classify_entity_poleo 相同的姓氏列表
    _cn_surnames_ent = (
        "王李张刘陈杨黄赵周吴徐孙马胡朱郭何罗高林郑梁谢唐许冯宋韩邓彭曹曾田萧"
        "潘袁蔡蒋余于杜叶程苏魏吕丁任卢姚钟姜崔谭陆汪范金石廖贾夏韦付方白邹孟"
        "熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤温康施文"
        "牛樊葛邢安齐易乔伍庞颜倪庄聂章鲁岳翟殷詹申欧耿关兰焦俞左柳甘祝包宁尚"
        "司马欧阳上官皇甫诸葛令狐司徒"
    )
    _concept_words_set = {
        "问题", "方法", "方案", "结果", "数据", "功能", "配置", "部署",
        "测试", "需求", "设计", "架构", "实现", "开发", "优化", "模块",
        "系统", "服务", "接口", "组件", "引擎", "管道", "通道", "版本",
        "发布", "编辑", "文档", "搜索", "索引", "缓存", "进程", "线程",
    }
    # 扫描所有2-4字中文序列，检测是否为人名
    # 使用滑动窗口，逐字从姓氏位置开始匹配
    chars = list(text)
    seen_names = set()
    # 句子连接词/动词/介词 — 不应出现在人名中
    _name_break_chars = set("的在了和是就也都很到说要去了不有这那个什么怎么可以需要通过使用进行包括负责参与处理完成检查确认执行部署配置优化升级返回发送接收")
    for i in range(len(chars)):
        if chars[i] not in _cn_surnames_ent:
            continue
        for name_len in (3, 2):  # 中文名通常 2-3 字
            if i + name_len > len(chars):
                continue
            cand = "".join(chars[i:i + name_len])
            if not re.match(r"^[\u4e00-\u9fff]+$", cand):
                continue
            # 末尾字符不应是句连接词
            if cand[-1] in _name_break_chars:
                continue
            if cand in entities or cand in seen_names:
                continue
            if cand in _concept_words_set:
                continue
            entities.add(cand)
            seen_names.add(cand)
            break

    # 去除太短的实体
    return [e for e in entities if len(e) >= 2]


def extract_triples(text: str, use_llm: bool = True) -> list[tuple[str, str, str]]:
    """从文本中提取 (主语, 关系, 宾语) 三元组。

    优先使用正则提取，结果不足时回退到 LLM 提取。
    Returns: List of (subject, predicate, object) tuples
    """
    triples: list[tuple[str, str, str]] = []

    # 1. 正则提取（快速）
    for pattern, predicate in _RELATION_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                subj, obj = match[0], match[1]
                if subj and obj and subj != obj:
                    triples.append((subj, predicate, obj))

    # 否定关系
    for pattern, predicate in _NEGATION_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                subj, obj = match[0], match[1]
                if subj and obj and subj != obj:
                    triples.append((subj, predicate + "_not", obj))

    # 2. LLM 回退：正则结果不足 2 条且文本较长时使用
    if use_llm and len(triples) < 2 and len(text) > 30:
        try:
            llm_triples = _extract_triples_llm(text)
            # 去重合并
            existing = {(s.lower(), p.lower(), o.lower()) for s, p, o in triples}
            for s, p, o in llm_triples:
                key = (s.lower(), p.lower(), o.lower())
                if key not in existing:
                    triples.append((s, p, o))
                    existing.add(key)
        except Exception as e:
            logger.debug("LLM triple extraction failed, using regex only: %s", e)

    return triples


def _extract_triples_llm(text: str) -> list[tuple[str, str, str]]:
    """使用 LLM 从文本中抽取知识三元组（轻量级，仅在正则不足时调用）。"""
    import yaml
    from openai import OpenAI

    # 读取 API 配置
    config_path = Path.home() / ".hermes" / "config.yaml"
    if not config_path.exists():
        return []
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    providers = cfg.get("providers", {})
    ds = providers.get("openai", {})
    if not ds.get("api_key"):
        return []

    client = OpenAI(api_key=ds["api_key"], base_url=ds.get("base_url", ""))

    prompt = f"""从以下文本中抽取知识三元组 (subject, predicate, object)。
要求：
- 抽取 1-3 个最重要的三元组
- subject/object 是有意义的实体（技术名、工具、概念、人名、地名）
- predicate 是中文或英文关系词（使用、属于、推荐、避免、解决、配置、导致等）
- 跳过信息量不足的文本

文本：
{text[:500]}

返回 JSON 数组：[{{"s": "...", "p": "...", "o": "..."}}]
只返回 JSON。"""

    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )
    raw = resp.choices[0].message.content or ""

    import json
    m = re.search(r"\[\s\S]*\]", raw)
    if not m:
        return []
    items = json.loads(m.group())
    return [
        (item["s"].strip(), item["p"].strip(), item["o"].strip())
        for item in items
        if item.get("s") and item.get("p") and item.get("o")
    ]
def infer_relations(existing_triples: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """基于已有三元组推理隐含关系。

    推理规则:
      - 传递性: A uses B, B uses C → A uses C (transitive)
      - 互逆: A belongs_to B → B contains A
      - 替代链: A replaces B, B replaces C → A replaces C
    """
    inferred: list[tuple[str, str, str]] = []

    # 建立主语→(关系→宾语)的索引
    subj_map: dict[str, dict[str, list[str]]] = {}
    for t in existing_triples:
        s, p, o = t.get("subject", ""), t.get("predicate", ""), t.get("object", "")
        if not s or not p or not o:
            continue
        subj_map.setdefault(s, {}).setdefault(p, []).append(o)

    # 传递性推理: uses, causes, replaces
    transitive_preds = {"uses", "causes", "replaces"}
    for subj, pred_map in subj_map.items():
        for pred in transitive_preds:
            if pred in pred_map:
                for obj in pred_map[pred]:
                    # obj 的关系传递到 subj
                    if obj in subj_map and pred in subj_map[obj]:
                        for trans_obj in subj_map[obj][pred]:
                            if trans_obj != subj:  # 避免循环
                                inferred.append((subj, pred, trans_obj))

    # 互逆推理: belongs_to ↔ contains
    for subj, pred_map in subj_map.items():
        if "belongs_to" in pred_map:
            for obj in pred_map["belongs_to"]:
                inferred.append((obj, "contains", subj))

    return inferred


# ─── KnowledgeGraph ────────────────────────────────────────────


class KnowledgeGraph:
    """SQLite 知识图谱，支持实体抽取、关系推理和图谱检索。"""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "knowledge_graph.db"
        self._conn: sqlite3.Connection | None = None
        self._triple_count = 0
        self._lock = threading.RLock()
        # ★ TTL 查询缓存：减少重复实体查询的 SQLite IO
        self._CACHE_TTL = 30.0
        self._query_cache: dict[str, tuple[Any, float]] = {}
        self._init_db()

    def _init_db(self) -> None:
        """初始化知识图谱数据库。"""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")

        # 三元组表（含时序有效性）
        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="triples",
            create_sql="""
                CREATE TABLE IF NOT EXISTS triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    source_memory_id TEXT,
                    confidence REAL DEFAULT 1.0,
                    is_negation INTEGER DEFAULT 0,
                    valid_from TEXT,
                    valid_to TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """,
            migrations=[],
        )
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_object ON triples(object)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predicate ON triples(predicate)
        """)

        # 实体表
        migrator.migrate(
            table_name="entities",
            create_sql="""
                CREATE TABLE IF NOT EXISTS entities (
                    name TEXT PRIMARY KEY,
                    entity_type TEXT DEFAULT 'unknown',
                    mention_count INTEGER DEFAULT 1,
                    first_seen TEXT,
                    last_seen TEXT
                )
            """,
            migrations=[],
        )

        self._conn.commit()

    # ─── 三元组操作 ────────────────────────────────────────────

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source_memory_id: str = "",
        confidence: float = 1.0,
        is_negation: bool = False,
        valid_from: str = "",
        valid_to: str = "",
    ) -> int:
        """添加三元组。"""
        with self._lock:
            assert self._conn is not None
            try:
                # 冲突检测：如果已有否定关系，不再添加肯定关系
                if not is_negation:
                    existing = self._conn.execute(
                        "SELECT id FROM triples WHERE subject = ? AND predicate = ? AND object = ? AND is_negation = 1",
                        (subject, predicate, obj),
                    ).fetchone()
                    if existing:
                        logger.warning(
                            "Triple blocked by negation: %s %s %s", subject, predicate, obj
                        )
                        return -1

                now = datetime.now(timezone.utc).isoformat()
                cursor = self._conn.execute(
                    """INSERT INTO triples (subject, predicate, object, source_memory_id,
                       confidence, is_negation, valid_from, valid_to, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        subject,
                        predicate,
                        obj,
                        source_memory_id,
                        confidence,
                        1 if is_negation else 0,
                        valid_from,
                        valid_to,
                        now,
                    ),
                )
                self._conn.commit()
                self._triple_count += 1

                # 同步更新实体表
                self._upsert_entity_locked(subject)
                self._upsert_entity_locked(obj)

                # 数据变更后清除查询缓存
                self._invalidate_cache()

                return cursor.lastrowid if cursor.lastrowid is not None else -1
            except Exception as e:
                logger.warning("Triple add failed: %s", e)
                return -1

    def add_triple_with_negation_check(
        self,
        subject: str,
        predicate: str,
        obj: str,
        content: str,
        source_memory_id: str = "",
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """添加三元组并自动检测否定关系。

        Returns:
            操作结果，包含是否有冲突
        """
        with self._lock:
            assert self._conn is not None
            # 检查内容是否包含否定
            is_negation = any(
                neg_word in content
                for neg_word in [
                    "不",
                    "并非",
                    "没有",
                    "无法",
                    "不能",
                    "不是",
                    "don't",
                    "not",
                    "no longer",
                ]
            )

            # 如果是新三元组，检查与已有三元组的否定冲突
            conflict = None
            if not is_negation:
                # 检查是否已有否定关系
                existing_neg = self._conn.execute(
                    "SELECT id, source_memory_id FROM triples WHERE subject = ? AND predicate = ? AND object = ? AND is_negation = 1",
                    (subject, predicate, obj),
                ).fetchone()
                if existing_neg:
                    conflict = {"type": "negation_exists", "triple_id": existing_neg[0]}
            else:
                # 否定关系：标记已有肯定关系为失效
                self._conn.execute(
                    "UPDATE triples SET valid_to = ? WHERE subject = ? AND predicate = ? AND object = ? AND is_negation = 0 AND valid_to = ''",
                    (datetime.now(timezone.utc).isoformat(), subject, predicate, obj),
                )
                self._conn.commit()
                self._invalidate_cache()

        triple_id = self.add_triple(
            subject,
            predicate,
            obj,
            source_memory_id=source_memory_id,
            confidence=confidence,
            is_negation=is_negation,
        )

        return {
            "triple_id": triple_id,
            "is_negation": is_negation,
            "conflict": conflict,
        }

    def _cached(self, key: str, fetch_fn: Callable[[], Any]) -> Any:
        """带 TTL 的查询缓存（CPython 下单个 dict 操作原子性足够）。"""
        now = time.monotonic()
        cached = self._query_cache.get(key)
        if cached:
            result, ts = cached
            if now - ts < self._CACHE_TTL:
                return result
            del self._query_cache[key]
        result = fetch_fn()
        self._query_cache[key] = (result, now)
        return result

    def _invalidate_cache(self) -> None:
        """数据变更后清除查询缓存。"""
        self._query_cache.clear()

    def query_by_subject(self, subject: str, include_expired: bool = False) -> list[dict[str, Any]]:
        """按主语查询三元组。"""

        def _fetch() -> list[dict[str, Any]]:
            assert self._conn is not None
            try:
                if include_expired:
                    rows = self._conn.execute(
                        "SELECT * FROM triples WHERE subject = ?",
                        (subject,),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT * FROM triples WHERE subject = ? AND (valid_to = '' OR valid_to IS NULL)",
                        (subject,),
                    ).fetchall()
                return self._rows_to_dicts(rows)
            except Exception as e:
                logger.warning("query_by_subject failed for %s: %s", subject, e)
                return []

        return self._cached(f"subj:{subject}:{include_expired}", _fetch)  # type: ignore[no-any-return]

    def query_by_object(self, obj: str, include_expired: bool = False) -> list[dict[str, Any]]:
        """按宾语查询三元组。"""

        def _fetch() -> list[dict[str, Any]]:
            assert self._conn is not None
            try:
                if include_expired:
                    rows = self._conn.execute(
                        "SELECT * FROM triples WHERE object = ?",
                        (obj,),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT * FROM triples WHERE object = ? AND (valid_to = '' OR valid_to IS NULL)",
                        (obj,),
                    ).fetchall()
                return self._rows_to_dicts(rows)
            except Exception as e:
                logger.warning("query_by_object failed for %s: %s", obj, e)
                return []

        return self._cached(f"obj:{obj}:{include_expired}", _fetch)  # type: ignore[no-any-return]

    def query_by_predicate(self, predicate: str, limit: int = 50) -> list[dict[str, Any]]:
        """按谓词查询三元组。"""
        try:
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT * FROM triples WHERE predicate = ? AND (valid_to = '' OR valid_to IS NULL) LIMIT ?",
                (predicate, limit),
            ).fetchall()
            return self._rows_to_dicts(rows)
        except Exception as e:
            logger.warning("query_by_predicate failed for %s: %s", predicate, e)
            return []

    def get_neighbors(self, entity: str, depth: int = 1) -> list[dict[str, Any]]:
        """获取实体的邻居（递归扩展查询），带 TTL 缓存。"""

        def _fetch() -> list[dict[str, Any]]:
            results = []
            visited: set[str] = set()

            def _expand(e: str, d: int) -> None:
                if d <= 0 or e in visited:
                    return
                visited.add(e)
                as_subj = self.query_by_subject(e)
                results.extend(as_subj)
                as_obj = self.query_by_object(e)
                results.extend(as_obj)
                if d > 1:
                    for t in as_subj:
                        _expand(t.get("object", ""), d - 1)
                    for t in as_obj:
                        _expand(t.get("subject", ""), d - 1)

            _expand(entity, depth)
            seen_ids: set[int] = set()
            unique_results = []
            for r in results:
                rid = r.get("id")
                if rid is not None:
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        unique_results.append(r)
            return unique_results

        return self._cached(f"neighbors:{entity}:{depth}", _fetch)  # type: ignore[no-any-return]

    # ─── 从记忆中自动抽取 ─────────────────────────────────────

    def extract_and_store(
        self, content: str, memory_id: str = "", confidence: float = 0.8
    ) -> dict[str, Any]:
        """从记忆内容中抽取实体和三元组并存储。

        Returns:
            抽取统计
        """
        # 提取实体
        entities = extract_entities(content)
        for entity in entities:
            self._upsert_entity(entity)

        # 提取三元组
        raw_triples = extract_triples(content)
        stored_triples = []
        conflicts = []

        for subj, pred, obj in raw_triples:
            result = self.add_triple_with_negation_check(
                subj,
                pred,
                obj,
                content=content,
                source_memory_id=memory_id,
                confidence=confidence,
            )
            if result["triple_id"] > 0:
                stored_triples.append(result)
            if result["conflict"]:
                conflicts.append(result["conflict"])

        # ★ P1方案四：增量局部推理（替代全表扫描）
        # 对新三元组的主语和宾语做 2-hop 邻居查询，仅对局部子图推理
        inferred_stored = []
        seen_inferred: set[tuple[str, str, str]] = set()
        for subj, pred, obj in raw_triples:
            local_triples: list[dict[str, Any]] = []
            try:
                local_triples.extend(self.query_by_subject(subj))
                local_triples.extend(self.query_by_object(subj))
                local_triples.extend(self.query_by_subject(obj))
                local_triples.extend(self.query_by_object(obj))
            except Exception as e:
                logger.warning("extract_and_store local query failed: %s", e)
                continue

            inferred = infer_relations(local_triples)
            assert self._conn is not None
            for isubj, ipred, iobj in inferred:
                key = (isubj, ipred, iobj)
                if key in seen_inferred:
                    continue
                seen_inferred.add(key)
                existing = self._conn.execute(
                    "SELECT id FROM triples WHERE subject = ? AND predicate = ? AND object = ?",
                    (isubj, ipred, iobj),
                ).fetchone()
                if not existing:
                    tid = self.add_triple(
                        isubj,
                        ipred,
                        iobj,
                        source_memory_id=f"inferred-from:{memory_id}",
                        confidence=0.5,
                    )
                    if tid > 0:
                        inferred_stored.append(
                            {"subject": isubj, "predicate": ipred, "object": iobj}
                        )

        return {
            "entities_extracted": len(entities),
            "triples_extracted": len(raw_triples),
            "triples_stored": len(stored_triples),
            "conflicts_found": len(conflicts),
            "inferred_triples": len(inferred_stored),
        }

    # ─── 图谱检索通道 ─────────────────────────────────────────

    def graph_search(self, query: str, max_depth: int = 2, limit: int = 20) -> list[dict[str, Any]]:
        """图谱检索通道：从查询中提取实体，然后扩展搜索。

        用于检索引擎的第6通道 (Graph Retriever)。
        """
        # 从查询中提取可能的实体
        query_entities = extract_entities(query)

        if not query_entities:
            # 尝试直接关键词匹配（转义 LIKE 通配符防止注入/误匹配）
            try:
                assert self._conn is not None
                escaped = query.replace("%", "\\%").replace("_", "\\_")
                rows = self._conn.execute(
                    "SELECT * FROM triples WHERE subject LIKE ? ESCAPE '\\' OR object LIKE ? ESCAPE '\\' LIMIT ?",
                    (f"%{escaped}%", f"%{escaped}%", limit),
                ).fetchall()
                return self._rows_to_dicts(rows)
            except Exception as e:
                logger.warning("graph_search keyword query failed: %s", e)
                return []

        # 对每个实体进行扩展搜索
        all_results: list[dict[str, Any]] = []
        for entity in query_entities[:3]:  # 最多3个实体
            neighbors = self.get_neighbors(entity, depth=max_depth)
            all_results.extend(neighbors)

        # 去重
        seen_ids: set[int] = set()
        unique = []
        for r in all_results:
            rid = r.get("id")
            if rid not in seen_ids:
                seen_ids.add(rid)  # type: ignore[arg-type]
                unique.append(r)

        return unique[:limit]

    def graph_rag_context(self, entity: str, depth: int = 1) -> str:
        """Graph RAG: 生成实体子图的可读上下文文本。

        将子图中的三元组格式化为自然语言描述，可直接注入LLM上下文窗口。
        参考 Cognee/Zep 的 Graph RAG 模式。

        Args:
            entity: 起始实体名
            depth: 扩展深度（1-hop/2-hop）

        Returns:
            格式化的子图上下文文本，无结果返回空字符串
        """
        neighbors = self.get_neighbors(entity, depth=max(depth, 1))
        if not neighbors:
            # 尝试用部分匹配
            try:
                assert self._conn is not None
                escaped = entity.replace("%", "\\%").replace("_", "\\_")
                rows = self._conn.execute(
                    "SELECT * FROM triples WHERE subject LIKE ? ESCAPE '\\' LIMIT 5",
                    (f"%{escaped}%",),
                ).fetchall()
                neighbors = self._rows_to_dicts(rows)
            except Exception as e:
                logger.warning("graph_rag_context partial match failed: %s", e)

        if not neighbors:
            return ""

        # 按关系类型分组
        grouped: dict[str, list[tuple[str, str]]] = {}
        for t in neighbors:
            subj = t.get("subject", "")
            obj = t.get("object", "")
            pred = t.get("predicate", "")
            if subj and obj:
                grouped.setdefault(pred, []).append((subj, obj))
            elif subj:
                grouped.setdefault("related", []).append((subj, "?"))

        # 生成自然语言上下文
        lines = []
        relation_labels = {
            "uses": "使用", "belongs_to": "属于", "causes": "导致",
            "replaces": "替代", "connects_to": "关联到", "contains": "包含",
            "located_in": "位于", "better_than": "优于",
            "not_uses": "不使用", "differs_from": "不同于",
            "related": "相关于",
        }
        seen_pairs: set[tuple[str, str, str]] = set()
        for pred, pairs in grouped.items():
            label = relation_labels.get(pred, pred)
            for subj, obj in pairs[:3]:  # Max 3 per relation type
                key = (subj, pred, obj)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if obj == "?":
                    lines.append(f"- {subj} {label}")
                else:
                    lines.append(f"- {subj} {label} {obj}")

        if not lines:
            return ""
        return f"[Knowledge Graph: {entity}]\n" + "\n".join(lines)

    def graph_rag_search(self, query: str, max_depth: int = 2, limit: int = 10) -> str:
        """完整的 Graph RAG 搜索：提取实体→扩展子图→生成上下文文本。

        Args:
            query: 自然语言查询
            max_depth: 扩展深度
            limit: 最多返回的三元组数

        Returns:
            Graph RAG 上下文字符串，可注入LLM
        """
        query_entities = extract_entities(query)
        if not query_entities:
            return ""

        contexts = []
        for entity in query_entities[:3]:
            ctx = self.graph_rag_context(entity, depth=max_depth)
            if ctx:
                contexts.append(ctx)

        return "\n\n".join(contexts)


    # ─── 实体操作 ─────────────────────────────────────────────

    def get_entity(self, name: str) -> dict[str, Any] | None:
        """获取实体信息。"""
        try:
            assert self._conn is not None
            row = self._conn.execute("SELECT * FROM entities WHERE name = ?", (name,)).fetchone()
            if row:
                keys = ["name", "entity_type", "mention_count", "first_seen", "last_seen"]
                return dict(zip(keys, row, strict=False))
            return None
        except Exception as e:
            logger.warning("get_entity failed for %s: %s", name, e)
            return None

    def get_all_entities(self, limit: int = 100) -> list[dict[str, Any]]:
        """获取所有实体。"""
        try:
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT * FROM entities ORDER BY mention_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
            keys = ["name", "entity_type", "mention_count", "first_seen", "last_seen"]
            return [dict(zip(keys, row, strict=False)) for row in rows]
        except Exception as e:
            logger.warning("get_all_entities failed: %s", e)
            return []

    def get_entity_graph(self, limit: int = 100) -> dict[str, list[dict[str, Any]]]:
        """获取按 POLE+O 类型分组的实体图谱摘要。

        Returns:
            {"Person": [...], "Organization": [...], "Location": [...],
             "Event": [...], "Object": [...]}
        """
        entities = self.get_all_entities(limit)
        graph: dict[str, list[dict[str, Any]]] = {
            "Person": [],
            "Organization": [],
            "Location": [],
            "Event": [],
            "Object": [],
        }
        for e in entities:
            etype = e.get("entity_type", "Object")
            if etype in graph:
                graph[etype].append(e)
            else:
                graph["Object"].append(e)
        return graph

    # ─── 时序图谱 API ──────────────────────────────────────────

    def get_timeline(self, entity: str, limit: int = 50) -> list[dict[str, Any]]:
        """获取实体的时间线：与其相关的所有三元组按创建时间排序。

        对标 Zep/kektordb 的时序图谱能力 — 追踪实体关系的演变。

        Returns:
            按 created_at 升序的三元组列表，形成实体关系演变时间线
        """
        if not self._conn:
            return []
        try:
            rows = self._conn.execute(
                "SELECT * FROM triples "
                "WHERE (subject = ? OR object = ?) AND (valid_to = '' OR valid_to IS NULL) "
                "ORDER BY created_at ASC LIMIT ?",
                (entity, entity, limit),
            ).fetchall()
            return self._rows_to_dicts(rows)
        except Exception as e:
            logger.warning("Timeline query failed: %s", e)
            return []

    def get_entity_timeline_text(self, entity: str, limit: int = 20) -> str:
        """生成实体时间线的可读文本，适合注入 LLM 上下文。

        Returns:
            格式化的时间线文本，无结果返回空字符串
        """
        timeline = self.get_timeline(entity, limit=limit)
        if not timeline:
            return ""

        relation_labels = {
            "uses": "开始使用", "belongs_to": "归属", "causes": "引起",
            "replaces": "取代", "connects_to": "关联到", "contains": "包含",
            "located_in": "位于", "better_than": "优于",
        }
        lines = [f"[{entity} 时间线]"]
        for t in timeline:
            created = t.get("created_at", "")[:10]  # 只取日期
            subj = t.get("subject", "")
            obj = t.get("object", "")
            pred = t.get("predicate", "")
            label = relation_labels.get(pred, pred)
            lines.append(f"  {created}: {subj} {label} {obj}")
        return "\n".join(lines)

    def get_recent_changes(self, since_days: int = 7, limit: int = 50) -> list[dict[str, Any]]:
        """获取最近N天的三元组变更。

        Args:
            since_days: 最近多少天
            limit: 最大返回数

        Returns:
            按创建时间降序的三元组列表
        """
        if not self._conn:
            return []
        try:
            from datetime import datetime, timedelta, timezone
            since = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
            rows = self._conn.execute(
                "SELECT * FROM triples WHERE created_at >= ? "
                "AND (valid_to = '' OR valid_to IS NULL) "
                "ORDER BY created_at DESC LIMIT ?",
                (since, limit),
            ).fetchall()
            return self._rows_to_dicts(rows)
        except Exception as e:
            logger.warning("Recent changes query failed: %s", e)
            return []

    # ─── 统计 ─────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """获取图谱统计。"""
        stats: dict[str, Any] = {"total_triples": 0, "total_entities": 0}
        if self._conn:
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM triples").fetchone()
                stats["total_triples"] = row[0] if row else 0

                row = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()
                stats["total_entities"] = row[0] if row else 0

                # 按谓词统计
                pred_rows = self._conn.execute(
                    "SELECT predicate, COUNT(*) as cnt FROM triples GROUP BY predicate ORDER BY cnt DESC LIMIT 10"
                ).fetchall()
                stats["predicates"] = {r[0]: r[1] for r in pred_rows}
            except Exception as e:
                logger.warning("get_stats query failed: %s", e)
        return stats

    # ─── 图算法 ─────────────────────────────────────────────

    def shortest_path(
        self,
        start: str,
        end: str,
        max_depth: int = 5,
    ) -> list[dict[str, Any]]:
        """返回两实体间的最短关系路径（BFS）。

        Args:
            start: 起始实体
            end: 目标实体
            max_depth: 最大搜索深度

        Returns:
            路径上的三元组列表（从 start 到 end）
        """
        if not self._conn:
            return []
        try:
            from collections import deque

            visited: dict[str, tuple[Any, ...]] = {start: ()}  # entity -> (prev_entity, triple_dict)
            queue: deque[str] = deque([start])
            depth = 0

            while queue and depth < max_depth:
                for _ in range(len(queue)):
                    current = queue.popleft()
                    if current == end:
                        # 回溯路径
                        path = []
                        node = end
                        while node != start:
                            prev, triple = visited[node]
                            path.append(triple)
                            node = prev
                        return list(reversed(path))

                    # 扩展邻居：作为 subject 或 object
                    rows = self._conn.execute(
                        "SELECT subject, predicate, object, confidence FROM triples "
                        "WHERE (subject = ? OR object = ?) AND (valid_to = '' OR valid_to IS NULL)",
                        (current, current),
                    ).fetchall()
                    for subj, pred, obj, conf in rows:
                        neighbor = obj if subj == current else subj
                        if neighbor not in visited:
                            visited[neighbor] = (
                                current,
                                {
                                    "subject": subj,
                                    "predicate": pred,
                                    "object": obj,
                                    "confidence": conf,
                                },
                            )
                            queue.append(neighbor)
                depth += 1
            return []
        except Exception as e:
            logger.warning("Shortest path failed: %s", e)
            return []

    def connected_components(self, min_size: int = 3, limit: int = 500) -> list[list[str]]:
        """发现知识社区（连通分量）。

        Args:
            min_size: 社区最小实体数
            limit: 最大扫描实体数

        Returns:
            每个社区是一个实体名称列表
        """
        if not self._conn:
            return []
        try:
            rows = self._conn.execute(
                "SELECT DISTINCT subject, object FROM triples "
                "WHERE valid_to = '' OR valid_to IS NULL LIMIT ?",
                (limit * 2,),
            ).fetchall()
            from collections import defaultdict

            graph: defaultdict[str, set[str]] = defaultdict(set)
            all_entities: set[str] = set()
            for subj, obj in rows:
                graph[subj].add(obj)
                graph[obj].add(subj)
                all_entities.add(subj)
                all_entities.add(obj)

            visited: set[str] = set()
            components: list[list[str]] = []
            for entity in all_entities:
                if entity in visited:
                    continue
                stack = [entity]
                comp: list[str] = []
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    comp.append(node)
                    stack.extend(graph[node] - visited)
                if len(comp) >= min_size:
                    components.append(comp)
            return components
        except Exception as e:
            logger.warning("Connected components failed: %s", e)
            return []

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ─── 内部方法 ─────────────────────────────────────────────

    def _upsert_entity(self, name: str) -> None:
        """更新或插入实体（外部调用，加锁）。"""
        with self._lock:
            self._upsert_entity_locked(name)

    def _upsert_entity_locked(self, name: str) -> None:
        """更新或插入实体（内部已持有锁时调用）。"""
        try:
            assert self._conn is not None
            now = datetime.now(timezone.utc).isoformat()
            existing = self._conn.execute(
                "SELECT name FROM entities WHERE name = ?", (name,)
            ).fetchone()
            if existing:
                self._conn.execute(
                    "UPDATE entities SET mention_count = mention_count + 1, last_seen = ? WHERE name = ?",
                    (now, name),
                )
            else:
                # 推断实体类型
                entity_type = self._infer_entity_type(name)
                self._conn.execute(
                    "INSERT INTO entities (name, entity_type, mention_count, first_seen, last_seen) VALUES (?, ?, 1, ?, ?)",
                    (name, entity_type, now, now),
                )
            self._conn.commit()
        except Exception as e:
            logger.warning("Entity upsert failed: %s", e)

    def _infer_entity_type(self, name: str) -> str:
        """推断实体类型 → POLE+O 五类 (Person/Organization/Location/Event/Object)。

        统一使用 _classify_entity_poleo 规则引擎，与 extract_entities_llm 保持一致。
        """
        poleo = _classify_entity_poleo(name)
        poleo_label = _POLEO_TYPES.get(poleo, "Object")
        return poleo_label

    def _get_all_triples(self, limit: int = 5000) -> list[dict[str, Any]]:
        """获取所有有效三元组。"""
        try:
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT * FROM triples WHERE (valid_to = '' OR valid_to IS NULL) LIMIT ?",
                (limit,),
            ).fetchall()
            return self._rows_to_dicts(rows)
        except Exception as e:
            logger.warning("_get_all_triples failed: %s", e)
            return []

    def _rows_to_dicts(self, rows: list[Any]) -> list[dict[str, Any]]:
        """将行转为字典。"""
        keys = [
            "id",
            "subject",
            "predicate",
            "object",
            "source_memory_id",
            "confidence",
            "is_negation",
            "valid_from",
            "valid_to",
            "created_at",
        ]
        return [dict(zip(keys, row, strict=False)) for row in rows]


# ─── POLE+O 实体分类 (参考 neo4j-labs/agent-memory) ────────

_POLEO_TYPES: dict[str, str] = {
    "person": "Person",
    "org": "Organization",
    "location": "Location",
    "event": "Event",
    "object": "Object",
}


def _classify_entity_poleo(name: str) -> str:
    """快速规则分类实体到 POLE+O 类型。LLM 提取后可覆盖此分类。

    优先级: Location/Organization已知 > Person > 后缀匹配Organization > Location > Event > Object
    """
    # ── 已知地名（优先级最高，避免姓氏误判）──
    known_locations = {"北京", "上海", "深圳", "杭州", "广州", "成都", "浙江", "金华",
                       "中国", "美国", "日本", "旧金山", "硅谷"}
    if name in known_locations:
        return "location"

    # ── 已知组织名（优先级高于人名模式）──
    known_orgs = {"OpenAI", "Google", "Meta", "Microsoft", "Apple", "Amazon",
                  "Anthropic", "Nous Research", "Hermes", "GitHub"}
    if name in known_orgs:
        return "org"

    # ── Person: 中文人名 + 角色称谓 ──
    # 常见中文姓氏（含复姓）前缀 + 2-3字
    _cn_surnames = (
        "王李张刘陈杨黄赵周吴徐孙马胡朱郭何罗高林郑梁谢唐许冯宋韩邓彭曹曾田萧"
        "潘袁蔡蒋余于杜叶程苏魏吕丁任卢姚钟姜崔谭陆汪范金石廖贾夏韦付方白邹孟"
        "熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤温康施文"
        "牛樊葛邢安齐易乔伍庞颜倪庄聂章鲁岳翟殷詹申欧耿关兰焦俞左柳甘祝包宁尚"
        "司马欧阳上官皇甫诸葛令狐司徒"
    )
    if len(name) >= 2 and len(name) <= 4 and re.match(r"^[\u4e00-\u9fff]+$", name):
        # 排除明确非人名的概念词
        concept_words = {
            "问题", "方法", "方案", "结果", "数据", "功能", "配置", "部署",
            "测试", "需求", "设计", "架构", "实现", "开发", "优化", "模块",
            "系统", "服务", "接口", "组件", "引擎", "管道", "通道",
        }
        # 姓氏匹配 或 结尾为常见人名后缀
        if name[0] in _cn_surnames and name not in concept_words:
            return "person"
        # 称谓后缀: 总/工/老师/经理/教授/博士/老板/先生/女士
        if re.search(r"(总|工|老师|经理|教授|博士|老板|先生|女士|同学)$", name):
            return "person"

    # 英文人名: First Last (CamelCase with space concept)
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", name):
        return "person"

    # ── Organization ──
    org_suffixes = (
        "公司|团队|部门|组|实验室|小组|工作室|组织|中心|研究院|研究所|"
        "学院|大学|集团|银行|基金|协会|联盟|社区|平台"
    )
    if re.search(f"({org_suffixes})$", name):
        return "org"

    # ── Location ──
    location_suffixes = "市|省|区|县|镇|乡|村|街道|路|大厦|广场|园区|国家|大陆|岛|湾|湖|山|海"
    if re.search(f"({location_suffixes})$", name):
        return "location"

    # ── Event ──
    event_keywords = (
        "会议|峰会|大会|发布会|上线|部署|测试|回归|修复|审查|合并|"
        "发布|版本|里程碑|迭代|冲刺|评审|复盘|事故|故障|演练"
    )
    if re.search(f"({event_keywords})", name):
        return "event"

    # ── Object (默认) ──
    return "object"


def extract_entities_llm(
    texts: list[str],
    llm_call: Callable[[str], str] | None = None,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """使用 LLM 深度提取实体和三元组（POLE+O 分类版）。

    设计: 批量处理，后台异步触发，不影响主路径的正则快速提取。
    无 LLM 客户端时回退到规则分类。

    Args:
        texts: 待提取的记忆文本列表
        llm_call: LLM 调用函数（接收prompt返回text），可选

    Returns:
        (entities, triples) — entities 带 POLE+O 分类前缀
    """
    if not texts or not llm_call:
        all_entities: set[str] = set()
        all_triples: list[tuple[str, str, str]] = []
        for text in texts:
            all_entities.update(extract_entities(text))
            all_triples.extend(extract_triples(text))
        classified = [f"{_classify_entity_poleo(e)}:{e}" for e in all_entities]
        return classified, all_triples

    prompt = (
        "Extract entities and relationships from the following texts. "
        "Classify each entity into: Person, Organization, Location, Event, Object. "
        "Return JSON: "
        '{"entities": [{"name": "...", "type": "..."}], '
        '"triples": [["subject", "predicate", "object"]]}'
        "\n\nTexts:\n"
    )
    combined = "\n---\n".join(texts[:5])
    prompt += combined[:3000]

    try:
        response = llm_call(prompt)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in response")
        data = json.loads(json_match.group())
        entities = data.get("entities", [])
        triples_data = data.get("triples", [])

        classified_entities = [
            f"{e.get('type', 'Object')}:{e.get('name', '')}" for e in entities
        ]
        llm_triples = [
            (t[0], t[1], t[2]) for t in triples_data if len(t) >= 3
        ]
        return classified_entities, llm_triples
    except Exception as e:
        logger.warning("LLM entity extraction failed, fallback regex: %s", e)
        all_entities = set()
        all_triples = []
        for text in texts:
            all_entities.update(extract_entities(text))
            all_triples.extend(extract_triples(text))
        return [f"Object:{e}" for e in all_entities], all_triples
