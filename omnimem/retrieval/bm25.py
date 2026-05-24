"""BM25Retriever — BM25 关键词检索。

使用 rank_bm25 库实现 BM25 算法，用于精确关键词匹配。
改进：add() 使用缓冲区 + 延迟重建，避免 O(n²) 性能问题。
OPT-6: 支持磁盘缓存，跨会话后快速恢复索引而无需全量重建。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

logger = logging.getLogger(__name__)


# ★ 高频噪声词集合（IDF值极低，会稀释有效词的区分度）
# 这些词在BM25中几乎无区分能力，需降权处理
_NOISE_WORDS = {
    "问题",
    "方法",
    "使用",
    "进行",
    "实现",
    "相关",
    "包括",
    "关于",
    "通过",
    "根据",
    "由于",
    "因此",
    "但是",
    "然而",
    "另外",
    "此外",
    "同时",
    "首先",
    "其次",
    "最后",
}


_MINIMAL_ZH_STOPWORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
}


_COMMON_ZH_WORDS: set[str] = set()


def _load_common_zh_words() -> set[str]:
    """从外部 JSON 加载中文词词典。

    加载策略：
      1. 尝试从 config/zh_words.json 加载
      2. 加载成功时使用外部词典
      3. 加载失败时回退到 _MINIMAL_ZH_STOPWORDS 并记录 warning
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "zh_words.json"
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            external: list[str] = json.load(f)
        if isinstance(external, list):
            return set(external)
    except FileNotFoundError:
        logger.warning("zh_words.json not found at %s, using minimal stopwords", config_path)
    except Exception:
        logger.warning("Failed to load zh_words.json from %s, using minimal stopwords", config_path)
    return set(_MINIMAL_ZH_STOPWORDS)


def _ensure_common_zh_words() -> None:
    """确保 _COMMON_ZH_WORDS 已加载（惰性初始化）。"""
    global _COMMON_ZH_WORDS
    if not _COMMON_ZH_WORDS:
        _COMMON_ZH_WORDS = _load_common_zh_words()


def _tokenize(text: str) -> list[str]:
    if _HAS_JIEBA:
        raw_tokens = jieba.lcut(text)
        processed = []
        for t in raw_tokens:
            if not re.search(r'[\u4e00-\u9fffa-zA-Z0-9]', t):
                continue
            processed.append(t)
        if not processed and re.search(r'[\u4e00-\u9fff]', text):
            zh_chars = re.findall(r'[\u4e00-\u9fff]', text)
            if zh_chars:
                processed = [zh_chars[0]]
        return processed

    _ensure_common_zh_words()
    raw_tokens = []
    raw_tokens.extend(re.findall(r"[a-zA-Z0-9]{2,}", text.lower()))

    zh_chars = re.findall(r"[\u4e00-\u9fff]+", text)
    for segment in zh_chars:
        i = 0
        while i < len(segment):
            matched = False
            for word_len in (4, 3, 2):
                if i + word_len > len(segment):
                    continue
                word = segment[i : i + word_len]
                if word in _COMMON_ZH_WORDS:
                    raw_tokens.append(word)
                    i += word_len
                    matched = True
                    break
            if not matched:
                raw_tokens.append(segment[i])
                i += 1

    filtered = [t for t in raw_tokens if len(t) >= 2 or not re.match(r"[\u4e00-\u9fff]", t)]

    if not filtered and raw_tokens:
        _stop_chars = {"的", "了", "是", "在", "和", "就", "也", "很", "到", "说", "要", "去", "不"}
        filtered = [t for t in raw_tokens if t not in _stop_chars]

    if not filtered and re.search(r'[\u4e00-\u9fff]', text):
        zh_chars = re.findall(r'[\u4e00-\u9fff]', text)
        if zh_chars:
            filtered = [zh_chars[0]]

    return filtered


class BM25Retriever:
    """BM25 关键词检索，带批量缓冲优化。

    改进策略：
      - add() 写入缓冲区，不立即重建索引
      - search() 标记脏数据，返回上次索引的结果（可接受轻微不一致）
      - 后台线程延迟重建，不阻塞搜索路径
      - flush() 显式刷新（会话结束时调用）
      - add_document() 增量添加单个文档，避免从 SQLite 全量读取
      - rebuild_from_entries() 全量重建公开接口
    """

    CACHE_VERSION = 3

    def __init__(self, buffer_size: int = 50, data_dir: Path | None = None, max_documents: int = 5000):
        self._corpus: list[list[str]] = []
        self._documents: list[dict[str, Any]] = []
        self._bm25: Any = None
        self._buffer: list[dict[str, Any]] = []
        self._buffer_size = buffer_size
        self._max_documents = max_documents
        self._data_dir = data_dir
        self._lock = threading.Lock()
        self._dirty = False
        self._rebuilding = False
        self._cache_loaded = False
        self._load_from_disk()

    def add(self, content: str, memory_id: str, metadata: dict[str, Any]) -> None:
        """添加文档到 BM25 缓冲区。达到阈值时自动刷新。"""
        entry = dict(metadata)
        entry["content"] = content
        entry["memory_id"] = memory_id
        entry["_tokens"] = _tokenize(content)

        with self._lock:
            self._buffer.append(entry)
            self._dirty = True
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()
            elif not self._rebuilding:
                # ★ R26优化：缓冲区未满但已脏，触发后台延迟重建
                # 避免小批量写入时 search() 前才刷新导致的延迟
                self._start_background_rebuild()

    def add_batch(self, documents: list[dict[str, Any]]) -> None:
        """批量添加文档。"""
        with self._lock:
            for doc in documents:
                content = doc.get("content", "")
                entry = dict(doc)
                entry["_tokens"] = _tokenize(content)
                self._buffer.append(entry)
            self._dirty = True
            self._flush_buffer()

    def warmup(self) -> None:
        """预热：刷新缓冲区确保 BM25 索引就绪。"""
        with self._lock:
            if self._buffer:
                self._flush_buffer()

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """BM25 关键词检索。搜索前同步刷新缓冲区，确保新写入条目可被检索。"""
        with self._lock:
            if self._buffer:
                self._flush_buffer()

        if not self._bm25 or not self._corpus:
            return []
        try:
            query_tokens = _tokenize(query)
            if not query_tokens:
                return []
            scores = self._bm25.get_scores(query_tokens)
            # ★ Step 1: 先基于原始 BM25 分数做阈值过滤（不受加权影响）
            if scores is not None and len(scores) > 0:
                raw_max = max(scores)
                # 原始分数阈值：绝对最低 0.01 + 相对最低 max * 0.05
                # ★ 阈值不宜太高：BM25 原始分数范围取决于语料规模，
                # 高阈值会过滤掉包含部分查询词但 IDF 较低的合理结果
                raw_min = max(0.01, raw_max * 0.05)
                # 过滤掉原始分数过低的文档
                ranked = [(idx, score) for idx, score in enumerate(scores) if score > raw_min]
                ranked.sort(key=lambda x: x[1], reverse=True)
            else:
                ranked = []

            # ★ Step 2: 对过滤后的结果加权排序（含噪声词降权）
            results = []

            # ★ R25优化：语料过少时 BM25 IDF 不可靠（N≤2时IDF为负/零）
            # 回退到简单的关键词匹配，避免漏召回
            if not ranked and len(self._corpus) <= 5 and len(self._corpus) > 0:
                query_set = set(query_tokens)
                for idx, doc_tokens in enumerate(self._corpus):
                    if idx >= len(self._documents):
                        continue
                    overlap = query_set & set(doc_tokens)
                    if overlap:
                        entry = dict(self._documents[idx])
                        entry.pop("_tokens", None)
                        entry["score"] = len(overlap) / len(query_set) * 0.5
                        results.append(entry)
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:top_k]

            # ★ 识别查询词中的噪声词和有效词
            query_set = set(query_tokens)
            noise_query = query_set & _NOISE_WORDS
            valid_query = query_set - _NOISE_WORDS

            for idx, raw_score in ranked[:top_k]:
                if idx < len(self._documents):
                    entry = dict(self._documents[idx])
                    entry.pop("_tokens", None)
                    score = raw_score

                    doc_tokens = self._corpus[idx] if idx < len(self._corpus) else []
                    doc_set = set(doc_tokens)

                    # ★ 有效词命中加权（高区分度词）
                    valid_overlap = valid_query & doc_set
                    if valid_overlap:
                        valid_ratio = len(valid_overlap) / len(valid_query) if valid_query else 0
                        if valid_ratio >= 0.5:
                            score *= 2.5  # 有效词半数以上命中 → 强力加权
                        elif valid_ratio > 0:
                            score *= 1.0 + valid_ratio * 1.5

                    # ★ 噪声词命中降权：仅噪声词匹配不应显著提升排名
                    noise_overlap = noise_query & doc_set
                    if noise_overlap and not valid_overlap:
                        # 只有噪声词命中、无有效词命中 → 惩罚性降权
                        score *= 0.3
                    elif noise_overlap and valid_overlap:
                        # 混合命中：噪声词贡献打折
                        noise_ratio = len(noise_overlap) / len(query_set)
                        score *= 1.0 - 0.3 * noise_ratio

                    entry["score"] = float(score)
                    results.append(entry)
            return results
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)
            return []

    def flush(self) -> None:
        """显式刷新缓冲区并保存磁盘缓存。"""
        with self._lock:
            if self._buffer:
                self._flush_buffer()
            self._save_to_disk()

    @property
    def pending_count(self) -> int:
        """缓冲区中待刷新的文档数。"""
        return len(self._buffer)

    @property
    def document_count(self) -> int:
        """已索引的文档总数（不含缓冲区）。"""
        return len(self._documents)

    def add_document(self, doc_id: str, text: str) -> None:
        tokens = _tokenize(text)
        with self._lock:
            self._corpus.append(tokens)
            self._documents.append({"memory_id": doc_id, "content": text})
            self._rebuild()
            self._dirty = True

    def delete(self, memory_id: str) -> None:
        """从 BM25 索引中删除指定条目。"""
        with self._lock:
            indices_to_remove = [
                i for i, doc in enumerate(self._documents)
                if doc.get("memory_id") == memory_id
            ]
            if indices_to_remove:
                for idx in reversed(indices_to_remove):
                    if idx < len(self._corpus):
                        self._corpus.pop(idx)
                    self._documents.pop(idx)
                self._rebuild()
                self._dirty = True

    def rebuild_from_entries(self, entries: list[dict[str, Any]]) -> int:
        with self._lock:
            self._corpus.clear()
            self._documents.clear()
            self._buffer.clear()
            self._bm25 = None
            self._dirty = False
        rebuilt = 0
        for entry in entries:
            content = entry.get("content", "") or entry.get("summary", "")
            memory_id = entry.get("memory_id", "")
            if content and memory_id:
                existing_ids = {doc.get("memory_id", ""): i for i, doc in enumerate(self._documents)}
                if memory_id in existing_ids:
                    old_content = self._documents[existing_ids[memory_id]].get("content", "")
                    if hashlib.md5(old_content.encode()).hexdigest() != hashlib.md5(content.encode()).hexdigest():
                        self.delete(memory_id)
                self.add_document(memory_id, content)
                rebuilt += 1
        with self._lock:
            self._dirty = False
        return rebuilt

    @property
    def cache_loaded(self) -> bool:
        return self._cache_loaded

    def _flush_buffer(self) -> None:
        for entry in self._buffer:
            tokens = entry.pop("_tokens", [])
            self._corpus.append(tokens)
            self._documents.append(entry)
        self._buffer.clear()
        # ★ P3 LRU淘汰：超出上限时删除最旧文档
        while len(self._documents) > self._max_documents:
            self._corpus.pop(0)
            self._documents.pop(0)
        self._rebuild()

    def _rebuild(self) -> None:
        """重建 BM25 索引。"""
        if not self._corpus:
            self._bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._corpus)
        except ImportError:
            logger.warning("rank_bm25 not installed — BM25 search unavailable")
            self._bm25 = None

    def _start_background_rebuild(self) -> None:
        """在后台线程中合并缓冲区并重建索引，不阻塞搜索。"""
        if self._rebuilding:
            return
        self._rebuilding = True

        def _do_rebuild() -> None:
            try:
                with self._lock:
                    if self._buffer:
                        self._flush_buffer()
                        self._save_to_disk()
                        self._dirty = False
            except Exception:
                logger.warning("BM25 background rebuild failed", exc_info=True)
            finally:
                self._rebuilding = False

        t = threading.Thread(target=_do_rebuild, daemon=True)
        t.start()

    # ─── Disk cache (OPT-6) ──────────────────────────────────

    def _disk_cache_path(self) -> Path | None:
        if self._data_dir is None:
            return None
        return self._data_dir / "bm25_cache.json"

    def _load_from_disk(self) -> None:
        cache_path = self._disk_cache_path()
        if cache_path is None or not cache_path.exists():
            return
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("version") != self.CACHE_VERSION:
                logger.warning("BM25 disk cache version mismatch (expected %d, got %d), rebuilding", self.CACHE_VERSION, cached.get("version", 0))
                try:
                    cache_path.unlink()
                except OSError:
                    pass
                return
            self._corpus = cached.get("corpus", [])
            self._documents = cached.get("documents", [])
            if self._corpus:
                self._rebuild()
                self._cache_loaded = True
                logger.warning("BM25 loaded %d entries from disk cache", len(self._documents))
        except Exception as e:
            logger.warning("BM25 disk cache load failed: %s", e)

    def _save_to_disk(self) -> None:
        cache_path = self._disk_cache_path()
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"version": self.CACHE_VERSION, "corpus": self._corpus, "documents": self._documents},
                    f,
                    ensure_ascii=False,
                )
            logger.warning("BM25 saved %d entries to disk cache", len(self._documents))
        except Exception as e:
            logger.warning("BM25 disk cache save failed: %s", e)
