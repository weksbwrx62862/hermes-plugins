"""OmniMem recall 处理器。

从 provider.py 的 _handle_recall() 方法提取，通过 provider 参数访问实例组件。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ★ 时序关键词：检测查询是否包含时间相关语义，优先使用时序图谱
_TEMPORAL_KEYWORDS = (
    "现在", "之前", "之后", "上个月", "下个月", "去年", "今年", "目前",
    "当前", "最近", "以前", "后来", "后来呢", "什么时候", "什么时候开始",
    "什么时候结束", "什么时候换", "什么时候改", "变化", "变了", "换了",
    "更新了", "升级了", "改成了", "变成了", "之前是", "原来是", "以前是",
    "last", "now", "before", "after", "previously", "currently",
    "used to", "changed", "updated", "replaced",
)

# ★ R26优化：提取公共正则常量，避免4处硬编码重复
_CJK_KEYWORD_RE = re.compile(
    r"[\u4e00-\u9fff]{2,}|[\uac00-\ud7af]{2,}|[\u3040-\u309f\u30a0-\u30ff]{2,}|[a-zA-Z]{3,}"
)

# ★ R27优化：模块级同义词映射，避免 llm 模式每次调用重建字典
_SYNONYM_MAP: dict[str, list[str]] = {
    "宠物": ["猫咪", "狗狗", "兔子", "仓鼠", "小鸟", "小鱼"],
    "饮食": ["食用", "喂食", "饲料", "鸡胸肉", "猫粮", "狗粮"],
    "编程": ["代码", "开发", "程序", "coding"],
    "部署": ["deploy", "上线", "发布", "运维"],
    "数据库": ["mysql", "postgres", "mongodb", "redis"],
}


def _extract_query_keywords(query: str) -> set[str]:
    """从查询中提取关键词集合，含 CJK 长词窗口切分。"""
    _raw_kw = _CJK_KEYWORD_RE.findall(query.lower())
    keywords = set()
    for kw in _raw_kw:
        # ★ R25修复：连续汉字 >4 字时按2-4字窗口切分
        if re.match(r"[\u4e00-\u9fff]+$", kw) and len(kw) > 4:
            for i in range(len(kw)):
                for n in (4, 3, 2):
                    if i + n <= len(kw):
                        keywords.add(kw[i : i + n])
        else:
            keywords.add(kw)
    return keywords


def handle_recall(provider: Any, args: dict[str, Any]) -> str:
    """主动检索记忆 — 经 ContextManager 精炼后返回精简摘要。

    与 prefetch 不同，recall 是 Agent 主动调用，预算更宽松。
    但仍然经过精炼/去重，并保留 original_content 供 omni_detail 按需拉取。

    检索流程（三种模式）:
      rag 模式（默认）:
        1. HybridRetriever.search() 执行向量+BM25+RRF融合检索
        2. 图谱检索通道补充（知识图谱三元组）
        3. 时间衰减 + 隐私过滤
        4. 主存储验证（过滤索引残留）
        5. 最低相关性过滤（关键词验证）
        6. ContextManager 精炼

      llm 模式（深度检索）:
        1. 同 rag 模式基础流程
        2. 额外 store 内容搜索补充通道（同义词扩展 + 关键词重叠过滤）
        3. 图谱检索通道补充
        4. 后续过滤和精炼同 rag

      无结果 fallback:
        向量+BM25均无结果时，回退到 store 全量关键词匹配

    Args:
        provider: OmniMemProvider 实例，用于访问子组件
        args: 工具调用参数，包含 query/mode(rag|llm)/max_tokens

    Returns:
        JSON 字符串，status 可能为:
          found — 找到相关记忆，包含精炼后的摘要列表
          no_results — 未找到任何相关记忆
    """
    query = args["query"]
    mode = args.get("mode", "rag")
    max_tokens = args.get("max_tokens", 1500)
    user_id = args.get("user_id", "default")
    enable_trace = args.get("enable_trace", False)  # ★ 新增：是否记录检索轨迹
    
    if hasattr(provider, "_rbac") and not provider._rbac.check_permission(user_id, "read"):
        return json.dumps({"status": "blocked", "reason": f"User '{user_id}' lacks 'read' permission"})

    # ★ R27优化：预提取查询关键词，避免同一函数内4次重复正则匹配与CJK切分
    _query_keywords = _extract_query_keywords(query)

    # ★ OPT: 检索超时保护 — ThreadPoolExecutor 替代直接调用
    recall_timeout = provider._config.get("recall_timeout_ms", 5000) / 1000.0
    from concurrent.futures import ThreadPoolExecutor
    import time as _time
    _recall_start = _time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            provider._retriever.search, query,
            max_tokens=max_tokens, mode=mode,
            enable_trace=enable_trace,  # ★ 新增：传递 enable_trace
        )
        try:
            results = future.result(timeout=recall_timeout)
        except TimeoutError:
            future.cancel()
            logger.warning("OmniMem recall timed out (%.1fs), returning empty", recall_timeout)
            results = []
        except Exception as e:
            logger.error("OmniMem recall failed: %s", e)
            results = []
    _recall_latency_ms = (_time.monotonic() - _recall_start) * 1000.0

    # ★ llm 模式补充通道：从 store 做内容搜索，弥补向量/BM25 可能遗漏的
    # 但需要过滤：只保留与 query 有关键词重叠的结果，避免噪音
    if mode == "llm":
        try:
            # ★ 同义词扩展：常见中文近义词/上下位词，弥补 BM25 词袋模型的语义鸿沟
            # 注意：单字会被 _tokenize 丢弃，所以用2+字词
            expanded_queries = [query]
            for key, synonyms in _SYNONYM_MAP.items():
                if key in query:
                    for syn in synonyms:
                        expanded_queries.append(query.replace(key, syn))

            all_store_results = []
            existing_ids = {r.get("memory_id", "") for r in results}
            for eq in expanded_queries:
                all_store_results.extend(provider._store.search_by_content(eq, limit=5))

            # 去重
            seen = set(existing_ids)
            query_keywords = _query_keywords
            for sr in all_store_results:
                mid = sr.get("memory_id", "")
                if mid in seen:
                    continue
                seen.add(mid)
                sr_content = sr.get("content", "").lower()
                # ★ 关键词重叠过滤：至少1个关键词在结果内容中出现（宽松，因为同义词已扩展）
                if query_keywords:
                    overlap_count = sum(1 for kw in query_keywords if kw in sr_content)
                    if overlap_count >= 1:
                        sr["_source"] = "store_supplement"
                        sr["score"] = 0.3
                        results.append(sr)
                # query 无关键词时不追加
        except (TimeoutError, ConnectionError) as e:
            logger.warning("OmniMem llm store supplement failed: %s", e)

    # 图谱检索通道
    if provider._knowledge_graph:
        try:
            # Graph RAG: 生成子图上下文文本（可读性强于原始三元组）
            graph_rag_ctx = provider._knowledge_graph.graph_rag_search(
                query, max_depth=2
            )
            if graph_rag_ctx:
                results.append({
                    "content": graph_rag_ctx,
                    "type": "graph_rag",
                    "confidence": 0.6,
                    "score": 0.5,
                    "_source": "graph_rag",
                })
        except (RuntimeError, ValueError, AttributeError) as e:
            # Fallback: 原始三元组搜索
            try:
                graph_results = provider._knowledge_graph.graph_search(
                    query, max_depth=2, limit=10
                )
                if graph_results:
                    for gr in graph_results[:5]:
                        gr["content"] = (
                            f"{gr.get('subject', '')} {gr.get('predicate', '')} {gr.get('object', '')}"
                        )
                        gr["type"] = "graph_triple"
                        gr["confidence"] = gr.get("confidence", 0.5)
                    results.extend(graph_results[:5])
            except (RuntimeError, ValueError) as e2:
                logger.warning("OmniMem graph recall failed: %s", e2)

    # ★ 时序图谱检索通道：当查询包含时间关键词时，补充时序图谱结果
    _has_temporal_intent = any(kw in query for kw in _TEMPORAL_KEYWORDS)
    if _has_temporal_intent and hasattr(provider, "_temporal_kg") and provider._temporal_kg:
        try:
            from omnimem.deep.knowledge_graph import extract_entities as _kg_extract_entities
            query_entities = _kg_extract_entities(query)
            if query_entities:
                temporal_ctx = provider._temporal_kg.temporal_rag_context(
                    query_entities
                )
                if temporal_ctx:
                    results.append({
                        "content": temporal_ctx,
                        "type": "temporal_kg",
                        "confidence": 0.7,
                        "score": 0.55,
                        "_source": "temporal_kg",
                    })
        except (RuntimeError, ValueError, AttributeError, ImportError) as e:
            logger.warning("OmniMem temporal KG recall failed: %s", e)

    results = provider._temporal_decay.apply(results)
    results = provider._privacy.filter(results, session_id=provider._session_id)

    # ★ 主存储验证：过滤掉向量/BM25索引中残留但主存储已删除的条目
    # 封存记忆（archived/forgotten）不删除，降权 + 标记 sealed 保留可召回
    valid_results = []
    for r in results:
        mid = r.get("memory_id", "")
        if mid:
            entry = provider._store.get(mid)
            if not entry:
                # 主存储中不存在 → 索引残留，跳过
                continue
            if entry.get("archived"):
                # 封存记忆：不跳过，降权 + 标记 sealed
                r["score"] = r.get("score", 0) * 0.3  # 降权到 30%
                r["sealed"] = True
        valid_results.append(r)
    results = valid_results

    # ★ 最低相关性过滤 — 统一所有来源的结果
    # 来源分类：
    #   RRF 融合: score = rrf_score (0.02-0.05), 已在 rrf.py 中过滤 < 0.015
    #   store_supplement: score = 0.3, 已做关键词重叠过滤
    #   graph_triple: confidence = 0.5, 无关键词过滤
    # ★ 关键词验证：对每条结果检查内容与 query 是否有实质关联
    # ★ R25修复：连续汉字需按2-4字窗口切分，避免6字整体匹配不到2字词
    query_keywords = _query_keywords
    filtered = []
    for r in results:
        score = r.get("score", 0)
        if score <= 0:
            continue
        source = r.get("_source", "")
        # ★ store_supplement 已在上方做过关键词过滤，直接通过
        if source == "store_supplement":
            filtered.append(r)
            continue
        # ★ graph_triple: 检查内容是否与 query 关键词有重叠
        if r.get("type") == "graph_triple":
            content = r.get("content", "").lower()
            if query_keywords and any(kw in content for kw in query_keywords):
                filtered.append(r)
            continue
        # ★ temporal_kg: 时序图谱结果，已按时间意图过滤，直接通过
        if source == "temporal_kg":
            filtered.append(r)
            continue
        # ★ RRF 融合结果: score = rrf_score
        # rrf_score < 0.015 的已在 rrf.py 中过滤
        # 但如果 rrf_score 很低（0.015-0.025），可能是单路低排名的噪音
        # 进一步验证：内容是否与 query 有任何关键词重叠
        if score < 0.025:
            if query_keywords:
                content = r.get("content", "").lower()
                has_overlap = any(kw in content for kw in query_keywords)
                if not has_overlap:
                    continue  # 低分且无关键词重叠 → 噪音，跳过
            else:
                # ★ R24修复QUAL-1：无关键词的垃圾查询（如 zzzzzxyz123），
                # 低分结果一定是噪音，直接跳过
                continue
        filtered.append(r)
    results = filtered

    if len(results) < 5:
        query_keywords = _query_keywords
        if query_keywords:
            existing_ids = {r.get("memory_id", "") for r in results}
            # 优先：FTS5 全文搜索
            try:
                fts_results = provider._store.search_by_content(query, limit=10)
                for sf in fts_results:
                    sf_mid = sf.get("memory_id", "")
                    if sf_mid not in existing_ids:
                        sf["_source"] = "store_fts_fallback"
                        sf["score"] = sf.get("score", 0) or 0.2
                        results.append(sf)
                        existing_ids.add(sf_mid)
                        if len(results) >= 5:
                            break
            except Exception as e:
                logger.warning("OmniMem FTS fallback failed: %s", e)
            # 补充：store 全量扫描 + 关键词匹配
            if len(results) < 5:
                try:
                    store_all = provider._store.search(limit=50)
                    for sf in store_all:
                        sf_mid = sf.get("memory_id", "")
                        if sf_mid in existing_ids:
                            continue
                        sf_content = sf.get("content", "").lower()
                        keyword_hits = sum(1 for kw in query_keywords if kw in sf_content)
                        if keyword_hits >= 1:
                            sf["_source"] = "store_fallback"
                            sf["score"] = min(0.15 + keyword_hits * 0.05, 0.35)
                            results.append(sf)
                            existing_ids.add(sf_mid)
                            if len(results) >= 5:
                                break
                except Exception as e:
                    logger.warning("OmniMem store fallback failed: %s", e)
    if not results:
            return json.dumps(
                {
                    "status": "no_results",
                    "query": query,
                    "message": "No relevant memories found.",
                }
            )

    # ★ 经 ContextManager 精炼 — 精简摘要 + 保留原文供 detail 拉取
    refined = provider._context_manager.refine_recall_results(results, max_tokens=max_tokens)

    # ★ 新增：提取检索轨迹
    trace_data = None
    if enable_trace and results:
        trace_data = results[-1].pop("_trace", None)

    # ★ 质量评估：enable_trace=True 时自动记录检索质量
    quality_data = None
    if enable_trace and hasattr(provider, "_quality_evaluator") and provider._quality_evaluator:
        try:
            from omnimem.retrieval.quality_eval import RetrievalQualityEvaluator
            from dataclasses import asdict
            relevant_ids = RetrievalQualityEvaluator.infer_relevant_ids(results)
            metrics = provider._quality_evaluator.evaluate(
                query=query,
                results=results,
                relevant_ids=relevant_ids,
                latency_ms=_recall_latency_ms,
            )
            provider._quality_evaluator.record_evaluation(metrics)
            quality_data = asdict(metrics)
        except Exception as e:
            logger.warning("质量评估记录失败: %s", e)

    provider._audit_logger.log(
        "recall",
        details={"query": query, "mode": mode, "count": len(refined)},
        result="success",
        instance_id=getattr(provider, "_instance_id", None),
    )

    # ★ 召回反馈循环：记录每条被召回的记忆，用于遗忘曲线和排序优化
    for r in refined:
        mid = r.get("memory_id", "")
        if mid:
            try:
                provider._forgetting.record_access(mid)
            except Exception as e:
                logger.debug("recall feedback record_access failed for %s: %s", mid, e)

    return json.dumps(
        {
            "status": "found",
            "query": query,
            "count": len(refined),
            "memories": refined,
            "hint": "Use omni_detail with a memory_id to fetch full content.",
            **({"trace": trace_data} if trace_data else {}),
            **({"_quality": quality_data} if quality_data else {}),
        },
        ensure_ascii=False,
    )
