"""OmniMem memorize 处理器。

从 provider.py 的 _handle_memorize() 方法提取，通过 provider 参数访问实例组件。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from omnimem.core.llm_memory_manager import LLMMemoryManager, MemoryAction
from omnimem.core.saga import SagaStep
from omnimem.memory.wing_room import _PRIVACY_TO_WING
from omnimem.utils.security import SecurityValidator

logger = logging.getLogger(__name__)


def _enrich_retriever_content(content: str, memory_type: str, room: str = "") -> str:
    """★ R34修复Minor-3：为 secret/skill/procedural 类型附加可搜索描述。

    问题：secret 内容（如 API 密钥 sk-abc123）与自然语言查询（如"API密钥"）
    语义相似度极低，向量搜索无法匹配。skill/procedural 的步骤化内容与概括性
    查询也存在语义鸿沟。

    方案：在写入向量索引时，为这些类型的内容附加类型标签和房间名作为
    可搜索的语义锚点，使自然语言查询能通过标签匹配找到对应记忆。
    """
    if memory_type == "secret":
        return f"[加密信息/密钥/凭证] {room} {content}"
    elif memory_type == "skill":
        return f"[技能/步骤/教程] {room} {content}"
    elif memory_type == "procedural":
        return f"[流程/操作/指南] {room} {content}"
    return content


def handle_memorize(provider: Any, args: dict[str, Any], llm_memory_manager: LLMMemoryManager | None = None) -> str:
    """处理 omni_memorize 工具调用。

    存储流程（安全扫描→去重→存储→索引）:
      1. 转义字符还原：将 LLM 传入的字面量 \\n/\\t/\\r 还原为实际控制字符
      2. 安全扫描：调用 SecurityValidator.scan_threats 检查注入攻击
      3. 反递归防护：拒绝存储系统注入内容，防止 prefetch→store→prefetch 循环
      4. privacy→scope 推导：根据 privacy 参数推导 wing 分类
      5. 精确内容去重：先检查是否有完全相同的内容
      6. 语义去重：通过 _semantic_dedup 检查相似记忆，高相似度合并更新
      7. 冲突检测：检查同 room/同 type 记忆的矛盾，冲突严重则拒绝
      8. 写入 L2 结构化记忆（DrawerClosetStore）
      9. 写入三层索引（ThreeLevelIndex）
     10. 写入检索索引（HybridRetriever 向量+BM25）
     11. L3: 提取实体和三元组到知识图谱 + 提交到 Consolidation 队列
     12. L4: 检查 KV Cache 自动预填充触发

    Args:
        provider: OmniMemProvider 实例，用于访问子组件
        args: 工具调用参数，包含 content/memory_type/confidence/scope/privacy

    Returns:
        JSON 字符串，status 可能为:
          stored — 成功存储
          duplicate_skipped — 精确或语义重复，跳过
          conflict_rejected — 冲突严重，拒绝存储
          blocked — 安全扫描拦截
          rejected — 反递归防护拦截
    """
    content = args["content"]
    dry_run = args.get("dry_run", False)
    user_id = args.get("user_id", "default")
    if hasattr(provider, "_rbac") and not provider._rbac.check_permission(user_id, "write"):
        return json.dumps({"status": "blocked", "reason": f"User '{user_id}' lacks 'write' permission"})

    # ★ 还原转义字符：LLM 传入的 content 可能含字面量 \\n \\t
    # 在 JSON 解析后这些变成 \n \t 字面量（两个字符），需要还原为实际控制字符
    # ★ R19修复Minor-1-v2：使用正则替换，避免破坏路径中的\normal、\test等
    #   判断依据：\n/\t 后面紧跟字母的是路径（如C:\new\test），不应还原
    #   \n/\t 后面是空格/标点/行尾/非字母的才是真正的转义字符，应还原
    content = re.sub(r"\\n(?![a-zA-Z])", "\n", content)
    content = re.sub(r"\\t(?![a-zA-Z])", "\t", content)
    content = re.sub(r"\\r(?![a-zA-Z])", "\r", content)

    # ★ 安全扫描（统一入口）
    scan_error = SecurityValidator.scan_threats(content)
    if scan_error:
        return json.dumps({"status": "blocked", "reason": scan_error})

    # ★ 反递归防护：拒绝存储系统注入内容，防止 prefetch → store → prefetch 循环
    if not provider._should_store(content):
        return json.dumps(
            {
                "status": "rejected",
                "reason": "Content appears to be a system injection or recursive memory",
            }
        )

    memory_type = args.get("memory_type", "fact")
    confidence = args.get("confidence", 3)
    scope = args.get("scope", "personal")
    privacy = args.get("privacy", "personal")

    # ★ R25修复BUG-1：直接从 privacy 映射到 wing
    # privacy 值直接作为 wing 名称：public→public, team→team, personal→personal, secret→personal
    wing = provider._wing_room.resolve_wing_from_privacy(privacy, memory_type)

    # scope 保留用于 index 存储等需要 scope 字段的场合
    # scope 与 privacy 保持一致，不再做二次映射
    if privacy in _PRIVACY_TO_WING:
        scope = privacy

    # ★ 统一候选搜索：向量+BM25搜索，共享给去重和冲突检测
    candidates = provider._unified_candidate_search(content)
    # 补充 FTS5 精确搜索（防止 ChromaDB 索引延迟导致遗漏）
    if len(candidates) < 5:
        fts_results = provider._store.search_by_content(content, limit=5)
        existing_ids = {m.get("memory_id", "") for m in candidates}
        for m in fts_results:
            if m.get("memory_id", "") not in existing_ids:
                candidates.append(m)

    # ★ 精确内容去重：在合并候选列表中检查完全相同内容
    for m in candidates:
        if m.get("content", "").strip() == content.strip():
            return json.dumps(
                {
                    "status": "duplicate_skipped",
                    "reason": "Exact content already exists",
                    "existing_id": m.get("memory_id", ""),
                }
            )

    # ★ 语义去重：复用已合并的候选结果
    dedup_result = provider._semantic_dedup(content, memory_type, candidates)

    # ★ LLM 决策层：当 LLM 客户端可用时，用 LLM 决策替代规则去重判断
    llm_decision_applied = False
    if llm_memory_manager and llm_memory_manager.is_available:
        try:
            llm_decision = llm_memory_manager.decide(content, memory_type, candidates)
            llm_dedup_result = llm_decision.to_dedup_result()
            llm_action = llm_dedup_result.get("action", "")

            if llm_action == "create":
                dedup_result = {"action": "create"}
                llm_decision_applied = True
                logger.info("OmniMem LLM 决策: ADD — %s", llm_decision.reason)
            elif llm_action == "update":
                dedup_result = llm_dedup_result
                llm_decision_applied = True
                logger.info(
                    "OmniMem LLM 决策: UPDATE target=%s — %s",
                    llm_decision.target_memory_id,
                    llm_decision.reason,
                )
            elif llm_action == "delete":
                # ★ DELETE 决策：删除矛盾记忆，然后继续存储新内容
                delete_target = llm_dedup_result.get("existing_id", "")
                if delete_target:
                    try:
                        provider._forgetting.archive(delete_target)
                        provider._retriever.delete(delete_target)
                        try:
                            provider._index.delete(delete_target)
                        except Exception:
                            pass
                        try:
                            provider._store.delete(delete_target)
                        except Exception:
                            pass
                        logger.info(
                            "OmniMem LLM 决策: DELETE target=%s — %s",
                            delete_target,
                            llm_decision.reason,
                        )
                    except Exception as e:
                        logger.warning("OmniMem LLM DELETE 执行失败: %s", e)
                dedup_result = {"action": "create"}
                llm_decision_applied = True
            elif llm_action == "skip":
                dedup_result = llm_dedup_result
                llm_decision_applied = True
                logger.info("OmniMem LLM 决策: NONE — %s", llm_decision.reason)
        except Exception as e:
            logger.warning("OmniMem LLM 决策层异常，回退到规则去重: %s", e)

    if not llm_decision_applied:
        # ★ 规则去重回退路径（LLM 不可用或异常时）
        pass

    if dedup_result["action"] == "update":
        existing_id = dedup_result["existing_id"]
        provider._forgetting.archive(existing_id)
        # ★ LLM UPDATE 决策：更新已有记忆内容，标记旧内容到 provenance
        updated_content = dedup_result.get("updated_content", "")
        if updated_content:
            try:
                existing_entry = provider._store.get(existing_id) if existing_id else {}
                old_provenance = existing_entry.get("provenance", {})
                if isinstance(old_provenance, str):
                    try:
                        old_provenance = json.loads(old_provenance)
                    except (json.JSONDecodeError, TypeError):
                        old_provenance = {}
                old_provenance["llm_update_reason"] = dedup_result.get("reason", "")
                old_provenance["replaced_by_llm_decision"] = True
                old_provenance["original_content_preview"] = (existing_entry.get("content", "") or "")[:200]
                try:
                    provider._store.update_field(existing_id, provenance=json.dumps(old_provenance, ensure_ascii=False))
                except Exception as e:
                    logger.warning("OmniMem: 更新 provenance 失败: %s", e)
            except Exception as e:
                logger.warning("OmniMem: UPDATE provenance 标记失败: %s", e)
        logger.info("OmniMem dedup: archived duplicate %s, storing updated version", existing_id)
    elif dedup_result["action"] == "skip":
        existing_id = dedup_result.get("existing_id", "")
        existing_entry = provider._store.get(existing_id) if existing_id else {}
        return json.dumps(
            {
                "status": "duplicate_skipped",
                "reason": dedup_result["reason"],
                "existing_id": existing_id,
                "wing": existing_entry.get("wing", ""),
                "privacy": existing_entry.get("privacy", ""),
            }
        )

    # ★ Dry-run 模式：执行去重检测和 wing 映射后返回预览，不执行写入操作
    if dry_run:
        room = provider._wing_room.resolve_room(content, wing, memory_type)
        return json.dumps({
            "status": "dry_run",
            "wing": wing,
            "room": room,
            "dedup_result": dedup_result,
            "content_preview": content[:200],
        }, ensure_ascii=False)

    # 治理：冲突检测
    # ★ 合并候选：语义搜索结果 + 同 room 的记忆（捕捉主题矛盾但语义不相似的情况）
    conflict_candidates = list(candidates[:5])
    _room = provider._wing_room.resolve_room(content, wing, memory_type)
    if wing and _room:
        try:
            same_room = provider._store.search(wing=wing, memory_type=memory_type, limit=10)
            existing_ids = {m.get("memory_id", "") for m in conflict_candidates}
            for m in same_room:
                if m.get("memory_id", "") not in existing_ids:
                    conflict_candidates.append(m)
        except (OSError, KeyError) as e:
            logger.warning("OmniMem same_room search failed: %s", e)

    conflict = provider._conflict_resolver.check(
        content,
        existing_memories=[
            {"content": m.get("content", ""), "memory_id": m.get("memory_id", "")}
            for m in conflict_candidates[:10]
        ],
    )
    conflict_info = None  # ★ 记录冲突信息，用于后续标记
    if conflict.has_conflict:
        resolution = provider._conflict_resolver.resolve(content, conflict)
        if resolution.action == "reject":
            return json.dumps(
                {
                    "status": "conflict_rejected",
                    "reason": resolution.reason,
                    "existing": conflict.existing_memory,
                }
            )
        # ★ 冲突被接受时记录信息，写入后标记到记忆
        conflict_info = {
            "conflict_type": conflict.conflict_type,
            "conflicting_with": conflict.existing_id,
            "reason": resolution.reason,
        }

    # 治理：溯源
    provenance = provider._provenance.track(
        content, source=provider._session_id, method="tool_call"
    )

    # 写入 L2 结构化记忆
    hall = provider._wing_room.resolve_hall(memory_type)
    room = provider._wing_room.resolve_room(content, wing, memory_type)
    # ★ 分布式向量时钟：为每条记忆附加逻辑时钟
    vc = provider.get_next_vc().to_json()
    memory_id = provider._store.add(
        wing=wing,
        room=room,
        content=content,
        memory_type=memory_type,
        confidence=confidence,
        privacy=privacy,
        provenance=provenance,
        vc=vc,
    )

    # ★ P0方案二：Saga 协调派生数据写入
    # 主存储（store.add）已在上方完成，作为唯一事实来源。
    # index / retriever / knowledge_graph 作为派生数据，通过 Saga 保证最终一致。
    now = datetime.now(timezone.utc).isoformat()
    summary = content[:200].replace("\n", " ").replace("\r", " ").replace("\t", " ")

    saga_result = provider._saga.execute(
        memory_id,
        [
            SagaStep(
                "three_level_index",
                lambda: provider._index.add(
                    memory_id=memory_id,
                    wing=wing,
                    hall=hall,
                    room=room,
                    content=content,
                    summary=summary,
                    type=memory_type,
                    confidence=confidence,
                    privacy=privacy,
                    scope=scope,
                    stored_at=now,
                    provenance=json.dumps(provenance) if provenance else "",
                ),
            ),
            SagaStep(
                "retriever",
                lambda: provider._retriever.add(
                    _enrich_retriever_content(content, memory_type, room),
                    memory_id=memory_id,
                    metadata={
                        "memory_id": memory_id,
                        "type": memory_type,
                        "confidence": confidence,
                        "scope": scope,
                        "privacy": privacy,
                        "wing": wing,
                        "room": room,
                        "stored_at": now,
                    },
                ),
            ),
            SagaStep(
                "knowledge_graph",
                lambda: provider._knowledge_graph.extract_and_store(
                    content, memory_id=memory_id, confidence=confidence / 5.0
                )
                if provider._knowledge_graph
                else None,
            ),
        ],
    )

    if not saga_result.success:
        logger.warning(
            "OmniMem memorize saga partial failure for %s at step '%s': %s",
            memory_id,
            saga_result.failed_step,
            saga_result.error,
        )

    # 记录溯源
    provider._provenance.record(memory_id, provenance)

    # 记录遗忘状态
    provider._forgetting.record_access(memory_id)

    # ★ OPT: 记录溯源链 L0 对话 → L1 原子事实
    if hasattr(provider, '_trace_chain') and provider._trace_chain:
        try:
            provider._trace_chain.record_derivation(
                parent_node_ids=[f"conv-{provider._session_id}-turn-{getattr(provider, '_turn_count', 0)}"],
                child_node_id=memory_id,
                child_layer="L1",
                ref_path=str(provider._data_dir / "conversations" / f"{provider._session_id}.jsonl"),
            )
        except Exception as e:
            logger.warning("TraceChain record failed for %s: %s", memory_id, e)

    # ★ OPT: 通知 PipelineScheduler 有新记忆写入（L3 画像触发）
    if hasattr(provider, '_pipeline_scheduler') and provider._pipeline_scheduler:
        provider._pipeline_scheduler.on_new_memory(session_key=provider._session_id)

    # ★ R25修复Minor-3：写入后确保向量索引就绪
    # ChromaDB client.persist() 已在 vector.py 的 add/upsert 中同步调用，
    # 此处仅确保 Saga 已完成的 retriever step 产生的结果已 flush
    provider._retriever.flush()

    # ★ R24修复EXT-5：写入后创建 event 记录（仅高置信度记忆，避免噪声）
    if confidence >= 3:
        try:
            provider._store.add(
                wing="auto",
                room=f"event-{memory_id[:8]}",
                content=f"[create] {content[:120]}",
                memory_type="event",
                confidence=1,
                privacy="personal",
                provenance={"trigger": "memorize", "source_memory_id": memory_id},
            )
        except (OSError, KeyError) as e:
            logger.warning("OmniMem event log creation failed: %s", e)

    # ★ 嵌入缓存持久化（确保新记忆的 embedding 写入磁盘）
    try:
        if hasattr(provider._retriever, '_vector') and hasattr(provider._retriever._vector, '_embedding_fn'):
            emb_fn = provider._retriever._vector._embedding_fn
            if emb_fn and hasattr(emb_fn, 'persist'):
                emb_fn.persist()
    except Exception as e:
        logger.debug("Embedding cache persist skipped: %s", e)

    # L3: 从 Saga 结果中获取知识图谱统计（避免重复执行）
    kg_stats = saga_result.step_results.get("knowledge_graph") or {}
    if not kg_stats and provider._knowledge_graph:
        # Fallback：Saga 未执行 kg 步骤时（如 kg 为 None 被跳过）单独提取
        try:
            kg_stats = provider._knowledge_graph.extract_and_store(
                content, memory_id=memory_id, confidence=confidence / 5.0
            )
        except (ValueError, RuntimeError) as e:
            logger.warning("KnowledgeGraph extraction failed: %s", e)

    # L3: 提交到 Consolidation 队列
    if provider._consolidation:
        provider._consolidation.submit(memory_id, content, memory_type=memory_type)

    # L4: 检查 KV Cache 自动预填充触发
    auto_preloaded = False
    if provider._kv_cache:
        auto_preloaded = provider._kv_cache.check_and_auto_preload(
            key=memory_id,
            content=content,
            metadata={"type": memory_type, "confidence": confidence, "wing": wing},
            source_memory_ids=[memory_id],
        )

    result = {
        "status": "stored",
        "memory_id": memory_id,
        "wing": wing,
        "room": room,
        "type": memory_type,
        "privacy": privacy,
        "kv_cached": auto_preloaded,
    }
    # ★ 冲突自动标记：写入存在冲突的记忆时，返回中包含冲突警告
    if conflict_info:
        result["conflict_warning"] = conflict_info
        logger.warning(
            "OmniMem: stored conflicting memory %s (conflicts with %s: %s)",
            memory_id,
            conflict_info["conflicting_with"],
            conflict_info["reason"],
        )
        # ★ R28v2修复BUG-3：将冲突信息持久化到 store 和 index，
        # 使 resolve_conflict 可直接读取而不需要重新搜索
        conflict_fields = {
            "conflicting_with": conflict_info["conflicting_with"],
            "conflict_type": conflict_info["conflict_type"],
        }
        try:
            provider._store.update_field(memory_id, **conflict_fields)
        except Exception as e:
            logger.warning("OmniMem: failed to persist conflict_info to store: %s", e)
        try:
            provider._index.update_field(memory_id, immediate=True, **conflict_fields)
        except Exception as e:
            logger.warning("OmniMem: failed to persist conflict_info to index: %s", e)

    provider._audit_logger.log(
        "memorize",
        memory_id=memory_id,
        details={"wing": wing, "room": room, "type": memory_type, "privacy": privacy},
        result="success",
        instance_id=getattr(provider, "_instance_id", None),
    )

    return json.dumps(result, ensure_ascii=False)
