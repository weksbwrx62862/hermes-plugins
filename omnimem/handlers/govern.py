"""OmniMem govern 处理器。

从 provider.py 的 _handle_govern() 和 _scan_memory_conflicts() 方法提取，
通过 provider 参数访问实例组件。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from omnimem.governance.conflict import ConflictResolver
from omnimem.memory.wing_room import _PRIVACY_TO_WING

logger = logging.getLogger(__name__)

_NEGATION_INDICATORS: tuple[str, ...] = (
    "不是",
    "不对",
    "并非",
    "不再",
    "改为",
    "而不是",
    "不用",
    "改用",
    "不要",
    "无法",
    "没能",
    "错误",
    "纠正",
    "not",
    "no longer",
    "instead of",
    "rather than",
)

# ★ R27优化：预编译正则，避免 _scan_memory_conflicts 中每条记忆重复编译
_CONFLICT_KEYWORD_RE = re.compile(r"[\u4e00-\u9fff]{2,4}|[a-zA-Z]{3,}")

class ActionRegistry:
    def __init__(self):
        self._actions = {}

    def register(self, name, handler):
        self._actions[name] = handler

    def unregister(self, name):
        self._actions.pop(name, None)

    def execute(self, name, **kwargs):
        handler = self._actions.get(name)
        if handler is None:
            return {"error": f"Unknown governance action: {name}", "available_actions": list(self._actions.keys())}
        return handler(**kwargs)

    def list_actions(self):
        return list(self._actions.keys())


_registry = ActionRegistry()


def register_action(name: str, handler) -> None:
    _registry.register(name, handler)


def unregister_action(name: str) -> None:
    _registry.unregister(name)


def _scan_memory_conflicts(provider: Any) -> list[dict[str, Any]]:
    """主动扫描所有记忆，检测同主题的矛盾对。

    策略：对所有 fact/preference/correction 类型的记忆，
    用内容关键词做粗筛分组，同组内检查否定词矛盾。
    """
    all_memories = provider._store.search(limit=500)
    checkable = [
        m for m in all_memories if m.get("type", "") in ("fact", "preference", "correction")
    ]
    if len(checkable) < 2:
        return []

    # ★ R17修复BUG-3：数据量少时全部同组，避免矛盾对被错误分组
    groups: dict[str, list[dict[str, Any]]] = {}
    if len(checkable) <= 4:
        groups["_all"] = list(checkable)
    else:
        for m in checkable:
            content = m.get("content", "")
            keywords = _CONFLICT_KEYWORD_RE.findall(content)
            key = "|".join(keywords[:2]) if keywords else "_other"
            groups.setdefault(key, []).append(m)

    conflicts = []
    resolver = ConflictResolver(strategy="latest")
    for key, group in groups.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                ca = a.get("content", "").lower()
                cb = b.get("content", "").lower()
                # ★ R31修复BUG-3：使用 ConflictResolver 统一检测逻辑
                result = resolver.check(
                    ca,
                    existing_memories=[{"content": cb, "memory_id": b.get("memory_id", "")}],
                )
                if result.has_conflict:
                    a_has_neg = any(ni in ca for ni in _NEGATION_INDICATORS)
                    b_has_neg = any(ni in cb for ni in _NEGATION_INDICATORS)
                    overlap = ConflictResolver._compute_overlap(ca, cb)
                    conflicts.append(
                        {
                            "memory_a": {
                                "memory_id": a.get("memory_id", ""),
                                "content": a.get("content", "")[:100],
                                "type": a.get("type", ""),
                            },
                            "memory_b": {
                                "memory_id": b.get("memory_id", ""),
                                "content": b.get("content", "")[:100],
                                "type": b.get("type", ""),
                            },
                            "overlap": round(overlap, 2),
                            "negation_in": "a" if a_has_neg else "b",
                            "conflict_type": result.conflict_type,
                        }
                    )
    return conflicts


# ---- 以下为各 action 处理函数 ----


def _action_resolve_conflict(provider: Any, params: dict) -> str:
    """冲突解决：无 target 时全局扫描，有 target 时检查指定记忆。"""
    target = params.get("target", "")
    # ★ R17修复BUG-3b：无 target 时走全局扫描，有 target 时检查指定记忆
    if not target or not target.strip():
        scan_results = _scan_memory_conflicts(provider)
        if scan_results:
            # ★ R24修复BUG-3：全局扫描时归档每对冲突中较旧的条目
            archived_ids = []
            for pair in scan_results[:5]:
                old_id = pair.get("memory_b", {}).get("memory_id") or pair.get("memory_a", {}).get(
                    "memory_id"
                )
                if old_id:
                    provider._forgetting.archive(old_id)
                    archived_ids.append(old_id)
            return json.dumps(
                {
                    "status": "conflicts_found",
                    "action_taken": "archived_old_entries",
                    "reason": f"Found {len(scan_results)} conflicting pairs, archived {len(archived_ids)} old entries",
                    "conflicts": scan_results[:5],
                    "archived": archived_ids,
                }
            )
        return json.dumps(
            {
                "status": "no_conflict",
                "reason": "No conflicting memories found (global scan)",
            }
        )

    target_entry = provider._store.get(target)
    if not target_entry:
        return json.dumps({"status": "error", "reason": f"Memory {target} not found"})

    target_content = target_entry.get("content", "")

    # ★ R28v2修复BUG-3：优先使用 conflict_warning 中记录的冲突对
    conflict_with_id = target_entry.get("conflicting_with", "") or ""
    if not conflict_with_id:
        index_entry = provider._index.get(target)
        if index_entry and isinstance(index_entry, dict):
            conflict_with_id = index_entry.get("conflicting_with", "") or ""
    if conflict_with_id:
        conflict_entry = provider._store.get(conflict_with_id)
        if conflict_entry:
            conflict = provider._conflict_resolver.check(
                target_content,
                existing_memories=[
                    {"content": conflict_entry.get("content", ""), "memory_id": conflict_with_id}
                ],
            )
            if conflict.has_conflict:
                resolution = provider._conflict_resolver.resolve(target_content, conflict)
                old_id = conflict.existing_id
                if old_id and old_id != target:
                    provider._forgetting.archive(old_id)
                return json.dumps(
                    {
                        "status": "resolved",
                        "action_taken": resolution.action,
                        "reason": resolution.reason,
                        "target": target,
                        "conflicting_with": old_id,
                        "source": "conflict_warning_direct",
                    }
                )

    # Fallback：conflict_warning 无记录时，走语义搜索路径
    try:
        semantic_candidates = []
        if hasattr(provider, "_unified_candidate_search"):
            try:
                semantic_candidates = provider._unified_candidate_search(target_content)
            except Exception:
                semantic_candidates = []
        all_memories = provider._store.search(limit=100)
        simple_candidates = [
            m for m in all_memories
            if m.get("memory_id", "") != target
            and m.get("type", "") in ("fact", "preference", "correction")
        ]
        seen_ids = set()
        merged_candidates = []
        for m in semantic_candidates + simple_candidates:
            mid = m.get("memory_id", "")
            if mid and mid != target and mid not in seen_ids:
                seen_ids.add(mid)
                merged_candidates.append(
                    {"content": m.get("content", ""), "memory_id": mid}
                )

        conflict = provider._conflict_resolver.check(
            target_content, existing_memories=merged_candidates
        )

        if conflict.has_conflict:
            resolution = provider._conflict_resolver.resolve(target_content, conflict)
            # ★ R24修复BUG-3：实际归档旧的冲突条目，而不是仅标记 pending
            old_id = conflict.existing_id
            if old_id and old_id != target:
                provider._forgetting.archive(old_id)
                logger.warning("OmniMem resolve_conflict: archived old entry %s", old_id)
            return json.dumps(
                {
                    "status": "resolved",
                    "action_taken": resolution.action,
                    "reason": resolution.reason,
                    "conflicting_with": old_id,
                    "conflict_type": conflict.conflict_type,
                    "archived_old": old_id if old_id and old_id != target else None,
                }
            )
    except (ValueError, KeyError) as e:
        logger.warning("OmniMem conflict detection failed: %s", e)

    # fallback: 全局否定词扫描（即使语义不矛盾，也检查否定词重叠）
    scan_results = _scan_memory_conflicts(provider)
    target_conflicts = [
        c
        for c in scan_results
        if c.get("memory_a", {}).get("memory_id") == target
        or c.get("memory_b", {}).get("memory_id") == target
    ]
    if target_conflicts:
        return json.dumps(
            {
                "status": "conflicts_found",
                "action_taken": "pending",
                "reason": f"Found {len(target_conflicts)} conflicting pairs",
                "conflicts": target_conflicts[:3],
            }
        )

    return json.dumps(
        {
            "status": "no_conflict",
            "reason": "No conflicting memories found for this target",
            "memory_id": target,
        }
    )


def _action_scan_conflicts(provider: Any, params: dict) -> str:
    """主动扫描矛盾记忆：对所有记忆做同主题+矛盾检测。"""
    conflicts = _scan_memory_conflicts(provider)
    return json.dumps(
        {
            "status": "scanned",
            "conflict_count": len(conflicts),
            "conflicts": conflicts[:10],
        },
        ensure_ascii=False,
    )


def _action_set_privacy(provider: Any, params: dict) -> str:
    """设置隐私级别：同步更新 index/store/wing。"""
    target = params.get("target", "")
    # ★ 从多个位置读取 level：params.level / params.privacy / args 顶层
    level = params.get(
        "level", params.get("privacy", "personal")
    )
    # ★ R33修复Minor-4：先计算 new_wing，再传给所有更新操作
    existing = provider._store.get(target)
    existing_type = existing.get("type", existing.get("memory_type", "")) if existing else ""
    new_wing = provider._wing_room.resolve_wing_from_privacy(level, existing_type)
    provider._privacy.set(target, level, new_wing=new_wing)
    provider._index.update_privacy(target, level)
    provider._store.update_privacy(target, level, new_wing=new_wing)
    provider._index.update_field(target, wing=new_wing, immediate=True)
    # ★ R28修复Minor-4回归：同步更新检索索引（向量/BM25）中的 metadata
    if existing and hasattr(provider, "_retriever") and provider._retriever:
        try:
            content = existing.get("content", "")
            if content:
                provider._retriever.update_metadata(
                    target,
                    {
                        "_content": content,
                        "memory_id": target,
                        "type": existing_type,
                        "wing": new_wing,
                        "privacy": level,
                    },
                )
        except Exception as e:
            logger.warning("OmniMem set_privacy retriever sync failed: %s", e)
    # ★ 验证：读回确认是否真正更新
    verify = provider._store.get(target)
    actual_privacy = verify.get("privacy", "personal") if verify else "unknown"
    actual_wing = verify.get("wing", "personal") if verify else "unknown"
    if actual_wing != new_wing:
        logger.warning(
            "govern set_privacy wing mismatch: expected=%s actual=%s for %s",
            new_wing, actual_wing, target,
        )
    provider._audit_logger.log("govern_set_privacy", memory_id=target, details={"privacy": actual_privacy, "wing": actual_wing}, result="success", instance_id=getattr(provider, "_instance_id", None))
    return json.dumps(
        {
            "status": "updated",
            "memory_id": target,
            "privacy": actual_privacy,
            "wing": actual_wing,
        }
    )


def _action_archive(provider: Any, params: dict) -> str:
    """封存记忆（不删除数据，仅标记阶段，保留可召回能力）。"""
    target = params.get("target", "")
    dry_run = params.get("dry_run", False)
    if dry_run:
        return json.dumps({
            "status": "dry_run",
            "action": "archive",
            "target": target,
            "hint": "Set dry_run=false to actually execute",
        })
    # ★ 封存模式：只更新遗忘曲线阶段，不删除底层数据
    provider._forgetting.archive(target)
    # 不再删除 retriever/index/store — 数据保留，召回时降权 + 标记 sealed
    result = {"status": "sealed", "memory_id": target, "note": "数据保留，召回时降权显示"}
    provider._audit_logger.log("govern_archive", memory_id=target, result="sealed", instance_id=getattr(provider, "_instance_id", None))
    return json.dumps(result)


def _action_reactivate(provider: Any, params: dict) -> str:
    """重新激活已归档的记忆。"""
    target = params.get("target", "")
    dry_run = params.get("dry_run", False)
    if dry_run:
        return json.dumps({
            "status": "dry_run",
            "action": "reactivate",
            "target": target,
            "hint": "Set dry_run=false to actually execute",
        })
    provider._forgetting.reactivate(target)
    provider._audit_logger.log("govern_reactivate", memory_id=target, result="success", instance_id=getattr(provider, "_instance_id", None))
    return json.dumps({"status": "reactivated", "memory_id": target})


def _action_provenance(provider: Any, params: dict) -> str:
    """查询记忆溯源信息。"""
    target = params.get("target", "")
    prov = provider._provenance.lookup(target)
    return json.dumps({"status": "found", "provenance": prov})


def _action_forgetting_status(provider: Any, params: dict) -> str:
    """查看遗忘曲线状态。"""
    status = provider._forgetting.get_status()
    return json.dumps({"status": "ok", "forgetting": status})


def _action_lora_train(provider: Any, params: dict) -> str:
    """L4: 触发 LoRA 训练。"""
    if not provider._lora_trainer:
        return json.dumps({"error": "LoRA trainer not available"})
    try:
        shade = params.get("shade", "default")
        epochs = params.get("epochs", 3)
        result = provider._lora_trainer.train(shade=shade, epochs=epochs)
        return json.dumps(result)
    except (RuntimeError, AttributeError) as e:
        logger.warning("OmniMem lora_train failed: %s", e)
        return json.dumps({"status": "error", "reason": f"LoRA train failed: {e}"})


def _action_shade_switch(provider: Any, params: dict) -> str:
    """L4: 切换 LoRA shade（人格切片）。"""
    target = params.get("target", "")
    dry_run = params.get("dry_run", False)
    if dry_run:
        shade_name = params.get("shade") or target or "default"
        return json.dumps({
            "status": "dry_run",
            "action": "shade_switch",
            "target": shade_name,
            "hint": "Set dry_run=false to actually execute",
        })
    if not provider._lora_trainer:
        return json.dumps({"status": "error", "reason": "LoRA trainer not available"})
    try:
        shade_name = params.get("shade") or target or "default"
        available_shades = [s["name"] for s in provider._lora_trainer.get_shades()]
        if shade_name not in available_shades:
            provider._lora_trainer.register_shade(shade_name, f"自定义模式：{shade_name}")
        result = provider._lora_trainer.switch_shade(shade_name)
        # ★ R34修复：switch_shade 可能返回 None
        if result is None:
            result = {"status": "switched", "shade": shade_name}
        verify_shade = provider._lora_trainer.active_shade
        result["verified_active"] = verify_shade
        if verify_shade != shade_name:
            result["status"] = "error"
            result["message"] = f"Switch failed: expected {shade_name}, got {verify_shade}"
        return json.dumps(result)
    except Exception as e:
        logger.warning("OmniMem shade_switch failed: %s", e)
        return json.dumps({"status": "error", "reason": f"Shade switch failed: {e}"})


def _action_shade_list(provider: Any, params: dict) -> str:
    """L4: 列出所有可用 shade。"""
    if not provider._lora_trainer:
        return json.dumps({"error": "LoRA trainer not available"})
    shades = provider._lora_trainer.get_shades()
    return json.dumps({"status": "ok", "shades": shades})


def _action_kv_cache_stats(provider: Any, params: dict) -> str:
    """L4: 查看 KV Cache 统计。"""
    if not provider._kv_cache:
        return json.dumps({"error": "KV Cache not available"})
    stats = provider._kv_cache.get_stats()
    return json.dumps({"status": "ok", "kv_cache": stats})


def _action_consolidation_stats(provider: Any, params: dict) -> str:
    """L3: 查看 Consolidation 统计。"""
    if not provider._consolidation:
        return json.dumps({"error": "Consolidation not available"})
    stats = provider._consolidation.get_stats()
    return json.dumps({"status": "ok", "consolidation": stats})


def _action_sync_status(provider: Any, params: dict) -> str:
    """查看同步引擎状态。"""
    if not provider._sync_engine:
        return json.dumps({"error": "Sync engine not available"})
    info = provider._sync_engine.get_instance_info()
    return json.dumps({"status": "ok", "sync": info})


def _action_sync_instances(provider: Any, params: dict) -> str:
    """查看活跃同步实例。"""
    if not provider._sync_engine:
        return json.dumps({"error": "Sync engine not available"})
    instances = provider._sync_engine.get_active_instances()
    return json.dumps({"status": "ok", "instances": instances})


def _action_export_memories(provider: Any, params: dict) -> str:
    """导出记忆到文件。"""
    from omnimem.core.import_export import MemoryExporter

    output_path = params.get("output_path", "")
    if not output_path:
        return json.dumps({"error": "output_path is required for export_memories"})
    fmt = params.get("format", "json")
    exporter = MemoryExporter(provider._store, provider._index, provider._store._meta_store)
    try:
        if fmt == "markdown":
            count = exporter.export_markdown(output_path, wing=params.get("wing"))
        else:
            count = exporter.export_json(
                output_path,
                wing=params.get("wing"),
                memory_type=params.get("memory_type"),
            )
        return json.dumps({"status": "exported", "count": count, "path": str(output_path)})
    except Exception as e:
        return json.dumps({"error": f"Export failed: {e}"})


def _action_import_memories(provider: Any, params: dict) -> str:
    """从文件导入记忆。"""
    from omnimem.core.import_export import MemoryImporter

    input_path = params.get("input_path", "")
    if not input_path:
        return json.dumps({"error": "input_path is required for import_memories"})
    skip_dup = params.get("skip_duplicates", True)
    resolve_conf = params.get("resolve_conflicts", True)
    importer = MemoryImporter(
        provider._store,
        provider._index,
        provider._retriever,
        provider._dedup,
        provider._conflict_resolver,
        provider._forgetting,
    )
    try:
        result = importer.import_json(
            input_path,
            skip_duplicates=skip_dup,
            resolve_conflicts=resolve_conf,
        )
        return json.dumps({"status": "imported", **result})
    except Exception as e:
        return json.dumps({"error": f"Import failed: {e}"})


def _action_audit_log(provider: Any, params: dict) -> str:
    """查询审计日志。"""
    target = params.get("target", "")
    operation = params.get("operation")
    memory_id = params.get("memory_id") or target or None
    from_time = params.get("from_time")
    to_time = params.get("to_time")
    limit = params.get("limit", 100)
    try:
        entries = provider._audit_logger.query(
            operation=operation,
            memory_id=memory_id,
            from_time=from_time,
            to_time=to_time,
            limit=limit,
        )
        return json.dumps({"status": "ok", "count": len(entries), "entries": entries}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Audit log query failed: {e}"})


def _action_assign_role(provider: Any, params: dict) -> str:
    """为用户分配角色。"""
    caller_id = params.get("caller_id", "default")
    if hasattr(provider, "_rbac") and provider._rbac is not None:
        if not provider._rbac.check_permission(caller_id, "govern"):
            return json.dumps({"status": "blocked", "reason": "govern permission required"})
    user_id = params.get("user_id", "default")
    role_name = params.get("role_name", "")
    if not role_name:
        return json.dumps({"error": "role_name is required"})
    provider._rbac.assign_role(user_id, role_name)
    return json.dumps({"status": "assigned", "user_id": user_id, "role": role_name})


def _action_revoke_role(provider: Any, params: dict) -> str:
    """撤销用户角色。"""
    caller_id = params.get("caller_id", "default")
    if hasattr(provider, "_rbac") and provider._rbac is not None:
        if not provider._rbac.check_permission(caller_id, "govern"):
            return json.dumps({"status": "blocked", "reason": "govern permission required"})
    user_id = params.get("user_id", "default")
    role_name = params.get("role_name", "")
    if not role_name:
        return json.dumps({"error": "role_name is required"})
    provider._rbac.revoke_role(user_id, role_name)
    return json.dumps({"status": "revoked", "user_id": user_id, "role": role_name})


def _action_add_role(provider: Any, params: dict) -> str:
    """创建新角色。"""
    caller_id = params.get("caller_id", "default")
    if hasattr(provider, "_rbac") and provider._rbac is not None:
        if not provider._rbac.check_permission(caller_id, "govern"):
            return json.dumps({"status": "blocked", "reason": "govern permission required"})
    role_name = params.get("role_name", "")
    permissions = params.get("permissions", [])
    if not role_name:
        return json.dumps({"error": "role_name is required"})
    provider._rbac.add_role(role_name, permissions)
    return json.dumps({"status": "created", "role": role_name, "permissions": permissions})


def _action_check_permission(provider: Any, params: dict) -> str:
    """检查用户权限。"""
    user_id = params.get("user_id", "default")
    permission = params.get("permission", "")
    if not permission:
        return json.dumps({"error": "permission is required"})
    allowed = provider._rbac.check_permission(user_id, permission)
    return json.dumps({"status": "ok", "user_id": user_id, "permission": permission, "allowed": allowed})


def _action_get_permissions(provider: Any, params: dict) -> str:
    """获取用户权限列表。"""
    user_id = params.get("user_id", "default")
    permissions = provider._rbac.get_user_permissions(user_id)
    return json.dumps({"status": "ok", "user_id": user_id, "permissions": permissions})


def _action_configure_kms(provider: Any, params: dict) -> str:
    """配置 KMS 提供商（local/aws/azure/gcp）。"""
    provider_name = params.get("provider", "local")
    # 使用原始 params 构建配置参数，避免合并注入的元数据键泄漏到 KMS 配置
    raw_params = params.get("_original_params", params)
    config_kwargs = {k: v for k, v in raw_params.items() if k != "provider"}
    try:
        provider._kms.configure_provider(provider_name, **config_kwargs)
        provider._audit_logger.log("govern_configure_kms", details={"provider": provider_name}, result="success", instance_id=getattr(provider, "_instance_id", None))
        return json.dumps({"status": "configured", "provider": provider_name})
    except ValueError as e:
        return json.dumps({"error": str(e)})


def _action_rotate_key(provider: Any, params: dict) -> str:
    """轮换密钥。"""
    key_id = params.get("key_id", "default")
    provider._kms.rotate_key(key_id)
    provider._audit_logger.log("govern_rotate_key", details={"key_id": key_id}, result="success", instance_id=getattr(provider, "_instance_id", None))
    return json.dumps({"status": "rotated", "key_id": key_id})


def _action_kms_status(provider: Any, params: dict) -> str:
    """查看 KMS 状态。"""
    return json.dumps({
        "status": "ok",
        "provider": provider._kms.provider,
        "config": provider._kms._config,
    })


def _action_rebuild_index(provider: Any, params: dict) -> str:
    """全量重建向量+BM25检索索引（修复历史向量退化）。"""
    try:
        entries = provider._index.search_all_for_retrieval(limit=5000)
        if not entries:
            return json.dumps({"status": "no_data", "reason": "No entries in index to rebuild"})
        stats = provider._retriever.rebuild_all_from_entries(entries)
        return json.dumps({
            "status": "rebuilt",
            "entries_processed": len(entries),
            "vector_rebuilt": stats.get("vector", 0),
            "bm25_rebuilt": stats.get("bm25", 0),
        })
    except Exception as e:
        logger.warning("OmniMem rebuild_index failed: %s", e)
        return json.dumps({"status": "error", "reason": str(e)})


def _action_purge_test_data(provider: Any, params: dict) -> str:
    """清理测试残留数据（恢复RAG精度）。"""
    dry_run = params.get("dry_run", True)
    test_patterns = params.get("patterns", ["test_", "QUAL-", "BUG-", "zzzzz", "mock"])
    purged = []
    all_entries = provider._index.search_all_for_retrieval(limit=5000)
    for entry in all_entries:
        content = entry.get("content", "")
        memory_id = entry.get("memory_id", "")
        if any(p in content for p in test_patterns):
            purged.append({"memory_id": memory_id, "content_preview": content[:80]})
            if not dry_run:
                try:
                    provider._forgetting.archive(memory_id)
                    provider._index.delete(memory_id)
                    if hasattr(provider, "_retriever") and provider._retriever:
                        provider._retriever.update_metadata(
                            memory_id, {"_content": "", "memory_id": memory_id}
                        )
                except Exception as e:
                    logger.warning("purge_test_data delete failed for %s: %s", memory_id, e)
    return json.dumps({
        "status": "dry_run" if dry_run else "purged",
        "test_entries_found": len(purged),
        "entries": purged[:20],
        "hint": "Set dry_run=false to actually delete" if dry_run else None,
    }, ensure_ascii=False)


def _action_wiki_upgrade(provider: Any, params: dict) -> str:
    """获取升级候选并执行升级。"""
    candidates = provider._forgetting.get_upgrade_candidates()
    if not candidates:
        return json.dumps({"status": "no_candidates", "message": "No memories eligible for Wiki upgrade."})
    return json.dumps({
        "status": "candidates",
        "count": len(candidates),
        "candidates": candidates[:10],
        "hint": "Use llm-wiki skill to create Wiki pages for these memories."
    }, ensure_ascii=False)


def _action_mark_wiki(provider: Any, params: dict) -> str:
    """标记记忆已升级到 Wiki。"""
    mid = params.get("memory_id") or params.get("target", "")
    path = params.get("wiki_page_path", "")
    if not mid:
        return json.dumps({"status": "error", "message": "memory_id required"})
    try:
        provider._forgetting.mark_upgraded_to_wiki(mid, path)
        return json.dumps({"status": "marked", "memory_id": mid, "wiki_page_path": path})
    except Exception as e:
        return json.dumps({"status": "error", "reason": str(e)})


def _action_forgetting_heat(provider: Any, params: dict) -> str:
    """返回热度分类统计。"""
    status = provider._forgetting.get_status()
    heat = status.get("heat", {})
    return json.dumps({"status": "ok", "heat": heat})


def _action_tree(provider: Any, params: dict) -> str:
    """展示记忆目录树（内化 OpenViking tree()）。"""
    wing = params.get("wing", "")
    hall = params.get("hall", "")
    tree_data = provider._wing_room.tree(wing=wing, hall=hall)
    return json.dumps({"status": "ok", "tree": tree_data}, ensure_ascii=False)


def _action_grep_rooms(provider: Any, params: dict) -> str:
    """搜索 Room 名称（内化 OpenViking grep()）。"""
    pattern = params.get("pattern", "")
    if not pattern:
        return json.dumps({"status": "error", "message": "pattern is required"})
    rooms = provider._wing_room.grep_rooms(pattern)
    return json.dumps({"status": "ok", "rooms": rooms}, ensure_ascii=False)


def _action_count(provider: Any, params: dict) -> str:
    """统计各目录记忆数量。"""
    wing = params.get("wing", "")
    hall = params.get("hall", "")
    room = params.get("room", "")
    counts = provider._wing_room.count_memories(wing=wing, hall=hall, room=room)
    return json.dumps({"status": "ok", "counts": counts}, ensure_ascii=False)


def _action_backup(provider: Any, params: dict) -> str:
    """手动触发数据目录备份。"""
    try:
        backup_path, backup_size = provider._create_backup()
        backup_max_copies = provider._config.get("backup_max_copies", 3)
        provider._cleanup_old_backups(backup_max_copies)
        return json.dumps({
            "status": "ok",
            "backup_path": backup_path,
            "backup_size_kb": round(backup_size / 1024, 1),
        })
    except Exception as e:
        logger.warning("OmniMem 手动备份失败: %s", e)
        return json.dumps({"status": "error", "reason": str(e)})


# ---- 注册所有 action ----

_registry.register("resolve_conflict", _action_resolve_conflict)
_registry.register("scan_conflicts", _action_scan_conflicts)
_registry.register("set_privacy", _action_set_privacy)
_registry.register("archive", _action_archive)
_registry.register("reactivate", _action_reactivate)
_registry.register("provenance", _action_provenance)
_registry.register("forgetting_status", _action_forgetting_status)
_registry.register("lora_train", _action_lora_train)
_registry.register("shade_switch", _action_shade_switch)
_registry.register("shade_list", _action_shade_list)
_registry.register("kv_cache_stats", _action_kv_cache_stats)
_registry.register("consolidation_stats", _action_consolidation_stats)
_registry.register("sync_status", _action_sync_status)
_registry.register("sync_instances", _action_sync_instances)
_registry.register("export_memories", _action_export_memories)
_registry.register("import_memories", _action_import_memories)
_registry.register("audit_log", _action_audit_log)
_registry.register("assign_role", _action_assign_role)
_registry.register("revoke_role", _action_revoke_role)
_registry.register("add_role", _action_add_role)
_registry.register("check_permission", _action_check_permission)
_registry.register("get_permissions", _action_get_permissions)
_registry.register("configure_kms", _action_configure_kms)
_registry.register("rotate_key", _action_rotate_key)
_registry.register("kms_status", _action_kms_status)
_registry.register("rebuild_index", _action_rebuild_index)
_registry.register("purge_test_data", _action_purge_test_data)
_registry.register("wiki_upgrade", _action_wiki_upgrade)
_registry.register("mark_wiki", _action_mark_wiki)
_registry.register("forgetting_heat", _action_forgetting_heat)
_registry.register("tree", _action_tree)
_registry.register("grep_rooms", _action_grep_rooms)
_registry.register("count", _action_count)
_registry.register("backup", _action_backup)


def handle_govern(provider: Any, args: dict[str, Any]) -> str:
    action = args["action"]
    target = args.get("target", "") or args.get("target_id", "")
    params = args.get("params", {})

    combined = {k: v for k, v in args.items() if k not in ("params", "action")}
    combined.update(params)
    combined["target"] = target
    combined["_original_params"] = params

    result = _registry.execute(action, provider=provider, params=combined)
    if isinstance(result, dict):
        return json.dumps(result)
    return result
