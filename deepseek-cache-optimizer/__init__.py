"""DeepSeek Cache Optimizer — prefix-cache 稳定性 + Reasonix 机制。

参考 Reasonix 四大支柱实现：

Pillar 1 — 缓存优先循环
  - 工具排序（字节级稳定）
  - 前缀保护压缩
  - 三分区消息结构（不可变前缀 + 追加日志 + 临时草稿）

Pillar 2 — 工具调用修复（简化版）
  - call-storm 检测（重复工具调用）
  - 失败信号计数 + 自动升级

Pillar 3 — 成本控制
  - 轮末自动压缩（工具结果 >3000 token → 压缩）
  - 失败信号自动升级（连续 3 次失败 → 升级模型）
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("deepseek_cache_optimizer")

# ─── 配置常量 ──────────────────────────────────────────────

# 轮末压缩：工具结果超过此 token 数会被压缩
TOOL_RESULT_CAP_TOKENS = 3000
# 每个中文字符 ≈ 1.5 token，每个英文字符 ≈ 0.25 token，取保守值
CHARS_PER_TOKEN = 2.5
TOOL_RESULT_CAP_CHARS = int(TOOL_RESULT_CAP_TOKENS * CHARS_PER_TOKEN)  # ~7500 字符

# 失败信号升级
FAILURE_ESCALATION_THRESHOLD = 3
# 升级目标模型
ESCALATION_MODEL_MAP = {
    "mimo-v2.5": "mimo-v2.5-pro",
    "mimo-v2": "mimo-v2.5-pro",
    "mimo-v2-pro": "mimo-v2.5-pro",
    "deepseek-v4-flash": "deepseek-v4-pro",
    "gpt-4o-mini": "gpt-4o",
    "claude-3-haiku": "claude-3-sonnet",
}

# call-storm 检测窗口
STORM_WINDOW = 5
STORM_THRESHOLD = 3

# ─── 会话级状态 ─────────────────────────────────────────────

_state_lock = threading.Lock()
# 每个 session 的状态
_session_states: Dict[str, Dict] = {}


def _get_session_state(session_id: str) -> Dict:
    """获取或创建会话状态。"""
    if session_id not in _session_states:
        _session_states[session_id] = {
            "failure_count": 0,
            "escalated_this_turn": False,
            "recent_tool_calls": [],  # [(tool_name, args_hash, timestamp)]
            "turn_count": 0,
            "compacted_results": 0,
            "storm_suppressed": 0,
            "escalations": 0,
        }
    # 清理过多的会话（保留最近 50 个）
    if len(_session_states) > 50:
        oldest = min(_session_states, key=lambda k: _session_states[k].get("turn_count", 0))
        del _session_states[oldest]
    return _session_states[session_id]


# ─── 缓存统计数据 ─────────────────────────────────────────

_stats_lock = threading.Lock()
_stats = {
    "total_requests": 0,
    "total_hit_tokens": 0,
    "total_miss_tokens": 0,
    "total_tokens": 0,
    "total_reasoning_tokens": 0,
    "total_compactions": 0,
    "total_storm_suppressions": 0,
    "total_escalations": 0,
    "by_model": {},
    "start_time": time.time(),
    "last_save": time.time(),
}

_stats_path = Path(os.path.expanduser("~/.hermes/deepseek_cache_stats.json"))


def _load_stats():
    global _stats
    try:
        if _stats_path.exists():
            with open(_stats_path) as f:
                saved = json.load(f)
                for k in ["total_hit_tokens", "total_miss_tokens", "total_tokens",
                          "total_requests", "total_reasoning_tokens",
                          "total_compactions", "total_storm_suppressions", "total_escalations"]:
                    _stats[k] = saved.get(k, 0)
                _stats["by_model"] = saved.get("by_model", {})
                logger.info("Loaded cache stats: %d requests, %.1f%% hit rate",
                           _stats["total_requests"],
                           _stats["total_hit_tokens"] / max(_stats["total_tokens"], 1) * 100)
    except Exception as e:
        logger.debug("Failed to load stats: %s", e)


def _save_stats():
    try:
        _stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_stats_path, "w") as f:
            json.dump(_stats, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.debug("Failed to save stats: %s", e)


_load_stats()


# ─── 工具函数 ──────────────────────────────────────────────

def _estimate_tokens_from_str(text: str) -> int:
    """粗略估算字符串的 token 数。"""
    return int(len(text) / CHARS_PER_TOKEN)


def _estimate_tokens(messages: List[Dict]) -> int:
    """粗略估算消息列表的 token 数。"""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total_chars += len(str(part.get("text", "")))
        for tc in msg.get("tool_calls", []):
            total_chars += len(json.dumps(tc.get("function", {}).get("arguments", "")))
    return int(total_chars / CHARS_PER_TOKEN)


def _has_cache_support(provider: str, model: str, base_url: str) -> bool:
    """判断 provider 是否支持 prefix caching。"""
    combined = f"{provider} {model} {base_url}".lower()
    cache_keywords = ["deepseek", "deep-seek", "mimo", "openai", "anthropic", "claude"]
    return any(kw in combined for kw in cache_keywords)


# ─── Pillar 1: 工具排序 ────────────────────────────────────

def _sort_tools(tools: List[Dict]) -> List[Dict]:
    """按工具名字典序排列 tools，确保字节级一致。"""
    if not tools or len(tools) <= 1:
        return tools
    return sorted(tools, key=lambda t: t.get("function", {}).get("name", "") or t.get("name", ""))


# ─── Pillar 1: 前缀保护压缩 ────────────────────────────────

def _compress_prefix_aware(
    messages: List[Dict],
    max_tokens: int,
    preserve_prefix_rounds: int = 4,
) -> List[Dict]:
    """前缀感知的上下文压缩。

    保留 system + 最近 N 轮，中间压缩为摘要。
    永远不修改前缀部分。
    """
    if not messages:
        return messages

    current_tokens = _estimate_tokens(messages)
    if current_tokens <= max_tokens:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    preserve_count = preserve_prefix_rounds * 2
    if len(other_msgs) <= preserve_count:
        return messages

    prefix_msgs = other_msgs[:2]
    middle_msgs = other_msgs[2:-preserve_count]
    recent_msgs = other_msgs[-preserve_count:]

    if not middle_msgs:
        return messages

    summary_parts = []
    for msg in middle_msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            preview = content[:150] + ("..." if len(content) > 150 else "")
            summary_parts.append(f"[{role}]: {preview}")

    summary = "[历史消息摘要]\n" + "\n".join(summary_parts[-6:])
    summary_msg = {"role": "user", "content": summary}

    result = system_msgs + prefix_msgs + [summary_msg] + recent_msgs
    logger.info("Prefix-aware compression: %d msgs → %d msgs, ~%d → ~%d tokens",
               len(messages), len(result), current_tokens, _estimate_tokens(result))
    return result


# ─── Pillar 3: 轮末压缩（transform_tool_result hook）──────

def _transform_tool_result(**kwargs) -> Optional[str]:
    """transform_tool_result 钩子：轮末自动压缩大工具结果。

    参考 Reasonix 的 Turn-End Auto-Compaction：
    - 工具结果超过 TOOL_RESULT_CAP_TOKENS → 截断保留头尾
    - 保留前 60% 和后 20%，中间插入 [...] 标记
    - 比每轮拖着完整结果便宜得多，需要时可重新调用工具
    """
    tool_name = kwargs.get("tool_name", "")
    result = kwargs.get("result", "")
    session_id = kwargs.get("session_id", "")

    if not result or not isinstance(result, str):
        return None

    result_chars = len(result)
    if result_chars <= TOOL_RESULT_CAP_CHARS:
        return None  # 不需要压缩

    # 保留头部 60% 和尾部 20%，中间压缩
    head_size = int(TOOL_RESULT_CAP_CHARS * 0.6)
    tail_size = int(TOOL_RESULT_CAP_CHARS * 0.2)

    head = result[:head_size]
    tail = result[-tail_size:]
    omitted = result_chars - head_size - tail_size

    compacted = f"{head}\n\n[... 省略 {omitted} 字符 ({_estimate_tokens_from_str(str(omitted))} tokens) ...]\n\n{tail}"

    with _state_lock:
        state = _get_session_state(session_id)
        state["compacted_results"] += 1
    with _stats_lock:
        _stats["total_compactions"] += 1

    logger.info(
        "Tool result compacted: %s %d → %d chars (%.1f%% reduction)",
        tool_name, result_chars, len(compacted),
        (1 - len(compacted) / result_chars) * 100,
    )

    return compacted


# ─── Pillar 2: Call-Storm 检测（post_tool_call hook）───────

def _post_tool_call(**kwargs) -> Optional[Dict]:
    """post_tool_call 钩子：call-storm 检测 + 失败计数。"""
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args", {})
    session_id = kwargs.get("session_id", "")
    result = kwargs.get("result", "")

    if not session_id:
        return None

    with _state_lock:
        state = _get_session_state(session_id)

        # call-storm 检测：相同 (tool, args) 在滑动窗口内重复
        args_str = json.dumps(args, sort_keys=True) if args else ""
        args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]
        now = time.time()

        state["recent_tool_calls"].append((tool_name, args_hash, now))
        # 保留最近 STORM_WINDOW 个
        if len(state["recent_tool_calls"]) > STORM_WINDOW * 2:
            state["recent_tool_calls"] = state["recent_tool_calls"][-STORM_WINDOW:]

        # 检测重复
        recent = state["recent_tool_calls"][-STORM_WINDOW:]
        duplicates = sum(1 for t, h, _ in recent if t == tool_name and h == args_hash)

        if duplicates >= STORM_THRESHOLD:
            state["storm_suppressed"] += 1
            with _stats_lock:
                _stats["total_storm_suppressions"] += 1
            logger.warning(
                "Call-storm detected: %s (args_hash=%s) repeated %d times in %d calls",
                tool_name, args_hash, duplicates, STORM_WINDOW,
            )
            # 返回 None（观察性钩子，不修改结果）
            # 但记录状态供 pre_llm_call 使用

        # 失败信号检测
        if isinstance(result, str) and ("error" in result.lower() or "Error" in result):
            state["failure_count"] += 1
            logger.debug("Failure signal: %s (count=%d)", tool_name, state["failure_count"])
        else:
            # 成功调用重置失败计数（但不清零，保留 1 以记录历史）
            state["failure_count"] = max(0, state["failure_count"] - 1)

    return None


# ─── Pillar 3: 失败信号升级（pre_llm_call hook）────────────

def _pre_llm_call(**kwargs) -> Optional[Dict]:
    """pre_llm_call 钩子：工具排序 + 前缀压缩 + 失败信号升级。

    参考 Reasonix 的 Failure-Signal Auto-Escalation：
    - 连续 FAILURE_ESCALATION_THRESHOLD 次工具失败
    - 自动将当前轮模型升级为更强的模型
    - 升级状态对用户可见（通过日志）
    """
    provider = kwargs.get("provider", "")
    model = kwargs.get("model", "")
    base_url = kwargs.get("base_url", "")
    messages = kwargs.get("messages")
    tools = kwargs.get("tools")
    session_id = kwargs.get("session_id", "")

    # 工具排序（对所有支持缓存的 provider 生效）
    optimized = False
    if _has_cache_support(provider, model, base_url):
        if tools and len(tools) > 1:
            sorted_tools = _sort_tools(tools)
            if json.dumps(sorted_tools, sort_keys=True) != json.dumps(tools, sort_keys=True):
                kwargs["tools"] = sorted_tools
                optimized = True

        # 前缀保护压缩
        if messages and len(messages) > 14:
            compressed = _compress_prefix_aware(messages, max_tokens=32000, preserve_prefix_rounds=4)
            if len(compressed) < len(messages):
                kwargs["messages"] = compressed
                optimized = True

    # 失败信号升级（对所有 provider 生效）
    if session_id:
        with _state_lock:
            state = _get_session_state(session_id)

            if state["failure_count"] >= FAILURE_ESCALATION_THRESHOLD:
                # 查找升级目标
                current_model = model.lower() if model else ""
                for src, dst in ESCALATION_MODEL_MAP.items():
                    if src in current_model:
                        kwargs["model"] = dst
                        state["escalations"] += 1
                        state["failure_count"] = 0  # 重置
                        state["escalated_this_turn"] = True
                        with _stats_lock:
                            _stats["total_escalations"] += 1
                        logger.warning(
                            "⚠️ Failure escalation: %s → %s (failures=%d)",
                            model, dst, FAILURE_ESCALATION_THRESHOLD,
                        )
                        return {"modified": True, "reason": f"escalation:{model}→{dst}"}
                        break

            # 新轮开始，重置升级标记
            state["turn_count"] += 1
            state["escalated_this_turn"] = False

    if optimized:
        return {"modified": True, "reason": "cache-optimized"}

    return None


# ─── 缓存统计收集（post_api_request hook）──────────────────

def _post_api_request(**kwargs) -> Optional[Dict]:
    """post_api_request 钩子：收集缓存命中统计。"""
    provider = kwargs.get("provider", "")
    model = kwargs.get("model", "")
    base_url = kwargs.get("base_url", "")
    response = kwargs.get("response")
    # 优先使用 normalized usage dict（由框架 normalize_usage 预处理），fallback 到 raw response.usage
    usage = kwargs.get("usage")
    if not usage and response:
        usage = getattr(response, "usage", None)

    if not usage:
        return None

    hit_tokens = 0
    miss_tokens = 0
    reasoning_tokens = 0

    # 从 normalized usage dict 或 raw usage object 提取缓存统计
    # 字段优先级：canonical (cache_read_tokens) > deepseek (prompt_cache_hit_tokens) > anthropic
    if isinstance(usage, dict):
        # normalized usage dict (来自 _usage_summary_for_api_request_hook)
        hit_tokens = usage.get("cache_read_tokens", 0) or usage.get("prompt_cache_hit_tokens", 0) or usage.get("cached_tokens", 0)
        miss_tokens = usage.get("cache_miss_tokens", 0) or usage.get("prompt_cache_miss_tokens", 0)
        total_input = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        reasoning_tokens = usage.get("reasoning_tokens", 0)
    else:
        # raw usage object
        for attr_name in ["cache_read_tokens", "prompt_cache_hit_tokens", "cached_tokens",
                          "cache_read_input_tokens"]:
            val = getattr(usage, attr_name, None)
            if val and val > 0:
                hit_tokens = val
                break
        # prompt_tokens_details.cached_tokens
        if not hit_tokens:
            details = getattr(usage, "prompt_tokens_details", None)
            if details:
                hit_tokens = getattr(details, "cached_tokens", 0) or 0

        for attr_name in ["cache_miss_tokens", "prompt_cache_miss_tokens"]:
            val = getattr(usage, attr_name, None)
            if val and val > 0:
                miss_tokens = val
                break

        total_input = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0)
        reasoning_tokens = getattr(usage, "reasoning_tokens", 0)

    # 如果有 hit 但没有 miss，用 total - hit 计算
    if hit_tokens > 0 and miss_tokens == 0 and total_input and total_input > hit_tokens:
        miss_tokens = total_input - hit_tokens

    if hit_tokens == 0 and miss_tokens == 0:
        return None

    total = hit_tokens + miss_tokens
    hit_rate = hit_tokens / max(total, 1) * 100

    with _stats_lock:
        _stats["total_requests"] += 1
        _stats["total_hit_tokens"] += hit_tokens
        _stats["total_miss_tokens"] += miss_tokens
        _stats["total_tokens"] += total
        _stats["total_reasoning_tokens"] += reasoning_tokens

        if model not in _stats["by_model"]:
            _stats["by_model"][model] = {"hit": 0, "miss": 0, "tokens": 0, "requests": 0}
        m = _stats["by_model"][model]
        m["hit"] += hit_tokens
        m["miss"] += miss_tokens
        m["tokens"] += total
        m["requests"] += 1

    now = time.time()
    if now - _stats["last_save"] > 30 or _stats["total_requests"] % 5 == 0:
        _save_stats()
        _stats["last_save"] = now

    total_hit_rate = _stats["total_hit_tokens"] / max(_stats["total_tokens"], 1) * 100
    logger.info(
        "Cache: hit=%d miss=%d rate=%.1f%% | 累计: %d请求 %.1f%%命中 %d万token",
        hit_tokens, miss_tokens, hit_rate,
        _stats["total_requests"], total_hit_rate,
        _stats["total_tokens"] // 10000,
    )

    return {
        "cache_hit_tokens": hit_tokens,
        "cache_miss_tokens": miss_tokens,
        "cache_hit_rate": round(hit_rate, 1),
    }


# ─── 注册 ──────────────────────────────────────────────────

def register(ctx):
    """注册所有钩子。"""
    ctx.register_hook("pre_llm_call", _pre_llm_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("transform_tool_result", _transform_tool_result)
    ctx.register_hook("post_api_request", _post_api_request)
    logger.info(
        "DeepSeek Cache Optimizer v1.1 registered: "
        "4 hooks (pre_llm_call, post_tool_call, transform_tool_result, post_api_request) | "
        "Pillar 1: tool sort + prefix compress | "
        "Pillar 2: storm detect + failure escalation | "
        "Pillar 3: turn-end compaction"
    )
