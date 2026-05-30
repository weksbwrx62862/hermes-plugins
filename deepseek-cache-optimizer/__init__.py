"""DeepSeek Cache Optimizer v2.0.0 — prefix-cache 稳定性 + 6 大优化。

参考 Reasonix 四大支柱 + 开源社区最佳实践：

Pillar 1 — 缓存优先循环
  - 工具排序（字典序稳定前缀）
  - 前缀保护压缩
  - 三分区消息结构（不可变前缀 + 追加日志 + 临时草稿）

Pillar 2 — 工具调用修复（简化版）
  - call-storm 检测（重复工具调用）
  - 失败信号计数 + 自动升级

Pillar 3 — 成本控制
  - 轮末自动压缩（工具结果 >3000 token → 压缩）
  - 失败信号自动升级（连续 3 次失败 → 升级模型）

v1.2.0 新增 (来自 GPTCache/Helicone/Portkey):
  - ★ Prompt 归一化层：剥离时间戳/UUID/计数器，稳定前缀
  - ★ 缓存命中率反馈循环：按工具追踪+滑动窗口+不友好工具识别
  - ★ 自适应压缩阈值：根据上下文长度和命中率动态调整

v2.0.0 新增 (来自 Reasonix):
  - ★ Reasoning 裁剪：丢弃过时 plain-turn reasoning，保留 API 必需的 tool-call reasoning
  - ★ Prefix 指纹监控：SHA-256 对比 system/tools hash，诊断缓存 miss 原因
  - ★ Cache Miss 诊断：cold-start / system-prompt-changed / tool-list-changed / append-log-drift
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("deepseek_cache_optimizer")

# ─── 配置常量 ──────────────────────────────────────────────

# 轮末压缩：工具结果超过此 token 数会被压缩 (基础值，自适应会动态调整)
TOOL_RESULT_CAP_TOKENS = 3000
CHARS_PER_TOKEN = 2.5
TOOL_RESULT_CAP_CHARS = int(TOOL_RESULT_CAP_TOKENS * CHARS_PER_TOKEN)  # ~7500 字符

# 自适应压缩阈值范围
ADAPTIVE_COMPRESS_MIN = 2000   # 最小阈值 (高命中率时更激进)
ADAPTIVE_COMPRESS_MAX = 6000   # 最大阈值 (低命中率时更保守)
ADAPTIVE_CONTEXT_TRIGGER = 50000  # 上下文超过此 token 数时开始降低阈值

# 失败信号升级
FAILURE_ESCALATION_THRESHOLD = 3
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

# ─── Prompt 归一化模式 (来自 GPTCache) ──────────────────────

# 时间戳模式
_TS_PATTERNS = [
    # ISO 8601: 2026-05-30T07:00:00Z, 2026-05-30T07:00:00+08:00
    re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"),
    # Unix timestamp (10-13 digits): 1748581200, 1748581200000
    re.compile(r"\b1[3-9]\d{8,11}\b"),
    # Common date formats: 2026/05/30, 05/30/2026, 30-May-2026
    re.compile(r"\b\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}\b"),
    # Time: 07:00:00, 7:00 AM
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b"),
]

# UUID 模式
_UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

# 请求 ID / 会话 ID 模式 (各种常见格式)
_ID_PATTERNS = [
    # req-xxx, request_xxx, id:xxx
    re.compile(r"\b(?:req|request|session|trace|span|correlation)[_\-]?id[:\s=]+\S+", re.IGNORECASE),
    # 随机 hex ID (16+ chars)
    re.compile(r"\b[0-9a-f]{16,}\b", re.IGNORECASE),
    # call_xxx (tool call IDs)
    re.compile(r"\bcall_[0-9a-z]{20,}\b", re.IGNORECASE),
    # fc_xxx (function call IDs)
    re.compile(r"\bfc_[0-9a-z]{20,}\b", re.IGNORECASE),
]

# 计数器/递增数字模式 (只替换独立出现的，不替换有意义的内容)
_COUNTER_PATTERNS = [
    # turn=123, count: 456, step 789
    re.compile(r"\b(?:turn|count|step|iteration|attempt|retry)[\s=:]+\d+", re.IGNORECASE),
]


def normalize_prompt(text: str) -> str:
    """
    归一化消息文本，剥离使前缀不稳定的动态内容。

    目的：让相同结构的消息产生相同前缀，提高缓存命中率。
    策略：保守替换，只替换明确的动态内容，保留语义。
    """
    if not text:
        return text

    result = text

    # 1. 替换 UUID
    result = _UUID_PATTERN.sub("<UUID>", result)

    # 2. 替换请求/会话 ID
    for pattern in _ID_PATTERNS:
        result = pattern.sub("<ID>", result)

    # 3. 替换时间戳 (最激进，放在后面)
    for pattern in _TS_PATTERNS:
        result = pattern.sub("<TS>", result)

    return result


def normalize_messages(messages: List[Dict]) -> List[Dict]:
    """
    归一化消息列表中的文本内容。
    只处理 role 不是 system 的消息 (system 消息通常稳定)。
    """
    normalized = []
    for msg in messages:
        role = msg.get("role", "")
        # system 消息和 tool 定义通常稳定，跳过
        if role in ("system", "tool"):
            normalized.append(msg)
            continue

        new_msg = dict(msg)
        content = msg.get("content", "")
        if isinstance(content, str):
            new_msg["content"] = normalize_prompt(content)
        elif isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    new_part = dict(part)
                    new_part["text"] = normalize_prompt(part["text"])
                    new_parts.append(new_part)
                else:
                    new_parts.append(part)
            new_msg["content"] = new_parts
        normalized.append(new_msg)
    return normalized


# ─── 缓存命中率反馈循环 (来自 Helicone) ─────────────────────

class CacheHitTracker:
    """
    按工具维度追踪缓存命中率，识别"缓存不友好"的工具。

    使用滑动窗口 (最近 N 个请求) 避免历史数据干扰。
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._lock = threading.RLock()

        # 全局滑动窗口
        self._global_hits: deque = deque(maxlen=window_size)

        # 按工具追踪: tool_name -> deque of (is_hit, tokens, timestamp)
        self._tool_stats: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

        # 命中率趋势: 最近 N 个窗口的命中率
        self._trend: deque = deque(maxlen=20)

    def record(self, hit_tokens: int, miss_tokens: int,
               tool_name: Optional[str] = None):
        """记录一次缓存命中/未命中。"""
        total = hit_tokens + miss_tokens
        if total == 0:
            return

        is_hit = hit_tokens > miss_tokens
        hit_ratio = hit_tokens / total
        now = time.time()

        with self._lock:
            self._global_hits.append((hit_ratio, total, now))

            if tool_name:
                self._tool_stats[tool_name].append((hit_ratio, total, now))

            # 更新趋势 (每 10 个请求算一个窗口)
            if len(self._global_hits) % 10 == 0:
                recent = list(self._global_hits)[-10:]
                avg_hit = sum(r for r, _, _ in recent) / len(recent)
                self._trend.append(avg_hit)

    def get_global_hit_rate(self) -> float:
        """获取全局滑动窗口命中率。"""
        with self._lock:
            if not self._global_hits:
                return 0.0
            return sum(r for r, _, _ in self._global_hits) / len(self._global_hits)

    def get_tool_hit_rates(self) -> Dict[str, float]:
        """获取每个工具的命中率。"""
        with self._lock:
            result = {}
            for tool_name, entries in self._tool_stats.items():
                if entries:
                    result[tool_name] = sum(r for r, _, _ in entries) / len(entries)
            return result

    def get_cache_hostile_tools(self, threshold: float = 0.3) -> List[Tuple[str, float, int]]:
        """
        识别"缓存不友好"的工具 (命中率低于阈值)。

        Returns: [(tool_name, hit_rate, request_count), ...]
        """
        with self._lock:
            hostile = []
            for tool_name, entries in self._tool_stats.items():
                if len(entries) < 3:  # 至少 3 个样本
                    continue
                hit_rate = sum(r for r, _, _ in entries) / len(entries)
                if hit_rate < threshold:
                    hostile.append((tool_name, round(hit_rate, 3), len(entries)))
            return sorted(hostile, key=lambda x: x[1])

    def get_trend(self) -> str:
        """获取命中率趋势描述。"""
        with self._lock:
            if len(self._trend) < 2:
                return "数据不足"
            recent = self._trend[-1]
            earlier = self._trend[0]
            diff = recent - earlier
            if diff > 0.05:
                return f"📈 上升 (+{diff:.1%})"
            elif diff < -0.05:
                return f"📉 下降 ({diff:.1%})"
            else:
                return f"➡️ 平稳 ({recent:.1%})"

    def get_report(self) -> Dict:
        """生成完整报告。"""
        with self._lock:
            global_rate = self.get_global_hit_rate()
            tool_rates = self.get_tool_hit_rates()
            hostile = self.get_cache_hostile_tools()

            return {
                "global_hit_rate": round(global_rate, 3),
                "window_size": len(self._global_hits),
                "trend": self.get_trend(),
                "tool_count": len(tool_rates),
                "tool_hit_rates": {k: round(v, 3) for k, v in
                                   sorted(tool_rates.items(), key=lambda x: x[1])},
                "cache_hostile_tools": [
                    {"tool": t, "hit_rate": r, "samples": n}
                    for t, r, n in hostile
                ],
            }


# 全局追踪器实例
_hit_tracker = CacheHitTracker(window_size=100)


# ─── 自适应压缩阈值 (来自 Portkey) ──────────────────────────

def get_adaptive_compress_chars(context_tokens: int = 0) -> int:
    """
    根据上下文长度和命中率趋势动态调整压缩阈值。

    策略：
    - 命中率高 (>=70%) → 更激进压缩 (小阈值)，减少 token 消耗
    - 命中率低 (<30%) → 更保守压缩 (大阈值)，保留更多内容以维持前缀
    - 上下文超长 → 降低阈值，强制压缩
    """
    base = TOOL_RESULT_CAP_CHARS

    hit_rate = _hit_tracker.get_global_hit_rate()

    # 命中率调整
    if hit_rate >= 0.7:
        # 高命中率：可以更激进压缩
        factor = 0.7
    elif hit_rate >= 0.5:
        # 中等命中率：保持基础值
        factor = 1.0
    elif hit_rate >= 0.3:
        # 低命中率：稍微保守
        factor = 1.3
    else:
        # 极低命中率：非常保守
        factor = 1.5

    # 上下文长度调整
    if context_tokens > ADAPTIVE_CONTEXT_TRIGGER:
        overflow_ratio = context_tokens / ADAPTIVE_CONTEXT_TRIGGER
        # 超长上下文时强制降低阈值
        factor *= max(0.5, 1.0 / (overflow_ratio ** 0.3))

    adaptive = int(base * factor)
    return max(ADAPTIVE_COMPRESS_MIN, min(ADAPTIVE_COMPRESS_MAX, adaptive))


# ─── 会话级状态 ─────────────────────────────────────────────

_state_lock = threading.Lock()
_session_states: Dict[str, Dict] = {}


def _get_session_state(session_id: str) -> Dict:
    """获取或创建会话状态。"""
    if session_id not in _session_states:
        _session_states[session_id] = {
            "failure_count": 0,
            "escalated_this_turn": False,
            "recent_tool_calls": [],
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
    "total_normalizations": 0,  # v1.2.0: 归一化次数
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
                          "total_compactions", "total_storm_suppressions",
                          "total_escalations", "total_normalizations"]:
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
    return int(total_chars / CHARS_PER_TOKEN)


def _has_cache_support(provider: str, model: str, base_url: str) -> bool:
    """检查 provider/model 是否支持 prompt cache。"""
    provider_lower = (provider or "").lower()
    model_lower = (model or "").lower()
    url_lower = (base_url or "").lower()

    if "deepseek" in provider_lower or "deepseek" in url_lower:
        return True
    if "mimo" in provider_lower or "xiaomimimo" in url_lower:
        return True
    if "openai" in provider_lower:
        return True
    if "anthropic" in provider_lower:
        return True
    # v2.0 修复: hook 不传 provider/base_url，通过 model 名称兜底
    if "mimo" in model_lower or "deepseek" in model_lower:
        return True
    return False


# ─── Pillar 1: 工具排序 ────────────────────────────────────

def _sort_tools(tools: List[Dict]) -> List[Dict]:
    """按字典序排序工具列表，确保前缀稳定。"""
    if not tools or len(tools) <= 1:
        return tools
    return sorted(tools, key=lambda t: json.dumps(t, sort_keys=True, ensure_ascii=False))


# ─── Pillar 1: 前缀保护压缩 ────────────────────────────────

def _compress_prefix_aware(
    text: str,
    max_chars: int,
    keep_prefix_ratio: float = 0.6,
    keep_suffix_ratio: float = 0.1,
) -> str:
    """
    前缀保护压缩：保留前 60% + 后 10%，中间用摘要替换。

    这样做是为了保留前缀不变，让 DeepSeek 的 prompt cache 继续命中。
    """
    if len(text) <= max_chars:
        return text

    keep_prefix = int(max_chars * keep_prefix_ratio)
    keep_suffix = int(max_chars * keep_suffix_ratio)
    mid_budget = max_chars - keep_prefix - keep_suffix

    prefix = text[:keep_prefix]
    suffix = text[-keep_suffix:] if keep_suffix > 0 else ""

    # 中间部分提取关键信息
    mid_section = text[keep_prefix:-keep_suffix] if keep_suffix > 0 else text[keep_prefix:]
    mid_lines = mid_section.split("\n")

    # 保留包含关键词的行
    keywords = {"error", "warning", "fail", "exception", "return", "result", "output",
                "错误", "警告", "失败", "返回", "结果", "输出"}
    important_lines = [l for l in mid_lines
                       if any(kw in l.lower() for kw in keywords)]

    mid_summary = "\n".join(important_lines)
    if len(mid_summary) > mid_budget:
        mid_summary = mid_summary[:mid_budget] + "..."

    omitted = len(mid_lines) - len(important_lines)
    marker = f"\n[{omitted} 行省略]\n" if omitted > 0 else "\n"

    result = prefix + marker + mid_summary
    if suffix:
        result += "\n[...]\n" + suffix

    return result[:max_chars]


def _semantic_compress(text: str, max_chars: int) -> str:
    """语义压缩：提取关键信息，丢弃低价值内容。"""
    if len(text) <= max_chars:
        return text

    lines = text.split("\n")

    # 价值评分
    high_value = {"error", "fail", "exception", "traceback", "return",
                  "错误", "失败", "异常", "返回", "警告"}
    mid_value = {"warning", "info", "debug", "print", "log",
                 "信息", "调试", "日志"}

    scored = []
    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in high_value):
            scored.append((2, line))
        elif any(kw in lower for kw in mid_value):
            scored.append((1, line))
        else:
            scored.append((0, line))

    # 先放高价值行，再放中价值行，直到预算用完
    result_parts = []
    current_len = 0

    for priority in [2, 1, 0]:
        for score, line in scored:
            if score != priority:
                continue
            if current_len + len(line) + 1 > max_chars:
                if priority == 2:  # 高价值行截断保留
                    remaining = max_chars - current_len - 1
                    if remaining > 50:
                        result_parts.append(line[:remaining] + "...")
                        current_len += remaining + 3
                continue
            result_parts.append(line)
            current_len += len(line) + 1

    return "\n".join(result_parts)


# ─── Pillar 3: 轮末压缩（transform_tool_result hook）───────

def _transform_tool_result(**kwargs) -> Optional[str]:
    """
    轮末压缩：工具结果超过阈值时自动压缩。

    v1.2.0: 使用自适应阈值。
    """
    result = kwargs.get("result")
    if not result or not isinstance(result, str):
        return None

    context_tokens = kwargs.get("context_tokens", 0)
    cap_chars = get_adaptive_compress_chars(context_tokens)

    if len(result) <= cap_chars:
        return None  # 不需要压缩

    original_len = len(result)

    # 策略 1：JSON 格式化结果 → 只保留摘要
    try:
        data = json.loads(result)
        if isinstance(data, dict):
            keys = list(data.keys())
            result = json.dumps(data, ensure_ascii=False, indent=None)
            if len(result) > cap_chars:
                result = _compress_prefix_aware(result, cap_chars)
        elif isinstance(data, list):
            result = json.dumps(data[:5], ensure_ascii=False)
            if len(data) > 5:
                result = result.rstrip("]") + f", ... ({len(data)} items total)]"
    except (json.JSONDecodeError, TypeError):
        # 非 JSON → 前缀保护压缩
        result = _compress_prefix_aware(result, cap_chars)

    if len(result) < original_len:
        with _stats_lock:
            _stats["total_compactions"] += 1
        saved = original_len - len(result)
        logger.debug(
            "Tool result compacted: %d → %d chars (saved %d, cap=%d, adaptive=%s)",
            original_len, len(result), saved, cap_chars,
            f"hit_rate={_hit_tracker.get_global_hit_rate():.1%}"
        )

    return result if len(result) < original_len else None


# ─── Pillar 2: Call-Storm 检测（post_tool_call hook）───────

def _post_tool_call(**kwargs) -> Optional[Dict]:
    """
    Call-Storm 检测：在滑动窗口内，如果同一工具被相同参数调用超过阈次，
    标记为可能的无限循环。
    """
    session_id = kwargs.get("session_id", "default")
    tool_name = kwargs.get("tool_name", "")
    tool_args = kwargs.get("tool_args", "")

    if not tool_name:
        return None

    state = _get_session_state(session_id)
    args_hash = hashlib.md5(str(tool_args).encode()).hexdigest()[:8]
    now = time.time()

    with _state_lock:
        state["recent_tool_calls"].append((tool_name, args_hash, now))

        # 清理窗口外的记录
        cutoff = now - 60  # 60 秒窗口
        state["recent_tool_calls"] = [
            (n, a, t) for n, a, t in state["recent_tool_calls"]
            if t > cutoff
        ]

        # 检测重复调用
        recent = state["recent_tool_calls"][-STORM_WINDOW:]
        duplicates = sum(1 for n, a, _ in recent
                         if n == tool_name and a == args_hash)

        if duplicates >= STORM_THRESHOLD:
            state["storm_suppressed"] += 1
            with _stats_lock:
                _stats["total_storm_suppressions"] += 1
            logger.warning(
                "Call-Storm detected: %s(%s) called %d times in window. "
                "Possible infinite loop.",
                tool_name, args_hash, duplicates
            )
            return {
                "action": "suppress",
                "reason": f"call-storm: {tool_name} called {duplicates}x with same args",
            }

    return None


# ─── Pillar 3: 失败信号升级（pre_llm_call hook）────────────

def _pre_llm_call(**kwargs) -> Optional[Dict]:
    """
    Pre-LLM call hook:
    1. 检测失败升级 → 注入 context 提示 (Pillar 3)
    2. Prefix 指纹监控 (v2.0, 只读观测)

    注意：此 hook 只能返回 context 字符串，不能修改 messages/tools/model。
    """
    session_id = kwargs.get("session_id", "default")
    model = kwargs.get("model", "")
    conversation_history = kwargs.get("conversation_history", [])

    state = _get_session_state(session_id)
    state["turn_count"] += 1

    # ── 1. Prefix 指纹监控 (v2.0, 只读) ──
    if conversation_history:
        # 从 conversation_history 提取 system 消息做指纹
        system_msgs = [m for m in conversation_history if m.get("role") == "system"]
        _prefix_fingerprint.compute(system_msgs, [])  # tools 不可用，只监控 system

    # ── 2. 失败信号升级 → 注入 context 提示 ──
    context_parts = []

    if state["failure_count"] >= FAILURE_ESCALATION_THRESHOLD:
        escalated = ESCALATION_MODEL_MAP.get(model)
        if escalated and escalated != model:
            state["escalations"] += 1
            with _stats_lock:
                _stats["total_escalations"] += 1
            # 注入升级提示到用户消息
            context_parts.append(
                f"[系统提示：当前模型 {model} 连续失败 {state['failure_count']} 次，"
                f"请使用更强大的模型 {escalated} 重试]"
            )
            logger.warning(
                "Failure escalation: %s → %s (session=%s, failures=%d)",
                model, escalated, session_id, state["failure_count"]
            )
        state["failure_count"] = 0

    state["escalated_this_turn"] = False

    # 只返回 context 字符串（框架会注入到用户消息中）
    if context_parts:
        return {"context": "\n\n".join(context_parts)}
    return None


# ─── 缓存统计收集（post_api_request hook）──────────────────

def _post_api_request(**kwargs) -> Optional[Dict]:
    """收集缓存命中率统计 + 反馈循环 (v1.2.0)。"""
    model = kwargs.get("model", "unknown")
    response = kwargs.get("response")
    usage = kwargs.get("usage")
    if not usage and response:
        usage = getattr(response, "usage", None)

    if not usage:
        return None

    hit_tokens = 0
    miss_tokens = 0
    reasoning_tokens = 0
    total_input = 0

    # 从 normalized usage dict 或 raw usage object 提取缓存统计
    if isinstance(usage, dict):
        hit_tokens = (usage.get("cache_read_tokens", 0)
                      or usage.get("prompt_cache_hit_tokens", 0)
                      or usage.get("cached_tokens", 0))
        miss_tokens = (usage.get("cache_miss_tokens", 0)
                       or usage.get("prompt_cache_miss_tokens", 0))
        total_input = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        reasoning_tokens = usage.get("reasoning_tokens", 0)
    else:
        for attr_name in ["cache_read_tokens", "prompt_cache_hit_tokens", "cached_tokens",
                          "cache_read_input_tokens"]:
            val = getattr(usage, attr_name, None)
            if val and val > 0:
                hit_tokens = val
                break
        details = getattr(usage, "prompt_tokens_details", None)
        if details and not hit_tokens:
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

    # ── 更新全局统计 ──
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

    # ── v1.2.0: 反馈循环 ──
    # 提取当前使用的工具名 (如果有的话)
    tool_name = kwargs.get("tool_name") or kwargs.get("last_tool_name")
    _hit_tracker.record(hit_tokens, miss_tokens, tool_name=tool_name)

    # ── v2.0: Cache Miss 诊断 ──
    miss_reason = _prefix_fingerprint.infer_miss_reason(hit_tokens, miss_tokens)
    if miss_reason:
        logger.info(
            "Cache miss diagnosis: reason=%s hit=%d miss=%d model=%s",
            miss_reason, hit_tokens, miss_tokens, model,
        )
        with _stats_lock:
            _stats["miss_reasons"] = _stats.get("miss_reasons", {})
            _stats["miss_reasons"][miss_reason] = _stats["miss_reasons"].get(miss_reason, 0) + 1

    # 定期保存
    now = time.time()
    if now - _stats["last_save"] > 30 or _stats["total_requests"] % 5 == 0:
        _save_stats()
        _stats["last_save"] = now

    # ── 日志输出 (增强版) ──
    total_hit_rate = _stats["total_hit_tokens"] / max(_stats["total_tokens"], 1) * 100
    window_rate = _hit_tracker.get_global_hit_rate() * 100
    trend = _hit_tracker.get_trend()

    log_msg = (
        f"Cache: hit={hit_tokens} miss={miss_tokens} rate={hit_rate:.1f}% | "
        f"累计: {_stats['total_requests']}请求 {total_hit_rate:.1f}%命中 "
        f"{_stats['total_tokens'] // 10000}万token | "
        f"窗口: {window_rate:.1f}% {trend}"
    )

    # 每 20 个请求报告一次工具维度命中率
    if _stats["total_requests"] % 20 == 0:
        report = _hit_tracker.get_report()
        hostile = report["cache_hostile_tools"]
        if hostile:
            tools_str = ", ".join(f"{t['tool']}({t['hit_rate']:.0%})" for t in hostile[:3])
            log_msg += f" | ⚠️ 缓存不友好: {tools_str}"

    logger.info(log_msg)

    return {
        "cache_hit_tokens": hit_tokens,
        "cache_miss_tokens": miss_tokens,
        "cache_hit_rate": round(hit_rate, 1),
    }


# ─── 公共 API ──────────────────────────────────────────────

def get_stats_report() -> Dict:
    """获取完整统计报告 (v1.2.0)。"""
    with _stats_lock:
        base = dict(_stats)

    base["hit_rate_pct"] = round(
        base["total_hit_tokens"] / max(base["total_tokens"], 1) * 100, 1
    )
    base["uptime_hours"] = round((time.time() - base["start_time"]) / 3600, 1)

    # v1.2.0: 添加反馈循环报告
    base["hit_tracker"] = _hit_tracker.get_report()
    base["adaptive_compress_chars"] = get_adaptive_compress_chars()
    base["prefix_diagnostics"] = _prefix_fingerprint.get_diagnostics()
    base["reasoning_stripped_chars"] = _stats.get("total_reasoning_stripped", 0)
    base["reasoning_strip_count"] = _stats.get("reasoning_strip_count", 0)
    base["miss_reasons"] = _stats.get("miss_reasons", {})

    return base


def get_hit_tracker() -> CacheHitTracker:
    """获取命中率追踪器实例。"""
    return _hit_tracker


# ─── v2.0 新增：Reasoning 裁剪 + Prefix 指纹监控 + Cache Miss 诊断 ──

# Reasoning 内容保留策略（参考 Reasonix reasoning-retention.ts）
_REASONING_KEEP_PATTERNS = [
    re.compile(r"<\|DSML\|", re.IGNORECASE),
    re.compile(r"function_call", re.IGNORECASE),
    re.compile(r"tool_use", re.IGNORECASE),
]


def _has_tool_calls(msg: Dict) -> bool:
    """判断 assistant 消息是否包含工具调用。"""
    if msg.get("tool_calls"):
        return True
    content = msg.get("content", "")
    if isinstance(content, str):
        return any(p.search(content) for p in _REASONING_KEEP_PATTERNS)
    return False


def strip_droppable_reasoning(messages: List[Dict]) -> Tuple[List[Dict], int]:
    """
    裁剪可丢弃的 reasoning_content。

    策略（参考 Reasonix）：
    - 保留：最后一条 user 消息之后的 assistant reasoning（当前活跃上下文）
    - 保留：带 tool_calls 的 assistant reasoning（API 验证需要）
    - 丢弃：其他过时的 plain-turn reasoning

    Returns: (裁剪后的消息列表, 丢弃的 reasoning 字符数)
    """
    if not messages:
        return messages, 0

    # 找到最后一条 user 消息的位置
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    result = []
    total_stripped = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning", "")

        if role == "assistant" and reasoning:
            # 保留：最后 user 之后的 reasoning
            if i > last_user_idx:
                result.append(msg)
                continue

            # 保留：带 tool_calls 的 reasoning
            if _has_tool_calls(msg):
                result.append(msg)
                continue

            # 丢弃：过时的 plain-turn reasoning
            new_msg = dict(msg)
            stripped_len = len(str(reasoning))
            total_stripped += stripped_len
            new_msg["reasoning_content"] = ""
            new_msg["reasoning"] = ""
            result.append(new_msg)
            logger.debug(
                "Stripped droppable reasoning: msg[%d], %d chars saved", i, stripped_len
            )
        else:
            result.append(msg)

    if total_stripped > 0:
        with _stats_lock:
            _stats["total_reasoning_stripped"] = _stats.get("total_reasoning_stripped", 0) + total_stripped
            _stats["reasoning_strip_count"] = _stats.get("reasoning_strip_count", 0) + 1

    return result, total_stripped


# Prefix 指纹监控（参考 Reasonix cache-diagnostics.ts）

class PrefixFingerprint:
    """
    监控 prefix 组件的 SHA-256 指纹，诊断缓存 miss 原因。

    组件：
    - system_hash: system prompt 的哈希
    - tools_hash: 工具列表的哈希
    - prefix_hash: 整体前缀的哈希
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._prev: Dict[str, str] = {}
        self._curr: Dict[str, str] = {}
        self._miss_log: deque = deque(maxlen=50)

    def compute(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, str]:
        """计算当前前缀组件的哈希。"""
        system_parts = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
                system_parts.append(content)
            else:
                break  # system 消息只在开头

        system_text = "\n".join(system_parts)
        tools_text = json.dumps(
            [t.get("function", {}).get("name", "") for t in (tools or [])],
            sort_keys=True
        )

        hashes = {
            "system_hash": hashlib.sha256(system_text.encode()).hexdigest()[:16],
            "tools_hash": hashlib.sha256(tools_text.encode()).hexdigest()[:16],
            "tool_count": len(tools or []),
            "prefix_hash": hashlib.sha256(
                (system_text + "||" + tools_text).encode()
            ).hexdigest()[:16],
        }

        with self._lock:
            self._prev = dict(self._curr)
            self._curr = hashes

        return hashes

    def infer_miss_reason(self, hit_tokens: int, miss_tokens: int) -> Optional[str]:
        """
        推断缓存 miss 原因。
        DeepSeek 只报告 token 数，不报告原因，所以全部本地推断。
        """
        if hit_tokens > miss_tokens:
            return None  # 命中为主，无需诊断

        with self._lock:
            prev = self._prev
            curr = self._curr

        if not prev:
            return "cold-start"

        reasons = []
        if prev.get("system_hash") != curr.get("system_hash"):
            reasons.append("system-prompt-changed")
        if prev.get("tools_hash") != curr.get("tools_hash"):
            reasons.append("tool-list-changed")
        if not reasons:
            reasons.append("append-log-drift")

        reason = reasons[0]
        self._miss_log.append({
            "time": time.time(),
            "reason": reason,
            "hit": hit_tokens,
            "miss": miss_tokens,
        })
        return reason

    def get_diagnostics(self) -> Dict:
        """获取诊断摘要。"""
        with self._lock:
            return {
                "current": dict(self._curr),
                "previous": dict(self._prev),
                "recent_misses": list(self._miss_log)[-10:],
                "miss_reasons": dict(
                    Counter(m["reason"] for m in self._miss_log)
                ) if self._miss_log else {},
            }


_prefix_fingerprint = PrefixFingerprint()


# ─── transform_request hook (v2.0 — 消息/工具修改) ────────

def _transform_request(**kwargs) -> Optional[Dict]:
    """
    Transform request hook — 在 API 调用前修改 messages/tools/model。

    这是唯一能真正修改请求的 hook。执行：
    1. 工具排序（codepoint 稳定前缀）
    2. 消息归一化（剥离时间戳/UUID/ID）
    3. Reasoning 裁剪（丢弃过时 reasoning_content）
    4. Prefix 指纹监控
    """
    messages = kwargs.get("messages", [])
    tools = kwargs.get("tools", [])
    model = kwargs.get("model", "")
    provider = kwargs.get("provider", "")
    base_url = kwargs.get("base_url", "")
    session_id = kwargs.get("session_id", "default")

    modifications = {}

    # ── 1. 工具排序 ──
    if tools and len(tools) > 1:
        sorted_tools = _sort_tools(tools)
        if sorted_tools != tools:
            modifications["tools"] = sorted_tools

    # ── 2. 消息归一化 ──
    if _has_cache_support(provider, model, base_url) and messages:
        normalized = normalize_messages(messages)
        if normalized != messages:
            modifications["messages"] = normalized
            with _stats_lock:
                _stats["total_normalizations"] += 1

    # ── 3. Reasoning 裁剪 ──
    work_messages = modifications.get("messages", messages)
    if _has_cache_support(provider, model, base_url) and work_messages:
        stripped_messages, stripped_chars = strip_droppable_reasoning(work_messages)
        if stripped_chars > 0:
            modifications["messages"] = stripped_messages
            logger.info(
                "Reasoning stripped: %d chars saved (session=%s)",
                stripped_chars, session_id,
            )

    # ── 4. Prefix 指纹监控 ──
    work_messages = modifications.get("messages", messages)
    work_tools = modifications.get("tools", tools)
    if _has_cache_support(provider, model, base_url):
        _prefix_fingerprint.compute(work_messages, work_tools)

    return modifications if modifications else None


# ─── 注册 ──────────────────────────────────────────────────

def register(ctx):
    """注册所有钩子。"""
    ctx.register_hook("pre_llm_call", _pre_llm_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("transform_tool_result", _transform_tool_result)
    ctx.register_hook("post_api_request", _post_api_request)
    ctx.register_hook("transform_request", _transform_request)
    logger.info(
        "DeepSeek Cache Optimizer v2.0.0 registered: "
        "5 hooks (pre_llm_call, post_tool_call, transform_tool_result, post_api_request, transform_request) | "
        "Pillar 1: tool sort + prefix compress + prompt normalization | "
        "Pillar 2: storm detect + failure escalation | "
        "Pillar 3: turn-end compaction + adaptive threshold | "
        "v2.0: transform_request(工具排序+消息归一化+reasoning裁剪) + prefix fingerprint + cache miss diagnosis"
    )
