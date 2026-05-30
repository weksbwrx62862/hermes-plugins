"""DeepSeek Cache Optimizer v1.2.0 测试"""

import json
import logging
import pytest
import importlib.util
from pathlib import Path

# 直接从文件路径加载
_spec = importlib.util.spec_from_file_location(
    "dco",
    str(Path(__file__).parent / "__init__.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# 导入要测试的组件
normalize_prompt = _mod.normalize_prompt
normalize_messages = _mod.normalize_messages
CacheHitTracker = _mod.CacheHitTracker
get_adaptive_compress_chars = _mod.get_adaptive_compress_chars
_sort_tools = _mod._sort_tools
_compress_prefix_aware = _mod._compress_prefix_aware
_semantic_compress = _mod._semantic_compress
_estimate_tokens = _mod._estimate_tokens
_estimate_tokens_from_str = _mod._estimate_tokens_from_str
_session_states = _mod._session_states
_stats = _mod._stats
_hit_tracker = _mod._hit_tracker


# ═══════════════════════════════════════════════════════════
# Prompt 归一化测试
# ═══════════════════════════════════════════════════════════

class TestPromptNormalization:
    """Prompt 归一化层测试"""

    def test_normalize_uuid(self):
        text = "session 550e8400-e29b-41d4-a716-446655440000 active"
        result = normalize_prompt(text)
        assert "550e8400" not in result
        assert "<UUID>" in result

    def test_normalize_iso_timestamp(self):
        text = "Request at 2026-05-30T07:00:00Z completed"
        result = normalize_prompt(text)
        assert "2026-05-30T07:00:00Z" not in result
        assert "<TS>" in result

    def test_normalize_iso_with_offset(self):
        text = "2026-05-30T07:00:00+08:00 received"
        result = normalize_prompt(text)
        assert "2026-05-30T07:00:00+08:00" not in result
        assert "<TS>" in result

    def test_normalize_unix_timestamp(self):
        text = "timestamp 1748581200000 processed"
        result = normalize_prompt(text)
        assert "1748581200000" not in result
        assert "<TS>" in result

    def test_normalize_request_id(self):
        text = "req-abc12345 request_id=xyz789 done"
        result = normalize_prompt(text)
        assert "<ID>" in result

    def test_normalize_call_id(self):
        text = "call_abc123def456ghi789jkl012 completed"
        result = normalize_prompt(text)
        assert "call_abc123def456ghi789jkl012" not in result
        assert "<ID>" in result

    def test_normalize_fc_id(self):
        text = "fc_abc123def456ghi789jkl012 result"
        result = normalize_prompt(text)
        assert "fc_abc123def456ghi789jkl012" not in result
        assert "<ID>" in result

    def test_normalize_preserves_stable_content(self):
        text = "tool terminal completed (2.5s, 1234 chars)"
        result = normalize_prompt(text)
        # 稳定内容应保留
        assert "tool terminal completed" in result

    def test_normalize_empty(self):
        assert normalize_prompt("") == ""
        assert normalize_prompt(None) == None

    def test_normalize_messages_skip_system(self):
        """system 消息不归一化"""
        messages = [
            {"role": "system", "content": "You are helpful at 2026-05-30T07:00:00Z"},
            {"role": "user", "content": "Request at 2026-05-30T07:00:00Z"},
        ]
        result = normalize_messages(messages)
        # system 消息保持原样
        assert "2026-05-30T07:00:00Z" in result[0]["content"]
        # user 消息被归一化
        assert "<TS>" in result[1]["content"]

    def test_normalize_messages_skip_tool(self):
        """tool 消息不归一化"""
        messages = [
            {"role": "tool", "content": "Result from call_abc123def456ghi789jkl012"},
        ]
        result = normalize_messages(messages)
        assert "call_abc123def456ghi789jkl012" in result[0]["content"]

    def test_normalize_messages_list_content(self):
        """处理 list 格式的 content"""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "at 2026-05-30T07:00:00Z"},
                {"type": "image", "url": "http://example.com"},
            ]},
        ]
        result = normalize_messages(messages)
        assert "<TS>" in result[0]["content"][0]["text"]
        # 非 text 部分不变
        assert result[0]["content"][1]["url"] == "http://example.com"

    def test_normalize_multiple_patterns(self):
        """一条消息中多个动态内容"""
        text = ("session 550e8400-e29b-41d4-a716-446655440000 "
                "at 2026-05-30T07:00:00Z "
                "call_abc123def456ghi789jkl012 done")
        result = normalize_prompt(text)
        assert "<UUID>" in result
        assert "<TS>" in result
        assert "<ID>" in result
        # 静态内容保留
        assert "session" in result
        assert "done" in result


# ═══════════════════════════════════════════════════════════
# 缓存命中率反馈循环测试
# ═══════════════════════════════════════════════════════════

class TestCacheHitTracker:
    """缓存命中率追踪器测试"""

    def setup_method(self):
        self.tracker = CacheHitTracker(window_size=50)

    def test_initial_state(self):
        assert self.tracker.get_global_hit_rate() == 0.0
        report = self.tracker.get_report()
        assert report["global_hit_rate"] == 0.0
        assert report["window_size"] == 0

    def test_record_hit(self):
        self.tracker.record(hit_tokens=1000, miss_tokens=200)
        rate = self.tracker.get_global_hit_rate()
        assert rate > 0.5  # 1000/(1000+200) ≈ 0.83

    def test_record_miss(self):
        self.tracker.record(hit_tokens=100, miss_tokens=900)
        rate = self.tracker.get_global_hit_rate()
        assert rate < 0.5  # 100/(100+900) = 0.1

    def test_sliding_window_eviction(self):
        """窗口满后旧数据被淘汰"""
        tracker = CacheHitTracker(window_size=5)
        for _ in range(10):
            tracker.record(hit_tokens=1000, miss_tokens=0)
        assert len(tracker._global_hits) == 5

    def test_per_tool_tracking(self):
        self.tracker.record(800, 200, tool_name="read_file")
        self.tracker.record(100, 900, tool_name="terminal")
        self.tracker.record(700, 300, tool_name="read_file")

        rates = self.tracker.get_tool_hit_rates()
        assert "read_file" in rates
        assert "terminal" in rates
        assert rates["read_file"] > rates["terminal"]

    def test_cache_hostile_detection(self):
        """命中率低的工具应被标记为缓存不友好"""
        for _ in range(5):
            self.tracker.record(50, 950, tool_name="browser_click")
        for _ in range(5):
            self.tracker.record(900, 100, tool_name="read_file")

        hostile = self.tracker.get_cache_hostile_tools(threshold=0.3)
        hostile_names = [t for t, _, _ in hostile]
        assert "browser_click" in hostile_names
        assert "read_file" not in hostile_names

    def test_hostile_min_samples(self):
        """少于 3 个样本的工具不报告"""
        self.tracker.record(10, 90, tool_name="rare_tool")
        hostile = self.tracker.get_cache_hostile_tools()
        hostile_names = [t for t, _, _ in hostile]
        assert "rare_tool" not in hostile_names

    def test_trend_insufficient_data(self):
        assert self.tracker.get_trend() == "数据不足"

    def test_trend_rising(self):
        """注入低命中率数据，然后高命中率数据"""
        for _ in range(10):
            self.tracker.record(100, 900)  # 10% hit
        for _ in range(10):
            self.tracker.record(900, 100)  # 90% hit
        trend = self.tracker.get_trend()
        assert "上升" in trend

    def test_report_structure(self):
        self.tracker.record(800, 200, tool_name="test_tool")
        report = self.tracker.get_report()
        assert "global_hit_rate" in report
        assert "window_size" in report
        assert "trend" in report
        assert "tool_hit_rates" in report
        assert "cache_hostile_tools" in report

    def test_zero_tokens_ignored(self):
        """hit=0, miss=0 的记录被忽略"""
        self.tracker.record(0, 0)
        assert self.tracker.get_global_hit_rate() == 0.0
        assert len(self.tracker._global_hits) == 0


# ═══════════════════════════════════════════════════════════
# 自适应压缩阈值测试
# ═══════════════════════════════════════════════════════════

class TestAdaptiveCompression:
    """自适应压缩阈值测试"""

    def test_base_value(self):
        """无数据时返回基础值附近"""
        val = get_adaptive_compress_chars()
        assert val > 0

    def test_high_hit_rate_low_threshold(self):
        """高命中率 → 更激进压缩 (小阈值)"""
        # 注入高命中率数据
        tracker = _mod._hit_tracker
        for _ in range(20):
            tracker.record(900, 100)
        val = get_adaptive_compress_chars()
        # 应该比基础值小
        assert val <= _mod.TOOL_RESULT_CAP_CHARS

    def test_low_hit_rate_high_threshold(self):
        """低命中率 → 更保守压缩 (大阈值)"""
        # 创建独立 tracker 模拟低命中率
        tracker = _mod.CacheHitTracker()
        for _ in range(20):
            tracker.record(100, 900)
        # 手动注入到模块级 tracker
        old_tracker = _mod._hit_tracker
        _mod._hit_tracker = tracker
        try:
            val_low = get_adaptive_compress_chars()
            assert val_low >= _mod.ADAPTIVE_COMPRESS_MIN
            # 对比高命中率
            _mod._hit_tracker = _mod.CacheHitTracker()
            for _ in range(20):
                _mod._hit_tracker.record(900, 100)
            val_high = get_adaptive_compress_chars()
            assert val_low >= val_high  # 低命中率阈值 >= 高命中率阈值
        finally:
            _mod._hit_tracker = old_tracker

    def test_long_context_reduces_threshold(self):
        """超长上下文 → 降低阈值"""
        val_normal = get_adaptive_compress_chars(context_tokens=10000)
        val_long = get_adaptive_compress_chars(context_tokens=100000)
        assert val_long <= val_normal

    def test_bounds_respected(self):
        """阈值在 [MIN, MAX] 范围内"""
        for _ in range(20):
            _mod._hit_tracker.record(50, 950)
        val = get_adaptive_compress_chars(context_tokens=200000)
        assert val >= _mod.ADAPTIVE_COMPRESS_MIN
        assert val <= _mod.ADAPTIVE_COMPRESS_MAX


# ═══════════════════════════════════════════════════════════
# 工具排序测试
# ═══════════════════════════════════════════════════════════

class TestToolSort:
    """工具排序测试"""

    def test_sort_stable(self):
        tools = [
            {"function": {"name": "zebra"}},
            {"function": {"name": "alpha"}},
            {"function": {"name": "middle"}},
        ]
        sorted1 = _sort_tools(tools)
        sorted2 = _sort_tools(tools)
        assert sorted1 == sorted2

    def test_sort_by_name(self):
        tools = [
            {"function": {"name": "zebra"}},
            {"function": {"name": "alpha"}},
        ]
        result = _sort_tools(tools)
        names = [t["function"]["name"] for t in result]
        assert names == ["alpha", "zebra"]

    def test_sort_empty(self):
        assert _sort_tools([]) == []

    def test_sort_single(self):
        tools = [{"function": {"name": "only"}}]
        assert _sort_tools(tools) == tools

    def test_sort_none(self):
        assert _sort_tools(None) is None


# ═══════════════════════════════════════════════════════════
# 压缩函数测试
# ═══════════════════════════════════════════════════════════

class TestCompression:
    """压缩函数测试"""

    def test_prefix_aware_short_text(self):
        """短文本不压缩"""
        text = "short text"
        assert _compress_prefix_aware(text, 1000) == text

    def test_prefix_aware_long_text(self):
        """长文本被压缩"""
        text = "a" * 20000
        result = _compress_prefix_aware(text, 7500)
        assert len(result) <= 7500

    def test_prefix_aware_preserves_prefix(self):
        """压缩后前缀保留"""
        text = "IMPORTANT_PREFIX " + "x" * 20000
        result = _compress_prefix_aware(text, 7500)
        assert result.startswith("IMPORTANT_PREFIX")

    def test_semantic_compress_short_text(self):
        text = "short"
        assert _semantic_compress(text, 1000) == text

    def test_semantic_compress_preserves_errors(self):
        """高价值行 (error/fail) 优先保留"""
        lines = ["noise " * 50] * 10
        lines.append("ERROR: critical failure in module X")
        lines.extend(["noise " * 50] * 10)
        text = "\n".join(lines)

        result = _semantic_compress(text, 500)
        assert "ERROR: critical failure" in result


# ═══════════════════════════════════════════════════════════
# Token 估算测试
# ═══════════════════════════════════════════════════════════

class TestTokenEstimation:
    """Token 估算测试"""

    def test_estimate_from_str(self):
        assert _estimate_tokens_from_str("hello") > 0
        assert _estimate_tokens_from_str("a" * 100) > _estimate_tokens_from_str("a" * 10)

    def test_estimate_messages(self):
        messages = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
        ]
        assert _estimate_tokens(messages) > 0

    def test_estimate_empty(self):
        assert _estimate_tokens([]) == 0

    def test_estimate_list_content(self):
        messages = [
            {"role": "user", "content": [
                {"text": "hello"},
                {"text": "world"},
            ]},
        ]
        assert _estimate_tokens(messages) > 0


# ═══════════════════════════════════════════════════════════
# Call-Storm 检测测试
# ═══════════════════════════════════════════════════════════

class TestCallStorm:
    """Call-Storm 检测测试"""

    def test_no_storm_single_call(self):
        result = _mod._post_tool_call(
            session_id="test-storm-1",
            tool_name="read_file",
            tool_args={"path": "/tmp/test"},
        )
        assert result is None

    def test_storm_detected(self):
        session_id = "test-storm-2"
        for _ in range(4):
            result = _mod._post_tool_call(
                session_id=session_id,
                tool_name="terminal",
                tool_args={"command": "ls"},
            )
        assert result is not None
        assert result["action"] == "suppress"
        assert "call-storm" in result["reason"]


# ═══════════════════════════════════════════════════════════
# 集成测试: pre_llm_call (归一化 + 工具排序 + 升级)
# ═══════════════════════════════════════════════════════════

class TestPreLLMCall:
    """Pre-LLM Call hook 集成测试 (v2.0: 只返回 context，不修改 tools/messages)"""

    def test_tool_sorting_ignored_by_hook(self):
        """pre_llm_call 不再修改 tools（框架不支持）"""
        tools = [
            {"function": {"name": "zebra"}},
            {"function": {"name": "alpha"}},
        ]
        result = _mod._pre_llm_call(
            session_id="test-pre-1",
            tools=tools,
            model="mimo-v2.5-pro",
            provider="mimo",
            base_url="https://xiaomimimo.com/v1",
        )
        # v2.0: hook 不返回 tools 修改
        if result:
            assert "tools" not in result

    def test_normalization_not_applied_by_hook(self):
        """pre_llm_call 不再修改 messages（框架不支持）"""
        messages = [
            {"role": "user", "content": "at 2026-05-30T07:00:00Z do something"},
        ]
        result = _mod._pre_llm_call(
            session_id="test-pre-2",
            messages=messages,
            model="deepseek-v4-pro",
            provider="deepseek",
            base_url="https://api.deepseek.com",
        )
        # v2.0: hook 不返回 messages 修改
        if result:
            assert "messages" not in result

    def test_failure_escalation_context(self):
        """失败升级通过 context 注入"""
        session_id = "test-pre-escalation"
        state = _mod._get_session_state(session_id)
        state["failure_count"] = 5  # 超过阈值
        result = _mod._pre_llm_call(
            session_id=session_id,
            model="deepseek-v4-flash",
            conversation_history=[],
        )
        assert result is not None
        assert "context" in result
        assert "deepseek-v4-pro" in result["context"]

    def test_prefix_fingerprint_with_history(self):
        """Prefix 指纹监控在有历史时工作"""
        result = _mod._pre_llm_call(
            session_id="test-pre-fp",
            model="mimo-v2.5-pro",
            conversation_history=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hello"},
            ],
        )
        diag = _mod._prefix_fingerprint.get_diagnostics()
        assert diag["current"]["system_hash"] != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
