"""log-translator 插件测试 v1.1.0"""

import logging
import pytest
import importlib.util
import sys
from pathlib import Path

# 直接从文件路径加载
_spec = importlib.util.spec_from_file_location(
    "log_translator",
    str(Path(__file__).parent / "__init__.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LogTranslator = _mod.LogTranslator
get_translator = _mod.get_translator
install_translator = _mod.install_translator
uninstall_translator = _mod.uninstall_translator
TRANSLATION_RULES = _mod.TRANSLATION_RULES
TranslationStats = _mod.TranslationStats


def _make_record(msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg=msg, args=None, exc_info=None
    )


class TestLogTranslator:
    """LogTranslator 类测试"""

    def setup_method(self):
        self.translator = LogTranslator()

    def test_initialization(self):
        assert len(self.translator._compiled_rules) == len(TRANSLATION_RULES)
        assert len(self.translator._compiled_rules) > 0

    def test_filter_returns_true(self):
        record = _make_record("test message")
        assert self.translator.filter(record) is True

    def test_translate_tool_completed(self):
        record = _make_record("tool terminal completed (2.5s, 1234 chars)")
        self.translator.filter(record)
        assert "工具" in record.getMessage()
        assert "完成" in record.getMessage()

    def test_translate_tool_failed(self):
        record = _make_record("tool read_file failed (1.0s): File not found")
        self.translator.filter(record)
        assert "工具" in record.getMessage()
        assert "失败" in record.getMessage()

    def test_translate_credential_pool(self):
        record = _make_record("credential pool: no available entries (all exhausted or empty)")
        self.translator.filter(record)
        assert "凭证池" in record.getMessage()
        assert "无可用条目" in record.getMessage()

    def test_translate_model_router(self):
        record = _make_record("Model Router: [abc12345] gpt-4/openai → mimo-v2.5-pro/mimo | 用户指定使用 MiMo")
        self.translator.filter(record)
        assert "模型路由" in record.getMessage()

    def test_translate_plugin_loaded(self):
        record = _make_record("Plugin my-plugin loaded")
        self.translator.filter(record)
        assert "插件" in record.getMessage()
        assert "已加载" in record.getMessage()

    def test_translate_error(self):
        record = _make_record("Error: Connection timeout")
        self.translator.filter(record)
        assert "错误" in record.getMessage()

    def test_translate_rate_limit(self):
        record = _make_record("Rate limit exceeded for provider openai")
        self.translator.filter(record)
        assert "速率限制" in record.getMessage()

    def test_translate_fallback(self):
        record = _make_record("Fallback: gpt-4 exhausted, trying mimo")
        self.translator.filter(record)
        assert "模型回退" in record.getMessage()

    def test_translate_session(self):
        record = _make_record("Gateway: new session abc123 from cli")
        self.translator.filter(record)
        assert "Gateway" in record.getMessage()
        assert "新会话" in record.getMessage()

    def test_no_translation_for_unknown(self):
        msg = "some random log message that should not be translated"
        record = _make_record(msg)
        self.translator.filter(record)
        assert record.getMessage() == msg

    def test_translate_conversation_turn(self):
        record = _make_record(
            "conversation turn: session=abc123 model=gpt-4 provider=openai platform=cli history=5 msg='hello'"
        )
        self.translator.filter(record)
        assert "会话轮次" in record.getMessage()


class TestTranslatorModes:
    """翻译模式测试"""

    def test_replace_mode(self):
        translator = LogTranslator(mode="replace")
        record = _make_record("Error: something broke")
        translator.filter(record)
        assert record.getMessage().startswith("错误:")
        assert "Error" not in record.getMessage()

    def test_append_mode(self):
        translator = LogTranslator(mode="append")
        record = _make_record("Error: something broke")
        translator.filter(record)
        msg = record.getMessage()
        assert "Error: something broke" in msg
        assert "└─" in msg
        assert "错误" in msg


class TestTranslatorSingleton:
    """单例管理测试"""

    def test_get_translator_singleton(self):
        t1 = get_translator()
        t2 = get_translator()
        assert t1 is t2

    def test_install_uninstall(self):
        install_translator(mode="append")
        t = get_translator()
        assert t.mode == "append"
        assert t in [f for f in logging.getLogger().filters if isinstance(f, LogTranslator)]

        uninstall_translator()
        assert not any(isinstance(f, LogTranslator) for f in logging.getLogger().filters)


class TestTranslationRules:
    """翻译规则测试"""

    def test_rules_count(self):
        assert len(TRANSLATION_RULES) > 40

    def test_rules_are_tuples(self):
        for rule in TRANSLATION_RULES:
            assert isinstance(rule, tuple)
            assert len(rule) == 3  # (pattern, replacement, keywords)
            pattern, replacement, keywords = rule
            assert isinstance(pattern, str)
            assert isinstance(replacement, str)
            assert isinstance(keywords, list)

    def test_rules_compile(self):
        import re
        for pattern, _, _ in TRANSLATION_RULES:
            try:
                re.compile(pattern)
            except re.error as e:
                pytest.fail(f"正则编译失败: {pattern}, 错误: {e}")

    def test_dynamic_add_rule(self):
        translator = LogTranslator()
        before = len(translator._compiled_rules)
        translator.add_rule(r"TEST_PATTERN_(\d+)", r"测试模式_\1")
        assert len(translator._compiled_rules) == before + 1

        record = _make_record("TEST_PATTERN_42")
        translator.filter(record)
        assert "测试模式_42" in record.getMessage()


class TestKeywordIndex:
    """关键词索引测试"""

    def test_keyword_index_populated(self):
        translator = LogTranslator()
        assert len(translator._keyword_index) > 0

    def test_keyword_index_covers_rules(self):
        translator = LogTranslator()
        # 每条规则至少有一个关键词被索引
        for idx in range(len(TRANSLATION_RULES)):
            found = False
            for kw, indices in translator._keyword_index.items():
                if idx in indices:
                    found = True
                    break
            assert found, f"规则 {idx} 未被任何关键词索引覆盖"


class TestLRUCache:
    """LRU 缓存测试"""

    def test_cache_enabled(self):
        translator = LogTranslator(cache_size=100)
        info = translator.get_cache_info()
        assert info["enabled"] is True
        assert info["max_size"] == 100
        assert info["size"] == 0

    def test_cache_disabled(self):
        translator = LogTranslator(cache_size=0)
        info = translator.get_cache_info()
        assert info["enabled"] is False

    def test_cache_hit(self):
        translator = LogTranslator(cache_size=100)
        msg = "Error: test cache"

        record1 = _make_record(msg)
        translator.filter(record1)
        assert translator.get_cache_info()["size"] == 1

        record2 = _make_record(msg)
        translator.filter(record2)
        # 翻译结果应该一致
        assert record1.getMessage() == record2.getMessage()

    def test_cache_eviction(self):
        translator = LogTranslator(cache_size=3)

        for i in range(5):
            record = _make_record(f"Error: message {i}")
            translator.filter(record)

        # 缓存大小不应超过 3
        assert translator.get_cache_info()["size"] <= 3

    def test_cache_does_not_affect_translation(self):
        """缓存不应改变翻译结果"""
        translator = LogTranslator(cache_size=100)

        # 第一次翻译
        record1 = _make_record("tool terminal completed (2.5s, 1234 chars)")
        translator.filter(record1)
        result1 = record1.getMessage()

        # 第二次翻译（缓存命中）
        record2 = _make_record("tool terminal completed (2.5s, 1234 chars)")
        translator.filter(record2)
        result2 = record2.getMessage()

        assert result1 == result2


class TestTranslationStats:
    """翻译统计测试"""

    def test_stats_enabled(self):
        translator = LogTranslator(enable_stats=True)
        assert translator._stats is not None

    def test_stats_disabled(self):
        translator = LogTranslator(enable_stats=False)
        assert translator._stats is None

    def test_stats_tracks_translations(self):
        translator = LogTranslator(enable_stats=True)

        # 翻译一条
        record = _make_record("Error: test")
        translator.filter(record)

        # 未翻译一条
        record = _make_record("random message")
        translator.filter(record)

        stats = translator.get_stats()
        assert stats["total_messages"] == 2
        assert stats["translated"] == 1
        assert stats["misses"] == 1

    def test_stats_tracks_cache_hits(self):
        translator = LogTranslator(cache_size=100, enable_stats=True)

        msg = "Error: cached message"
        record1 = _make_record(msg)
        translator.filter(record1)

        record2 = _make_record(msg)
        translator.filter(record2)

        stats = translator.get_stats()
        assert stats["cache_hits"] == 1
        assert stats["translated"] == 1  # 实际正则翻译只发生一次

    def test_stats_reset(self):
        translator = LogTranslator(enable_stats=True)

        record = _make_record("Error: test")
        translator.filter(record)

        translator.reset_stats()
        stats = translator.get_stats()
        assert stats["total_messages"] == 0
        assert stats["translated"] == 0

    def test_stats_summary_format(self):
        translator = LogTranslator(enable_stats=True)

        for msg in ["Error: a", "Error: b", "tool x completed (1s, 10 chars)"]:
            record = _make_record(msg)
            translator.filter(record)

        stats = translator.get_stats()
        assert "total_messages" in stats
        assert "hit_rate_pct" in stats
        assert "cache_hit_rate_pct" in stats
        assert "top_rules" in stats
        assert isinstance(stats["top_rules"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
