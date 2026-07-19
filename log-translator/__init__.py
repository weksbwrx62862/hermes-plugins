"""
Hermes Log Translator Plugin v1.1.0

把 Hermes 的英文日志/报错信息自动翻译成中文显示。

v1.1.0 优化:
  - LRU 缓存: 相同消息只翻译一次，后续直接命中缓存
  - 关键词预过滤: 按首关键词分组，跳过不相关的规则组
  - 翻译统计: 追踪命中率、缓存命中、最活跃规则
  - 去重: 移除重复规则

功能：
  - 通过 logging.Filter 拦截日志输出
  - 使用模式匹配翻译常见的英文日志消息
  - 支持运行时动态添加翻译规则
  - 零侵入：不修改原始代码，纯粹的日志过滤器

配置 (~/.hermes/config.yaml):
  plugins:
    log-translator:
      enabled: true
      mode: "replace"          # replace=替换原文, append=追加中文注释
      cache_size: 512           # LRU 缓存大小 (0=禁用)
      enable_stats: true        # 启用翻译统计
"""

import logging
import re
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

# ── 翻译规则 ──────────────────────────────────────────────────────────────────

# 格式: (正则模式, 中文替换, 关键词列表)
# 关键词用于预过滤，提高匹配速度
# 支持 %s, %d, %r 等 Python logging 格式占位符
TRANSLATION_RULES: List[Tuple[str, str, List[str]]] = [
    # ── 会话/对话相关 ──
    (r"conversation turn: session=(\S+) model=(\S+) provider=(\S+) platform=(\S+) history=(\d+) msg=(.+)",
     r"会话轮次: 会话=\1 模型=\2 提供者=\3 平台=\4 历史消息数=\5 消息=\6",
     ["conversation", "turn"]),

    # ── 工具调用相关 ──
    (r"tool (\S+) completed \(([0-9.]+)s, (\d+) chars\)",
     r"工具 \1 完成 (耗时 \2 秒, \3 字符)",
     ["tool", "completed"]),
    (r"tool (\S+) failed \(([0-9.]+)s\): (.+)",
     r"工具 \1 失败 (耗时 \2 秒): \3",
     ["tool", "failed"]),

    # ── 凭证池相关 ──
    (r"credential pool: no available entries \(all exhausted or empty\)",
     "凭证池: 无可用条目 (全部耗尽或为空)",
     ["credential"]),
    (r"credential pool: rotated to (\S+)",
     r"凭证池: 已轮转到 \1",
     ["credential"]),
    (r"Credential auth failure — refreshed pool entry (\S+)",
     r"凭证认证失败 — 已刷新池条目 \1",
     ["Credential", "auth"]),

    # ── Model Router 相关 ──
    (r"Model Router: strategy=(\S+)",
     r"模型路由: 策略=\1",
     ["Model", "Router"]),
    (r"Model Router: \[(\S+)\] (\S+)/(\S+) → (\S+)/(\S+) \| (.+)",
     r"模型路由: [\1] \2/\3 → \4/\5 | \6",
     ["Model", "Router"]),
    (r"Model Router: loaded (\d+) MiMo keys for rotation",
     r"模型路由: 已加载 \1 个 MiMo 密钥用于轮换",
     ["Model", "Router", "loaded"]),
    (r"Model Router v(\S+) registered: (\d+) models, (\d+) tools",
     r"模型路由 v\1 已注册: \2 个模型, \3 个工具",
     ["Model", "Router"]),
    (r"Model Router: pre_llm_call hook registered",
     r"模型路由: pre_llm_call 钩子已注册",
     ["Model", "Router", "pre_llm_call"]),

    # ── 插件加载相关 ──
    (r"Plugin '(\S+)' registered (\S+) provider: (\S+)",
     r"插件 '\1' 已注册 \2 提供者: \3",
     ["Plugin"]),
    (r"Plugin '?(\S+)'? loaded",
     r"插件 '\1' 已加载",
     ["Plugin"]),

    # ── Curator 相关 ──
    (r"Curator snapshot created: (\S+) \((\S+)\)",
     r"Curator 快照已创建: \1 (\2)",
     ["Curator", "snapshot"]),
    (r"Curator rollback: restored from (\S+) \(cron_report=(\S+)\)",
     r"Curator 回滚: 已从 \1 恢复 (定时报告=\2)",
     ["Curator", "rollback"]),
    (r"Curator snapshot failed: (.+)",
     r"Curator 快照失败: \1",
     ["Curator", "snapshot"]),
    (r"Cron skill-link restore failed: (.+)",
     r"定时技能链接恢复失败: \1",
     ["Cron", "skill-link"]),

    # ── Model Router 高级 ──
    (r"Model Router: (\S+) 加入冷却黑名单 \(([0-9.]+)s\)",
     r"模型路由: \1 加入冷却黑名单 (\2 秒)",
     ["Model", "Router", "冷却"]),
    (r"Model Router: PyYAML not installed, cannot read config\.yaml",
     "模型路由: PyYAML 未安装，无法读取 config.yaml",
     ["Model", "Router", "PyYAML"]),
    (r"Model Router: failed to read config\.yaml: (.+)",
     r"模型路由: 读取 config.yaml 失败: \1",
     ["Model", "Router", "failed"]),
    (r"Model Router: loaded (\d+) NVIDIA NIM keys for rotation",
     r"模型路由: 已加载 \1 个 NVIDIA NIM 密钥用于轮换",
     ["Model", "Router", "NVIDIA"]),
    (r"Model Router: 所有 (\S+) Key 均不健康，降级使用最少使用策略",
     r"模型路由: 所有 \1 密钥均不健康，降级使用最少使用策略",
     ["Model", "Router", "不健康"]),
    (r"Model Router: 所有 provider 均被限流，使用评分最高模型作为保底",
     "模型路由: 所有提供者均被限流，使用评分最高模型作为保底",
     ["Model", "Router", "限流"]),
    (r"Model Router: _create_openai_client failed: (.+)",
     r"模型路由: 创建 OpenAI 客户端失败: \1",
     ["Model", "Router", "_create_openai_client"]),
    (r"Model Router: OpenAI client fallback also failed: (.+)",
     r"模型路由: OpenAI 客户端回退也失败: \1",
     ["Model", "Router", "OpenAI"]),
    (r"Model Router: 已设置 (\d+) 个降级备选模型 \[([^\]]+)\]",
     r"模型路由: 已设置 \1 个降级备选模型 [\2]",
     ["Model", "Router", "降级"]),
    (r"Model Router: apply routing failed: (.+)",
     r"模型路由: 应用路由失败: \1",
     ["Model", "Router", "apply"]),

    # ── 速率限制相关 ──
    (r"Rate limit exceeded for provider (\S+)",
     r"速率限制: 提供者 \1 超出速率限制",
     ["Rate", "limit"]),
    (r"Rate limit: (\S+) bucket full, retry after ([0-9.]+)s",
     r"速率限制: \1 桶已满，\2 秒后重试",
     ["Rate", "limit", "bucket"]),

    # ── 模型回退相关 ──
    (r"Fallback: (\S+) exhausted, trying (\S+)",
     r"模型回退: \1 已耗尽，尝试 \2",
     ["Fallback"]),
    (r"All models exhausted for provider (\S+)",
     r"所有模型已耗尽: 提供者 \1",
     ["All", "models", "exhausted"]),

    # ── 上下文/压缩相关 ──
    (r"Context window exceeded: (\d+) tokens > (\d+) max",
     r"上下文窗口超限: \1 tokens > \2 最大值",
     ["Context", "window"]),
    (r"Compressing context from (\d+) to (\d+) tokens",
     r"压缩上下文: 从 \1 到 \2 tokens",
     ["Compressing", "context"]),

    # ── Gateway 相关 ──
    (r"Gateway started on port (\d+)",
     r"Gateway 已启动，端口 \1",
     ["Gateway", "started"]),
    (r"Gateway shutdown initiated",
     "Gateway 正在关闭",
     ["Gateway", "shutdown"]),
    (r"Gateway: new session (\S+) from (\S+)",
     r"Gateway: 新会话 \1 来自 \2",
     ["Gateway", "session"]),

    # ── Cron 相关 ──
    (r"Cron job '(\S+)' scheduled: (.+)",
     r"定时任务 '\1' 已调度: \2",
     ["Cron", "job"]),
    (r"Cron job '(\S+)' completed \((\S+)\)",
     r"定时任务 '\1' 已完成 (\2)",
     ["Cron", "completed"]),
    (r"Cron job '(\S+)' failed: (.+)",
     r"定时任务 '\1' 失败: \2",
     ["Cron", "failed"]),

    # ── Delegation 相关 ──
    (r"Delegating task to (\S+): (.+)",
     r"委派任务给 \1: \2",
     ["Delegating"]),
    (r"Subagent (\S+) completed: (.+)",
     r"子代理 \1 已完成: \2",
     ["Subagent", "completed"]),
    (r"Subagent (\S+) failed: (.+)",
     r"子代理 \1 失败: \2",
     ["Subagent", "failed"]),

    # ── 文件操作相关 ──
    (r"File read: (\S+) \((\d+) bytes\)",
     r"文件读取: \1 (\2 字节)",
     ["File", "read"]),
    (r"File written: (\S+) \((\d+) bytes\)",
     r"文件写入: \1 (\2 字节)",
     ["File", "written"]),
    (r"File not found: (\S+)",
     r"文件未找到: \1",
     ["File", "not found"]),

    # ── 通用错误模式 ──
    (r"Error: (.+)",
     r"错误: \1",
     ["Error"]),
    (r"Failed to (.+)",
     r"失败: 无法\1",
     ["Failed"]),
    (r"Warning: (.+)",
     r"警告: \1",
     ["Warning"]),
    (r"Timeout: (.+)",
     r"超时: \1",
     ["Timeout"]),
    (r"Connection refused: (.+)",
     r"连接被拒绝: \1",
     ["Connection", "refused"]),
    (r"Permission denied: (.+)",
     r"权限被拒绝: \1",
     ["Permission", "denied"]),
]


class TranslationStats:
    """翻译统计"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_messages = 0
        self.translated = 0
        self.cache_hits = 0
        self.misses = 0
        self.rule_hits: Dict[str, int] = {}  # rule_index -> count
        self.start_time = time.time()

    def record_hit(self, rule_index: int):
        self.translated += 1
        self.rule_hits[rule_index] = self.rule_hits.get(rule_index, 0) + 1

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_miss(self):
        self.misses += 1

    def record_total(self):
        self.total_messages += 1

    def get_summary(self) -> Dict:
        elapsed = time.time() - self.start_time
        hit_rate = (self.translated / self.total_messages * 100) if self.total_messages else 0
        cache_rate = (self.cache_hits / self.total_messages * 100) if self.total_messages else 0

        # Top 5 most hit rules
        top_rules = sorted(self.rule_hits.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_messages": self.total_messages,
            "translated": self.translated,
            "cache_hits": self.cache_hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 1),
            "cache_hit_rate_pct": round(cache_rate, 1),
            "uptime_seconds": round(elapsed, 1),
            "top_rules": [(idx, count) for idx, count in top_rules],
        }


class LogTranslator(logging.Filter):
    """日志翻译过滤器"""

    def __init__(self, mode: str = "replace", cache_size: int = 512,
                 enable_stats: bool = True):
        super().__init__()
        self.mode = mode
        self._enable_stats = enable_stats
        self._stats = TranslationStats() if enable_stats else None

        # 预编译正则 + 建立关键词索引
        self._compiled_rules: List[Tuple[re.Pattern, str, int]] = []
        self._keyword_index: Dict[str, List[int]] = {}  # keyword -> [rule_indices]

        for idx, (pattern, replacement, keywords) in enumerate(TRANSLATION_RULES):
            compiled = re.compile(pattern)
            self._compiled_rules.append((compiled, replacement, idx))
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower not in self._keyword_index:
                    self._keyword_index[kw_lower] = []
                self._keyword_index[kw_lower].append(idx)

        # LRU 缓存: msg -> translated_msg
        self._cache_size = cache_size
        self._cache: OrderedDict[str, str] = OrderedDict() if cache_size > 0 else None

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()

        if self._stats:
            self._stats.record_total()

        # 1. 查缓存
        if self._cache is not None and msg in self._cache:
            translated = self._cache[msg]
            self._cache.move_to_end(msg)
            if self._stats:
                self._stats.record_cache_hit()
            self._apply_translation(record, msg, translated)
            return True

        # 2. 关键词预过滤 + 正则匹配
        msg_lower = msg.lower()
        candidate_indices = self._get_candidate_indices(msg_lower)

        translated = None
        for idx in candidate_indices:
            compiled, replacement, rule_idx = self._compiled_rules[idx]
            m = compiled.search(msg)
            if m:
                translated = m.expand(replacement)
                if self._stats:
                    self._stats.record_hit(rule_idx)
                break

        # 3. 兜底: 尝试未被关键词索引覆盖的规则
        if translated is None:
            for compiled, replacement, rule_idx in self._compiled_rules:
                if rule_idx not in candidate_indices:
                    m = compiled.search(msg)
                    if m:
                        translated = m.expand(replacement)
                        if self._stats:
                            self._stats.record_hit(rule_idx)
                        break

        # 4. 应用翻译
        if translated:
            # 存缓存
            if self._cache is not None:
                self._cache[msg] = translated
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            self._apply_translation(record, msg, translated)
        else:
            if self._stats:
                self._stats.record_miss()

        return True

    def _get_candidate_indices(self, msg_lower: str) -> set:
        """从消息中提取关键词，返回候选规则索引集合"""
        candidates = set()
        # 按空格分割取前几个词作为关键词
        words = msg_lower.split()[:5]
        for word in words:
            # 清理标点
            clean = word.strip("':,\"()[]{}|/\\")
            if clean in self._keyword_index:
                for idx in self._keyword_index[clean]:
                    candidates.add(idx)
        return candidates

    def _apply_translation(self, record: logging.LogRecord, original: str, translated: str):
        """应用翻译到日志记录"""
        if self.mode == "replace":
            record.msg = translated
            record.args = None
        elif self.mode == "append":
            record.msg = f"{original}\n  └─ {translated}"
            record.args = None

    def add_rule(self, pattern: str, replacement: str, keywords: Optional[List[str]] = None):
        """动态添加翻译规则"""
        compiled = re.compile(pattern)
        idx = len(self._compiled_rules)
        self._compiled_rules.append((compiled, replacement, idx))

        if keywords is None:
            # 自动提取关键词: 取正则中的字面量单词
            keywords = re.findall(r'[a-zA-Z]+', pattern)[:3]

        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in self._keyword_index:
                self._keyword_index[kw_lower] = []
            self._keyword_index[kw_lower].append(idx)

    def get_stats(self) -> Optional[Dict]:
        """获取翻译统计"""
        return self._stats.get_summary() if self._stats else None

    def reset_stats(self):
        """重置统计"""
        if self._stats:
            self._stats.reset()

    def get_cache_info(self) -> Dict:
        """获取缓存信息"""
        if self._cache is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "size": len(self._cache),
            "max_size": self._cache_size,
        }


# ── 单例管理 ──────────────────────────────────────────────────────────────────

_translator: Optional[LogTranslator] = None


def get_translator() -> LogTranslator:
    """获取翻译器单例"""
    global _translator
    if _translator is None:
        _translator = LogTranslator()
    return _translator


def install_translator(mode: str = "replace", cache_size: int = 512,
                       enable_stats: bool = True) -> LogTranslator:
    """安装翻译器到根 logger"""
    global _translator
    uninstall_translator()
    _translator = LogTranslator(mode=mode, cache_size=cache_size,
                                enable_stats=enable_stats)
    logging.getLogger().addFilter(_translator)
    return _translator


def uninstall_translator():
    """卸载翻译器"""
    global _translator
    if _translator is not None:
        logging.getLogger().removeFilter(_translator)
        _translator = None


# ── Hermes 插件接口 ──────────────────────────────────────────────────────────

def register(ctx=None):
    """Hermes 插件注册入口"""
    try:
        return install_translator()
    except Exception as e:
        logging.getLogger(__name__).exception("log-translator: register 失败: %s", e)
        return None


def on_config_update(config: dict):
    """配置更新回调"""
    try:
        plugin_config = config.get("plugins", {}).get("log-translator", {})
        mode = plugin_config.get("mode", "replace")
        cache_size = plugin_config.get("cache_size", 512)
        enable_stats = plugin_config.get("enable_stats", True)

        global _translator
        if _translator:
            _translator.mode = mode
            old_cache_size = _translator._cache_size
            _translator._cache_size = cache_size
            _translator._enable_stats = enable_stats

            # 缓存大小变化时重建缓存，裁剪超出的条目
            if cache_size != old_cache_size and _translator._cache is not None:
                if cache_size <= 0:
                    # 禁用缓存
                    _translator._cache = None
                else:
                    # 裁剪超出新容量的条目（LRU 淘汰最旧的）
                    while len(_translator._cache) > cache_size:
                        _translator._cache.popitem(last=False)
    except Exception as e:
        logging.getLogger(__name__).exception("log-translator: on_config_update 执行失败: %s", e)


def on_post_api_request(**kwargs):
    """API 请求后回调 — 报告翻译统计"""
    try:
        translator = get_translator()
        stats = translator.get_stats()
        if stats and stats["total_messages"] > 0:
            logging.getLogger("log-translator").info(
                f"翻译统计: 总消息={stats['total_messages']}, "
                f"已翻译={stats['translated']}, "
                f"缓存命中={stats['cache_hits']}, "
                f"命中率={stats['hit_rate_pct']}%"
            )
    except Exception as e:
        logging.getLogger(__name__).exception("log-translator: on_post_api_request 执行失败: %s", e)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hermes 日志翻译器")
    parser.add_argument("--test", action="store_true", help="测试翻译规则")
    parser.add_argument("--list", action="store_true", help="列出所有翻译规则")
    parser.add_argument("--stats", action="store_true", help="显示翻译统计")
    parser.add_argument("--install", action="store_true", help="安装翻译器")
    parser.add_argument("--mode", choices=["replace", "append"], default="replace")
    parser.add_argument("--cache-size", type=int, default=512, help="LRU 缓存大小")
    args = parser.parse_args()

    translator = LogTranslator(mode=args.mode, cache_size=args.cache_size)

    if args.list:
        print(f"翻译规则列表 ({len(TRANSLATION_RULES)} 条):\n")
        for i, (pattern, replacement, keywords) in enumerate(TRANSLATION_RULES):
            kw_str = ", ".join(keywords)
            print(f"  [{i:2d}] 关键词=[{kw_str}]")
            print(f"       模式: {pattern[:60]}{'...' if len(pattern) > 60 else ''}")
            print(f"       翻译: {replacement[:60]}{'...' if len(replacement) > 60 else ''}")
            print()

        # 关键词索引统计
        print(f"关键词索引: {len(translator._keyword_index)} 个关键词")
        top_kw = sorted(translator._keyword_index.items(),
                        key=lambda x: len(x[1]), reverse=True)[:10]
        for kw, indices in top_kw:
            print(f"  '{kw}' → {len(indices)} 条规则")
        return

    if args.stats:
        install_translator(mode=args.mode, cache_size=args.cache_size)
        translator = get_translator()

        # 模拟一些翻译
        test_messages = [
            "tool terminal completed (2.5s, 1234 chars)",
            "tool terminal completed (2.5s, 1234 chars)",  # 缓存命中
            "tool read_file failed (1.0s): File not found",
            "Error: Connection timeout",
            "Model Router: strategy=auto",
            "unknown message that won't match anything",
        ]

        for msg in test_messages:
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg=msg, args=None, exc_info=None
            )
            translator.filter(record)

        stats = translator.get_stats()
        cache = translator.get_cache_info()

        print("翻译统计:")
        for k, v in stats.items():
            if k == "top_rules":
                print(f"  top_rules:")
                for idx, count in v:
                    print(f"    规则[{idx}]: {count} 次命中")
            else:
                print(f"  {k}: {v}")

        print(f"\n缓存信息:")
        for k, v in cache.items():
            print(f"  {k}: {v}")
        return

    if args.test:
        test_messages = [
            "conversation turn: session=abc123 model=gpt-4 provider=openai platform=cli history=5 msg='hello'",
            "tool terminal completed (2.5s, 1234 chars)",
            "tool read_file failed (1.0s): File not found",
            "credential pool: no available entries (all exhausted or empty)",
            "Model Router: [abc12345] gpt-4/openai → mimo-v2.5-pro/mimo | 用户指定使用 MiMo",
            "Plugin my-plugin loaded",
            "Error: Connection timeout",
        ]

        for msg in test_messages:
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg=msg, args=None, exc_info=None
            )
            translator.filter(record)
            print(f"原始: {msg}")
            print(f"翻译: {record.getMessage()}")
            print()
        return

    if args.install:
        translator = install_translator(mode=args.mode, cache_size=args.cache_size)
        print(f"✅ 日志翻译器已安装 (模式: {args.mode}, 缓存: {args.cache_size})")
        print(f"   已加载 {len(translator._compiled_rules)} 条翻译规则")
        print(f"   关键词索引: {len(translator._keyword_index)} 个关键词")
        return

    # 默认：安装翻译器
    translator = register()
    print(f"✅ 日志翻译器插件已注册")
    print(f"   已加载 {len(translator._compiled_rules)} 条翻译规则")
    print(f"   关键词索引: {len(translator._keyword_index)} 个关键词")


if __name__ == "__main__":
    main()
