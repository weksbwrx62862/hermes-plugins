"""
Model Router Plugin v2.1 — 多模型智能路由

核心能力：
  - 时间感知路由：非高峰期（00:00-08:00）优先 MiMo（0.8x 系数），高峰期优先 DeepSeek
  - 多密钥轮转：MiMo 3 账号自动轮转均衡负载
  - Provider 令牌桶限流：单 provider 请求限制，超限自动降级到备选模型
  - 智能评分：任务复杂度 / 成本 / 速度 / 质量四维评分
  - 自动切换：monkey-patch _build_api_kwargs，在 API 调用前无缝切换 provider + client

架构：
  pre_llm_call hook → 计算路由决策 → 存入路由表（session_id 索引）
  → agent 调用 _build_api_kwargs（已被 monkey-patch）→ 取出路由决策 → 切换 provider / client
  → agent 使用新 provider 发起 API 调用

多 key 轮转策略：
  从 .env 读取 MIMO_API_KEY / MIMO_API_KEY_2 / MIMO_API_KEY_3，记录每个 key 的使用次数，
  每次选择使用次数最少的 key，使用后递增计数。

配置 (~/.hermes/config.yaml)：
  providers:
    mimo:   { api_key, base_url, models: [...] }
    openai: { api_key, base_url, models: [...] }
  plugins.model-router.strategy: auto  # cheapest|fastest|smartest|auto

环境变量 (~/.hermes/.env)：
  MIMO_API_KEY        MiMo 账号 1（主）
  MIMO_API_KEY_2      MiMo 账号 2（轮转）
  MIMO_API_KEY_3      MiMo 账号 3（轮转）
"""

import os
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

BEIJING_TZ = timezone(timedelta(hours=8))
_MIMO_KEY_ENV_VARS = ["MIMO_API_KEY", "MIMO_API_KEY_2", "MIMO_API_KEY_3"]
_NVIDIA_KEY_ENV_VARS = ["NVIDIA_NIM_API_KEY", "NVIDIA_NIM_API_KEY_2", "NVIDIA_NIM_API_KEY_3", "NVIDIA_NIM_API_KEY_4", "NVIDIA_NIM_API_KEY_5"]

_ENV_KEYS_CACHE: Optional[List[str]] = None
_ENV_KEYS_CACHE_TIME: float = 0.0
_ENV_KEYS_CACHE_TTL: float = 30.0

_POOL_CACHE: Optional[List[Dict]] = None
_POOL_CACHE_TIME: float = 0.0
_POOL_CACHE_TTL: float = 10.0

_CONFIG_CACHE: Optional[Dict] = None
_CONFIG_CACHE_TIME: float = 0.0
_CONFIG_CACHE_TTL: float = 30.0

_KEY_USAGE: Dict[str, int] = {}
_KEY_USAGE_LOCK = threading.Lock()

# ── Round-Robin 轮转指针 ──
_KEY_ROUND_ROBIN: Dict[str, int] = {}  # provider → 当前索引
_KEY_ROUND_ROBIN_LOCK = threading.Lock()

# ── Provider 限流冷却黑名单 ──
_PROVIDER_COOLDOWN: Dict[str, float] = {}  # provider → 冷却到期的 timestamp
_PROVIDER_COOLDOWN_LOCK = threading.Lock()
_PROVIDER_COOLDOWN_DURATION: float = 10.0  # 被限流后冷却 10 秒

_KEY_HEALTH: Dict[str, Dict] = {}
_KEY_HEALTH_LOCK = threading.Lock()
_KEY_FAIL_THRESHOLD: float = 0.5
_KEY_COOLDOWN: float = 300.0

# ── Provider 级别令牌桶限流 ──
# 格式: {provider_name: {"tokens": float, "last_refill": float, "rate": float, "capacity": int}}
_RATE_LIMIT_BUCKETS: Dict[str, Dict] = {}
_RATE_LIMIT_LOCK = threading.Lock()

# 默认限流配置（可被 config.yaml 覆盖）
_DEFAULT_RATE_LIMITS = {
    "nvidia-nim": {"rate": 40.0 / 60.0, "capacity": 40},  # 40次/分钟 → 0.667 tokens/s
}

def _load_rate_limits() -> Dict[str, Dict]:
    """从 config 读取 provider 级别限流配置，合并默认值"""
    cfg = _load_config()
    mr_cfg = cfg.get("plugins", {}).get("model-router", {})
    configured = mr_cfg.get("rate_limits", {})
    merged = dict(_DEFAULT_RATE_LIMITS)
    for prov, limits in configured.items():
        if isinstance(limits, dict):
            merged[prov] = {
                "rate": limits.get("rate", _DEFAULT_RATE_LIMITS.get(prov, {}).get("rate", 1.0)),
                "capacity": limits.get("capacity", _DEFAULT_RATE_LIMITS.get(prov, {}).get("capacity", 60)),
            }
    return merged


def _check_rate_limit(provider: str) -> bool:
    """令牌桶算法：检查 provider 是否有可用令牌。
    有令牌则消耗 1 个并返回 True，否则返回 False。
    """
    limits = _load_rate_limits()
    limit = limits.get(provider)
    if not limit:
        return True  # 无限流配置，放行

    with _RATE_LIMIT_LOCK:
        now = time.monotonic()
        if provider not in _RATE_LIMIT_BUCKETS:
            _RATE_LIMIT_BUCKETS[provider] = {
                "tokens": float(limit["capacity"]),
                "last_refill": now,
                "rate": limit["rate"],
                "capacity": limit["capacity"],
            }
        bucket = _RATE_LIMIT_BUCKETS[provider]

        # 补齐令牌
        elapsed = now - bucket["last_refill"]
        refill = elapsed * bucket["rate"]
        bucket["tokens"] = min(float(bucket["capacity"]), bucket["tokens"] + refill)
        bucket["last_refill"] = now

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        else:
            return False


def get_rate_limit_status(provider: str = None) -> Dict:
    """查询限流状态"""
    limits = _load_rate_limits()
    with _RATE_LIMIT_LOCK:
        # 先补齐所有桶
        now = time.monotonic()
        for pname, bucket in list(_RATE_LIMIT_BUCKETS.items()):
            elapsed = now - bucket["last_refill"]
            refill = elapsed * bucket["rate"]
            bucket["tokens"] = min(float(bucket["capacity"]), bucket["tokens"] + refill)
            bucket["last_refill"] = now

        if provider:
            bucket = _RATE_LIMIT_BUCKETS.get(provider)
            if not bucket:
                return {"provider": provider, "limited": False, "reason": "无限制"}
            return {
                "provider": provider,
                "limited": bucket["tokens"] < 1.0,
                "available_tokens": round(bucket["tokens"], 1),
                "capacity": bucket["capacity"],
                "rate_per_sec": bucket["rate"],
                "rate_per_min": round(bucket["rate"] * 60, 1),
            }

        result = {}
        for pname, bucket in _RATE_LIMIT_BUCKETS.items():
            result[pname] = {
                "limited": bucket["tokens"] < 1.0,
                "available_tokens": round(bucket["tokens"], 1),
                "capacity": bucket["capacity"],
                "rate_per_sec": bucket["rate"],
                "rate_per_min": round(bucket["rate"] * 60, 1),
            }
        return result

def _get_provider_cooldown_duration() -> float:
    cfg = _load_config()
    return cfg.get("plugins", {}).get("model-router", {}).get("provider_cooldown_seconds", _PROVIDER_COOLDOWN_DURATION)


def _add_provider_cooldown(provider: str) -> None:
    """将被限流的 provider 加入冷却黑名单"""
    duration = _get_provider_cooldown_duration()
    with _PROVIDER_COOLDOWN_LOCK:
        _PROVIDER_COOLDOWN[provider] = time.time() + duration
        logger.info("Model Router: %s 加入冷却黑名单 (%.1fs)", provider, duration)


def _is_provider_in_cooldown(provider: str) -> bool:
    """检查 provider 是否在冷却中"""
    with _PROVIDER_COOLDOWN_LOCK:
        if provider not in _PROVIDER_COOLDOWN:
            return False
        if time.time() > _PROVIDER_COOLDOWN[provider]:
            del _PROVIDER_COOLDOWN[provider]
            return False
        return True

_plugin_ctx: Optional[Any] = None
_current_strategy: str = "auto"
_active_model: Optional[Dict] = None
_state_lock = threading.Lock()


def _get_active_model() -> Optional[Dict]:
    with _state_lock:
        return _active_model

def _set_active_model(model: Optional[Dict]) -> None:
    global _active_model
    with _state_lock:
        _active_model = model

def _get_current_strategy() -> str:
    with _state_lock:
        return _current_strategy

def _set_current_strategy(strategy: str) -> None:
    global _current_strategy
    with _state_lock:
        _current_strategy = strategy

_routing_decisions: Dict[str, Dict] = {}
_routing_lock = threading.Lock()
_routing_cleanup_time: float = 0.0
_ROUTING_CLEANUP_INTERVAL: float = 300.0
_ROUTING_TTL: float = 600.0

_DEFAULT_SCORING: Dict[str, Any] = {
    "off_peak_mimo_bonus": 25,
    "off_peak_mimo_pro_bonus": 10,
    "off_peak_non_mimo_penalty": -15,
    "peak_deepseek_bonus": 20,
    "peak_mimo_penalty": -15,
    "nvidia_nim_priority_bonus": 50,  # NVIDIA NIM 优先加成（仅当 MiMo 不可用时生效）
    "force_bonus": 80,
    "force_penalty": -80,
    "strategy_multiplier": 6,
    "feedback_multiplier": 10,
    "auto_speed_weight": 2,
    # 任务类型路由加成
    "task_code_bonus": 10,       # 代码任务 → 优先 coder/flash/fast 模型
    "task_knowledge_bonus": 5,   # 知识任务 → 优先 pro 模型
    "task_chat_bonus": 5,        # 聊天任务 → 优先 flash/fast 模型
    "long_context_bonus": 8,     # 长文档任务 → 优先大上下文窗口模型
}

# ── AMA 联动：任务权重 → 策略覆盖 ──
_ama_task_weights: Dict[str, float] = {}  # session_id → complexity_score
_ama_task_lock = threading.Lock()

# ── AMA 执行反馈：模型质量回流 ──
_model_feedback: Dict[str, Dict] = {}  # model_name → {successes, failures, total_tokens, recent_penalty}
_FEEDBACK_TTL: float = 86400.0  # 24 小时衰减
_feedback_lock = threading.Lock()


def record_model_feedback(model_name: str, success: bool, token_usage: int = 0) -> None:
    """AMA 执行后调用：记录模型成功/失败，影响后续评分"""
    import time as _time
    with _feedback_lock:
        if model_name not in _model_feedback:
            _model_feedback[model_name] = {
                "successes": 0, "failures": 0, "total_tokens": 0,
                "last_time": 0, "penalty": 0.0,
            }
        fb = _model_feedback[model_name]
        # 衰减旧 penalty
        elapsed = _time.time() - fb["last_time"] if fb["last_time"] else _FEEDBACK_TTL
        if elapsed > 0:
            decay = max(0, 1 - elapsed / _FEEDBACK_TTL)
            fb["penalty"] *= decay

        if success:
            fb["successes"] += 1
            fb["penalty"] = max(0, fb["penalty"] - 0.05)  # 成功减 penalty
        else:
            fb["failures"] += 1
            fb["penalty"] += 0.3  # 失败加 penalty
        fb["total_tokens"] += token_usage
        fb["last_time"] = _time.time()


def get_model_feedback_score(model_name: str) -> float:
    """返回模型反馈调整分（0~0.5），用于 _score 加减"""
    with _feedback_lock:
        fb = _model_feedback.get(model_name)
        if not fb:
            return 0.0
        total = fb["successes"] + fb["failures"]
        if total < 2:
            return 0.0  # 样本太少不调整
        return -fb["penalty"]  # penalty 越大，返回越负


def get_active_model_quality() -> int:
    """返回当前活跃模型的 quality 分（1-5），供 AMA 调整选型"""
    if _get_active_model():
        _, _, quality = _infer_attrs(_get_active_model().get("name", ""), _get_active_model().get("provider", ""))
        return quality
    return 3  # 默认中等


def set_task_weight(session_id: str, score: float) -> str:
    """AMA 调用：下发任务权重，由 Router 据此选策略
    返回推荐策略名（cheapest/auto/smartest）"""
    global _ama_task_weights
    with _ama_task_lock:
        _ama_task_weights[session_id] = score
    # 按复杂度返回推荐策略
    if score <= 3:
        return "cheapest"
    elif score >= 7:
        return "smartest"
    return "auto"

_original_build_api_kwargs = None
_patch_applied = False
_patch_lock = threading.Lock()
_patch_retry_count: int = 0
_PATCH_MAX_RETRIES: int = 10


# ═══════════════════════════════════════════════════════════════════
# 时间工具
# ═══════════════════════════════════════════════════════════════════

def _beijing_now() -> datetime:
    return datetime.now(BEIJING_TZ)


def _is_off_peak() -> bool:
    cfg = _load_config()
    peak_cfg = cfg.get("plugins", {}).get("model-router", {}).get("peak_hours", {})
    off_peak_start = peak_cfg.get("off_peak_start", 0)   # 非高峰开始（含）
    off_peak_end = peak_cfg.get("off_peak_end", 8)        # 非高峰结束（不含）
    hour = _beijing_now().hour
    if off_peak_start <= off_peak_end:
        return off_peak_start <= hour < off_peak_end
    else:  # 跨午夜，如 22~6
        return hour >= off_peak_start or hour < off_peak_end


# ═══════════════════════════════════════════════════════════════════
# 配置读取
# ═══════════════════════════════════════════════════════════════════

def _load_config() -> Dict:
    global _CONFIG_CACHE, _CONFIG_CACHE_TIME
    now = time.time()
    if _CONFIG_CACHE is not None and (now - _CONFIG_CACHE_TIME) < _CONFIG_CACHE_TTL:
        return _CONFIG_CACHE
    try:
        import yaml
    except ImportError:
        logger.warning("Model Router: PyYAML not installed, cannot read config.yaml")
        return {}
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        _CONFIG_CACHE = cfg
        _CONFIG_CACHE_TIME = now
        return cfg
    except Exception as exc:
        logger.warning("Model Router: failed to read config.yaml: %s", exc)
        return {}


def invalidate_config_cache() -> None:
    global _CONFIG_CACHE, _CONFIG_CACHE_TIME
    _CONFIG_CACHE = None
    _CONFIG_CACHE_TIME = 0.0


def _get_strategy() -> str:
    cfg = _load_config()
    strategy = cfg.get("plugins", {}).get("model-router", {}).get("strategy", "auto")
    if strategy in ("cheapest", "fastest", "smartest", "auto"):
        return strategy
    return "auto"


def _load_scoring_config() -> Dict[str, Any]:
    cfg = _load_config()
    mr_cfg = cfg.get("plugins", {}).get("model-router", {})
    scoring = mr_cfg.get("scoring", {})
    merged = dict(_DEFAULT_SCORING)
    merged.update({k: v for k, v in scoring.items() if k in _DEFAULT_SCORING})
    return merged


# ═══════════════════════════════════════════════════════════════════
# 多 key 轮转（MiMo）
# ═══════════════════════════════════════════════════════════════════

def _load_mimo_keys() -> List[str]:
    global _ENV_KEYS_CACHE, _ENV_KEYS_CACHE_TIME
    now = time.time()
    if _ENV_KEYS_CACHE is not None and (now - _ENV_KEYS_CACHE_TIME) < _ENV_KEYS_CACHE_TTL:
        return _ENV_KEYS_CACHE

    keys = []
    for var in _MIMO_KEY_ENV_VARS:
        val = os.environ.get(var, "").strip()
        if val and val not in keys:
            keys.append(val)

    _ENV_KEYS_CACHE = keys
    _ENV_KEYS_CACHE_TIME = now
    if len(keys) > 1:
        logger.info("Model Router: loaded %d MiMo keys for rotation", len(keys))
    return keys


_NVIDIA_KEYS_CACHE: Optional[List[str]] = None
_NVIDIA_KEYS_CACHE_TIME: float = 0.0
_NVIDIA_KEYS_CACHE_TTL: float = 30.0


def _load_nvidia_keys() -> List[str]:
    """从环境变量加载 NVIDIA NIM 多个 API Key"""
    global _NVIDIA_KEYS_CACHE, _NVIDIA_KEYS_CACHE_TIME
    now = time.time()
    if _NVIDIA_KEYS_CACHE is not None and (now - _NVIDIA_KEYS_CACHE_TIME) < _NVIDIA_KEYS_CACHE_TTL:
        return _NVIDIA_KEYS_CACHE

    keys = []
    for var in _NVIDIA_KEY_ENV_VARS:
        val = os.environ.get(var, "").strip()
        if val and val not in keys:
            keys.append(val)
    config_key = _load_config().get("providers", {}).get("nvidia-nim", {}).get("api_key", "")
    if config_key and config_key not in keys:
        keys.append(config_key)

    _NVIDIA_KEYS_CACHE = keys
    _NVIDIA_KEYS_CACHE_TIME = now
    if len(keys) > 1:
        logger.info("Model Router: loaded %d NVIDIA NIM keys for rotation", len(keys))
    return keys


def _pick_key_for_provider(provider: str, config_provider_name: str, env_vars: List[str]) -> str:
    """通用多 key 轮转：Round-Robin 平均分流 + 健康检测跳过不健康 key"""
    keys = []
    for var in env_vars:
        val = os.environ.get(var, "").strip()
        if val and val not in keys:
            keys.append(val)
    if not keys:
        return _load_config().get("providers", {}).get(config_provider_name, {}).get("api_key", "")
    if len(keys) == 1:
        return keys[0]

    with _KEY_ROUND_ROBIN_LOCK:
        if provider not in _KEY_ROUND_ROBIN:
            _KEY_ROUND_ROBIN[provider] = 0

        # 从上次位置开始，找第一个健康的 key
        n = len(keys)
        for offset in range(n):
            idx = (_KEY_ROUND_ROBIN[provider] + offset) % n
            key = keys[idx]
            if _is_key_healthy(key):
                _KEY_ROUND_ROBIN[provider] = (idx + 1) % n
                # 同时记录用量统计
                with _KEY_USAGE_LOCK:
                    if key not in _KEY_USAGE:
                        _KEY_USAGE[key] = 0
                    _KEY_USAGE[key] += 1
                return key

        # 所有 key 都不健康，强制用最少使用的
        logger.warning("Model Router: 所有 %s Key 均不健康，降级使用最少使用策略", provider)
        with _KEY_USAGE_LOCK:
            min_key = min(keys, key=lambda k: _KEY_USAGE.get(k, 0))
            _KEY_USAGE[min_key] = _KEY_USAGE.get(min_key, 0) + 1
            return min_key


def _pick_mimo_key() -> str:
    return _pick_key_for_provider("mimo", "mimo", _MIMO_KEY_ENV_VARS)


def _pick_nvidia_key() -> str:
    return _pick_key_for_provider("nvidia-nim", "nvidia-nim", _NVIDIA_KEY_ENV_VARS)


def _get_key_usage() -> Dict[str, int]:
    with _KEY_USAGE_LOCK:
        return dict(_KEY_USAGE)


def record_key_failure(key: str) -> None:
    with _KEY_HEALTH_LOCK:
        if key not in _KEY_HEALTH:
            _KEY_HEALTH[key] = {"successes": 0, "failures": 0, "last_fail_time": 0.0}
        _KEY_HEALTH[key]["failures"] += 1
        _KEY_HEALTH[key]["last_fail_time"] = time.time()


def record_key_success(key: str) -> None:
    with _KEY_HEALTH_LOCK:
        if key not in _KEY_HEALTH:
            _KEY_HEALTH[key] = {"successes": 0, "failures": 0, "last_fail_time": 0.0}
        _KEY_HEALTH[key]["successes"] += 1


def _is_key_healthy(key: str) -> bool:
    with _KEY_HEALTH_LOCK:
        health = _KEY_HEALTH.get(key)
        if not health:
            return True
        total = health["successes"] + health["failures"]
        if total < 2:
            return True
        fail_rate = health["failures"] / total
        if fail_rate > _KEY_FAIL_THRESHOLD:
            elapsed = time.time() - health.get("last_fail_time", 0)
            if elapsed < _KEY_COOLDOWN:
                return False
            health["successes"] = 0
            health["failures"] = 0
            return True
        return True


def _mask_key(key: str) -> str:
    if not key or len(key) <= 8:
        return "***"
    return key[:4] + "***" + key[-4:]


_COST_TABLE = {
    "deepseek-v4-pro": {"input": 2.70, "output": 10.80},
    "deepseek-v4-flash": {"input": 0.10, "output": 0.40},
    "mimo-v2.5-pro": {"input": 1.50, "output": 6.00},
    "mimo-v2.5": {"input": 0.80, "output": 3.20},
    "mimo-v2-pro": {"input": 0.80, "output": 3.20},
    "mimo-v2-omni": {"input": 1.00, "output": 4.00},
}


def _estimate_cost(model: str, provider: str, usage: Dict) -> float:
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    pricing = _COST_TABLE.get(model)
    if not pricing:
        return 0.0
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


# ═══════════════════════════════════════════════════════════════════
# 模型池
# ═══════════════════════════════════════════════════════════════════

def _is_tts(name: str) -> bool:
    return "tts" in name.lower()


def _infer_attrs(name: str, provider: str) -> tuple:
    """从 config 模型属性表查，查不到再用启发式推断。
    
    Returns: (cost, speed, quality, context_window)
    """
    # 优先查 config
    cfg = _load_config()
    mr_cfg = cfg.get("plugins", {}).get("model-router", {})
    model_attrs = mr_cfg.get("model_attrs", {})
    for pattern, attrs in model_attrs.items():
        if pattern.lower() in name.lower() and isinstance(attrs, dict):
            return (
                attrs.get("cost", 2),
                attrs.get("speed", 3),
                attrs.get("quality", 4),
                attrs.get("context_window", 128_000),
            )

    # fallback 启发式
    lower = name.lower()
    prov = provider.lower()

    if _is_tts(name):
        return (0, 3, 4, 128_000)

    if prov == "mimo":
        if "pro" in lower and ("2.5" in lower or "v2.5" in lower):
            return (3, 3, 5, 1_000_000)
        if "pro" in lower:
            return (3, 3, 5, 1_000_000)
        if "omni" in lower:
            return (3, 2, 5, 256_000)
        return (2, 3, 4, 1_000_000)

    # NVIDIA NIM 必须优先于通用 deepseek 匹配（否则 cost 计算错误）
    if prov == "nvidia-nim":
        if "deepseek" in lower and "pro" in lower:
            return (0, 2, 5, 1_000_000)   # V4 Pro
        if "deepseek" in lower and "flash" in lower:
            return (0, 5, 4, 1_000_000)   # V4 Flash
        if "gpt-oss" in lower:
            return (0, 3, 4, 128_000)     # GPT-OSS
        if "llama-4" in lower or "maverick" in lower:
            return (0, 5, 4, 1_000_000)   # Llama4
        if "glm" in lower:
            return (0, 3, 4, 128_000)     # GLM 5.1
        if "nemotron-3" in lower or "super-120" in lower:
            return (0, 2, 4, 128_000)     # Nemotron3
        if "qwen3.5" in lower or "397b" in lower:
            return (0, 2, 3, 131_072)     # Qwen 397B
        if "qwen3-coder" in lower or "coder-480" in lower:
            return (0, 3, 3, 262_144)     # Qwen Coder
        if "nemotron-super" in lower and "49b" in lower:
            return (0, 1, 1, 128_000)     # ⛔ Nemotron 49B
        return (0, 3, 4, 128_000)

    if "deepseek" in lower or prov in ("openai", "deepseek"):
        if "pro" in lower or "v4-pro" in lower:
            return (3, 2, 5, 1_000_000)
        if "flash" in lower or "v4-flash" in lower:
            return (1, 5, 3, 1_000_000)
        return (2, 3, 4, 128_000)

    return (2, 3, 3, 128_000)


# ═══════════════════════════════════════════════════════════════════
# 长上下文绕路策略
# ═══════════════════════════════════════════════════════════════════

def _estimate_tokens(messages: list) -> int:
    """估算消息列表的 token 数量（中文约 1.5 token/字，英文约 0.75 token/word）。
    
    只估算 user_message 部分（即当前请求的输入），不包括 system prompt。
    """
    total_chars = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total_chars += len(part["text"])
    # 粗略估算：中文约 1.5 token/字，英文约 0.75 token/word
    # 取折中 1.2 token/字
    return int(total_chars * 1.2)


def _estimate_user_message_tokens(user_message: str) -> int:
    """估算单条用户消息的 token 数量。"""
    if not user_message:
        return 0
    return int(len(user_message) * 1.2)


# 长上下文阈值（token 数）
_LONG_CONTEXT_THRESHOLD = 200_000
_VERY_LONG_CONTEXT_THRESHOLD = 800_000


def _find_best_long_context_model(pool: List[Dict], threshold: int) -> Optional[Dict]:
    """在模型池中找到最适合长上下文的模型。
    
    策略：优先选 context_window >= threshold 的模型，按 quality 降序排列。
    如果没有满足条件的，选 context_window 最大的。
    """
    if not pool:
        return None

    # 过滤出 context_window >= threshold 的模型
    capable = [m for m in pool if m.get("context_window", 128_000) >= threshold]
    if capable:
        # 按 quality 降序，quality 相同时按 cost 升序
        capable.sort(key=lambda m: (-m.get("quality", 3), m.get("cost", 3)))
        return capable[0]

    # 没有满足条件的，选 context_window 最大的
    pool.sort(key=lambda m: -m.get("context_window", 128_000))
    return pool[0]


def _build_pool() -> List[Dict]:
    global _POOL_CACHE, _POOL_CACHE_TIME
    now = time.time()
    if _POOL_CACHE is not None and (now - _POOL_CACHE_TIME) < _POOL_CACHE_TTL:
        return _POOL_CACHE

    cfg = _load_config()
    providers = cfg.get("providers", {})
    if not providers:
        _POOL_CACHE = _load_pool_from_env()
        _POOL_CACHE_TIME = now
        return _POOL_CACHE

    pool = []
    for pname, pcfg in providers.items():
        if not isinstance(pcfg, dict):
            continue
        pkey = pcfg.get("api_key", "")
        pbase = pcfg.get("base_url", "")
        pmodels = pcfg.get("models", [])
        if not pkey or not pbase:
            continue
        if not isinstance(pmodels, list):
            pmodels = [pmodels] if pmodels else []

        for m in pmodels:
            if _is_tts(m):
                continue
            name_short = m.replace("/", "-").replace(":", "-")
            cost, speed, quality, context_window = _infer_attrs(m, pname)
            pool.append({
                "name": name_short,
                "display_name": m,
                "provider": pname,
                "model": m,
                "base_url": pbase,
                "key": pkey,
                "cost": cost,
                "speed": speed,
                "quality": quality,
                "context_window": context_window,
            })

    if not pool:
        _POOL_CACHE = _load_pool_from_env()
        _POOL_CACHE_TIME = now
        return _POOL_CACHE

    _POOL_CACHE = pool
    _POOL_CACHE_TIME = now
    return _POOL_CACHE


def _load_pool_from_env() -> List[Dict]:
    raw = os.environ.get("MODEL_POOL", "")
    if not raw:
        return []
    try:
        pool = json.loads(raw)
    except json.JSONDecodeError:
        return []
    _fix_pool_providers(pool)
    return [m for m in pool if not _is_tts(m.get("name", ""))]


def _fix_pool_providers(pool: List[Dict]) -> None:
    try:
        cfg = _load_config()
        providers = cfg.get("providers", {})
        url_map = {}
        for pname, pcfg in providers.items():
            if isinstance(pcfg, dict) and pcfg.get("base_url"):
                url_map[pcfg["base_url"].rstrip("/")] = pname
        for entry in pool:
            url = entry.get("base_url", "").rstrip("/")
            if url in url_map and entry.get("provider") != url_map[url]:
                entry["provider"] = url_map[url]
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# 模型评分与路由
# ═══════════════════════════════════════════════════════════════════

def _score(model: Dict, strategy: str, complexity: int, off_peak: bool,
           force_mimo: bool = False, force_deepseek: bool = False,
           task_type: str = "chat") -> float:
    name = model.get("name", "")
    prov = model.get("provider", "")
    c = model.get("cost", 3)
    s = model.get("speed", 3)
    q = model.get("quality", 3)
    ctx = model.get("context_window", 128_000)

    sc = _load_scoring_config()

    feedback_adj = get_model_feedback_score(name)

    base = 0.0
    is_mimo = prov == "mimo" or "mimo" in name.lower()
    is_ds = prov in ("openai", "deepseek") or "deepseek" in name.lower()
    is_nvidia = prov == "nvidia-nim"

    # ── 任务类型路由加成 ──
    base += _task_type_bonus(name, task_type, sc)

    # ── 长文档任务：大上下文窗口加成 ──
    if task_type == "long_doc":
        if ctx >= 1_000_000:
            base += sc.get("long_context_bonus", 8)
        elif ctx >= 256_000:
            base += sc.get("long_context_bonus", 8) * 0.5

    if force_mimo:
        base += (sc["force_bonus"] if is_mimo else sc["force_penalty"])
    elif force_deepseek:
        base += (sc["force_bonus"] if is_ds else sc["force_penalty"])
    else:
        # ═══ MiMo 全天优先 ═══
        if is_mimo:
            base += sc["off_peak_mimo_bonus"]  # MiMo 全天 +25
            if "pro" in name.lower() and strategy != "cheapest":
                base += sc["off_peak_mimo_pro_bonus"]  # Pro 再 +10
        elif is_nvidia:
            # NVIDIA 第二优先 — 高峰+20, 非高峰不受惩罚
            if not off_peak:
                base += sc["peak_deepseek_bonus"]  # +20
        elif is_ds:
            if off_peak:
                base += sc["off_peak_non_mimo_penalty"]  # DeepSeek 非高峰 -15
            else:
                base += sc["peak_deepseek_bonus"]  # DeepSeek 高峰 +20
        else:
            if off_peak:
                base += sc["off_peak_non_mimo_penalty"]

    sm = sc["strategy_multiplier"]
    fm = sc["feedback_multiplier"]

    if strategy == "cheapest":
        return base + (5 - c) * sm + feedback_adj * fm
    elif strategy == "fastest":
        return base + s * sm + feedback_adj * fm
    elif strategy == "smartest":
        return base + q * sm + feedback_adj * fm
    elif strategy == "auto":
        wq = min(5, max(1, complexity))
        wc = max(1, 5 - wq)
        ws = sc["auto_speed_weight"]
        denom = wc + ws + wq
        return base + (wq * q + ws * s - wc * c) * (sm / denom) + feedback_adj * fm
    return base


def _estimate_complexity(query: str) -> int:
    score = 3.0
    high_kw = [
        "分析", "优化", "设计", "架构", "重构", "review", "refactor",
        "debug", "调试", "安全", "security", "性能", "performance",
        "多步骤", "复杂", "系统", "部署", "deploy", "explain", "实现",
        "诊断", "排查", "漏洞", "攻击", "渗透", "加密", "认证",
    ]
    low_kw = [
        "多少钱", "价格", "天气", "时间", "翻译", "translate",
        "什么是", "定义", "简单", "快捷", "hello", "hi", "你好",
        "echo", "重复", "ping",
    ]
    for kw in high_kw:
        if kw in query.lower():
            score += 0.3
    for kw in low_kw:
        if kw in query.lower():
            score -= 0.3
    return max(1, min(5, int(score)))


def _detect_task_type(query: str) -> str:
    """检测任务类型: classify / extract / simple_qa / long_doc / code / math / complex_reasoning / agent"""
    lower = query.lower()

    classify_kw = ["分类", "归类", "判断是否", "classify", "categorize", "是真是假"]
    extract_kw = ["提取", "抽取", "摘录", "extract", "摘要", "summarize", "总结"]
    simple_qa_kw = ["多少钱", "价格", "天气", "时间", "翻译", "什么是", "定义", "hello", "hi", "ping"]
    long_doc_kw = ["文档", "论文", "长文", "报告", "document", "paper", "report", "阅读理解"]
    code_kw = [
        "代码", "code", "编程", "函数", "class", "def", "import",
        "bug", "修复", "重构", "refactor", "debug", "测试", "test",
        "python", "java", "go", "rust", "js", "ts", "react", "vue",
        "api", "接口", "算法", "algorithm", "sql", "数据库",
        "编译", "部署", "deploy", "docker", "git",
    ]
    math_kw = ["计算", "数学", "方程", "公式", "calculate", "math", "equation", "证明", "积分", "微分"]
    complex_reasoning_kw = [
        "分析", "优化", "设计", "架构", "安全", "性能",
        "诊断", "排查", "漏洞", "explain", "analyze",
        "比较", "对比", "区别", "差异", "优缺点",
    ]
    agent_kw = ["帮我", "执行", "操作", "调用工具", "agent", "工具", "搜索", "联网"]

    scores = {
        "classify": sum(1 for kw in classify_kw if kw in lower),
        "extract": sum(1 for kw in extract_kw if kw in lower),
        "simple_qa": sum(1 for kw in simple_qa_kw if kw in lower),
        "long_doc": sum(1 for kw in long_doc_kw if kw in lower),
        "code": sum(1 for kw in code_kw if kw in lower),
        "math": sum(1 for kw in math_kw if kw in lower),
        "complex_reasoning": sum(1 for kw in complex_reasoning_kw if kw in lower),
        "agent": sum(1 for kw in agent_kw if kw in lower),
    }

    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return "simple_qa"
    return best_type


def _complexity_to_level(complexity: int) -> str:
    if complexity <= 2:
        return "light"
    elif complexity <= 3:
        return "medium"
    return "heavy"


def _task_type_bonus(name: str, task_type: str, sc: Dict) -> float:
    lower = name.lower()
    if task_type == "code":
        if "flash" in lower or "coder" in lower or "fast" in lower:
            return sc.get("task_code_bonus", 10)
        if "maverick" in lower:
            return sc.get("task_code_bonus", 10) * 0.5
    elif task_type in ("complex_reasoning", "long_doc"):
        if "pro" in lower and "flash" not in lower:
            return sc.get("task_knowledge_bonus", 5)
    elif task_type in ("simple_qa", "classify", "extract"):
        if "flash" in lower or "fast" in lower:
            return sc.get("task_chat_bonus", 5)
    elif task_type == "math":
        if "pro" in lower:
            return sc.get("task_knowledge_bonus", 5)
    elif task_type == "agent":
        if "pro" in lower or "omni" in lower:
            return sc.get("task_knowledge_bonus", 5)
    return 0.0


def _route(query: str, strategy: str,
           force_mimo: bool = False, force_deepseek: bool = False,
           estimated_tokens: int = 0) -> Optional[Dict]:
    pool = _build_pool()
    if not pool:
        return None

    off_peak = _is_off_peak()
    complexity = _estimate_complexity(query) if query else 3
    task_type = _detect_task_type(query) if query else "simple_qa"

    # ── 长上下文自动绕路 ──
    # 超过阈值时，直接选 context_window 最大的模型，跳过常规路由
    if estimated_tokens > _VERY_LONG_CONTEXT_THRESHOLD:
        best = _find_best_long_context_model(pool, _VERY_LONG_CONTEXT_THRESHOLD)
        if best:
            reasons = [f"超长上下文({estimated_tokens:,} tokens > {_VERY_LONG_CONTEXT_THRESHOLD:,})→直选大上下文模型"]
            _set_active_model(best)
            return {
                "name": best["name"], "provider": best["provider"],
                "model": best["model"], "base_url": best["base_url"],
                "key": best.get("key", ""), "key_masked": _mask_key(best.get("key", "")),
                "strategy": "long_context_bypass",
                "complexity": complexity, "task_type": task_type,
                "pool_size": len(pool),
                "alternatives": [m["name"] for m in pool if m["name"] != best["name"]][:3],
                "score_breakdown": {best["name"]: "long_context_bypass"},
                "fallback_chain": [],
                "time_info": {
                    "beijing_time": _beijing_now().strftime("%Y-%m-%d %H:%M:%S"),
                    "is_off_peak": off_peak,
                    "period": "非高峰期（0.8x 系数）" if off_peak else "高峰期",
                },
                "selection_reason": "；".join(reasons),
                "long_context_bypass": True,
                "estimated_tokens": estimated_tokens,
            }
    elif estimated_tokens > _LONG_CONTEXT_THRESHOLD:
        best = _find_best_long_context_model(pool, _LONG_CONTEXT_THRESHOLD)
        if best:
            reasons = [f"长上下文({estimated_tokens:,} tokens > {_LONG_CONTEXT_THRESHOLD:,})→优选大上下文模型"]
            _set_active_model(best)
            return {
                "name": best["name"], "provider": best["provider"],
                "model": best["model"], "base_url": best["base_url"],
                "key": best.get("key", ""), "key_masked": _mask_key(best.get("key", "")),
                "strategy": "long_context_prefer",
                "complexity": complexity, "task_type": task_type,
                "pool_size": len(pool),
                "alternatives": [m["name"] for m in pool if m["name"] != best["name"]][:3],
                "score_breakdown": {best["name"]: "long_context_prefer"},
                "fallback_chain": [],
                "time_info": {
                    "beijing_time": _beijing_now().strftime("%Y-%m-%d %H:%M:%S"),
                    "is_off_peak": off_peak,
                    "period": "非高峰期（0.8x 系数）" if off_peak else "高峰期",
                },
                "selection_reason": "；".join(reasons),
                "long_context_bypass": True,
                "estimated_tokens": estimated_tokens,
            }

    from .cost_monitor import get_routing_matrix
    _matrix_strategy = get_routing_matrix().get_model(task_type, _complexity_to_level(complexity))
    if _matrix_strategy == "balanced":
        _matrix_strategy = "auto"
    if not force_mimo and not force_deepseek and _matrix_strategy in ("cheapest", "fastest", "smartest", "auto"):
        strategy = _matrix_strategy

    scored = [(m, _score(m, strategy, complexity, off_peak, force_mimo, force_deepseek, task_type)) for m in pool]
    scored.sort(key=lambda x: -x[1])

    # ── 令牌桶限流检查 + 冷却黑名单 ──
    fallback_chain = []
    seen = set()
    best = None
    rate_limited_providers = []
    cooldown_providers = []
    for idx, (m, score_val) in enumerate(scored):
        if m["name"] in seen:
            continue
        seen.add(m["name"])
        prov = m.get("provider", "")
        # 检查冷却黑名单
        if _is_provider_in_cooldown(prov):
            cooldown_providers.append(prov)
            fallback_chain.append({
                "provider": m["provider"],
                "model": m["model"],
                "base_url": m["base_url"],
                "api_key": m.get("key", ""),
                "cooldown": True,
            })
            continue
        # 检查令牌桶限流
        if not _check_rate_limit(prov):
            rate_limited_providers.append(prov)
            _add_provider_cooldown(prov)  # 限流后加入冷却
            fallback_chain.append({
                "provider": m["provider"],
                "model": m["model"],
                "base_url": m["base_url"],
                "api_key": m.get("key", ""),
                "rate_limited": True,
            })
            continue
        best = m
        break

    if best is None:
        # 所有 provider 都被限流了，选评分最高的（保底策略）
        logger.warning("Model Router: 所有 provider 均被限流，使用评分最高模型作为保底")
        best = scored[0][0]
        if best.get("provider") == "mimo":
            best = dict(best)
            best["key"] = _pick_mimo_key()
        elif best.get("provider") == "nvidia-nim":
            best = dict(best)
            best["key"] = _pick_nvidia_key()

    if best.get("provider") == "mimo":
        best = dict(best)
        best["key"] = _pick_mimo_key()
    elif best.get("provider") == "nvidia-nim":
        best = dict(best)
        best["key"] = _pick_nvidia_key()

    _set_active_model(best)

    reasons = []
    if force_mimo:
        reasons.append("用户指定使用 MiMo")
    elif force_deepseek:
        reasons.append("用户指定使用 DeepSeek")
    elif best.get("provider") == "mimo":
        reasons.append("MiMo 全天优先使用")
    elif best.get("provider") == "nvidia-nim":
        reasons.append("NVIDIA NIM 第二优先（MiMo 不可用，免费 100 年配额）")

    if rate_limited_providers:
        reasons.append(f"以下 provider 被限流已跳过: {', '.join(rate_limited_providers)}")
    if cooldown_providers:
        reasons.append(f"以下 provider 冷却中: {', '.join(cooldown_providers)}")

    # 构建完整降级链（包含被限流已跳过的和未被选中的）
    for s in scored:
        m = s[0]
        if m["name"] in seen:
            continue
        seen.add(m["name"])
        if m.get("provider", "") in rate_limited_providers:
            continue  # 限流的已在前置循环中加入了
        fallback_chain.append({
            "provider": m["provider"],
            "model": m["model"],
            "base_url": m["base_url"],
            "api_key": m.get("key", ""),
        })

    _safe_fallback_chain = []
    for fb in fallback_chain:
        safe_fb = dict(fb)
        if "api_key" in safe_fb:
            safe_fb["api_key_masked"] = _mask_key(safe_fb.pop("api_key"))
        _safe_fallback_chain.append(safe_fb)

    return {
        "name": best["name"],
        "provider": best["provider"],
        "model": best["model"],
        "base_url": best["base_url"],
        "key": best["key"],
        "key_masked": _mask_key(best["key"]),
        "strategy": strategy,
        "complexity": complexity,
        "task_type": task_type,
        "pool_size": len(pool),
        "alternatives": [s[0]["name"] for s in scored[1:4]],
        "score_breakdown": {s[0]["name"]: round(s[1], 1) for s in scored[:5]},
        "fallback_chain": _safe_fallback_chain,
        "time_info": {
            "beijing_time": _beijing_now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_off_peak": off_peak,
            "period": "非高峰期（0.8x 系数）" if off_peak else "高峰期",
        },
        "selection_reason": "；".join(reasons),
    }


# ═══════════════════════════════════════════════════════════════════
# Agent 状态切换（通过 monkey-patch 调用）
# ═══════════════════════════════════════════════════════════════════

def _apply_routing(agent, routing: Dict) -> bool:
    try:
        new_model = routing.get("model", "")
        new_provider = routing.get("provider", "")
        new_base_url = routing.get("base_url", "")
        new_key = routing.get("key", "")

        if not new_model:
            return False

        api_mode = getattr(agent, "api_mode", "chat_completions")
        if api_mode not in ("chat_completions", "codex_responses"):
            return True

        old_model = getattr(agent, "model", "")
        old_provider = getattr(agent, "provider", "")
        old_key = getattr(agent, "api_key", "") or getattr(agent, "_api_key", "")

        model_changed = new_model != old_model
        provider_changed = new_provider != old_provider
        key_changed = new_key and new_key != old_key

        if not model_changed and not provider_changed and not key_changed:
            return True

        agent.model = new_model
        agent.provider = new_provider
        if hasattr(agent, "base_url"):
            agent.base_url = new_base_url

        if new_key:
            for attr in ("api_key", "_api_key"):
                if hasattr(agent, attr):
                    setattr(agent, attr, new_key)

        if provider_changed or key_changed:
            if hasattr(agent, "_transport_cache"):
                agent._transport_cache.clear()

            if hasattr(agent, "_client_kwargs"):
                old_ck = dict(getattr(agent, "_client_kwargs", {}))
                old_ck["api_key"] = new_key
                old_ck["base_url"] = new_base_url
                agent._client_kwargs = old_ck

            if hasattr(agent, "_create_openai_client") and callable(agent._create_openai_client):
                try:
                    agent.client = agent._create_openai_client(
                        dict(getattr(agent, "_client_kwargs", {})),
                        reason="model-router",
                        shared=True,
                    )
                    if new_key:
                        record_key_success(new_key)
                except Exception as exc:
                    logger.warning("Model Router: _create_openai_client failed: %s", exc)
                    if new_key:
                        record_key_failure(new_key)
                    try:
                        from openai import OpenAI
                        agent.client = OpenAI(api_key=new_key, base_url=new_base_url)
                    except Exception as exc2:
                        logger.warning("Model Router: OpenAI client fallback also failed: %s", exc2)
                        if new_key:
                            record_key_failure(new_key)
                        # ── 反馈联动：记录模型失败 ──
                        record_model_feedback(new_model, success=False)
                else:
                    # ── 反馈联动：记录模型成功 ──
                    record_model_feedback(new_model, success=True)

        reason = routing.get("selection_reason", "")
        logger.info(
            "Model Router: [%s] %s/%s → %s/%s | %s",
            getattr(agent, "session_id", "")[:8] if hasattr(agent, "session_id") else "?",
            old_model, old_provider, new_model, new_provider, reason,
        )

        fb_chain = routing.get("fallback_chain", [])
        if fb_chain:
            agent._fallback_chain = fb_chain
            agent._fallback_index = 0
            agent._fallback_activated = False
            logger.info(
                "Model Router: 已设置 %d 个降级备选模型 [%s]",
                len(fb_chain),
                ", ".join(fb["model"] for fb in fb_chain[:3]),
            )

        try:
            from .cost_monitor import get_cost_monitor, CallStats
            get_cost_monitor().record(CallStats(
                model=new_model,
                provider=new_provider,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=0.0,
                task_type=routing.get("task_type", "simple_qa"),
                complexity=str(routing.get("complexity", 3)),
                timestamp=time.time(),
            ))
        except Exception:
            pass

        return True
    except Exception as exc:
        logger.warning("Model Router: apply routing failed: %s", exc)
        return False


_FALLBACK_RETRYABLE_ERRORS = {429, 500, 502, 503, 504}


def _try_fallback(agent, exc: Exception) -> bool:
    chain = getattr(agent, "_fallback_chain", [])
    idx = getattr(agent, "_fallback_index", 0)

    if not chain or idx >= len(chain):
        return False

    status_code = getattr(exc, "status_code", None)
    if status_code and status_code not in _FALLBACK_RETRYABLE_ERRORS:
        return False

    for i in range(idx, len(chain)):
        fb = chain[i]
        if fb.get("cooldown") or fb.get("rate_limited"):
            continue

        logger.warning(
            "Model Router: 降级 %s → %s (原因: %s)",
            getattr(agent, "model", ""), fb["model"], str(exc)[:100],
        )

        agent.model = fb["model"]
        agent.provider = fb["provider"]
        if hasattr(agent, "base_url"):
            agent.base_url = fb["base_url"]

        fb_key = fb.get("api_key", "")
        if fb_key:
            for attr in ("api_key", "_api_key"):
                if hasattr(agent, attr):
                    setattr(agent, attr, fb_key)
            if hasattr(agent, "_client_kwargs"):
                ck = dict(getattr(agent, "_client_kwargs", {}))
                ck["api_key"] = fb_key
                ck["base_url"] = fb.get("base_url", "")
                agent._client_kwargs = ck
            if hasattr(agent, "_transport_cache"):
                agent._transport_cache.clear()
            if hasattr(agent, "_create_openai_client") and callable(agent._create_openai_client):
                try:
                    agent.client = agent._create_openai_client(
                        dict(getattr(agent, "_client_kwargs", {})),
                        reason="model-router-fallback",
                        shared=True,
                    )
                except Exception:
                    try:
                        from openai import OpenAI
                        agent.client = OpenAI(api_key=fb_key, base_url=fb.get("base_url", ""))
                    except Exception:
                        pass

        agent._fallback_index = i + 1
        agent._fallback_activated = True
        record_model_feedback(fb["model"], success=False)
        return True

    return False


# ═══════════════════════════════════════════════════════════════════
# Monkey-patch：拦截 _build_api_kwargs
# ═══════════════════════════════════════════════════════════════════

def _build_api_kwargs_patched(self, api_messages: list) -> dict:
    global _routing_decisions, _routing_lock, _original_build_api_kwargs
    session_id = getattr(self, "session_id", "") or ""
    with _routing_lock:
        entry = _routing_decisions.pop(session_id, None)
    routing = entry.get("decision") if isinstance(entry, dict) else entry
    if routing:
        _apply_routing(self, routing)
    try:
        return _original_build_api_kwargs(self, api_messages)
    except Exception as exc:
        if _try_fallback(self, exc):
            try:
                return _original_build_api_kwargs(self, api_messages)
            except Exception:
                raise
        raise


def _apply_monkey_patch():
    global _original_build_api_kwargs, _patch_applied, _patch_lock, _patch_retry_count
    if _patch_applied:
        return
    if _patch_retry_count >= _PATCH_MAX_RETRIES:
        return
    with _patch_lock:
        if _patch_applied:
            return
        if _patch_retry_count >= _PATCH_MAX_RETRIES:
            return
        try:
            import sys as _sys
            run_agent = _sys.modules.get("run_agent")
            if run_agent is None:
                try:
                    from run_agent import AIAgent as _AIAgent
                except ImportError:
                    _patch_retry_count += 1
                    if _patch_retry_count >= _PATCH_MAX_RETRIES:
                        logger.warning(
                            "Model Router: monkey-patch failed after %d attempts, giving up",
                            _PATCH_MAX_RETRIES,
                        )
                    else:
                        logger.debug(
                            "Model Router: run_agent not loaded yet, retry %d/%d",
                            _patch_retry_count, _PATCH_MAX_RETRIES,
                        )
                    return
            else:
                _AIAgent = run_agent.AIAgent
            if not hasattr(_AIAgent, "_build_api_kwargs"):
                _patch_retry_count += 1
                if _patch_retry_count >= _PATCH_MAX_RETRIES:
                    logger.warning(
                        "Model Router: AIAgent._build_api_kwargs not found after %d attempts, giving up",
                        _PATCH_MAX_RETRIES,
                    )
                else:
                    logger.debug(
                        "Model Router: AIAgent._build_api_kwargs not found, retry %d/%d",
                        _patch_retry_count, _PATCH_MAX_RETRIES,
                    )
                return
            _original_build_api_kwargs = _AIAgent._build_api_kwargs
            _AIAgent._build_api_kwargs = _build_api_kwargs_patched
            _patch_applied = True
            logger.info("Model Router: monkey-patch applied to AIAgent._build_api_kwargs")
        except Exception as exc:
            _patch_retry_count += 1
            logger.warning("Model Router: monkey-patch failed (attempt %d/%d): %s", _patch_retry_count, _PATCH_MAX_RETRIES, exc)


# ═══════════════════════════════════════════════════════════════════
# 路由表清理
# ═══════════════════════════════════════════════════════════════════

def _cleanup_stale_routing():
    global _routing_decisions, _routing_lock, _routing_cleanup_time
    now = time.time()
    if now - _routing_cleanup_time < _ROUTING_CLEANUP_INTERVAL:
        return
    _routing_cleanup_time = now
    with _routing_lock:
        expired = [
            sid for sid, entry in _routing_decisions.items()
            if isinstance(entry, dict) and (now - entry.get("_created_at", 0)) > _ROUTING_TTL
        ]
        for sid in expired:
            del _routing_decisions[sid]
        if expired:
            logger.debug("Model Router: cleaned %d expired routing entries", len(expired))


# ═══════════════════════════════════════════════════════════════════
# Hook 处理
# ═══════════════════════════════════════════════════════════════════

def handle_pre_llm_call(**kwargs) -> Optional[Dict]:
    global _routing_decisions, _routing_lock, _ama_task_weights, _ama_task_lock
    _apply_monkey_patch()
    _cleanup_stale_routing()

    session_id = kwargs.get("session_id", "")
    user_message = kwargs.get("user_message", "")
    messages = kwargs.get("messages", [])  # 完整消息列表（含历史）

    if not session_id or not user_message:
        return None

    _set_current_strategy(_get_strategy())

    # ── AMA 联动：检查任务权重，覆盖策略 ──
    effective_strategy = _get_current_strategy()
    ama_weight_reason = ""
    with _ama_task_lock:
        ama_score = _ama_task_weights.pop(session_id, None)
    if ama_score is not None:
        if ama_score <= 3:
            effective_strategy = "cheapest"
            ama_weight_reason = f" (AMA评分={ama_score:.1f}→cheapest)"
        elif ama_score >= 7:
            effective_strategy = "smartest"
            ama_weight_reason = f" (AMA评分={ama_score:.1f}→smartest)"

    # ── 估算 token 数量（用于长上下文绕路）──
    estimated_tokens = _estimate_tokens(messages) if messages else _estimate_user_message_tokens(user_message)

    result = _route(user_message, effective_strategy, estimated_tokens=estimated_tokens)
    if not result:
        return None

    with _routing_lock:
        _routing_decisions[session_id] = {"decision": result, "_created_at": time.time()}

    # 构建路由日志
    ctx_parts = [
        f"[Model Router] 策略={effective_strategy}{ama_weight_reason}",
        f"复杂度={result['complexity']}/5",
        f"选中: {result['name']} ({result['provider']})",
        f"原因: {result['selection_reason']}",
        f"时间段: {result['time_info']['period']}",
    ]
    if result.get("long_context_bypass"):
        ctx_parts.insert(2, f"长上下文绕路: {estimated_tokens:,} tokens")
    if result.get("alternatives"):
        ctx_parts.append(f"降级链: {result['name']} → {' → '.join(result.get('alternatives', []))}")

    return {"context": " | ".join(ctx_parts)}


def handle_post_llm_call(**kwargs) -> None:
    usage = kwargs.get("usage")
    if not usage:
        return
    try:
        from .cost_monitor import get_cost_monitor, CallStats
        model = kwargs.get("model", "")
        provider = kwargs.get("provider", "")
        latency = kwargs.get("latency_ms", 0)
        get_cost_monitor().record(CallStats(
            model=model,
            provider=provider,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cost_usd=_estimate_cost(model, provider, usage),
            latency_ms=latency,
            task_type=kwargs.get("task_type", "simple_qa"),
            complexity=str(kwargs.get("complexity", 3)),
            timestamp=time.time(),
        ))
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# Tool 处理
# ═══════════════════════════════════════════════════════════════════

def handle_model_route(args: Dict, **kwargs) -> str:
    query = args.get("query", "")
    strategy = args.get("strategy") or _get_current_strategy()
    force_mimo = args.get("force_mimo", False)
    force_deepseek = args.get("force_deepseek", False)

    result = _route(query, strategy, force_mimo=force_mimo, force_deepseek=force_deepseek)
    if not result:
        return json.dumps({"error": "模型池为空，请检查 config.yaml providers 配置"}, ensure_ascii=False)
    safe_result = {k: v for k, v in result.items() if k != "key"}
    return json.dumps(safe_result, ensure_ascii=False, indent=2)


def handle_model_pool(args: Dict, **kwargs) -> str:
    pool = _build_pool()
    now = _beijing_now()
    off_peak = _is_off_peak()
    key_usage = _get_key_usage()
    return json.dumps({
        "strategy": _get_current_strategy(),
        "active_model": m["name"] if (m := _get_active_model()) else "none",
        "time": {
            "beijing": now.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": now.hour,
            "is_off_peak": off_peak,
            "period": "非高峰期（0.8x 系数）" if off_peak else "高峰期",
        },
        "key_rotation": {
            "keys_loaded": len(_load_mimo_keys()),
            "usage": key_usage,
        } if key_usage else None,
        "pool": [
            {
                "name": m["name"], "provider": m["provider"],
                "cost": m.get("cost"), "speed": m.get("speed"),
                "quality": m.get("quality"),
                "context_window": m.get("context_window", 128_000),
            }
            for m in pool
        ],
        "pool_size": len(pool),
    }, ensure_ascii=False, indent=2)


def handle_model_balance(args: Dict, **kwargs) -> str:
    strategy = args.get("strategy", "auto")
    valid = ["cheapest", "fastest", "smartest", "auto"]
    if strategy not in valid:
        return json.dumps({"error": f"无效策略: {strategy}", "valid": valid}, ensure_ascii=False)
    _set_current_strategy(strategy)
    try:
        import subprocess
        subprocess.run(
            ["hermes", "config", "set", "plugins.model-router.strategy", strategy],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass
    return json.dumps({"strategy": strategy, "status": "active"}, ensure_ascii=False)


def handle_time_info(args: Dict, **kwargs) -> str:
    now = _beijing_now()
    off_peak = _is_off_peak()
    return json.dumps({
        "beijing_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": now.hour,
        "is_off_peak": off_peak,
        "period": "非高峰期（0.8x 系数）" if off_peak else "高峰期",
        "off_peak_range": "00:00-08:00 北京时间",
        "strategy": _get_current_strategy(),
        "rules": {
            "off_peak": "优先 MiMo 模型（0.8x 系数优惠）",
            "peak": "优先 DeepSeek 模型",
        },
    }, ensure_ascii=False, indent=2)


def handle_model_route_for_cron(args: Dict, **kwargs) -> str:
    query = args.get("query", "")
    strategy = args.get("strategy") or _get_current_strategy()
    if not query:
        return json.dumps({"error": "缺少 task_description 参数"}, ensure_ascii=False)
    result = _route(query, strategy)
    if not result or not result.get("model"):
        return json.dumps({"error": "模型路由失败"}, ensure_ascii=False)
    return json.dumps({
        "provider": result.get("provider"),
        "model": result.get("model"),
        "complexity": result.get("complexity"),
        "strategy": result.get("strategy"),
        "reason": result.get("selection_reason"),
        "time_info": result.get("time_info"),
        "alternatives": result.get("alternatives"),
    }, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════
# 对外 API（供外部脚本 / cron 调用）
# ═══════════════════════════════════════════════════════════════════

def route_for_cron(task_description: str, strategy: str = None) -> Dict:
    if not task_description or not task_description.strip():
        return {"provider": None, "model": None, "complexity": 0, "reason": "任务描述为空"}
    result = _route(task_description, strategy or _get_current_strategy())
    if not result or not result.get("model"):
        return {"provider": None, "model": None, "complexity": 0, "reason": "模型池未配置"}
    return {
        "provider": result.get("provider"),
        "model": result.get("model"),
        "complexity": result.get("complexity"),
        "strategy": result.get("strategy"),
        "reason": result.get("selection_reason"),
        "time_info": result.get("time_info"),
        "alternatives": result.get("alternatives"),
    }


# ═══════════════════════════════════════════════════════════════════
# 插件注册
# ═══════════════════════════════════════════════════════════════════

def register(ctx) -> None:
    global _plugin_ctx
    _plugin_ctx = ctx

    import sys as _sys
    _sys.modules.setdefault("plugins.model-router", _sys.modules[__name__])

    _set_current_strategy(_get_strategy())
    logger.info("Model Router: strategy=%s", _get_current_strategy())

    ctx.register_hook("pre_llm_call", handle_pre_llm_call)
    logger.info("Model Router: pre_llm_call hook registered (monkey-patch applies lazily)")

    ctx.register_hook("post_llm_call", handle_post_llm_call)

    ctx.register_tool(
        name="model_route",
        toolset="model-router",
        schema={
            "name": "model_route",
            "description": (
                "根据任务自动选择最佳大模型。支持时间感知路由："
                "非高峰期（00:00-08:00 北京时间）优先 MiMo（0.8x 系数优惠），高峰期优先 DeepSeek。"
                "支持强制指定：设置 force_mimo=true 或 force_deepseek=true"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "任务描述，用于评估复杂度并选择合适模型"},
                    "strategy": {"type": "string", "enum": ["cheapest", "fastest", "smartest", "auto"], "description": "路由策略"},
                    "force_mimo": {"type": "boolean", "description": "强制使用 MiMo 模型"},
                    "force_deepseek": {"type": "boolean", "description": "强制使用 DeepSeek 模型"},
                },
                "required": ["query"],
            },
        },
        handler=lambda args, **kw: handle_model_route(args, **kw),
        emoji="🔀",
    )

    ctx.register_tool(
        name="model_pool",
        toolset="model-router",
        schema={
            "name": "model_pool",
            "description": "查看当前模型池中所有可用模型、当前策略、关键使用统计及时间信息",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        handler=lambda args, **kw: handle_model_pool(args, **kw),
        emoji="🏊",
    )

    ctx.register_tool(
        name="model_balance",
        toolset="model-router",
        schema={
            "name": "model_balance",
            "description": "切换模型路由策略（cheapest=省钱 / fastest=极速 / smartest=最强 / auto=自适应）",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string", "enum": ["cheapest", "fastest", "smartest", "auto"], "description": "目标策略"},
                },
                "required": ["strategy"],
            },
        },
        handler=lambda args, **kw: handle_model_balance(args, **kw),
        emoji="⚖️",
    )

    ctx.register_tool(
        name="time_info",
        toolset="model-router",
        schema={
            "name": "time_info",
            "description": "查看当前北京时间、是否为非高峰期及路由策略说明",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        handler=lambda args, **kw: handle_time_info(args, **kw),
        emoji="🕐",
    )

    ctx.register_tool(
        name="model_route_for_cron",
        toolset="model-router",
        schema={
            "name": "model_route_for_cron",
            "description": "为定时任务评估复杂度并选择最优模型，返回 provider/model 配置",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "定时任务描述"},
                    "strategy": {"type": "string", "enum": ["cheapest", "fastest", "smartest", "auto"]},
                },
                "required": ["query"],
            },
        },
        handler=lambda args, **kw: handle_model_route_for_cron(args, **kw),
        emoji="⏰",
    )

    def handle_rate_limit_status(args: Dict, **kwargs) -> str:
        provider = args.get("provider") or None
        result = get_rate_limit_status(provider)
        return json.dumps(result, ensure_ascii=False, indent=2)

    ctx.register_tool(
        name="rate_limit_status",
        toolset="model-router",
        schema={
            "name": "rate_limit_status",
            "description": "查看 provider 级别的令牌桶限流状态。可选参数 provider 来查询特定 provider（如 'nvidia-nim'），不传则返回所有被限流 provider 的状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "description": "provider 名称，如 'nvidia-nim'。不传则返回所有。"},
                },
                "required": [],
            },
        },
        handler=lambda args, **kw: handle_rate_limit_status(args, **kw),
        emoji="🚦",
    )

    # ── 成本监控工具（v2.2 新增）──────────────────────────────
    from .cost_monitor import get_cost_monitor, print_cost_dashboard

    def handle_cost_dashboard(args: Dict, **kwargs) -> str:
        """成本仪表盘"""
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            print_cost_dashboard()
        return f.getvalue()

    ctx.register_tool(
        name="cost_dashboard",
        toolset="model-router",
        schema={
            "name": "cost_dashboard",
            "description": "查看模型调用成本监控仪表盘，包括各模型的调用次数、成本、延迟等统计",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        handler=lambda args, **kw: handle_cost_dashboard(args, **kw),
        emoji="💰",
    )

    pool = _build_pool()
    logger.info(
        "Model Router v2.2 registered: %d models, 7 tools, pre_llm_call hook, %d MiMo keys, rate limits=%s",
        len(pool), len(_load_mimo_keys()),
        list(_load_rate_limits().keys()),
    )
    logger.info(
        "当前北京时间: %s | %s",
        _beijing_now().strftime("%Y-%m-%d %H:%M:%S"),
        "非高峰期（MiMo 优先）" if _is_off_peak() else "高峰期（DeepSeek 优先）",
    )
