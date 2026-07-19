"""
Model Router v2.2 — 借鉴优化

新增能力：
  1. 成本监控 — 记录每次调用的 token 和成本
  2. 影子流量 — 1% 抽样对比质量
  3. 路由矩阵配置化 — JSON 热更新
  4. 任务分类器 — 用便宜模型做前置分类
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# 成本监控
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CallStats:
    """单次调用统计"""
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    task_type: str
    complexity: str
    timestamp: float


class CostMonitor:
    """成本监控器"""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file or str(Path.home() / ".hermes" / "model_router_costs.jsonl")
        self.stats: dict[str, dict] = defaultdict(lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "latencies": [],
        })
        self._lock = threading.Lock()
        self._load_history()
    
    def record(self, stats: CallStats):
        """记录一次调用"""
        with self._lock:
            key = f"{stats.provider}/{stats.model}"
            s = self.stats[key]
            s["calls"] += 1
            s["input_tokens"] += stats.input_tokens
            s["output_tokens"] += stats.output_tokens
            s["cost_usd"] += stats.cost_usd
            s["latencies"].append(stats.latency_ms)
            
            # 保持最近 1000 个延迟记录
            if len(s["latencies"]) > 1000:
                s["latencies"] = s["latencies"][-1000:]
            
            # 写日志
            self._write_log(stats)
    
    def get_summary(self) -> dict:
        """获取汇总统计"""
        with self._lock:
            summary = {}
            for key, s in self.stats.items():
                avg_latency = sum(s["latencies"]) / len(s["latencies"]) if s["latencies"] else 0
                summary[key] = {
                    "calls": s["calls"],
                    "input_tokens": s["input_tokens"],
                    "output_tokens": s["output_tokens"],
                    "cost_usd": round(s["cost_usd"], 4),
                    "avg_latency_ms": round(avg_latency, 1),
                }
            return summary
    
    def get_daily_cost(self) -> float:
        """获取今日成本"""
        today = time.strftime("%Y-%m-%d")
        total = 0.0
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("date") == today:
                            total += entry.get("cost_usd", 0)
                    except:
                        pass
        except FileNotFoundError:
            pass
        return round(total, 4)
    
    def _write_log(self, stats: CallStats):
        """写 JSONL 日志"""
        try:
            entry = {
                "date": time.strftime("%Y-%m-%d"),
                "time": time.strftime("%H:%M:%S"),
                "model": stats.model,
                "provider": stats.provider,
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "cost_usd": stats.cost_usd,
                "latency_ms": stats.latency_ms,
                "task_type": stats.task_type,
                "complexity": stats.complexity,
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Cost log write error: {e}")
    
    def _load_history(self):
        """加载历史统计"""
        try:
            if not os.path.exists(self.log_file):
                return
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        key = f"{entry.get('provider', '')}/{entry.get('model', '')}"
                        s = self.stats[key]
                        s["calls"] += 1
                        s["input_tokens"] += entry.get("input_tokens", 0)
                        s["output_tokens"] += entry.get("output_tokens", 0)
                        s["cost_usd"] += entry.get("cost_usd", 0)
                        latency = entry.get("latency_ms", 0)
                        if isinstance(latency, (int, float)) and latency > 0:
                            s["latencies"].append(latency)
                    except:
                        pass
            # 加载后按最近 1000 条截断，避免历史过长
            for s in self.stats.values():
                if len(s["latencies"]) > 1000:
                    s["latencies"] = s["latencies"][-1000:]
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# 影子流量
# ═══════════════════════════════════════════════════════════════════

class ShadowEvaluator:
    """影子流量评估器 — 1% 抽样对比质量"""
    
    def __init__(self, sample_rate: float = 0.01, log_file: str = None):
        self.sample_rate = sample_rate
        self.log_file = log_file or str(Path.home() / ".hermes" / "model_router_shadow.jsonl")
        self._lock = threading.Lock()
    
    def should_sample(self) -> bool:
        """是否应该抽样"""
        import random
        return random.random() < self.sample_rate

    def _evaluate_responses(self, primary_response: str, shadow_response: str) -> dict:
        """轻量级自动评估主/影子响应的相似度（无需额外 LLM 调用）。"""
        if not primary_response or not shadow_response:
            return {"score": 0.0, "reason": "空响应"}

        # 长度相似度
        len_primary = len(primary_response)
        len_shadow = len(shadow_response)
        length_ratio = (
            min(len_primary, len_shadow) / max(len_primary, len_shadow)
            if max(len_primary, len_shadow) > 0 else 0
        )
        length_score = 0.3 * length_ratio

        # 关键词重叠：含中文时按字符分词，否则按空格分词
        def tokens(text: str) -> set[str]:
            if any("\u4e00" <= ch <= "\u9fff" for ch in text):
                return set(text)
            return set(text.lower().split())

        primary_tokens = tokens(primary_response)
        shadow_tokens = tokens(shadow_response)
        if not primary_tokens:
            overlap_score = 0.0
        else:
            overlap = len(primary_tokens & shadow_tokens) / len(primary_tokens)
            overlap_score = 0.4 * overlap

        # 结构一致性：代码块、列表、表格
        def structure_features(text: str) -> set[str]:
            features = set()
            if "```" in text:
                features.add("code_block")
            if any(line.strip().startswith(("- ", "* ", "1. ")) for line in text.splitlines()):
                features.add("list")
            if "|" in text and "---" in text:
                features.add("table")
            return features

        primary_struct = structure_features(primary_response)
        shadow_struct = structure_features(shadow_response)
        if primary_struct:
            struct_score = 0.3 * len(primary_struct & shadow_struct) / len(primary_struct)
        else:
            struct_score = 0.3

        total = length_score + overlap_score + struct_score
        return {
            "score": round(total, 3),
            "length_score": round(length_score, 3),
            "overlap_score": round(overlap_score, 3),
            "struct_score": round(struct_score, 3),
        }

    def record_comparison(
        self,
        user_input: str,
        primary_model: str,
        primary_response: str,
        shadow_model: str,
        shadow_response: str,
    ):
        """记录对比结果，并附加自动评估分数。"""
        with self._lock:
            evaluation = self._evaluate_responses(primary_response, shadow_response)
            entry = {
                "timestamp": time.time(),
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_preview": user_input[:200],
                "primary_model": primary_model,
                "primary_response_preview": primary_response[:200],
                "shadow_model": shadow_model,
                "shadow_response_preview": shadow_response[:200],
                "evaluation": evaluation,
            }
            try:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.debug(f"Shadow log write error: {e}")
    
    def get_comparison_count(self) -> int:
        """获取对比记录数"""
        try:
            with open(self.log_file, "r") as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0


# ═══════════════════════════════════════════════════════════════════
# 路由矩阵配置化
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_ROUTING_MATRIX = {
    # (task_type, complexity) → model_key
    ("classify", "light"):           "cheapest",
    ("classify", "medium"):          "cheapest",
    ("extract", "light"):            "cheapest",
    ("extract", "medium"):           "cheapest",
    ("simple_qa", "light"):          "cheapest",
    ("simple_qa", "medium"):         "balanced",
    ("long_doc", "light"):           "balanced",
    ("long_doc", "medium"):          "balanced",
    ("long_doc", "heavy"):           "smartest",
    ("code", "light"):               "cheapest",
    ("code", "medium"):              "balanced",
    ("code", "heavy"):               "smartest",
    ("math", "light"):               "balanced",
    ("math", "medium"):              "balanced",
    ("math", "heavy"):               "smartest",
    ("complex_reasoning", "light"):  "balanced",
    ("complex_reasoning", "medium"): "smartest",
    ("complex_reasoning", "heavy"):  "smartest",
    ("agent", "light"):              "balanced",
    ("agent", "medium"):             "smartest",
    ("agent", "heavy"):              "smartest",
}


class RoutingMatrix:
    """路由矩阵 — 支持 JSON 热更新"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or str(Path.home() / ".hermes" / "routing_matrix.json")
        self._matrix: dict = {}
        self._cache_time: float = 0
        self._cache_ttl: float = 60  # 60 秒刷新一次
        self._lock = threading.Lock()
        
        # 初始化：如果配置文件不存在，创建默认配置
        if not os.path.exists(self.config_file):
            self._save_default()
        
        self._load()
    
    def get_model(self, task_type: str, complexity: str) -> str:
        """根据任务类型和复杂度获取模型策略"""
        with self._lock:
            self._refresh_if_stale()
            key = (task_type, complexity)
            return self._matrix.get(key, "balanced")
    
    def _load(self):
        """加载配置"""
        try:
            with open(self.config_file, "r") as f:
                data = json.load(f)
            # 转换 key 格式
            self._matrix = {}
            for key_str, value in data.items():
                if isinstance(key_str, str) and "," in key_str:
                    parts = key_str.strip("()").split(",")
                    key = (parts[0].strip().strip("'"), parts[1].strip().strip("'"))
                    self._matrix[key] = value
            self._cache_time = time.time()
        except Exception as e:
            logger.debug(f"Routing matrix load error: {e}")
            self._matrix = dict(_DEFAULT_ROUTING_MATRIX)
    
    def _refresh_if_stale(self):
        """如果缓存过期，重新加载"""
        if time.time() - self._cache_time > self._cache_ttl:
            self._load()
    
    def _save_default(self):
        """保存默认配置"""
        try:
            # 转换 key 为字符串
            data = {}
            for key, value in _DEFAULT_ROUTING_MATRIX.items():
                data[str(key)] = value
            with open(self.config_file, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Created default routing matrix: {self.config_file}")
        except Exception as e:
            logger.debug(f"Routing matrix save error: {e}")


# ═══════════════════════════════════════════════════════════════════
# 全局实例
# ═══════════════════════════════════════════════════════════════════

_cost_monitor = CostMonitor()
_shadow_evaluator = ShadowEvaluator()
_routing_matrix = RoutingMatrix()


def get_cost_monitor() -> CostMonitor:
    return _cost_monitor


def get_shadow_evaluator() -> ShadowEvaluator:
    return _shadow_evaluator


def get_routing_matrix() -> RoutingMatrix:
    return _routing_matrix


def print_cost_dashboard():
    """打印成本仪表盘"""
    summary = _cost_monitor.get_summary()
    daily_cost = _cost_monitor.get_daily_cost()
    
    print(f"\n{'='*70}")
    print(f"Model Router 成本监控")
    print(f"{'='*70}")
    print(f"今日成本: ${daily_cost:.4f}")
    print(f"\n{'Model':<30}{'Calls':<10}{'Cost($)':<12}{'AvgLat(ms)':<12}")
    print("-" * 64)
    
    for key, s in sorted(summary.items(), key=lambda x: -x[1]["cost_usd"]):
        print(f"{key:<30}{s['calls']:<10}{s['cost_usd']:<12.4f}{s['avg_latency_ms']:<12.1f}")
    
    shadow_count = _shadow_evaluator.get_comparison_count()
    print(f"\n影子流量对比记录: {shadow_count}")
    print(f"{'='*70}")
