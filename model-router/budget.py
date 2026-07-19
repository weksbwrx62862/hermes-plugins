"""LiteLLM 风格的预算管控模块。

在现有的 CostMonitor 基础上增加：
1. 每日/每月预算限额
2. 阈值告警（50%/80%/95%/100%）
3. 超预算自动降级/拒绝
4. 预算日志和告警通知

配置 (~/.hermes/config.yaml):
  plugins.model-router.budget:
    enabled: true
    daily_max_usd: 1.0
    monthly_max_usd: 20.0
    warn_thresholds: [0.5, 0.8, 0.95]
    on_exceed: warn  # warn | block | degrade
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

BEIJING_TZ = timezone(timedelta(hours=8))


class BudgetManager:
    """预算管理器。

    每次 API 调用前检查预算，防止意外超支。
    """

    _CONFIG_DEFAULTS = {
        "enabled": True,
        "daily_max_usd": 1.0,
        "monthly_max_usd": 20.0,
        "warn_thresholds": [0.5, 0.8, 0.95],
        "on_exceed": "warn",  # warn | block | degrade
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._alert_log: Path = Path.home() / ".hermes" / "budget_alerts.jsonl"
        self._fired_alerts: dict[str, float] = {}  # 防噪：同一个阈值一天只告警一次

    def _load_config(self) -> dict:
        """加载预算配置。"""
        try:
            config_path = Path.home() / ".hermes" / "config.yaml"
            if not config_path.exists():
                return self._CONFIG_DEFAULTS
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            plugins = cfg.get("plugins", {})
            mr = plugins.get("model-router", {})
            budget_cfg = mr.get("budget", {})
            if not budget_cfg:
                return self._CONFIG_DEFAULTS
            return {**self._CONFIG_DEFAULTS, **budget_cfg}
        except Exception:
            return self._CONFIG_DEFAULTS

    def check(self, call_cost_usd: float = 0.0) -> dict[str, Any]:
        """检查预算状态。

        Args:
            call_cost_usd: 即将发生的调用预估成本

        Returns:
            {"allowed": bool, "reason": str, "daily": float, "monthly": float}
        """
        config = self._load_config()
        if not config.get("enabled", True):
            return {"allowed": True, "reason": "预算管控未启用"}

        daily_used = self._get_daily_cost()
        monthly_used = self._get_monthly_cost()

        daily_max = config.get("daily_max_usd", 1.0)
        monthly_max = config.get("monthly_max_usd", 20.0)
        thresholds = config.get("warn_thresholds", [0.5, 0.8, 0.95])
        on_exceed = config.get("on_exceed", "warn")

        # 检查每日预算
        daily_ratio = daily_used / daily_max if daily_max > 0 else 0.0
        projected = (daily_used + call_cost_usd) / daily_max if daily_max > 0 else 0.0

        # 阈值告警
        for threshold in thresholds:
            if daily_ratio >= threshold and not self._alert_fired("daily", threshold):
                self._fire_alert(
                    "daily",
                    f"每日预算已达 {threshold*100:.0f}% "
                    f"(${daily_used:.4f}/${daily_max:.2f})",
                )

        if projected > 1.0:
            if on_exceed == "block":
                return {
                    "allowed": False,
                    "reason": f"超过每日预算限额 (${daily_used:.2f}/${daily_max:.2f})",
                    "suggested_strategy": "cheapest",
                }
            elif on_exceed == "degrade":
                return {
                    "allowed": True,
                    "reason": "超预算，已降级到廉价模型",
                    "degraded": True,
                    "suggested_strategy": "cheapest",
                }
            else:
                # warn + allow
                self._fire_alert("daily_exceeded",
                    f"⚠️ 已超过每日预算！(${daily_used:.2f}/${daily_max:.2f})")

        return {
            "allowed": True,
            "daily_used": daily_used,
            "daily_max": daily_max,
            "daily_ratio": round(daily_ratio, 4),
            "monthly_used": monthly_used,
            "monthly_max": monthly_max,
        }

    def _get_daily_cost(self) -> float:
        """获取今日累计成本。"""
        try:
            cost_log = Path.home() / ".hermes" / "model_router_costs.jsonl"
            if not cost_log.exists():
                return 0.0
            today = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d")
            total = 0.0
            with open(cost_log) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        ts = entry.get("timestamp", 0)
                        dt = datetime.fromtimestamp(ts, tz=BEIJING_TZ).strftime("%Y-%m-%d")
                        if dt == today:
                            total += float(entry.get("cost_usd", 0))
                    except (json.JSONDecodeError, ValueError):
                        continue
            return round(total, 4)
        except Exception:
            return 0.0

    def _get_monthly_cost(self) -> float:
        """获取本月累计成本。"""
        try:
            cost_log = Path.home() / ".hermes" / "model_router_costs.jsonl"
            if not cost_log.exists():
                return 0.0
            now = datetime.now(BEIJING_TZ)
            month_prefix = now.strftime("%Y-%m")
            total = 0.0
            with open(cost_log) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        ts = entry.get("timestamp", 0)
                        dt = datetime.fromtimestamp(ts, tz=BEIJING_TZ).strftime("%Y-%m")
                        if dt == month_prefix:
                            total += float(entry.get("cost_usd", 0))
                    except (json.JSONDecodeError, ValueError):
                        continue
            return round(total, 4)
        except Exception:
            return 0.0

    def _alert_fired(self, category: str, threshold: float) -> bool:
        """检查告警是否已发出（防噪）。"""
        key = f"{category}:{threshold}:{datetime.now(BEIJING_TZ).strftime('%Y-%m-%d')}"
        return self._fired_alerts.get(key, 0) > 0

    def _fire_alert(self, category: str, message: str) -> None:
        """发出预算告警并记录日志。"""
        now = datetime.now(BEIJING_TZ)
        key = f"{category}:{now.strftime('%Y-%m-%d')}"
        self._fired_alerts[key] = time.time()

        alert = {
            "timestamp": now.isoformat(),
            "category": category,
            "message": message,
        }
        try:
            self._alert_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self._alert_log, "a") as f:
                f.write(json.dumps(alert, ensure_ascii=False) + "\n")
        except Exception:
            pass
        logger.warning("💰 Budget Alert: %s", message)

    def get_summary(self) -> dict[str, Any]:
        """获取预算摘要。"""
        config = self._load_config()
        daily = self._get_daily_cost()
        monthly = self._get_monthly_cost()
        return {
            "daily_used": daily,
            "daily_max": config.get("daily_max_usd", 1.0),
            "daily_ratio": round(daily / config.get("daily_max_usd", 1.0), 4) if config.get("daily_max_usd", 1.0) > 0 else 0.0,
            "monthly_used": monthly,
            "monthly_max": config.get("monthly_max_usd", 20.0),
            "enabled": config.get("enabled", True),
            "on_exceed": config.get("on_exceed", "warn"),
        }

    time = __import__("time")  # noqa


# ─── 全局单例 ────────────────────────────────────────────

_budget_manager: BudgetManager | None = None
_budget_lock = threading.Lock()


def get_budget_manager() -> BudgetManager:
    """获取全局 BudgetManager 单例。"""
    global _budget_manager
    if _budget_manager is None:
        with _budget_lock:
            if _budget_manager is None:
                _budget_manager = BudgetManager()
    return _budget_manager


def check_budget(estimated_cost: float = 0.0) -> dict[str, Any]:
    """快捷函数：检查预算。"""
    return get_budget_manager().check(estimated_cost)
