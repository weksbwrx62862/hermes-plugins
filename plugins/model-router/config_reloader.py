"""model-router 配置热重载器。

监听 ``~/.hermes/config.yaml`` 文件变更，在独立后台线程中轮询 mtime，
检测到变更后在 30 秒内完成：

- 模型池重新构建
- 路由矩阵重新加载
- 平台加分（scoring）重新加载
- 策略配置重新加载

并通过读写锁 + 原子替换保证热重载期间正在进行的请求不受影响。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from . import cache as _cache_module

logger = logging.getLogger(__name__)

RouteCache = _cache_module.RouteCache


class _RWLock:
    """简单的读者-写者锁。

    支持多个并发读者，写者独占。用于热重载期间隔离「构建新快照」
    与「读取当前快照」的并发访问。
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self) -> None:
        with self._cond:
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        self._cond.acquire()
        while self._readers > 0:
            self._cond.wait()

    def release_write(self) -> None:
        self._cond.release()

    def read(self) -> "_RWLockContext":
        return _RWLockContext(self, write=False)

    def write(self) -> "_RWLockContext":
        return _RWLockContext(self, write=True)


class _RWLockContext:
    def __init__(self, lock: _RWLock, write: bool) -> None:
        self._lock = lock
        self._write = write

    def __enter__(self) -> "_RWLockContext":
        if self._write:
            self._lock.acquire_write()
        else:
            self._lock.acquire_read()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._write:
            self._lock.release_write()
        else:
            self._lock.release_read()


class ConfigReloader:
    """``config.yaml`` 配置热重载器。

    参数:
        config_path: 监听的 YAML 配置文件路径，默认 ``~/.hermes/config.yaml``。
        routing_matrix_path: 可选的路由矩阵 JSON 文件路径；
            未指定时使用 ``cost_monitor`` 中的全局单例。
        poll_interval: 轮询间隔（秒），默认 5 秒。
        max_reload_time: 单次重载允许的最大耗时（秒），默认 30 秒。
        route_cache: 路由结果缓存实例；未指定时内部创建一个 ``RouteCache``。
        config_loader: 可选的配置加载函数，返回配置字典；
            主要用于测试场景覆盖默认 YAML 读取。
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        routing_matrix_path: Optional[str] = None,
        poll_interval: float = 5.0,
        max_reload_time: float = 30.0,
        route_cache: Optional[RouteCache] = None,
        config_loader: Optional[Callable[[], dict]] = None,
    ) -> None:
        self.config_path = Path(config_path or Path.home() / ".hermes" / "config.yaml").resolve()
        self.routing_matrix_path = Path(routing_matrix_path) if routing_matrix_path else None
        self.poll_interval = float(poll_interval)
        self.max_reload_time = float(max_reload_time)
        self._route_cache = route_cache if route_cache is not None else RouteCache()
        self._config_loader = config_loader

        self._rwlock = _RWLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._last_mtime: float = 0.0
        self._last_size: int = 0
        self._last_hash: str = ""

        self._snapshot: dict[str, Any] = {}
        self._reload_count: int = 0
        self._last_summary: dict[str, Any] = {}
        self._routing_matrix_instance: Optional[Any] = None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动后台监听线程。"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("配置热重载线程已运行，忽略重复启动")
            return
        self._stop_event.clear()
        # 先执行一次强制加载，确保启动时快照可用
        try:
            self.force_reload()
        except Exception:
            logger.exception("配置热重载器初始加载失败")

        self._thread = threading.Thread(
            target=self._watch_loop,
            name="model-router-config-reloader",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "配置热重载监听已启动: path=%s interval=%ss",
            self.config_path,
            self.poll_interval,
        )

    def stop(self) -> None:
        """停止后台监听线程。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def force_reload(self) -> bool:
        """立即执行一次强制重载。"""
        return self._check_and_reload(force=True)

    def is_alive(self) -> bool:
        """后台监听线程是否存活。"""
        return self._thread is not None and self._thread.is_alive()

    def get_pool(self) -> list[dict[str, Any]]:
        """线程安全地获取当前模型池快照。"""
        with self._rwlock.read():
            return list(self._snapshot.get("model_pool", []))

    def get_snapshot(self) -> dict[str, Any]:
        """线程安全地获取当前完整快照。"""
        with self._rwlock.read():
            return dict(self._snapshot)

    def get_last_summary(self) -> dict[str, Any]:
        """线程安全地获取最近一次热重载变更摘要。"""
        with self._rwlock.read():
            return dict(self._last_summary)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _watch_loop(self) -> None:
        """后台轮询循环。"""
        while not self._stop_event.is_set():
            try:
                self._check_and_reload()
            except Exception:
                logger.exception("配置热重载检查失败")
            self._stop_event.wait(self.poll_interval)

    def _read_config(self) -> dict[str, Any]:
        """读取配置文件。"""
        if self._config_loader is not None:
            return self._config_loader()
        if yaml is None:
            raise RuntimeError("PyYAML 未安装，无法读取 config.yaml")
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _file_changed(self) -> bool:
        """基于 mtime + 文件大小 + sha256 判断文件是否发生实质变更。"""
        try:
            st = self.config_path.stat()
            mtime, size = st.st_mtime, st.st_size
        except FileNotFoundError:
            return False

        if mtime != self._last_mtime or size != self._last_size:
            try:
                new_hash = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
            except Exception:
                new_hash = f"{mtime}:{size}"
            return new_hash != self._last_hash
        return False

    def _check_and_reload(self, force: bool = False) -> bool:
        """检查文件变更并执行热重载。"""
        if not force and not self._file_changed():
            return False

        # 获取变更后的文件元信息
        try:
            st = self.config_path.stat()
            new_mtime, new_size = st.st_mtime, st.st_size
            new_hash = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        except FileNotFoundError:
            return False
        except Exception:
            new_mtime, new_size, new_hash = self._last_mtime, self._last_size, ""

        # 先不带锁构建新快照，允许读取端继续使用旧快照
        try:
            new_snapshot, summary = self._reload(new_hash)
        except Exception:
            logger.exception("配置重载失败")
            return False

        # 原子替换快照
        with self._rwlock.write():
            self._snapshot = new_snapshot
            self._last_mtime = new_mtime
            self._last_size = new_size
            self._last_hash = new_hash
            self._reload_count += 1
            self._last_summary = summary

        self._log_reload_event(summary)
        return True

    def _reload(self, config_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """执行实际重载逻辑并返回新快照与变更摘要。"""
        import importlib

        pkg_name = "plugins.model-router"
        try:
            model_router = importlib.import_module(pkg_name)
        except Exception as exc:
            raise RuntimeError(f"无法加载 {pkg_name}: {exc}") from exc

        # 失效 model-router 内部缓存，确保后续 _build_pool 读取到最新配置
        if hasattr(model_router, "invalidate_config_cache"):
            model_router.invalidate_config_cache()
        if hasattr(model_router, "_POOL_CACHE"):
            model_router._POOL_CACHE = None
            model_router._POOL_CACHE_TIME = 0.0

        config = self._read_config()
        mr_cfg = config.get("plugins", {}).get("model-router", {})

        start = time.monotonic()

        _build_pool = getattr(model_router, "_build_pool", None)
        if _build_pool is None:
            raise RuntimeError("model_router._build_pool 不存在")
        new_pool = _build_pool(config)
        elapsed = time.monotonic() - start

        _load_scoring_config = getattr(model_router, "_load_scoring_config", lambda config=None: {})
        _get_strategy = getattr(model_router, "_get_strategy", lambda config=None: "auto")
        _set_current_strategy = getattr(model_router, "_set_current_strategy", None)

        new_scoring = _load_scoring_config(config)
        new_strategy = _get_strategy(config)
        if _set_current_strategy is not None:
            _set_current_strategy(new_strategy)

        # 重新加载路由矩阵
        matrix = self._reload_routing_matrix()

        old_pool = self._snapshot.get("model_pool", [])
        old_scoring = self._snapshot.get("scoring", {})
        old_strategy = self._snapshot.get("strategy", "auto")

        summary = self._compute_summary(
            old_pool,
            new_pool,
            old_scoring,
            new_scoring,
            old_strategy,
            new_strategy,
            mr_cfg,
            elapsed,
        )

        snapshot: dict[str, Any] = {
            "config_hash": config_hash,
            "model_pool": new_pool,
            "scoring": new_scoring,
            "strategy": new_strategy,
            "model_router_config": mr_cfg,
            "routing_matrix": matrix,
            "reload_time": elapsed,
        }

        # 触发缓存失效
        try:
            invalidated_count = self._route_cache.invalidate_all()
        except Exception as exc:
            logger.warning("路由缓存失效失败: %s", exc)
            invalidated_count = 0
        summary["cache_invalidated"] = invalidated_count

        if elapsed > self.max_reload_time:
            logger.warning(
                "配置重载耗时 %.2fs，超过阈值 %.0fs",
                elapsed,
                self.max_reload_time,
            )

        return snapshot, summary

    def _reload_routing_matrix(self) -> Optional[Any]:
        """重新加载路由矩阵。"""
        try:
            from .cost_monitor import RoutingMatrix, get_routing_matrix
        except Exception as exc:
            logger.warning("路由矩阵模块加载失败: %s", exc)
            return None

        if self.routing_matrix_path is not None:
            if self._routing_matrix_instance is None:
                self._routing_matrix_instance = RoutingMatrix(
                    config_file=str(self.routing_matrix_path)
                )
            else:
                self._routing_matrix_instance._load()
            return self._routing_matrix_instance

        matrix = get_routing_matrix()
        matrix._load()
        matrix._cache_time = time.time()
        return matrix

    def _compute_summary(
        self,
        old_pool: list[dict[str, Any]],
        new_pool: list[dict[str, Any]],
        old_scoring: dict[str, Any],
        new_scoring: dict[str, Any],
        old_strategy: str,
        new_strategy: str,
        mr_cfg: dict[str, Any],
        elapsed: float,
    ) -> dict[str, Any]:
        """计算两次快照之间的变更摘要。"""
        old_by_name: dict[str, dict[str, Any]] = {
            m.get("name"): m for m in old_pool if m.get("name")
        }
        new_by_name: dict[str, dict[str, Any]] = {
            m.get("name"): m for m in new_pool if m.get("name")
        }
        old_names = set(old_by_name)
        new_names = set(new_by_name)

        added_models = sorted(new_names - old_names)
        removed_models = sorted(old_names - new_names)

        model_attrs = ("provider", "cost", "speed", "quality", "context_window", "base_url", "key")
        modified_models = []
        for name in sorted(old_names & new_names):
            old_m = old_by_name[name]
            new_m = new_by_name[name]
            changes: dict[str, Any] = {}
            for attr in model_attrs:
                old_v = old_m.get(attr)
                new_v = new_m.get(attr)
                if old_v != new_v:
                    changes[attr] = {"old": old_v, "new": new_v}
            if changes:
                modified_models.append({"name": name, "changes": changes})

        scoring_changes: dict[str, Any] = {}
        for key in sorted(set(old_scoring) | set(new_scoring)):
            old_v = old_scoring.get(key)
            new_v = new_scoring.get(key)
            if old_v != new_v:
                scoring_changes[key] = {"old": old_v, "new": new_v}

        old_allowed = set(
            self._snapshot.get("model_router_config", {}).get("allowed_providers", [])
        )
        new_allowed = set(mr_cfg.get("allowed_providers", []))

        return {
            "added_models": added_models,
            "removed_models": removed_models,
            "modified_models": modified_models,
            "model_count": len(new_pool),
            "scoring_changes": scoring_changes,
            "strategy_change": (
                {"old": old_strategy, "new": new_strategy}
                if old_strategy != new_strategy
                else None
            ),
            "allowed_provider_changes": {
                "added": sorted(new_allowed - old_allowed),
                "removed": sorted(old_allowed - new_allowed),
            },
            "reload_time": elapsed,
        }

    def _log_reload_event(self, summary: dict[str, Any]) -> None:
        """记录热重载事件日志。"""
        strategy_old = summary["strategy_change"]["old"] if summary["strategy_change"] else "-"
        strategy_new = summary["strategy_change"]["new"] if summary["strategy_change"] else "-"
        scoring_keys = list(summary["scoring_changes"].keys()) if summary["scoring_changes"] else "-"
        logger.info(
            "配置热重载完成: 模型数=%d 新增=%d 删除=%d 修改=%d "
            "策略=%s→%s 平台加分变化=%s 缓存失效=%d 耗时=%.3fs",
            summary["model_count"],
            len(summary["added_models"]),
            len(summary["removed_models"]),
            len(summary["modified_models"]),
            strategy_old,
            strategy_new,
            scoring_keys,
            summary.get("cache_invalidated", 0),
            summary["reload_time"],
        )
