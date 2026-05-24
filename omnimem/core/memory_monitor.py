import gc
import sys
import threading
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MemoryMonitor:
    def __init__(self, interval: float = 60.0, warning_mb: float = 1200.0):
        self._interval = interval
        self._warning_mb = warning_mb
        self._timer: threading.Timer | None = None
        self._running = False
        self._callbacks: list = []

    def start(self) -> None:
        self._running = True
        self._schedule()

    def stop(self) -> None:
        self._running = False
        if self._timer:
            self._timer.cancel()

    def on_warning(self, callback) -> None:
        self._callbacks.append(callback)

    def get_usage(self) -> dict:
        usage: dict[str, Any] = {"rss_mb": 0, "objects": 0}
        try:
            import psutil
            process = psutil.Process()
            usage["rss_mb"] = process.memory_info().rss / 1024 / 1024
        except ImportError:
            usage["rss_mb"] = 0
        usage["objects"] = len(gc.get_objects())
        return usage

    def _schedule(self) -> None:
        if not self._running:
            return
        self._check()
        self._timer = threading.Timer(self._interval, self._schedule)
        self._timer.daemon = True
        self._timer.start()

    def _check(self) -> None:
        usage = self.get_usage()
        if usage["rss_mb"] > self._warning_mb:
            logger.warning("Memory usage: %.1fMB exceeds %.1fMB threshold", usage["rss_mb"], self._warning_mb)
            for cb in self._callbacks:
                try:
                    cb(usage)
                except Exception as e:
                    logger.warning("MemoryMonitor callback failed: %s", e)
