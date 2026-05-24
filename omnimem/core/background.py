"""BackgroundTaskExecutor — 统一后台任务执行器。

替代 provider.py 中每轮新建 threading.Thread 的模式，提供：
  1. 固定线程池（避免线程创建/销毁开销）
  2. 背压控制（统一队列，顺序或并发执行）
  3. 失败观测（异常捕获与日志，不静默丢失）

设计为 SagaCoordinator 的底层执行基础设施。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


class BackgroundTaskExecutor:
    """统一后台任务执行器。

    所有后台任务（索引更新、Saga 补偿、治理巡检）都通过此类提交，
    避免分散的 threading.Thread 造成资源浪费和失败不可观测。
    """

    def __init__(self, max_workers: int = 2):
        """初始化后台执行器。

        Args:
            max_workers: 线程池大小。默认 2：
                1 个用于索引更新，1 个用于 Saga 补偿/治理巡检。
        """
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="omnimem_bg",
        )
        self._pending_tasks = 0
        self._lock = threading.Lock()

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        """提交任务到后台线程池。

        Args:
            fn: 可调用对象
            *args, **kwargs: 传递给 fn 的参数

        Returns:
            Future 对象，可用于获取结果或检查异常
        """
        with self._lock:
            self._pending_tasks += 1
        future = self._executor.submit(self._wrap, fn, *args, **kwargs)
        return future

    def _wrap(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """包装任务：捕获异常、更新计数。"""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error("Background task failed: %s", e, exc_info=True)
            raise
        finally:
            with self._lock:
                self._pending_tasks -= 1

    def shutdown(self, wait: bool = True) -> None:
        """优雅关闭线程池。"""
        with self._lock:
            pending = self._pending_tasks
        logger.info("BackgroundTaskExecutor shutting down (pending=%d)", pending)
        self._executor.shutdown(wait=wait)

    @property
    def pending_tasks(self) -> int:
        """当前待执行或执行中的任务数。"""
        with self._lock:
            return self._pending_tasks
