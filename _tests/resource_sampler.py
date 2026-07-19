"""资源占用采样模块

在插件 register(ctx) 执行前后采样资源占用情况:
- CPU 时间(用户态 + 系统态)
- 壁钟时间
- 内存当前值与峰值(通过 tracemalloc 跟踪 Python 分配)
- 文件描述符数量
- 子进程创建数量(通过 monkey-patch subprocess.Popen.__init__)

注意: tracemalloc 不跟踪 C 扩展分配的内存, 内存数据仅反映 Python 层分配。
"""

from __future__ import annotations

import os
import resource
import subprocess
import threading
import time
import tracemalloc
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


# 资源异常阈值
CPU_TIME_WARN_SECONDS = 5.0
MEM_WARN_BYTES = 100 * 1024 * 1024  # 100 MB
FD_WARN_COUNT = 200


@dataclass
class ResourceSnapshot:
    """单次资源采样快照。"""

    cpu_time: float = 0.0       # CPU 时间差(秒)
    wall_time: float = 0.0      # 壁钟时间差(秒)
    mem_current: float = 0.0    # 当前内存占用(字节)
    mem_peak: float = 0.0       # 峰值内存占用(字节)
    fd_count: int = 0           # 文件描述符数量(采样时刻)
    subprocess_count: int = 0   # 子进程创建数量


class ResourceSampler:
    """资源采样器。

    用法:
        sampler = ResourceSampler()
        sampler.start()
        ... 执行被测代码 ...
        snapshot = sampler.stop()
    """

    def __init__(self) -> None:
        self._start_cpu: float = 0.0
        self._start_wall: float = 0.0
        self._start_fd: int = 0
        self._subprocess_count: int = 0
        self._subprocess_lock = threading.Lock()
        self._orig_popen_init: Any = None
        self._tracemalloc_started_by_us: bool = False

    # -- 子进程计数 monkey-patch ------------------------------------------

    def _patch_subprocess(self) -> None:
        """monkey-patch subprocess.Popen.__init__ 以统计子进程创建数。"""
        self._orig_popen_init = subprocess.Popen.__init__

        def _counting_init(popen_self, *args: Any, **kwargs: Any):
            with self._subprocess_lock:
                self._subprocess_count += 1
            # 调用原始构造函数; 若失败则不真正创建进程
            return self._orig_popen_init(popen_self, *args, **kwargs)

        subprocess.Popen.__init__ = _counting_init

    def _unpatch_subprocess(self) -> None:
        """恢复原始 subprocess.Popen.__init__。"""
        if self._orig_popen_init is not None:
            subprocess.Popen.__init__ = self._orig_popen_init
            self._orig_popen_init = None

    # -- 文件描述符计数 ----------------------------------------------------

    @staticmethod
    def _current_fd_count() -> int:
        """返回当前进程打开的文件描述符数量。"""
        try:
            return len(os.listdir(f"/proc/{os.getpid()}/fd"))
        except OSError:
            # 非 Linux 环境降级处理
            return 0

    # -- 生命周期 ----------------------------------------------------------

    def start(self) -> None:
        """启动采样: 启动 tracemalloc, 记录初始 CPU/壁钟时间/FD 数, patch 子进程。"""
        # tracemalloc 可能已被其他代码启动, 记录是否由本采样器启动
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_started_by_us = True
        else:
            # 已在跟踪则重置峰值统计, 以获得本次采样的纯净峰值
            tracemalloc.reset_peak()
            self._tracemalloc_started_by_us = False

        # CPU 时间 = 用户态 + 系统态 (ru_utime + ru_stime, 单位秒)
        ru = resource.getrusage(resource.RUSAGE_SELF)
        self._start_cpu = ru.ru_utime + ru.ru_stime
        self._start_wall = time.perf_counter()
        self._start_fd = self._current_fd_count()
        self._subprocess_count = 0
        self._patch_subprocess()

    def stop(self) -> ResourceSnapshot:
        """停止采样并返回资源快照。"""
        # 先取壁钟与内存, 尽量减少停止操作本身的计入
        wall_now = time.perf_counter()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        cpu_now = ru.ru_utime + ru.ru_stime

        mem_current, mem_peak = 0.0, 0.0
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            mem_current = float(current)
            mem_peak = float(peak)

        fd_now = self._current_fd_count()

        # 恢复 subprocess patch
        self._unpatch_subprocess()

        # 若由本采样器启动 tracemalloc, 则停止以恢复原状
        if self._tracemalloc_started_by_us and tracemalloc.is_tracing():
            tracemalloc.stop()

        snap = ResourceSnapshot(
            cpu_time=max(0.0, cpu_now - self._start_cpu),
            wall_time=max(0.0, wall_now - self._start_wall),
            mem_current=mem_current,
            mem_peak=mem_peak,
            fd_count=fd_now,
            subprocess_count=self._subprocess_count,
        )
        return snap

    # -- 格式化 ------------------------------------------------------------

    @staticmethod
    def format_snapshot(snap: ResourceSnapshot) -> Dict[str, Any]:
        """将快照转为可序列化字典, 并标注异常项。"""
        anomalies: list = []
        if snap.cpu_time > CPU_TIME_WARN_SECONDS:
            anomalies.append(f"CPU 时间 {snap.cpu_time:.2f}s 超过 {CPU_TIME_WARN_SECONDS}s")
        if snap.mem_peak > MEM_WARN_BYTES:
            anomalies.append(f"内存峰值 {snap.mem_peak / 1024 / 1024:.1f}MB 超过 100MB")
        if snap.fd_count > FD_WARN_COUNT:
            anomalies.append(f"FD 数 {snap.fd_count} 超过 {FD_WARN_COUNT}")

        data = asdict(snap)
        # 附加人类可读的内存单位
        data["mem_current_mb"] = round(snap.mem_current / 1024 / 1024, 2)
        data["mem_peak_mb"] = round(snap.mem_peak / 1024 / 1024, 2)
        data["anomalies"] = anomalies
        data["has_anomaly"] = len(anomalies) > 0
        return data
