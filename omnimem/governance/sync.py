"""SyncEngine — 分布式同步引擎。

解决多实例并发写入 SQLite 时的数据不一致和锁冲突问题。

同步机制:
  - 单实例模式 (默认): 无需额外开销，直接写入
  - 单主机多进程模式: 使用 fcntl 文件锁 + WAL 模式
  - 多主机分布式模式: 使用基于文件的变更日志 + 合并策略

设计原则:
  - 零外部依赖: 不依赖 Redis/MQ/consul
  - 向后兼容: 默认不改变现有行为
  - 按需启用: 通过配置开启同步
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnimem.governance.vector_clock import (
    VectorClock,
    merge_records,
)

logger = logging.getLogger(__name__)

# ★ 平台兼容：fcntl 仅在 Unix 可用，Windows 上降级为进程内锁
try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False


# ─── 数据模型 ────────────────────────────────────────────────


@dataclass
class SyncConfig:
    """同步配置。"""

    mode: str = "none"  # none / file_lock / changelog
    instance_id: str = ""
    instance_name: str = ""
    sync_interval: int = 30  # 同步间隔(秒)
    conflict_resolution: str = "latest_wins"  # latest_wins / manual
    changelog_path: str = ""

    def __post_init__(self) -> None:
        if not self.instance_id:
            self.instance_id = f"instance-{uuid.uuid4().hex[:8]}"
        if not self.instance_name:
            self.instance_name = f"omnimem-{os.getpid()}"


# ─── 文件锁管理器 ─────────────────────────────────────────────


class FileLockManager:
    """基于 fcntl 的跨进程文件锁（Unix）；Windows 上降级为进程内线程锁。

    适用于单主机多进程场景，防止多个进程同时写入 SQLite。
    """

    def __init__(self, lock_dir: Path):
        self._lock_dir = lock_dir
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file = self._lock_dir / "omnimem.lock"
        self._lock_file.touch(exist_ok=True)
        self._fd: int | None = None
        self._lock_count = 0
        self._wait_time = 0.0
        # Windows 降级：进程内线程锁（多进程并发写入仍依赖 SQLite WAL）
        self._fallback_lock = threading.Lock()
        self._has_fcntl = _HAS_FCNTL
        if not self._has_fcntl:
            logger.warning(
                "fcntl unavailable on this platform — FileLockManager falls back to threading.Lock (intra-process only)"
            )

    def acquire(self, timeout: float = 5.0, exclusive: bool = True) -> bool:
        """获取文件锁。

        Args:
            timeout: 超时时间(秒)
            exclusive: True=排他锁, False=共享锁

        Returns:
            是否成功获取锁
        """
        if not self._has_fcntl:
            self._fallback_lock.acquire()
            self._lock_count += 1
            return True

        if self._fd is None:
            self._fd = os.open(str(self._lock_file), os.O_RDWR)

        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH  # type: ignore[attr-defined]
        start_time = time.monotonic()

        while True:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)  # type: ignore[attr-defined]
                self._lock_count += 1
                return True
            except OSError:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    self._wait_time += elapsed
                    return False
                time.sleep(0.05)

    def release(self) -> None:
        """释放文件锁。"""
        if not self._has_fcntl:
            self._fallback_lock.release()
            return
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)  # type: ignore[attr-defined]
            except OSError as e:
                logger.warning("FileLock release failed: %s", e)

    def stats(self) -> dict[str, Any]:
        """获取锁统计。"""
        return {
            "acquisitions": self._lock_count,
            "total_wait_time_ms": round(self._wait_time * 1000, 2),
        }

    def close(self) -> None:
        """关闭锁。"""
        self.release()
        if self._has_fcntl and self._fd is not None:
            os.close(self._fd)
            self._fd = None


# ─── 变更日志 ────────────────────────────────────────────────


class ChangeLog:
    """变更日志，用于多主机分布式同步。

    每个实例写入自己的变更日志，其他实例定期读取并合并。
    """

    def __init__(self, changelog_dir: Path, instance_id: str):
        self._changelog_dir = changelog_dir
        self._changelog_dir.mkdir(parents=True, exist_ok=True)
        self._instance_id = instance_id
        self._my_log = self._changelog_dir / f"{instance_id}.jsonl"
        self._my_log.touch(exist_ok=True)
        self._lock = threading.Lock()

    def append(self, operation: str, table: str, data: dict[str, Any], vc: str = "") -> None:
        """追加一条变更记录。

        Args:
            operation: INSERT / UPDATE / DELETE
            table: 表名
            data: 变更数据
            vc: 可选的向量时钟 JSON
        """
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "instance_id": self._instance_id,
            "operation": operation,
            "table": table,
            "data": data,
        }
        if vc:
            entry["vc"] = vc
        with self._lock, open(self._my_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_new(self, since_ts: str, exclude_instance: str = "") -> list[dict[str, Any]]:
        """读取指定时间后的变更（排除指定实例）。

        Args:
            since_ts: 起始时间戳
            exclude_instance: 要排除的实例ID

        Returns:
            变更条目列表
        """
        changes = []
        log_files = list(self._changelog_dir.glob("*.jsonl"))

        for log_file in log_files:
            try:
                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        if entry.get("ts", "") > since_ts:
                            if exclude_instance and entry.get("instance_id") == exclude_instance:
                                continue
                            changes.append(entry)
            except Exception as e:
                logger.warning("ChangeLog read failed for %s: %s", log_file, e)

        # 按时间排序
        changes.sort(key=lambda x: x.get("ts", ""))
        return changes

    def get_last_ts(self) -> str:
        """获取最后一条变更的时间戳。"""
        last_ts = ""
        try:
            with open(self._my_log, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        last_ts = entry.get("ts", "")
        except Exception as e:
            logger.warning("ChangeLog get_last_ts failed: %s", e)
        return last_ts

    def trim(self, keep_last_n: int = 1000) -> None:
        """清理旧变更，只保留最近N条。"""
        with self._lock:
            try:
                lines = []
                with open(self._my_log, encoding="utf-8") as f:
                    lines = f.readlines()
                if len(lines) > keep_last_n:
                    with open(self._my_log, "w", encoding="utf-8") as f:
                        f.writelines(lines[-keep_last_n:])
            except Exception as e:
                logger.warning("ChangeLog trim failed: %s", e)


# ─── 同步引擎 ────────────────────────────────────────────────


class SyncEngine:
    """分布式同步引擎。

    协调文件锁和变更日志，提供统一的同步接口。
    """

    def __init__(self, data_dir: Path, config: SyncConfig | None = None):
        self._data_dir = data_dir
        self._config = config or SyncConfig()
        self._active = self._config.mode != "none"

        # 文件锁(单主机多进程)
        self._file_lock: FileLockManager | None = None
        if self._config.mode in ("file_lock", "changelog"):
            self._file_lock = FileLockManager(data_dir / "locks")

        # 变更日志(多主机分布式)
        self._changelog: ChangeLog | None = None
        if self._config.mode == "changelog":
            self._changelog = ChangeLog(
                data_dir / "changelogs",
                self._config.instance_id,
            )

        # 实例注册
        self._register_instance()

        logger.info(
            "SyncEngine initialized: mode=%s, instance=%s",
            self._config.mode,
            self._config.instance_id,
        )

    def write_with_lock(self, write_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """带锁的写入操作。

        Args:
            write_fn: 写入函数
            *args, **kwargs: 传递给写入函数的参数

        Returns:
            写入函数的返回值
        """
        if not self._active or not self._file_lock:
            # 无同步模式：直接执行
            return write_fn(*args, **kwargs)

        # 获取锁
        acquired = self._file_lock.acquire(timeout=5.0)
        if not acquired:
            logger.warning("SyncEngine: failed to acquire lock after timeout")
            return None

        try:
            result = write_fn(*args, **kwargs)

            # 记录变更日志
            if self._changelog:
                # write_fn 应该返回变更信息
                if isinstance(result, dict) and "sync_log" in result:
                    sync_info = result["sync_log"]
                    self._changelog.append(
                        operation=sync_info.get("operation", "UPDATE"),
                        table=sync_info.get("table", "unknown"),
                        data=sync_info.get("data", {}),
                    )

            return result
        finally:
            self._file_lock.release()

    def sync_from_others(
        self,
        apply_fn: Callable[..., Any],
        since_ts: str = "",
        get_local_fn: Callable[[str], dict[str, Any] | None] | None = None,
    ) -> int:
        """从其他实例同步变更并应用。

        Args:
            apply_fn: 应用变更的函数，签名: (change_entry) -> bool
            since_ts: 起始时间戳，为空则使用本地最新时间
            get_local_fn: 读取本地记录的函数，签名: (memory_id) -> Optional[Dict]
                          若提供，则启用 VectorClock 冲突检测与合并

        Returns:
            应用的变更数量
        """
        if not self._changelog:
            return 0

        if not since_ts:
            since_ts = self._changelog.get_last_ts()

        changes = self._changelog.read_new(since_ts, exclude_instance=self._config.instance_id)
        if not changes:
            return 0

        applied = 0
        skipped = 0
        merged = 0

        for change in changes:
            try:
                data = change.get("data", {})
                memory_id = data.get("memory_id", "")
                remote_vc = VectorClock.from_dict(change.get("vc", {}))

                # 若提供了本地查询函数，执行向量时钟冲突检测
                if get_local_fn and memory_id:
                    local_record = get_local_fn(memory_id)
                    if local_record:
                        local_vc = VectorClock.from_dict(
                            json.loads(local_record.get("vc", "{}"))
                            if isinstance(local_record.get("vc"), str)
                            else local_record.get("vc", {})
                        )
                        cmp = local_vc.compare(remote_vc)

                        if cmp == 1:
                            # 本地版本更新，跳过远程变更
                            skipped += 1
                            continue
                        elif cmp == 0 and local_vc != remote_vc:
                            # 并发冲突，需要合并
                            if self._config.conflict_resolution == "latest_wins":
                                memory_type = data.get("type", "fact")
                                merged_data = merge_records(
                                    local_record, data, memory_type=memory_type
                                )
                                change["data"] = merged_data
                                merged += 1
                            else:
                                # manual 模式：记录冲突但不自动合并，跳过
                                logger.warning(
                                    "SyncEngine: conflict detected for %s in manual mode, skipping",
                                    memory_id,
                                )
                                skipped += 1
                                continue

                if apply_fn(change):
                    applied += 1
            except Exception as e:
                logger.warning("SyncEngine: failed to apply change: %s", e)

        if applied > 0 or merged > 0 or skipped > 0:
            logger.info(
                "SyncEngine: applied=%d merged=%d skipped=%d changes from other instances",
                applied,
                merged,
                skipped,
            )

        # 定期清理旧日志
        if self._changelog:
            self._changelog.trim()

        return applied

    def get_instance_info(self) -> dict[str, Any]:
        """获取当前实例信息。"""
        info: dict[str, Any] = {
            "instance_id": self._config.instance_id,
            "instance_name": self._config.instance_name,
            "sync_mode": self._config.mode,
        }
        if self._file_lock:
            info["file_lock_stats"] = self._file_lock.stats()
        return info

    def get_active_instances(self) -> list[dict[str, Any]]:
        """获取活跃实例列表。

        通过扫描 changelog 目录中的日志文件来发现其他实例。
        """
        instances = [
            {
                "instance_id": self._config.instance_id,
                "instance_name": self._config.instance_name,
                "is_self": True,
            }
        ]

        if self._changelog:
            changelog_dir = self._changelog._changelog_dir
            for log_file in changelog_dir.glob("*.jsonl"):
                other_id = log_file.stem
                if other_id != self._config.instance_id:
                    instances.append(
                        {
                            "instance_id": other_id,
                            "instance_name": f"omnimem-{other_id}",
                            "is_self": False,
                        }
                    )

        return instances

    def close(self) -> None:
        """关闭同步引擎。"""
        if self._file_lock:
            self._file_lock.close()
        self._unregister_instance()

    # ─── 实例注册 ─────────────────────────────────────────────

    def _register_instance(self) -> None:
        """注册当前实例。"""
        registry_path = self._data_dir / "instance_registry.json"
        registry = self._read_registry(registry_path)

        registry[self._config.instance_id] = {
            "name": self._config.instance_name,
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "mode": self._config.mode,
        }

        self._write_registry(registry_path, registry)

    def _unregister_instance(self) -> None:
        """注销当前实例。"""
        registry_path = self._data_dir / "instance_registry.json"
        registry = self._read_registry(registry_path)

        if self._config.instance_id in registry:
            del registry[self._config.instance_id]
            self._write_registry(registry_path, registry)

    @staticmethod
    def _read_registry(path: Path) -> dict[str, Any]:
        """读取实例注册表。"""
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    result: dict[str, Any] = json.load(f)
                    return result
            except Exception as e:
                logger.warning("SyncEngine: registry read failed: %s", e)
        return {}

    @staticmethod
    def _write_registry(path: Path, registry: dict[str, Any]) -> None:
        """写入实例注册表。"""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("SyncEngine: registry write failed: %s", e)
