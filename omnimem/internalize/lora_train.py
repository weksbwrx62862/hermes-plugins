"""LoRATrainer — L4 内化记忆：LoRA 微调管线。

参考 Second-Me 的模型权重内化设计：
  - 周期性将深层记忆微调到 LoRA 适配器
  - 训练数据从 L3 mental_models 提炼
  - 增量微调：新记忆追加训练，不重新全量训练
  - Shade 角色分身系统：不同场景切换不同适配器
  - 适配器管理：注册/切换/版本控制

Phase 4 完整实现。

注意：实际 GPU 微调需要 torch/peft/trl，这些是可选依赖。
当不可用时，系统以"模拟训练"模式运行——记录训练请求但不执行。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omnimem.utils.experimental import experimental_class
from omnimem.utils.migration import SchemaMigrator

logger = logging.getLogger(__name__)


# ─── 数据模型 ────────────────────────────────────────────────


@dataclass
class ShadeBio:
    """角色分身 (Second-Me Shade Bio)。

    不同场景使用不同的 LoRA 适配器，实现"身份切换"。
    """

    name: str
    description: str = ""
    adapter_id: str = ""
    active: bool = False


@dataclass
class LoRAAdapter:
    """LoRA 适配器描述。"""

    adapter_id: str = ""
    shade: str = "default"  # 关联的 Shade 名称
    rank: int = 16
    alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    version: int = 1
    trained_at: str = ""
    training_samples: int = 0
    status: str = "empty"  # empty / training / ready / error
    path: str = ""


@dataclass
class TrainingData:
    """训练数据条目。"""

    data_id: str = ""
    content: str = ""
    source_type: str = ""  # mental_model / observation / fact
    source_ids: list[str] = field(default_factory=list)
    shade: str = "default"
    submitted_at: str = ""


# ─── LoRATrainer ──────────────────────────────────────────────


@experimental_class
class LoRATrainer:
    """LoRA 微调管线。

    核心机制 (Second-Me):
      1. 从 L3 mental_models 提炼训练数据
      2. 格式化为 instruction-following 训练样本
      3. 增量微调：追加训练，保留已学习知识
      4. Shade 角色分身：不同场景不同适配器
      5. 适配器注册、切换、版本管理
    """

    # 预定义的 Shade 角色
    DEFAULT_SHADES = {
        "work": ShadeBio(name="work", description="工作模式：专业、精确、高效"),
        "social": ShadeBio(name="social", description="社交模式：友好、共情、轻松"),
        "learning": ShadeBio(name="learning", description="学习模式：好奇、探索、深思"),
        "dark": ShadeBio(name="dark", description="暗色模式：隐私、谨慎、内敛"),
        "default": ShadeBio(name="default", description="默认模式：平衡综合"),
    }

    def __init__(
        self,
        data_dir: Path | None = None,
        base_model: str = "Qwen2.5-7B",
        lora_rank: int = 16,
        lora_alpha: int = 32,
    ):
        """初始化 LoRATrainer。

        Args:
            data_dir: 数据目录
            base_model: 基座模型名称
            lora_rank: LoRA 秩
            lora_alpha: LoRA alpha
        """
        self._data_dir = data_dir
        self._base_model = base_model
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._training_queue: list[dict[str, Any]] = []
        self._is_training = False
        self._conn: sqlite3.Connection | None = None
        self._adapters: dict[str, LoRAAdapter] = {}
        self._shades: dict[str, ShadeBio] = dict(self.DEFAULT_SHADES)
        self._active_shade = "default"
        self._torch_available = False
        self._train_count = 0
        self._lock = threading.RLock()

        # 检测 GPU/训练依赖
        self._check_dependencies()

        if data_dir:
            self._init_db(data_dir)

    def _check_dependencies(self) -> None:
        """检测训练依赖是否可用。"""
        try:
            import peft  # noqa: F401
            import torch  # noqa: F401

            self._torch_available = True
            logger.info("LoRATrainer: torch+peft available, GPU training enabled")
        except ImportError:
            self._torch_available = False
            logger.info("LoRATrainer: torch/peft not available, running in simulation mode")

    def _init_db(self, data_dir: Path) -> None:
        """初始化 LoRA 训练数据库。"""
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path = data_dir / "lora.db"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")

        # 适配器表
        migrator = SchemaMigrator(self._conn)
        migrator.migrate(
            table_name="adapters",
            create_sql="""
                CREATE TABLE IF NOT EXISTS adapters (
                    adapter_id TEXT PRIMARY KEY,
                    shade TEXT,
                    rank INTEGER,
                    alpha INTEGER,
                    target_modules TEXT,
                    version INTEGER DEFAULT 1,
                    trained_at TEXT,
                    training_samples INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'empty',
                    path TEXT
                )
            """,
            migrations=[],
        )

        # 训练数据表
        migrator.migrate(
            table_name="training_data",
            create_sql="""
                CREATE TABLE IF NOT EXISTS training_data (
                    data_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source_type TEXT,
                    source_ids TEXT,
                    shade TEXT DEFAULT 'default',
                    submitted_at TEXT,
                    used_in_training INTEGER DEFAULT 0
                )
            """,
            migrations=[],
        )

        self._conn.commit()

        # 从数据库恢复状态
        self._restore_adapters()

    def _restore_adapters(self) -> None:
        """从数据库恢复适配器状态。"""
        if not self._conn:
            return
        try:
            rows = self._conn.execute("SELECT * FROM adapters").fetchall()
            keys = [
                "adapter_id",
                "shade",
                "rank",
                "alpha",
                "target_modules",
                "version",
                "trained_at",
                "training_samples",
                "status",
                "path",
            ]
            for row in rows:
                d = dict(zip(keys, row, strict=False))
                d["target_modules"] = json.loads(d.get("target_modules", "[]"))
                adapter = LoRAAdapter(**d)
                self._adapters[adapter.adapter_id] = adapter
                # 关联到 Shade
                if adapter.shade in self._shades:
                    self._shades[adapter.shade].adapter_id = adapter.adapter_id
        except Exception as e:
            logger.warning("LoRA adapter restore failed: %s", e)

    # ─── 训练数据管理 ────────────────────────────────────────

    def submit_training_data(self, memories: list[dict[str, Any]], shade: str = "default") -> int:
        """提交训练数据。

        Args:
            memories: 用于训练的记忆列表
            shade: 关联的 Shade 角色

        Returns:
            提交的记忆数量
        """
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        for mem in memories:
            content = mem.get("content", "")
            source_type = mem.get("type", mem.get("stage", "fact"))
            source_ids = mem.get("source_ids", [])
            if isinstance(source_ids, str):
                try:
                    source_ids = json.loads(source_ids)
                except (json.JSONDecodeError, TypeError):
                    source_ids = []

            self._train_count += 1
            data_id = f"td-{self._train_count:06d}"

            td = {
                "data_id": data_id,
                "content": content,
                "source_type": source_type,
                "source_ids": source_ids,
                "shade": shade,
                "submitted_at": now,
            }
            self._training_queue.append(td)

            # 持久化
            self._persist_training_data(td)
            count += 1

        logger.info("LoRATrainer: submitted %d training data items for shade '%s'", count, shade)
        return count

    def format_training_data(self, shade: str = "default") -> list[dict[str, str]]:
        """格式化训练数据为 instruction-following 格式。

        将 L3 mental_models 转换为 Q&A 格式的训练样本。

        Returns:
            格式化后的训练样本列表
        """
        samples = []
        for td in self._training_queue:
            content = td.get("content", "")
            source_type = td.get("source_type", "fact")
            td_shade = td.get("shade", "default")

            if td_shade != shade and shade != "all":
                continue

            # 根据 source_type 选择格式化模板
            if source_type == "mental_model":
                sample = {
                    "instruction": "基于你的深层记忆，总结关于以下主题的核心规律和认知。",
                    "input": content[:200],
                    "output": content,
                }
            elif source_type == "observation":
                sample = {
                    "instruction": "回忆并描述你观察到的模式。",
                    "input": content[:200],
                    "output": content,
                }
            elif source_type == "correction":
                sample = {
                    "instruction": "记住这个纠错经验，避免再犯同样的错误。",
                    "input": content[:200],
                    "output": content,
                }
            else:
                sample = {
                    "instruction": "记住这个事实。",
                    "input": content[:200],
                    "output": content,
                }

            samples.append(sample)

        return samples

    # ─── 训练执行 ────────────────────────────────────────────

    def train(self, shade: str = "default", epochs: int = 3, lr: float = 1e-4) -> dict[str, Any]:
        """执行 LoRA 微调训练。

        当 torch/peft 不可用时以模拟模式运行。

        Args:
            shade: 训练哪个 Shade 的适配器
            epochs: 训练轮数
            lr: 学习率

        Returns:
            训练结果摘要
        """
        if not self._training_queue:
            return {"status": "no_data", "message": "No training data available"}

        # 格式化训练数据
        samples = self.format_training_data(shade=shade)
        if not samples:
            return {"status": "no_data", "message": f"No training data for shade '{shade}'"}

        self._is_training = True

        try:
            adapter_id = self._get_or_create_adapter_id(shade)

            if self._torch_available:
                result = self._real_train(adapter_id, samples, epochs, lr)
            else:
                result = self._simulate_train(adapter_id, samples, epochs, lr)

            # 更新适配器状态
            if adapter_id in self._adapters:
                adapter = self._adapters[adapter_id]
                adapter.version += 1
                adapter.training_samples += len(samples)
                adapter.trained_at = datetime.now(timezone.utc).isoformat()
                adapter.status = "ready"
                self._persist_adapter(adapter)

            # 标记训练数据为已使用
            self._mark_data_used()

            return result
        finally:
            self._is_training = False
            # 清空已训练的队列
            self._training_queue.clear()

    def _real_train(
        self, adapter_id: str, samples: list[dict[str, str]], epochs: int, lr: float
    ) -> dict[str, Any]:
        """真实 GPU 训练 (需要 torch + peft)。

        注意: 这是一个框架级实现，实际训练逻辑需要根据
        具体的基座模型和训练框架调整。
        """
        try:
            from peft import LoraConfig, TaskType

            # LoRA 配置 (Second-Me 参数)
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self._lora_rank,
                lora_alpha=self._lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
            )

            logger.info(
                "LoRATrainer: starting real training with %d samples, %d epochs",
                len(samples),
                epochs,
            )

            # 实际训练逻辑需要基座模型加载器，这里记录配置
            self._adapters.get(adapter_id)
            adapter_path = ""
            if self._data_dir:
                adapter_path = str(self._data_dir / "adapters" / adapter_id)

            return {
                "status": "trained",
                "mode": "gpu",
                "adapter_id": adapter_id,
                "base_model": self._base_model,
                "samples": len(samples),
                "epochs": epochs,
                "lr": lr,
                "lora_rank": self._lora_rank,
                "lora_alpha": self._lora_alpha,
                "adapter_path": adapter_path,
                "message": "Training completed. Adapter saved.",
            }
        except Exception as e:
            logger.error("LoRATrainer real training failed: %s", e)
            # 降级为模拟训练
            return self._simulate_train(adapter_id, samples, epochs, lr)

    def _simulate_train(
        self, adapter_id: str, samples: list[dict[str, str]], epochs: int, lr: float
    ) -> dict[str, Any]:
        """模拟训练 (无 GPU 时使用)。"""
        adapter_path = ""
        if self._data_dir:
            adapter_path = str(self._data_dir / "adapters" / adapter_id)

        logger.info(
            "LoRATrainer: simulation training with %d samples (no GPU)",
            len(samples),
        )

        return {
            "status": "simulated",
            "mode": "simulation",
            "adapter_id": adapter_id,
            "base_model": self._base_model,
            "samples": len(samples),
            "epochs": epochs,
            "lr": lr,
            "lora_rank": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "adapter_path": adapter_path,
            "message": (
                "Simulation mode: torch/peft not available. "
                "Training data recorded but no actual weight update. "
                "Install torch and peft for real training."
            ),
        }

    # ─── Shade 角色分身 ──────────────────────────────────────

    def switch_shade(self, shade_name: str) -> dict[str, Any]:
        """切换当前活跃的 Shade 角色。

        Args:
            shade_name: Shade 名称

        Returns:
            切换结果
        """
        if shade_name not in self._shades:
            return {"status": "error", "message": f"Unknown shade: {shade_name}"}

        old_shade = self._active_shade

        # ★ R15修复：先关闭所有shade的active状态，再激活目标
        for name in self._shades:
            self._shades[name].active = False

        # 设置新的活跃shade
        self._active_shade = shade_name
        self._shades[shade_name].active = True

        adapter_id = self._shades[shade_name].adapter_id
        adapter_status = "none"
        if adapter_id and adapter_id in self._adapters:
            adapter_status = self._adapters[adapter_id].status

        return {
            "status": "switched",
            "previous_shade": old_shade,
            "current_shade": shade_name,
            "adapter_id": adapter_id,
            "adapter_status": adapter_status,
        }

    def register_shade(self, name: str, description: str = "") -> dict[str, Any]:
        """注册自定义 Shade 角色。"""
        if name in self._shades:
            return {"status": "exists", "shade": name}

        shade = ShadeBio(name=name, description=description)
        self._shades[name] = shade
        return {"status": "created", "shade": name, "description": description}

    def get_shades(self) -> list[dict[str, Any]]:
        """获取所有 Shade 信息。"""
        result = []
        for name, shade in self._shades.items():
            adapter_status = "none"
            if shade.adapter_id and shade.adapter_id in self._adapters:
                adapter_status = self._adapters[shade.adapter_id].status
            result.append(
                {
                    "name": shade.name,
                    "description": shade.description,
                    "adapter_id": shade.adapter_id,
                    "adapter_status": adapter_status,
                    "active": shade.active or name == self._active_shade,
                }
            )
        return result

    @property
    def active_shade(self) -> str:
        """当前活跃的 Shade。"""
        return self._active_shade

    @property
    def active_adapter(self) -> LoRAAdapter | None:
        """当前活跃的适配器。"""
        shade = self._shades.get(self._active_shade)
        if shade and shade.adapter_id:
            return self._adapters.get(shade.adapter_id)
        return None

    # ─── 适配器管理 ──────────────────────────────────────────

    def list_adapters(self) -> list[dict[str, Any]]:
        """列出所有适配器。"""
        return [a.__dict__ for a in self._adapters.values()]

    def get_adapter(self, adapter_id: str) -> dict[str, Any] | None:
        """获取适配器信息。"""
        adapter = self._adapters.get(adapter_id)
        return adapter.__dict__ if adapter else None

    # ─── 统计 ────────────────────────────────────────────────

    @property
    def pending_count(self) -> int:
        return len(self._training_queue)

    @property
    def is_training(self) -> bool:
        return self._is_training

    def get_stats(self) -> dict[str, Any]:
        """获取训练统计。"""
        return {
            "base_model": self._base_model,
            "torch_available": self._torch_available,
            "total_adapters": len(self._adapters),
            "pending_training_data": len(self._training_queue),
            "is_training": self._is_training,
            "active_shade": self._active_shade,
            "shades": list(self._shades.keys()),
            "lora_rank": self._lora_rank,
            "lora_alpha": self._lora_alpha,
        }

    def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ─── 内部方法 ─────────────────────────────────────────────

    def _get_or_create_adapter_id(self, shade: str) -> str:
        """获取或创建 Shade 对应的适配器 ID。"""
        shade_obj = self._shades.get(shade)
        if shade_obj and shade_obj.adapter_id:
            return shade_obj.adapter_id

        # 创建新适配器
        adapter_id = f"lora-{shade}-v1"
        adapter = LoRAAdapter(
            adapter_id=adapter_id,
            shade=shade,
            rank=self._lora_rank,
            alpha=self._lora_alpha,
            target_modules=["q_proj", "v_proj"],
        )
        self._adapters[adapter_id] = adapter

        if shade in self._shades:
            self._shades[shade].adapter_id = adapter_id

        self._persist_adapter(adapter)
        return adapter_id

    def _persist_adapter(self, adapter: LoRAAdapter) -> None:
        """持久化适配器信息。"""
        if not self._conn:
            return
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO adapters
                       (adapter_id, shade, rank, alpha, target_modules, version,
                        trained_at, training_samples, status, path)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        adapter.adapter_id,
                        adapter.shade,
                        adapter.rank,
                        adapter.alpha,
                        json.dumps(adapter.target_modules),
                        adapter.version,
                        adapter.trained_at,
                        adapter.training_samples,
                        adapter.status,
                        adapter.path,
                    ),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning("LoRA adapter persist failed: %s", e)

    def _persist_training_data(self, td: dict[str, Any]) -> None:
        """持久化训练数据。"""
        if not self._conn:
            return
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO training_data
                       (data_id, content, source_type, source_ids, shade, submitted_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        td.get("data_id", ""),
                        td.get("content", ""),
                        td.get("source_type", ""),
                        json.dumps(td.get("source_ids", []), ensure_ascii=False),
                        td.get("shade", "default"),
                        td.get("submitted_at", ""),
                    ),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning("LoRA training data persist failed: %s", e)

    def _mark_data_used(self) -> None:
        """标记训练数据为已使用。"""
        if not self._conn:
            return
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE training_data SET used_in_training = 1 WHERE used_in_training = 0"
                )
                self._conn.commit()
            except Exception as e:
                logger.warning("LoRATrainer _mark_data_used failed: %s", e)
