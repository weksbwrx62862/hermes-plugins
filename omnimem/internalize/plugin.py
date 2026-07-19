"""L4 内化记忆插件系统 — 将 KVCache 和 LoRA 解耦为可选插件。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from omnimem.utils.experimental import experimental_class

logger = logging.getLogger(__name__)


@experimental_class
class InternalizationPlugin(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def initialize(self, config: Any, storage_dir: Any) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class KVCachePlugin(InternalizationPlugin):
    def __init__(self) -> None:
        self._manager: Any = None

    def name(self) -> str:
        return "kv_cache"

    def is_available(self) -> bool:
        return True

    def initialize(self, config: Any, storage_dir: Any) -> None:
        from omnimem.internalize.kv_cache import KVCacheManager

        self._manager = KVCacheManager(
            storage_dir,
            auto_preload_threshold=config.get("kv_cache_threshold", 10),
            max_cache_size=config.get("kv_cache_max", 100),
        )

    def close(self) -> None:
        if self._manager:
            self._manager.close()


class LoRAPlugin(InternalizationPlugin):
    def __init__(self) -> None:
        self._trainer: Any = None

    def name(self) -> str:
        return "lora"

    def is_available(self) -> bool:
        # Always available: runs in simulation mode when peft is not installed
        return True

    def initialize(self, config: Any, storage_dir: Any) -> None:
        from omnimem.internalize.lora_train import LoRATrainer

        self._trainer = LoRATrainer(
            storage_dir,
            base_model=config.get("lora_base_model", "Qwen2.5-7B"),
            lora_rank=config.get("lora_rank", 16),
            lora_alpha=config.get("lora_alpha", 32),
        )

    def close(self) -> None:
        if self._trainer:
            self._trainer.close()


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, InternalizationPlugin] = {}

    def register(self, plugin: InternalizationPlugin) -> None:
        if plugin.is_available():
            self._plugins[plugin.name()] = plugin
        else:
            logger.warning("Plugin %s skipped: dependencies not available", plugin.name())

    def get(self, name: str) -> InternalizationPlugin | None:
        return self._plugins.get(name)

    def initialize_all(self, config: Any, storage_dir: Any) -> None:
        for plugin in self._plugins.values():
            try:
                plugin.initialize(config, storage_dir)
            except Exception:
                logger.warning("Failed to initialize plugin %s", plugin.name(), exc_info=True)

    def close_all(self) -> None:
        for plugin in self._plugins.values():
            try:
                plugin.close()
            except Exception:
                logger.warning("Failed to close plugin %s", plugin.name(), exc_info=True)
