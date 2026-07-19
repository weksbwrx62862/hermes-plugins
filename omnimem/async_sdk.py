"""AsyncOmniMemSDK — 异步 SDK，提供原生 async/await 接口。

内部使用 OmniMemSDK 实例 + asyncio.to_thread() 包装同步操作，
为 asyncio 事件循环提供非阻塞的记忆操作接口。

用法:
    from omnimem.async_sdk import AsyncOmniMemSDK

    async with AsyncOmniMemSDK(storage_dir="~/.omnimem") as sdk:
        await sdk.memorize("用户喜欢Python", memory_type="preference")
        result = await sdk.recall("用户喜欢什么")
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from omnimem.sdk import OmniMemSDK

logger = logging.getLogger(__name__)


class AsyncOmniMemSDK:
    """异步 SDK — 原生 async/await 接口。

    内部持有 OmniMemSDK 实例，所有同步操作通过 asyncio.to_thread()
    包装为异步调用，避免阻塞事件循环。
    """

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._sdk = OmniMemSDK(storage_dir=storage_dir, config=config)

    async def memorize(self, content: str, memory_type: str = "fact", **kwargs: Any) -> dict:
        """异步存储记忆。"""
        args = {"content": content, "memory_type": memory_type, **kwargs}
        raw = await asyncio.to_thread(self._sdk._provider_style_memorize, args)
        return json.loads(raw)

    async def recall(self, query: str, mode: str = "rag", **kwargs: Any) -> dict:
        """异步检索记忆。"""
        args = {"query": query, "mode": mode, **kwargs}
        raw = await asyncio.to_thread(self._sdk._provider_style_recall, args)
        return json.loads(raw)

    async def reflect(self, query: str, **kwargs: Any) -> dict:
        """异步深层反思。"""
        args = {"query": query, **kwargs}
        raw = await asyncio.to_thread(self._sdk._provider_style_reflect, args)
        return json.loads(raw)

    async def govern(self, action: str, **kwargs: Any) -> dict:
        """异步治理操作。"""
        args = {"action": action, **kwargs}
        raw = await asyncio.to_thread(self._sdk._provider_style_govern, args)
        return json.loads(raw)

    async def compact(self, **kwargs: Any) -> dict:
        """异步压缩前准备。"""
        raw = await asyncio.to_thread(self._sdk.compact, **kwargs)
        return raw

    async def detail(self, memory_id: str, **kwargs: Any) -> dict:
        """异步按需拉取记忆细节。"""
        return await asyncio.to_thread(self._sdk.detail, memory_id, **kwargs)

    async def health_check(self) -> dict:
        """异步健康检查。"""
        return await asyncio.to_thread(self._sdk.health_check)

    async def close(self) -> None:
        """异步关闭 SDK，释放资源。"""
        await asyncio.to_thread(self._sdk.close)

    async def __aenter__(self) -> AsyncOmniMemSDK:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
