"""异步非阻塞路由决策包装器。

将 model-router 的同步 ``_route()`` 逻辑包装为可在 asyncio 事件循环中
await 的 API，避免网关主事件循环被评分、模型池构建、嵌入推理等同步
计算阻塞。

主要组件：
  - AsyncRouter: 提供 ``async def route()`` 入口。
  - 内部通过 ``asyncio.to_thread`` 将同步调用提交到线程池。
  - 可选集成 ``RouteCache`` 与 ``EmbedTaskClassifier``，并暴露其异步接口。

用法示例::

    router = AsyncRouter(
        route_cache=RouteCache(capacity=256, ttl=60.0),
        embed_classifier=EmbedTaskClassifier(lazy=True),
    )
    result = await router.route(
        query="帮我优化这段 Python 代码",
        strategy="auto",
        messages=[{"role": "user", "content": "..."}],
        context={"dev_stage": "build"},
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AsyncRouter:
    """model-router 的异步非阻塞包装器。

    参数:
        executor: 可选的线程池执行器。未指定时使用 ``asyncio`` 默认线程池
            （可通过 ``asyncio.to_thread`` 提交任务）。
        route_cache: 可选的 ``RouteCache`` 实例，用于缓存完整路由结果。
        embed_classifier: 可选的 ``EmbedTaskClassifier`` 实例，用于异步语义
            任务分类与 embedding 编码。
    """

    def __init__(
        self,
        executor: Optional[ThreadPoolExecutor] = None,
        route_cache: Optional[Any] = None,
        embed_classifier: Optional[Any] = None,
    ) -> None:
        self.executor = executor
        self.route_cache = route_cache
        self.embed_classifier = embed_classifier

        # 动态加载 plugins/model-router（目录名含连字符，无法直接 import）
        self._model_router = self._load_model_router()

    @staticmethod
    def _load_model_router() -> Any:
        """通过 importlib 加载 model-router 插件模块。"""
        import importlib.util
        import pathlib
        import sys

        # 若已通过 sys.modules 加载则直接复用
        mod = sys.modules.get("plugins.model-router")
        if mod is not None:
            return mod

        path = pathlib.Path(__file__).resolve().parent / "__init__.py"
        spec = importlib.util.spec_from_file_location("plugins.model-router", str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载 plugins/model-router/__init__.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["plugins.model-router"] = mod
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        """估算文本 token 数（与 __init__.py 逻辑对齐的简化版）。"""
        if not text:
            return 0
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
        cn_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        other_chars = len(text) - cn_chars
        return int(cn_chars * 1.5 + other_chars * 0.5)

    @classmethod
    def _estimate_tokens(cls, messages: list[dict[str, Any]]) -> int:
        """估算消息列表 token 数量。"""
        total = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                total += cls._estimate_text_tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total += cls._estimate_text_tokens(part["text"])
        return total

    @staticmethod
    def _cache_key(query: str, strategy: str, context: dict[str, Any]) -> str:
        """构造缓存 key，综合查询、策略与关键上下文。"""
        canonical = json.dumps(
            {"query": query, "strategy": strategy, "context": context},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def _run_in_thread(self, func, *args, **kwargs) -> Any:
        """在线程池中执行同步可调用对象。"""
        if self.executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))
        return await asyncio.to_thread(func, *args, **kwargs)

    async def classify(self, query: str, timeout: Optional[float] = 30.0) -> tuple[str, int, float]:
        """异步执行语义任务分类（若分类器不可用则返回空信息）。

        返回:
            (task_type, complexity, confidence)
        """
        if self.embed_classifier is None:
            return "", 0, 0.0

        def _do_classify():
            return self.embed_classifier.classify(query, timeout=timeout)

        try:
            return await self._run_in_thread(_do_classify)
        except Exception as exc:
            logger.warning("AsyncRouter 语义分类失败: %s", exc)
            return "", 0, 0.0

    async def encode(self, query: str) -> Optional[list[float]]:
        """异步编码查询为 embedding 向量（若分类器不可用则返回 None）。"""
        if self.embed_classifier is None:
            return None

        def _do_encode():
            vec = self.embed_classifier._encode(query)
            return vec.flatten().tolist()

        try:
            return await self._run_in_thread(_do_encode)
        except Exception as exc:
            logger.warning("AsyncRouter embedding 编码失败: %s", exc)
            return None

    async def _check_cache(self, query: str, strategy: str, context: dict[str, Any]) -> Optional[dict[str, Any]]:
        """异步查询路由缓存。"""
        if self.route_cache is None:
            return None

        key = self._cache_key(query, strategy, context)

        def _get():
            embedding = None
            if self.route_cache.semantic:
                # 语义缓存需要 embedding；同步获取后传入
                try:
                    vec = self.embed_classifier._encode(query) if self.embed_classifier else None
                    if vec is not None:
                        embedding = vec.flatten().tolist()
                except Exception:
                    embedding = None
            return self.route_cache.get(key, embedding=embedding)

        try:
            cached = await self._run_in_thread(_get)
            if cached and cached.get("routing_result"):
                return dict(cached["routing_result"])
            return None
        except Exception as exc:
            logger.warning("AsyncRouter 缓存查询失败: %s", exc)
            return None

    async def _store_cache(
        self,
        query: str,
        strategy: str,
        context: dict[str, Any],
        result: dict[str, Any],
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """异步存储路由结果到缓存。"""
        if self.route_cache is None:
            return

        key = self._cache_key(query, strategy, context)

        def _set():
            embedding = None
            if self.route_cache.semantic:
                try:
                    vec = self.embed_classifier._encode(query) if self.embed_classifier else None
                    if vec is not None:
                        embedding = vec.flatten().tolist()
                except Exception:
                    embedding = None
            value = {
                "complexity": result.get("complexity", 3),
                "task_type": result.get("task_type", "simple_qa"),
                "strategy": result.get("strategy", strategy),
                "model_scores_sorted": list(result.get("score_breakdown", {}).items()),
                "routing_result": result,
            }
            self.route_cache.set(key, value, embedding=embedding, config=config)

        try:
            await self._run_in_thread(_set)
        except Exception as exc:
            logger.warning("AsyncRouter 缓存写入失败: %s", exc)

    async def route(
        self,
        query: str,
        strategy: str = "auto",
        messages: Optional[list[dict[str, Any]]] = None,
        context: Optional[dict[str, Any]] = None,
        **route_kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """异步执行模型路由决策。

        参数:
            query: 当前用户查询文本。
            strategy: 路由策略（cheapest/fastest/smartest/auto）。
            messages: 完整消息列表，用于长上下文 token 估算。
            context: 附加上下文，可包含 ``dev_stage``、``cache_hit_rate`` 等。
            **route_kwargs: 透传给 ``_route()`` 的额外参数（如 ``force_mimo``、
                ``force_deepseek``）。

        返回:
            与 ``_route()`` 返回格式一致的字典，包含选中的模型、Provider、
            降级链和决策上下文；模型池为空时返回 None。
        """
        context = context or {}
        messages = messages or []

        # 1. 尝试命中缓存
        cached = await self._check_cache(query, strategy, context)
        if cached is not None:
            logger.debug("AsyncRouter 缓存命中: %s", query[:40])
            cached["from_cache"] = True
            return cached

        # 2. 可选：异步语义分类（用于上下文 enrich，不强制覆盖 _route 内部逻辑）
        embed_task_type, embed_complexity, embed_confidence = await self.classify(query)
        if embed_task_type:
            route_kwargs.setdefault("dev_stage", context.get("dev_stage", ""))
            # 将语义分类结果写入决策上下文，供调用方使用
            route_kwargs["_embed_context"] = {
                "task_type": embed_task_type,
                "complexity": embed_complexity,
                "confidence": embed_confidence,
            }

        # 3. token 估算（长上下文绕路）
        estimated_tokens = route_kwargs.pop("estimated_tokens", None)
        if not estimated_tokens:
            estimated_tokens = self._estimate_tokens(messages)
        if not estimated_tokens and query:
            estimated_tokens = self._estimate_text_tokens(query)

        # 4. 在线程池中执行同步 _route
        def _do_route():
            return self._model_router._route(
                query,
                strategy,
                estimated_tokens=estimated_tokens or 0,
                cache_hit_rate=float(context.get("cache_hit_rate", 0.0)),
                **{k: v for k, v in route_kwargs.items() if not k.startswith("_")},
            )

        result = await self._run_in_thread(_do_route)
        if result is None:
            return None

        # 5.  enrich 语义分类上下文（若存在）
        if "_embed_context" in route_kwargs:
            result["embed_context"] = route_kwargs.pop("_embed_context")

        # 6. 写入缓存
        await self._store_cache(query, strategy, context, result)

        result["from_cache"] = False
        return result


# 暴露一个默认实例，方便直接 await 调用
default_async_router: Optional[AsyncRouter] = None
_default_router_lock = asyncio.Lock()


async def get_default_async_router(
    route_cache: Optional[Any] = None,
    embed_classifier: Optional[Any] = None,
) -> AsyncRouter:
    """获取/创建默认 AsyncRouter 单例。"""
    global default_async_router
    async with _default_router_lock:
        if default_async_router is None:
            default_async_router = AsyncRouter(
                route_cache=route_cache,
                embed_classifier=embed_classifier,
            )
        return default_async_router
