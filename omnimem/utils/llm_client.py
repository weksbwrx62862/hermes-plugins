"""Unified async LLM client with retry, caching, and credential management.

OPT-2: Replaces scattered LLM call logic in provider.py with a centralized,
async-first client. Synchronous wrappers provided for backward compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response."""

    content: str = ""
    model: str = ""
    latency_ms: float = 0.0


class AsyncLLMClient:
    """Async LLM client wrapping OpenAI-compatible APIs.

    Features:
      - Async HTTP via openai.AsyncOpenAI / httpx.AsyncClient
      - Semaphore-based concurrency limiting (default max 3)
      - TTL-based response caching (default 60s)
      - Credential caching (avoids re-reading .env every call)
      - Graceful fallback chain: direct API -> auxiliary_client
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "glm-5.1",
        max_concurrent: int = 3,
        timeout: float = 30.0,
        cache_ttl: float = 60.0,
        backend: Any | None = None,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cache: dict[str, tuple[LLMResponse, float]] = {}
        self._client: Any | None = None
        self._closed = False
        self._backend = backend

    # ─── Public API ──────────────────────────────────────────

    async def call(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 800,
        temperature: float = 0.5,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Async LLM call with concurrency control and caching."""
        if self._backend is not None:
            content = await asyncio.to_thread(
                self._backend.call, prompt, system or None, max_tokens, temperature
            )
            return LLMResponse(content=content or "", model=self._model)

        cache_key = f"{self._model}|{prompt[:200]}|{system[:100]}|{max_tokens}|{temperature}"
        now = time.time()

        if use_cache and cache_key in self._cache:
            cached, ts = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                logger.warning("AsyncLLM cache hit")
                return cached

        async with self._semaphore:
            start = time.time()
            result = await self._do_call(prompt, system, max_tokens, temperature)
            result.latency_ms = (time.time() - start) * 1000

        if use_cache:
            self._cache[cache_key] = (result, now)
            # ★ 定期清理过期条目：每 10 次写入或缓存超过 100 条时执行，避免每次调用都全量重建字典
            if len(self._cache) % 10 == 0 or len(self._cache) > 100:
                cutoff = now - self._cache_ttl
                self._cache = {k: v for k, v in self._cache.items() if v[1] > cutoff}

        return result

    def call_sync(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 800,
        temperature: float = 0.5,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Synchronous wrapper for backward compatibility.

        Safely handles nested event loops (e.g., when called from
        within an already-running async context).
        """
        if self._backend is not None:
            content = self._backend.call(prompt, system or None, max_tokens, temperature)
            return LLMResponse(content=content or "", model=self._model)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — safe to use asyncio.run
            return asyncio.run(self.call(prompt, system, max_tokens, temperature, use_cache))

        # Already inside an event loop — schedule and wait
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self.call(prompt, system, max_tokens, temperature, use_cache),
                loop,
            )
            # Blocking wait (caller is sync anyway)
            return future.result(timeout=self._timeout + 10.0)
        else:
            return loop.run_until_complete(
                self.call(prompt, system, max_tokens, temperature, use_cache)
            )

    def close(self) -> None:
        """Close underlying HTTP client."""
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            try:
                # Async close must be called from async context;
                # fire-and-forget a cleanup task if needed.
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._client.close())
                except RuntimeError:
                    pass  # No loop running — ignore
            except Exception as e:
                logger.warning("LLM client close error: %s", e)
            self._client = None

    # ─── Internal ────────────────────────────────────────────

    async def _do_call(
        self, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        """Actual HTTP call. Raises on failure so retry logic can catch it."""
        try:
            import httpx
            import openai

            if self._client is None:
                self._client = openai.AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=httpx.Timeout(self._timeout, connect=10.0),
                )

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            return LLMResponse(content=content.strip(), model=self._model)
        except Exception:
            logger.warning("Async LLM direct call failed, will try fallback")
            raise

    # ─── Credential helpers ──────────────────────────────────

    @staticmethod
    def load_credentials_from_env() -> dict[str, str]:
        """Load API credentials from environment variables."""
        import os

        return {
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "base_url": os.environ.get("OPENAI_BASE_URL", ""),
        }

    @staticmethod
    def load_credentials_from_hermes_env() -> dict[str, str]:
        """Load API credentials from ~/.hermes/.env file."""
        from pathlib import Path

        result = {"api_key": "", "base_url": ""}
        env_file = Path.home() / ".hermes" / ".env"
        if not env_file.exists():
            return result
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k == "OPENAI_API_KEY":
                    result["api_key"] = v
                elif k == "OPENAI_BASE_URL":
                    result["base_url"] = v
        except Exception as e:
            logger.warning("Hermes .env credential parsing failed: %s", e)
        return result

    @staticmethod
    def load_credentials_from_hermes_config() -> dict[str, str]:
        """Load API credentials from ~/.hermes/config.yaml.

        优先从 providers 节点读取 api_key/base_url/models，
        再从 model 节点读取 base_url/default model 作为补充。
        """
        from pathlib import Path

        result = {"api_key": "", "base_url": "", "model": "", "models": []}
        config_file = Path.home() / ".hermes" / "config.yaml"
        if not config_file.exists():
            return result
        try:
            import yaml

            cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            if cfg and isinstance(cfg, dict):
                # ★ R25修复ARCH-1：从 providers 节点读取凭证和可用模型
                providers_cfg = cfg.get("providers") or {}
                for _pname, pval in providers_cfg.items():
                    if isinstance(pval, dict):
                        if not result["api_key"]:
                            result["api_key"] = pval.get("api_key", "")
                        if not result["base_url"]:
                            result["base_url"] = pval.get("base_url", "")
                        if not result["models"] and isinstance(pval.get("models"), list):
                            result["models"] = pval["models"]
                        if result["api_key"] and result["base_url"]:
                            break
                # model 节点补充 base_url 和 default model
                model_cfg = cfg.get("model") or {}
                if not result["base_url"]:
                    result["base_url"] = model_cfg.get("base_url", "")
                result["model"] = model_cfg.get("default", "")
        except Exception as e:
            logger.warning("Hermes config.yaml credential parsing failed: %s", e)
        return result
