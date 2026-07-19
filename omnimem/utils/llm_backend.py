from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    @abstractmethod
    def call(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str | None:
        ...

    @abstractmethod
    def call_sync(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Any:
        ...


class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ):
        from omnimem.utils.llm_client import AsyncLLMClient

        self._client = AsyncLLMClient(
            api_key=api_key or "",
            base_url=base_url or "",
            model=model,
        )

    def call(self, prompt, system=None, max_tokens=1024, temperature=0.7):
        result = self._client.call_sync(
            prompt=prompt, system=system or "", max_tokens=max_tokens, temperature=temperature
        )
        return result.content if result else None

    def call_sync(self, prompt, system=None, max_tokens=1024, temperature=0.7):
        return self._client.call_sync(
            prompt=prompt, system=system or "", max_tokens=max_tokens, temperature=temperature
        )


class OllamaBackend(LLMBackend):
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url

    def call(self, prompt, system=None, max_tokens=1024, temperature=0.7):
        import json
        import urllib.request

        data = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "system": system or "",
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                return result.get("response")
        except Exception as e:
            logger.warning("OllamaBackend call failed: %s", e)
            return None

    def call_sync(self, prompt, system=None, max_tokens=1024, temperature=0.7):
        content = self.call(prompt, system, max_tokens, temperature)
        if content is None:
            return None
        return type("Result", (), {"content": content})()


class AnthropicBackend(LLMBackend):
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self._api_key = api_key
        self._model = model

    def call(self, prompt, system=None, max_tokens=1024, temperature=0.7):
        import json
        import urllib.request

        data = json.dumps(
            {
                "model": self._model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system or "You are a helpful assistant.",
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key or "",
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                return result.get("content", [{}])[0].get("text")
        except Exception as e:
            logger.warning("AnthropicBackend call failed: %s", e)
            return None

    def call_sync(self, prompt, system=None, max_tokens=1024, temperature=0.7):
        content = self.call(prompt, system, max_tokens, temperature)
        if content is None:
            return None
        return type("Result", (), {"content": content})()


def create_llm_backend(backend: str = "openai", **kwargs) -> LLMBackend:
    if backend == "openai":
        return OpenAIBackend(**kwargs)
    elif backend == "ollama":
        return OllamaBackend(**kwargs)
    elif backend == "anthropic":
        return AnthropicBackend(**kwargs)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")
