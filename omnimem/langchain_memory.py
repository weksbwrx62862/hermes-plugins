from __future__ import annotations

from typing import Any


class OmniMemLangChainMemory:
    def __init__(self, storage_dir: str | None = None, config: dict | None = None):
        from omnimem.sdk import OmniMemSDK

        self._sdk = OmniMemSDK(storage_dir=storage_dir, config=config)
        self.buffer = ""

    @property
    def memory_variables(self):
        return ["chat_history", "omnimem_context"]

    def load_memory_variables(self, inputs: dict) -> dict:
        query = inputs.get("input", "")
        if query:
            result = self._sdk.recall(query=query, mode="rag")
            context = result.get("result", "") if isinstance(result, dict) else ""
        else:
            context = ""
        return {"chat_history": self.buffer, "omnimem_context": context}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        human = inputs.get("input", "")
        ai = outputs.get("output", "")
        if human:
            self._sdk.memorize(content=human, memory_type="fact")
        self.buffer += f"Human: {human}\nAI: {ai}\n"

    def clear(self) -> None:
        self.buffer = ""

    def close(self) -> None:
        self._sdk.close()
