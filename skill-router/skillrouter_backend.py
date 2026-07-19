"""
SkillRouter Backend — Qwen3-0.6B based embedding + reranking

Provides an alternative backend for the skill-router plugin using:
  - SR-Emb-0.6B: Embedding model (last_token_pool + L2 normalize)
  - SR-Rank-0.6B: Reranker model (logit("yes") - logit("no"))

All inference runs on CPU with torch.float32, lazy-loaded on first use.
"""

import logging
import os
import threading
from typing import Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Instruction prefix for query encoding
_QUERY_PREFIX = (
    "Instruct: Given a task description, retrieve the most relevant "
    "skill document that would help an agent complete the task\nQuery:"
)

# Reranker prompt template
_RERANK_INSTRUCT = (
    "Given a task description, judge whether the skill document is "
    "relevant and useful for completing the task"
)

# System prompt for reranker (Qwen3 thinking mode)
_RERANK_SYSTEM = (
    "You are a helpful assistant that evaluates document relevance. "
    "Think step by step, then answer with only 'yes' or 'no'."
)


def _last_token_pool(last_hidden_state, attention_mask):
    """Pool by taking the last token's hidden state (left-padded sequences)."""
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device), seq_lengths
    ]


def _l2_normalize(embeddings):
    """L2-normalize embeddings along the last dimension."""
    return torch.nn.functional.normalize(embeddings, p=2, dim=-1)


class SkillRouterBackend:
    """SkillRouter backend using Qwen3-0.6B models for embedding and reranking.

    Models are lazy-loaded on first use. All inference runs on CPU with float32.
    Thread-safe via locks.
    """

    def __init__(self, emb_model_path: str, rank_model_path: str):
        self._emb_model_path = os.path.expanduser(emb_model_path)
        self._rank_model_path = os.path.expanduser(rank_model_path)

        # Lazy-loaded components
        self._emb_model = None
        self._emb_tokenizer = None
        self._rank_model = None
        self._rank_tokenizer = None

        # Thread locks
        self._emb_lock = threading.Lock()
        self._rank_lock = threading.Lock()

        self._loaded = False

    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded

    def _load_embedding_model(self):
        """Load the embedding model (lazy, thread-safe)."""
        if self._emb_model is not None:
            return

        with self._emb_lock:
            if self._emb_model is not None:
                return

            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                logger.info("Loading SkillRouter embedding model: %s", self._emb_model_path)
                self._emb_tokenizer = AutoTokenizer.from_pretrained(
                    self._emb_model_path, trust_remote_code=True
                )
                self._emb_model = AutoModel.from_pretrained(
                    self._emb_model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
                self._emb_model.eval()
                logger.info("SkillRouter embedding model loaded successfully")
            except Exception as e:
                logger.error("Failed to load SkillRouter embedding model: %s", e)
                self._emb_model = None
                self._emb_tokenizer = None
                raise

    def _load_ranker_model(self):
        """Load the reranker model (lazy, thread-safe)."""
        if self._rank_model is not None:
            return

        with self._rank_lock:
            if self._rank_model is not None:
                return

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info("Loading SkillRouter reranker model: %s", self._rank_model_path)
                self._rank_tokenizer = AutoTokenizer.from_pretrained(
                    self._rank_model_path, trust_remote_code=True
                )
                self._rank_model = AutoModelForCausalLM.from_pretrained(
                    self._rank_model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
                self._rank_model.eval()
                logger.info("SkillRouter reranker model loaded successfully")
            except Exception as e:
                logger.error("Failed to load SkillRouter reranker model: %s", e)
                self._rank_model = None
                self._rank_tokenizer = None
                raise

    def _ensure_loaded(self):
        """Ensure both models are loaded."""
        self._load_embedding_model()
        self._load_ranker_model()
        self._loaded = True

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Batch encode document texts, return L2-normalized embeddings.

        Args:
            texts: List of document texts in format "name | description | body"

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        import torch

        self._load_embedding_model()
        assert self._emb_model is not None and self._emb_tokenizer is not None

        all_embeddings = []
        batch_size = 8  # Process in small batches for CPU

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self._emb_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=4096,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self._emb_model(**encoded)
                embeddings = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
                embeddings = _l2_normalize(embeddings)

            all_embeddings.append(embeddings.float().cpu().numpy())

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query with the instruction prefix.

        Args:
            query: The search query text

        Returns:
            numpy array of shape (1, embedding_dim)
        """
        import torch

        self._load_embedding_model()
        assert self._emb_model is not None and self._emb_tokenizer is not None

        prefixed_query = f"{_QUERY_PREFIX} {query}"
        encoded = self._emb_tokenizer(
            [prefixed_query],
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._emb_model(**encoded)
            embedding = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embedding = _l2_normalize(embedding)

        return embedding.float().cpu().numpy()

    def rerank(self, query: str, candidates: List[Dict], skills: Dict, top_k: int) -> List[Dict]:
        """Score each candidate with the reranker model, return sorted results.

        Args:
            query: The search query
            candidates: List of dicts with 'name' and 'score' keys
            skills: Dict mapping skill name to skill info
            top_k: Number of results to return

        Returns:
            List of candidates sorted by reranker score (descending)
        """
        import torch

        if not candidates:
            return []

        self._load_ranker_model()
        assert self._rank_model is not None and self._rank_tokenizer is not None

        scored = []
        for item in candidates:
            name = item["name"]
            skill = skills.get(name, {})
            desc = skill.get("description", "")[:500]
            body = skill.get("body_text", "")[:2000]

            document = f"{name} | {desc} | {body}"

            # Build the prompt using Qwen3 chat template
            user_content = (
                f"<Instruct>: {_RERANK_INSTRUCT}\n"
                f"<Query>: {query}\n"
                f"<Document>: {document}"
            )

            messages = [
                {"role": "system", "content": _RERANK_SYSTEM},
                {"role": "user", "content": user_content},
            ]

            text = self._rank_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )

            encoded = self._rank_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )

            # Get yes/no token ids
            yes_token_id = self._rank_tokenizer.encode("yes", add_special_tokens=False)[0]
            no_token_id = self._rank_tokenizer.encode("no", add_special_tokens=False)[0]

            with torch.no_grad():
                outputs = self._rank_model(**encoded)
                # Get logits for the last token
                last_logits = outputs.logits[0, -1, :]
                yes_logit = last_logits[yes_token_id].item()
                no_logit = last_logits[no_token_id].item()
                score = yes_logit - no_logit

            scored.append({
                **item,
                "rerank_score": score,
                "original_score": item["score"],
                "score": score,
            })

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.debug("SkillRouter reranker scored %d candidates, returning top %d", len(scored), top_k)
        return scored[:top_k]
