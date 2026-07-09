"""LLM/embedding provider config shared by the indexer and apps/api.

The query service MUST use the same provider/embedding model the index was
built with — embedding spaces are incompatible across models.

Gemini specifics learned the hard way:
- The free tier enforces 15 requests/MINUTE on the LLM, so calls go through an
  AsyncLimiter window, not just a concurrency cap.
- The OpenAI-compatible /embeddings endpoint returns a wrong vector count for
  batched inputs, so embeddings use the native google-genai SDK instead.
"""

from __future__ import annotations

import os

import numpy as np
from aiolimiter import AsyncLimiter

PROVIDERS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "llm_model": "gpt-4.1-mini",
        "embed_model": "text-embedding-3-small",
        "embed_dim": 1536,
        "llm_price": (0.40, 1.60),  # USD per 1M tokens (input, output)
        "embed_price": 0.02,
        "max_async": 4,
        "max_gleaning": 1,
        "embed_batch": 32,
        "embed_max_async": 8,
        "llm_rpm": 480,
        "embed_rpm": 100,
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        # On this account's free tier only gemini-3.1-flash-lite has a usable
        # daily quota (15 RPM / 500 RPD); 2.5/3/3.5 flash are capped at 20 RPD.
        "llm_model": "gemini-3.1-flash-lite",
        "embed_model": "gemini-embedding-001",  # free: 100 RPM / 30K TPM / 1K RPD
        "embed_dim": 3072,
        "llm_price": (0.0, 0.0),  # free tier
        "embed_price": 0.0,
        "max_async": 2,
        "max_gleaning": 0,  # 1 extraction call per chunk — halves quota burn
        # 1K embedding requests/DAY is the scarcest quota, so batches are big
        # (16 x ~1200-token chunks ≈ 19K tokens/request) and pacing is done by
        # a token bucket against the 30K TPM limit, not a request counter.
        "embed_batch": 16,
        "embed_max_async": 1,
        "llm_rpm": 12,  # limit is 15/min; leave headroom for retries
        "embed_tpm": 24_000,
    },
}


def resolve_provider() -> tuple[str, dict]:
    name = os.environ.get("LLM_PROVIDER") or (
        "gemini" if os.environ.get("GEMINI_API_KEY") else "openai"
    )
    cfg = dict(PROVIDERS[name])
    cfg["llm_model"] = os.environ.get("LLM_MODEL", cfg["llm_model"])
    cfg["api_key"] = os.environ.get(cfg["api_key_env"])
    return name, cfg


def make_llm_func(provider: str, cfg: dict, token_tracker=None):
    from lightrag.llm.openai import openai_complete_if_cache

    limiter = AsyncLimiter(cfg["llm_rpm"], 60)

    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        async with limiter:
            return await openai_complete_if_cache(
                cfg["llm_model"], prompt, system_prompt=system_prompt,
                history_messages=history_messages, base_url=cfg["base_url"],
                api_key=cfg["api_key"], token_tracker=token_tracker, **kwargs,
            )

    return llm_model_func


def make_embedding_func(provider: str, cfg: dict, token_tracker=None):
    from lightrag.utils import EmbeddingFunc

    if provider == "gemini":
        from google import genai

        client = genai.Client(api_key=cfg["api_key"])
        # token bucket: rough estimate 3 chars/token for Spanish text
        token_bucket = AsyncLimiter(cfg["embed_tpm"], 60)

        async def embed(texts: list[str]) -> np.ndarray:
            est_tokens = min(sum(len(t) // 3 + 16 for t in texts), cfg["embed_tpm"])
            await token_bucket.acquire(est_tokens)
            res = await client.aio.models.embed_content(
                model=cfg["embed_model"], contents=texts,
            )
            if len(res.embeddings) != len(texts):
                raise ValueError(
                    f"Gemini devolvió {len(res.embeddings)} vectores para {len(texts)} textos"
                )
            return np.array([e.values for e in res.embeddings], dtype=np.float32)

    else:
        from lightrag.llm.openai import openai_embed

        limiter = AsyncLimiter(cfg["embed_rpm"], 60)

        async def embed(texts: list[str]) -> np.ndarray:
            async with limiter:
                return await openai_embed(
                    texts, model=cfg["embed_model"], base_url=cfg["base_url"],
                    api_key=cfg["api_key"], token_tracker=token_tracker,
                )

    return EmbeddingFunc(embedding_dim=cfg["embed_dim"], func=embed)


async def build_rag(working_dir: str, provider: str, cfg: dict,
                    llm_tracker=None, embed_tracker=None):
    """Create and initialize a LightRAG instance for this provider config."""
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_pipeline_status

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=make_llm_func(provider, cfg, llm_tracker),
        llm_model_name=cfg["llm_model"],
        llm_model_max_async=cfg["max_async"],
        entity_extract_max_gleaning=cfg["max_gleaning"],
        embedding_batch_num=cfg["embed_batch"],
        embedding_func_max_async=cfg["embed_max_async"],
        embedding_func=make_embedding_func(provider, cfg, embed_tracker),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag
