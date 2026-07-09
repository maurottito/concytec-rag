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

import asyncio
import os
import time

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
        "max_gleaning": 0,  # 1 extraction call per chunk; raise for graph quality at ~2x cost
        # embeddings: 40K TPM / 100 RPM / 2K requests-per-DAY on this account
        # (a batch counts as ONE request, unlike Gemini) — TPM is the binding
        # limit, so batches of 16 x ~1200-token chunks pace against it.
        "embed_batch": 16,
        "embed_max_async": 1,
        # tier 1 allows 500 RPM but only 200K TPM; extraction calls average
        # ~3K tokens, so 50/min keeps worst case near 165K TPM.
        "llm_rpm": 50,
        # tier 1 embeddings: 1M TPM / 3K RPM (confirmed via response headers);
        # 400K leaves ample margin for the rough char-based token estimate.
        "embed_tpm": 400_000,
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
        # WARNING: Gemini counts EVERY TEXT in a batch as one embedding
        # request against its 100/min and 1,000/DAY free quotas, so batching
        # saves nothing and any real corpus exhausts the daily cap. Use
        # EMBED_PROVIDER=openai for actual indexing; this stays only for
        # completeness. Google also meters on a sliding minute window, so
        # pacing must be burst-free.
        "embed_batch": 8,
        "embed_max_async": 1,
        "llm_rpm": 12,  # limit is 15/min; leave headroom for retries
        "embed_tpm": 20_000,  # limit is 30K/min; char-based estimate is rough
    },
}


def resolve_provider() -> tuple[str, dict]:
    name = os.environ.get("LLM_PROVIDER") or (
        "gemini" if os.environ.get("GEMINI_API_KEY") else "openai"
    )
    cfg = dict(PROVIDERS[name])
    cfg["llm_model"] = os.environ.get("LLM_MODEL", cfg["llm_model"])
    cfg["api_key"] = os.environ.get(cfg["api_key_env"])

    # EMBED_PROVIDER splits embeddings off to another provider (e.g. free
    # Gemini LLM + paid OpenAI embeddings — the Gemini free tier counts every
    # text in a batch against its 1K/day embedding request quota, which makes
    # it unusable beyond toy corpora). The index must always be queried with
    # the same embedding model it was built with.
    embed_name = os.environ.get("EMBED_PROVIDER", name)
    if embed_name != name:
        e = PROVIDERS[embed_name]
        for k in ("embed_model", "embed_dim", "embed_price", "embed_batch",
                  "embed_max_async", "embed_tpm"):
            cfg[k] = e[k]
        cfg["embed_base_url"] = e["base_url"]
        cfg["embed_api_key"] = os.environ.get(e["api_key_env"])
    else:
        cfg["embed_base_url"] = cfg["base_url"]
        cfg["embed_api_key"] = cfg["api_key"]
    cfg["embed_provider"] = embed_name
    return name, cfg


def make_llm_func(provider: str, cfg: dict, token_tracker=None):
    from lightrag.llm.openai import openai_complete_if_cache

    # 1 request per (60/rpm)s instead of AsyncLimiter(rpm, 60): the latter
    # allows an initial burst of `rpm` instant requests, which lands ~2x the
    # cap inside Google's first minute window and triggers 429 retry storms.
    limiter = AsyncLimiter(1, 60 / cfg["llm_rpm"])

    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        async with limiter:
            return await openai_complete_if_cache(
                cfg["llm_model"], prompt, system_prompt=system_prompt,
                history_messages=history_messages, base_url=cfg["base_url"],
                api_key=cfg["api_key"], token_tracker=token_tracker, **kwargs,
            )

    return llm_model_func


def _token_pacer(tpm: float):
    """Even, burst-free pacing: each call reserves a time slot sized by its
    estimated tokens (~3 chars/token for Spanish — overestimates, which is the
    safe direction). A 1s floor between requests also keeps the request rate
    under 60 RPM when batches are tiny (entity names during graph building).
    Rate-limit windows slide, so bursts breach them even at a legal average
    rate — this pacer never bursts."""
    rate = tpm / 60  # tokens per second
    next_slot = 0.0
    lock = asyncio.Lock()

    async def pace(texts: list[str]) -> None:
        nonlocal next_slot
        est_tokens = sum(len(t) // 3 + 16 for t in texts)
        async with lock:
            now = time.monotonic()
            start = max(now, next_slot)
            next_slot = start + max(est_tokens / rate, 1.0)
            wait = start - now
        if wait > 0:
            await asyncio.sleep(wait)

    return pace


def make_embedding_func(provider: str, cfg: dict, token_tracker=None):
    from lightrag.utils import EmbeddingFunc

    pace = _token_pacer(cfg["embed_tpm"])

    if cfg["embed_provider"] == "gemini":
        from google import genai

        client = genai.Client(api_key=cfg["embed_api_key"])

        async def embed(texts: list[str]) -> np.ndarray:
            await pace(texts)
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

        async def embed(texts: list[str]) -> np.ndarray:
            await pace(texts)
            return await openai_embed(
                texts, model=cfg["embed_model"], base_url=cfg["embed_base_url"],
                api_key=cfg["embed_api_key"], token_tracker=token_tracker,
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
