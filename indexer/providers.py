"""LLM/embedding provider config shared by the indexer and apps/api.

The query service MUST use the same provider/embedding model the index was
built with — embedding spaces are incompatible across models.
"""

from __future__ import annotations

import os

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
        "max_async": 2,  # stay under 15 RPM; retries handle 429s
        # 1 extraction call per chunk instead of 2 — halves the daily quota burn
        "max_gleaning": 0,
        # embeddings free tier is 30K TPM; 10 chunks x ~1200 tokens stays under
        "embed_batch": 10,
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
