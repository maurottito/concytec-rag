"""FastAPI retrieval service over the LightRAG index in ../../rag_storage.

Retrieval only: returns the assembled context (with [Doc: ... | Página N]
markers) for a question; answer generation happens in apps/web. Must run with
the same provider config the index was built with (indexer/.env).

Run:  uv run uvicorn main:app --port 8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "indexer"))  # shared provider config
load_dotenv(ROOT / "indexer" / ".env")

from providers import build_rag, resolve_provider  # noqa: E402

WORKING_DIR = ROOT / "rag_storage"

rag = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    provider, cfg = resolve_provider()
    if not cfg["api_key"]:
        raise RuntimeError(f"Falta {cfg['api_key_env']} en indexer/.env")
    if not WORKING_DIR.exists():
        raise RuntimeError("No existe rag_storage/ — ejecuta primero el indexado")

    rag = await build_rag(str(WORKING_DIR), provider, cfg)
    yield
    await rag.finalize_storages()


app = FastAPI(title="concytec-rag API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    mode: Literal["naive", "local", "global", "hybrid", "mix"] = "naive"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query(req: QueryRequest):
    from lightrag import QueryParam

    if not req.question.strip():
        raise HTTPException(400, "Pregunta vacía")
    context = await rag.aquery(
        req.question,
        param=QueryParam(mode=req.mode, only_need_context=True),
    )
    return {"mode": req.mode, "context": context or ""}
