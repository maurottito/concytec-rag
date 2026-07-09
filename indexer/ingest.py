"""Download PDFs for a year range, extract text per page and index with LightRAG.

Each page is prefixed with a provenance marker so retrieved chunks can be cited
with document name, handle link and page number (see CLAUDE.md):

    [Doc: <título> | Handle: <url> | Página N]

Usage:
  uv run ingest.py --years 2026            # download + extract + index
  uv run ingest.py --years 2026 --dry-run  # download + extract only, report sizes
  uv run ingest.py --years 2021-2026 --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF
import httpx
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
WORKING_DIR = Path(__file__).parent.parent / "rag_storage"

# Provider is chosen via LLM_PROVIDER in .env (default: gemini if GEMINI_API_KEY
# is set, else openai). Gemini free tier costs $0 but is rate-limited and Google
# may use free-tier content to improve its products (docs here are public/open).
from providers import resolve_provider  # noqa: E402

MIN_CHARS_PER_PAGE = 50  # below this average we assume a scanned/image PDF


def parse_years(spec: str) -> set[str]:
    if "-" in spec:
        a, b = spec.split("-")
        return {str(y) for y in range(int(a), int(b) + 1)}
    return {s.strip() for s in spec.split(",")}


def select_items(years: set[str], limit: int | None) -> list[dict]:
    items = json.load(open(DATA_DIR / "metadata.json", encoding="utf-8"))
    selected = [it for it in items if it["year"] in years and it["pdfs"]]
    selected.sort(key=lambda it: (it["year"], it["title"] or ""))
    return selected[:limit] if limit else selected


def download_pdfs(items: list[dict]) -> dict[str, list[Path]]:
    """Download every PDF of every item; returns item uuid -> local paths."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, list[Path]] = {}
    with httpx.Client(timeout=120, headers={"User-Agent": "concytec-rag/0.1"}, follow_redirects=True) as client:
        for it in items:
            paths[it["uuid"]] = []
            for k, pdf in enumerate(it["pdfs"]):
                dest = PDF_DIR / f"{it['uuid']}_{k}.pdf"
                paths[it["uuid"]].append(dest)
                if dest.exists() and dest.stat().st_size > 0:
                    continue
                print(f"  descargando {pdf['name']} ({(pdf['size_bytes'] or 0)/1e6:.1f} MB)")
                try:
                    with client.stream("GET", pdf["download_url"]) as r:
                        r.raise_for_status()
                        with open(dest, "wb") as f:
                            for chunk in r.iter_bytes(1 << 20):
                                f.write(chunk)
                except httpx.HTTPStatusError as e:
                    print(f"    NO DESCARGABLE (HTTP {e.response.status_code}): {pdf['name']}")
                    dest.unlink(missing_ok=True)
                    paths[it["uuid"]].remove(dest)
                except httpx.HTTPError as e:
                    print(f"    NO DESCARGABLE ({type(e).__name__}): {pdf['name']}")
                    dest.unlink(missing_ok=True)
                    paths[it["uuid"]].remove(dest)
    return paths


def extract_document(item: dict, pdf_paths: list[Path]) -> tuple[str | None, int, str]:
    """Return (marked_text, page_count, status) for one item; None text if unusable.

    The scanned-PDF check runs per file, so an item that mixes a scanned study
    with a text-based nota técnica keeps its usable files. Page numbers are per
    file and the marker names the file when the item has several PDFs.
    """
    title = (item["title"] or "Sin título").strip()
    handle = item["handle_url"]
    if not pdf_paths:
        return None, 0, "acceso restringido (401)"
    pages: list[str] = []
    kept_pages = 0
    scanned: list[str] = []
    for k, path in enumerate(pdf_paths):
        file_name = item["pdfs"][k]["name"] if k < len(item["pdfs"]) else path.name
        file_label = f" | Archivo: {file_name}" if len(pdf_paths) > 1 else ""
        file_pages: list[str] = []
        file_chars = 0
        with fitz.open(path) as doc:
            for page_no, page in enumerate(doc, 1):
                text = page.get_text("text").strip()
                file_chars += len(text)
                if text:
                    file_pages.append(
                        f"[Doc: {title}{file_label} | Handle: {handle} | Página {page_no}]\n{text}"
                    )
            n_pages = doc.page_count
        if n_pages and file_chars / n_pages < MIN_CHARS_PER_PAGE:
            scanned.append(file_name)
            continue
        pages.extend(file_pages)
        kept_pages += n_pages
    if not pages:
        return None, 0, f"escaneado: {'; '.join(scanned) or 'sin texto'}"
    status = "ok" if not scanned else f"ok (omite escaneados: {'; '.join(scanned)})"
    return "\n\n".join(pages), kept_pages, status


async def index_documents(docs: list[tuple[dict, str]], provider: str, cfg: dict) -> None:
    from lightrag.utils import TokenTracker

    from providers import build_rag

    llm_tracker = TokenTracker()
    embed_tracker = TokenTracker()
    rag = await build_rag(str(WORKING_DIR), provider, cfg, llm_tracker, embed_tracker)

    try:
        for i, (item, text) in enumerate(docs, 1):
            print(f"[{i}/{len(docs)}] indexando: {item['title'][:70]}")
            await rag.ainsert(text, ids=item["uuid"], file_paths=item["title"])
            print(f"    LLM acumulado: {llm_tracker}")
    finally:
        await rag.finalize_storages()

    llm, emb = llm_tracker.get_usage(), embed_tracker.get_usage()
    cost_llm = (llm["prompt_tokens"] * cfg["llm_price"][0]
                + llm["completion_tokens"] * cfg["llm_price"][1]) / 1e6
    cost_emb = emb["total_tokens"] * cfg["embed_price"] / 1e6
    print("\n=== Uso de tokens ===")
    print(f"Proveedor: {provider}")
    print(f"LLM ({cfg['llm_model']}): {llm}")
    print(f"Embeddings ({cfg['embed_model']}): {emb}")
    print(f"Costo LLM: ${cost_llm:.2f} | Costo embeddings: ${cost_emb:.4f} | TOTAL: ${cost_llm + cost_emb:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", required=True, help="ej. 2026, 2021-2026, 2024,2026")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="solo descarga y extrae, no indexa")
    args = ap.parse_args()

    load_dotenv(Path(__file__).parent / ".env")

    items = select_items(parse_years(args.years), args.limit)
    print(f"{len(items)} documentos seleccionados ({args.years})")
    paths = download_pdfs(items)

    docs: list[tuple[dict, str]] = []
    skipped: list[dict] = []
    total_pages = total_chars = 0
    for it in items:
        text, n_pages, status = extract_document(it, paths[it["uuid"]])
        if text is None:
            skipped.append({"uuid": it["uuid"], "title": it["title"], "year": it["year"],
                            "handle_url": it["handle_url"], "reason": status})
            print(f"  OMITIDO ({status}): {it['title'][:70]}")
            continue
        if status != "ok":
            print(f"  PARCIAL ({status}): {it['title'][:70]}")
        docs.append((it, text))
        total_pages += n_pages
        total_chars += len(text)

    if skipped:
        with open(DATA_DIR / "skipped.csv", "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["uuid", "title", "year", "handle_url", "reason"])
            if f.tell() == 0:
                w.writeheader()
            w.writerows(skipped)

    est_tokens = total_chars // 4  # rough: ~4 chars/token for Spanish
    print(f"\nExtraíbles: {len(docs)} docs, {total_pages} páginas, {total_chars:,} chars (~{est_tokens:,} tokens)")
    print(f"Omitidos: {len(skipped)} (ver data/skipped.csv)")

    if args.dry_run:
        return
    provider, cfg = resolve_provider()
    if not cfg["api_key"]:
        sys.exit(f"Falta {cfg['api_key_env']} en indexer/.env (proveedor: {provider})")
    if not cfg["embed_api_key"]:
        sys.exit(f"Falta la API key del proveedor de embeddings ({cfg['embed_provider']}) en indexer/.env")
    if not docs:
        sys.exit("No hay documentos indexables.")
    print(f"Proveedor: {provider} | LLM: {cfg['llm_model']} | Embeddings: {cfg['embed_model']}")
    asyncio.run(index_documents(docs, provider, cfg))


if __name__ == "__main__":
    main()
