"""Harvest metadata for all items in the CONCYTEC community
"1. Publicaciones y eventos institucionales" via the DSpace 7 REST API.

Outputs:
  data/metadata.json  - one record per item (full detail, incl. all ORIGINAL bitstreams)
  data/metadata.csv   - one row per item (primary PDF flattened)

Usage:
  uv run harvest.py
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

import httpx

BASE = "https://repositorio.concytec.gob.pe/server/api"
COMMUNITY = "8e1992ce-1102-4cbf-a985-7af02bb6c039"
PAGE_SIZE = 100
DATA_DIR = Path(__file__).parent / "data"

client = httpx.Client(timeout=60, headers={"User-Agent": "concytec-rag-harvester/0.1"})


def md(item: dict, field: str, first: bool = True):
    """Read a metadata field from a DSpace item; first value or all values."""
    values = [v["value"] for v in item.get("metadata", {}).get(field, [])]
    if first:
        return values[0] if values else None
    return values


def fetch_page(page: int) -> dict:
    r = client.get(
        f"{BASE}/discover/search/objects",
        params={
            "scope": COMMUNITY,
            "dsoType": "item",
            "size": PAGE_SIZE,
            "page": page,
            "embed": ["bundles/bitstreams/format", "owningCollection", "metrics"],
        },
    )
    r.raise_for_status()
    return r.json()


def bitstream_record(bs: dict) -> dict:
    fmt = bs.get("_embedded", {}).get("format") or {}
    return {
        "name": bs.get("name"),
        "size_bytes": bs.get("sizeBytes"),
        "mimetype": fmt.get("mimetype"),
        "download_url": bs.get("_links", {}).get("content", {}).get("href"),
    }


def parse_metrics(emb: dict) -> dict:
    """CRIS stored metrics: snapshot of the view/download counters shown in the UI."""
    metrics = (emb.get("metrics") or {}).get("_embedded", {}).get("metrics", [])
    out = {"views": None, "downloads": None, "metrics_date": None}
    for m in metrics:
        key = {"view": "views", "download": "downloads"}.get(m.get("metricType"))
        if key and m.get("metricCount") is not None:
            out[key] = int(m["metricCount"])
            out["metrics_date"] = (m.get("acquisitionDate") or "")[:10] or None
    return out


def parse_item(obj: dict) -> dict:
    item = obj["_embedded"]["indexableObject"]
    emb = item.get("_embedded", {})

    original, text = [], []
    bundles = (emb.get("bundles") or {}).get("_embedded", {}).get("bundles", [])
    for bundle in bundles:
        bits = (bundle.get("_embedded") or {}).get("bitstreams", {})
        bits = (bits.get("_embedded") or {}).get("bitstreams", [])
        if bundle["name"] == "ORIGINAL":
            original = [bitstream_record(b) for b in bits]
        elif bundle["name"] == "TEXT":
            text = [bitstream_record(b) for b in bits]

    date_issued = md(item, "dc.date.issued") or ""
    owning = emb.get("owningCollection") or {}

    pdfs = [
        b
        for b in original
        if b["mimetype"] == "application/pdf"
        or (b["name"] or "").lower().endswith(".pdf")
    ]
    non_pdfs = [b for b in original if b not in pdfs]

    return {
        "uuid": item["uuid"],
        "handle_url": md(item, "dc.identifier.uri") or f"https://hdl.handle.net/{item.get('handle')}",
        "title": item.get("name"),
        "authors": md(item, "dc.contributor.author", first=False),
        "date_issued": date_issued,
        "year": date_issued[:4] if date_issued else None,
        "type": md(item, "dc.type"),
        "language": md(item, "dc.language.iso"),
        "collection": owning.get("name"),
        "abstract": md(item, "dc.description.abstract"),
        "subjects": md(item, "dc.subject", first=False),
        "subject_ocde": md(item, "dc.subject.ocde", first=False),
        "publisher": md(item, "dc.publisher"),
        "rights": md(item, "dc.rights"),
        "rights_uri": md(item, "dc.rights.uri"),
        "sponsorship": md(item, "dc.description.sponsorship", first=False),
        **parse_metrics(emb),
        "pdfs": pdfs,
        "non_pdf_files": non_pdfs,
        "text_bitstreams": text,  # DSpace pre-extracted full text, may help scanned PDFs
    }


def human_type(dc_type: str | None) -> str:
    """'info:eu-repo/semantics/book' -> 'book'."""
    return (dc_type or "unknown").rsplit("/", 1)[-1]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    items: list[dict] = []
    page = 0
    while True:
        data = fetch_page(page)
        result = data["_embedded"]["searchResult"]
        objects = result["_embedded"]["objects"]
        items.extend(parse_item(o) for o in objects)
        total_pages = result["page"]["totalPages"]
        print(f"page {page + 1}/{total_pages}: {len(items)} items so far")
        page += 1
        if page >= total_pages:
            break
        time.sleep(0.5)

    with open(DATA_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    with open(DATA_DIR / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["uuid", "year", "type", "collection", "title", "handle_url",
             "views", "downloads", "subjects", "publisher", "rights_uri", "sponsorship",
             "pdf_count", "pdf_names", "pdf_mb", "pdf_urls", "non_pdf_count", "has_text_bundle"]
        )
        for it in items:
            pdf_mb = sum(b["size_bytes"] or 0 for b in it["pdfs"]) / 1e6
            writer.writerow(
                [it["uuid"], it["year"], human_type(it["type"]), it["collection"],
                 it["title"], it["handle_url"],
                 it["views"], it["downloads"],
                 "; ".join(it["subjects"] or []), it["publisher"], it["rights_uri"],
                 "; ".join(it["sponsorship"] or []),
                 len(it["pdfs"]),
                 "; ".join(b["name"] or "" for b in it["pdfs"]), f"{pdf_mb:.1f}",
                 "; ".join(b["download_url"] or "" for b in it["pdfs"]),
                 len(it["non_pdf_files"]), bool(it["text_bitstreams"])]
            )

    # --- summary -------------------------------------------------------------
    with_pdf = [it for it in items if it["pdfs"]]
    total_mb = sum(b["size_bytes"] or 0 for it in with_pdf for b in it["pdfs"]) / 1e6

    print(f"\nTotal items: {len(items)}")
    print(f"Items with >=1 PDF: {len(with_pdf)} (total {total_mb:,.0f} MB)")
    print(f"Items without PDF: {len(items) - len(with_pdf)}")

    by_year_type: Counter[tuple[str, str]] = Counter()
    mb_by_year: Counter[str] = Counter()
    for it in with_pdf:
        year = it["year"] or "????"
        by_year_type[(year, human_type(it["type"]))] += 1
        mb_by_year[year] += sum(b["size_bytes"] or 0 for b in it["pdfs"]) / 1e6

    types = sorted({t for (_, t) in by_year_type})
    print(f"\n{'year':<6}" + "".join(f"{t[:12]:>14}" for t in types) + f"{'total':>8}{'MB':>10}")
    for year in sorted(mb_by_year):
        counts = [by_year_type.get((year, t), 0) for t in types]
        print(f"{year:<6}" + "".join(f"{c:>14}" for c in counts)
              + f"{sum(counts):>8}{mb_by_year[year]:>10,.0f}")


if __name__ == "__main__":
    main()
