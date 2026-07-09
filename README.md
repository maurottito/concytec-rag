# concytec-rag

Chatbot RAG (vectorial + grafo) sobre la comunidad **"1. Publicaciones y eventos institucionales"** del [repositorio institucional de CONCYTEC](https://repositorio.concytec.gob.pe/communities/8e1992ce-1102-4cbf-a985-7af02bb6c039).

## Estructura

- `indexer/` — scripts Python (uv) de recolección e indexación
  - `harvest.py` — inventario de metadatos vía la API REST de DSpace 7 (`data/metadata.csv` / `.json`)
  - `ingest.py` — descarga de PDFs, extracción por página e indexación con LightRAG
  - `providers.py` — configuración de proveedor LLM/embeddings (Gemini gratis u OpenAI)
- `apps/api/` — servicio de consulta FastAPI + LightRAG (solo recuperación)
- `apps/web/` — chat Next.js + AI SDK (interfaz en español, modos vector/grafo, citas con página)
- `rag_storage/` — índice local de LightRAG (no versionado)

## Uso

```bash
# 1. Inventario de metadatos (sin API key)
cd indexer && uv run harvest.py

# 2. Indexación (requiere GEMINI_API_KEY u OPENAI_API_KEY en indexer/.env)
cd indexer && uv run ingest.py --years 2026

# 3. Servicio de consulta
cd apps/api && uv run uvicorn main:app --port 8000

# 4. Chat web (requiere apps/web/.env.local con GEMINI_API_KEY y RAG_API_URL)
cd apps/web && npm run dev   # http://localhost:3000
```

Por ahora todo corre en local; el despliegue (Vercel) y el almacenamiento gestionado quedan para una fase posterior.
