# concytec-rag

Chatbot RAG (vectorial + grafo) sobre la comunidad **"1. Publicaciones y eventos institucionales"** del [repositorio institucional de CONCYTEC](https://repositorio.concytec.gob.pe/communities/8e1992ce-1102-4cbf-a985-7af02bb6c039).

## Estructura

- `indexer/` — scripts Python (uv) de recolección e indexación
  - `harvest.py` — inventario de metadatos vía la API REST de DSpace 7 (`data/metadata.csv` / `.json`)
  - `ingest.py` — (próximamente) descarga de PDFs, extracción por página e indexación con LightRAG
- `apps/api/` — (próximamente) servicio de consulta FastAPI + LightRAG
- `apps/web/` — (próximamente) chat Next.js + AI SDK (interfaz en español, modos vector/grafo)
- `rag_storage/` — índice local de LightRAG (no versionado)

## Uso

```bash
cd indexer
uv run harvest.py
```

Por ahora todo corre en local; el despliegue (Vercel) y el almacenamiento gestionado quedan para una fase posterior.
