# serve/app.py
"""
Query-serving app. Owns the single IndexCache instance for the process's
lifetime. Threat model: single process (see cache.py docstring) — do not
run this under multiple worker processes without revisiting that.
"""

from contextlib import asynccontextmanager
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config
from embedding.registry import get
from vector_db.index_store.cache import IndexCache

_cache: IndexCache | None = None
_embedder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cache, _embedder
    if not config.MANIFEST.exists():
        raise RuntimeError(
            f"no index found at {config.MANIFEST} — run ingest/run_ingest.py build first."
        )
    _cache = IndexCache(config.INDEX_ROOT, config.MANIFEST)
    _embedder = get(config.EMBEDDING_MODEL)  # must match what built the index — see note below
    _cache.get()  # warm the cache at startup rather than on first request
    yield


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    text: str
    top_k: int = 5


class RetrievedChunk(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    distance: float


class QueryResponse(BaseModel):
    results: List[RetrievedChunk]
    generation: int


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    index, meta = _cache.get()

    query_vector = np.array(_embedder.embed(req.text), dtype="float32").reshape(1, -1)
    if query_vector.shape[1] != index.d:
        raise HTTPException(
            status_code=500,
            detail=f"query embedding dim {query_vector.shape[1]} does not match index dim {index.d}",
        )

    k = min(req.top_k, index.ntotal)
    if k == 0:
        return QueryResponse(results=[], generation=_cache._gen_cached or 0)

    distances, indices = index.search(query_vector, k)

    results = [
        RetrievedChunk(
            doc_id=meta[i]["doc_id"],
            chunk_id=meta[i]["chunk_id"],
            text=meta[i]["text"],
            distance=float(d),
        )
        for d, i in zip(distances[0], indices[0])
        if i != -1  # FAISS pads with -1 if fewer than k results exist
    ]
    return QueryResponse(results=results, generation=_cache._gen_cached or 0)


@app.get("/health")
def health():
    if _cache is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    index, meta = _cache.get()
    return {"status": "ok", "generation": _cache._gen_cached, "vectors": index.ntotal, "meta_count": len(meta)}