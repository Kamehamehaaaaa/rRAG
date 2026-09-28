from knowledge_store.index_store.cache import IndexCache
from knowledge_store.graph_store.graph_cache import GraphCache
from pathlib import Path
from embedding.registry import get
from ingest.pipeline import add_data as _add_data

from .rrf import reciprocal_rank_fusion

from typing import List, Dict, Any


class SemanticSearch:
    def __init__(self, index_root: Path, manifest_path: Path, embedding: str = "sentence_transformer"):
        self.index_root = index_root
        self.manifest_path = manifest_path
        self.index_cache = IndexCache(index_root, manifest_path)
        self.index_cache.get()
        self.graph_cache = GraphCache(index_root, manifest_path)
        self.graph_cache.get()
        self.embedder = get(embedding) 

    def add_data(self, data_path: Path, chunk_size: int = 6, 
                captioner_name: str = "local_vlm", graph_extractor_name: str = None):
        gen = _add_data(
            data_path, 
            self.index_root, 
            self.manifest_path, 
            self.embedder.name, 
            chunk_size,
            captioner_name,
            graph_extractor_name)
        self.index_cache.get()
        return gen

    def search(self, query, top_k=5):
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        
        index, meta = self.index_cache.get()
        k = min(top_k, index.ntotal)
        if k == 0:
            return []

        query_vector = self.embedder.embed([query]).astype('float32')

        distances, indices = index.search(query_vector, k)
        results = [
            {
                "doc_id": meta[i]["doc_id"],
                "chunk_id": meta[i]["chunk_id"],
                "text": meta[i]["text"],
                "distance": float(distances[0][j]),
            }
            for j, i in enumerate(indices[0])
        ]
        return results

    def hybrid_search(self, query: str, top_k: int = 5, graph_hops: int = 2) -> List[Dict[str, Any]]:
        vector_hits = self.search(query, top_k=top_k * 3)
        vector_ranked_positions = [h["position"] for h in vector_hits]
        graph_ranked_positions = self.graph_cache.expand(vector_ranked_positions, hops=graph_hops)

        fused = reciprocal_rank_fusion(
            [str(p) for p in vector_ranked_positions],
            [str(p) for p in graph_ranked_positions],
        )
        top_positions = [int(p) for p, _ in fused[:top_k]]
        return [c for pos in top_positions if (c := self.index_cache.get_chunk(pos)) is not None]


    def get_context(self, query: str, top_k: int = 5, max_chars: int = 3000) -> str:
        results = self.search(query, top_k=top_k)
        blocks, total = [], 0
        for r in results:
            block = f"[{r['doc_id']}#{r['chunk_id']}]\n{r['text']}"
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)
        return "\n\n".join(blocks)