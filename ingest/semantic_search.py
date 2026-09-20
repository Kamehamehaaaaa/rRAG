from vector_db.index_store.cache import IndexCache
from pathlib import Path
from embedding.registry import get
from ingest.pipeline import add_data

class SemanticSearch:
    def __init__(self, index_root: Path, manifest_path: Path, embedding: str = "sentence_transformer"):
        self.index_root = index_root
        self.manifest_path = manifest_path
        self.index_cache = IndexCache(index_root, manifest_path)
        self.index_cache._ensure_loaded()
        self.embedder = get(embedding) 

    def add_data(self, data_path: Path, chunk_size: int = 6):
        add_data(data_path, self.index_root, self.manifest_path, self.embedder.name, chunk_size)
        self.index_cache._ensure_loaded()

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