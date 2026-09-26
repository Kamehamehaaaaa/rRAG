from pathlib import Path
from typing import List
from vector_db.chunk import Chunk
from embedding.registry import get
import numpy as np
from loaders.registry import get_loader
from captioning import AbstractCaptioner

def make_chunks_from_dir(folder: Path,
                         max_sentences_per_chunk: int = 6,
                         max_chars: int | None = None,
                         captioner: AbstractCaptioner | None = None) -> List[Chunk]:
    if not folder.is_dir():
        return get_loader(folder, captioner=captioner).load(folder, chunk_size=max_sentences_per_chunk, max_chars=max_chars)
    all_chunks = []
    for file in folder.rglob('*'):
        if not file.is_file():
            continue
        try:
            loader = get_loader(file, captioner=captioner)
        except ValueError:
            continue  # unsupported extension
        all_chunks.extend(loader.load(file, max_sentences_per_chunk, max_chars=max_chars))
    return all_chunks

def embed_chunks(chunks: List[Chunk], embedder, batch_size: int = 64) -> List[Chunk]:
    if embedder is None:
        embedder = get("sentence_transformer")
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c.text for c in batch]
        vectors = embedder.embed_batch(texts)
        if len(vectors) != len(batch):
            raise ValueError(f"embedder returned {len(vectors)} vectors for {len(batch)} texts")
        for chunk, vec in zip(batch, vectors):
            chunk.vector = np.asarray(vec, dtype="float32")
    return chunks
