from pathlib import Path
from typing import List
from vector_db.chunk import Chunk
from embedding.registry import get
import numpy as np

def split_into_chunks(text: str,
                      max_sentences: int = 6,
                      delimiter: str = '\n') -> List[str]:
    paragraphs = [p for p in text.split(delimiter) if p.strip()]
    chunks = []
    buffer = []
    for p in paragraphs:
        buffer.append(p.strip())
        if len(buffer) >= max_sentences:
            chunks.append(' '.join(buffer))
            buffer = []
    if buffer:
        chunks.append(' '.join(buffer))
    return chunks

def make_chunks_from_dir(folder: Path,
                         max_sentences_per_chunk: int = 6) -> List[Chunk]:
    all_chunks = []
    for txt_file in folder.glob('*.txt'):
        raw = txt_file.read_text(encoding='utf-8')
        chunks = split_into_chunks(raw,
                                 max_sentences=max_sentences_per_chunk,
                                 delimiter='\n')
        for idx, sent in enumerate(chunks, start=1):
            all_chunks.append(Chunk(doc_id=txt_file.stem,
                                    chunk_id=idx,
                                    text=sent))
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
