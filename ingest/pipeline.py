from pathlib import Path

from utils import make_chunks_from_dir, embed_chunks
from embedding.registry import get
from vector_db.index_store.faiss_index import persist_index, update_index
from vector_db.chunk import Chunk

from embedding import AbstractEmbedder


def _prepare_chunks(data_dir: Path, embedder: AbstractEmbedder, chunk_size: int) -> list[Chunk]:
    chunks = make_chunks_from_dir(data_dir, chunk_size)
    return embed_chunks(chunks, embedder)


def data_pipeline(
    data_dir: Path,
    index_root: Path,
    manifest_path: Path,
    embedder: AbstractEmbedder,
    chunk_size: int = 6,
) -> int:
    """First-ever build. Returns the generation number written (0)."""
    chunks = _prepare_chunks(data_dir, embedder, chunk_size)
    gen = persist_index(chunks, index_root, manifest_path)
    print(f"Indexed {len(chunks)} chunks (gen={gen})")
    return gen


def add_data(
    data_path: Path,
    index_root: Path,
    manifest_path: Path,
    embedder: AbstractEmbedder,
    chunk_size: int = 6,
) -> int:
    """Append new data. Returns the new generation number."""
    chunks = _prepare_chunks(data_path, embedder, chunk_size)
    gen = update_index(chunks, index_root, manifest_path)
    print(f"Added {len(chunks)} chunks (gen={gen})")
    return gen