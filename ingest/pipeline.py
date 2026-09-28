from pathlib import Path

from captioning import AbstractCaptioner
from ingest.utils import make_chunks_from_dir, embed_chunks
from embedding.registry import get
from knowledge_store.builder import persist_index, update_index
from chunk import Chunk

from embedding.abstract_embedding import AbstractEmbedding

from captioning.registry import get as get_captioner
from knowledge_store.graph_store.registry import get as get_extractor

FALLBACK_MAX_CHARS = 1000
CHARS_PER_TOKEN_ESTIMATE = 4  # rough estimate, varies by language and model

def _resolve_max_chars(embedder) -> int:
    if embedder.max_seq_length is None:
        return FALLBACK_MAX_CHARS
    # 10% margin for [CLS]/[SEP] and the estimate is rough
    return int(embedder.max_seq_length * CHARS_PER_TOKEN_ESTIMATE * 0.9)


def _prepare_chunks(data_dir: Path, embedder: AbstractEmbedding, chunk_size: int, captioner: AbstractCaptioner | None = None) -> list[Chunk]:
    max_chars = _resolve_max_chars(embedder)
    chunks = make_chunks_from_dir(data_dir, chunk_size, max_chars, captioner=captioner)
    return embed_chunks(chunks, embedder)


def data_pipeline(
    data_dir: Path,
    index_root: Path,
    manifest_path: Path,
    embedder: AbstractEmbedding,
    chunk_size: int = 6,
    captioner_name: str = "local_vlm",
    graph_extractor_name: str = "local_ollama"
) -> int:
    """First-ever build. Returns the generation number written (0)."""
    captioner = get_captioner(captioner_name)
    gextractor = get_extractor(graph_extractor_name)
    chunks = _prepare_chunks(data_dir, embedder, chunk_size, captioner)
    gen = persist_index(chunks, index_root, manifest_path, graph_extractor=gextractor)
    print(f"Indexed {len(chunks)} chunks (gen={gen})")
    return gen


def add_data(
    data_path: Path,
    index_root: Path,
    manifest_path: Path,
    embedder: AbstractEmbedding,
    chunk_size: int = 6,
    captioner_name: str = "local_vlm",
    graph_extractor_name: str = "local_ollama"
) -> int:
    """Append new data. Returns the new generation number."""
    captioner = get_captioner(captioner_name)
    gextractor = get_extractor(graph_extractor_name)
    chunks = _prepare_chunks(data_path, embedder, chunk_size, captioner)
    gen = update_index(chunks, index_root, manifest_path, graph_extractor=gextractor)
    print(f"Added {len(chunks)} chunks (gen={gen})")
    return gen