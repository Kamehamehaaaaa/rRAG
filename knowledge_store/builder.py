import logging

import faiss
from chunk import Chunk
from typing import List, Optional
import numpy as np
import json
from pathlib import Path
from manifest import _write_generation, _write_manifest, _read_generation
from paths import next_gen_number, new_gen_dir, current_gen_dir, gc_old_generations

from .graph_store.schema import GraphData
from .graph_store.utils import merge_extraction_into_graph, save_graph, load_graph
from .graph_store import AbstractGraphExtractor

logger = logging.getLogger(__name__)

def _stack_vectors(chunks: List[Chunk]) -> np.ndarray:
    if not chunks:
        raise ValueError("cannot build/update an index from an empty chunk list")
    dims = {c.vector.shape[0] for c in chunks}
    if len(dims) > 1:
        raise ValueError(f"chunks have mixed embedding dimensions: {sorted(dims)}")
    return np.vstack([c.vector for c in chunks]).astype("float32")

def build_faiss_index(chunks: List[Chunk]) -> faiss.IndexFlatL2:
    """
    Build a FAISS index from a list of Chunk objects.

    Args:
        chunks (List[Chunk]): A list of Chunk objects containing text and vector embeddings.

    Returns:
        faiss.IndexFlatL2: A FAISS index built from the provided chunks.
    """
    # Extract vectors from chunks
    # vectors = [chunk.vector for chunk in chunks]
    # vectors = np.array(vectors).astype('float32')
    vectors = _stack_vectors(chunks)
    
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    index.add(vectors)
    
    return index

def _chunk_meta(chunks: List[Chunk]) -> List[dict]:
    return [
        {
            "doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text,
            "modality": c.modality,
            "image_path": str(c.image_path) if c.image_path else None,
        }
        for c in chunks
    ]

def _extract_graph(chunks: List[Chunk], extractor: AbstractGraphExtractor,
                    base: GraphData, base_offset: int = 0) -> GraphData:
    for position, chunk in enumerate(chunks, start=base_offset):
        if chunk.text.strip():
            merge_extraction_into_graph(base, extractor.extract(chunk.text), position)
    return base

def persist_index(
        chunks: List[Chunk], 
        index_root: Path, 
        manifest_path: Path, 
        graph_extractor: Optional[AbstractGraphExtractor] = None):
    gen = next_gen_number(manifest_path)
    target_dir = new_gen_dir(index_root, gen)

    index = build_faiss_index(chunks)
    faiss.write_index(index, str(target_dir / "index.faiss"))

    with open(target_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(_chunk_meta(chunks), f, indent=2, ensure_ascii=False)

    if graph_extractor is not None:
        # base_offset=0 by default: meta.json is empty on the first build,
        # so positions start at 0.
        save_graph(_extract_graph(chunks, graph_extractor, GraphData()), target_dir)


    _write_generation(manifest_path, gen)
    logger.info("index built: gen=%d, vectors=%d", gen, index.ntotal)
    return gen


def update_index(
        chunks: List[Chunk], 
        index_root: Path, 
        manifest_path: Path,
        graph_extractor: Optional[AbstractGraphExtractor] = None):
    new_vectors = _stack_vectors(chunks)

    src_dir = current_gen_dir(index_root, manifest_path)
    index = faiss.read_index(str(src_dir / "index.faiss"))
    if index.d != new_vectors.shape[1]:
        raise ValueError(
            f"existing index has dim {index.d}, new vectors have dim {new_vectors.shape[1]}"
        )
    with open(src_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    base_offset = len(meta)

    index.add(new_vectors)
    meta.extend(_chunk_meta(chunks))

    new_gen = next_gen_number(manifest_path)
    dst_dir = new_gen_dir(index_root, new_gen)
    faiss.write_index(index, str(dst_dir / "index.faiss"))
    with open(dst_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if graph_extractor is not None:
        graph = load_graph(src_dir)  # carry the accumulated graph forward
        save_graph(_extract_graph(chunks, graph_extractor, graph, base_offset=base_offset), dst_dir)


    _write_generation(manifest_path, new_gen)  # atomic pointer swap — last step
    logger.info("index updated: gen=%d, vectors=%d", new_gen, index.ntotal)

    removed = gc_old_generations(index_root, manifest_path, keep=3)
    if removed:
        logger.info("gc'd old generations: %s", [str(p) for p in removed])

    return new_gen


# --- If ingestion and serving ever become separate OS processes ---
# Wrap the bodies of persist_index / update_index in:
#
#   from filelock import FileLock
#   with FileLock(str(manifest_path) + ".lock"):
#       ...
#
# to serialize concurrent writers. Not needed for a single process calling
# these functions sequentially.