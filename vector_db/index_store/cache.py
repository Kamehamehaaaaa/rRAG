from pathlib import Path
import faiss
import json
from typing import List, Tuple
from vector_db.index_store.manifest import _read_generation
from .paths import current_gen_dir, next_gen_number, new_gen_dir
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndexCache:
    def __init__(self, index_root: Path, manifest_path: Path):
        self.index_root = index_root
        self.manifest_path = manifest_path
        self._gen_cached: int | None = None
        self.index: faiss.IndexFlatL2 | None = None
        self.meta: List[dict] | None = None

    def _load(self) -> None:
        """Slow path: resolve current generation and read its files from disk."""
        gen_dir = current_gen_dir(self.index_root, self.manifest_path)
        idx = faiss.read_index(str(gen_dir / "index.faiss"))
        with open(gen_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        self.index = idx
        self.meta = meta
        self._gen_cached = _read_generation(self.manifest_path) or 0
        logger.info("index cache loaded: gen=%d, vectors=%d", self._gen_cached, idx.ntotal)

    def _ensure_loaded(self) -> None:
        if self.index is None or self.meta is None:
            self._load()
            return

        cur_gen = _read_generation(self.manifest_path)
        cur_gen = 0 if cur_gen is None else cur_gen
        cached_gen = 0 if self._gen_cached is None else self._gen_cached

        if cur_gen > cached_gen:
            self._load()

    def get(self) -> Tuple[faiss.IndexFlatL2, List[dict]]:
        self._ensure_loaded()
        return self.index, self.meta