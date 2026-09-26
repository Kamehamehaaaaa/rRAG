from . import AbstractLoader
from captioning import AbstractCaptioner
from vector_db.chunk import Chunk

from typing import List, Optional
from pathlib import Path

class ImageLoader(AbstractLoader):
    def __init__(self, captioner: AbstractCaptioner):
        self.captioner = captioner

    def load(self, path: Path, chunk_size: int) -> List[Chunk]:
        context = self._sibling_text(path)
        caption = self.captioner.caption(path, context)
        return [
            Chunk(
                doc_id=path.stem, 
                chunk_id="0", 
                text=caption, 
                modality="image",
                source_path=path, 
                image_path=path
            )
        ]

    @staticmethod
    def _sibling_text(path: Path) -> Optional[str]:
        sibling = path.with_suffix(".txt")
        if sibling.is_file():
            return sibling.read_text(encoding="utf-8").strip() or None
        return None