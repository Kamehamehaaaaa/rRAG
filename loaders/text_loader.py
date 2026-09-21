# loaders/pdf_loader.py
from pathlib import Path
from typing import List
from pypdf import PdfReader
from vector_db.chunk import Chunk
from . import AbstractLoader
from .text_utils import split_into_chunks

class TextLoader(AbstractLoader):
    def load(self, path: Path, chunk_size: int, max_chars: int | None = None) -> List[Chunk]:
        text = path.read_text(encoding='utf-8')
        chunks = []
        if not text.strip():
            return []
        for i, piece in enumerate(split_into_chunks(text=text, max_sentences=chunk_size, max_chars=max_chars)):
            chunks.append(Chunk(
                            doc_id=path.stem, 
                            chunk_id=f"i",
                            text=piece
            ))
        return chunks