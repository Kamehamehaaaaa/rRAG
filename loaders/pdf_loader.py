# loaders/pdf_loader.py
from pathlib import Path
from typing import List
from pypdf import PdfReader
from vector_db.chunk import Chunk
from . import AbstractLoader
from ingest.utils import split_into_chunks

class PDFLoader(AbstractLoader):
    def load(self, path: Path, chunk_size: int) -> List[Chunk]:
        reader = PdfReader(str(path))
        chunks = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            for i, piece in enumerate(split_into_chunks(text=text, max_sentences=chunk_size)):
                chunks.append(Chunk(
                                doc_id=path.stem, 
                                chunk_id=f"p{page_num}_c{i}",
                                text=piece
                ))
        return chunks