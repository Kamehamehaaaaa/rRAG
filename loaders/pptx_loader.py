# loaders/pptx_loader.py
from pathlib import Path
from typing import List
from pptx import Presentation
from vector_db.chunk import Chunk
from . import AbstractLoader
from ingest.utils import split_into_chunks

class PPTXLoader(AbstractLoader):
    def load(self, path: Path, chunk_size: int) -> List[Chunk]:
        prs = Presentation(str(path))
        chunks = []
        for slide_num, slide in enumerate(prs.slides):
            texts = [s.text_frame.text for s in slide.shapes if s.has_text_frame and s.text_frame.text.strip()]
            slide_text = "\n".join(texts)
            if not slide_text.strip():
                continue
            for i, piece in enumerate(split_into_chunks(text=slide_text, max_sentences=chunk_size)):
                chunks.append(Chunk(
                    doc_id=path.stem, 
                    chunk_id=f"s{slide_num}_c{i}",
                    text=piece
                ))
        return chunks