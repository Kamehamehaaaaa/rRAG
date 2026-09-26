# loaders/pdf_loader.py
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from captioning import AbstractCaptioner
from vector_db.chunk import Chunk
from . import AbstractLoader
from .text_utils import split_into_chunks

EXTRACTED_IMAGE_DIR = Path("data/extracted_images")


class PDFLoader(AbstractLoader):
    def __init__(self, captioner: AbstractCaptioner):
        self.captioner = captioner

    def load(self, path: Path, chunk_size: int, max_chars: Optional[int] = None) -> List[Chunk]:
        doc = fitz.open(str(path))
        chunks: List[Chunk] = []

        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                for i, piece in enumerate(split_into_chunks(page_text, chunk_size, max_chars=max_chars)):
                    chunks.append(Chunk(
                        doc_id=path.stem, chunk_id=f"p{page_num}_c{i}",
                        text=piece, source_path=path, page_num=page_num,
                    ))
            chunks.extend(self._extract_images(doc, page, page_num, path, page_text))

        doc.close()
        return chunks

    def _extract_images(self, doc, page, page_num: int, source_path: Path, page_text: str) -> List[Chunk]:
        chunks = []
        EXTRACTED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            extracted = doc.extract_image(xref)
            image_path = EXTRACTED_IMAGE_DIR / f"{source_path.stem}_p{page_num}_i{img_index}.{extracted['ext']}"
            image_path.write_bytes(extracted["image"])

            caption = self.captioner.caption(image_path, context=page_text.strip() or None)
            chunks.append(Chunk(
                doc_id=source_path.stem, chunk_id=f"p{page_num}_img{img_index}",
                text=caption, modality="image",
                source_path=source_path, page_num=page_num, image_path=image_path,
            ))
        return chunks