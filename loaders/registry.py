# loaders/registry.py
from pathlib import Path

from captioning import AbstractCaptioner
from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .pptx_loader import PPTXLoader
from .image_loader import ImageLoader
from . import AbstractLoader

class UnsupportedExtension(ValueError):
    """Genuinely unrecognized file type — safe to skip during a directory walk."""


class MissingCaptioner(ValueError):
    """Extension is supported but needs a captioner that wasn't provided —
    a caller mistake, should propagate rather than be silently skipped."""


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
_TEXT_EXTENSIONS = {".txt", ".md"}
_NEEDS_CAPTIONER = {".pdf", ".pptx", *_IMAGE_EXTENSIONS}


_LOADERS: dict[str, AbstractLoader] = {} 

def get_loader(path: Path, captioner: AbstractCaptioner | None = None) -> AbstractLoader:
    suffix = path.suffix.lower()

    if suffix in _NEEDS_CAPTIONER and captioner is None:
        raise MissingCaptioner(f"extension {suffix} needs a captioner")

    if suffix in _TEXT_EXTENSIONS:
        return _LOADERS.setdefault(suffix, TextLoader())
    
    if suffix == ".pdf":
        return _LOADERS.setdefault(suffix, PDFLoader())
    
    if suffix == ".pptx":
        return _LOADERS.setdefault(suffix, PPTXLoader())
    
    if suffix in _IMAGE_EXTENSIONS:
        return _LOADERS.setdefault(suffix, ImageLoader(captioner))
    
    raise UnsupportedExtension(f"no loader registered for extension {suffix!r}")