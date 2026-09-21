# loaders/registry.py
from pathlib import Path
from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .pptx_loader import PPTXLoader

_LOADERS = {".txt": TextLoader(), ".md": TextLoader(), ".pdf": PDFLoader(), ".pptx": PPTXLoader()}

def get_loader(path: Path):
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"no loader registered for extension {path.suffix!r}")
    return loader