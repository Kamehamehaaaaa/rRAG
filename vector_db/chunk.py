from typing import List, Optional
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    vector: Optional[np.ndarray] = None
    modality: str = "text"
    source_path: Optional[Path] = None
    page_num: Optional[int] = None
    image_path: Optional[Path] = None  