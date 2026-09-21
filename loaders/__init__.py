from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from vector_db.chunk import Chunk

class AbstractLoader(ABC):
    @abstractmethod
    def load(self, path: Path, chunk_size: int) -> List[Chunk]:
        ...