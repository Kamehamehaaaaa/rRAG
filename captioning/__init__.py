from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List

class AbstractCaptioner(ABC):
    @abstractmethod
    def caption(self, path: Path, context: Optional[str]) -> str:
        pass

    def caption_batch(
        self, image_paths: List[Path], contexts: Optional[List[Optional[str]]] = None
    ) -> List[str]:
        if contexts is None:
            contexts = [None] * len(image_paths)
        if len(contexts) != len(image_paths):
            raise ValueError("image_paths and contexts must be the same length")
        return [self.caption(p, c) for p, c in zip(image_paths, contexts)]