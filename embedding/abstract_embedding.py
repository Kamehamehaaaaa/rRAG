from abc import ABC, abstractmethod
import numpy as np

class AbstractEmbedding(ABC):

    @property
    def max_seq_length(self) -> int | None:
        """Max input tokens the model accepts, if known. None means
        unknown — callers fall back to a conservative static default."""
        return None

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    def embed_batch(self, texts: list, batch_size: int = 64) -> list[np.ndarray]:
        pass