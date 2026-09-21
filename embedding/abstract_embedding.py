from abc import ABC, abstractmethod
import numpy as np

class AbstractEmbedding(ABC):

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    def embed_batch(self, texts: list, batch_size: int = 64) -> list[np.ndarray]:
        pass