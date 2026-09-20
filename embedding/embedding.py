from abc import ABC, abstractmethod

class AbstractEmbedding(ABC):

    @abstractmethod
    def embed(self, text: str) -> list:
        pass

    @abstractmethod
    def embed_batch(self, texts: list, batch_size: int = 64) -> list[list]:
        pass