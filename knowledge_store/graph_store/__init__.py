from abc import ABC, abstractmethod
from .schema import ExtractedGraph

class AbstractGraphExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> ExtractedGraph:
        pass