from typing import List
import numpy as np

class Chunk:
    text: str
    vector: np.ndarray
    doc_id: str
    chunk_id: int
    metadata: dict  

    def __init__(self, text: str, vector: np.ndarray = None, doc_id: str = None, chunk_id: int = None, metadata: dict = None):
        self.text = text
        self.vector = vector
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.metadata = metadata or {}