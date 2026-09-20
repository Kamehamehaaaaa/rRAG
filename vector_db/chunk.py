from typing import List

class Chunk:
    text: str
    vector: List[float]
    doc_id: str
    chunk_id: int
    metadata: dict  