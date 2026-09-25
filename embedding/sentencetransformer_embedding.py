from embedding.abstract_embedding import AbstractEmbedding
from sentence_transformers import SentenceTransformer
from typing import List

class SentenceTransformerEmbedding(AbstractEmbedding):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    @property
    def max_seq_length(self) -> int | None:
        return self.model.max_seq_length

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 500,
        )
        return vectors.tolist()