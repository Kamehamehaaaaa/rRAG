from embedding.abstract_embedding import AbstractEmbedding
import os
import openai
import numpy as np

class OpenAIEmbeddings(AbstractEmbedding):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def embed(self, texts):
        r = openai.Embedding.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [np.asarray(data["embedding"], dtype="float32") for data in r["data"]]