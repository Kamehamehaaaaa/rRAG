from typing import Dict, Type
from embedding import AbstractEmbedding

_registry: Dict[str, Type[AbstractEmbedding]] = {}

def register(name: str, provider_cls: Type[AbstractEmbedding]):
    _registry[name] = provider_cls

def get(name: str, **kwargs) -> AbstractEmbedding:
    cls = _registry.get(name)
    if cls is None:
        raise KeyError(f"Unknown provider '{name}'.")
    return cls(**kwargs)

def list_providers() -> Dict[str, Type[AbstractEmbedding]]:
    return _registry.copy()

def clear_registry():
    _registry.clear()

def remove_provider(name: str):
    if name in _registry:
        del _registry[name]

def default_registry() -> Dict[str, Type[AbstractEmbedding]]:
    from embedding.openai_embedding import OpenAIEmbeddings
    from embedding.sentencetransformer_embedding import SentenceTransformerEmbedding

    register("openai", OpenAIEmbeddings)
    register("sentence_transformer", SentenceTransformerEmbedding)