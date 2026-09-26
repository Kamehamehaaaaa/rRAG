from . import AbstractCaptioner
from typing import Callable

_FACTORIES: dict[str, Callable[[], AbstractCaptioner]] = {}
_INSTANCE: dict[str, AbstractCaptioner] = {}

def register(name:str, factory: Callable[[], AbstractCaptioner]):
    _FACTORIES[name] = factory

def get(name: str):
    if name in _INSTANCE:
        return _INSTANCE[name]
    elif name in _FACTORIES:
        captioner = _FACTORIES[name]()
        _INSTANCE[name] = captioner
        return captioner
    else:
        raise KeyError(f"Unknown captioner '{name}'.")