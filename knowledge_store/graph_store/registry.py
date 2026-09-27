from typing import Callable, Dict
from . import AbstractGraphExtractor

_FACTORIES: Dict[str, Callable[[], AbstractGraphExtractor]] = {}
_INSTANCES: Dict[str, AbstractGraphExtractor] = {}

def register(name: str, factory: Callable[[], AbstractGraphExtractor]) -> None:
    _FACTORIES[name] = factory

def get(name: str) -> AbstractGraphExtractor:
    if name not in _INSTANCES:
        if name not in _FACTORIES:
            raise ValueError(f"no graph extractor registered under {name!r}. Known: {list(_FACTORIES)}")
        _INSTANCES[name] = _FACTORIES[name]()
    return _INSTANCES[name]