# dataset_core/io/loaders/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type, Optional


class LoaderError(RuntimeError):
    pass


class BaseLoader:
    """All loaders should subclass this."""
    extension: str  # e.g. ".acc"

    @classmethod
    def can_load(cls, ext: str) -> bool:
        '''Edit this to be smarter when we have more complex loading needs.'''
        return ext.lower() == cls.extension.lower()


@dataclass(frozen=True)
class LoaderInfo:
    extension: str
    cls: Type[BaseLoader]


_REGISTRY: Dict[str, Type[BaseLoader]] = {}


def register_loader(cls: Type[BaseLoader]) -> Type[BaseLoader]:
    ext = getattr(cls, "extension", None)
    if not isinstance(ext, str) or not ext.startswith("."):
        raise LoaderError(f"{cls.__name__} must define extension like '.acc'")

    key = ext.lower()
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise LoaderError(
            f"Duplicate loader for extension {ext}: "
            f"{_REGISTRY[key].__name__} and {cls.__name__}"
        )

    _REGISTRY[key] = cls
    return cls


def get_loader_for_extension(ext: str) -> Optional[Type[BaseLoader]]:
    key = ext.lower()
    loader = _REGISTRY.get(key)
    if loader is None:
        import warnings
        available = list(_REGISTRY.keys())
        warnings.warn(
            f"No loader registered for extension '{ext}'. "
            f"Available loaders: {available}. Skipping file."
        )
    return loader


def registered_extensions() -> Dict[str, Type[BaseLoader]]:
    return dict(_REGISTRY)
