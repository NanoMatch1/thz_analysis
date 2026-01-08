# io/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type, Optional


class LoaderError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoaderInfo:
    extension: str
    cls: Type["BaseLoader"]


_REGISTRY: Dict[str, LoaderInfo] = {}


def normalize_ext(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext:
        raise LoaderError("Empty extension")
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def register_loader(cls: Type["BaseLoader"]) -> Type["BaseLoader"]:
    ext = normalize_ext(getattr(cls, "extension", ""))
    if ext in _REGISTRY:
        raise LoaderError(
            f"Duplicate loader for {ext}: "
            f"{_REGISTRY[ext].cls.__name__} vs {cls.__name__}"
        )
    _REGISTRY[ext] = LoaderInfo(extension=ext, cls=cls)
    return cls


def get_loader(ext: str) -> Optional[Type["BaseLoader"]]:
    info = _REGISTRY.get(normalize_ext(ext))
    return None if info is None else info.cls


def all_loaders() -> Dict[str, Type["BaseLoader"]]:
    return {ext: info.cls for ext, info in _REGISTRY.items()}


class BaseLoader:
    # each subclass must set: extension = ".acc" etc.
    extension: str

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError
