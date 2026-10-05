from __future__ import annotations
from datetime import datetime, timezone
import inspect
import json
import hashlib
from typing import Any, Callable, Dict, Optional

'''Provenance is designed to track the history of processing steps applied to data objects. It provides a decorator that can be applied to any function that takes a data object (with a .history attribute) as its first argument. Each time the decorated function is called, a record of the call (including timestamp, function name, arguments, and optionally the return value) is appended to the data object's history. This allows for easy tracking and reproducibility of data processing steps.'''

def _jsonable(x: Any) -> Any:
    """Best-effort conversion to something JSON-serializable."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    # Numpy scalars/arrays, Path, custom objects, etc.
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return _jsonable(x.item())
        except Exception:
            pass
    if hasattr(x, "__array__"):
        # avoid dumping big arrays; keep shape/dtype only
        try:
            arr = x.__array__()
            return {"__ndarray__": True, "shape": getattr(arr, "shape", None), "dtype": str(getattr(arr, "dtype", ""))}
        except Exception:
            pass
    return {"__repr__": repr(x)}

def _return_summary(ret: Any, max_chars: int = 500) -> Any:
    # keep it light; don't store giant arrays
    j = _jsonable(ret)
    s = json.dumps(j, default=str)
    if len(s) > max_chars:
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        return {"__summary__": True, "sha256": h, "truncated": True}
    return j

def record_provenance(*, store_return: bool = False, note: Optional[str] = None):
    """
    Decorator for functions that take a 'data' object as first arg.
    Appends a record to data.history (list[dict]).
    """
    def deco(func: Callable):
        sig = inspect.signature(func)

        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            data = bound.arguments.get("data", None) or (args[0] if args else None)
            if data is None or not hasattr(data, "history"):
                raise TypeError("Expected first arg (or 'data') to have a .history attribute.")

            rec: Dict[str, Any] = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "func": f"{func.__module__}.{func.__qualname__}",
                "args": {k: _jsonable(v) for k, v in bound.arguments.items() if k != "data"},
            }
            if note:
                rec["note"] = note

            ret = func(*args, **kwargs)

            if store_return:
                rec["return"] = _return_summary(ret)

            data.history.append(rec)
            return ret

        wrapper.__name__ = func.__name__
        wrapper.__qualname__ = func.__qualname__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return deco


# ── code and input identity, for saved results ──────────────────────────────────

def git_state(path: str) -> Dict[str, Any]:
    """Commit, branch and dirtiness of the git repo containing ``path`` (best effort).

    ``dirty`` matters for provenance: a result made from uncommitted code cannot be reproduced
    from the SHA alone, so it is recorded rather than assumed clean.
    """
    import os
    import subprocess

    directory = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))

    def run(*arguments):
        try:
            result = subprocess.run(["git", *arguments], cwd=directory, capture_output=True,
                                    text=True, timeout=5)
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def code_versions() -> Dict[str, Any]:
    """git state of thz_analysis and of the thz-core checkout it imports."""
    import os

    versions = {"thz_analysis": git_state(os.path.dirname(os.path.abspath(__file__)))}
    try:
        import thz_core.thz_core as core
        versions["thz_core"] = git_state(os.path.realpath(os.path.dirname(core.__file__)))
    except Exception:
        versions["thz_core"] = None
    return versions


def file_sha256(path: str, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 of a file's contents: identifies the exact raw data a result came from."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()
