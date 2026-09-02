"""Resolve the catalogue root directory — the one place saved analyses are indexed.

Resolution order (first hit wins), so the location is configurable and never hardcoded
into call sites:

    1. explicit argument
    2. environment variable  THZ_CATALOG_ROOT
    3. per-user config file   ~/.thz/catalog.toml   (key: ``root = "..."``)
    4. documented default     C:/Users/Samuel/Data/THz

The catalogue index file lives *at* the resolved root (see ``store.default_catalog_path``),
so "find the catalogue" reduces to "find the root".
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

ENV_VAR_NAME = "THZ_CATALOG_ROOT"
USER_CONFIG_PATH = Path.home() / ".thz" / "catalog.toml"
DEFAULT_ROOT = r"C:/Users/Samuel/Data/THz"


def _read_root_from_config_file(config_path: Path = USER_CONFIG_PATH) -> str | None:
    """Return the ``root`` key from the TOML config file, or None if unavailable/malformed."""
    try:
        with open(config_path, "rb") as config_file:
            parsed = tomllib.load(config_file)
    except (FileNotFoundError, tomllib.TOMLDecodeError, OSError):
        return None
    root_value = parsed.get("root")
    return str(root_value) if root_value else None


def resolve_catalog_root(explicit_root: str | None = None) -> str:
    """Resolve the catalogue root directory following the documented precedence."""
    if explicit_root:
        return os.path.abspath(os.path.expanduser(explicit_root))

    env_root = os.environ.get(ENV_VAR_NAME)
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))

    file_root = _read_root_from_config_file()
    if file_root:
        return os.path.abspath(os.path.expanduser(file_root))

    return os.path.abspath(os.path.expanduser(DEFAULT_ROOT))
