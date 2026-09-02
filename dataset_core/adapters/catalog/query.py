"""Shared query vocabulary for catalogue filters (used by the CLI and the interactive REPL).

One place defines how a short ``key=value`` token maps to a :meth:`Catalog.find` keyword, so a
new filter is added here once — not in every front-end that parses filters.
"""

from __future__ import annotations

# Short CLI/REPL token -> Catalog.find keyword.
FILTER_KEYS: dict[str, str] = {
    "type": "measurement_type",
    "pol": "polarization",
    "sample": "sample",
    "since": "since",
    "until": "until",
    "notes": "notes_contains",
    "fits": "has_fits",
    "flags": "has_flags",
    "text": "text",
}
BOOLEAN_FILTERS = {"has_fits", "has_flags"}
_TRUE_WORDS = {"yes", "true", "1", "y"}


def parse_filter_tokens(tokens: list[str]) -> dict:
    """Turn ``["type=reflection", "fits=yes"]`` into ``Catalog.find`` keyword arguments.

    Raises ``ValueError`` on a malformed token or unknown key, so each front-end can present
    the error however it likes (SystemExit for a CLI, a reprint for a REPL).
    """
    filters: dict = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Filter '{token}' must be key=value. Keys: {', '.join(FILTER_KEYS)}.")
        key, value = token.split("=", 1)
        keyword = FILTER_KEYS.get(key.strip())
        if keyword is None:
            raise ValueError(f"Unknown filter key '{key}'. Keys: {', '.join(FILTER_KEYS)}.")
        filters[keyword] = value.lower() in _TRUE_WORDS if keyword in BOOLEAN_FILTERS else value
    return filters
