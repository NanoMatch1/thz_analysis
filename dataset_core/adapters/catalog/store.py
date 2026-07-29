"""Catalogue persistence: a swappable store behind a small protocol.

``JsonCatalogStore`` keeps the whole catalogue in one human-readable, git-diffable JSON file
at ``<root>/_thz_catalog.json``. The volume (dozens–hundreds of runs) makes a single JSON the
right call; a ``SqliteCatalogStore`` with the same protocol can replace it later without the
facade changing.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Protocol, runtime_checkable

from dataset_core.adapters.catalog.record import CatalogRecord

CATALOG_FILENAME = "_thz_catalog.json"


def default_catalog_path(root: str) -> str:
    """Path of the index file that lives at the catalogue root."""
    return os.path.join(root, CATALOG_FILENAME)


@runtime_checkable
class CatalogStore(Protocol):
    """Persistence contract for catalogue records, keyed by ``bundle_id``."""

    def upsert(self, record: CatalogRecord) -> None: ...
    def remove(self, bundle_id: str) -> None: ...
    def get(self, bundle_id: str) -> CatalogRecord | None: ...
    def all(self) -> list[CatalogRecord]: ...
    def replace_all(self, records: Iterable[CatalogRecord]) -> None: ...


class JsonCatalogStore:
    """A single-JSON-file :class:`CatalogStore`, cached in memory and persisted on write."""

    def __init__(self, path: str):
        self.path = path
        self._records: dict[str, CatalogRecord] | None = None  # lazily loaded cache

    # ── internal ─────────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> dict[str, CatalogRecord]:
        if self._records is None:
            self._records = self._load_from_disk()
        return self._records

    def _load_from_disk(self) -> dict[str, CatalogRecord]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as index_file:
                payload = json.load(index_file)
        except (json.JSONDecodeError, OSError) as error:
            print(f"[catalog] WARNING: could not read {self.path} ({error}); treating as empty.")
            return {}
        records = {}
        for entry in payload.get("records", []):
            try:
                record = CatalogRecord.from_dict(entry)
            except TypeError:
                continue  # skip a malformed/partial row rather than fail the whole load
            records[record.bundle_id] = record
        return records

    def _persist(self) -> None:
        records = self._ensure_loaded()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "catalog_format": "thz-catalog",
            "records": [record.to_dict() for record in records.values()],
        }
        with open(self.path, "w", encoding="utf-8") as index_file:
            json.dump(payload, index_file, indent=2, default=str)

    # ── protocol ─────────────────────────────────────────────────────────────
    def upsert(self, record: CatalogRecord) -> None:
        self._ensure_loaded()[record.bundle_id] = record
        self._persist()

    def remove(self, bundle_id: str) -> None:
        if self._ensure_loaded().pop(bundle_id, None) is not None:
            self._persist()

    def get(self, bundle_id: str) -> CatalogRecord | None:
        return self._ensure_loaded().get(bundle_id)

    def all(self) -> list[CatalogRecord]:
        return list(self._ensure_loaded().values())

    def replace_all(self, records: Iterable[CatalogRecord]) -> None:
        self._records = {record.bundle_id: record for record in records}
        self._persist()
