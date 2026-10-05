"""Catalog — the public entry point for indexing and querying saved analyses.

Everything here except :meth:`Catalog.open` stays on the pure metadata read path (recipe +
report only). ``open`` is the single method that touches the analysis runtime, and it imports
``DataSet`` lazily so a query session never pays for it.
"""

from __future__ import annotations

import os
from pathlib import Path

from dataset_core.adapters.catalog.config import resolve_catalog_root
from dataset_core.adapters.catalog.record import (
    RECIPE_FILENAME,
    CatalogRecord,
    extract_record,
)
from dataset_core.adapters.catalog.store import (
    CatalogStore,
    JsonCatalogStore,
    default_catalog_path,
)

BUNDLE_SUFFIX = ".thzbundle"


class Catalog:
    """A rebuildable index over ``.thzbundle`` analyses under a single configurable root."""

    def __init__(self, root: str | None = None, store: CatalogStore | None = None):
        self.root = resolve_catalog_root(root)
        # Never create a missing root: a mis-resolved root (e.g. a Windows default on Linux,
        # which abspath turns into '<cwd>/C:/Users/...') would silently become an empty
        # catalogue inside whatever directory the script was launched from.
        if not os.path.isdir(self.root):
            print(
                f"[catalog] WARNING: catalogue root '{self.root}' does not exist. Set "
                f"THZ_CATALOG_ROOT or 'root' in ~/.thz/catalog.toml to your data directory."
            )
        self.store: CatalogStore = store or JsonCatalogStore(default_catalog_path(self.root))

    # ── indexing ─────────────────────────────────────────────────────────────
    def update(self, bundle_dir: str) -> CatalogRecord:
        """Index (insert or refresh) a single bundle and return its record."""
        record = extract_record(bundle_dir, self.root)
        self.store.upsert(record)
        return record

    def rebuild(self) -> int:
        """Rescan the root for bundles and rebuild the index from scratch. Returns the count.

        A single malformed bundle (bad recipe.json, unreadable report) is skipped with a warning
        rather than aborting the whole scan — one corrupt run must not sink the catalogue.
        """
        records = []
        for bundle_dir in self._discover_bundles():
            try:
                records.append(extract_record(str(bundle_dir), self.root))
            except Exception as error:
                print(f"[catalog] WARNING: skipping unreadable bundle {bundle_dir} ({error}).")
        self.store.replace_all(records)
        return len(records)

    def verify(self) -> list[str]:
        """Return relative paths of indexed bundles whose directory no longer exists on disk."""
        missing = []
        for record in self.store.all():
            bundle_dir = os.path.join(self.root, record.relative_path)
            if not os.path.exists(os.path.join(bundle_dir, RECIPE_FILENAME)):
                missing.append(record.relative_path)
        return missing

    def _discover_bundles(self) -> list[Path]:
        root_path = Path(self.root)
        return [
            path
            for path in root_path.rglob(f"*{BUNDLE_SUFFIX}")
            if path.is_dir() and (path / RECIPE_FILENAME).exists()
        ]

    # ── querying ─────────────────────────────────────────────────────────────
    def list_all(self) -> list[CatalogRecord]:
        return sorted(self.store.all(), key=lambda record: record.created or "")

    def find(
        self,
        *,
        measurement_type: str | None = None,
        polarization: str | None = None,
        sample: str | None = None,
        since: str | None = None,
        until: str | None = None,
        notes_contains: str | None = None,
        has_fits: bool | None = None,
        has_flags: bool | None = None,
        text: str | None = None,
    ) -> list[CatalogRecord]:
        """Return records matching all supplied filters (AND-combined)."""
        results = []
        for record in self.list_all():
            if measurement_type is not None and record.measurement_type != measurement_type:
                continue
            if polarization is not None and record.polarization != polarization:
                continue
            if sample is not None and not self._matches_sample(record, sample):
                continue
            if since is not None and (record.created or "") < since:
                continue
            if until is not None and not self._before_or_within(record.created, until):
                continue
            if notes_contains is not None and notes_contains.lower() not in record.notes.lower():
                continue
            if has_fits is not None and bool(record.fit_models) != has_fits:
                continue
            if has_flags is not None and bool(record.flags_raised) != has_flags:
                continue
            if text is not None and not self._matches_text(record, text):
                continue
            results.append(record)
        return results

    def match(self, identifier: str) -> list[CatalogRecord]:
        """Records a typed identifier refers to: bundle-id prefix first, else series substring.

        The single lookup rule behind every front-end that accepts a short id or a name
        (``open_session.py``, ``catalog_browse.py --open``, :meth:`open`). An id prefix wins
        outright so ``49ad3cc4`` never also matches a series that happens to contain it.
        """
        records = self.list_all()
        by_id = [record for record in records if record.bundle_id.startswith(identifier)]
        if by_id:
            return by_id
        needle = identifier.lower()
        return [record for record in records if needle in (record.series_name or "").lower()]

    @staticmethod
    def _matches_sample(record: CatalogRecord, sample: str) -> bool:
        needle = sample.lower()
        haystacks = list(record.sample_tags) + [record.series_name or "", record.source_dir or ""]
        return any(needle in item.lower() for item in haystacks)

    @staticmethod
    def _before_or_within(created: str | None, until: str) -> bool:
        """True if ``created`` is on/before ``until`` (lenient on 'YYYY' / 'YYYY-MM' prefixes)."""
        if not created:
            return False
        return created[: len(until)] <= until

    @staticmethod
    def _matches_text(record: CatalogRecord, text: str) -> bool:
        needle = text.lower()
        haystacks = [
            record.series_name or "",
            record.notes,
            record.source_dir or "",
            *record.sample_tags,
        ]
        return any(needle in item.lower() for item in haystacks)

    # ── opening (the only DataSet dependency, imported lazily) ────────────────
    def resolve_bundle_dir(self, record_or_id) -> str:
        """Absolute bundle directory for a record or a bundle id."""
        record = self._as_record(record_or_id)
        if record is None:
            raise KeyError(f"No catalogue entry for {record_or_id!r}.")
        return os.path.join(self.root, record.relative_path)

    def open(self, record_or_id):
        """Load the bundle's saved results into a ``DataSet`` (no recompute)."""
        from dataset_core.adapters import session_bundle

        record = self._as_record(record_or_id)
        if record is not None and record.bundle_kind != "session":
            raise ValueError(
                f"{record.relative_path} is a '{record.bundle_kind}' bundle, not a DataSet "
                "session; load it with the code shown at the end of its report.md")
        return session_bundle.load_session(self.resolve_bundle_dir(record_or_id))

    # ── annotating ───────────────────────────────────────────────────────────
    def annotate(self, record_or_id, notes: str, *, append: bool = False) -> CatalogRecord:
        """Replace (or append to) a bundle's notes after the fact, keep the history, re-index.

        The previous notes are kept in ``recipe['notes_history']`` with a timestamp, so editing
        the description never loses what was said before; ``report.md`` gets the change appended.
        """
        import json
        from datetime import datetime, timezone

        bundle_dir = self.resolve_bundle_dir(record_or_id)
        recipe_path = os.path.join(bundle_dir, "recipe.json")
        with open(recipe_path, "r", encoding="utf-8") as handle:
            recipe = json.load(handle)
        previous = recipe.get("notes", "") or ""
        updated = f"{previous}\n{notes}".strip() if append and previous else notes.strip()
        now = datetime.now(timezone.utc).isoformat()
        recipe.setdefault("notes_history", []).append(
            {"replaced_at": now, "previous": previous, "source": "annotate"})
        recipe["notes"] = updated
        with open(recipe_path, "w", encoding="utf-8") as handle:
            json.dump(recipe, handle, indent=2)
        report_path = os.path.join(bundle_dir, "report.md")
        with open(report_path, "a", encoding="utf-8") as handle:
            handle.write(f"\n## Notes edited {now}\n\n{updated}\n")
        return self.update(bundle_dir)

    def _as_record(self, record_or_id) -> CatalogRecord | None:
        if isinstance(record_or_id, CatalogRecord):
            return record_or_id
        exact = self.store.get(str(record_or_id))
        if exact is not None:
            return exact
        matches = self.match(str(record_or_id))
        if len(matches) > 1:
            raise KeyError(f"{record_or_id!r} is ambiguous: matches {len(matches)} catalogue entries.")
        return matches[0] if matches else None
