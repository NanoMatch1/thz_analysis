"""CatalogRecord + the single extractor that maps a bundle to a queryable record.

`extract_record` is the ONE place that decides which fields are searchable. Adding a new
query dimension is a one-line change here — not a parallel schema edit elsewhere.

The read path is pure metadata: it reads only ``recipe.json`` (+ ``report.md``), never loads
``snapshot.pkl`` and never imports ``DataSet``. That keeps querying the catalogue cheap and
free of the analysis runtime.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone

CATALOG_SCHEMA_VERSION = "1.0"

RECIPE_FILENAME = "recipe.json"
REPORT_FILENAME = "report.md"

# Deterministic namespace for fallback ids of bundles saved before bundle_id existed.
_FALLBACK_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "thz-catalog-bundle")


@dataclass(frozen=True)
class CatalogRecord:
    """One lightweight, fully-queryable row describing a saved ``.thzbundle`` (no arrays)."""

    bundle_id: str
    relative_path: str            # bundle dir relative to the catalogue root (portable pointer)
    series_name: str | None
    source_dir: str | None
    created: str | None
    git_sha: str | None
    notes: str
    measurement_type: str         # 'transmission' | 'reflection' | 'gold' | 'unknown'
    polarization: str | None      # 's' | 'p' | None
    geometry_detail: str | None   # 'window' | 'gold' | ... (from the reflection invert stage)
    sample_tags: list[str]
    n_files: int | None
    quantities: list[str]         # subset of ['n', 'k', 'sigma', 'fit']
    fit_models: list[str]
    flags_raised: list[str]
    indexed_at: str
    catalog_schema_version: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CatalogRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


# ── derivation helpers (all pure, recipe/report only) ───────────────────────────


def _relative_path(bundle_dir: str, root: str) -> str:
    """Bundle path relative to root, forward-slashed so it survives OS/drive moves."""
    try:
        relative = os.path.relpath(os.path.abspath(bundle_dir), os.path.abspath(root))
    except ValueError:
        # Different drive on Windows — fall back to the basename so the record is still usable.
        relative = os.path.basename(os.path.normpath(bundle_dir))
    return relative.replace(os.sep, "/")


def _steps_by_stage(recipe: dict) -> dict[str, dict]:
    """Map stage name -> its recorded kwargs (last occurrence wins)."""
    by_stage: dict[str, dict] = {}
    for step in recipe.get("steps", []) or []:
        stage = step.get("stage")
        if stage:
            by_stage[stage] = dict(step.get("kwargs") or {})
    return by_stage


def _derive_measurement_type(steps_by_stage: dict[str, dict]) -> tuple[str, str | None]:
    """Return (measurement_type, geometry_detail) from which stages ran.

    Transmission vs reflection is carried by *which invert stage ran*, not by config —
    ``invert_nk`` for transmission, ``invert_nk_reflection`` (geometry ``window``/``gold``)
    for reflection.
    """
    if "invert_nk_reflection" in steps_by_stage:
        geometry_detail = steps_by_stage["invert_nk_reflection"].get("geometry")
        measurement_type = "gold" if geometry_detail == "gold" else "reflection"
        return measurement_type, geometry_detail
    if "invert_nk" in steps_by_stage:
        return "transmission", None
    if any(stage.endswith("_reflection") for stage in steps_by_stage):
        return "reflection", None
    return "unknown", None


def _derive_quantities(steps_by_stage: dict[str, dict], has_fits: bool) -> list[str]:
    quantities: list[str] = []
    if "invert_nk" in steps_by_stage or "invert_nk_reflection" in steps_by_stage:
        quantities += ["n", "k"]
    if "derive_eps_sigma" in steps_by_stage:
        quantities.append("sigma")
    if has_fits:
        quantities.append("fit")
    return quantities


def _stringify_tags(value) -> list[str]:
    """Flatten a nested config/metadata value into a list of short string tokens."""
    tokens: list[str] = []
    if isinstance(value, dict):
        for inner in value.values():
            tokens += _stringify_tags(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            tokens += _stringify_tags(inner)
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value).strip()
        if text:
            tokens.append(text)
    return tokens


def _derive_sample_tags(recipe: dict) -> list[str]:
    """Best-effort sample descriptors from series name, config['sample'] and dataset metadata."""
    config = recipe.get("config") or {}
    tokens: list[str] = []
    series_name = recipe.get("series_name")
    if series_name:
        tokens += [part for part in re.split(r"[\s_\-]+", str(series_name)) if part]
    tokens += _stringify_tags(config.get("sample"))
    tokens += _stringify_tags(recipe.get("metadata"))
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique = []
    for token in tokens:
        if token.lower() not in seen:
            seen.add(token.lower())
            unique.append(token)
    return unique


def _parse_report(report_text: str) -> tuple[list[str], list[str]]:
    """Extract (fit_models, flags_raised) from a report.md body."""
    fit_models = re.findall(r"—\s+`([^`]+)`\s+\(R", report_text)
    flags_raised = re.findall(r"flag \*\*([^*]+)\*\*", report_text)
    # De-duplicate, preserve order.
    def _unique(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out
    return _unique(fit_models), _unique(flags_raised)


# ── the extractor ───────────────────────────────────────────────────────────────


def extract_record(bundle_dir: str, root: str) -> CatalogRecord:
    """Build a :class:`CatalogRecord` from a bundle's ``recipe.json`` (+ ``report.md``)."""
    recipe_path = os.path.join(bundle_dir, RECIPE_FILENAME)
    with open(recipe_path, "r", encoding="utf-8") as recipe_file:
        recipe = json.load(recipe_file)

    report_path = os.path.join(bundle_dir, REPORT_FILENAME)
    fit_models: list[str] = []
    flags_raised: list[str] = []
    if os.path.exists(report_path):
        # Best-effort enrichment: never let an odd-encoded report crash indexing (errors="replace").
        with open(report_path, "r", encoding="utf-8", errors="replace") as report_file:
            fit_models, flags_raised = _parse_report(report_file.read())

    relative_path = _relative_path(bundle_dir, root)
    config = recipe.get("config") or {}
    steps_by_stage = _steps_by_stage(recipe)
    measurement_type, geometry_detail = _derive_measurement_type(steps_by_stage)
    has_fits = bool(fit_models)

    bundle_id = recipe.get("bundle_id") or str(
        uuid.uuid5(_FALLBACK_ID_NAMESPACE, relative_path)
    )

    return CatalogRecord(
        bundle_id=bundle_id,
        relative_path=relative_path,
        series_name=recipe.get("series_name"),
        source_dir=recipe.get("source_dir"),
        created=recipe.get("created"),
        git_sha=recipe.get("git_sha"),
        notes=recipe.get("notes", "") or "",
        measurement_type=measurement_type,
        polarization=(config.get("geometry") or {}).get("polarization"),
        geometry_detail=geometry_detail,
        sample_tags=_derive_sample_tags(recipe),
        n_files=recipe.get("n_files"),
        quantities=_derive_quantities(steps_by_stage, has_fits),
        fit_models=fit_models,
        flags_raised=flags_raised,
        indexed_at=datetime.now(timezone.utc).isoformat(),
        catalog_schema_version=CATALOG_SCHEMA_VERSION,
    )
