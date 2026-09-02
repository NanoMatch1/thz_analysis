"""Analysis Session Bundle: save / load / report for a processed dataset.

A *bundle* is a directory that makes a run self-describing and reopenable:

    myrun.thzbundle/
        recipe.json    # source dir + full config + ordered steps + git SHA + schema version
                       #   -> human-readable, editable, replayable (see pipeline_registry.replay_recipe)
        snapshot.pkl   # per-sample processing_dict + grouping + config + recipe
                       #   -> instant reload for display, NO recompute
        report.md      # ordered steps + any raised per-stage metric flags/warnings

Two ways to reopen:

    load_session(bundle_dir)                 -> DataSet, fully populated, ready to display (fast)
    replay_recipe(read_recipe(bundle_dir))   -> DataSet, recomputed from raw data (reproducible,
                                                overridable) — see pipeline_registry

The snapshot is pickle (full fidelity: FitResult, nested metric dicts). The recipe is JSON so
you can read/edit it by hand and re-run with tweaks.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import subprocess
import uuid
import zipfile
from datetime import datetime, timezone

from dataset_core.services.provenance import _jsonable

SCHEMA_VERSION = "1.0"
RECIPE_FILENAME = "recipe.json"
SNAPSHOT_FILENAME = "snapshot.pkl"
REPORT_FILENAME = "report.md"


# ── helpers ────────────────────────────────────────────────────────────────────


def _git_sha(repo_dir: str) -> str | None:
    """Best-effort current commit SHA of the repo containing ``repo_dir`` (None if unavailable)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _count_files(dataset) -> int | None:
    """Best-effort file count for the catalogue (None if the dataset can't report it)."""
    try:
        return len(dataset.data.data_dict)
    except Exception:
        return None


def _build_recipe(dataset, notes: str = "", bundle_id: str | None = None) -> dict:
    """Assemble the replayable recipe dict from a dataset's recorded context.

    ``bundle_id`` is a stable identity for the bundle (re-used across re-saves so the catalogue
    keeps one entry). ``metadata`` and ``n_files`` are lightweight fields the catalogue reads
    without loading the snapshot.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": bundle_id or str(uuid.uuid4()),
        "created": datetime.now(timezone.utc).isoformat(),
        "source_dir": getattr(dataset, "file_dir", None),
        "series_name": getattr(dataset, "seriesname", None),
        "git_sha": _git_sha(os.path.dirname(os.path.abspath(__file__))),
        "notes": notes,
        # Dataset-level tags, kept in the recipe so the catalogue read path stays pure metadata.
        "metadata": _jsonable(getattr(dataset, "metadata", {}) or {}),
        "n_files": _count_files(dataset),
        # config as-is (JSON turns tuples into lists — fine for replay, which indexes them).
        "config": _jsonable(getattr(dataset, "config", {}) or {}),
        # the ordered pipeline calls; includes load_all_data / group_files if recording was active.
        "steps": list(getattr(dataset, "recipe", []) or []),
    }


def _existing_bundle_id(bundle_dir: str) -> str | None:
    """Return the bundle_id already stored in a bundle's recipe.json, if any (for stable re-saves)."""
    recipe_path = os.path.join(bundle_dir, RECIPE_FILENAME)
    if not os.path.exists(recipe_path):
        return None
    try:
        with open(recipe_path, "r", encoding="utf-8") as f:
            return json.load(f).get("bundle_id")
    except Exception:
        return None


def _build_snapshot(dataset) -> dict:
    """Full-fidelity pickle payload (mirrors DataSet.save_state, config-aware)."""
    return {
        "data_dict": dataset.data.data_dict,
        "grouping_state": dataset.grouping.get_state(),
        "config": getattr(dataset, "config", {}),
        "recipe": list(getattr(dataset, "recipe", []) or []),
        "metadata": getattr(dataset, "metadata", {}),
        "file_metadata": getattr(dataset, "file_metadata", {}),
        "seriesname": getattr(dataset, "seriesname", None),
        "file_dir": getattr(dataset, "file_dir", None),
    }


# ── save ─────────────────────────────────────────────────────────────────────────


def save_session(dataset, bundle_dir: str, notes: str = "") -> str:
    """Write a directory bundle (recipe.json + snapshot.pkl + report.md).

    Returns the bundle directory path. Overwrites the three member files if the directory
    already exists (idempotent re-save of the same run).
    """
    os.makedirs(bundle_dir, exist_ok=True)

    # Re-use the id already on disk so a re-save updates one catalogue entry instead of forking it.
    recipe = _build_recipe(dataset, notes=notes, bundle_id=_existing_bundle_id(bundle_dir))
    with open(os.path.join(bundle_dir, RECIPE_FILENAME), "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2, default=str)

    with open(os.path.join(bundle_dir, SNAPSHOT_FILENAME), "wb") as f:
        pickle.dump(_build_snapshot(dataset), f)

    write_processing_report(dataset, os.path.join(bundle_dir, REPORT_FILENAME), notes=notes)

    print(
        f"[save_session] wrote bundle -> {bundle_dir} "
        f"({len(recipe['steps'])} recorded steps, "
        f"{len(dataset.data.data_dict)} files)."
    )
    _index_in_catalog(bundle_dir)
    return bundle_dir


def _index_in_catalog(bundle_dir: str) -> None:
    """Best-effort: add/refresh this bundle in the catalogue. Never fails a save.

    Only auto-indexes bundles saved *under* the catalogue root — the root defines the
    catalogue's scope. A bundle saved elsewhere is left for an explicit ``rebuild`` under a
    root that contains it, so ad-hoc/temporary saves never pollute the managed index.
    """
    from pathlib import Path

    try:
        from dataset_core.adapters.catalog import Catalog

        catalog = Catalog()
        if not Path(os.path.abspath(bundle_dir)).is_relative_to(catalog.root):
            print(
                "[save_session] bundle is outside the catalogue root; not auto-indexed "
                "(use `catalog_browse.py --rebuild` under its root to include it)."
            )
            return
        record = catalog.update(bundle_dir)
        print(f"[save_session] indexed in catalogue as '{record.relative_path}'.")
    except Exception as error:  # catalogue is an index, not a source of record — never block a save
        print(f"[save_session] catalogue update skipped ({error}).")


# ── load (fast, no recompute) ────────────────────────────────────────────────────


def read_recipe(bundle_dir: str) -> dict:
    """Read and return the recipe dict from a bundle (for replay / inspection / editing)."""
    with open(os.path.join(bundle_dir, RECIPE_FILENAME), "r", encoding="utf-8") as f:
        return json.load(f)


# ── portability (single-file transfer between machines) ───────────────────────────


def pack_bundle(bundle_dir: str, zip_path: str | None = None) -> str:
    """Zip a bundle directory into a single portable file (default ``<bundle_dir>.zip``).

    The bundle is already self-describing; this just makes it one file to carry between
    machines — replacing the old single-file pickle database's only real advantage.
    """
    bundle_dir = os.path.normpath(bundle_dir)
    if zip_path is None:
        zip_path = bundle_dir + ".zip"
    bundle_name = os.path.basename(bundle_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for root_dir, _dirs, files in os.walk(bundle_dir):
            for filename in files:
                absolute = os.path.join(root_dir, filename)
                # Store paths under the bundle's own name so unpacking recreates the directory.
                arcname = os.path.join(bundle_name, os.path.relpath(absolute, bundle_dir))
                archive.write(absolute, arcname)
    print(f"[pack_bundle] wrote {zip_path}.")
    return zip_path


def unpack_bundle(zip_path: str, destination_dir: str) -> str:
    """Extract a packed bundle zip into ``destination_dir``; return the bundle directory path."""
    os.makedirs(destination_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination_dir)
        top_levels = {name.split("/", 1)[0] for name in archive.namelist() if name}
    bundle_dir = (
        os.path.join(destination_dir, next(iter(top_levels)))
        if len(top_levels) == 1 else destination_dir
    )
    print(f"[unpack_bundle] extracted {zip_path} -> {bundle_dir}.")
    return bundle_dir


def replay_session(bundle_dir: str, *, override_config: dict | None = None,
                   override_steps: dict | None = None, save_to: str | None = None,
                   notes: str = ""):
    """Re-derive a bundle from raw data with optional edits, and optionally re-save.

    The principled "edit and re-derive" path: reads the bundle's recipe, replays it against the
    raw data (recomputing with the CURRENT code), applying ``override_config`` (shallow-merged
    over the stored config, e.g. ``{'sample': {...}}`` — note nested dicts replace wholesale) and
    ``override_steps`` (``{stage: {kwarg: value}}``). If ``save_to`` is given, writes a fresh
    bundle there. Returns the reproduced dataset.

    For a quick in-place tweak instead, ``load_session`` a bundle, mutate ``processing_dict``
    directly in Python, and ``save_session`` it back.
    """
    from dataset_core.adapters.pipeline_registry import replay_recipe

    recipe = read_recipe(bundle_dir)
    dataset = replay_recipe(recipe, override_config=override_config, override_steps=override_steps)
    if save_to is not None:
        save_session(dataset, save_to, notes=notes or f"edited replay of {bundle_dir}")
    return dataset


def load_session(bundle_dir: str):
    """Reconstruct a fully-populated ``DataSet`` from a bundle's snapshot — no recompute.

    The returned dataset carries the restored processing_dicts (n, k, sigma, fit_result, all
    metrics), grouping, config and recipe, so viewers/exports work immediately.
    """
    from dataset_core.dataset import DataSet

    with open(os.path.join(bundle_dir, SNAPSHOT_FILENAME), "rb") as f:
        snapshot = pickle.load(f)

    dataset = DataSet(
        snapshot.get("file_dir") or bundle_dir,
        config=snapshot.get("config", {}),
        seriesname=snapshot.get("seriesname"),
    )
    dataset.data._data_dict = snapshot.get("data_dict", {})
    grouping_state = snapshot.get("grouping_state")
    if grouping_state is not None:
        dataset.grouping.restore_state(grouping_state)
    dataset._restore_context(snapshot)
    print(
        f"[load_session] restored {len(dataset.data.data_dict)} files from {bundle_dir} "
        f"(no recompute; {len(dataset.recipe)} steps in recipe)."
    )
    return dataset


# ── processing report ────────────────────────────────────────────────────────────


def write_processing_report(dataset, path: str, notes: str = "") -> str:
    """Write a human-readable markdown report: ordered steps + raised per-stage metric flags."""
    lines: list[str] = []
    lines.append(f"# Processing report — {getattr(dataset, 'seriesname', 'dataset')}")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Source: `{getattr(dataset, 'file_dir', '')}`")
    if notes:
        lines.append(f"- Notes: {notes}")
    lines.append("")

    lines.append("## Pipeline steps (in order)")
    recipe = getattr(dataset, "recipe", []) or []
    if recipe:
        for i, step in enumerate(recipe, 1):
            kwargs = step.get("kwargs", {})
            arg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            lines.append(f"{i}. `{step['stage']}({arg_str})`")
    else:
        lines.append("_No recorded steps (recording was not active for this run)._")
    lines.append("")

    lines.append("## Per-stage diagnostics (raised flags / warnings)")
    any_flag = False
    for filename, data_obj in dataset.data.data_dict.items():
        processing = getattr(data_obj, "processing_dict", {}) or {}
        metric_keys = [k for k in processing if k.endswith("_metrics")]
        file_flags: list[str] = []
        for key in metric_keys:
            metrics = processing.get(key) or {}
            if not isinstance(metrics, dict):
                continue
            raised = [name for name, value in (metrics.get("flags", {}) or {}).items() if value]
            warnings = metrics.get("warnings", []) or []
            for name in raised:
                file_flags.append(f"  - `{key}` flag **{name}**")
            for warning in warnings:
                file_flags.append(f"  - `{key}` warning: {warning}")
        if file_flags:
            any_flag = True
            lines.append(f"- **{filename}**")
            lines.extend(file_flags)
    if not any_flag:
        lines.append("_No flags or warnings raised._")
    lines.append("")

    lines.append("## Fits")
    any_fit = False
    for filename, data_obj in dataset.data.data_dict.items():
        result = (getattr(data_obj, "processing_dict", {}) or {}).get("fit_result")
        if result is None or not getattr(result, "success", False):
            continue
        any_fit = True
        params = {**result.param_values, **result.fixed_params}
        uncertainties = getattr(result, "param_uncertainties", {}) or {}
        r_squared = getattr(result, "r_squared", float("nan"))
        lines.append(f"- **{filename}** — `{result.model_name}`  (R² = {r_squared:.4f})")
        for name, value in params.items():
            u = uncertainties.get(name)
            if isinstance(u, (int, float)) and math.isfinite(u):
                lines.append(f"  - {name} = {value:.6g} ± {u:.3g}")
            else:
                lines.append(f"  - {name} = {value:.6g}")
    if not any_fit:
        lines.append("_No fits stored._")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
