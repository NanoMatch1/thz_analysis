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
import os
import pickle
import subprocess
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


def _build_recipe(dataset, notes: str = "") -> dict:
    """Assemble the replayable recipe dict from a dataset's recorded context."""
    return {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "source_dir": getattr(dataset, "file_dir", None),
        "series_name": getattr(dataset, "seriesname", None),
        "git_sha": _git_sha(os.path.dirname(os.path.abspath(__file__))),
        "notes": notes,
        # config as-is (JSON turns tuples into lists — fine for replay, which indexes them).
        "config": _jsonable(getattr(dataset, "config", {}) or {}),
        # the ordered pipeline calls; includes load_all_data / group_files if recording was active.
        "steps": list(getattr(dataset, "recipe", []) or []),
    }


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

    recipe = _build_recipe(dataset, notes=notes)
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
    return bundle_dir


# ── load (fast, no recompute) ────────────────────────────────────────────────────


def read_recipe(bundle_dir: str) -> dict:
    """Read and return the recipe dict from a bundle (for replay / inspection / editing)."""
    with open(os.path.join(bundle_dir, RECIPE_FILENAME), "r", encoding="utf-8") as f:
        return json.load(f)


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

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
