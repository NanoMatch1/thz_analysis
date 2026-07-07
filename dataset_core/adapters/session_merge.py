"""Merge processed datasets / session bundles into one for combined display + export.

Use case: compare samples across runs (e.g. N-type transmission vs reflection, a temperature
series, or s- vs p-pol) in a single registry-driven viewer. Merging combines the per-sample
``processing_dict`` results (not raw traces), namespacing each file by its source so nothing
collides and provenance is preserved.

    merge_datasets([ds_a, ds_b], labels=["trans", "refl"])   -> DataSet   (in-memory)
    merge_sessions([bundle_a, bundle_b], labels=[...])        -> DataSet   (from .thzbundle dirs)

The merged dataset works everywhere a normal dataset does: ``launch_results_viewer``,
``export_quantities``, ``save_session``. Each merged file key is ``"<label>::<original>"`` and its
source is recorded in ``file_metadata`` and as a ``merge_datasets`` provenance step in the recipe.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _default_labels(datasets) -> list[str]:
    labels, seen = [], {}
    for index, dataset in enumerate(datasets):
        base = getattr(dataset, "seriesname", None) or f"ds{index}"
        # de-duplicate identical series names so keys stay unique
        count = seen.get(base, 0)
        seen[base] = count + 1
        labels.append(base if count == 0 else f"{base}#{count + 1}")
    return labels


def merge_datasets(datasets, labels: list[str] | None = None, *, config: dict | None = None):
    """Merge several processed datasets into one (namespaced by source label).

    Brings over every file (samples AND references) with its grouping classification intact, so
    the combined set displays correctly (references still recognised, samples still samples).
    Returns a new ``DataSet``.
    """
    from dataset_core.dataset import DataSet

    datasets = list(datasets)
    if not datasets:
        raise ValueError("merge_datasets requires at least one dataset.")
    if labels is None:
        labels = _default_labels(datasets)
    if len(labels) != len(datasets):
        raise ValueError("labels must match the number of datasets.")

    first = datasets[0]
    merged = DataSet(
        getattr(first, "file_dir", "merged"),
        config=config if config is not None else dict(getattr(first, "config", {}) or {}),
        seriesname="merged__" + "_".join(labels),
    )

    merged_data: dict = {}
    merged_file_items: dict = {}
    ordered_keys: list = []
    reference_identifiers: set = set()
    sample_identifiers: set = set()

    for label, dataset in zip(labels, datasets):
        state = dataset.grouping.get_state()
        source_file_items = state.get("file_items", {}) or {}
        reference_identifiers |= set(getattr(dataset.grouping, "reference_identifiers", set()))
        sample_identifiers |= set(getattr(dataset.grouping, "sample_identifiers", set()))
        for filename, data_obj in dataset.data.data_dict.items():
            key = f"{label}::{filename}"
            merged_data[key] = data_obj
            if filename in source_file_items:
                merged_file_items[key] = source_file_items[filename]
            ordered_keys.append(key)
            merged.file_metadata[key] = {"source": label, "original": filename}

    merged.data._data_dict = merged_data
    if reference_identifiers:
        merged.grouping.reference_identifiers = reference_identifiers
    if sample_identifiers:
        merged.grouping.sample_identifiers = sample_identifiers
    merged.grouping.restore_state({
        "file_items": merged_file_items,
        "filelist": ordered_keys,
        "current_data_list": ordered_keys,
    })

    merged.recipe.append({
        "stage": "merge_datasets",
        "kwargs": {"labels": list(labels), "n_files": len(ordered_keys)},
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    print(f"[merge_datasets] merged {len(datasets)} datasets -> {len(ordered_keys)} files "
          f"(labels: {', '.join(labels)}).")
    return merged


def merge_sessions(bundle_dirs, labels: list[str] | None = None, *, config: dict | None = None):
    """Load several ``.thzbundle`` directories and merge them (see :func:`merge_datasets`)."""
    from dataset_core.adapters.session_bundle import load_session

    bundle_dirs = list(bundle_dirs)
    datasets = [load_session(bundle) for bundle in bundle_dirs]
    if labels is None:
        import os
        labels = [os.path.basename(os.path.normpath(bundle)).replace(".thzbundle", "")
                  for bundle in bundle_dirs]
    merged = merge_datasets(datasets, labels=labels, config=config)
    # record the source bundles too, so the merge is reproducible from disk
    merged.recipe[-1]["kwargs"]["sources"] = [str(b) for b in bundle_dirs]
    return merged
