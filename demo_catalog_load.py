"""Demo: find and load a saved analysis through the catalogue — no path to remember.

The catalogue indexes every ``.thzbundle`` under the configured root (see ``~/.thz/catalog.toml``
or env ``THZ_CATALOG_ROOT``). Instead of remembering *where* a run was saved, you ask the
catalogue for it by what it *is* (measurement type, sample, has-fits, date...) and load the
saved results straight into a ``DataSet`` — instantly, with no recompute.

This walks the whole workflow in five steps:

    1. open the catalogue
    2. browse what's indexed
    3. query for the analysis you want
    4. load it (a fully-populated DataSet: n, k, sigma, fits — all restored)
    5. use it (inspect the restored quantities and plot them)

Run it:

    python demo_catalog_load.py            # loads a reflection-with-fit run and plots it
    python demo_catalog_load.py --viewer   # also launch the interactive results viewer

The equivalent one-liners from the command line are in ``catalog_browse.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from dataset_core.adapters.catalog import Catalog
from dataset_core.adapters import thz_adapter as thz

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "explorations", "output")


# ── step 5 helper: plot the restored optical constants ─────────────────────────


def plot_optical_constants(dataset, title=""):
    """Plot n, k (left axis) and sigma (right axis) for every sample in a loaded dataset."""
    figure, axis_nk = plt.subplots(figsize=(8, 5))
    axis_sigma = axis_nk.twinx()
    colormap = plt.get_cmap("tab10")

    for index, (filename, data_object) in enumerate(thz._sample_items(dataset)):
        processing = data_object.processing_dict
        frequency_hz = processing.get("fft_freq")
        refractive_index = processing.get("n")
        extinction = processing.get("k")
        conductivity = processing.get("sigma")
        if frequency_hz is None or refractive_index is None:
            continue
        color = colormap(index)
        frequency_thz = frequency_hz * thz._HZ_TO_THZ
        axis_nk.plot(frequency_thz, refractive_index, color=color, label=f"{filename}  n")
        if extinction is not None:
            axis_nk.plot(frequency_thz, extinction, color=color, linestyle="--", label=f"{filename}  k")
        if conductivity is not None:
            axis_sigma.plot(frequency_thz, np.real(conductivity), color=color, linestyle=":", alpha=0.6)

    axis_nk.set_xlabel("Frequency (THz)")
    axis_nk.set_ylabel("n (solid) / k (dashed)")
    axis_sigma.set_ylabel("Re σ (dotted, S/m)")
    axis_nk.set_title(title)
    axis_nk.legend(fontsize=8, loc="upper right")
    return figure


def describe_record(record) -> str:
    """One-line human summary of a catalogue record."""
    fit_note = f", fits={record.fit_models}" if record.fit_models else ""
    flag_note = "  [flags]" if record.flags_raised else ""
    return (
        f"{record.bundle_id[:8]}  {record.measurement_type:12} pol={record.polarization or '-':2} "
        f"quantities={record.quantities}  '{record.series_name}'{fit_note}{flag_note}"
    )


def main() -> None:
    launch_viewer = "--viewer" in sys.argv

    # ── step 1: open the catalogue ────────────────────────────────────────────
    # No arguments: the root is resolved from env THZ_CATALOG_ROOT / ~/.thz/catalog.toml /
    # the documented default — so this same call works on any machine that's configured.
    catalog = Catalog()
    print(f"[1] Catalogue root: {catalog.root}\n")

    # ── step 2: browse what's indexed ─────────────────────────────────────────
    all_records = catalog.list_all()
    print(f"[2] {len(all_records)} analyses indexed:")
    for record in all_records:
        print("     " + describe_record(record))
    print()

    # ── step 3: query for the analysis you want ───────────────────────────────
    # "reflection runs that have a stored fit" — you don't need to know the path.
    matches = catalog.find(measurement_type="reflection", has_fits=True)
    print(f"[3] find(measurement_type='reflection', has_fits=True) -> {len(matches)} match(es):")
    for record in matches:
        print("     " + describe_record(record))
    if not matches:
        print("     (no reflection-with-fit runs found — try `python catalog_browse.py` to browse)")
        return
    chosen = matches[0]
    print(f"\n    Loading: '{chosen.series_name}'  ({chosen.relative_path})\n")

    # ── step 4: load it (no recompute) ────────────────────────────────────────
    # catalog.open() resolves the bundle from the record and restores the saved snapshot:
    # a fully-populated DataSet with every processing_dict (n, k, sigma, fit_result) intact.
    dataset = catalog.open(chosen)

    # ── step 5: use it ────────────────────────────────────────────────────────
    print("[5] Restored samples and their available quantities:")
    for filename, data_object in thz._sample_items(dataset):
        available = [key for key in ("n", "k", "sigma", "fit_result")
                     if data_object.processing_dict.get(key) is not None]
        print(f"     {filename}: {', '.join(available)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figure = plot_optical_constants(dataset, title=f"Loaded from catalogue: {chosen.series_name}")
    figure_path = os.path.join(OUTPUT_DIR, "demo_catalog_load.png")
    figure.savefig(figure_path, dpi=120, bbox_inches="tight")
    print(f"\n    Saved figure -> {figure_path}")

    if launch_viewer:
        thz.launch_results_viewer(dataset)
    plt.show()


if __name__ == "__main__":
    main()
