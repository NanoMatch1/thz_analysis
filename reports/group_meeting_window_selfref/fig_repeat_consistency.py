"""Figure D: Repeat-measurement consistency + anisotropy.

Runs the full analysis pipeline (including n, k inversion) for all sample
files in CNT-13/D, with and without front-pulse self-referencing. Shows:
  - how much the extracted n(f) spread shrinks between repeats at the same
    orientation (11% -> 4% according to ANALYSIS_NOTES §11);
  - that the 90°-rotated measurement stays cleanly separated from the 0°
    repeats after self-referencing, revealing real optical anisotropy that
    was previously buried under drift artefacts.

Run:
    .venv/Scripts/python.exe reports/group_meeting_window_selfref/fig_repeat_consistency.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SECOND_REFLECTION_DIR = (
    r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\D\segmented\second_reflection"
)

# Orientation groups (filename stems): parallel = 0°, perp = 90°
PARALLEL_STEMS = {"sample_a-45", "sample_a-45_1", "sample_a-45_2"}
PERP_STEMS = {"sample_a-45_p_5_0.05"}

SAMPLE_LABELS = {
    "sample_a-45": "repeat 1",
    "sample_a-45_1": "repeat 2",
    "sample_a-45_2": "repeat 3",
    "sample_a-45_p_5_0.05": "90° rotated",
}

GEOMETRY_KWARGS = {
    "geometry": "window",
    "theta_deg": 45.0,
    "polarization": "s",
    "n_window": 1.95,
}

SNR_THRESH_DB = 20
ANALYSIS_BAND_THZ = (0.3, 2.5)


def _run_full_pipeline(second_dir: str, self_reference: bool) -> DataSet:
    """Run phase-2 pipeline through invert_nk_reflection, return dataset."""
    dataset = DataSet(second_dir)
    dataset.load_all_data(case_insensitive=True)
    thz.subtract_baseline(dataset, show_graph=False)
    dataset.group_files(keywords=["type"])
    thz.global_truncate(dataset)
    thz.window_time(
        dataset,
        config={"window": {"type": "hann", "alpha": 0.2}},
        show_graph=False,
    )
    thz.zero_pad(dataset, config={"pad": {"extend_factor": 3.0}}, show_graph=False)
    thz.fft_spectrum(dataset)
    thz.transfer_function(
        dataset,
        config={
            "transfer": {"apply_snr_mask": True, "self_reference": self_reference},
            "mask": {
                "snr_thresh_db": SNR_THRESH_DB,
                "tail_fraction": 0.25,
                "min_contiguous_bins": 3,
            },
        },
        ref_type="reference",
    )
    thz.invert_nk_reflection(dataset, **GEOMETRY_KWARGS)
    return dataset


def _n_in_band(proc_dict: dict, band_thz: tuple) -> np.ndarray | None:
    n_vals = proc_dict.get("n")
    freq_thz_arr = proc_dict.get("fft_freq")
    mask_arr = proc_dict.get("transfer_mask")
    if n_vals is None or freq_thz_arr is None:
        return None
    freq_thz = freq_thz_arr * 1e-12
    band_mask = (freq_thz >= band_thz[0]) & (freq_thz <= band_thz[1])
    if mask_arr is not None:
        band_mask = band_mask & mask_arr
    return n_vals[band_mask]


def _spread_rms(n_matrix: list[np.ndarray]) -> float:
    """RMS standard deviation across repeat curves (common length required)."""
    if len(n_matrix) < 2:
        return float("nan")
    min_len = min(len(row) for row in n_matrix)
    stacked = np.column_stack([row[:min_len] for row in n_matrix])
    return float(np.mean(np.std(stacked, axis=1)))


def main() -> None:
    if not os.path.isdir(SECOND_REFLECTION_DIR):
        print(f"ERROR: directory not found: {SECOND_REFLECTION_DIR}")
        return

    print("Running pipeline — conventional reference (self_reference=False) ...")
    dataset_old = _run_full_pipeline(SECOND_REFLECTION_DIR, self_reference=False)
    print("\nRunning pipeline — self-referenced (self_reference=True) ...")
    dataset_new = _run_full_pipeline(SECOND_REFLECTION_DIR, self_reference=True)

    sample_files = sorted(
        fname for fname, _ in dataset_old.data.items()
        if not dataset_old.data.is_reference(fname)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False, layout="constrained")
    ax_old, ax_new = axes
    fig.suptitle(
        "Repeat consistency before / after front-pulse self-referencing — CNT-13/D",
        fontsize=11,
    )

    parallel_colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(PARALLEL_STEMS)))
    perp_colors = plt.cm.Oranges(np.linspace(0.5, 0.9, len(PERP_STEMS)))

    parallel_color_iter = iter(parallel_colors)
    perp_color_iter = iter(perp_colors)

    n_parallel_old: list[np.ndarray] = []
    n_parallel_new: list[np.ndarray] = []

    for filename in sample_files:
        stem = filename.replace(".acc", "")
        label = SAMPLE_LABELS.get(stem, stem)
        is_parallel = stem in PARALLEL_STEMS
        color = next(parallel_color_iter) if is_parallel else next(perp_color_iter)

        proc_old = dataset_old.data[filename].processing_dict
        proc_new = dataset_new.data[filename].processing_dict

        freq_thz = proc_old["fft_freq"] * 1e-12
        mask_old = proc_old.get("transfer_mask", np.ones(freq_thz.shape, dtype=bool))
        mask_new = proc_new.get("transfer_mask", np.ones(freq_thz.shape, dtype=bool))
        band_old = mask_old & (freq_thz >= ANALYSIS_BAND_THZ[0]) & (freq_thz <= ANALYSIS_BAND_THZ[1])
        band_new = mask_new & (freq_thz >= ANALYSIS_BAND_THZ[0]) & (freq_thz <= ANALYSIS_BAND_THZ[1])

        n_old = proc_old.get("n")
        n_new = proc_new.get("n")
        if n_old is None or n_new is None:
            print(f"  WARNING: n not found for {filename}; skipping.")
            continue

        ax_old.plot(freq_thz[band_old], n_old[band_old], color=color, lw=1.4, label=label)
        ax_new.plot(freq_thz[band_new], n_new[band_new], color=color, lw=1.4, label=label)

        if is_parallel:
            n_band_old = n_old[band_old]
            n_band_new = n_new[band_new]
            if len(n_band_old):
                n_parallel_old.append(n_band_old)
            if len(n_band_new):
                n_parallel_new.append(n_band_new)

    spread_old = _spread_rms(n_parallel_old)
    spread_new = _spread_rms(n_parallel_new)

    for ax, title, spread in [
        (ax_old, f"Conventional (H_old)\nrepeat spread std(n)~ {spread_old:.3f}", spread_old),
        (ax_new, f"Self-referenced (H_new)\nrepeat spread std(n)~ {spread_new:.3f}", spread_new),
    ]:
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("Refractive index n")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    print(f"\nRepeat spread (rms std of n in {ANALYSIS_BAND_THZ[0]}–{ANALYSIS_BAND_THZ[1]} THz):")
    print(f"  Conventional :  std(n) = {spread_old:.4f}")
    print(f"  Self-referenced: std(n) = {spread_new:.4f}")
    if not np.isnan(spread_old) and spread_old > 0:
        print(f"  Improvement:    {spread_old / spread_new:.1f}× reduction in spread")

    out_path = os.path.join(OUTPUT_DIR, "fig_repeat_consistency.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
