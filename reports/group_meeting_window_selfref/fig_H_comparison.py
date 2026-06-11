"""Figure C: H_old vs H_new — the fake spectral features removed.

Runs the full phase-2 spectral pipeline on CNT-13/D data twice, with and
without front-pulse self-referencing, and overlays |H| for each sample.
The structured oscillations in H_old (caused by mount drift) that are absent
in H_new are the artefacts the method removes.

Run:
    .venv/Scripts/python.exe reports/group_meeting_window_selfref/fig_H_comparison.py
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

# Friendly labels: filename stem → display label
SAMPLE_LABELS = {
    "sample_a-45": "repeat 1 (0°)",
    "sample_a-45_1": "repeat 2 (0°)",
    "sample_a-45_2": "repeat 3 (0°)",
    "sample_a-45_p_5_0.05": "perpendicular (90°)",
}

SNR_THRESH_DB = 20
PLOT_BAND_THZ = (0.2, 3.0)


def _run_spectral_pipeline(second_dir: str, self_reference: bool) -> DataSet:
    """Run phase-2 pipeline up to transfer_function, return loaded dataset."""
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
    return dataset


def _sample_label(filename: str) -> str:
    stem = filename.replace(".acc", "")
    return SAMPLE_LABELS.get(stem, stem)


def main() -> None:
    if not os.path.isdir(SECOND_REFLECTION_DIR):
        print(f"ERROR: directory not found: {SECOND_REFLECTION_DIR}")
        return

    print("Running pipeline — conventional reference (self_reference=False) ...")
    dataset_old = _run_spectral_pipeline(SECOND_REFLECTION_DIR, self_reference=False)
    print("Running pipeline — self-referenced (self_reference=True) ...")
    dataset_new = _run_spectral_pipeline(SECOND_REFLECTION_DIR, self_reference=True)

    # Collect sample filenames
    sample_files = [
        fname for fname, _ in dataset_old.data.items()
        if not dataset_old.data.is_reference(fname)
    ]

    n_samples = len(sample_files)
    fig, axes = plt.subplots(
        n_samples, 2, figsize=(12, 3 * n_samples), sharex=True, layout="constrained"
    )
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "H_old (conventional) vs H_new (self-referenced) — CNT-13/D",
        fontsize=11,
    )

    print(f"\n  {'Label':<35} | |H_old| mean | |H_new| mean | rms change")
    print(f"  {'-'*35}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

    for row_index, filename in enumerate(sorted(sample_files)):
        label = _sample_label(filename)
        proc_old = dataset_old.data[filename].processing_dict
        proc_new = dataset_new.data[filename].processing_dict

        freq_thz = proc_old["fft_freq"] * 1e-12
        H_old = proc_old["transfer_H"]
        H_new = proc_new["transfer_H"]
        mask_old = proc_old["transfer_mask"]
        mask_new = proc_new["transfer_mask"]
        mask_common = mask_old & mask_new & (freq_thz >= PLOT_BAND_THZ[0]) & (freq_thz <= PLOT_BAND_THZ[1])

        ax_left = axes[row_index, 0]
        ax_right = axes[row_index, 1]

        ax_left.plot(freq_thz[mask_common], np.abs(H_old[mask_common]), lw=1.3, color="tab:red", label="|H_old|")
        ax_left.plot(freq_thz[mask_common], np.abs(H_new[mask_common]), lw=1.3, color="tab:blue", label="|H_new|")
        ax_left.set_ylabel(f"{label}\n|H|")
        ax_left.legend(fontsize=8)
        ax_left.grid(True, alpha=0.2)

        # Phase comparison
        ph_old = np.unwrap(np.angle(H_old))
        ph_new = np.unwrap(np.angle(H_new))
        ax_right.plot(freq_thz[mask_common], np.degrees(ph_old[mask_common]), lw=1.3, color="tab:red", label="arg(H_old)")
        ax_right.plot(freq_thz[mask_common], np.degrees(ph_new[mask_common]), lw=1.3, color="tab:blue", label="arg(H_new)")
        ax_right.set_ylabel("arg(H)  (°)")
        ax_right.legend(fontsize=8)
        ax_right.grid(True, alpha=0.2)

        H_old_band = H_old[mask_common]
        H_new_band = H_new[mask_common]
        rms_change = float(np.sqrt(np.mean(np.abs(H_new_band - H_old_band) ** 2)))
        print(
            f"  {label:<35} | {np.mean(np.abs(H_old_band)):>12.3f} | "
            f"{np.mean(np.abs(H_new_band)):>12.3f} | {rms_change:>10.4f}"
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency (THz)")
    if n_samples > 0:
        axes[0, 0].set_title("Amplitude |H|")
        axes[0, 1].set_title("Phase arg(H)")

    out_path = os.path.join(OUTPUT_DIR, "fig_H_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
