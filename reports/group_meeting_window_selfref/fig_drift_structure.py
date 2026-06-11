"""Figure B: The measured drift D = Y1_sample / Y1_reference.

Since the front reflection never touches the sample, D ≡ 1 with no drift.
Any deviation is pure mount-to-mount alignment drift — the very quantity that
self-referencing cancels. This figure plots |D|−1 (%) and arg(D) from the real
CNT-13/D first-reflection measurements.

Run:
    .venv/Scripts/python.exe reports/group_meeting_window_selfref/fig_drift_structure.py
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

from dataset_core.io.loaders.acc_loader import ACCLoader
from dataset_core.adapters import thz_adapter as thz

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIRST_REFLECTION_DIR = (
    r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\D\segmented\first_reflection"
)

REFERENCE_FILENAME = "reference_a-45.acc"
SAMPLE_LABELS = {
    "sample_a-45.acc": "sample (repeat 1, 0°)",
    "sample_a-45_1.acc": "sample (repeat 2, 0°)",
    "sample_a-45_2.acc": "sample (repeat 3, 0°)",
    "sample_a-45_p_5_0.05.acc": "sample (perpendicular, 90°)",
}

FFT_LENGTH = 8192
PLOT_BAND_THZ = (0.2, 3.0)
# Show the SNR-trusted portion: mask based on reference dynamic range
SNR_FLOOR_DB = -15.0


def _load_acc_averaged(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_s, amplitude) for the averaged trace from a .acc file."""
    averaged = np.asarray(ACCLoader(filepath).load().data, dtype=float)
    return averaged[:, 0], averaged[:, 1]


def main() -> None:
    ref_path = os.path.join(FIRST_REFLECTION_DIR, REFERENCE_FILENAME)
    if not os.path.isfile(ref_path):
        print(f"ERROR: reference file not found: {ref_path}")
        return

    time_ref_s, amp_ref = _load_acc_averaged(ref_path)
    freq_hz, Y_ref = thz._absolute_time_spectrum(time_ref_s, amp_ref, FFT_LENGTH)
    f_thz = freq_hz * 1e-12

    ref_peak_db = np.max(20 * np.log10(np.abs(Y_ref) + 1e-30))
    ref_power_db = 20 * np.log10(np.abs(Y_ref) + 1e-30) - ref_peak_db
    snr_mask = (ref_power_db >= SNR_FLOOR_DB) & (f_thz >= PLOT_BAND_THZ[0]) & (f_thz <= PLOT_BAND_THZ[1])

    fig, (ax_amp, ax_phase) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, layout="constrained"
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(SAMPLE_LABELS)))

    print("Drift D = Y1_sample / Y1_reference")
    print(f"  {'Label':<35} | |D|-1 rms (%)  | phase rms (°)")
    print(f"  {'-'*35}-+-{'-'*14}-+-{'-'*14}")

    for color, (samp_filename, label) in zip(colors, SAMPLE_LABELS.items()):
        samp_path = os.path.join(FIRST_REFLECTION_DIR, samp_filename)
        if not os.path.isfile(samp_path):
            print(f"  Skipping {samp_filename} (not found)")
            continue

        time_samp_s, amp_samp = _load_acc_averaged(samp_path)
        _, Y_samp = thz._absolute_time_spectrum(time_samp_s, amp_samp, FFT_LENGTH)

        with np.errstate(divide="ignore", invalid="ignore"):
            D = Y_samp / Y_ref

        D_mag_pct = (np.abs(D) - 1.0) * 100.0
        D_phase_deg = np.degrees(np.unwrap(np.angle(D)))

        ax_amp.plot(f_thz[snr_mask], D_mag_pct[snr_mask], color=color, lw=1.4, label=label)
        ax_phase.plot(f_thz[snr_mask], D_phase_deg[snr_mask], color=color, lw=1.4)

        amp_rms = float(np.sqrt(np.mean(D_mag_pct[snr_mask] ** 2)))
        phase_rms = float(np.sqrt(np.mean(D_phase_deg[snr_mask] ** 2)))
        print(f"  {label:<35} | {amp_rms:>12.1f}% | {phase_rms:>12.1f}°")

    ax_amp.axhline(0, color="0.5", lw=0.8, ls="--", zorder=0)
    ax_amp.set_ylabel("|D| − 1  (%)")
    ax_amp.set_title(
        "Mount drift D = Y₁_sample / Y₁_reference  (front reflection, CNT-13/D)"
    )
    ax_amp.legend(fontsize=8)
    ax_amp.grid(True, which="both", alpha=0.2)

    ax_phase.axhline(0, color="0.5", lw=0.8, ls="--", zorder=0)
    ax_phase.set_ylabel("arg(D)  (°)")
    ax_phase.set_xlabel("Frequency (THz)")
    ax_phase.grid(True, which="both", alpha=0.2)

    out_path = os.path.join(OUTPUT_DIR, "fig_drift_structure.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
