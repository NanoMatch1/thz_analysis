"""Raw vs recentered |H|, n, k for the angular-misalignment gold-mirror series.

Makes the point at a glance: re-centering the pulses to a common T0 (a pure delay
correction for angular misalignment) leaves the AMPLITUDE distortion untouched. One row
per misaligned mirror, columns |H| / n / k, raw (solid) vs recentered (dashed).

  * |H|  : solid and dashed OVERLAP -> magnitude distortion is immune to the delay fix.
  * n, k : phase-derived, so they shift between raw/recentered, but NEITHER is the
           physical mirror (a perfect gold mirror would give n -> large) -> recentering
           changes the numbers without rescuing the measurement.

Run:  PYTHONPATH=. .venv/Scripts/python.exe explorations/reflection_phase_and_windowing/misalignment_recenter_nk_figure.py
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, REPO)

from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz

DATA_DIR = r"C:\Users\Samuel\Data\THz\Sam\reflection_testing\2026-06-25_misalign_tests"
BAND_THZ = (0.3, 2.0)


def _config():
    return {
        "general": {"show_graph": False},
        "geometry": {"theta_external_deg": 45.0, "polarization": "s", "r_reference": -1.0},
        "window": {"type": "hann", "alpha": 1.0, "half_width_ps": 3.0},
        "fft": {"norm": "backward", "amplitude_scale": 1.0, "n_fft": 2000},
        "transfer": {"self_reference": False, "apply_snr_mask": True, "min_ref_amp_rel": 1e-3,
                     "regularization_eps": 1e-30, "unwrap_phase": True},
        "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
        "derive": {"eps_background": 1.0},
    }


def run(recalibrate: bool) -> dict:
    """Process the misalignment series; return {name: dict(freq, H, n, k, mask)}."""
    dataset = DataSet(DATA_DIR, config=_config())
    dataset.load_all_data()
    dataset.group_files(keywords=["type"])
    thz.subtract_baseline(dataset)
    if recalibrate:
        thz.recenter_peaks_to_common_t0(dataset, target_t0_ps=None, subsample=True, show_graph=False)
    thz.window_single_pulse_fixed_width(dataset, half_width_ps=3.0, show_graph=False)
    thz.fft_spectrum(dataset, n_fft=2000)
    thz.transfer_function(dataset, ref_type="reference")
    thz.invert_nk_reflection(dataset, geometry="gold", theta_deg=45.0,
                             polarization="s", r_reference=-1.0)
    out = {}
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        p = data_obj.processing_dict
        out[filename] = dict(freq=p["fft_freq"], H=p["transfer_H"],
                             n=p["n"], k=p["k"], mask=p["transfer_mask"])
    return out


def _mrad(filename: str) -> float:
    match = re.search(r"(minus|plus)-([\d.]+)-mrad", filename)
    if not match:
        return 0.0
    value = float(match.group(2))
    return -value if match.group(1) == "minus" else value


def main():
    raw = run(False)
    recentered = run(True)
    names = sorted(raw, key=_mrad)

    fig, axes = plt.subplots(len(names), 3, figsize=(14, 2.7 * len(names)), squeeze=False)
    fig.suptitle("Angular misalignment: raw (solid) vs recentered (dashed) — "
                 "delay-correction leaves |H| untouched", fontsize=12)
    for row, name in enumerate(names):
        label = f"{_mrad(name):+.1f} mrad"
        for col, key, ylab in ((0, "H", "|H|"), (1, "n", "n"), (2, "k", "k")):
            ax = axes[row][col]
            for source, style in ((raw, "-"), (recentered, "--")):
                rec = source[name]
                f_thz = rec["freq"] * 1e-12
                band = rec["mask"] & (f_thz >= BAND_THZ[0]) & (f_thz <= BAND_THZ[1])
                y = np.abs(rec["H"]) if key == "H" else rec[key]
                ax.plot(f_thz[band], np.asarray(y)[band], style, lw=1.4,
                        label="raw" if style == "-" else "recentered")
            if col == 0:
                ax.axhline(1.0, color="0.6", ls=":", lw=0.8)  # ideal mirror-vs-mirror |H|=1
                ax.set_ylabel(label, fontsize=10, fontweight="bold")
            ax.set_title(f"{ylab}" if row == 0 else "", fontsize=10)
            ax.grid(alpha=0.3)
            if row == len(names) - 1:
                ax.set_xlabel("Frequency (THz)")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    out = os.path.join(HERE, "misalignment_recenter_nk_figure.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved figure: {out}")
    # one-line numeric confirmation that |H| is unchanged by recentering
    print("\n|H| median (raw vs recentered) — should be identical:")
    for name in names:
        def med(src):
            r = src[name]; f_thz = r["freq"] * 1e-12
            b = r["mask"] & (f_thz >= BAND_THZ[0]) & (f_thz <= BAND_THZ[1]) & np.isfinite(r["H"])
            return np.nanmedian(np.abs(r["H"])[b])
        print(f"  {_mrad(name):+6.1f} mrad : raw {med(raw):.3f}  recentered {med(recentered):.3f}")


if __name__ == "__main__":
    main()
