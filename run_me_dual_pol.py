"""Dual-polarisation reflection pipeline: process an s-pol and a p-pol dataset, then jointly fit.

Runs the SAME windowed self-referenced reflection pipeline as ``run_me_low-level.py`` on two
directories (one measured s-pol, one p-pol), pairs the samples, and fits ONE Drude(+gap) model to
both polarisations at once (``thz.fit_dual_pol_reflection`` -> ``core.fit_reflection_gap_dual_pol``).

Why: a single windowed reflection is ill-conditioned (near the r=-1 mirror) and gap-degenerate;
s AND p over-determine one material n(omega) plus one shared contact gap d, which is far more
robust (dual-polarisation study; lab notebook F21).  p-pol also suppresses the parasitic front
reflection (Brewster) and moves the CNT off the mirror pole.

Set the two directories + the shared processing config below and run.  For lower-level control use
``core.invert_nk_reflection(..., polarization='p')`` and ``core.fit_reflection_gap_dual_pol(...)``
directly (both are unit-tested and DataSet-free).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz

# ── Data paths (one directory per polarisation) ─────────────────────────────
data_dir_s = r"C:\Users\Samuel\Data\THz\Sam\2026-06-30_CNT\export\s_pol"
data_dir_p = r"C:\Users\Samuel\Data\THz\Sam\2026-06-30_CNT\export\p_pol"

# ── Shared processing config (mirrors run_me_low-level.py; polarization set per run) ──
config: dict = {
    "general": {"show_graph": False, "save_database": False},
    "geometry": {"theta_external_deg": 45.0, "n_sio2": 1.96},
    "regions": {
        "first_reflection": (146.5, 159.5),
        "second_reflection": (170.4, 186.9),
    },
    "window": {"type": "hann", "alpha": 1.0, "half_width_ps": 4},
    "fft": {"n_fft": 2000},
    "transfer": {
        "self_reference": True,
        "apply_snr_mask": True,
        "min_ref_amp_rel": 1e-3,
        "regularization_eps": 1e-30,
        "unwrap_phase": True,
    },
    "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
    "derive": {"eps_background": 1},
    "dual_pol": {
        # Gap handling: fit d freely by default; set fixed_gap_um to anchor it (e.g. from Si).
        "fixed_gap_um": None,
        "initial_gap_um": 1.0,
        "gap_bounds_um": (0.0, 60.0),
    },
}


def process_polarization(data_dir: str, polarization: str, config: dict) -> DataSet:
    """Run the windowed self-referenced reflection pipeline for one polarisation.

    Mirrors run_me_low-level.py up to and including invert_nk_reflection, which stores the
    measured reflection (reflection_r) and window front reference (r_reference) the joint fit needs.
    """
    run_config = {**config, "geometry": {**config["geometry"], "polarization": polarization}}
    dataset = DataSet(data_dir, config=run_config)
    dataset.load_all_data()

    thz.build_full_trace_reflection(dataset)
    thz.taper_and_pad_traces(dataset)
    dataset.group_files(keywords=["type"])
    thz.define_reflection_regions(dataset, run_config)
    thz.subtract_baseline(dataset)
    thz.window_pulses_fixed_width(
        dataset, half_width_ps=run_config["window"]["half_width_ps"], show_graph=False)

    required_n_fft = thz.minimum_fft_length(dataset)
    shared_n_fft = run_config["fft"]["n_fft"]
    assert shared_n_fft >= required_n_fft, (
        f"n_fft={shared_n_fft} would truncate the windowed trace ({required_n_fft} samples). "
        f"Set n_fft >= {required_n_fft}."
    )
    thz.fft_spectrum(dataset, segment="second_reflection", n_fft=shared_n_fft)
    thz.fft_spectrum(dataset, segment="first_reflection", n_fft=shared_n_fft)
    thz.transfer_function(dataset, ref_type="reference")
    thz.selfref_quality(dataset)
    thz.invert_nk_reflection(
        dataset,
        geometry="window",
        theta_deg=run_config["geometry"]["theta_external_deg"],
        polarization=polarization,
        n_window=run_config["geometry"]["n_sio2"],
    )
    return dataset


def main() -> None:
    print("===== s-pol =====")
    dataset_s = process_polarization(data_dir_s, "s", config)
    print("===== p-pol =====")
    dataset_p = process_polarization(data_dir_p, "p", config)

    print("===== joint dual-pol Drude + gap fit =====")
    fits = thz.fit_dual_pol_reflection(dataset_s, dataset_p, config)

    if config["general"].get("show_graph", False):
        for key, fit in fits.items():
            frequency_thz = fit["frequency_hz"] * 1e-12
            band = fit["mask"]
            figure, (ax_n, ax_k) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
            figure.suptitle(f"Dual-pol joint fit — {key} (gap {fit['gap_um']:.1f} um)")
            ax_n.plot(frequency_thz[band], fit["n"][band], "C2", label="joint-fit n")
            ax_n.set_xlabel("THz"); ax_n.set_ylabel("n"); ax_n.legend(); ax_n.grid(alpha=0.3)
            ax_k.plot(frequency_thz[band], fit["k"][band], "C3", label="joint-fit k")
            ax_k.set_xlabel("THz"); ax_k.set_ylabel("k"); ax_k.legend(); ax_k.grid(alpha=0.3)
            plt.show()

    return fits


if __name__ == "__main__":
    main()
