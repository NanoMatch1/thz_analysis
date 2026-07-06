"""De-embed validation on the two silicon controls (known n = 3.418).

For Si-p (new FZ, P-pol) and Si-s-old (S-pol): estimate the contact gap with the
phase-excess estimator (Hilbert and MEM engines), de-embed, re-invert with air incidence at
the gap angle, and compare against the ORACLE gap — the d that best flattens n onto 3.418.
If estimator ~ oracle and the de-embedded n is flat at 3.418, the de-embed chain is
validated end-to-end on real data.

Run from repo root:
  PYTHONPATH=".;./explorations/air_gap_cnt_reflection" ./.venv/Scripts/python.exe \
      explorations/air_gap_cnt_reflection/deembed_si_controls.py
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thz_core.thz_core as core
from explore_air_gap_deembedding import deembed_gap_layer, remove_round_trip_phase
from deembed_air_gap_iterative import estimate_gap_minimum_phase

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "cnt21_channel_results")
N_SILICON = 3.418
N_SIO2 = 1.96
THETA_EXTERNAL = np.deg2rad(45.0)
THETA_GAP = np.deg2rad(45.0)          # Snell returns the beam to 45 deg in the air gap
FIT_BAND_HZ = (0.4e12, 2.2e12)        # band for the phase-excess linear fit
SCORE_BAND_HZ = (0.5e12, 2.0e12)      # band for the oracle |n - 3.418| score
CROP_BAND_HZ = (0.15e12, 3.0e12)      # protect the MEM AR fit from out-of-band noise

CHANNELS = ["Si-p", "Si-s-old"]


def invert_air_gap_frame(freq, x_deembedded, mask, polarization):
    """Invert r_back (air incidence at the gap angle) with the right polarisation."""
    n_values, k_values, _ = core.invert_nk_reflection(
        freq, x_deembedded, mask, theta_rad=THETA_GAP, n_incident=1.0,
        polarization=polarization)
    return n_values, k_values


def flatness_score(freq, n_values, mask):
    band = mask & (freq >= SCORE_BAND_HZ[0]) & (freq <= SCORE_BAND_HZ[1]) & np.isfinite(n_values)
    if not band.any():
        return np.inf
    return float(np.nanmedian(np.abs(n_values[band] - N_SILICON)))


def main():
    figure, axes = plt.subplots(len(CHANNELS), 2, figsize=(13, 4.6 * len(CHANNELS)),
                                layout="constrained", squeeze=False)
    figure.suptitle("Si controls: gap de-embed vs oracle (target n = 3.418)")

    for row, channel in enumerate(CHANNELS):
        data = np.load(os.path.join(RESULTS_DIR, f"channel_{channel}.npz"))
        freq = data["freq"]; mask = data["mask"].astype(bool)
        r_meas = data["reflection_r"]; r_front = complex(data["r_reference"])
        polarization = str(data["polarization"])
        n_naive = data["n"]; k_naive = data["k"]

        crop = (freq >= CROP_BAND_HZ[0]) & (freq <= CROP_BAND_HZ[1])
        freq_c = freq[crop]; mask_c = mask[crop]; r_c = r_meas[crop]
        x_c = deembed_gap_layer(r_c, r_front)

        print(f"\n===== {channel} (pol {polarization}, r_front {r_front:+.3f}) =====")

        # -- gap estimators ----------------------------------------------------
        estimates = {}
        for engine in ("hilbert", "mem"):
            d_est, _ = estimate_gap_minimum_phase(
                freq_c, x_c, mask_c, THETA_GAP, FIT_BAND_HZ, phase_engine=engine)
            estimates[engine] = d_est
            print(f"  phase-excess ({engine:7s}): d = {d_est*1e6:+7.2f} um")

        # -- oracle: grid-search the d that best flattens n onto silicon --------
        gap_grid_um = np.linspace(-10.0, 40.0, 201)
        scores = []
        for gap_um in gap_grid_um:
            x_shift = remove_round_trip_phase(x_c, freq_c, gap_um * 1e-6, THETA_GAP)
            n_test, _ = invert_air_gap_frame(freq_c, x_shift, mask_c, polarization)
            scores.append(flatness_score(freq_c, n_test, mask_c))
        oracle_d_um = float(gap_grid_um[int(np.argmin(scores))])
        print(f"  ORACLE (best |n-3.418|) : d = {oracle_d_um:+7.2f} um "
              f"(score {np.min(scores):.3f})")

        # -- de-embedded spectra for plotting -----------------------------------
        curves = {"naive (window frame)": (freq, n_naive, k_naive, mask)}
        for label, d_value in (("hilbert", estimates["hilbert"]),
                               ("mem", estimates["mem"]),
                               ("oracle", oracle_d_um * 1e-6)):
            x_shift = remove_round_trip_phase(x_c, freq_c, d_value, THETA_GAP)
            n_de, k_de = invert_air_gap_frame(freq_c, x_shift, mask_c, polarization)
            curves[f"{label} d={d_value*1e6:+.1f}um"] = (freq_c, n_de, k_de, mask_c)
            print(f"    {label:8s}: median n {flatness_score(freq_c, n_de, mask_c) + 0:.3f} "
                  f"|n-3.418| in band")

        ax_n, ax_k = axes[row]
        for (label, (f_arr, n_arr, k_arr, m_arr)), color in zip(
                curves.items(), ("0.6", "C3", "C0", "C2")):
            f_thz = f_arr * 1e-12
            band = m_arr & (f_thz >= 0.3) & (f_thz <= 2.4) & np.isfinite(n_arr)
            style = dict(color=color, lw=1.5, ls="--" if label.startswith("naive") else "-")
            ax_n.plot(f_thz[band], n_arr[band], label=label, **style)
            ax_k.plot(f_thz[band], k_arr[band], label=label, **style)
        ax_n.axhline(N_SILICON, color="k", ls=":", lw=1.0, label="n_Si=3.418")
        ax_n.set_title(f"{channel}: n", fontsize=10); ax_n.set_ylim(0, 5)
        ax_k.set_title(f"{channel}: k", fontsize=10)
        for ax in (ax_n, ax_k):
            ax.set_xlabel("THz"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    out = os.path.join(RESULTS_DIR, "deembed_si_controls.png")
    figure.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
