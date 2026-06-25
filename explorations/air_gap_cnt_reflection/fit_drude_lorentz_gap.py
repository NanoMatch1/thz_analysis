"""Fit the rough-gap models (A: statistical-gap FP, B: graded EMT) to the latest
well-pressed CNT reflection, with a DRUDE (not Drude-Smith) + LORENTZ material.

Question (Samuel, 2026-06-25): the CNTs now press in with almost no gap delay (~25-50 fs),
but a Lorentzian-like peak in n is CONSISTENTLY present across the latest datasets. Is it
real material response, or a gap/roughness artefact?

Strategy — fit four combinations per sample and compare:
    {pure Drude, Drude + 1 Lorentz oscillator} x {Route A gap, Route B gap}
and judge the Lorentzian by two independent tells:
  (1) does adding the oscillator MATERIALLY lower the fit RMS (vs pure Drude)?
  (2) do the TWO very different gap models (A flat-gap FP, B graded EMT) INDEPENDENTLY
      converge on the SAME resonance frequency w0?  A shared spurious feature across two
      unrelated gap physics is unlikely, so agreement => real; scatter / vanishing strength
      => the gap can absorb it (artefact-suspect).

Builds the measured reflection with the CURRENT pipeline (fixed-width window, symmetric
phase ref, self-referencing) via air_gap_slider_explorer.samples_from_dataset, so it reflects
every latest correction. reflection_r from the pipeline IS r_back,sample (SiO2-side), exactly
what the gap forward models output; r_front = SiO2->air Fresnel.

Run:  PYTHONPATH=. .venv/Scripts/python.exe explorations/air_gap_cnt_reflection/fit_drude_lorentz_gap.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz
import air_gap_models as models
from air_gap_slider_explorer import samples_from_dataset, internal_and_gap_angles

VACUUM_PERMITTIVITY = models.VACUUM_PERMITTIVITY
DATA_DIR = r"C:\Users\Samuel\Data\THz\Sam\2026-06-24_refl_testing\main_alignment_tests\CNT\de-embed_testing"
DISPLAY_BAND_THZ = (0.2, 2.5)

PIPELINE_CONFIG = {
    "general": {"show_graph": False},
    "geometry": {"theta_external_deg": 45.0, "polarization": "s", "n_sio2": 1.95},
    "regions": {"first_reflection": (152.5, 159), "second_reflection": (177, 183.8)},
    "centering": {"mode": "crop"},
    "window": {"type": "hann", "alpha": 1.0, "half_width_ps": 3.0},
    "fft": {"norm": "backward", "amplitude_scale": 1.0, "n_fft": 2000},
    "transfer": {"self_reference": True, "apply_snr_mask": True, "min_ref_amp_rel": 1e-3,
                 "regularization_eps": 1e-30, "unwrap_phase": True},
    "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
    "derive": {"eps_background": 1.0},
}


def build_measured_samples(data_dir):
    """Run the CURRENT reflection pipeline and return slider-style sample records."""
    config = dict(PIPELINE_CONFIG)
    dataset = DataSet(data_dir, config=config)
    dataset.load_all_data()
    thz.build_full_trace_reflection(dataset)
    dataset.group_files(keywords=["type"])
    thz.define_reflection_regions(dataset, config)
    thz.subtract_baseline(dataset)
    thz.window_pulses_fixed_width(dataset, half_width_ps=config["window"]["half_width_ps"],
                                  show_graph=False)
    n_fft = config["fft"]["n_fft"]
    thz.fft_spectrum(dataset, segment="second_reflection", n_fft=n_fft)
    thz.fft_spectrum(dataset, segment="first_reflection", n_fft=n_fft)
    thz.transfer_function(dataset, ref_type="reference")
    thz.invert_nk_reflection(dataset, geometry="window",
                             theta_deg=config["geometry"]["theta_external_deg"],
                             polarization=config["geometry"]["polarization"],
                             n_window=config["geometry"]["n_sio2"])
    return samples_from_dataset(dataset)


# ── Material permittivity from a natural-unit vector ─────────────────────────
# Material block: [eps_inf, plasma_THz, tau_fs]  (+ [lor_strength, w0_THz, gamma_THz] if Lorentz)
# Gap block A: [d_mean_um, sigma_d_um];  gap block B: [sigma_h_um, z0_um].

def material_eps(frequency_hz, material_vector, with_lorentz):
    eps_inf, plasma_thz, tau_fs = material_vector[0], material_vector[1], material_vector[2]
    eps = models.drude_permittivity(frequency_hz, eps_inf, 2 * np.pi * plasma_thz * 1e12, tau_fs * 1e-15)
    if with_lorentz:
        strength, w0_thz, gamma_thz = material_vector[3], material_vector[4], material_vector[5]
        eps = eps + models.lorentz_permittivity_term(
            frequency_hz, strength, 2 * np.pi * w0_thz * 1e12, 2 * np.pi * gamma_thz * 1e12)
    return eps


def _n_material(with_lorentz):
    return 6 if with_lorentz else 3


def forward(route, vector, with_lorentz, frequency_hz, r_front, gap_angle_rad, sio2_angle_rad):
    n_mat = _n_material(with_lorentz)
    eps = material_eps(frequency_hz, vector[:n_mat], with_lorentz)
    gap1, gap2 = vector[n_mat], vector[n_mat + 1]
    if route == "A":
        return models.rough_gap_reflection_from_eps(
            frequency_hz, eps, gap1 * 1e-6, gap2 * 1e-6, r_front, gap_angle_rad)
    return models.graded_emt_reflection_from_eps(
        frequency_hz, eps, gap1 * 1e-6, gap2 * 1e-6, sio2_angle_rad)


def conductivity(frequency_hz, vector, with_lorentz, eps_background=1.0):
    """sigma = -i w eps0 (eps - eps_background) over the FULL (Drude+Lorentz) permittivity."""
    eps = material_eps(frequency_hz, vector[:_n_material(with_lorentz)], with_lorentz)
    omega = 2.0 * np.pi * frequency_hz
    return -1j * omega * VACUUM_PERMITTIVITY * (eps - eps_background)


def nk(frequency_hz, vector, with_lorentz):
    n_hat = models.index_hat_from_eps(material_eps(frequency_hz, vector[:_n_material(with_lorentz)], with_lorentz))
    return np.real(n_hat), -np.imag(n_hat)


# ── Bounds / initial guesses ─────────────────────────────────────────────────
DRUDE_LOW = [1.0, 0.5, 3.0]
DRUDE_HIGH = [30.0, 60.0, 500.0]
DRUDE_INIT = [4.0, 10.0, 30.0]
LORENTZ_LOW = [0.0, 0.2, 0.05]
LORENTZ_HIGH = [60.0, 3.0, 4.0]
LORENTZ_INIT = [3.0, 0.8, 0.5]
GAP_A_LOW, GAP_A_HIGH, GAP_A_INIT = [0.0, 0.1], [40.0, 30.0], [5.0, 3.0]
GAP_B_LOW, GAP_B_HIGH, GAP_B_INIT = [0.5, 0.0], [30.0, 40.0], [3.0, 5.0]


def assemble(route, with_lorentz):
    low = list(DRUDE_LOW); high = list(DRUDE_HIGH); init = list(DRUDE_INIT)
    if with_lorentz:
        low += LORENTZ_LOW; high += LORENTZ_HIGH; init += LORENTZ_INIT
    gap_low, gap_high, gap_init = (
        (GAP_A_LOW, GAP_A_HIGH, GAP_A_INIT) if route == "A" else (GAP_B_LOW, GAP_B_HIGH, GAP_B_INIT))
    return init + gap_init, (low + gap_low, high + gap_high)


def fit(route, with_lorentz, record, r_front, gap_angle_rad, sio2_angle_rad):
    freq = record["frequency_hz"]; r_meas = record["r_meas"]
    f_thz = freq * 1e-12
    init, bounds = assemble(route, with_lorentz)
    # The Drude permittivity ~1/omega diverges toward DC, so confine the fit to a positive
    # band and drop any bin where the initial model is non-finite (keeps the residual length
    # fixed and finite across the optimisation).
    init_model = forward(route, init, with_lorentz, freq, r_front, gap_angle_rad, sio2_angle_rad)
    band = (record["mask"] & np.isfinite(r_meas) & np.isfinite(init_model)
            & (f_thz >= DISPLAY_BAND_THZ[0]) & (f_thz <= DISPLAY_BAND_THZ[1]))

    def residual(vector):
        model = forward(route, vector, with_lorentz, freq, r_front, gap_angle_rad, sio2_angle_rad)
        diff = (model - r_meas)[band]
        return np.concatenate([diff.real, diff.imag])

    result = least_squares(residual, init, bounds=bounds, method="trf", x_scale="jac", max_nfev=6000)
    rms = float(np.sqrt(np.mean(result.fun**2)))
    return result.x, rms


def main():
    sio2_angle_rad, gap_angle_rad = internal_and_gap_angles()
    r_front = models.reflection_sio2_to_air(sio2_angle_rad)
    print(f"data: {DATA_DIR}")
    print(f"r_front (SiO2->air) = {r_front:.3f}; gap angle {np.rad2deg(gap_angle_rad):.2f} deg")
    samples = build_measured_samples(DATA_DIR)
    print(f"samples: {[s['name'] for s in samples]}\n")

    rows = []
    for record in samples:
        name = record["name"]
        print(f"=== {name} ===")
        fits = {}
        for route in ("A", "B"):
            for with_lorentz in (False, True):
                vec, rms = fit(route, with_lorentz, record, r_front, gap_angle_rad, sio2_angle_rad)
                fits[(route, with_lorentz)] = (vec, rms)
                n_mat = _n_material(with_lorentz)
                tag = "Drude+Lorentz" if with_lorentz else "Drude       "
                gap_label = ("d_mean %.1f sig_d %.1f um" % (vec[n_mat], vec[n_mat + 1]) if route == "A"
                             else "sig_h %.1f z0 %.1f um" % (vec[n_mat], vec[n_mat + 1]))
                lor = ("  | Lorentz w0 %.2f THz, strength %.2f, gamma %.2f THz"
                       % (vec[4], vec[3], vec[5])) if with_lorentz else ""
                print(f"  Route {route} {tag}: RMS {rms:.2e}  "
                      f"eps_inf {vec[0]:.1f}, wp {vec[1]:.1f} THz, tau {vec[2]:.0f} fs, {gap_label}{lor}")
        # verdicts
        for route in ("A", "B"):
            rms_d = fits[(route, False)][1]; rms_dl = fits[(route, True)][1]
            improve = 100 * (rms_d - rms_dl) / rms_d
            print(f"  -> Route {route}: Lorentz changes RMS by {improve:+.0f}% "
                  f"({rms_d:.2e} -> {rms_dl:.2e})")
        w0_A = fits[("A", True)][0][4]; w0_B = fits[("B", True)][0][4]
        print(f"  -> A vs B Lorentz w0: {w0_A:.2f} vs {w0_B:.2f} THz "
              f"(|diff| {abs(w0_A - w0_B):.2f})\n")
        rows.append((record, fits))

    make_figure(rows, r_front, gap_angle_rad, sio2_angle_rad)
    print("Reminder: single measurement is degeneracy-limited (conditioning report). "
          "The A/B w0 agreement + RMS-improvement test is the robust read here.")


def make_figure(rows, r_front, gap_angle_rad, sio2_angle_rad):
    fig, axes = plt.subplots(len(rows), 4, figsize=(19, 4.7 * len(rows)), squeeze=False)
    fig.suptitle("Drude vs Drude+Lorentz x Route A/B — is the Lorentzian real?", fontsize=13)
    for row, (record, fits) in enumerate(rows):
        freq = record["frequency_hz"]; f_thz = freq * 1e-12
        band = record["mask"] & (f_thz >= DISPLAY_BAND_THZ[0]) & (f_thz <= DISPLAY_BAND_THZ[1])
        styles = {("A", False): ("C0", "-", "A Drude"), ("A", True): ("C0", "--", "A D+Lor"),
                  ("B", False): ("C3", "-", "B Drude"), ("B", True): ("C3", "--", "B D+Lor")}

        ax = axes[row][0]
        ax.plot(f_thz[band], record["n_naive"][band], "0.6", ls=":", label="naive")
        for key, (vec, _rms) in fits.items():
            n, _k = nk(freq, vec, key[1]); c, ls, lab = styles[key]
            ax.plot(f_thz[band], n[band], color=c, ls=ls, label=lab)
        ax.axhline(1.0, color="0.5", ls=":", lw=0.8); ax.set_ylim(0, 6)
        ax.set_title(f"{record['name']}: n"); ax.set_xlabel("THz"); ax.set_ylabel("n")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[row][1]
        ax.plot(f_thz[band], record["k_naive"][band], "0.6", ls=":", label="naive")
        for key, (vec, _rms) in fits.items():
            _n, k = nk(freq, vec, key[1]); c, ls, _lab = styles[key]
            ax.plot(f_thz[band], k[band], color=c, ls=ls)
        ax.set_title("k"); ax.set_xlabel("THz"); ax.set_ylabel("k"); ax.grid(alpha=0.3)

        ax = axes[row][2]
        for key, (vec, _rms) in fits.items():
            sig = conductivity(freq, vec, key[1]); c, ls, lab = styles[key]
            ax.plot(f_thz[band], np.real(sig)[band], color=c, ls=ls, label=lab)
        ax.set_title("Re sigma (S/m)"); ax.set_xlabel("THz"); ax.set_ylabel("S/m")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[row][3]
        ax.plot(f_thz[band], np.abs(record["r_meas"])[band], "k", lw=1.7, label="measured |r|")
        for key, (vec, rms) in fits.items():
            model = forward(key[0], vec, key[1], freq, r_front, gap_angle_rad, sio2_angle_rad)
            c, ls, lab = styles[key]
            ax.plot(f_thz[band], np.abs(model)[band], color=c, ls=ls,
                    label=f"{lab} ({rms:.1e})")
        ax.set_title("|r| fit quality"); ax.set_xlabel("THz"); ax.set_ylabel("|r|")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    out = os.path.join(HERE, "fit_drude_lorentz_gap.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
