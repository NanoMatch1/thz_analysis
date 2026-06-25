"""Fit a pure DRUDE to the Route-A de-embedded CNT reflection — "where are we at?".

Runs the current low-level reflection pipeline INCLUDING the air-gap de-embed step
(thz.deembed_air_gap_reflection), pulls the de-embedded reflection r_back, and fits a bare
Drude (eps_inf, omega_p, tau) by forward-modelling r_back = r_{air->CNT}(Drude). Fitting the
REFLECTION (not the inverted n,k) keeps it robust to the near-|r|=1 inversion sensitivity.

Reports the Drude parameters + DC conductivity, and plots: de-embedded n,k,Re sigma with the
Drude curve overlaid, and the |r_back| fit quality. Change the gap in PIPELINE_CONFIG['air_gap']
(or dial it in the slider) to see how the Drude fit responds.

Run:  PYTHONPATH=. .venv/Scripts/python.exe explorations/air_gap_cnt_reflection/fit_drude_deembedded.py
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
from air_gap_slider_explorer import internal_and_gap_angles, conductivity_from_nk

VACUUM_PERMITTIVITY = models.VACUUM_PERMITTIVITY
DATA_DIR = r"C:\Users\Samuel\Data\THz\Sam\2026-06-24_refl_testing\main_alignment_tests\CNT\de-embed_testing"
DISPLAY_BAND_THZ = (0.2, 2.5)

PIPELINE_CONFIG = {
    "general": {"show_graph": False},
    "geometry": {"theta_external_deg": 45.0, "polarization": "s", "n_sio2": 1.95},
    "regions": {"first_reflection": (152.5, 159), "second_reflection": (177, 183.8)},
    "window": {"type": "hann", "alpha": 1.0, "half_width_ps": 3.0},
    "fft": {"norm": "backward", "amplitude_scale": 1.0, "n_fft": 2000},
    "transfer": {"self_reference": True, "apply_snr_mask": True, "min_ref_amp_rel": 1e-3,
                 "regularization_eps": 1e-30, "unwrap_phase": True},
    "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
    "air_gap": {"enabled": True, "position_um": 5.0, "width_um": 0.0, "max_boost": 1.0e3},
    "derive": {"eps_background": 1.0},
}

# Drude natural-unit vector: [eps_inf, plasma_THz, tau_fs]
DRUDE_LOW = [1.0, 0.5, 3.0]
DRUDE_HIGH = [40.0, 80.0, 500.0]
DRUDE_INIT = [4.0, 12.0, 30.0]


def build_deembedded_samples(data_dir):
    """Run the current pipeline + air-gap de-embed; return per-sample records."""
    config = dict(PIPELINE_CONFIG)
    dataset = DataSet(data_dir, config=config)
    dataset.load_all_data()
    thz.build_full_trace_reflection(dataset)
    dataset.group_files(keywords=["type"])
    thz.define_reflection_regions(dataset, config)
    thz.subtract_baseline(dataset)
    thz.window_pulses_fixed_width(dataset, half_width_ps=config["window"]["half_width_ps"],
                                  show_graph=False)
    thz.fft_spectrum(dataset, segment="second_reflection", n_fft=config["fft"]["n_fft"])
    thz.fft_spectrum(dataset, segment="first_reflection", n_fft=config["fft"]["n_fft"])
    thz.transfer_function(dataset, ref_type="reference")
    thz.invert_nk_reflection(dataset, geometry="window",
                             theta_deg=config["geometry"]["theta_external_deg"],
                             polarization=config["geometry"]["polarization"],
                             n_window=config["geometry"]["n_sio2"])
    thz.deembed_air_gap_reflection(dataset)
    records = []
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        p = data_obj.processing_dict
        records.append(dict(
            name=filename,
            frequency_hz=np.asarray(p["fft_freq"], float),
            mask=np.asarray(p["transfer_mask"], bool),
            r_back=np.asarray(p["reflection_r_deembedded"], complex),
            n_deembed=np.asarray(p["n"], float),
            k_deembed=np.asarray(p["k"], float),
        ))
    return records


def drude_eps(frequency_hz, vector):
    return models.drude_permittivity(frequency_hz, vector[0], 2 * np.pi * vector[1] * 1e12, vector[2] * 1e-15)


def fit_drude(record, gap_angle_rad):
    freq = record["frequency_hz"]; r_back = record["r_back"]; f_thz = freq * 1e-12
    band = (record["mask"] & np.isfinite(r_back)
            & (f_thz >= DISPLAY_BAND_THZ[0]) & (f_thz <= DISPLAY_BAND_THZ[1]))

    def residual(vector):
        model = models.reflection_air_to_cnt_from_eps(freq, drude_eps(freq, vector), gap_angle_rad)
        diff = (model - r_back)[band]
        return np.concatenate([diff.real, diff.imag])

    result = least_squares(residual, DRUDE_INIT, bounds=(DRUDE_LOW, DRUDE_HIGH),
                           method="trf", x_scale="jac", max_nfev=6000)
    return result.x, float(np.sqrt(np.mean(result.fun**2)))


def main():
    sio2_angle_rad, gap_angle_rad = internal_and_gap_angles()
    gap_um = PIPELINE_CONFIG["air_gap"]["position_um"]
    print(f"data: {DATA_DIR}")
    print(f"de-embed gap: d_mean {gap_um} um, sigma_d {PIPELINE_CONFIG['air_gap']['width_um']} um; "
          f"gap angle {np.rad2deg(gap_angle_rad):.1f} deg\n")
    records = build_deembedded_samples(DATA_DIR)

    fig, axes = plt.subplots(len(records), 4, figsize=(19, 4.7 * len(records)), squeeze=False)
    fig.suptitle(f"Drude fit to Route-A de-embedded CNT (gap d={gap_um} um)", fontsize=13)
    for row, record in enumerate(records):
        vec, rms = fit_drude(record, gap_angle_rad)
        eps_inf, plasma_thz, tau_fs = vec
        sigma_dc = VACUUM_PERMITTIVITY * (2 * np.pi * plasma_thz * 1e12) ** 2 * (tau_fs * 1e-15)
        print(f"=== {record['name']} ===")
        print(f"  Drude fit: eps_inf {eps_inf:.2f}, omega_p {plasma_thz:.2f} THz, tau {tau_fs:.0f} fs")
        print(f"             sigma_DC = {sigma_dc:.0f} S/m,  RMS(|r| residual) = {rms:.2e}")

        freq = record["frequency_hz"]; f_thz = freq * 1e-12
        band = record["mask"] & (f_thz >= DISPLAY_BAND_THZ[0]) & (f_thz <= DISPLAY_BAND_THZ[1])
        eps = drude_eps(freq, vec)
        n_hat = models.index_hat_from_eps(eps)
        n_fit, k_fit = np.real(n_hat), -np.imag(n_hat)
        sigma_fit = conductivity_from_nk(freq, n_fit, k_fit, 1.0)
        sigma_deembed = conductivity_from_nk(freq, record["n_deembed"], record["k_deembed"], 1.0)
        r_model = models.reflection_air_to_cnt_from_eps(freq, eps, gap_angle_rad)

        ax = axes[row][0]
        ax.plot(f_thz[band], record["n_deembed"][band], "k", lw=1.5, label="de-embedded")
        ax.plot(f_thz[band], n_fit[band], "C0--", label="Drude fit")
        ax.axhline(1.0, color="0.6", ls=":", lw=0.8); ax.set_ylim(0, 6)
        ax.set_title(f"{record['name'][:26]}: n"); ax.set_xlabel("THz"); ax.set_ylabel("n")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row][1]
        ax.plot(f_thz[band], record["k_deembed"][band], "k", lw=1.5, label="de-embedded")
        ax.plot(f_thz[band], k_fit[band], "C0--", label="Drude fit")
        ax.set_title("k"); ax.set_xlabel("THz"); ax.set_ylabel("k"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row][2]
        ax.plot(f_thz[band], np.real(sigma_deembed)[band], "k", lw=1.5, label="de-embedded")
        ax.plot(f_thz[band], np.real(sigma_fit)[band], "C0--", label="Drude fit")
        ax.set_title("Re sigma (S/m)"); ax.set_xlabel("THz"); ax.set_ylabel("S/m")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row][3]
        ax.plot(f_thz[band], np.abs(record["r_back"])[band], "k", lw=1.6, label="de-embedded |r_back|")
        ax.plot(f_thz[band], np.abs(r_model)[band], "C0--", label=f"Drude (RMS {rms:.1e})")
        ax.set_title("|r_back| fit quality"); ax.set_xlabel("THz"); ax.set_ylabel("|r|")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    out = os.path.join(HERE, "fit_drude_deembedded.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure: {out}")
    print("Note: gap d is not yet calibrated (Si benchmark pending) — vary air_gap.position_um "
          "to see the Drude fit / sigma_DC respond.")


if __name__ == "__main__":
    main()
