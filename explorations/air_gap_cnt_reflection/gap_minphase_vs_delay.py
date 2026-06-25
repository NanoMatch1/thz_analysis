"""Separate the gap delay from the material group delay on the good-contact CNT.

Two tasks (Samuel, 2026-06-25):
  (1) QUANTIFY the material contamination: a reflected pulse's delay = group delay =
      geometric gap (2d cosθ/c) + the CNT's own reflection-phase group delay
      -d(arg r_back)/dω. Compute the Drude material group delay and see what fraction of
      the measured ~25-50 fs is material vs gap.
  (2) MINIMUM-PHASE gap extraction: |r_back| fixes the material's causal phase via
      Hilbert(ln|r|); the EXCESS over that is the pure gap (-2beta). This is the ONE method
      that separates them (unlike Denís's slope-flattening, which lumps both into t_delay).

Identity used: arg(x) = φ_material(min-phase, from |x|) + excess(-2beta, gap). Slopes add, so
  tau_total(raw arg-slope)  =  tau_material(min-phase)  +  tau_gap(excess).
tau_total is what a slope-flattening / arg-x method calls "the delay"; only tau_gap is the gap.

Run:  PYTHONPATH=. .venv/Scripts/python.exe explorations/air_gap_cnt_reflection/gap_minphase_vs_delay.py
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

import air_gap_models as models
from air_gap_slider_explorer import internal_and_gap_angles
from deembed_air_gap_iterative import minimum_phase_from_magnitude
from fit_drude_lorentz_gap import build_measured_samples, DATA_DIR

SPEED_OF_LIGHT = 299_792_458.0
BAND_THZ = (0.3, 2.0)


def group_delay_fs(omega, phase, band):
    """tau_g = -dφ/dω over the band, in fs (phase already unwrapped)."""
    slope = np.polyfit(omega[band], phase[band], 1)[0]   # rad per (rad/s) = s
    return -slope * 1e15


def gap_from_delay(tau_seconds, gap_angle_rad):
    """d (um) from a gap round-trip delay tau = 2 d cosθ_gap / c."""
    return tau_seconds * SPEED_OF_LIGHT / (2.0 * np.cos(gap_angle_rad)) * 1e6


def fit_drude_to_rback(freq, r_back, band, gap_angle_rad):
    def residual(v):
        eps = models.drude_permittivity(freq, v[0], 2 * np.pi * v[1] * 1e12, v[2] * 1e-15)
        model = models.reflection_air_to_cnt_from_eps(freq, eps, gap_angle_rad)
        diff = (model - r_back)[band]
        return np.concatenate([diff.real, diff.imag])
    result = least_squares(residual, [4.0, 12.0, 30.0], bounds=([1, 0.5, 3], [40, 80, 500]),
                           x_scale="jac", max_nfev=6000)
    return result.x


def main():
    sio2_angle_rad, gap_angle_rad = internal_and_gap_angles()
    cos_gap = np.cos(gap_angle_rad)
    samples = build_measured_samples(DATA_DIR)   # raw r_meas, r_front (no de-embed yet)
    print(f"data: {DATA_DIR}\ngap angle {np.rad2deg(gap_angle_rad):.1f} deg\n")

    fig, axes = plt.subplots(len(samples), 2, figsize=(13, 5 * len(samples)), squeeze=False)
    for row, record in enumerate(samples):
        name = record["name"]
        freq = record["frequency_hz"]; f_thz = freq * 1e-12; omega = 2 * np.pi * freq
        r_meas = record["r_meas"]; r_front = record["r_front"]; mask = record["mask"]
        band = mask & (f_thz >= BAND_THZ[0]) & (f_thz <= BAND_THZ[1]) & np.isfinite(r_meas)

        # de-embed the SiO2/air front interface -> x = r_back * e^{-i2beta}
        x = (r_meas - r_front) / (1.0 - r_front * r_meas)
        arg_x = np.unwrap(np.angle(x))

        # minimum-phase material phase from |x| (causality), restricted to the band
        material_phase = np.full_like(arg_x, np.nan)
        material_phase[band] = minimum_phase_from_magnitude(np.abs(x[band]))
        excess = arg_x - material_phase    # should be the pure gap -2beta where defined

        # group delays (fs) over the band
        tau_total = group_delay_fs(omega, arg_x, band)
        tau_material = group_delay_fs(omega, np.nan_to_num(material_phase), band)
        tau_gap = group_delay_fs(omega, np.nan_to_num(excess), band)

        # gaps (um) implied by the biased (arg-x) vs min-phase (excess) delays
        d_argx = gap_from_delay(tau_total * 1e-15, gap_angle_rad)
        d_minphase = gap_from_delay(tau_gap * 1e-15, gap_angle_rad)

        # Task 1 cross-check: fit a Drude to the min-phase de-embedded r_back, get ITS group delay
        r_back_mp = x * np.exp(1j * omega / SPEED_OF_LIGHT * 2 * (d_minphase * 1e-6) * cos_gap)
        drude_vec = fit_drude_to_rback(freq, r_back_mp, band, gap_angle_rad)
        eps_drude = models.drude_permittivity(freq, drude_vec[0], 2 * np.pi * drude_vec[1] * 1e12,
                                              drude_vec[2] * 1e-15)
        r_drude = models.reflection_air_to_cnt_from_eps(freq, eps_drude, gap_angle_rad)
        tau_material_drude = group_delay_fs(omega, np.unwrap(np.angle(r_drude)), band)

        print(f"=== {name} ===")
        print(f"  tau_total  (raw arg-x slope; what slope-flattening calls 'the delay') = {tau_total:7.1f} fs")
        print(f"  tau_material (min-phase, from |r|)                                    = {tau_material:7.1f} fs")
        print(f"  tau_gap    (excess = pure geometric gap)                              = {tau_gap:7.1f} fs")
        print(f"     check: tau_material + tau_gap = {tau_material + tau_gap:7.1f} fs  (= tau_total)")
        print(f"  tau_material (independent Drude fit cross-check)                      = {tau_material_drude:7.1f} fs")
        print(f"  -> gap thickness:  arg-x (biased) d = {d_argx:5.2f} um   "
              f"min-phase d = {d_minphase:5.2f} um")
        material_fraction = 100 * tau_material / tau_total if tau_total != 0 else float("nan")
        print(f"  -> material is {material_fraction:.0f}% of the raw delay; "
              f"gap is {100 - material_fraction:.0f}%\n")

        ax = axes[row][0]
        ax.plot(f_thz[band], arg_x[band], "k", label="arg(x) total")
        ax.plot(f_thz[band], material_phase[band], "C0", label="min-phase material (from |r|)")
        ax.plot(f_thz[band], excess[band], "C3", label="excess = gap (-2beta)")
        ax.set_title(f"{name[:24]}: phase decomposition"); ax.set_xlabel("THz")
        ax.set_ylabel("phase (rad)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row][1]
        bars = ["tau_total\n(arg-x)", "tau_material\n(min-phase)", "tau_gap\n(min-phase)",
                "tau_material\n(Drude)"]
        vals = [tau_total, tau_material, tau_gap, tau_material_drude]
        ax.bar(bars, vals, color=["k", "C0", "C3", "C2"])
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title("group-delay budget (fs)"); ax.set_ylabel("fs"); ax.grid(alpha=0.3, axis="y")

    out = os.path.join(HERE, "gap_minphase_vs_delay.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved figure: {out}")
    print("Reading: tau_total is what Denís's slope-flattening / the arg-x method would remove as "
          "'the delay'. Only tau_gap is the real gap; tau_material is CNT dispersion that must NOT be "
          "stripped. Min-phase splits them using causality.")


if __name__ == "__main__":
    main()
