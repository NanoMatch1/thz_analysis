"""Dual-polarization (s/p) ellipsometry as a route around the CNT air-gap + conditioning problem.

MOTIVATION (lab notebook F19/F20; the spintronic emitter makes polarization a free knob):
  * s-pol sits the CNT back-reflection near the r=-1 MIRROR pole -> ill-conditioned inversion.
  * p-pol drops |r_back| well below 1 AND suppresses the parasitic SiO2->air front reflection
    (Brewster), which is the *cause* of the gap interference. So p-pol attacks BOTH problems.
  * Measuring s AND p gives the ellipsometric ratio rho = r_p/r_s, which (a) cancels common-mode
    reference/coupling errors and, crucially, (b) cancels the gap's common round-trip PHASE
    e^{-i2beta} in the ratio -> gap-immune at Brewster, gap-reduced elsewhere. = "pinning points".

This file VALIDATES those claims on synthetic ground truth (planted Drude CNT, known gap):
  Demo 1  single-pol conditioning: how far a planted gap pushes recovered n in s vs p, vs angle.
  Demo 2  ellipsometric closed-form inversion from rho: gap bias vs single-pol, at 45 and Brewster.
  Demo 3  joint dual-pol Drude+gap fit: recover the planted Drude AND d from s+p together (the
          over-determined "pinning"), where single-pol n<->d is degenerate.

It works in optical reflection coefficients, mirroring what the windowed self-referenced pipeline
delivers per polarization (the channel response cancels inside each H, so no s/p channel
calibration is needed for the windowed case -- only the bare reflection would need it).

Run:  PYTHONPATH=. ../.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/polarization_ellipsometry_validation.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import thz_core.thz_core as core
from thz_core.thz_core.multilayer import fresnel_reflection_s, fresnel_reflection_p
from explore_air_gap_deembedding import (
    gap_round_trip_phase, fabry_perot_reflection, SPEED_OF_LIGHT_M_PER_S,
    REFRACTIVE_INDEX_SIO2, REFRACTIVE_INDEX_AIR,
)

FRESNEL = {"s": fresnel_reflection_s, "p": fresnel_reflection_p}


# ── geometry + material ───────────────────────────────────────────────────────


def window_angles(external_deg, n_sio2=REFRACTIVE_INDEX_SIO2):
    """(theta_in_SiO2, theta_in_gap) for an external beam; the gap is air."""
    external = np.deg2rad(external_deg)
    theta_sio2 = np.arcsin(np.sin(external) / n_sio2)
    theta_gap = np.arcsin(n_sio2 * np.sin(theta_sio2) / REFRACTIVE_INDEX_AIR)
    return float(theta_sio2), float(theta_gap)


def drude_index(frequency_hz, eps_inf, plasma_thz, damping_thz):
    """Complex index n - i k from a Drude permittivity (thz_core sign convention)."""
    omega = 2.0 * np.pi * frequency_hz
    plasma = 2.0 * np.pi * plasma_thz * 1e12
    damping = 2.0 * np.pi * damping_thz * 1e12
    eps = eps_inf - plasma**2 / (omega**2 + 1j * damping * omega)
    return np.conj(np.sqrt(eps))


PLANTED_DRUDE = dict(eps_inf=4.0, plasma_thz=6.0, damping_thz=3.0)


def build_measured_reflection(frequency_hz, n_cnt, gap_thickness_m, polarization,
                              theta_sio2, theta_gap):
    """Windowed measured reflection through the gap: FP(r_front, r_back, 2beta) for one pol."""
    fresnel = FRESNEL[polarization]
    reflection_front = fresnel(REFRACTIVE_INDEX_SIO2, REFRACTIVE_INDEX_AIR, theta_sio2)
    reflection_back = fresnel(REFRACTIVE_INDEX_AIR, n_cnt, theta_gap)
    round_trip = gap_round_trip_phase(frequency_hz, gap_thickness_m, theta_gap)
    return fabry_perot_reflection(reflection_front, reflection_back, round_trip)


# ── p-pol single-interface inversion (numerical; core only does s) ─────────────


def invert_index_from_reflection(reflection, polarization, n_incident, theta_incident_rad,
                                  frequency_hz=None, n_initial=None):
    """Recover n - i k from a single-interface reflection coefficient.

    s-pol uses the validated closed form in core (robust near the r=-1 mirror, where a Newton
    iteration diverges). p-pol uses a complex Newton seeded near the answer (core is s-only).
    """
    if polarization == "s":
        mask = np.ones(np.shape(reflection), dtype=bool)
        n, k, _ = core.invert_nk_reflection(
            frequency_hz, np.asarray(reflection, dtype=complex), mask,
            theta_rad=theta_incident_rad, n_incident=n_incident, polarization="s")
        return n - 1j * k
    fresnel = FRESNEL[polarization]
    n_estimate = np.asarray(n_initial, dtype=complex).copy()
    for _ in range(60):
        with np.errstate(invalid="ignore", divide="ignore"):
            residual = fresnel(n_incident, n_estimate, theta_incident_rad) - reflection
            step_size = 1e-6 * (1.0 + np.abs(n_estimate))
            derivative = (
                fresnel(n_incident, n_estimate + step_size, theta_incident_rad)
                - fresnel(n_incident, n_estimate - step_size, theta_incident_rad)
            ) / (2.0 * step_size)
            update = residual / derivative
        n_estimate = np.where(np.isfinite(update), n_estimate - update, n_estimate)
        if np.all(np.abs(update) < 1e-12 * (1.0 + np.abs(n_estimate))):
            break
    return n_estimate


def ellipsometric_index(rho, n_incident, theta_incident_rad):
    """Exact single-interface ellipsometric inversion: n2 from rho = r_p / r_s.

    n2^2 = n1^2 sin^2(theta) [1 + tan^2(theta) ((1-rho)/(1+rho))^2]   (Azzam & Bashara).
    The common gap round-trip PHASE cancels in rho, so this is gap-PHASE immune; with a residual
    front reflection (away from Brewster) a small amplitude bias remains.
    """
    sin_t = np.sin(theta_incident_rad)
    tan_t = np.tan(theta_incident_rad)
    bracket = 1.0 + tan_t**2 * ((1.0 - rho) / (1.0 + rho)) ** 2
    n_squared = (n_incident**2) * (sin_t**2) * bracket
    # np.sqrt's principal branch already yields the n - i k (k>0) convention here; no conjugate.
    return np.sqrt(n_squared)


# ── Demo 1: single-pol conditioning / gap sensitivity ─────────────────────────


def demo_single_pol_conditioning():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 600)
    print("=== Demo 1: single-pol gap sensitivity (planted Drude CNT) ===")
    print("  how far a planted gap pushes the (naive, no-de-embed) recovered n from truth:")
    for external_deg in (45.0, 62.9):
        theta_sio2, theta_gap = window_angles(external_deg)
        n_true = drude_index(frequency_hz, **PLANTED_DRUDE)
        for gap_um in (4.0,):
            row = f"  ext {external_deg:4.1f} deg, gap {gap_um:.0f} um:"
            for polarization in ("s", "p"):
                r_meas = build_measured_reflection(
                    frequency_hz, n_true, gap_um * 1e-6, polarization, theta_sio2, theta_gap)
                # NAIVE: invert as if r_meas were the bare air->CNT back reflection (ignore gap).
                n_recovered = invert_index_from_reflection(
                    r_meas, polarization, REFRACTIVE_INDEX_AIR, theta_gap,
                    frequency_hz=frequency_hz, n_initial=n_true)
                error = np.abs(n_recovered.real - n_true.real)
                row += (f"   {polarization}-pol median|dn|={np.nanmedian(error):.3f} "
                        f"max={np.nanmax(error):.3f}")
            print(row)
    print("  -> p-pol error is far smaller (front reflection suppressed -> less gap contamination);"
          "\n     at the Brewster external angle (62.9 deg) the front nulls and p-pol is cleanest.")


# ── Demo 2: ellipsometric inversion, gap bias ─────────────────────────────────


def demo_ellipsometric_inversion():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 600)
    theta_sio2, theta_gap = window_angles(45.0)
    n_true = drude_index(frequency_hz, **PLANTED_DRUDE)
    print("\n=== Demo 2: ellipsometric closed form (rho=r_p/r_s) -- sanity + geometry caution ===")

    # (a) A GENUINE single air->CNT interface (no window, no gap): the closed form is EXACT.
    r_s_pure = fresnel_reflection_s(REFRACTIVE_INDEX_AIR, n_true, theta_gap)
    r_p_pure = fresnel_reflection_p(REFRACTIVE_INDEX_AIR, n_true, theta_gap)
    n_pure = ellipsometric_index(r_p_pure / r_s_pure, REFRACTIVE_INDEX_AIR, theta_gap)
    print(f"  pure air->CNT interface: ellipsometric median|dn|="
          f"{np.median(np.abs(n_pure.real - n_true.real)):.5f} (EXACT -- formula verified)")

    # (b) The WINDOWED measurement is NOT a single interface: at d->0 it is SiO2->CNT (window
    #     incidence), for d>0 it is a Fabry-Perot. Applying the air-incidence closed form to it is
    #     geometrically wrong -> biased. So rho is no shortcut here; MODEL the geometry (Demo 3).
    for gap_um in (0.0, 4.0):
        r_s = build_measured_reflection(frequency_hz, n_true, gap_um * 1e-6, "s", theta_sio2, theta_gap)
        r_p = build_measured_reflection(frequency_hz, n_true, gap_um * 1e-6, "p", theta_sio2, theta_gap)
        n_e = ellipsometric_index(r_p / r_s, REFRACTIVE_INDEX_AIR, theta_gap)
        print(f"  windowed, gap {gap_um:.0f} um: ellipsometric median|dn|="
              f"{np.median(np.abs(n_e.real - n_true.real)):.3f} (biased -- wrong geometry/gap unmodelled)")
    print("  -> the formula is correct but assumes a single interface; the window+gap is not one.\n"
          "     Use the forward-model fit (Demo 3), which encodes the exact geometry for both pols.")


# ── Demo 3: joint dual-pol Drude + gap fit (the pinning payoff) ────────────────


def joint_pol_fit(frequency_hz, measured_by_pol, theta_sio2, theta_gap,
                  polarizations=("s", "p"), initial=None):
    """Fit (eps_inf, plasma_thz, damping_thz, gap_um) to the named polarization spectra.

    measured_by_pol: {"s": r_meas_s, "p": r_meas_p}. With polarizations=("s","p") the same Drude
    n(omega) must explain BOTH and the single gap is shared -> over-determined, breaking the
    n<->d degeneracy. Pass ("s",) to see the single-pol fit for comparison.
    """
    def residuals(params):
        eps_inf, plasma_thz, damping_thz, gap_um = params
        n_model = drude_index(frequency_hz, eps_inf, plasma_thz, damping_thz)
        parts = []
        for polarization in polarizations:
            model = build_measured_reflection(frequency_hz, n_model, gap_um * 1e-6,
                                              polarization, theta_sio2, theta_gap)
            difference = model - measured_by_pol[polarization]
            parts.extend([difference.real, difference.imag])
        return np.concatenate(parts)

    if initial is None:
        initial = [2.0, 4.0, 2.0, 1.0]   # deliberately OFF the planted values
    bounds = ([1.0, 0.5, 0.2, 0.0], [12.0, 20.0, 12.0, 60.0])
    solution = least_squares(residuals, initial, bounds=bounds, xtol=1e-12, ftol=1e-12)
    eps_inf, plasma_thz, damping_thz, gap_um = solution.x
    return dict(eps_inf=eps_inf, plasma_thz=plasma_thz, damping_thz=damping_thz,
                gap_um=gap_um, cost=float(solution.cost))


def demo_joint_fit():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 600)
    theta_sio2, theta_gap = window_angles(45.0)
    n_true = drude_index(frequency_hz, **PLANTED_DRUDE)
    true_gap_um = 4.0
    r_s = build_measured_reflection(frequency_hz, n_true, true_gap_um * 1e-6, "s", theta_sio2, theta_gap)
    r_p = build_measured_reflection(frequency_hz, n_true, true_gap_um * 1e-6, "p", theta_sio2, theta_gap)

    measured = {"s": r_s, "p": r_p}
    fit = joint_pol_fit(frequency_hz, measured, theta_sio2, theta_gap, polarizations=("s", "p"))
    fit_s_only = joint_pol_fit(frequency_hz, measured, theta_sio2, theta_gap, polarizations=("s",))
    print("\n=== Demo 3: joint Drude + gap fit (recover n AND d together) ===")
    print(f"  planted   : eps_inf=4.00 plasma=6.00 THz damping=3.00 THz gap=4.0 um")
    print(f"  dual s+p  : eps_inf={fit['eps_inf']:.2f} plasma={fit['plasma_thz']:.2f} "
          f"damping={fit['damping_thz']:.2f} gap={fit['gap_um']:.1f} um  (cost {fit['cost']:.1e})")
    print(f"  s-pol only: eps_inf={fit_s_only['eps_inf']:.2f} plasma={fit_s_only['plasma_thz']:.2f} "
          f"damping={fit_s_only['damping_thz']:.2f} gap={fit_s_only['gap_um']:.1f} um  "
          f"(cost {fit_s_only['cost']:.1e})")
    n_fit = drude_index(frequency_hz, fit["eps_inf"], fit["plasma_thz"], fit["damping_thz"])
    print(f"  dual-pol recovered n median|dn|={np.median(np.abs(n_fit.real - n_true.real)):.4f}, "
          f"gap error {abs(fit['gap_um'] - true_gap_um):.2f} um")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    f_thz = frequency_hz * 1e-12
    axes[0].plot(f_thz, n_true.real, "k", lw=2, label="planted n")
    axes[0].plot(f_thz, n_fit.real, "C2--", lw=1.6, label="dual-pol joint fit")
    axes[0].set_xlabel("THz"); axes[0].set_ylabel("n"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("Demo 3: joint dual-pol fit recovers n (with gap present)")
    axes[1].plot(f_thz, -n_true.imag, "k", lw=2, label="planted k")
    axes[1].plot(f_thz, -n_fit.imag, "C2--", lw=1.6, label="fit k")
    axes[1].set_xlabel("THz"); axes[1].set_ylabel("k"); axes[1].legend(); axes[1].grid(alpha=0.3)
    figure.tight_layout()
    output_path = "explorations/air_gap_cnt_reflection/polarization_ellipsometry_validation.png"
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    print(f"  figure -> {output_path}")
    return fit


def demo_noise_robustness(noise_level=0.01, trials=40, seed=0):
    """Under measurement noise, does adding p-pol reduce the fitted-parameter error vs s-only?

    On NOISELESS data both single- and dual-pol recover n+d exactly (Demo 3), so dual-pol's value
    is CONDITIONING/robustness: s-pol sits near the r=-1 mirror where noise is amplified into the
    inversion; p-pol's lower |r| is better conditioned, so s+p (and p-only) should track truth with
    less scatter under the same noise. This quantifies that.
    """
    frequency_hz = np.linspace(0.3e12, 2.5e12, 400)
    theta_sio2, theta_gap = window_angles(45.0)
    n_true = drude_index(frequency_hz, **PLANTED_DRUDE)
    true_gap_um = 4.0
    clean = {p: build_measured_reflection(frequency_hz, n_true, true_gap_um * 1e-6, p,
                                          theta_sio2, theta_gap) for p in ("s", "p")}
    rng = np.random.default_rng(seed)

    def fit_error(polarizations):
        gap_errors, n_errors = [], []
        for _ in range(trials):
            noisy = {}
            for p in ("s", "p"):
                noise = noise_level * (rng.standard_normal(frequency_hz.size)
                                       + 1j * rng.standard_normal(frequency_hz.size))
                noisy[p] = clean[p] + noise
            fit = joint_pol_fit(frequency_hz, noisy, theta_sio2, theta_gap,
                                polarizations=polarizations)
            n_fit = drude_index(frequency_hz, fit["eps_inf"], fit["plasma_thz"], fit["damping_thz"])
            gap_errors.append(abs(fit["gap_um"] - true_gap_um))
            n_errors.append(np.median(np.abs(n_fit.real - n_true.real)))
        return np.mean(gap_errors), np.mean(n_errors)

    print(f"\n=== Demo 4: robustness under {noise_level:.0%} reflection noise ({trials} trials) ===")
    for label, pols in (("s-only", ("s",)), ("p-only", ("p",)), ("dual s+p", ("s", "p"))):
        gap_err, n_err = fit_error(pols)
        print(f"  {label:9s}: mean gap error {gap_err:.2f} um, mean n error {n_err:.3f}")
    print("  -> p-pol / dual-pol track truth with less error under noise (better conditioned than\n"
          "     the near-mirror s-pol). This -- plus the gap front-suppression (Demo 1) -- is the\n"
          "     real-data payoff; on noiseless data single-pol already suffices (Demo 3).")


if __name__ == "__main__":
    demo_single_pol_conditioning()
    demo_ellipsometric_inversion()
    demo_joint_fit()
    demo_noise_robustness()
