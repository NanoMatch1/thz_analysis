"""Tests for the dual-polarization / ellipsometry validation (explorations/air_gap_cnt_reflection).

Encodes the validated claims:
  * the ellipsometric closed form is EXACT on a genuine single interface;
  * the p-pol numerical inversion round-trips;
  * the joint Drude+gap forward-model fit recovers planted n AND d (clean data);
  * under noise, adding p-pol reduces the fitted-n error vs s-only (the conditioning payoff).
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "explorations", "air_gap_cnt_reflection")
)

from thz_core.thz_core.multilayer import fresnel_reflection_s, fresnel_reflection_p  # noqa: E402
from polarization_ellipsometry_validation import (  # noqa: E402
    window_angles, drude_index, build_measured_reflection, invert_index_from_reflection,
    ellipsometric_index, joint_pol_fit, PLANTED_DRUDE, REFRACTIVE_INDEX_AIR,
)


def test_ellipsometric_exact_on_single_interface():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 200)
    _, theta_gap = window_angles(45.0)
    n_true = drude_index(frequency_hz, **PLANTED_DRUDE)
    r_s = fresnel_reflection_s(REFRACTIVE_INDEX_AIR, n_true, theta_gap)
    r_p = fresnel_reflection_p(REFRACTIVE_INDEX_AIR, n_true, theta_gap)
    n_recovered = ellipsometric_index(r_p / r_s, REFRACTIVE_INDEX_AIR, theta_gap)
    assert np.max(np.abs(n_recovered.real - n_true.real)) < 1e-6
    assert np.max(np.abs(n_recovered.imag - n_true.imag)) < 1e-6


def test_p_pol_inversion_roundtrips():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 200)
    _, theta_gap = window_angles(45.0)
    n_true = drude_index(frequency_hz, **PLANTED_DRUDE)
    r_p = fresnel_reflection_p(REFRACTIVE_INDEX_AIR, n_true, theta_gap)
    n_recovered = invert_index_from_reflection(
        r_p, "p", REFRACTIVE_INDEX_AIR, theta_gap, n_initial=2.0 - 2.0j * np.ones_like(n_true))
    assert np.max(np.abs(n_recovered.real - n_true.real)) < 1e-4


def test_s_pol_inversion_roundtrips():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 200)
    _, theta_gap = window_angles(45.0)
    n_true = drude_index(frequency_hz, **PLANTED_DRUDE)
    r_s = fresnel_reflection_s(REFRACTIVE_INDEX_AIR, n_true, theta_gap)
    n_recovered = invert_index_from_reflection(
        r_s, "s", REFRACTIVE_INDEX_AIR, theta_gap, frequency_hz=frequency_hz)
    assert np.max(np.abs(n_recovered.real - n_true.real)) < 1e-4


def test_joint_dual_pol_fit_recovers_planted():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 400)
    theta_sio2, theta_gap = window_angles(45.0)
    n_true = drude_index(frequency_hz, **PLANTED_DRUDE)
    true_gap_um = 4.0
    measured = {p: build_measured_reflection(frequency_hz, n_true, true_gap_um * 1e-6, p,
                                             theta_sio2, theta_gap) for p in ("s", "p")}
    fit = joint_pol_fit(frequency_hz, measured, theta_sio2, theta_gap, polarizations=("s", "p"))
    assert abs(fit["eps_inf"] - 4.0) < 1e-3
    assert abs(fit["plasma_thz"] - 6.0) < 1e-3
    assert abs(fit["damping_thz"] - 3.0) < 1e-3
    assert abs(fit["gap_um"] - true_gap_um) < 1e-2


def test_dual_pol_beats_s_only_under_noise():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 300)
    theta_sio2, theta_gap = window_angles(45.0)
    n_true = drude_index(frequency_hz, **PLANTED_DRUDE)
    true_gap_um = 4.0
    clean = {p: build_measured_reflection(frequency_hz, n_true, true_gap_um * 1e-6, p,
                                          theta_sio2, theta_gap) for p in ("s", "p")}
    rng = np.random.default_rng(1)
    noise_level = 0.01

    def mean_n_error(polarizations, trials=12):
        errors = []
        for _ in range(trials):
            noisy = {}
            for p in ("s", "p"):
                noise = noise_level * (rng.standard_normal(frequency_hz.size)
                                       + 1j * rng.standard_normal(frequency_hz.size))
                noisy[p] = clean[p] + noise
            fit = joint_pol_fit(frequency_hz, noisy, theta_sio2, theta_gap,
                                polarizations=polarizations)
            n_fit = drude_index(frequency_hz, fit["eps_inf"], fit["plasma_thz"], fit["damping_thz"])
            errors.append(np.median(np.abs(n_fit.real - n_true.real)))
        return float(np.mean(errors))

    assert mean_n_error(("s", "p")) < mean_n_error(("s",))
