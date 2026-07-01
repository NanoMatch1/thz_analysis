"""Tests for the contact-gap forward model + dual-pol joint fit (thz_core.reflection_gap)
and the adapter's sample-pairing helper.
"""

import warnings

import numpy as np
import pytest

import thz_core.thz_core as core
from thz_core.thz_core.multilayer import fresnel_reflection_s, fresnel_reflection_p
from dataset_core.adapters.thz_adapter import _dual_pol_sample_key


def _geometry(n_sio2=1.95, external_deg=45.0):
    external = np.deg2rad(external_deg)
    theta_sio2 = np.arcsin(np.sin(external) / n_sio2)
    theta_gap = np.arcsin(n_sio2 * np.sin(theta_sio2) / 1.0)
    return theta_sio2, theta_gap


def _planted_measurement(frequency_hz, gap_um):
    n_sio2 = 1.95
    theta_sio2, theta_gap = _geometry(n_sio2)
    n_true = core.drude_index(frequency_hz, 4.0, 6.0, 3.0)
    front = {
        "s": fresnel_reflection_s(n_sio2, 1.0, theta_sio2),
        "p": fresnel_reflection_p(n_sio2, 1.0, theta_sio2),
    }
    measured = {
        p: core.modelled_windowed_reflection(
            frequency_hz, n_true, front[p], gap_um * 1e-6, theta_gap, p)
        for p in ("s", "p")
    }
    front_arrays = {p: front[p] * np.ones_like(frequency_hz) for p in ("s", "p")}
    return measured, front_arrays, theta_gap, n_true


def test_drude_index_is_passive_conductor():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 100)
    n_hat = core.drude_index(frequency_hz, 4.0, 6.0, 3.0)
    assert np.all(n_hat.real > 0)
    assert np.all(-n_hat.imag > 0)  # k > 0 (absorbing)


def test_gap_round_trip_phase_linear():
    frequency_hz = np.array([1e12, 2e12])
    phase = core.gap_round_trip_phase(frequency_hz, 10e-6, np.deg2rad(45.0))
    assert phase[1] == pytest.approx(2 * phase[0])  # linear in frequency
    assert core.gap_round_trip_phase(frequency_hz, 20e-6, np.deg2rad(45.0))[0] == pytest.approx(
        2 * phase[0])  # linear in d


def test_single_gap_reflection_zero_gap_is_composite():
    # d=0: FP reduces to the two-interface composite (r_front+r_back)/(1+r_front r_back).
    r_front, r_back = 0.3 + 0.0j, -0.8 + 0.1j
    r = core.single_gap_reflection(r_front, r_back, 0.0)
    assert r == pytest.approx((r_front + r_back) / (1 + r_front * r_back))


def test_joint_fit_dual_pol_recovers_planted():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 500)
    mask = np.ones(frequency_hz.size, dtype=bool)
    measured, front, theta_gap, n_true = _planted_measurement(frequency_hz, 4.0)
    fit = core.fit_reflection_gap_dual_pol(frequency_hz, mask, measured, front, theta_gap)
    assert fit["eps_inf"] == pytest.approx(4.0, abs=1e-3)
    assert fit["plasma_thz"] == pytest.approx(6.0, abs=1e-3)
    assert fit["damping_thz"] == pytest.approx(3.0, abs=1e-3)
    assert fit["gap_um"] == pytest.approx(4.0, abs=1e-2)
    assert np.max(np.abs(fit["n"] - n_true.real)) < 1e-3


def test_joint_fit_fixed_gap_holds_gap():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 400)
    mask = np.ones(frequency_hz.size, dtype=bool)
    measured, front, theta_gap, _ = _planted_measurement(frequency_hz, 4.0)
    fit = core.fit_reflection_gap_dual_pol(
        frequency_hz, mask, measured, front, theta_gap, fixed_gap_um=4.0)
    assert fit["gap_fitted"] is False
    assert fit["gap_um"] == 4.0
    assert fit["eps_inf"] == pytest.approx(4.0, abs=1e-2)


def test_joint_fit_single_polarization_runs():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 400)
    mask = np.ones(frequency_hz.size, dtype=bool)
    measured, front, theta_gap, _ = _planted_measurement(frequency_hz, 4.0)
    fit = core.fit_reflection_gap_dual_pol(
        frequency_hz, mask, {"s": measured["s"]}, {"s": front["s"]}, theta_gap)
    assert fit["polarizations"] == ["s"]
    assert fit["gap_um"] == pytest.approx(4.0, abs=1e-1)


def test_dual_pol_under_noise_dual_beats_s_only():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 300)
    mask = np.ones(frequency_hz.size, dtype=bool)
    measured, front, theta_gap, n_true = _planted_measurement(frequency_hz, 4.0)
    rng = np.random.default_rng(3)

    def mean_error(polarizations, trials=8):
        errors = []
        for _ in range(trials):
            noisy = {}
            for p in ("s", "p"):
                noise = 0.01 * (rng.standard_normal(frequency_hz.size)
                                + 1j * rng.standard_normal(frequency_hz.size))
                noisy[p] = measured[p] + noise
            sub = {p: noisy[p] for p in polarizations}
            fit = core.fit_reflection_gap_dual_pol(
                frequency_hz, mask, sub, front, theta_gap)
            errors.append(np.median(np.abs(fit["n"] - n_true.real)))
        return float(np.mean(errors))

    assert mean_error(("s", "p")) < mean_error(("s",))


def test_dual_pol_sample_key_pairs_s_and_p():
    assert (_dual_pol_sample_key("sample_CNT-0-deg_colinear_A-75_s.acc")
            == _dual_pol_sample_key("sample_CNT-0-deg_colinear_A-75_p.acc"))
    assert (_dual_pol_sample_key("sample_CNT-90-deg_s-pol.acc")
            != _dual_pol_sample_key("sample_CNT-0-deg_s-pol.acc"))
