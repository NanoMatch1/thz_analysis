"""Tests for the amplitude-KK gap-shift estimator (deembed_air_gap_iterative).

These encode the VALIDATED regime of the method (found by synthetic ground truth):
  * super-fringe gap, clean phase  -> recovers the gap;
  * super-fringe gap, alignment-corrupted phase -> still recovers it (the arXiv robustness);
  * small (sub-fringe) gap -> flagged sub_fringe, defer to the phase-slope estimator;
  * the conductor sign (~pi) is found by the automatic sign search.
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "explorations", "air_gap_cnt_reflection")
)

import thz_core.thz_core as core  # noqa: E402
from explore_air_gap_deembedding import (  # noqa: E402
    internal_angles, gap_round_trip_phase, fabry_perot_reflection, deembed_gap_layer,
    planted_cnt_index, REFRACTIVE_INDEX_SIO2, REFRACTIVE_INDEX_AIR, SPEED_OF_LIGHT_M_PER_S,
)
from deembed_air_gap_iterative import (  # noqa: E402
    estimate_gap_amplitude_kk, estimate_gap_minimum_phase,
)


def _build_measurement(gap_thickness_m):
    frequency_hz = np.linspace(0.1e12, 4.0e12, 1000)
    theta_sio2, theta_gap = internal_angles()
    n_cnt = planted_cnt_index(frequency_hz)
    reflection_front = core.fresnel_reflection_s(
        REFRACTIVE_INDEX_SIO2, REFRACTIVE_INDEX_AIR, theta_sio2)
    reflection_back = core.fresnel_reflection_s(REFRACTIVE_INDEX_AIR, n_cnt, theta_gap)
    round_trip = gap_round_trip_phase(frequency_hz, gap_thickness_m, theta_gap)
    reflection_measured = fabry_perot_reflection(reflection_front, reflection_back, round_trip)
    return frequency_hz, reflection_measured, reflection_front, theta_gap


def test_super_fringe_clean_recovers_gap():
    true_gap_m = 120e-6
    frequency_hz, reflection_measured, reflection_front, theta_gap = _build_measurement(true_gap_m)
    mask = np.ones(frequency_hz.size, dtype=bool)
    gap, diagnostics = estimate_gap_amplitude_kk(
        frequency_hz, reflection_measured, reflection_front, mask, theta_gap,
        (0.4e12, 3.0e12), phase_engine="hilbert")
    assert not diagnostics["sub_fringe"]
    assert abs(gap - true_gap_m) < 10e-6


def test_super_fringe_robust_to_alignment_corruption():
    # A spurious sample-reference misplacement = pure linear phase; |r| unchanged, so the
    # amplitude method should be (nearly) immune while the phase-slope method is biased.
    true_gap_m = 120e-6
    frequency_hz, reflection_measured, reflection_front, theta_gap = _build_measurement(true_gap_m)
    mask = np.ones(frequency_hz.size, dtype=bool)
    misplacement_m = 8e-6
    spurious_delay_s = 2 * misplacement_m * np.cos(theta_gap) / SPEED_OF_LIGHT_M_PER_S
    corrupted = reflection_measured * np.exp(-1j * 2 * np.pi * frequency_hz * spurious_delay_s)

    gap_amp, _ = estimate_gap_amplitude_kk(
        frequency_hz, corrupted, reflection_front, mask, theta_gap, (0.4e12, 3.0e12),
        phase_engine="hilbert")
    x_corrupted = deembed_gap_layer(corrupted, reflection_front)
    gap_phase_slope, _ = estimate_gap_minimum_phase(
        frequency_hz, x_corrupted, mask, theta_gap, (0.4e12, 3.0e12), phase_engine="hilbert")

    # amplitude-KK stays close to truth; phase-slope is pulled away by the misplacement.
    assert abs(gap_amp - true_gap_m) < 15e-6
    assert abs(gap_phase_slope - true_gap_m) > abs(gap_amp - true_gap_m)


def test_small_gap_flagged_sub_fringe():
    true_gap_m = 20e-6
    frequency_hz, reflection_measured, reflection_front, theta_gap = _build_measurement(true_gap_m)
    mask = np.ones(frequency_hz.size, dtype=bool)
    _, diagnostics = estimate_gap_amplitude_kk(
        frequency_hz, reflection_measured, reflection_front, mask, theta_gap,
        (0.4e12, 3.0e12), phase_engine="hilbert")
    assert diagnostics["sub_fringe"]
    assert diagnostics["fringes_across_band"] < 0.5


def test_sign_search_picks_conductor_sign():
    # The planted CNT is metal-like (r_back ~ -1), so the automatic search should prefer the
    # negative material sign over +1 for the forward model.
    true_gap_m = 120e-6
    frequency_hz, reflection_measured, reflection_front, theta_gap = _build_measurement(true_gap_m)
    mask = np.ones(frequency_hz.size, dtype=bool)
    _, diagnostics = estimate_gap_amplitude_kk(
        frequency_hz, reflection_measured, reflection_front, mask, theta_gap,
        (0.4e12, 3.0e12), phase_engine="hilbert")
    assert diagnostics["material_sign"] == -1.0
