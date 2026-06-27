"""Unit tests for MEM (maximum-entropy) phase retrieval — explorations/air_gap_cnt_reflection.

These are deterministic known-answer tests:
  * for a system handed its FULL period, MEM must reproduce the exact analytic min phase;
  * the anchor helper must pin known phase values;
  * on a finite band with a near-edge pole, MEM must beat Hilbert at the low edge.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "explorations", "air_gap_cnt_reflection")
)

from mem_phase_retrieval import (  # noqa: E402
    autoregressive_coefficients_from_reflectivity,
    maximum_entropy_phase,
    hilbert_minimum_phase,
    apply_phase_anchor,
    known_minimum_phase_reflection,
    synthetic_validation,
)


def _phase_shape_residual(recovered, truth, frequency):
    """Residual after removing an affine (offset+slope) term truth-vs-recovered allows."""
    residual = recovered - truth
    slope, intercept = np.polyfit(frequency, residual, 1)
    return recovered - (slope * frequency + intercept) - truth


def test_ar_coefficient_count_and_finiteness():
    normalized_frequency = np.arange(512) / 512
    reflection = known_minimum_phase_reflection(
        normalized_frequency, zeros=[0.5], poles=[0.8, 0.3])
    coefficients, error_power = autoregressive_coefficients_from_reflectivity(
        np.abs(reflection) ** 2, num_coefficients=20)
    assert coefficients.shape == (20,)
    assert np.all(np.isfinite(coefficients))
    assert error_power > 0.0


def test_order_must_be_smaller_than_samples():
    with pytest.raises(ValueError):
        autoregressive_coefficients_from_reflectivity(np.ones(10), num_coefficients=10)


def test_full_period_recovers_known_minimum_phase():
    # Given the WHOLE period, MEM must reproduce the exact analytic minimum phase (shape).
    num_samples = 2048
    normalized_frequency = np.arange(num_samples) / num_samples
    reflection = known_minimum_phase_reflection(
        normalized_frequency,
        zeros=[0.6 * np.exp(1j * 0.7), 0.6 * np.exp(-1j * 0.7)],
        poles=[0.85 * np.exp(1j * 0.4), 0.85 * np.exp(-1j * 0.4)],
        gain=0.5,
    )
    phase_true = np.unwrap(np.angle(reflection))
    phase_mem = np.unwrap(maximum_entropy_phase(np.abs(reflection), num_coefficients=40))
    residual = _phase_shape_residual(phase_mem, phase_true, normalized_frequency)
    assert np.sqrt(np.mean(residual ** 2)) < 0.05


def test_anchor_single_point_pins_value():
    frequency_hz = np.linspace(0.2e12, 3.0e12, 200)
    phase = np.linspace(-1.0, 1.0, 200)
    anchored = apply_phase_anchor(frequency_hz, phase, 1.0e12, 0.5)
    index = int(np.argmin(np.abs(frequency_hz - 1.0e12)))
    assert anchored[index] == pytest.approx(0.5, abs=1e-9)


def test_anchor_two_points_pins_both():
    frequency_hz = np.linspace(0.2e12, 3.0e12, 200)
    phase = np.sin(frequency_hz / 1e12)  # arbitrary shape
    anchored = apply_phase_anchor(
        frequency_hz, phase, [0.5e12, 2.5e12], [0.1, -0.3])
    index_low = int(np.argmin(np.abs(frequency_hz - 0.5e12)))
    index_high = int(np.argmin(np.abs(frequency_hz - 2.5e12)))
    assert anchored[index_low] == pytest.approx(0.1, abs=1e-9)
    assert anchored[index_high] == pytest.approx(-0.3, abs=1e-9)


def test_mem_beats_hilbert_at_low_edge():
    scores = synthetic_validation(num_coefficients=30)
    assert scores["mem_edge"] < scores["hilbert_edge"]
    # And by a clear margin in our region of interest.
    assert scores["mem_edge"] < 0.5 * scores["hilbert_edge"]


def test_hilbert_helper_matches_manual():
    magnitude = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8])
    from scipy.signal import hilbert as _hilbert
    expected = -np.imag(_hilbert(np.log(magnitude)))
    np.testing.assert_allclose(hilbert_minimum_phase(magnitude), expected)
