"""Unit tests for thz_core.self_referenced_transfer — the self-reference MATH.

These justify the math in isolation (pure spectra in, H out), independent of the
dataset wrapper:
  * H equals the two nested divisions (Y2_s/Y1_s)/(Y2_r/Y1_r),
  * the within-trace ratio cancels DRIFT (any factor multiplying a whole trace),
  * H equals the algebraically-equivalent composed form (Y2_s/Y2_r)*(Y1_r/Y1_s),
  * the reported front-pulse correction is C = Y1_r/Y1_s.

Run with:
    .venv/Scripts/python.exe tests/test_self_referenced_transfer.py
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import thz_core.thz_core as core

_CONFIG = {"transfer": {"unwrap_phase": False, "min_ref_amp_rel": 1e-4}}


def _spectra(seed=0):
    """Four well-conditioned complex spectra (no near-zero bins -> nothing masked)."""
    rng = np.random.default_rng(seed)
    freq = np.linspace(0.3e12, 2.5e12, 64)
    def smooth_complex(scale):
        mag = scale * (1.0 + 0.3 * np.sin(freq / 4e11))
        phase = 0.5 * np.cos(freq / 6e11)
        return mag * np.exp(1j * phase) + 0.05 * (rng.standard_normal(freq.size) + 1j)
    first_ref = smooth_complex(1.0)
    first_samp = smooth_complex(1.0) * 0.9   # small front-pulse drift between shots
    second_ref = smooth_complex(0.5)
    second_samp = smooth_complex(0.7)
    return freq, second_samp, first_samp, second_ref, first_ref


def _band(freq):
    return (freq >= 0.4e12) & (freq <= 2.2e12)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_H_equals_two_nested_divisions():
    freq, second_samp, first_samp, second_ref, first_ref = _spectra()
    H, _, _ = core.self_referenced_transfer(freq, second_samp, first_samp, second_ref, first_ref, _CONFIG)
    expected = (second_samp / first_samp) / (second_ref / first_ref)
    band = _band(freq)
    assert np.allclose(H[band], expected[band], rtol=1e-10, atol=1e-12)


def test_H_equals_composed_form():
    """H == (Y2_s/Y2_r) * (Y1_r/Y1_s) — the form the old adapter used."""
    freq, second_samp, first_samp, second_ref, first_ref = _spectra(seed=3)
    H, _, _ = core.self_referenced_transfer(freq, second_samp, first_samp, second_ref, first_ref, _CONFIG)
    composed = (second_samp / second_ref) * (first_ref / first_samp)
    band = _band(freq)
    assert np.allclose(H[band], composed[band], rtol=1e-10, atol=1e-12)


def test_within_trace_ratio_cancels_drift():
    """Multiplying the WHOLE sample trace (both pulses) by any drift G(f) leaves H unchanged."""
    freq, second_samp, first_samp, second_ref, first_ref = _spectra(seed=7)
    H0, _, _ = core.self_referenced_transfer(freq, second_samp, first_samp, second_ref, first_ref, _CONFIG)

    rng = np.random.default_rng(11)
    drift = (1.0 + 0.4 * np.sin(freq / 3e11)) * np.exp(1j * 0.7 * np.cos(freq / 5e11))
    # drift multiplies BOTH the front and back pulse of the sample acquisition
    H_drift, _, _ = core.self_referenced_transfer(
        freq, second_samp * drift, first_samp * drift, second_ref, first_ref, _CONFIG,
    )
    band = _band(freq)
    assert np.allclose(H0[band], H_drift[band], rtol=1e-10, atol=1e-12)


def test_front_correction_diagnostic():
    freq, second_samp, first_samp, second_ref, first_ref = _spectra(seed=5)
    _, _, metrics = core.self_referenced_transfer(freq, second_samp, first_samp, second_ref, first_ref, _CONFIG)
    C = metrics["diagnostics"]["front_correction"]
    band = _band(freq)
    assert np.allclose(C[band], (first_ref / first_samp)[band], rtol=1e-10, atol=1e-12)


def test_noise_floor_masks_low_amplitude_bins():
    """Bins where a denominator is buried below the noise floor come back NaN, not huge."""
    freq, second_samp, first_samp, second_ref, first_ref = _spectra(seed=1)
    first_samp = first_samp.copy()
    first_samp[10] = 1e-20   # front sample pulse vanishes at one bin -> that bin invalid
    H, mask, _ = core.self_referenced_transfer(freq, second_samp, first_samp, second_ref, first_ref, _CONFIG)
    assert not np.isfinite(H[10])
    assert not mask[10]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

_TESTS = [
    test_H_equals_two_nested_divisions,
    test_H_equals_composed_form,
    test_within_trace_ratio_cancels_drift,
    test_front_correction_diagnostic,
    test_noise_floor_masks_low_amplitude_bins,
]


def main() -> int:
    passed = failed = 0
    for test in _TESTS:
        try:
            test()
            passed += 1
            print(f"PASS  {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed ({len(_TESTS)} total)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
