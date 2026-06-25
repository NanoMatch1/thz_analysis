"""Regression tests for Audit 1: the shared-axis reflection W = Y2/Y1 must be
INVARIANT to a rigid shift of the trace's time axis.

A rigid axis shift (what align_to_reference does: data[:,0] += integer_shift, values
unchanged) is physically meaningless — it only relabels the time origin. Before the fix,
fft_spectrum applied the absolute-time phase factor exp(-2πi f t0) to the FIRST reflection
ONLY, so W picked up an uncancelled exp(2πi f t0) and a shift leaked a spurious linear
phase into the self-referenced H. The fix applies the factor SYMMETRICALLY to both
reflection segments, so the shift cancels in W.

These tests build THzDataReflection objects directly and drive the real fft_spectrum.

Run with:
    .venv/Scripts/python.exe tests/test_align_shift_invariance.py
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.data_structures.thz import BaseTHzData, THzData, THzDataReflection
from dataset_core.adapters import thz_adapter as thz

_DT_PS = 0.05
_N_FFT = 2048


def _gaussian(time_ps, centre_ps, amp, width_ps):
    return amp * np.exp(-0.5 * ((time_ps - centre_ps) / width_ps) ** 2)


def _make_shared_reflection(filename, front_ps=155.0, back_ps=180.0, shift_ps=0.0):
    """Two-pulse reflection on ONE shared axis; `shift_ps` rigidly relabels the time axis.

    The pulse VALUES stay at the same array indices (rigid relabel), exactly like
    align_to_reference's `data[:,0] += shift`.
    """
    base_ps = np.arange(150.0, 190.0, _DT_PS)
    front_only = _gaussian(base_ps, front_ps, 1.0, 0.30)
    back_only = _gaussian(base_ps, back_ps, 0.5, 0.40)
    time_ps = base_ps + shift_ps  # rigid axis relabel (values unchanged)

    first = THzData(
        data=[BaseTHzData(data=np.column_stack([time_ps, front_only]), headers=[])],
        header=None, filename=filename,
    )
    second = THzData(
        data=[BaseTHzData(data=np.column_stack([time_ps, back_only]), headers=[])],
        header=None, filename=filename,
    )
    return THzDataReflection.from_thzdata(second, first)


def _fake_dataset(items: dict):
    class _FakeData:
        def items(self_):
            return list(items.items())

    class _FakeDataset:
        data = _FakeData()
        config = {}

    return _FakeDataset()


def _reflection_ratio(refl):
    """W = Y2/Y1 from the stored segment spectra after fft_spectrum."""
    y1 = refl.first_reflection.processing_dict["fft_spectrum"]
    y2 = refl.second_reflection.processing_dict["fft_spectrum"]
    return y2 / y1


def _fft_both_segments(refl):
    dataset = _fake_dataset({refl.filename: refl})
    thz.fft_spectrum(dataset, segment="second_reflection", n_fft=_N_FFT)
    thz.fft_spectrum(dataset, segment="first_reflection", n_fft=_N_FFT)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_reflection_ratio_invariant_to_rigid_axis_shift():
    """W = Y2/Y1 must be identical whether or not the sample axis is shifted."""
    refl_unshifted = _make_shared_reflection("s.acc", shift_ps=0.0)
    refl_shifted = _make_shared_reflection("s.acc", shift_ps=3 * _DT_PS)  # +3 samples
    _fft_both_segments(refl_unshifted)
    _fft_both_segments(refl_shifted)

    w0 = _reflection_ratio(refl_unshifted)
    w_shift = _reflection_ratio(refl_shifted)
    freq = refl_unshifted.second_reflection.processing_dict["fft_freq"]
    band = (freq >= 0.3e12) & (freq <= 3.0e12)
    assert np.allclose(w0[band], w_shift[band], rtol=1e-8, atol=1e-10)


def test_reflection_ratio_encodes_true_interpulse_delay():
    """W's phase slope must equal the physical front->back delay (t2 - t1), shift-free."""
    front_ps, back_ps = 155.0, 180.0
    refl = _make_shared_reflection("s.acc", front_ps=front_ps, back_ps=back_ps)
    _fft_both_segments(refl)
    freq = refl.second_reflection.processing_dict["fft_freq"]
    w = _reflection_ratio(refl)
    band = (freq >= 0.3e12) & (freq <= 2.0e12)
    slope = np.polyfit(2 * np.pi * freq[band], np.unwrap(np.angle(w[band])), 1)[0]
    measured_delay_s = -slope                      # exp(-2πi f Δt) -> slope = -Δt
    expected_delay_s = (back_ps - front_ps) * 1e-12
    assert abs(measured_delay_s - expected_delay_s) < 0.02e-12


def test_factor_not_applied_to_plain_thzdata_transmission():
    """Transmission (plain THzData) must keep trace-relative phase (no absolute-time factor)."""
    base_ps = np.arange(150.0, 190.0, _DT_PS)
    pulse = _gaussian(base_ps, 165.0, 1.0, 0.30)
    obj = THzData(
        data=[BaseTHzData(data=np.column_stack([base_ps, pulse]), headers=[])],
        header=None, filename="trans.acc",
    )
    dataset = _fake_dataset({"trans.acc": obj})
    thz.fft_spectrum(dataset, segment="second_reflection", n_fft=_N_FFT)

    stored = obj.processing_dict["fft_spectrum"]
    time_s = obj.data  # already frequency domain now; recompute from the prefft copy
    prefft = obj.processing_dict["time_domain_prefft"]
    expected = np.fft.rfft(prefft[:, 1], n=_N_FFT)  # NO exp(-2πi f t0) factor
    assert np.allclose(stored, expected, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

_TESTS = [
    test_reflection_ratio_invariant_to_rigid_axis_shift,
    test_reflection_ratio_encodes_true_interpulse_delay,
    test_factor_not_applied_to_plain_thzdata_transmission,
]


def main() -> int:
    passed = 0
    failed = 0
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
