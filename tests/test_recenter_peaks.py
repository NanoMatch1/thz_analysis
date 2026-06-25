"""Unit tests for recenter_peaks_to_common_t0 — the centering recalibration.

Resets every plain THzData pulse to a common peak T0 (removes relative timing), used to
SIMULATE a delay-correction for angular misalignment. These pin: peaks end up aligned,
the time axis is preserved, the target selection (reference / explicit) works, and the
applied shift is recorded.

Run with:
    .venv/Scripts/python.exe tests/test_recenter_peaks.py
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.data_structures.thz import BaseTHzData, THzData
from dataset_core.adapters.thz_adapter import recenter_peaks_to_common_t0

_DT_PS = 0.05


def _gaussian(time_ps, peak_ps, width_ps=0.3):
    return 5.0 * np.exp(-0.5 * ((time_ps - peak_ps) / width_ps) ** 2)


def _make_plain(filename, peak_ps, end_ps=40.0):
    time_ps = np.arange(0.0, end_ps, _DT_PS)
    trace = np.column_stack([time_ps, _gaussian(time_ps, peak_ps)])
    return THzData(data=[BaseTHzData(data=trace.copy(), headers=[])], header=None, filename=filename)


def _fake_dataset(items, references=()):
    refs = set(references)
    class _Data:
        def items(self_):
            return list(items.items())
        def is_reference(self_, name):
            return name in refs
    class _DS:
        data = _Data()
    return _DS()


def _peak_time_ps(obj):
    return obj.data[np.argmax(np.abs(obj.data[:, 1])), 0] * 1e12


# ---------------------------------------------------------------------------

def test_peaks_aligned_to_reference():
    ref = _make_plain("reference_x.acc", peak_ps=12.0)
    sam = _make_plain("sample_y.acc", peak_ps=20.0)
    time_s = sam.data[:, 0].copy()
    recenter_peaks_to_common_t0(
        _fake_dataset({"reference_x.acc": ref, "sample_y.acc": sam},
                      references={"reference_x.acc"}))
    # both peaks now at the reference peak time (12 ps), within a sub-sample
    assert abs(_peak_time_ps(ref) - 12.0) < _DT_PS
    assert abs(_peak_time_ps(sam) - 12.0) < _DT_PS
    # time axis preserved
    assert np.array_equal(sam.data[:, 0], time_s)


def test_explicit_target_t0():
    a = _make_plain("a.acc", peak_ps=10.0)
    b = _make_plain("b.acc", peak_ps=25.0)
    recenter_peaks_to_common_t0(_fake_dataset({"a.acc": a, "b.acc": b}), target_t0_ps=18.0)
    assert abs(_peak_time_ps(a) - 18.0) < _DT_PS
    assert abs(_peak_time_ps(b) - 18.0) < _DT_PS


def test_shift_recorded_and_reference_unmoved():
    ref = _make_plain("reference_x.acc", peak_ps=15.0)
    sam = _make_plain("sample_y.acc", peak_ps=21.0)
    recenter_peaks_to_common_t0(
        _fake_dataset({"reference_x.acc": ref, "sample_y.acc": sam},
                      references={"reference_x.acc"}))
    # reference is the anchor -> zero shift; sample shifts by -6 ps (21 -> 15)
    assert abs(ref.processing_dict['recenter']['shift_seconds']) < 1e-15
    assert abs(sam.processing_dict['recenter']['shift_seconds'] * 1e12 - (-6.0)) < 1e-6


def test_integer_roll_mode_runs_and_aligns():
    a = _make_plain("a.acc", peak_ps=12.0)
    b = _make_plain("b.acc", peak_ps=20.0)
    recenter_peaks_to_common_t0(
        _fake_dataset({"a.acc": a, "b.acc": b}), target_t0_ps=12.0, subsample=False)
    assert abs(_peak_time_ps(a) - 12.0) < _DT_PS
    assert abs(_peak_time_ps(b) - 12.0) < _DT_PS


_TESTS = [
    test_peaks_aligned_to_reference,
    test_explicit_target_t0,
    test_shift_recorded_and_reference_unmoved,
    test_integer_roll_mode_runs_and_aligns,
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
