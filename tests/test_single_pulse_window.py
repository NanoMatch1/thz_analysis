"""Unit tests for window_single_pulse_fixed_width — single-pulse fixed-width window.

The single-reflection counterpart of window_pulses_fixed_width. Same guarantees:
identical window for every trace, peak-centred, and the data is NEVER shifted (so the
relative sample-vs-reference timing — the misalignment signature — is preserved).

Run with:
    .venv/Scripts/python.exe tests/test_single_pulse_window.py
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.data_structures.thz import BaseTHzData, THzData
from dataset_core.adapters.thz_adapter import window_single_pulse_fixed_width

_DT_PS = 0.05


def _gaussian(time_ps, peak_ps, width_ps=0.3):
    return 1.0 + 5.0 * np.exp(-0.5 * ((time_ps - peak_ps) / width_ps) ** 2)


def _make_plain(filename, peak_ps, end_ps=40.0):
    time_ps = np.arange(0.0, end_ps, _DT_PS)
    trace = np.column_stack([time_ps, _gaussian(time_ps, peak_ps)])
    obj = THzData(data=[BaseTHzData(data=trace.copy(), headers=[])], header=None, filename=filename)
    return obj, obj.data[:, 0].copy()


def _fake_dataset(items):
    class _Data:
        def items(self_):
            return list(items.items())
    class _DS:
        data = _Data()
    return _DS()


def _window_from(obj):
    amplitude = obj.processing_dict['pre_fixed_window'][:, 1]
    return obj.data[:, 1] / amplitude


# ---------------------------------------------------------------------------

def test_peak_centred_and_no_shift():
    obj, time_s = _make_plain("s.acc", peak_ps=12.0)
    window_single_pulse_fixed_width(_fake_dataset({"s.acc": obj}), half_width_ps=2.0)
    # time axis unchanged (no shift)
    assert np.array_equal(obj.data[:, 0], time_s)
    plan = obj.processing_dict['fixed_window']
    peak, half = plan['peak_index'], plan['half_width_samples']
    window = _window_from(obj)
    assert abs(window[peak] - 1.0) < 1e-9
    assert np.all(window[:peak - half] == 0.0)
    assert np.all(window[peak + half + 1:] == 0.0)


def test_identical_window_across_different_peaks():
    obj_a, _ = _make_plain("a.acc", peak_ps=12.0)
    obj_b, _ = _make_plain("b.acc", peak_ps=20.0)
    window_single_pulse_fixed_width(_fake_dataset({"a.acc": obj_a, "b.acc": obj_b}), half_width_ps=2.0)
    half = obj_a.processing_dict['fixed_window']['half_width_samples']
    cores = []
    for obj in (obj_a, obj_b):
        peak = obj.processing_dict['fixed_window']['peak_index']
        cores.append(_window_from(obj)[peak - half: peak + half + 1])
    assert np.allclose(cores[0], cores[1])


def test_relative_timing_preserved():
    """Two pulses at different times keep their peak separation (data not shifted)."""
    obj_a, _ = _make_plain("a.acc", peak_ps=12.0)
    obj_b, _ = _make_plain("b.acc", peak_ps=20.0)
    window_single_pulse_fixed_width(_fake_dataset({"a.acc": obj_a, "b.acc": obj_b}), half_width_ps=2.0)
    peak_a = obj_a.processing_dict['fixed_window']['peak_index']
    peak_b = obj_b.processing_dict['fixed_window']['peak_index']
    # 8 ps separation at 0.05 ps/sample = 160 samples, preserved
    assert peak_b - peak_a == 160


def test_region_constrains_peak_search():
    """A region must select a local peak, not the global max outside it."""
    obj, _ = _make_plain("s.acc", peak_ps=20.0)
    # plant a taller spike OUTSIDE the region; the region must ignore it
    time_ps = np.arange(0.0, 40.0, _DT_PS)
    obj.data[np.argmin(np.abs(time_ps - 5.0)), 1] = 100.0
    window_single_pulse_fixed_width(
        _fake_dataset({"s.acc": obj}), half_width_ps=2.0, region_ps=(17.0, 23.0))
    peak_time = obj.processing_dict['fixed_window']['peak_time_seconds'] * 1e12
    assert 19.0 < peak_time < 21.0   # found the in-region pulse, not the 5 ps spike


def test_mismatched_dt_raises():
    obj_a, _ = _make_plain("a.acc", peak_ps=12.0)
    time_ps = np.arange(0.0, 40.0, 0.04)  # different dt
    trace = np.column_stack([time_ps, _gaussian(time_ps, 12.0)])
    obj_b = THzData(data=[BaseTHzData(data=trace, headers=[])], header=None, filename="b.acc")
    raised = False
    try:
        window_single_pulse_fixed_width(_fake_dataset({"a.acc": obj_a, "b.acc": obj_b}), half_width_ps=2.0)
    except ValueError as error:
        raised = "sample step" in str(error)
    assert raised


_TESTS = [
    test_peak_centred_and_no_shift,
    test_identical_window_across_different_peaks,
    test_relative_timing_preserved,
    test_region_constrains_peak_search,
    test_mismatched_dt_raises,
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
