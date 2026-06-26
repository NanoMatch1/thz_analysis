"""Unit tests for window_time_fixed_width — peak-centred fixed-width transmission window.

Centers the pulse by EXTENDING the axis (preserving the pulse's absolute time, so the
sample<->reference group delay that carries n is not disturbed) and applies a fixed-width
symmetric window. These pin: peak centred after the step, absolute peak TIME preserved
(the timing-safety property), window symmetric, and region-constrained peak search.

Run with:
    .venv/Scripts/python.exe tests/test_window_time_fixed_width.py
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.data_structures.thz import BaseTHzData, THzData
from dataset_core.adapters.thz_adapter import (
    window_time_fixed_width,
    _local_max_index_near,
)

_DT_PS = 0.05


def _gaussian(time_ps, peak_ps, width_ps=0.3):
    return 5.0 * np.exp(-0.5 * ((time_ps - peak_ps) / width_ps) ** 2)


def _make_plain(filename, peak_ps, end_ps=40.0):
    time_ps = np.arange(0.0, end_ps, _DT_PS)
    trace = np.column_stack([time_ps, _gaussian(time_ps, peak_ps)])
    return THzData(data=[BaseTHzData(data=trace.copy(), headers=[])], header=None, filename=filename)


def _fake_dataset(items):
    class _Data:
        def items(self_):
            return list(items.items())
    class _DS:
        data = _Data()
    return _DS()


def _peak_time_ps(obj):
    return obj.data[np.argmax(np.abs(obj.data[:, 1])), 0] * 1e12


# ---------------------------------------------------------------------------

def test_absolute_peak_time_preserved():
    """The timing-safety property: centering must NOT move the pulse's absolute time."""
    obj = _make_plain("s.acc", peak_ps=8.0)  # early pulse -> needs prepend to centre
    before = _peak_time_ps(obj)
    window_time_fixed_width(_fake_dataset({"s.acc": obj}), half_width_ps=2.0, center_in_trace=True)
    after = _peak_time_ps(obj)
    assert abs(after - before) < _DT_PS, (before, after)
    assert abs(before - 8.0) < _DT_PS


def test_peak_centred_in_trace():
    """After centering, the peak sits at the trace midpoint (balanced samples each side)."""
    obj = _make_plain("s.acc", peak_ps=8.0)
    window_time_fixed_width(_fake_dataset({"s.acc": obj}), half_width_ps=2.0, center_in_trace=True)
    peak_index = int(np.argmax(np.abs(obj.data[:, 1])))
    midpoint = obj.data.shape[0] // 2
    assert abs(peak_index - midpoint) <= 1


def test_window_symmetric_and_zero_outside():
    obj = _make_plain("s.acc", peak_ps=20.0)
    window_time_fixed_width(_fake_dataset({"s.acc": obj}), half_width_ps=2.0, center_in_trace=False)
    plan = obj.processing_dict['fixed_window']
    peak, half = plan['peak_index'], plan['half_width_samples']
    amplitude = obj.processing_dict['pre_fixed_window'][:, 1]
    window = obj.data[:, 1] / np.where(amplitude == 0, 1e-30, amplitude)
    assert np.all(obj.data[:peak - half, 1] == 0.0)
    assert np.all(obj.data[peak + half + 1:, 1] == 0.0)
    # symmetric weights about the peak
    assert np.allclose(window[peak - half: peak], window[peak + half: peak: -1], atol=1e-6)


def test_no_center_preserves_length_and_time():
    obj = _make_plain("s.acc", peak_ps=20.0)
    time_before = obj.data[:, 0].copy()
    window_time_fixed_width(_fake_dataset({"s.acc": obj}), half_width_ps=2.0, center_in_trace=False)
    assert np.array_equal(obj.data[:, 0], time_before)  # no axis change without centering


def test_region_constrains_peak_search():
    obj = _make_plain("s.acc", peak_ps=20.0)
    time_ps = np.arange(0.0, 40.0, _DT_PS)
    obj.data[np.argmin(np.abs(time_ps - 5.0)), 1] = 100.0  # taller spike outside region
    window_time_fixed_width(_fake_dataset({"s.acc": obj}), half_width_ps=2.0,
                            region_ps=(17.0, 23.0), center_in_trace=False)
    assert 19.0 < obj.processing_dict['fixed_window']['peak_time_seconds'] * 1e12 < 21.0


def test_peak_ps_snaps_to_chosen_pulse_not_global_max():
    """The large-delay case: a taller spike far away must NOT be chosen when peak_ps
    targets the (smaller) pulse of interest — the snap window isolates it."""
    obj = _make_plain("s.acc", peak_ps=22.0)        # the wanted pulse (amplitude 5)
    time_ps = np.arange(0.0, 40.0, _DT_PS)
    obj.data[np.argmin(np.abs(time_ps - 6.0)), 1] = 100.0  # a much taller spike elsewhere
    window_time_fixed_width(_fake_dataset({"s.acc": obj}), half_width_ps=2.0,
                            peak_ps=22.0, snap_halfwidth_ps=2.0, center_in_trace=False)
    chosen_ps = obj.processing_dict['fixed_window']['peak_time_seconds'] * 1e12
    assert 21.0 < chosen_ps < 23.0     # snapped to the wanted pulse, not the 6 ps spike


def test_local_max_index_near_picks_local_not_global():
    time_ps = np.arange(0.0, 40.0, _DT_PS)
    time_s = time_ps * 1e-12
    amplitude = _gaussian(time_ps, 22.0)
    amplitude[np.argmin(np.abs(time_ps - 6.0))] = 100.0   # global max far from target
    idx = _local_max_index_near(time_s, amplitude, 22.0e-12, snap_halfwidth_ps=2.0)
    assert abs(time_ps[idx] - 22.0) < 0.5


_TESTS = [
    test_absolute_peak_time_preserved,
    test_peak_centred_in_trace,
    test_window_symmetric_and_zero_outside,
    test_no_center_preserves_length_and_time,
    test_region_constrains_peak_search,
    test_peak_ps_snaps_to_chosen_pulse_not_global_max,
    test_local_max_index_near_picks_local_not_global,
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
