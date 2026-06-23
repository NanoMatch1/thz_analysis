"""Unit tests for window_pulses_fixed_width — the fixed-width symmetric window.

The whole point of this window is that it is IDENTICAL for every pulse: the
region only locates the pulse, the window length is fixed by half_width_ps, and
the data is never time-shifted. These tests pin exactly those guarantees.

Run with:
    .venv/Scripts/python.exe tests/test_fixed_width_window.py
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.data_structures.thz import BaseTHzData, THzData, THzDataReflection
from dataset_core.adapters.thz_adapter import (
    window_pulses_fixed_width,
    _build_symmetric_window,
)

_S_TO_PS = 1e12
_DT_PS = 0.05  # 0.05 ps sampling step (raw files store time in picoseconds)


def _gaussian_pulse(time_ps: np.ndarray, peak_ps: float, width_ps: float = 0.3) -> np.ndarray:
    """A unit-amplitude Gaussian bump centred at peak_ps (on a +1.0 baseline)."""
    return 1.0 + 5.0 * np.exp(-0.5 * ((time_ps - peak_ps) / width_ps) ** 2)


def _make_reflection(filename: str, first_peak_ps: float, second_peak_ps: float):
    """Full-trace reflection object: both segments share one axis, two pulses on it.

    The raw trace is built in PICOSECONDS (the file convention); THzData converts
    column 0 to SI seconds when it averages. Returns the resulting seconds axis.
    """
    time_ps = np.arange(0.0, 40.0, _DT_PS)
    amplitude = (_gaussian_pulse(time_ps, first_peak_ps)
                 + (_gaussian_pulse(time_ps, second_peak_ps) - 1.0))  # keep one +1 baseline
    trace = np.column_stack([time_ps, amplitude])
    first = THzData(data=[BaseTHzData(data=trace.copy(), headers=[])], header=None, filename=filename)
    second = THzData(data=[BaseTHzData(data=trace.copy(), headers=[])], header=None, filename=filename)
    refl = THzDataReflection.from_thzdata(second, first)
    refl.first_region = (first_peak_ps - 3.0, first_peak_ps + 3.0)
    refl.second_region = (second_peak_ps - 3.0, second_peak_ps + 3.0)
    time_s = refl.first_reflection.data[:, 0].copy()
    return refl, time_s


def _fake_dataset(items: dict):
    class _FakeData:
        def items(self_):
            return list(items.items())

    class _FakeDataset:
        data = _FakeData()

    return _FakeDataset()


def _window_function_from(holder) -> np.ndarray:
    """Recover the applied window (windowed / amplitude; amplitude is never zero)."""
    amplitude = holder.processing_dict['pre_fixed_window'][:, 1]
    windowed = holder.data[:, 1]
    return windowed / amplitude


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_build_symmetric_window_matches_scipy():
    from scipy.signal.windows import hann, tukey
    assert np.allclose(_build_symmetric_window(81, 'hann', 1.0), hann(81))
    assert np.allclose(_build_symmetric_window(81, 'tukey', 0.4), tukey(81, alpha=0.4))
    assert np.allclose(_build_symmetric_window(81, 'boxcar', 1.0), np.ones(81))


def test_window_is_peak_centred_and_symmetric():
    refl, _ = _make_reflection("s.acc", first_peak_ps=10.0, second_peak_ps=25.0)
    window_pulses_fixed_width(_fake_dataset({"s.acc": refl}), half_width_ps=2.0)

    plan = refl.first_reflection.processing_dict['fixed_window']
    peak = plan['peak_index']
    half = plan['half_width_samples']
    window = _window_function_from(refl.first_reflection)
    # weight 1.0 at the peak, symmetric about it, zero outside +/- half
    assert abs(window[peak] - 1.0) < 1e-9
    assert np.allclose(window[peak - half: peak], window[peak + half: peak: -1])
    assert np.all(window[:peak - half] == 0.0)
    assert np.all(window[peak + half + 1:] == 0.0)


def test_window_identical_across_pulses_despite_different_peaks():
    """Two files with DIFFERENT peak positions get a byte-identical window shape."""
    refl_a, _ = _make_reflection("a.acc", first_peak_ps=10.0, second_peak_ps=25.0)
    refl_b, _ = _make_reflection("b.acc", first_peak_ps=10.7, second_peak_ps=25.4)
    window_pulses_fixed_width(
        _fake_dataset({"a.acc": refl_a, "b.acc": refl_b}), half_width_ps=2.0,
    )

    half = refl_a.first_reflection.processing_dict['fixed_window']['half_width_samples']
    cores = []
    for refl in (refl_a, refl_b):
        for segment in ('first_reflection', 'second_reflection'):
            holder = getattr(refl, segment)
            peak = holder.processing_dict['fixed_window']['peak_index']
            window = _window_function_from(holder)
            cores.append(window[peak - half: peak + half + 1])
    reference_core = cores[0]
    for core in cores[1:]:
        assert np.allclose(core, reference_core), "window shape must be identical for every pulse"


def test_window_length_fixed_independent_of_region_width():
    """A wider second region must NOT produce a wider window (the original leak)."""
    refl, _ = _make_reflection("s.acc", first_peak_ps=10.0, second_peak_ps=25.0)
    refl.second_region = (20.0, 30.0)  # deliberately much wider than first_region
    window_pulses_fixed_width(_fake_dataset({"s.acc": refl}), half_width_ps=2.0)

    first_len = refl.first_reflection.processing_dict['fixed_window']['window_length']
    second_len = refl.second_reflection.processing_dict['fixed_window']['window_length']
    assert first_len == second_len, "window length must not depend on region width"


def test_time_axis_unchanged_no_shift():
    refl, time_s = _make_reflection("s.acc", first_peak_ps=10.0, second_peak_ps=25.0)
    window_pulses_fixed_width(_fake_dataset({"s.acc": refl}), half_width_ps=2.0)
    assert np.array_equal(refl.first_reflection.data[:, 0], time_s)
    assert np.array_equal(refl.second_reflection.data[:, 0], time_s)


def test_other_pulse_is_zeroed():
    """The first-reflection window must zero the region around the second pulse."""
    refl, time_s = _make_reflection("s.acc", first_peak_ps=10.0, second_peak_ps=25.0)
    window_pulses_fixed_width(_fake_dataset({"s.acc": refl}), half_width_ps=2.0)
    windowed_first = refl.first_reflection.data[:, 1]
    second_peak_index = int(np.argmin(np.abs(time_s - 25e-12)))
    assert windowed_first[second_peak_index] == 0.0


def test_clip_flag_set_when_window_runs_off_edge():
    refl, _ = _make_reflection("s.acc", first_peak_ps=1.0, second_peak_ps=25.0)
    refl.first_region = (0.0, 3.0)  # peak near the very start of the trace
    window_pulses_fixed_width(_fake_dataset({"s.acc": refl}), half_width_ps=3.0)
    assert refl.first_reflection.processing_dict['fixed_window']['clipped'] is True


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

_TESTS = [
    test_build_symmetric_window_matches_scipy,
    test_window_is_peak_centred_and_symmetric,
    test_window_identical_across_pulses_despite_different_peaks,
    test_window_length_fixed_independent_of_region_width,
    test_time_axis_unchanged_no_shift,
    test_other_pulse_is_zeroed,
    test_clip_flag_set_when_window_runs_off_edge,
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


if __name__ == '__main__':
    sys.exit(main())
