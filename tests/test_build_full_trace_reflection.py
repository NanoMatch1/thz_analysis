"""Regression tests for build_full_trace_reflection's region-driven isolation.

The function takes the generous, dataset-independent search ranges in
``config['regions']`` and:
  1. finds the actual non-zero data extent inside each range,
  2. clips the shared axis to [first_found_start, second_found_end] and zeros the
     section between (and any padding beyond) the two pulses,
  3. writes the found (tightened) ranges back into config['regions'],
while preserving each pulse's true position on ONE shared axis and preserving the
per-scan structure (so per-scan SNR statistics survive).

Dependency-free; run with the project venv:
    .venv/Scripts/python.exe tests/test_build_full_trace_reflection.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acquisition_editor import save_acc
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz
from dataset_core.data_structures.thz import THzDataReflection


DT_PS = 0.05
FIRST_PS = 155.0   # front (alignment) reflection peak
SECOND_PS = 180.0  # back (sample) reflection peak

# Generous general search ranges (kept constant across datasets); each extends
# past the acquired data / real pulse so the "find non-zero" logic has work to do.
FIRST_SEARCH = (146.0, 160.0)
SECOND_SEARCH = (170.0, 188.0)


def _pulse(t_ps, centre, amp, width=0.3):
    """Single-cycle-ish pulse: derivative of a Gaussian."""
    x = (t_ps - centre) / width
    return amp * -x * np.exp(-(x**2) / 2.0)


def _write_multiscan_acc(path, name, n_scans, *, t_start_ps, t_stop_ps,
                         second_amp, rng, lead_zeros_ps=0.0):
    """Write an .acc spanning [t_start, t_stop] ps with n_scans noisy repeats.

    ``lead_zeros_ps`` forces an explicit zero-padded stretch at the very start of
    the record (mimicking a shorter acquisition padded up to a common grid), so the
    non-zero-bounds trimming is exercised even when samples nominally exist there.
    """
    t_ps = np.arange(t_start_ps, t_stop_ps, DT_PS)
    columns = [t_ps]
    scan_headers = []
    for scan_index in range(n_scans):
        clean = _pulse(t_ps, FIRST_PS, 1.0) + _pulse(t_ps, SECOND_PS, second_amp)
        noisy = clean + rng.normal(0.0, 0.01, size=t_ps.size)
        if lead_zeros_ps > 0.0:
            noisy[t_ps < t_start_ps + lead_zeros_ps] = 0.0
        columns.append(noisy)
        scan_headers.append([f"title {name} acc {scan_index + 1}", "type 0"])
    data = np.column_stack(columns)
    save_acc({"data": data, "scan_headers": scan_headers, "header": scan_headers[0]}, path)


def _load_dataset(tmpdir, *, regions, t_start_ps=150.0, t_stop_ps=185.0,
                  lead_zeros_ps=0.0):
    rng = np.random.default_rng(0)
    _write_multiscan_acc(
        os.path.join(tmpdir, "reference_x.acc"), "reference_x", 4,
        t_start_ps=t_start_ps, t_stop_ps=t_stop_ps, second_amp=-0.4, rng=rng,
        lead_zeros_ps=lead_zeros_ps,
    )
    _write_multiscan_acc(
        os.path.join(tmpdir, "sample_x.acc"), "sample_x", 6,
        t_start_ps=t_start_ps, t_stop_ps=t_stop_ps, second_amp=-0.25, rng=rng,
        lead_zeros_ps=lead_zeros_ps,
    )
    config = {"regions": regions} if regions is not None else {}
    dataset = DataSet(tmpdir, config=config)
    dataset.load_all_data(case_insensitive=True)
    return dataset, config


def test_config_regions_are_tightened_to_data():
    """The generous search ranges collapse to the acquired data extent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset, config = _load_dataset(
            tmpdir,
            regions={"first_reflection": FIRST_SEARCH, "second_reflection": SECOND_SEARCH},
            t_start_ps=150.0, t_stop_ps=185.0,
        )
        thz.build_full_trace_reflection(dataset, config)

        first = config["regions"]["first_reflection"]
        second = config["regions"]["second_reflection"]
        # First search started at 146 but data starts at 150 -> found start ~150.
        assert 149.9 <= first[0] <= 150.2, first
        assert first[1] <= FIRST_SEARCH[1] + 1e-6, first
        # Second search ended at 188 but data ends at ~185 -> found end ~185.
        assert second[1] <= 185.0 + 1e-6, second
        assert second[0] >= SECOND_SEARCH[0] - 1e-6, second


def test_gap_between_pulses_is_zeroed_and_pulses_preserved():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset, config = _load_dataset(
            tmpdir,
            regions={"first_reflection": FIRST_SEARCH, "second_reflection": SECOND_SEARCH},
        )
        thz.build_full_trace_reflection(dataset, config)

        obj = dataset.data["sample_x.acc"]
        assert isinstance(obj, THzDataReflection)
        trace = obj.second_reflection.data          # averaged [time_s, mean, stderr]
        time_ps = trace[:, 0] * thz._S_TO_PS
        amplitude = trace[:, 1]

        first = config["regions"]["first_reflection"]
        second = config["regions"]["second_reflection"]

        # Everything strictly between the two found regions must be exactly zero.
        # Stay half a sample step clear of the found boundaries: the samples ON the
        # boundaries are legitimately KEPT, and a ps->s->ps roundtrip perturbs their
        # times by ~1e-15 ps, which would otherwise pull a kept edge sample into the gap.
        boundary_guard_ps = DT_PS / 2.0
        in_gap = (time_ps > first[1] + boundary_guard_ps) & (time_ps < second[0] - boundary_guard_ps)
        assert np.any(in_gap), "expected a non-empty inter-pulse gap"
        assert np.allclose(amplitude[in_gap], 0.0), "inter-pulse gap not zeroed"

        # Both pulse peaks survive inside their found regions. (The test pulse is a
        # derivative-of-Gaussian, whose |amplitude| extremum sits at centre ± width
        # = ±0.3 ps, so allow a 0.4 ps tolerance around the nominal centre.)
        in_first = (time_ps >= first[0]) & (time_ps <= first[1])
        in_second = (time_ps >= second[0]) & (time_ps <= second[1])
        assert abs(time_ps[in_first][np.argmax(np.abs(amplitude[in_first]))] - FIRST_PS) < 0.4
        assert abs(time_ps[in_second][np.argmax(np.abs(amplitude[in_second]))] - SECOND_PS) < 0.4


def test_outer_axis_is_clipped_to_found_bounds():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset, config = _load_dataset(
            tmpdir,
            regions={"first_reflection": FIRST_SEARCH, "second_reflection": SECOND_SEARCH},
            t_start_ps=150.0, t_stop_ps=185.0,
        )
        thz.build_full_trace_reflection(dataset, config)

        obj = dataset.data["sample_x.acc"]
        time_ps = obj.second_reflection.data[:, 0] * thz._S_TO_PS
        first = config["regions"]["first_reflection"]
        second = config["regions"]["second_reflection"]
        # No samples before the first found start or after the second found end.
        assert time_ps.min() >= first[0] - 1e-6, time_ps.min()
        assert time_ps.max() <= second[1] + 1e-6, time_ps.max()


def test_leading_zero_padding_is_trimmed():
    """An explicit zero-padded stretch inside the first search range is trimmed off."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Data spans [148,185] but first 4 ps are zero-padded -> real data from 152.
        dataset, config = _load_dataset(
            tmpdir,
            regions={"first_reflection": (147.0, 160.0), "second_reflection": SECOND_SEARCH},
            t_start_ps=148.0, t_stop_ps=185.0, lead_zeros_ps=4.0,
        )
        thz.build_full_trace_reflection(dataset, config)
        first = config["regions"]["first_reflection"]
        # Found start should skip the zero pad (>= ~152), not sit at 148.
        assert first[0] >= 151.9, first


def test_per_scan_structure_preserved():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset, config = _load_dataset(
            tmpdir,
            regions={"first_reflection": FIRST_SEARCH, "second_reflection": SECOND_SEARCH},
        )
        thz.build_full_trace_reflection(dataset, config)
        obj = dataset.data["sample_x.acc"]
        # raw_data columns = [time, scan1, ..., scanN]; sample had 6 scans.
        assert np.asarray(obj.second_reflection.raw_data).shape[1] - 1 == 6
        assert np.asarray(obj.first_reflection.raw_data).shape[1] - 1 == 6


def test_whole_trace_fallback_when_regions_absent():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset, config = _load_dataset(tmpdir, regions=None,
                                        t_start_ps=150.0, t_stop_ps=185.0)
        thz.build_full_trace_reflection(dataset, config)
        obj = dataset.data["sample_x.acc"]
        assert isinstance(obj, THzDataReflection)
        # Regions deferred (None) and the whole trace is kept (spans ~[150,185]).
        assert obj.first_region is None and obj.second_region is None
        time_ps = obj.second_reflection.data[:, 0] * thz._S_TO_PS
        assert time_ps.min() < 151.0 and time_ps.max() > 184.0


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"FAIL  {fn.__name__}: {error}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
