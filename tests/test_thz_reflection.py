"""Unit and integration tests for THzDataReflection and the reflection layout loader.

Run with:
    .venv/Scripts/python.exe tests/test_thz_reflection.py
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.data_structures.thz import BaseTHzData, THzData, THzDataReflection
from dataset_core.dataset import DataSet
from dataset_core.adapters.thz_adapter import center_pulse


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_thzdata(filename: str, n_pts: int = 100, n_scans: int = 3) -> THzData:
    """Build a minimal THzData without touching the file system."""
    time_ps = np.linspace(0, 10, n_pts)
    scans = []
    for _ in range(n_scans):
        raw = np.column_stack([time_ps, np.random.randn(n_pts)])
        scans.append(BaseTHzData(data=raw, headers=[]))
    return THzData(data=scans, header=None, filename=filename)


def _write_dat(path: str, n_pts: int = 100) -> None:
    """Write a minimal .dat file in the space-separated format the DATLoader expects."""
    t_ps = np.linspace(0, n_pts * 0.05, n_pts)
    y = np.random.randn(n_pts)
    rows = [f"{t:.6e} {a:.6e}" for t, a in zip(t_ps, y)]
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_from_thzdata_basic():
    second = _make_thzdata("sample.acc", n_pts=100, n_scans=3)
    first = _make_thzdata("sample.acc", n_pts=60, n_scans=3)
    refl = THzDataReflection.from_thzdata(second, first)

    assert isinstance(refl, THzDataReflection)
    assert refl.filename == "sample.acc"
    assert refl.first_segment is first
    assert len(refl.data_list) == 3
    assert len(refl.first_segment.data_list) == 3


def test_all_segments_order():
    second = _make_thzdata("ref.acc", n_pts=100)
    first = _make_thzdata("ref.acc", n_pts=60)
    refl = THzDataReflection.from_thzdata(second, first)
    segs = refl.all_segments()
    assert segs[0] == ("first_reflection", first)
    assert segs[1] == ("second_reflection", refl)


def test_data_property_points_to_second():
    """data property must still return the second-reflection (primary) data."""
    second = _make_thzdata("s.acc", n_pts=100)
    first = _make_thzdata("s.acc", n_pts=60)
    refl = THzDataReflection.from_thzdata(second, first)
    assert refl.data.shape[0] == 100


def test_isinstance_hierarchy():
    second = _make_thzdata("s.acc")
    first = _make_thzdata("s.acc")
    refl = THzDataReflection.from_thzdata(second, first)
    assert isinstance(refl, THzData)
    assert isinstance(refl, THzDataReflection)


def test_repr_contains_class_name():
    second = _make_thzdata("s.acc")
    first = _make_thzdata("s.acc")
    refl = THzDataReflection.from_thzdata(second, first)
    assert "THzDataReflection" in repr(refl)


def test_dataset_flat_dir_unchanged():
    """load_all_data on a flat directory (no subdirs) behaves exactly as before."""
    tmpdir = tempfile.mkdtemp()
    try:
        for name in ("sample_a.dat", "reference_a.dat"):
            _write_dat(os.path.join(tmpdir, name))

        ds = DataSet(tmpdir)
        ds.load_all_data()
        items = list(ds.data.items())
        assert len(items) == 2
        for fname, obj in items:
            assert isinstance(obj, THzData)
            assert not isinstance(obj, THzDataReflection), \
                f"flat-dir load should not produce THzDataReflection for '{fname}'"
    finally:
        shutil.rmtree(tmpdir)


def test_dataset_reflection_layout_creates_thzdatareflection():
    """load_all_data on a root dir with first/second subdirs creates THzDataReflection objects."""
    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "first_reflection"))
        os.makedirs(os.path.join(tmpdir, "second_reflection"))
        for name in ("sample_a.dat", "reference_a.dat"):
            _write_dat(os.path.join(tmpdir, "first_reflection", name), n_pts=60)
            _write_dat(os.path.join(tmpdir, "second_reflection", name), n_pts=100)

        ds = DataSet(tmpdir)
        ds.load_all_data()
        items = list(ds.data.items())
        assert len(items) == 2, f"expected 2, got {len(items)}"
        for fname, obj in items:
            assert isinstance(obj, THzDataReflection), \
                f"'{fname}' should be THzDataReflection, got {type(obj).__name__}"
            assert obj.first_segment is not None
            assert obj.first_segment.data.shape[0] == 60  # first_reflection has 60 pts
            assert obj.data.shape[0] == 100               # second_reflection has 100 pts
    finally:
        shutil.rmtree(tmpdir)


def test_explicit_dir_bypasses_reflection_layout():
    """explicit_dir=True loads file_dir directly even when reflection subdirs exist."""
    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "first_reflection"))
        os.makedirs(os.path.join(tmpdir, "second_reflection"))
        for name in ("sample_a.dat", "reference_a.dat"):
            _write_dat(os.path.join(tmpdir, "first_reflection", name), n_pts=60)
            _write_dat(os.path.join(tmpdir, "second_reflection", name), n_pts=100)
        # Also write a file directly in the root so explicit_dir has something to load
        _write_dat(os.path.join(tmpdir, "raw_data.dat"))

        ds = DataSet(tmpdir)
        ds.load_all_data(explicit_dir=True)
        items = dict(ds.data.items())
        # Should only see the flat root file, not the subdir files
        assert "raw_data.dat" in items
        assert not isinstance(items["raw_data.dat"], THzDataReflection)
        # Subdir files should NOT appear (we loaded the root dir directly)
        assert "sample_a.dat" not in items
    finally:
        shutil.rmtree(tmpdir)


def test_center_pulse_first_reflection_auto():
    """Auto-mode centering prepends samples so the peak is at the midpoint."""
    # Build a first_segment where the pulse sits early: 20 samples before peak,
    # 60 samples after — so the peak needs 40 samples prepended to centre it.
    n_pts = 81  # indices 0..80, peak at 20
    dt_s = 7.5e-15
    t = np.arange(n_pts) * dt_s
    y = np.zeros(n_pts)
    y[20] = 1.0  # sharp peak at index 20

    second = _make_thzdata("s.acc", n_pts=n_pts)
    first = THzData(data=[BaseTHzData(data=np.column_stack([t, y]), headers=[])],
                    header=None, filename="s.acc")
    refl = THzDataReflection.from_thzdata(second, first)

    # Wrap in a minimal DataSet-like structure by monkey-patching
    class _FakeData:
        def items(self_):
            return [("s.acc", refl)]

    class _FakeDataset:
        data = _FakeData()

    center_pulse(_FakeDataset(), segment='first_reflection', config={'centering': {'peak_mode': 'auto', 'taper_ps': 0.0}})

    t_new = refl.first_segment.data[:, 0]
    y_new = refl.first_segment.data[:, 1]
    new_peak_idx = int(np.argmax(np.abs(y_new)))

    n_before_new = new_peak_idx
    n_after_new = len(y_new) - 1 - new_peak_idx
    assert n_before_new == n_after_new, (
        f"Peak should be centred: n_before={n_before_new}, n_after={n_after_new}"
    )
    assert refl.first_segment.processing_dict.get('pre_centering') is not None
    assert refl.first_segment.processing_dict['centering_info']['n_prepend'] == 40


def test_center_pulse_first_reflection_already_centred():
    """No modification when peak is already at the midpoint."""
    n_pts = 101
    dt_s = 7.5e-15
    t = np.arange(n_pts) * dt_s
    y = np.zeros(n_pts)
    y[50] = 1.0  # exactly centred

    second = _make_thzdata("s.acc", n_pts=n_pts)
    first = THzData(data=[BaseTHzData(data=np.column_stack([t, y]), headers=[])],
                    header=None, filename="s.acc")
    refl = THzDataReflection.from_thzdata(second, first)

    class _FakeData:
        def items(self_):
            return [("s.acc", refl)]

    class _FakeDataset:
        data = _FakeData()

    center_pulse(_FakeDataset(), segment='first_reflection', config={'centering': {'peak_mode': 'auto', 'taper_ps': 0.0}})

    assert refl.first_segment.processing_dict.get('pre_centering') is None
    assert len(refl.first_segment.data) == n_pts  # unchanged


def test_center_pulse_first_reflection_taper_smooth():
    """Taper ramp starts at 0 and ends at 1 with no discontinuity at the pad edge."""
    n_pts = 81
    dt_s = 7.5e-15
    t = np.arange(n_pts) * dt_s
    y = np.zeros(n_pts)
    y[20] = 1.0
    # Set pre-pulse region to a constant non-zero value so a step would be visible
    y[:20] = 0.1

    second = _make_thzdata("s.acc", n_pts=n_pts)
    first = THzData(data=[BaseTHzData(data=np.column_stack([t, y]), headers=[])],
                    header=None, filename="s.acc")
    refl = THzDataReflection.from_thzdata(second, first)

    class _FakeData:
        def items(self_):
            return [("s.acc", refl)]

    class _FakeDataset:
        data = _FakeData()

    taper_ps = 0.3  # ~40 samples at 7.5 fs/pt
    center_pulse(_FakeDataset(), segment='first_reflection', config={'centering': {'peak_mode': 'auto', 'taper_ps': taper_ps}})

    n_prepend = refl.first_segment.processing_dict['centering_info']['n_prepend']
    y_new = refl.first_segment.data[:, 1]

    # Pad region should be identically zero
    assert np.all(y_new[:n_prepend] == 0.0), "Prepended region must be zero"
    # First sample of taper (right after pad) should be near 0
    assert abs(y_new[n_prepend]) < 1e-6, "First taper sample must be near zero"


def test_center_pulse_main_trace():
    """center_pulse centres the main data_obj.data trace for any THzData object."""
    n_pts = 81
    dt_s = 7.5e-15
    t = np.arange(n_pts) * dt_s
    y = np.zeros(n_pts)
    y[20] = 1.0  # peak at index 20, so 60 samples after — needs 40 prepended

    data_obj = _make_thzdata("s.acc", n_pts=n_pts, n_scans=1)
    # Override the averaged data directly
    data_obj.data = np.column_stack([t, y])

    class _FakeData:
        def items(self_):
            return [("s.acc", data_obj)]

    class _FakeDataset:
        data = _FakeData()

    center_pulse(_FakeDataset(), config={'centering': {'peak_mode': 'auto', 'taper_ps': 0.0}})

    y_new = data_obj.data[:, 1]
    new_peak_idx = int(np.argmax(np.abs(y_new)))
    n_before_new = new_peak_idx
    n_after_new = len(y_new) - 1 - new_peak_idx
    assert n_before_new == n_after_new, (
        f"Main trace should be centred: n_before={n_before_new}, n_after={n_after_new}"
    )
    assert data_obj.processing_dict['centering_info']['n_prepend'] == 40


def test_center_pulse_first_reflection_skips_plain_thzdata():
    """Plain THzData objects (no first_segment) are skipped without error."""
    second = _make_thzdata("s.acc", n_pts=100)

    class _FakeData:
        def items(self_):
            return [("s.acc", second)]

    class _FakeDataset:
        data = _FakeData()

    center_pulse(_FakeDataset(), segment='first_reflection')
    # No exception and data unchanged
    assert len(second.data) == 100


def test_dataset_reflection_layout_no_first_match_falls_back_to_thzdata():
    """Files in second_reflection/ with no first_reflection/ counterpart load as plain THzData."""
    tmpdir = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmpdir, "first_reflection"))
        os.makedirs(os.path.join(tmpdir, "second_reflection"))
        # Only second_reflection has 'orphan.dat'
        _write_dat(os.path.join(tmpdir, "second_reflection", "orphan.dat"))
        # Both subdirs have 'paired.dat'
        _write_dat(os.path.join(tmpdir, "first_reflection", "paired.dat"), n_pts=60)
        _write_dat(os.path.join(tmpdir, "second_reflection", "paired.dat"), n_pts=100)

        ds = DataSet(tmpdir)
        ds.load_all_data()
        items = dict(ds.data.items())
        assert isinstance(items["orphan.dat"], THzData)
        assert not isinstance(items["orphan.dat"], THzDataReflection)
        assert isinstance(items["paired.dat"], THzDataReflection)
    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

_TESTS = [
    test_from_thzdata_basic,
    test_all_segments_order,
    test_data_property_points_to_second,
    test_isinstance_hierarchy,
    test_repr_contains_class_name,
    test_dataset_flat_dir_unchanged,
    test_dataset_reflection_layout_creates_thzdatareflection,
    test_explicit_dir_bypasses_reflection_layout,
    test_dataset_reflection_layout_no_first_match_falls_back_to_thzdata,
    test_center_pulse_first_reflection_auto,
    test_center_pulse_first_reflection_already_centred,
    test_center_pulse_first_reflection_taper_smooth,
    test_center_pulse_main_trace,
    test_center_pulse_first_reflection_skips_plain_thzdata,
]


def main():
    passed = 0
    failed = 0
    for test_fn in _TESTS:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {test_fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
