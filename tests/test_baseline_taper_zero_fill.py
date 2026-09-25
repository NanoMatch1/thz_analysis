"""Regression tests: baseline subtraction and edge tapering respect zero-fill.

build_full_trace_reflection zeros the gap between the two pulses. The bug these
tests pin down: subtract_baseline removed the DC offset from the WHOLE shared-axis
trace, so the zero-filled gap was pushed to -offset while the measured data went to
~0. That left a step at every pulse-region edge, which the Hann window then drew as
a curve from 0 to -offset (most visible on the early side of the second reflection).
taper_and_pad_traces also only tapered the padded array end, never the edges that
border the gap.

Fixed behaviour:
  * baseline is estimated from, and subtracted from, the MEASURED samples only —
    the gap stays at exactly 0 (averaged trace and per-scan matrix alike);
  * every block of measured data is half-cosine tapered to zero at both edges.

Dependency-free; run with the project venv:
    .venv/bin/python tests/test_baseline_taper_zero_fill.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acquisition_editor import save_acc
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz


DT_PS = 0.05
FIRST_PS = 155.0
SECOND_PS = 180.0
DC_OFFSET = 0.3     # detector offset present on every measured sample
TAPER_PS = 0.5
REGIONS = {"first_reflection": (150.0, 160.0), "second_reflection": (175.0, 185.0)}


def _pulse(time_ps, centre_ps, amplitude, width_ps=0.3):
    scaled_time = (time_ps - centre_ps) / width_ps
    return amplitude * -scaled_time * np.exp(-(scaled_time ** 2) / 2.0)


def _write_offset_acc(path, name, n_scans, rng):
    time_ps = np.arange(150.0, 185.0, DT_PS)
    columns = [time_ps]
    scan_headers = []
    for scan_index in range(n_scans):
        clean = _pulse(time_ps, FIRST_PS, 1.0) + _pulse(time_ps, SECOND_PS, -0.4)
        columns.append(clean + DC_OFFSET + rng.normal(0.0, 0.005, size=time_ps.size))
        scan_headers.append([f"title {name} acc {scan_index + 1}", "type 0"])
    save_acc({"data": np.column_stack(columns), "scan_headers": scan_headers,
              "header": scan_headers[0]}, path)


def _built_dataset(tmpdir):
    rng = np.random.default_rng(1)
    _write_offset_acc(os.path.join(tmpdir, "reference_x.acc"), "reference_x", 3, rng)
    _write_offset_acc(os.path.join(tmpdir, "sample_x.acc"), "sample_x", 4, rng)
    config = {"regions": dict(REGIONS), "centering": {"peak_mode": "auto", "taper_ps": TAPER_PS}}
    dataset = DataSet(tmpdir, config=config)
    dataset.load_all_data(case_insensitive=True)
    thz.build_full_trace_reflection(dataset, config)
    thz.define_reflection_regions(dataset, config)
    return dataset


def _gap_mask(time_ps):
    return (time_ps > REGIONS["first_reflection"][1] + 0.5) & (time_ps < REGIONS["second_reflection"][0] - 0.5)


# ---------------------------------------------------------------- pure helpers

def test_measured_support_mask_marks_zero_runs_only():
    amplitude = np.array([0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
    support = thz.measured_support_mask(amplitude)
    # The leading pair and the triple are zero-fill; the lone interior zero is a
    # genuine zero-crossing and stays measured.
    assert support.tolist() == [False, False, True, True, True, False, False, False, True]


def test_taper_reaches_zero_at_every_block_edge():
    amplitude = np.concatenate([np.ones(20), np.zeros(10), np.ones(20)])
    tapered, blocks = thz.taper_measured_block_edges(
        amplitude, thz.measured_support_mask(amplitude), taper_samples=5)
    assert blocks == [(0, 20), (30, 50)]
    for block_start, block_stop in blocks:
        assert tapered[block_start] == 0.0
        assert abs(tapered[block_stop - 1]) < 0.1
        assert tapered[block_start + 5] == 1.0     # untouched interior
    assert np.all(tapered[20:30] == 0.0)            # zero-fill untouched


def test_taper_clamps_to_half_block():
    amplitude = np.ones(6)
    tapered, _ = thz.taper_measured_block_edges(amplitude, np.ones(6, dtype=bool), taper_samples=50)
    assert np.all(np.isfinite(tapered)) and tapered[0] == 0.0


# ----------------------------------------------------------- pipeline workflow

def test_baseline_keeps_gap_at_zero_and_removes_offset():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _built_dataset(tmpdir)
        thz.subtract_baseline(dataset)
        for filename, data_obj in dataset.data.items():
            for holder in (data_obj.first_reflection, data_obj.second_reflection):
                time_ps = holder.data[:, 0] * thz._S_TO_PS
                amplitude = holder.data[:, 1]
                gap = _gap_mask(time_ps)
                assert np.all(amplitude[gap] == 0.0), f"{filename}: gap shifted by baseline"
                # Measured data just inside the second region's leading edge sits
                # near zero, i.e. continuous with the gap (was DC_OFFSET-sized step).
                leading = (time_ps >= REGIONS["second_reflection"][0]) & (time_ps < REGIONS["second_reflection"][0] + 1.0)
                assert np.max(np.abs(amplitude[leading])) < 0.05, filename

            scan_matrix = thz._ensure_scan_matrix(data_obj)
            scan_gap = _gap_mask(scan_matrix[:, 0] * thz._S_TO_PS)
            assert np.all(scan_matrix[scan_gap, 1:] == 0.0), f"{filename}: scan gap shifted"


def test_taper_smooths_second_reflection_leading_edge():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _built_dataset(tmpdir)
        thz.subtract_baseline(dataset)
        thz.taper_and_pad_traces(dataset)
        for filename, data_obj in dataset.data.items():
            holder = data_obj.second_reflection
            amplitude = holder.data[:, 1]
            blocks = holder.processing_dict['centering_info']['measured_blocks']
            assert len(blocks) == 2, blocks
            for block_start, block_stop in blocks:
                assert amplitude[block_start] == 0.0, filename
                assert amplitude[block_stop - 1] == 0.0, filename
            # Largest sample-to-sample jump on the leading edge of the second pulse
            # is small compared with the removed offset (no step into the gap).
            second_start = blocks[1][0]
            edge_steps = np.abs(np.diff(amplitude[second_start - 3:second_start + 3]))
            assert np.max(edge_steps) < 0.1 * DC_OFFSET, (filename, edge_steps)


def _run_all():
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    failures = 0
    for test_function in tests:
        try:
            test_function()
            print(f"PASS  {test_function.__name__}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"FAIL  {test_function.__name__}: {error}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
