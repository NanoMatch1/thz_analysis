"""Regression tests: segment_reflections must preserve per-scan statistical power.

The preprocessing pipeline (baseline -> align -> normalise) transforms the
*averaged* working trace ``data_obj.data`` ([time_s, mean, stderr]). A parallel
per-scan working matrix is carried through those steps so that segmentation can
write every individual acquisition back out (not just the mean). These tests
assert that:

  1. every input scan survives into the segmented .acc files, and
  2. the averaged segmented trace is bit-for-bit the averaged pipeline result
     (baseline/normalise are linear and alignment is a shared x-shift, so the
     mean of the per-scan-processed traces equals the processed average).

Dependency-free; run with the project venv:
    .venv/Scripts/python.exe tests/test_segment_preserves_scans.py
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
T0_PS = 150.0
FIRST_PS = 154.0   # front (alignment) reflection
SECOND_PS = 167.0  # back (sample) reflection


def _pulse(t_ps, centre, amp, width=0.3):
    """Single-cycle-ish pulse: derivative of a Gaussian."""
    x = (t_ps - centre) / width
    return amp * -x * np.exp(-(x**2) / 2.0)


def _write_multiscan_acc(path, name, n_scans, *, second_amp, rng, time_jitter=0.0):
    """Write an .acc with n_scans noisy repeats sharing one time grid."""
    t_ps = np.arange(T0_PS, T0_PS + 20.0, DT_PS)
    columns = [t_ps]
    scan_headers = []
    for scan_index in range(n_scans):
        clean = _pulse(t_ps, FIRST_PS, 1.0) + _pulse(t_ps, SECOND_PS, second_amp)
        noisy = clean + rng.normal(0.0, 0.01, size=t_ps.size) + 0.002  # + tiny DC
        columns.append(noisy)
        # Header lines WITHOUT a leading '%' — save_acc adds the percent prefix.
        scan_headers.append([f"title {name} acc {scan_index + 1}", "type 0"])
    data = np.column_stack(columns)
    save_acc({"data": data, "scan_headers": scan_headers, "header": scan_headers[0]}, path)


def _build_preprocessed(tmpdir, rng):
    _write_multiscan_acc(
        os.path.join(tmpdir, "reference_x.acc"), "reference_x", 4,
        second_amp=-0.4, rng=rng,
    )
    _write_multiscan_acc(
        os.path.join(tmpdir, "sample_x.acc"), "sample_x", 6,
        second_amp=-0.25, rng=rng,
    )
    dataset = DataSet(tmpdir)
    dataset.load_all_data(case_insensitive=True)
    dataset.group_files(keywords=["type"])
    thz.subtract_baseline(dataset)
    thz.align_to_reference(
        dataset, ref_type="reference", roi=(FIRST_PS - 2, FIRST_PS + 2),
        subsample_correction=True, show_graph=False,
    )
    thz.normalise(dataset, config={"bounds": (FIRST_PS - 2, FIRST_PS + 2)}, show_graph=False)
    return dataset


def test_every_scan_is_preserved_in_segmented_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(0)
        dataset = _build_preprocessed(tmpdir, rng)
        out = os.path.join(tmpdir, "segmented")
        thz.segment_reflections(
            dataset, segments={"second_reflection": (SECOND_PS - 3, SECOND_PS + 3)},
            components=("second_reflection",), output_dir=out, show_graph=False,
        )
        reloaded = DataSet(os.path.join(out, "second_reflection"))
        reloaded.load_all_data(case_insensitive=True)
        scan_counts = {
            fn: np.asarray(obj.raw_data).shape[1] - 1
            for fn, obj in reloaded.data.items()
        }
        # The originals had reference=4 scans, sample=6 scans; both must survive.
        assert scan_counts["reference_x.acc"] == 4, scan_counts
        assert scan_counts["sample_x.acc"] == 6, scan_counts


def test_segmented_mean_matches_averaged_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(1)
        dataset = _build_preprocessed(tmpdir, rng)

        # Averaged-pipeline second-reflection mean (crop the averaged trace directly).
        sample = dataset.data["sample_x.acc"]
        t_ps = sample.data[:, 0] * thz._S_TO_PS
        gate = (t_ps >= SECOND_PS - 3) & (t_ps <= SECOND_PS + 3)
        averaged_pipeline_mean = sample.data[gate, 1].copy()

        out = os.path.join(tmpdir, "segmented")
        thz.segment_reflections(
            dataset, segments={"second_reflection": (SECOND_PS - 3, SECOND_PS + 3)},
            components=("second_reflection",), output_dir=out, show_graph=False,
        )
        reloaded = DataSet(os.path.join(out, "second_reflection"))
        reloaded.load_all_data(case_insensitive=True)
        reloaded_mean = reloaded.data["sample_x.acc"].data[:, 1]

        n = min(averaged_pipeline_mean.size, reloaded_mean.size)
        assert n > 10
        assert np.allclose(averaged_pipeline_mean[:n], reloaded_mean[:n], atol=1e-12)


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
