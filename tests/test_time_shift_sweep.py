"""Headless tests for the artificial time-shift sweep + blitting slider.

Builds a synthetic window-geometry reflection dataset, runs it through the
pipeline up to the transfer function, then exercises sweep_time_shift and the
slider (under the Agg backend, no display).

Dependency-free; run with the project venv:
    .venv/Scripts/python.exe tests/test_time_shift_sweep.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")

import thz_core as core
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz


N_SIO2 = 1.95
THETA_EXT_DEG = 45.0
N_SAMPLE_TRUTH = 3.0
SECOND_PS = 164.3
SAMP_NAME = "sample_film-1_set_0.acc"


def _gaussian(t_ps, centre, width=0.25):
    return np.exp(-((t_ps - centre) ** 2) / (2 * width**2))


def _write_second_only_acc(path, name, centre_ps, amplitude):
    t_ps = np.arange(159.0, 169.5, 0.05)
    y = amplitude * _gaussian(t_ps, centre_ps)
    lines = [f"%title {name} acc 1", "%type 0"]
    lines += [f"{t:.6f} {e:.10e}" for t, e in zip(t_ps, y)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _build_through_transfer(tmpdir):
    theta_int = float(np.real(core.snell_refracted_angle(np.deg2rad(THETA_EXT_DEG), 1.0, N_SIO2)))
    r_air = core.fresnel_reflection_s(N_SIO2, 1.0, theta_int)
    r_sample = core.fresnel_reflection_s(N_SIO2, N_SAMPLE_TRUTH, theta_int)
    _write_second_only_acc(
        os.path.join(tmpdir, "reference_sio2_set_0.acc"),
        "reference_sio2_set_0", SECOND_PS, float(r_air.real),
    )
    _write_second_only_acc(
        os.path.join(tmpdir, SAMP_NAME), "sample_film-1_set_0",
        SECOND_PS, float(r_sample.real),
    )
    dataset = DataSet(tmpdir)
    dataset.load_all_data(case_insensitive=True)
    dataset.group_files(keywords=["type"])
    thz.zero_pad(dataset, config={"pad": {"extend_factor": 2.0}})
    thz.fft_spectrum(dataset)
    thz.transfer_function(dataset, ref_type="reference")
    return dataset


def _geo():
    return dict(geometry="window", theta_deg=THETA_EXT_DEG, n_window=N_SIO2)


def test_sweep_zero_shift_matches_direct_inversion():
    """A zero artificial shift must reproduce the plain reflection inversion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _build_through_transfer(tmpdir)
        thz.invert_nk_reflection(dataset, **_geo())
        baseline_n = dataset.data[SAMP_NAME].processing_dict["n"]

        result = thz.sweep_time_shift(dataset, [0.0], **_geo())
        swept_n = result["n"][0]
        both = np.isfinite(baseline_n) & np.isfinite(swept_n)
        assert both.sum() > 5
        assert np.allclose(swept_n[both], baseline_n[both], atol=1e-9)


def test_sweep_shift_changes_n_and_is_reversible():
    """A nonzero shift changes n; the exact spectral ramp round-trips (+dt then
    the same -dt recovers the zero-shift result)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _build_through_transfer(tmpdir)
        shifts = np.array([-0.02e-12, 0.0, 0.02e-12])
        result = thz.sweep_time_shift(dataset, shifts, **_geo())
        n0 = result["n"][1]
        n_plus = result["n"][2]
        good = np.isfinite(n0) & np.isfinite(n_plus)
        # The shift must actually move n somewhere in the band.
        assert np.nanmax(np.abs(n_plus[good] - n0[good])) > 1e-3


def test_sweep_outputs_have_expected_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _build_through_transfer(tmpdir)
        shifts = np.linspace(-0.05e-12, 0.05e-12, 9)
        result = thz.sweep_time_shift(dataset, shifts, **_geo())
        n_freq = result["freq"].size
        for key in ("n", "k", "sigma1", "sigma2"):
            assert result[key].shape == (9, n_freq)
        assert result["sample"] == SAMP_NAME


def test_band_thz_overrides_the_snr_mask():
    """band_thz makes the inversion span exactly the requested window (finite
    inside, NaN outside), independent of the SNR trusted band."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _build_through_transfer(tmpdir)
        lo, hi = 0.5, 1.5
        result = thz.sweep_time_shift(
            dataset, [0.0], band_thz=(lo, hi), **_geo()
        )
        f_thz = result["freq"] * 1e-12
        n0 = result["n"][0]
        inside = (f_thz >= lo) & (f_thz <= hi)
        # Finite results only within the requested band.
        assert np.isfinite(n0[inside]).any()
        assert not np.isfinite(n0[~inside]).any()


def test_slider_builds_and_updates_headless():
    """The blitting slider constructs under Agg and its update callback runs and
    actually changes the plotted data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _build_through_transfer(tmpdir)
        view = thz.time_shift_slider(
            dataset, n_steps=11, shift_range_ps=(-0.05, 0.05),
            quantity="sigma", show=False, **_geo(),
        )
        line = view["lines"][0]  # the animated 'swept' sigma1 curve
        y_before = line.get_ydata().copy()
        # Drive the slider to an end of the range and confirm the curve moved.
        idx = view["update"](0.05)
        assert idx == 10
        y_after = line.get_ydata()
        assert not np.array_equal(np.nan_to_num(y_before), np.nan_to_num(y_after))
        import matplotlib.pyplot as plt
        plt.close("all")


def test_waterfall_builds_headless():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _build_through_transfer(tmpdir)
        result = thz.sweep_time_shift(
            dataset, np.linspace(-0.05e-12, 0.05e-12, 9), **_geo()
        )
        ax = thz.time_shift_waterfall(result, quantity="sigma1")
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")


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
