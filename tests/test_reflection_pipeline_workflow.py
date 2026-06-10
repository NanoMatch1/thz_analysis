"""Headless full-workflow test for the window-geometry reflection pipeline.

Builds synthetic .acc files (a SiO2-only reference and a sample with a known
lossless index pressed on the window), runs the whole adapter chain, and asserts
the pipeline recovers the known sample index. The first and second reflections
are placed at *identical* timing across files so there is no phase ramp — this
isolates the geometry/reference/inversion plumbing from the (separately handled)
timing-correction step.

Dependency-free (no pytest); run with the project venv:
    .venv/Scripts/python.exe tests/test_reflection_pipeline_workflow.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")

import thz_core.thz_core as core
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz


N_SIO2 = 1.95
THETA_EXT_DEG = 45.0
N_SAMPLE_TRUTH = 3.0  # lossless -> real reflection, clean recovery
FRONT_PS = 153.4
SECOND_PS = 164.3


def _gaussian(t_ps, centre, width=0.25):
    return np.exp(-((t_ps - centre) ** 2) / (2 * width**2))


def _write_acc(path, name, second_amplitude):
    """Write a single-scan .acc with a first reflection + a scaled second
    reflection (same timing)."""
    t_ps = np.arange(151.0, 169.0, 0.05)
    first = 1.0 * _gaussian(t_ps, FRONT_PS)
    second = second_amplitude * _gaussian(t_ps, SECOND_PS)
    y = first + second
    lines = [f"%title {name} acc 1", "%type 0"]
    lines += [f"{t:.6f} {e:.10e}" for t, e in zip(t_ps, y)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _build_dataset(tmpdir):
    theta_int = float(np.real(core.snell_refracted_angle(np.deg2rad(THETA_EXT_DEG), 1.0, N_SIO2)))
    r_air = core.fresnel_reflection_s(N_SIO2, 1.0, theta_int)            # ~ +0.44
    r_sample = core.fresnel_reflection_s(N_SIO2, N_SAMPLE_TRUTH, theta_int)  # negative (n>n_sio2)

    # Reference second reflection carries r_{SiO2->air}; sample second reflection
    # carries r_{SiO2->sample}. H = sample/ref, then r_recovered = r_air * H.
    _write_acc(os.path.join(tmpdir, "reference_sio2_set_0.acc"), "reference_sio2_set_0", float(r_air.real))
    _write_acc(os.path.join(tmpdir, "sample_film-1_set_0.acc"), "sample_film-1_set_0", float(r_sample.real))
    return r_air, r_sample, theta_int


GATES = {"first_reflection": (151.2, 158.5), "second_reflection": (159.5, 168.8)}
SAMP_NAME = "sample_film-1_set_0.acc"


def _segment_then_process(tmpdir):
    """Phase 1: crop second reflections to .acc. Phase 2: load + process normally."""
    _build_dataset(tmpdir)

    seg_root = os.path.join(tmpdir, "segmented")
    raw_ds = DataSet(tmpdir)
    raw_ds.load_all_data(case_insensitive=True)
    written = thz.segment_reflections(raw_ds, segments=GATES, output_dir=seg_root)
    # Both components written for both files (4 files), original names preserved.
    assert len(written) == 4
    assert os.path.exists(os.path.join(seg_root, "second_reflection", SAMP_NAME))
    assert os.path.exists(os.path.join(seg_root, "first_reflection", SAMP_NAME))

    dataset = DataSet(os.path.join(seg_root, "second_reflection"))
    dataset.load_all_data(case_insensitive=True)
    dataset.group_files(keywords=["type"])
    assert dataset.get_reference(SAMP_NAME, ref_type="reference") is not None

    thz.zero_pad(dataset, config={"pad": {"extend_factor": 2.0}})
    thz.fft_spectrum(dataset)
    thz.transfer_function(dataset, ref_type="reference")
    thz.invert_nk_reflection(
        dataset, geometry="window", theta_deg=THETA_EXT_DEG, n_window=N_SIO2,
    )
    return dataset


def test_window_reflection_pipeline_recovers_known_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _segment_then_process(tmpdir)
        pd_ = dataset.data[SAMP_NAME].processing_dict

        # Internal angle correctly computed by Snell.
        assert abs(np.rad2deg(pd_["theta_internal_rad"]) - 21.26) < 0.2

        freq = pd_["fft_freq"]
        n = pd_["n"]
        k = pd_["k"]
        f_thz = freq * 1e-12
        band = (f_thz > 0.5) & (f_thz < 3.0) & np.isfinite(n)
        assert band.sum() > 10, "too few finite bins to assess recovery"

        n_mean = float(np.nanmean(n[band]))
        k_mean = float(np.nanmean(k[band]))
        # Lossless sample with no timing offset -> clean recovery of n=3, k~0.
        assert abs(n_mean - N_SAMPLE_TRUTH) < 0.05, f"n recovered {n_mean:.3f}, expected 3.0"
        assert k_mean < 0.05, f"k recovered {k_mean:.3f}, expected ~0"


def test_recovered_reflection_has_pi_phase_for_high_index_sample():
    """A sample of higher index than SiO2 must give a sign-flipped (negative) r."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = _segment_then_process(tmpdir)
        pd_ = dataset.data[SAMP_NAME].processing_dict
        r = pd_["reflection_r"]
        mask = pd_["transfer_mask"]
        r_band = r[mask & np.isfinite(r)]
        assert np.mean(r_band.real) < 0, "high-index sample should give negative (pi-flipped) r"


def _write_second_only_acc(path, name, centre_ps, amplitude):
    """Write a single-scan .acc holding only one (second-reflection) pulse."""
    t_ps = np.arange(159.0, 169.5, 0.05)
    y = amplitude * _gaussian(t_ps, centre_ps)
    lines = [f"%title {name} acc 1", "%type 0"]
    lines += [f"{t:.6f} {e:.10e}" for t, e in zip(t_ps, y)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def test_align_to_reference_removes_injected_offset():
    """A sample second reflection offset by +0.15 ps is pulled back to T0,
    and the inversion then recovers the known index."""
    offset_ps = 0.15
    with tempfile.TemporaryDirectory() as tmpdir:
        theta_int = float(np.real(core.snell_refracted_angle(np.deg2rad(THETA_EXT_DEG), 1.0, N_SIO2)))
        r_air = core.fresnel_reflection_s(N_SIO2, 1.0, theta_int)
        r_sample = core.fresnel_reflection_s(N_SIO2, N_SAMPLE_TRUTH, theta_int)
        _write_second_only_acc(
            os.path.join(tmpdir, "reference_sio2_set_0.acc"),
            "reference_sio2_set_0", SECOND_PS, float(r_air.real),
        )
        _write_second_only_acc(
            os.path.join(tmpdir, SAMP_NAME),
            "sample_film-1_set_0", SECOND_PS + offset_ps, float(r_sample.real),
        )

        dataset = DataSet(tmpdir)
        dataset.load_all_data(case_insensitive=True)
        dataset.group_files(keywords=["type"])

        thz.align_to_reference(dataset, ref_type="reference")
        applied_ps = dataset.data[SAMP_NAME].processing_dict["align_metrics"]["values"][
            "applied_shift_seconds"
        ] * 1e12
        # The sample sat 0.15 ps late, so it must be shifted ~0.15 ps earlier.
        assert abs(abs(applied_ps) - offset_ps) < 0.02, f"applied {applied_ps:.3f} ps"

        thz.zero_pad(dataset, config={"pad": {"extend_factor": 2.0}})
        thz.fft_spectrum(dataset)
        thz.transfer_function(dataset, ref_type="reference")
        thz.invert_nk_reflection(
            dataset, geometry="window", theta_deg=THETA_EXT_DEG, n_window=N_SIO2,
        )
        pd_ = dataset.data[SAMP_NAME].processing_dict
        f_thz = pd_["fft_freq"] * 1e-12
        n = pd_["n"]
        band = (f_thz > 0.5) & (f_thz < 3.0) & np.isfinite(n)
        # Correct sign of the shift -> clean recovery; a wrong sign would double
        # the offset and badly corrupt n.
        assert abs(float(np.nanmean(n[band])) - N_SAMPLE_TRUTH) < 0.1, (
            f"n after alignment {np.nanmean(n[band]):.3f}, expected ~3.0"
        )


def _recover_n_with_subsample_offset(tmpdir, offset_ps, subsample_correction):
    """Build a sample whose second reflection is offset by a (sub-sample) amount,
    run align + invert with the correction on/off, return mean n in band."""
    theta_int = float(np.real(core.snell_refracted_angle(np.deg2rad(THETA_EXT_DEG), 1.0, N_SIO2)))
    r_air = core.fresnel_reflection_s(N_SIO2, 1.0, theta_int)
    r_sample = core.fresnel_reflection_s(N_SIO2, N_SAMPLE_TRUTH, theta_int)
    _write_second_only_acc(
        os.path.join(tmpdir, "reference_sio2_set_0.acc"),
        "reference_sio2_set_0", SECOND_PS, float(r_air.real),
    )
    _write_second_only_acc(
        os.path.join(tmpdir, SAMP_NAME),
        "sample_film-1_set_0", SECOND_PS + offset_ps, float(r_sample.real),
    )

    dataset = DataSet(tmpdir)
    dataset.load_all_data(case_insensitive=True)
    dataset.group_files(keywords=["type"])
    thz.align_to_reference(dataset, ref_type="reference", subsample_correction=subsample_correction)
    thz.zero_pad(dataset, config={"pad": {"extend_factor": 2.0}})
    thz.fft_spectrum(dataset)
    thz.transfer_function(dataset, ref_type="reference")
    thz.invert_nk_reflection(
        dataset, geometry="window", theta_deg=THETA_EXT_DEG, n_window=N_SIO2,
    )
    pd_ = dataset.data[SAMP_NAME].processing_dict
    f_thz = pd_["fft_freq"] * 1e-12
    n = pd_["n"]
    band = (f_thz > 0.5) & (f_thz < 3.0) & np.isfinite(n)
    return float(np.nanmean(n[band]))


def test_subsample_correction_beats_integer_only_for_half_sample_offset():
    """A half-sample (0.025 ps at dt=0.05) offset is pure sub-sample: integer-only
    alignment can't remove it, the spectral phase ramp can. The correction must
    recover n=3 cleanly and beat the uncorrected case."""
    offset_ps = 0.025  # exactly half of the 0.05 ps sampling step
    with tempfile.TemporaryDirectory() as tmp_on:
        n_corrected = _recover_n_with_subsample_offset(tmp_on, offset_ps, True)
    with tempfile.TemporaryDirectory() as tmp_off:
        n_uncorrected = _recover_n_with_subsample_offset(tmp_off, offset_ps, False)

    error_corrected = abs(n_corrected - N_SAMPLE_TRUTH)
    error_uncorrected = abs(n_uncorrected - N_SAMPLE_TRUTH)
    assert error_corrected < 0.05, f"corrected n={n_corrected:.4f}, expected ~3.0"
    assert error_corrected < error_uncorrected, (
        f"sub-sample correction ({error_corrected:.4f}) should beat integer-only "
        f"({error_uncorrected:.4f})"
    )


def test_subsample_correction_is_a_clean_noop_when_disabled_via_toggle():
    """With correction off, the residual is recorded but applied_residual is 0 and
    no phase ramp is applied — the H result equals the integer-only pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _recover_n_with_subsample_offset(tmpdir, 0.025, False)
        # Re-run to inspect the stored metadata directly.
        dataset = DataSet(tmpdir)
        dataset.load_all_data(case_insensitive=True)
        dataset.group_files(keywords=["type"])
        thz.align_to_reference(dataset, ref_type="reference", subsample_correction=False)
        pd_ = dataset.data[SAMP_NAME].processing_dict
        assert pd_["subsample_correction_enabled"] is False
        assert pd_["subsample_shift_seconds"] == 0.0
        # The measured residual is still recorded for transparency/diagnostics.
        assert abs(pd_["subsample_shift_measured_seconds"]) > 0.0


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
