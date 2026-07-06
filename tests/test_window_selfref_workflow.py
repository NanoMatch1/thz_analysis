"""Headless workflow tests for front-pulse self-referencing and single-trace
window characterisation.

Synthetic two-pulse traces are built from the analytic window model
(thz_core.window_transfer_model): the second reflection is the first pulse
filtered by W, exactly as in the real window-coupled geometry. A known
frequency-structured drift filter G(f) is applied to the whole sample trace to
emulate mount-to-mount alignment drift; the test asserts that

    H_old = Y2_s / Y2_ref          is contaminated by G, while
    H_new = (Y2_s/Y1_s)/(Y2_r/Y1_r)  (self_reference=True) recovers H_true.

Dependency-free (no pytest); run with the project venv:
    .venv/Scripts/python.exe tests/test_window_selfref_workflow.py
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


N_WINDOW_TRUE = 1.96 - 0.006j
THICKNESS_M = 0.9e-3
THETA_EXT_DEG = 45.0
H_TRUE = -1.5  # flat sample response (sign-flipped, CNT-like magnitude)

DT_PS = 0.05
TIME_PS = np.arange(151.0, 169.5, DT_PS)
# Keep each pulse near the centre of its gate: the Hann taper then weighs the
# drift filter's time-domain echoes (+/- DRIFT_ECHO_PS around each pulse)
# almost identically in both segments, which is what the real gating aims for.
FRONT_CENTRE_PS = 154.8
PULSE_WIDTH_PS = 0.25
DRIFT_ECHO_PS = 0.3

GATES = {"first_reflection": (151.2, 158.6), "second_reflection": (162.0, 169.0)}
REF_NAME = "reference_sio2_set_0.acc"
SAMP_NAME = "sample_film-1_set_0.acc"

ASSESS_BAND_THZ = (0.4, 2.0)


def _full_trace_freq_hz():
    return np.fft.rfftfreq(TIME_PS.size, DT_PS * 1e-12)


def _front_pulse():
    return np.exp(-((TIME_PS - FRONT_CENTRE_PS) ** 2) / (2 * PULSE_WIDTH_PS**2))


def _apply_spectral_filter(trace, filter_complex):
    """Filter a real trace with a (Hermitian-implied) rfft-domain filter."""
    return np.fft.irfft(np.fft.rfft(trace) * filter_complex, n=trace.size)


def _two_pulse_trace(second_pulse_filter):
    """Front pulse + (front pulse filtered by W * extra) second reflection."""
    freq_hz = _full_trace_freq_hz()
    w_model = core.window_transfer_model(
        freq_hz, N_WINDOW_TRUE, THICKNESS_M, np.deg2rad(THETA_EXT_DEG)
    )
    front = _front_pulse()
    second = _apply_spectral_filter(front, w_model * second_pulse_filter)
    return front + second


def _write_acc(path, name, trace):
    lines = [f"%title {name} acc 1", "%type 0"]
    lines += [f"{t:.6f} {e:.10e}" for t, e in zip(TIME_PS, trace)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _drift_filter():
    """Real, zero-phase, frequency-structured drift (15% magnitude ripple)."""
    freq_hz = _full_trace_freq_hz()
    return 1.0 + 0.15 * np.cos(2 * np.pi * freq_hz * DRIFT_ECHO_PS * 1e-12)


def _delay_filter(delay_s):
    """Pure timing offset: a linear-phase spectral filter exp(-i*2*pi*f*delay)."""
    freq_hz = _full_trace_freq_hz()
    return np.exp(-2j * np.pi * freq_hz * delay_s)


def _build_and_segment(tmpdir, apply_drift, sample_filter=None):
    reference_trace = _two_pulse_trace(1.0)
    sample_trace = _two_pulse_trace(H_TRUE)
    if apply_drift:
        sample_trace = _apply_spectral_filter(sample_trace, _drift_filter())
    if sample_filter is not None:
        sample_trace = _apply_spectral_filter(sample_trace, sample_filter)

    _write_acc(os.path.join(tmpdir, REF_NAME), REF_NAME[:-4], reference_trace)
    _write_acc(os.path.join(tmpdir, SAMP_NAME), SAMP_NAME[:-4], sample_trace)

    raw_dataset = DataSet(tmpdir)
    raw_dataset.load_all_data(case_insensitive=True)
    segment_root = os.path.join(tmpdir, "segmented")
    thz.segment_reflections(raw_dataset, segments=GATES, output_dir=segment_root)
    return segment_root


def _run_phase2(segment_root, self_reference, self_phase=False):
    dataset = DataSet(os.path.join(segment_root, "second_reflection"))
    dataset.load_all_data(case_insensitive=True)
    dataset.group_files(keywords=["type"])
    thz.zero_pad(dataset, config={"pad": {"extend_factor": 2.0}})
    thz.fft_spectrum(dataset)
    thz.transfer_function(
        dataset,
        config={"transfer": {"self_reference": self_reference, "self_phase": self_phase}},
        ref_type="reference",
    )
    return dataset


def _band_h(dataset):
    processing = dataset.data[SAMP_NAME].processing_dict
    freq_thz = processing["fft_freq"] * 1e-12
    band = (
        (freq_thz >= ASSESS_BAND_THZ[0]) & (freq_thz <= ASSESS_BAND_THZ[1])
        & processing["transfer_mask"]
    )
    return processing["transfer_H"][band]


def test_self_reference_removes_injected_drift():
    with tempfile.TemporaryDirectory() as tmpdir:
        segment_root = _build_and_segment(tmpdir, apply_drift=True)

        h_old = _band_h(_run_phase2(segment_root, self_reference=False))
        h_new = _band_h(_run_phase2(segment_root, self_reference=True))

        error_old = np.sqrt(np.mean(np.abs(h_old / H_TRUE - 1.0) ** 2))
        error_new = np.sqrt(np.mean(np.abs(h_new / H_TRUE - 1.0) ** 2))
        # The 15% ripple drift must show up in H_old...
        assert error_old > 0.05, f"drift not visible in H_old (rms {error_old:.4f})"
        # ...and be removed by the front-pulse correction.
        assert error_new < 0.02, f"self-referenced H rms error {error_new:.4f}"
        assert error_new < error_old / 4


def test_self_phase_removes_timing_offset():
    """self_phase must cancel a planted timing delay (like self_reference does)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # A pure delay on the whole sample trace: shifts both pulses together, so the
        # front-pulse phase carries the same offset the back ratio does.
        segment_root = _build_and_segment(
            tmpdir, apply_drift=False, sample_filter=_delay_filter(0.10e-12))

        h_plain = _band_h(_run_phase2(segment_root, self_reference=False))
        h_phase = _band_h(_run_phase2(segment_root, self_reference=False, self_phase=True))

        error_plain = np.sqrt(np.mean(np.abs(h_plain / H_TRUE - 1.0) ** 2))
        error_phase = np.sqrt(np.mean(np.abs(h_phase / H_TRUE - 1.0) ** 2))
        # The delay corrupts the plain ratio (linear phase) ...
        assert error_plain > 0.1, f"timing offset not visible in plain H (rms {error_plain:.4f})"
        # ... and self_phase removes it, recovering the flat H_TRUE.
        assert error_phase < 0.02, f"self_phase H rms error {error_phase:.4f}"
        assert error_phase < error_plain / 4


def test_self_phase_keeps_amplitude_drift_that_selfref_removes():
    """self_phase is PHASE-only: a zero-phase amplitude drift stays (self_reference kills it)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        segment_root = _build_and_segment(tmpdir, apply_drift=True)  # 15% zero-phase ripple

        h_plain = _band_h(_run_phase2(segment_root, self_reference=False))
        h_phase = _band_h(_run_phase2(segment_root, self_reference=False, self_phase=True))
        h_selfref = _band_h(_run_phase2(segment_root, self_reference=True))

        err = lambda h: np.sqrt(np.mean(np.abs(h / H_TRUE - 1.0) ** 2))
        # self_reference removes the amplitude drift; self_phase (phase-only) does not,
        # so it stays as corrupted as the plain ratio on this zero-phase drift.
        assert err(h_selfref) < 0.02, f"self_reference should remove drift (rms {err(h_selfref):.4f})"
        assert err(h_phase) > 0.05, f"self_phase should NOT remove amplitude drift (rms {err(h_phase):.4f})"
        assert abs(err(h_phase) - err(h_plain)) < 0.02


def test_self_reference_is_noop_without_drift():
    """With no drift the correction C is ~1 and both paths agree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        segment_root = _build_and_segment(tmpdir, apply_drift=False)

        h_old = _band_h(_run_phase2(segment_root, self_reference=False))
        h_new = _band_h(_run_phase2(segment_root, self_reference=True))
        assert np.sqrt(np.mean(np.abs(h_new - h_old) ** 2)) < 0.02 * abs(H_TRUE)


def test_selfref_correction_is_stored_for_diagnostics():
    with tempfile.TemporaryDirectory() as tmpdir:
        segment_root = _build_and_segment(tmpdir, apply_drift=True)
        dataset = _run_phase2(segment_root, self_reference=True)
        correction = dataset.data[SAMP_NAME].processing_dict.get("selfref_correction")
        assert correction is not None
        assert np.iscomplexobj(correction)


def test_missing_first_reflection_folder_raises():
    """Self-referencing must fail loudly, not silently skip the correction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        segment_root = _build_and_segment(tmpdir, apply_drift=False)
        # Remove the sibling folder the correction depends on.
        first_dir = os.path.join(segment_root, "first_reflection")
        for name in os.listdir(first_dir):
            os.remove(os.path.join(first_dir, name))
        os.rmdir(first_dir)

        try:
            _run_phase2(segment_root, self_reference=True)
        except FileNotFoundError as error:
            assert "first" in str(error).lower()
        else:
            raise AssertionError("expected FileNotFoundError for missing folder")


def test_characterise_window_recovers_known_index():
    """The single-trace characterisation recovers the index used to build W."""
    with tempfile.TemporaryDirectory() as tmpdir:
        segment_root = _build_and_segment(tmpdir, apply_drift=False)
        result = thz.characterise_window(
            os.path.join(segment_root, "first_reflection", REF_NAME),
            os.path.join(segment_root, "second_reflection", REF_NAME),
            thickness_m=THICKNESS_M,
            theta_deg=THETA_EXT_DEG,
            band_thz=ASSESS_BAND_THZ,
        )
        mask = result["mask"]
        n_median = float(np.nanmedian(result["n"][mask]))
        k_median = float(np.nanmedian(result["k"][mask]))
        assert abs(n_median - N_WINDOW_TRUE.real) < 0.01, f"n {n_median:.4f}"
        assert abs(k_median - (-N_WINDOW_TRUE.imag)) < 0.01, f"k {k_median:.4f}"

        # The measured envelope delay must match the model group delay.
        beta = np.sqrt(N_WINDOW_TRUE.real**2 - np.sin(np.deg2rad(THETA_EXT_DEG)) ** 2)
        expected_delay_s = 2.0 * THICKNESS_M * beta / 299_792_458.0
        assert abs(result["delay_s"] - expected_delay_s) < 0.05e-12


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
