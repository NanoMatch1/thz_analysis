"""The per-scan matrix must survive the pipeline, and stay honest while it does.

Phase 0 of the noise work. Every repeat-based estimate (noise amplitudes, drift,
error bars) reads ``processing_dict['working_scans']``. Before this, the first
pipeline step changed the row count and the matrix was silently replaced by the
averaged trace as a single "scan" — so a noise estimator downstream would have
measured the scatter of one trace, which is zero, and said nothing about it.

Two invariants are pinned here:

  1. **Survival** — the individual scans are still there after the steps that
     previously destroyed them.
  2. **Consistency** — the mean of the scan columns equals the averaged trace at
     every stage. This is what makes carrying them free: every step applied to the
     scans is linear, so the averaged pipeline is bit-for-bit unchanged.

Plus the third, which is the actual comprehension-debt fix: when the scatter *is*
lost, it is reported rather than silently swallowed.
"""
from __future__ import annotations

import numpy as np
import pytest

from dataset_core.adapters import thz_adapter as thz
from dataset_core.data_structures.thz import BaseTHzData, THzData


DT_PS = 0.05
N_SAMPLES = 120
N_SCANS = 12


def make_thz_data(filename: str = "sample_test.acc", n_scans: int = N_SCANS,
                  peak_fraction: float = 0.3, seed: int = 0) -> THzData:
    """A THzData carrying n_scans noisy repeats of an off-centre pulse.

    The peak sits off-centre on purpose: that is what makes the padding step change
    the row count, which is the condition the matrix used to be destroyed by.
    """
    rng = np.random.default_rng(seed)
    time_ps = np.arange(N_SAMPLES) * DT_PS + 100.0
    centre = time_ps[0] + peak_fraction * (time_ps[-1] - time_ps[0])
    argument = (time_ps - centre) / 0.2
    pulse = -argument * np.exp(-0.5 * argument ** 2)

    scans = []
    for index in range(n_scans):
        amplitude = pulse * (1.0 + 0.01 * index) + rng.standard_normal(N_SAMPLES) * 1e-3
        scans.append(BaseTHzData(data=np.column_stack((time_ps, amplitude)),
                                 headers=[f"scan {index}"]))
    return THzData(data=scans, header=None, filename=filename, data_type="acc")


class FakeDataSet:
    """Minimal stand-in: the steps under test only touch ``dataset.data`` and config."""

    def __init__(self, objects: dict, config: dict | None = None):
        self.data = objects
        self.config = config or {}


def scan_mean(data_obj) -> np.ndarray:
    matrix = np.asarray(data_obj.processing_dict[thz._SCAN_MATRIX_KEY], dtype=float)
    return matrix[:, 1:].mean(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# plan / apply split
# ─────────────────────────────────────────────────────────────────────────────


def test_plan_then_apply_reproduces_the_combined_call():
    data_obj = make_thz_data()
    time_seconds = np.asarray(data_obj.data)[:, 0]
    amplitude = np.asarray(data_obj.data)[:, 1]

    combined_time, combined_amplitude, info = thz._center_pulse_trace(
        time_seconds, amplitude, peak_mode="auto", taper_ps=0.5)
    plan = thz._plan_center_pad(time_seconds, amplitude, peak_mode="auto", taper_ps=0.5)
    split_time, split_amplitude = thz._apply_center_pad(time_seconds, amplitude, plan)

    assert np.array_equal(combined_time, split_time)
    assert np.array_equal(combined_amplitude, split_amplitude)
    assert info == plan


def test_apply_center_pad_is_linear_in_the_amplitude():
    """The property that makes carrying the scans free."""
    data_obj = make_thz_data()
    time_seconds = np.asarray(data_obj.data)[:, 0]
    first = np.asarray(data_obj.data)[:, 1]
    second = np.roll(first, 7) * 0.5
    plan = thz._plan_center_pad(time_seconds, first)

    _, padded_first = thz._apply_center_pad(time_seconds, first, plan)
    _, padded_second = thz._apply_center_pad(time_seconds, second, plan)
    _, padded_sum = thz._apply_center_pad(time_seconds, first + second, plan)

    assert np.allclose(padded_first + padded_second, padded_sum, atol=1e-15)


def test_the_plan_fully_determines_the_taper_length():
    """Two callers must not be able to resolve the taper differently."""
    data_obj = make_thz_data()
    time_seconds = np.asarray(data_obj.data)[:, 0]
    amplitude = np.asarray(data_obj.data)[:, 1]
    plan = thz._plan_center_pad(time_seconds, amplitude, taper_ps=0.5)
    assert plan["taper_samples"] > 0
    assert isinstance(plan["taper_samples"], int)


# ─────────────────────────────────────────────────────────────────────────────
# survival through the pipeline
# ─────────────────────────────────────────────────────────────────────────────


def test_padding_changes_the_row_count_and_the_scans_survive_it():
    """The regression this whole phase exists for."""
    data_obj = make_thz_data()
    dataset = FakeDataSet({data_obj.filename: data_obj})
    rows_before = np.asarray(data_obj.data).shape[0]

    thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)

    rows_after = np.asarray(data_obj.data).shape[0]
    assert rows_after > rows_before, "test needs a row-count-changing pad"

    status = thz.scan_matrix_status(data_obj)
    assert status["intact"], f"scans were dropped: {status['reason']}"
    assert status["n_scans"] == N_SCANS


def test_scan_mean_still_equals_the_averaged_trace_after_padding():
    data_obj = make_thz_data()
    dataset = FakeDataSet({data_obj.filename: data_obj})
    thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)
    assert np.allclose(scan_mean(data_obj), np.asarray(data_obj.data)[:, 1], atol=1e-12)


def test_scans_survive_padding_then_baseline_then_windowing():
    """End to end over the single-reflection pipeline's preprocessing chain."""
    data_obj = make_thz_data()
    dataset = FakeDataSet(
        {data_obj.filename: data_obj},
        config={"window": {"type": "hann", "alpha": 1.0}, "baseline": {"n_points": 10}},
    )

    thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)
    thz.subtract_baseline(dataset)
    thz.window_single_pulse_fixed_width(dataset, half_width_ps=2.0, show_graph=False)

    status = thz.scan_matrix_status(data_obj)
    assert status["intact"]
    assert status["n_scans"] == N_SCANS
    assert np.allclose(scan_mean(data_obj), np.asarray(data_obj.data)[:, 1], atol=1e-12)


def test_the_window_applied_to_scans_is_the_one_applied_to_the_average():
    data_obj = make_thz_data()
    dataset = FakeDataSet({data_obj.filename: data_obj},
                          config={"window": {"type": "hann", "alpha": 1.0}})
    thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)
    pre_window = np.asarray(data_obj.processing_dict[thz._SCAN_MATRIX_KEY],
                            dtype=float)[:, 1:].copy()
    thz.window_single_pulse_fixed_width(dataset, half_width_ps=2.0, show_graph=False)

    window_function = data_obj.processing_dict["window_function"]
    post_window = np.asarray(data_obj.processing_dict[thz._SCAN_MATRIX_KEY],
                             dtype=float)[:, 1:]
    assert np.allclose(post_window, pre_window * window_function[:, None], atol=1e-15)


def test_scans_reaching_the_noise_estimator_recover_the_planted_drift():
    """The point of the whole exercise: the estimator sees real repeats at the end."""
    from thz_core.thz_core.noise import drift_corrected_scatter

    data_obj = make_thz_data(n_scans=24)
    dataset = FakeDataSet({data_obj.filename: data_obj},
                          config={"window": {"type": "hann", "alpha": 1.0}})
    thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)

    matrix = np.asarray(data_obj.processing_dict[thz._SCAN_MATRIX_KEY], dtype=float)
    waveforms = matrix[:, 1:].T
    dt = float(np.median(np.diff(matrix[:, 0])))

    estimate = drift_corrected_scatter(waveforms, dt)
    # make_thz_data plants a 1%-per-scan amplitude ramp: 23% across 24 scans.
    assert np.ptp(estimate.amplitudes) == pytest.approx(0.23, rel=0.25)
    assert estimate.drift_inflation > 1.5


# ─────────────────────────────────────────────────────────────────────────────
# the loss is reported, not swallowed
# ─────────────────────────────────────────────────────────────────────────────


def test_a_row_changing_step_that_does_not_carry_reports_the_loss(capsys):
    data_obj = make_thz_data()
    thz._ensure_scan_matrix(data_obj)                      # build it in sync
    trace = np.asarray(data_obj.data)
    data_obj.data = trace[:-10]                            # a step that drops rows
    thz._ensure_scan_matrix(data_obj)                      # next consumer notices

    status = thz.scan_matrix_status(data_obj)
    assert not status["intact"]
    assert "row count changed" in status["reason"]
    assert "DROPPED" in capsys.readouterr().out


def test_loss_is_reported_once_not_on_every_call(capsys):
    data_obj = make_thz_data()
    thz._ensure_scan_matrix(data_obj)
    data_obj.data = np.asarray(data_obj.data)[:-10]
    for _ in range(4):
        thz._ensure_scan_matrix(data_obj)
    assert capsys.readouterr().out.count("[scan_matrix]") == 1


def test_dataset_level_report_lists_every_affected_file(capsys):
    intact = make_thz_data("sample_intact.acc")
    damaged = make_thz_data("sample_damaged.acc", seed=1)
    dataset = FakeDataSet({intact.filename: intact, damaged.filename: damaged})
    thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)

    damaged.data = np.asarray(damaged.data)[:-5]
    thz._ensure_scan_matrix(damaged)
    capsys.readouterr()

    losses = thz.discarded_scan_matrix_report(dataset)
    assert [loss["filename"] for loss in losses] == ["sample_damaged.acc"]
    assert "sample_damaged.acc" in capsys.readouterr().out


def test_single_scan_input_is_not_reported_as_a_loss(capsys):
    """One acquisition is a legitimate state, not a failure to preserve anything."""
    data_obj = make_thz_data(n_scans=1)
    dataset = FakeDataSet({data_obj.filename: data_obj})
    thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)

    status = thz.scan_matrix_status(data_obj)
    assert not status["intact"]
    assert status["reason"] is None
    assert "DROPPED" not in capsys.readouterr().out
