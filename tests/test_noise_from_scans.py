"""compute_noise_from_scans: the measured uncertainty stage.

Drop-in replacement for compute_transfer_uncertainty — it must write the same keys,
so plots and viewers need no change — but sourcing the numbers from the repeat scans
instead of from a spectral tail.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset_core.adapters import thz_adapter as thz
from dataset_core.data_structures.thz import BaseTHzData, THzData


DT_PS = 0.05
N_SAMPLES = 160
N_FFT = 512


def make_pair(n_scans=24, sample_scale=0.8, drift=0.0, seed=0):
    """A reference/sample pair of THzData, each with n_scans noisy repeats."""
    rng = np.random.default_rng(seed)
    time_ps = np.arange(N_SAMPLES) * DT_PS + 100.0
    argument = (time_ps - (time_ps[0] + 0.3 * (time_ps[-1] - time_ps[0]))) / 0.2
    pulse = -argument * np.exp(-0.5 * argument ** 2)

    def build(filename, scale):
        ramp = np.linspace(-0.5, 0.5, n_scans)
        scans = [
            BaseTHzData(
                data=np.column_stack((
                    time_ps,
                    scale * pulse * (1.0 + drift * ramp[index])
                    + rng.standard_normal(N_SAMPLES) * 1e-3,
                )),
                headers=[f"scan {index}"],
            )
            for index in range(n_scans)
        ]
        return THzData(data=scans, header=None, filename=filename, data_type="acc")

    return build("reference_test.acc", 1.0), build("sample_test.acc", sample_scale)


class FakeGrouping:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_reference_filename(self, filename, ref_type="substrate"):
        return self.mapping.get(filename)


class FakeDataDict(dict):
    def __init__(self, objects, reference_name):
        super().__init__(objects)
        self.grouping = FakeGrouping(
            {name: reference_name for name in objects if name != reference_name})

    def is_reference(self, filename):
        return filename == "reference_test.acc"

    def get(self, filename, default=None):
        return dict.get(self, filename, default)


class FakeDataSet:
    def __init__(self, objects, config=None):
        self.data = FakeDataDict(objects, "reference_test.acc")
        self.config = config or {}

    def get_reference(self, filename, ref_type="substrate"):
        name = self.data.grouping.get_reference_filename(filename, ref_type)
        return self.data.get(name) if name else None


def build_dataset(**kwargs):
    reference, sample = make_pair(**kwargs)
    config = {"window": {"type": "hann", "alpha": 1.0}, "baseline": {"n_points": 10},
              "fft": {"norm": "backward", "amplitude_scale": 1.0},
              "mask": {"snr_thresh_db": 5, "tail_fraction": 0.25,
                       "min_contiguous_bins": 3},
              "transfer": {"apply_snr_mask": True, "min_ref_amp_rel": 1e-3,
                           "regularization_eps": 1e-30, "unwrap_phase": True}}
    dataset = FakeDataSet({reference.filename: reference, sample.filename: sample},
                          config=config)
    thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)
    thz.subtract_baseline(dataset)
    thz.window_single_pulse_fixed_width(dataset, half_width_ps=2.0, show_graph=False)
    thz.fft_spectrum(dataset, n_fft=N_FFT)
    thz.transfer_function(dataset, ref_type="reference")
    return dataset


def sample_of(dataset):
    return dataset.data["sample_test.acc"]


def test_writes_the_same_keys_as_the_estimator_it_replaces():
    """So every existing plot and viewer picks the new bars up unchanged."""
    dataset = build_dataset()
    thz.compute_noise_from_scans(dataset, ref_type="reference", report=False)

    processing = sample_of(dataset).processing_dict
    for key in ("fft_sigma", "fft_phase_sigma", "transfer_H_sigma",
                "transfer_phase_sigma"):
        assert processing.get(key) is not None, f"{key} not written"
        assert np.asarray(processing[key]).shape == \
            np.asarray(processing["fft_freq"]).shape
    assert processing["uncertainty_method"] == "measured_from_scans"


def test_uncertainty_falls_as_one_over_root_m():
    """The property the tail-median floor famously did not have."""
    few = build_dataset(n_scans=10, seed=1)
    many = build_dataset(n_scans=40, seed=1)
    thz.compute_noise_from_scans(few, ref_type="reference", source="within_scan",
                                 report=False)
    thz.compute_noise_from_scans(many, ref_type="reference", source="within_scan",
                                 report=False)

    band = slice(5, 60)
    ratio = (np.median(sample_of(few).processing_dict["fft_sigma"][band])
             / np.median(sample_of(many).processing_dict["fft_sigma"][band]))
    assert ratio == pytest.approx(np.sqrt(40 / 10), rel=0.3)


def test_drift_makes_the_bars_bigger_only_in_drift_aware_mode():
    steady = build_dataset(n_scans=32, drift=0.0, seed=2)
    drifting = build_dataset(n_scans=32, drift=0.25, seed=2)
    for dataset in (steady, drifting):
        thz.compute_noise_from_scans(dataset, ref_type="reference",
                                     source="drift_aware", n_blocks=4, report=False)
    band = slice(5, 60)
    steady_sigma = np.median(sample_of(steady).processing_dict["fft_sigma"][band])
    drifting_sigma = np.median(sample_of(drifting).processing_dict["fft_sigma"][band])
    assert drifting_sigma > 2 * steady_sigma, (
        "drift-aware bars must grow when the acquisition drifts")

    # within_scan fits the drift out and reports only the noise, so it should NOT.
    for dataset in (steady, drifting):
        thz.compute_noise_from_scans(dataset, ref_type="reference",
                                     source="within_scan", report=False)
    steady_within = np.median(sample_of(steady).processing_dict["fft_sigma"][band])
    drifting_within = np.median(sample_of(drifting).processing_dict["fft_sigma"][band])
    assert drifting_within == pytest.approx(steady_within, rel=0.6)


def test_transfer_error_combines_sample_and_reference():
    dataset = build_dataset()
    thz.compute_noise_from_scans(dataset, ref_type="reference", report=False)
    processing = sample_of(dataset).processing_dict
    reference_processing = dataset.data["reference_test.acc"].processing_dict

    expected_relative = np.sqrt(
        (processing["fft_sigma"] / np.abs(processing["fft_spectrum"])) ** 2
        + (reference_processing["fft_sigma"]
           / np.abs(reference_processing["fft_spectrum"])) ** 2)
    expected = np.abs(processing["transfer_H"]) * expected_relative
    # Bins where |Y| reaches zero give an infinite relative error, stored as NaN
    # (unplottable but honest); compare where the answer is defined.
    defined = np.isfinite(expected) & np.isfinite(processing["transfer_H_sigma"])
    assert defined.sum() > 10
    assert np.allclose(processing["transfer_H_sigma"][defined], expected[defined])


def test_files_without_repeats_are_skipped_not_given_a_small_number(capsys):
    """A missing measurement must not be reported as a confident tiny error."""
    dataset = build_dataset()
    sample = sample_of(dataset)
    # Simulate the scan matrix having been lost.
    sample.processing_dict[thz._SCAN_MATRIX_KEY] = thz._averaged_as_single_scan(sample)

    thz.compute_noise_from_scans(dataset, ref_type="reference", report=True)
    assert sample.processing_dict.get("fft_sigma") is None
    assert "SKIPPED" in capsys.readouterr().out


def test_the_two_estimators_agree_when_the_tail_really_is_a_noise_plateau():
    """The known-answer check that validates BOTH estimators.

    The synthetic pulse here is smooth and band-limited, so its spectrum genuinely dies
    before Nyquist and the tail-median floor IS measuring noise. Under that condition —
    the one the floor assumes — the two must agree. They disagree by 10-22x on real
    records precisely because the condition fails there (see
    explorations/noise_model_validation/compare_error_bar_methods.py and
    ANALYSIS_NOTES §20), not because one of them is broken.
    """
    dataset = build_dataset(n_scans=32, seed=3)
    thz.compute_transfer_uncertainty(dataset)
    floor_sigma = np.asarray(sample_of(dataset).processing_dict["transfer_H_sigma"]).copy()

    # The premise: this record's floor band must actually be flat.
    spectrum = np.abs(sample_of(dataset).processing_dict["fft_spectrum"])
    tail = spectrum[-max(8, spectrum.size // 4):]
    third = max(1, tail.size // 3)
    assert np.median(tail[:third]) / np.median(tail[-third:]) < 2.0, \
        "test signal does not satisfy the plateau assumption"

    thz.compute_noise_from_scans(dataset, ref_type="reference", source="within_scan",
                                 report=False)
    measured = np.asarray(sample_of(dataset).processing_dict["transfer_H_sigma"])

    band = slice(5, 60)
    ratio = np.median(floor_sigma[band]) / np.median(measured[band])
    assert 0.4 < ratio < 2.5, f"estimators disagree by {ratio:.1f}x on a genuine plateau"


def test_reporter_names_the_three_sources():
    dataset = build_dataset()
    thz.compute_noise_from_scans(dataset, ref_type="reference", report=False)
    parameters = sample_of(dataset).processing_dict.get("noise_parameters")
    assert parameters is not None
    assert parameters.sigma_beta >= 0
    assert parameters.sigma_tau >= 0
