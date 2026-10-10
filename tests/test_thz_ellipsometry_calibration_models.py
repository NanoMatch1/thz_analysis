"""Calibration models: known answers for each registered model, the pipeline with a planted p/s
delay, and the whole driver run headlessly with a model chosen in the config."""

from __future__ import annotations

import copy

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import thz_ellipsometry_run_me
from thz_ellipsometry.adapters import inspection
from thz_ellipsometry.adapters.export import load_saved_run, save_run
from thz_ellipsometry.adapters.stages import run_ellipsometry
from thz_ellipsometry.core import materials
from thz_ellipsometry.core.calibration import (
    apply_calibration_model,
    channel_ratio_from_reference,
)
from thz_ellipsometry.core.calibration_models import (
    CALIBRATION_MODELS,
    calibration_model_names,
    fit_calibration_model,
)
from thz_ellipsometry.core.pipeline import analyse_polarisation_series
from thz_ellipsometry.core.simulate import synthesize_spectra

FREQUENCIES_HZ = np.linspace(0.3e12, 3.0e12, 120)
BAND = (FREQUENCIES_HZ > 0.5e12) & (FREQUENCIES_HZ < 2.6e12)


def _planted(delay_fs=4.0, slope_percent_per_thz=3.0, noise=0.0, seed=1):
    centre = FREQUENCIES_HZ[BAND].mean()
    offset = FREQUENCIES_HZ - centre
    truth = (0.82 * (1.0 + slope_percent_per_thz / 100.0 * offset / 1e12)
             * np.exp(1j * (np.deg2rad(170.0) - 2.0 * np.pi * offset * delay_fs * 1e-15)))
    generator = np.random.default_rng(seed)
    measured = truth + noise * (generator.normal(size=truth.shape)
                                + 1j * generator.normal(size=truth.shape))
    return truth, measured


def test_the_registry_lists_every_model_and_explains_unknown_ones():
    assert set(calibration_model_names()) == {
        "constant", "constant_plus_delay", "constant_plus_delay_and_slope", "per_frequency"}
    assert all(entry.summary for entry in CALIBRATION_MODELS.values())
    with pytest.raises(KeyError, match="registered"):
        fit_calibration_model("nope", FREQUENCIES_HZ, np.ones(FREQUENCIES_HZ.size), BAND)


def test_delay_and_slope_are_recovered_exactly_without_noise():
    truth, measured = _planted()
    fit = fit_calibration_model("constant_plus_delay_and_slope", FREQUENCIES_HZ, measured, BAND)
    assert fit.parameters["delay_fs"] == pytest.approx(4.0, abs=1e-6)
    assert fit.parameters["slope_percent_per_thz"] == pytest.approx(3.0, abs=1e-6)
    np.testing.assert_allclose(fit.applied_per_frequency, truth, rtol=1e-8)


def test_the_delay_fit_stays_within_its_error_bar_under_noise():
    _, measured = _planted(slope_percent_per_thz=0.0, noise=0.01)
    variance = np.full(FREQUENCIES_HZ.shape, 2 * 0.01**2)
    fit = fit_calibration_model("constant_plus_delay", FREQUENCIES_HZ, measured, BAND, variance)
    error = fit.standard_errors["delay_fs"]
    assert 0.0 < error < 1.0
    assert fit.parameters["delay_fs"] == pytest.approx(4.0, abs=4 * error)


def test_the_constant_model_is_the_original_band_mean():
    """Backward compatible: the default model reproduces what was applied before models."""
    _, measured = _planted(noise=0.005)
    expected = materials.gold_index(FREQUENCIES_HZ)
    from thz_ellipsometry.core.model import ellipsometric_ratio
    reference = measured * ellipsometric_ratio(expected, np.deg2rad(45.0))
    source = channel_ratio_from_reference(reference, expected, np.deg2rad(45.0),
                                          frequency_mask=BAND)
    modelled = apply_calibration_model(source, "constant", FREQUENCIES_HZ, BAND)
    assert modelled.ratio == pytest.approx(source.ratio)
    np.testing.assert_allclose(modelled.applied_ratio, source.ratio)
    assert modelled.residual_relative_scatter == pytest.approx(source.relative_scatter)
    assert modelled.standard_error == pytest.approx(source.standard_error)


def test_per_frequency_applies_the_measurement_itself():
    _, measured = _planted(noise=0.005)
    fit = fit_calibration_model("per_frequency", FREQUENCIES_HZ, measured, BAND)
    np.testing.assert_array_equal(fit.applied_per_frequency, measured)


def _series(index, channel_delay_s, seed):
    return synthesize_spectra(index, FREQUENCIES_HZ, incidence_angle_rad=np.deg2rad(45.0),
                              channel_delay_s=channel_delay_s, relative_noise=1e-4, seed=seed)


@pytest.mark.parametrize("model, tolerance", [("constant", None),
                                              ("constant_plus_delay", 0.01),
                                              ("per_frequency", 0.01)])
def test_a_planted_instrument_delay_biases_k_unless_the_model_carries_it(model, tolerance):
    """4 fs of p/s delay at 45 deg on HR-Si: the constant model leaves ~0.1-0.4 in k."""
    silicon = materials.high_resistivity_silicon_index(FREQUENCIES_HZ)
    gold = materials.gold_index(FREQUENCIES_HZ)
    sample, reference = _series(silicon, 4e-15, 0), _series(gold, 4e-15, 1)
    result = analyse_polarisation_series(
        frequencies_hz=FREQUENCIES_HZ, sample_spectra=sample.spectra,
        sample_emitter_angles_rad=sample.emitter_angles_rad,
        reference_spectra=reference.spectra, reference_index=gold,
        incidence_angle_rad=np.deg2rad(45.0), frequency_min_hz=0.5e12,
        frequency_max_hz=2.6e12, drift_model="none", calibration_model=model,
        branch_reference_index=3.4175)
    error = np.abs(result.index - silicon[result.band])
    assert result.calibration.model_name == model
    if tolerance is None:
        assert error.max() > 0.1
    else:
        assert error.max() < tolerance
    if model == "constant_plus_delay":
        assert result.calibration.model_parameters["delay_fs"] == pytest.approx(4.0, abs=0.05)


@pytest.fixture(scope="module")
def delayed_run(tmp_path_factory):
    """The whole driver on simulated .acc files with a 4 fs p/s instrument delay planted."""
    data = tmp_path_factory.mktemp("data")
    thz_ellipsometry_run_me.build_simulated_dataset(
        str(data), "hr_silicon", incidence_angle_deg=45.0, channel_delay_fs=4.0,
        emitter_angles_deg=(0.0, 90.0, 180.0, 270.0, 180.0, 90.0, 0.0), cycles=1)
    outcomes = {}
    for model in ("constant", "constant_plus_delay_and_slope"):
        config = copy.deepcopy(thz_ellipsometry_run_me.config)
        config["data"]["directory"] = str(data)
        config["general"]["show_graph"] = False
        config["validation"].update(expect="hr_silicon", cross_check_material=None)
        config["calibration"]["model"] = model
        config["acquisition"]["rows"] = "acquisition"
        outcomes[model] = run_ellipsometry(config)
    return outcomes


def _k_error(outcome):
    inversion = outcome.result.inversion
    truth = materials.high_resistivity_silicon_index(inversion.frequencies_hz)
    return float(np.median(np.abs(inversion.extinction - (-truth.imag))))


def test_the_driver_applies_the_configured_model_end_to_end(delayed_run):
    constant, delayed = delayed_run["constant"], delayed_run["constant_plus_delay_and_slope"]
    parameters = delayed.result.calibration.model_parameters
    # The fitted delay is the instrument's AS SEEN THROUGH the schedule and drift model: on a
    # single-pass-like schedule at C ~ -1 the drift fit absorbs a few tenths of a fs of a p/s
    # delay (and the window a little more). Gold and sample share the schedule, so that part
    # cancels in rho -- which is why k, not the delay, is the assertion that matters.
    assert parameters["delay_fs"] == pytest.approx(4.0, abs=1.0)
    assert abs(parameters["slope_percent_per_thz"]) < 2.0
    assert _k_error(delayed) < 0.5 * _k_error(constant)
    assert _k_error(delayed) < 0.03


def test_the_model_reaches_the_bundle_and_the_calibration_figure(delayed_run, tmp_path):
    outcome = delayed_run["constant_plus_delay_and_slope"]
    bundle = save_run(outcome, directory=str(tmp_path), notes_mode="none", figure=False,
                      inspection_figures=False, output_function=lambda *_: None)
    loaded = load_saved_run(bundle)
    assert loaded.recipe["summary"]["calibration_model"] == "constant_plus_delay_and_slope"
    assert "delay_fs" in loaded.recipe["summary"]["calibration_model_parameters"]
    np.testing.assert_allclose(loaded.arrays["calibration_applied_per_frequency"],
                               outcome.result.calibration.applied_ratio)
    record = inspection.record_from_outcome(outcome)
    back = inspection.load_inspection(inspection.save_inspection(record, str(tmp_path / "i.npz")))
    assert back.calibration_model == "constant_plus_delay_and_slope"
    assert "delay" in back.calibration_model_description
    np.testing.assert_allclose(back.calibration_applied_per_frequency,
                               outcome.result.calibration.applied_ratio)
    figures = inspection.make_inspection_figures(back, ("calibration",))
    (figure,) = figures.values()
    assert len(figure.axes) >= 6
