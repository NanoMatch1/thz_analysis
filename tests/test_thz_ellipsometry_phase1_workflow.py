"""Headless workflow tests for Phase 1 of thz_ellipsometry.

Every test writes real ``.acc`` files with the simulator, then drives the real driver -- loader,
repeat-scan noise model, harmonic fit, calibration registry, inversion, diagnostics -- exactly as
a run on bench data would. Fault-injection tests prove that each diagnostic fires on the fault it
guards and stays quiet on a clean run.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import thz_ellipsometry_run_me
from thz_ellipsometry.adapters import loader, noise
from thz_ellipsometry.adapters.report import format_run_report, plot_run
from thz_ellipsometry.adapters.stages import ACQUISITION_MODES, mode_names, run_ellipsometry
from thz_ellipsometry.core import materials, preprocess

INCIDENCE_ANGLE_DEG = thz_ellipsometry_run_me.config["geometry"]["incidence_angle_deg"]
PRIMARY_PROBE = thz_ellipsometry_run_me.config["detection"]["probe_azimuth_deg"]


def _write(directory, sample_name="doped_silicon", **options):
    options.setdefault("incidence_angle_deg", INCIDENCE_ANGLE_DEG)
    thz_ellipsometry_run_me.build_simulated_dataset(str(directory), sample_name, **options)
    return str(directory)


def _config(directory, *, expect=None, **sections):
    config = copy.deepcopy(thz_ellipsometry_run_me.config)
    config["data"]["directory"] = directory
    config["general"]["show_graph"] = False
    config["validation"]["expect"] = expect
    config["validation"]["cross_check_material"] = None
    if expect is None:
        config["validation"]["branch_reference_index"] = None
    for name, values in sections.items():
        config[name].update(values)
    return config


def _diagnostics(outcome):
    return [finding.diagnostic for finding in outcome.findings]


def _index_error(outcome, sample_name="doped_silicon"):
    frequencies = outcome.result.inversion.frequencies_hz
    truth = thz_ellipsometry_run_me.SIMULATED_SAMPLES[sample_name](frequencies)
    return float(np.median(np.abs(outcome.result.index - truth)))


# ---------------------------------------------------------------------------
# Loader: roles, probe settings, timestamps, magnet offset
# ---------------------------------------------------------------------------

def test_loader_groups_by_role_and_probe_and_orders_rows_in_time(tmp_path):
    _write(tmp_path, second_probe_azimuth_deg=15.0, angle_reference=True)
    series = loader.load_measurement(str(tmp_path), default_probe_azimuth_deg=PRIMARY_PROBE)
    assert set(series) == {("sample", PRIMARY_PROBE), ("sample", 15.0),
                           ("channel_reference", PRIMARY_PROBE),
                           ("angle_reference", PRIMARY_PROBE)}
    sample = series[("sample", PRIMARY_PROBE)]
    assert np.all(np.diff(sample.elapsed_seconds) > 0)
    assert np.allclose(sample.angles_deg[:4], [0.0, 90.0, 180.0, 270.0])
    later = series[("channel_reference", PRIMARY_PROBE)]
    assert later.elapsed_seconds.min() > sample.elapsed_seconds.max()


def test_loader_subtracts_the_magnet_angle_that_gives_p(tmp_path):
    _write(tmp_path, magnet_calibration={0.0: 7.0})
    raw = loader.load_measurement(str(tmp_path), default_probe_azimuth_deg=PRIMARY_PROBE)
    corrected = loader.load_measurement(str(tmp_path), magnet_calibration={0.0: 7.0},
                                        default_probe_azimuth_deg=PRIMARY_PROBE)
    key = ("sample", PRIMARY_PROBE)
    assert np.allclose(raw[key].angles_deg[:4], [7.0, 97.0, 187.0, 277.0])
    assert np.allclose(corrected[key].angles_deg[:4], [0.0, 90.0, 180.0, 270.0])


def test_a_silicon_sample_is_not_mistaken_for_the_angle_reference(tmp_path):
    _write(tmp_path, "hr_silicon")
    series = loader.load_measurement(str(tmp_path), default_probe_azimuth_deg=PRIMARY_PROBE)
    assert ("sample", PRIMARY_PROBE) in series
    assert not any(role == "angle_reference" for role, _ in series)


# ---------------------------------------------------------------------------
# Noise adapter
# ---------------------------------------------------------------------------

def test_noise_model_matches_the_scatter_of_the_repeat_scans(tmp_path):
    _write(tmp_path, scans_per_acquisition=6)
    sample = loader.load_measurement(
        str(tmp_path), default_probe_azimuth_deg=PRIMARY_PROBE)[("sample", PRIMARY_PROBE)]
    transformed = preprocess.transform_traces(sample.time_ps, sample.traces,
                                              window_half_width_ps=3.0)
    estimate = noise.estimate_series_noise(sample, transformed)
    band = (transformed.frequencies_hz > 0.8e12) & (transformed.frequencies_hz < 3.0e12)
    ratios = []
    for row, file in enumerate(sample.files):
        scans = file.scans - file.scans[:, :51].mean(axis=1, keepdims=True)
        spectra = np.fft.rfft(scans * transformed.window, n=transformed.padded_length, axis=1)
        empirical = (np.abs(spectra - spectra.mean(axis=0)) ** 2).sum(axis=0) / (
            (scans.shape[0] - 1) * scans.shape[0])
        ratios.append(np.median(estimate.spectral_variance[row][band])
                      / np.median(empirical[band]))
    assert np.median(ratios) == pytest.approx(1.0, abs=0.35)


def test_too_few_repeats_fall_back_and_say_why(tmp_path):
    outcome = run_ellipsometry(_config(_write(tmp_path, scans_per_acquisition=2)))
    reason = outcome.noise[("sample", PRIMARY_PROBE)].reason
    assert "repeat scans" in reason
    assert "noise_model_available" in _diagnostics(outcome)
    assert outcome.result.index_standard_error is not None, "residual-scatter bars remain"


# ---------------------------------------------------------------------------
# End to end: the clean run, and each calibration / angle / tilt route
# ---------------------------------------------------------------------------

def test_clean_doped_silicon_run_is_accurate_with_honest_bars_and_no_findings(tmp_path):
    outcome = run_ellipsometry(_config(_write(tmp_path)))
    result = outcome.result
    assert _index_error(outcome) < 0.02
    assert result.sample_fit.reduced_chi_square == pytest.approx(1.0, abs=0.5)
    assert result.sample_fit.fitted_drift_fs == pytest.approx(30.0, abs=3.0)
    # |dN| of a circular complex error is ~1.25x the per-component bar (Rayleigh mean); the
    # bars must be the right size, not merely present.
    assert 0.6 < _index_error(outcome) / np.median(result.index_standard_error) < 2.5
    assert outcome.findings == []


def test_probe_rotation_calibrates_as_well_as_gold_and_the_two_agree(tmp_path):
    directory = _write(tmp_path, second_probe_azimuth_deg=15.0)
    with_gold = run_ellipsometry(_config(directory))
    with_probe = run_ellipsometry(_config(directory, calibration={"channel": "probe_rotation"}))
    assert with_probe.result.calibration.ratio == pytest.approx(
        with_gold.result.calibration.ratio, rel=3e-3)
    assert _index_error(with_probe) < 0.02
    assert "calibration_sources_agree" not in _diagnostics(with_gold)


def test_incidence_angle_fitted_on_the_angle_reference_corrects_a_wrong_nominal(tmp_path):
    directory = _write(tmp_path, angle_reference=True)
    config = _config(directory, calibration={"incidence_angle": "fit_from_reference"},
                     geometry={"incidence_angle_deg": INCIDENCE_ANGLE_DEG - 1.0})
    outcome = run_ellipsometry(config)
    used = np.rad2deg(outcome.result.inversion.incidence_angle_rad)
    assert used == pytest.approx(INCIDENCE_ANGLE_DEG, abs=0.05)
    assert "fit_from_reference" in outcome.result.incidence_angle_source
    assert _index_error(outcome) < 0.02


def test_fitted_tilt_rescues_a_tilted_dispersive_sample(tmp_path):
    directory = _write(tmp_path, out_of_plane_tilt_deg=0.7)
    ignored = run_ellipsometry(_config(directory))
    fitted = run_ellipsometry(_config(directory, inversion={
        "fit_out_of_plane_tilt": True, "tilt_fixed_parameters": {"eps_inf": 11.68}}))
    assert _index_error(ignored) > 0.1
    assert _index_error(fitted) < 0.02
    assert fitted.result.tilt_fit.tilt_deg == pytest.approx(0.7, abs=0.03)
    resistivity = 10.0 ** fitted.result.tilt_fit.model_parameters["log10_resistivity_ohm_cm"]
    assert resistivity == pytest.approx(1.0, rel=0.1)


def test_a_known_magnet_offset_in_the_config_is_removed_exactly(tmp_path):
    directory = _write(tmp_path, magnet_calibration={0.0: 7.0})
    unknown = run_ellipsometry(_config(directory))
    known = run_ellipsometry(_config(directory, geometry={"magnet_calibration": {0.0: 7.0}}))
    assert _index_error(unknown) > 0.1
    assert _index_error(known) < 0.02


# ---------------------------------------------------------------------------
# Diagnostics: each fires on its fault
# ---------------------------------------------------------------------------

def test_large_background_is_removed_but_flagged(tmp_path):
    # The simulated background is a broader pulse, mostly below the band: 4x the peak in time
    # is ~15% of the signal in 0.8-3 THz, over the 10% threshold.
    outcome = run_ellipsometry(_config(_write(tmp_path, background_relative=4.0)))
    assert "background_is_small" in _diagnostics(outcome)
    assert _index_error(outcome) < 0.02, "the background term still removes it"


def test_a_hann_window_under_drift_breaks_the_harmonic_model_and_tukey_does_not(tmp_path):
    """The reason the default window is a flat-top Tukey (core/preprocess.py docstring)."""
    directory = _write(tmp_path, background_relative=2.0)
    hann = run_ellipsometry(_config(directory, preprocess={"window_shape": "hann"}))
    tukey = run_ellipsometry(_config(directory))
    assert hann.result.sample_fit.reduced_chi_square > 3.0
    assert tukey.result.sample_fit.reduced_chi_square == pytest.approx(1.0, abs=0.3)
    assert "harmonic_model_holds" in _diagnostics(hann)
    assert "harmonic_model_holds" not in _diagnostics(tukey)


def test_an_unmodelled_drift_breaks_the_harmonic_model_and_is_flagged(tmp_path):
    directory = _write(tmp_path, drift_span_fs=80.0)
    outcome = run_ellipsometry(_config(directory, acquisition={"drift_model": "none"}))
    assert "harmonic_model_holds" in _diagnostics(outcome)


def test_a_sign_flipped_stored_calibration_is_caught(tmp_path):
    directory = _write(tmp_path, "hr_silicon")
    good = run_ellipsometry(_config(directory, expect="hr_silicon"))
    flipped = run_ellipsometry(_config(
        directory, expect="hr_silicon",
        calibration={"channel": "stored",
                     "stored_channel_ratio": -complex(good.result.calibration.ratio)}))
    assert "sign_convention_consistent" not in _diagnostics(good)
    assert "sign_convention_consistent" in _diagnostics(flipped)


def test_a_large_fitted_tilt_is_flagged(tmp_path):
    directory = _write(tmp_path, out_of_plane_tilt_deg=1.5)
    outcome = run_ellipsometry(_config(directory, inversion={
        "fit_out_of_plane_tilt": True, "tilt_fixed_parameters": {"eps_inf": 11.68}}))
    assert "tilt_is_plausible" in _diagnostics(outcome)


def test_ellipsometry_assumptions_appear_in_the_generated_ledger():
    import thz_ellipsometry.adapters.diagnostics  # noqa: F401 -- registers the checks
    from dataset_core.adapters.diagnostics import DIAGNOSTIC_REGISTRY, render_assumptions_ledger
    ledger = render_assumptions_ledger()
    entries = [entry for entry in DIAGNOSTIC_REGISTRY.values()
               if entry.stage.startswith("ellipsometry.")]
    assert len(entries) >= 10
    for entry in entries:
        assert entry.assumption in ledger and entry.remedy in ledger


def test_ellipsometry_checks_ignore_a_dataset_that_is_not_an_ellipsometry_run():
    import thz_ellipsometry.adapters.diagnostics  # noqa: F401
    from dataset_core.adapters.diagnostics import run_diagnostics
    findings = run_diagnostics(object(), {}, stage="ellipsometry.inversion")
    assert findings == []


# ---------------------------------------------------------------------------
# Registry, report, driver
# ---------------------------------------------------------------------------

def test_acquisition_modes_are_registered_and_unknown_ones_explained(tmp_path):
    assert "isotropic" in mode_names()
    assert ACQUISITION_MODES["isotropic"].summary
    with pytest.raises(KeyError, match="registered"):
        run_ellipsometry({"mode": "no_such_mode"})


def test_report_shows_bars_systematic_and_findings_and_the_figure_renders(tmp_path):
    outcome = run_ellipsometry(_config(_write(tmp_path, background_relative=4.0)))
    text = format_run_report(outcome)
    assert "median noise bar" in text
    assert "angle uncertainty" in text
    assert "[check_assumptions]" in text and "background_is_small" in text
    figure_path = tmp_path / "run.png"
    plot_run(outcome, show=False, save_path=str(figure_path))
    assert figure_path.stat().st_size > 0


def test_driver_runs_the_doped_silicon_simulation_headlessly():
    assert thz_ellipsometry_run_me.main(["--simulate", "doped_silicon", "--no-graph"]) == 0


def test_per_state_magnet_scale_errors_need_the_four_null_table(tmp_path):
    """An offset-only calibration leaves per-state errors in; the four-state table removes them."""
    true_table = {0.0: 86.3, 90.0: 176.8, 180.0: 266.0, 270.0: 356.9}
    directory = _write(tmp_path, magnet_calibration=true_table)
    offset_only = run_ellipsometry(_config(directory, geometry={"magnet_calibration":
                                                                 {0.0: 86.3}}))
    full = run_ellipsometry(_config(directory, geometry={"magnet_calibration": true_table}))
    assert _index_error(full) < 0.02
    assert _index_error(offset_only) > 2 * _index_error(full)


def test_amplitude_drift_is_fitted_end_to_end(tmp_path):
    outcome = run_ellipsometry(_config(_write(tmp_path, amplitude_drift=0.03)))
    fit = outcome.result.sample_fit
    assert fit.fitted_amplitude_change == pytest.approx(0.03, abs=0.005)
    assert _index_error(outcome) < 0.02
