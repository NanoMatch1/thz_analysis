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
    # Written for one row per file on staircase synthetic drift; the per-scan layout has its
    # own tests with continuous drift (test_thz_ellipsometry_phase1_workflow.py, 'scan rows').
    config["acquisition"]["rows"] = "acquisition"
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
    series = loader.load_measurement(str(tmp_path), magnet_calibration=thz_ellipsometry_run_me.config["geometry"][
                                          "magnet_calibration"],
                                      default_probe_azimuth_deg=PRIMARY_PROBE)
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


def test_a_measured_purge_transient_is_removed_from_a_palindrome_ten_minutes_after_closing(
        tmp_path):
    """F40: delay (tau 38 min) + water tilt (tau 47 min), spans from the August purge data.

    Noise-free, so the claim is exact rather than statistical: the 'auto' models (exponential
    settling delay + settling tilt) remove the transient completely, the linear ramps do not.
    (With noise, 6 seeds give a median index error of 0.0133 with the purge vs 0.0129 without.)
    """
    from thz_ellipsometry.adapters.synthetic_files import write_accumulation_files
    from thz_ellipsometry.core.simulate import AcquisitionSchedule
    palindrome = np.array([0.0, 90.0, 180.0, 270.0, 270.0, 180.0, 90.0, 0.0])
    elapsed = 600.0 + np.arange(palindrome.size) * 600.0      # starts 10 min after closing
    delays = -40e-15 * np.exp(-elapsed / (38 * 60.0))
    tilts = -0.096 * np.exp(-elapsed / (47 * 60.0))
    schedule = AcquisitionSchedule(np.deg2rad(palindrome), elapsed - elapsed[0],
                                   np.repeat([1, 2], 4))
    shared = dict(schedule=schedule, scans_per_angle=4, angle_token="mag",
                  incidence_angle_rad=np.deg2rad(INCIDENCE_ANGLE_DEG),
                  crystal_orientation_rad=np.deg2rad(90.0), relative_noise=0.0,
                  probe_azimuth_deg=PRIMARY_PROBE, drift_span_s=delays - delays[0],
                  magnet_calibration=thz_ellipsometry_run_me.config["geometry"][
                      "magnet_calibration"],
                  tilt_profile=tilts - tilts[0])
    write_accumulation_files(str(tmp_path), sample_name="doped_silicon", seed=0,
                             index_sample_function=thz_ellipsometry_run_me.SIMULATED_SAMPLES[
                                 "doped_silicon"], **shared)
    write_accumulation_files(str(tmp_path), sample_name="ref-gold", seed=1,
                             index_sample_function=materials.gold_index,
                             time_offset_seconds=elapsed[-1] + 600.0, **shared)
    noise_off = {"enabled": False}
    modelled = run_ellipsometry(_config(str(tmp_path), noise=noise_off))
    unmodelled = run_ellipsometry(_config(str(tmp_path), noise=noise_off, acquisition={
        "drift_model": "linear_ramp", "amplitude_model": "linear_ramp"}))
    assert modelled.result.sample_fit.drift_model == "settling"
    assert modelled.result.sample_fit.amplitude_model == "tilt_settling"
    assert _index_error(modelled) < 1e-3
    assert _index_error(unmodelled) > 0.02


def test_a_single_pass_with_c_near_minus_one_is_flagged_as_inseparable(tmp_path):
    """F40: in 0/90/180/270 on gold with C ~ -1, drift and the channel ratio trade."""
    directory = _write(tmp_path, cycles=1, crystal_001_from_p_deg=90.0,
                       emitter_angles_deg=(0.0, 90.0, 180.0, 270.0))
    outcome = run_ellipsometry(_config(directory))
    assert "drift_separable_from_ratio" in _diagnostics(outcome)


# ---------------------------------------------------------------------------
# Per-scan rows (acquisition.rows = 'scan')
# ---------------------------------------------------------------------------

SINGLE_PASS_SCAN_OPTIONS = dict(cycles=1, scans_per_acquisition=8, seconds_per_scan=48.0,
                                drift_span_fs=30.0, drift_within_acquisition=True)


def test_scan_rows_keep_each_scan_with_its_own_time_and_file(tmp_path):
    _write(tmp_path, **SINGLE_PASS_SCAN_OPTIONS)
    config = _config(str(tmp_path))
    series = loader.load_measurement(
        str(tmp_path), magnet_calibration=config["geometry"]["magnet_calibration"],
        default_probe_azimuth_deg=PRIMARY_PROBE)[("sample", PRIMARY_PROBE)]
    rows = loader.expand_to_scan_rows(series)
    assert len(rows) == int(series.scan_counts.sum()) == 4 * 8
    assert np.array_equal(rows.segment_ids, np.repeat(np.arange(4), 8))
    assert np.array_equal(rows.angles_deg, np.repeat(series.angles_deg, 8))
    assert np.all(np.diff(rows.elapsed_seconds) > 0)
    for index in range(4):     # each file's scans are centred on the file's own elapsed time
        inside = rows.elapsed_seconds[rows.segment_ids == index]
        assert inside.mean() == pytest.approx(series.elapsed_seconds[index])
        assert np.diff(inside) == pytest.approx(48.0)
    assert np.allclose(rows.traces[rows.segment_ids == 2].mean(axis=0), series.traces[2])


def test_scan_rows_need_timestamps(tmp_path):
    _write(tmp_path, **SINGLE_PASS_SCAN_OPTIONS)
    series = loader.load_measurement(str(tmp_path))[("sample", None)]
    from dataclasses import replace
    with pytest.raises(ValueError, match="timestamp"):
        loader.expand_to_scan_rows(replace(series, elapsed_seconds=None))


def test_per_scan_noise_is_the_variance_of_one_scan_not_of_the_mean(tmp_path):
    _write(tmp_path, relative_noise=0.01, **SINGLE_PASS_SCAN_OPTIONS)
    series = loader.load_measurement(str(tmp_path))[("sample", None)]
    rows = loader.expand_to_scan_rows(series)
    transformed = preprocess.transform_traces(series.time_ps, rows.traces,
                                              window_half_width_ps=3.0)
    per_file = noise.estimate_series_noise(series, transformed)
    per_scan = noise.estimate_series_noise(rows, transformed)
    assert per_scan.spectral_variance.shape[0] == len(rows)
    assert np.allclose(per_scan.spectral_variance[rows.segment_ids == 1],
                       8 * per_file.spectral_variance[1])


def test_single_pass_with_drift_inside_the_files_is_recovered_from_scan_rows(tmp_path):
    """Continuous drift through a single pass: per-scan rows fix it; per-file rows cannot (F40)."""
    _write(tmp_path, relative_noise=0.002, **SINGLE_PASS_SCAN_OPTIONS)
    outcome = run_ellipsometry(_config(str(tmp_path), acquisition={"rows": "scan"}))
    sample = outcome.fits[("sample", PRIMARY_PROBE)]
    assert sample.drift_model == "segment_settling"
    assert sample.segment_ids is not None and sample.delays_s.size == 4 * 8
    assert _index_error(outcome) < 0.02
    assert sample.reduced_chi_square == pytest.approx(1.0, abs=0.3)
    # Measured 2026-10-07: 0.010 per scan vs 0.149 per file on this dataset.
    per_file = run_ellipsometry(_config(str(tmp_path), acquisition={"rows": "acquisition"}))
    assert per_file.fits[("sample", PRIMARY_PROBE)].segment_ids is None
    assert _index_error(per_file) > 5 * _index_error(outcome)


def test_a_permanent_step_at_each_opening_is_the_known_limit_of_scan_rows(tmp_path):
    """The one assumption: an opening leaves no lasting step. A staircase drift breaks it, and
    in a single pass nothing in the data can show it -- only a palindrome can. Kept as a test so
    the limit stays documented and visible if the model changes."""
    options = dict(SINGLE_PASS_SCAN_OPTIONS, drift_within_acquisition=False)
    _write(tmp_path, relative_noise=0.002, **options)
    outcome = run_ellipsometry(_config(str(tmp_path), acquisition={"rows": "scan"}))
    assert _index_error(outcome) > 0.1
    assert outcome.fits[("sample", PRIMARY_PROBE)].reduced_chi_square < 2.0   # looks fine


def test_an_unknown_row_layout_is_explained(tmp_path):
    _write(tmp_path, **SINGLE_PASS_SCAN_OPTIONS)
    with pytest.raises(ValueError, match="acquisition.rows"):
        run_ellipsometry(_config(str(tmp_path), acquisition={"rows": "per_file"}))
