"""Headless end-to-end workflow test for the ellipsometry driver.

Drives the real pipeline -- real `.acc` files on disk, the real loader, the real preprocessing,
fit, calibration, inversion and validation -- on synthesized data. No hardware, no GUI, no
network. This is gate V5 of the MVP plan and the test that catches integration rot between the
pure core and the adapter layer.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from thz_ellipsometry.adapters import loader, synthetic_files
from thz_ellipsometry.core import materials, preprocess, simulate
from thz_ellipsometry.adapters.report import format_run_report, plot_run
from thz_ellipsometry.adapters.stages import ELLIPSOMETRY_STAGES, run_ellipsometry, stage_names

import thz_ellipsometry_run_me


#: The plan's magnet states (sec. 4.2), cycled CYCLES times.
EMITTER_ANGLES_DEG = np.array([0.0, 90.0, 180.0, 270.0])
CYCLES = 3
# Follow the driver's own geometry rather than duplicating it, so a change to the specced
# incidence angle does not silently desynchronise the synthesized data from the analysis.
INCIDENCE_ANGLE_DEG = thz_ellipsometry_run_me.config["geometry"]["incidence_angle_deg"]


def _base_config(directory, expect="hr_silicon"):
    import copy
    config = copy.deepcopy(thz_ellipsometry_run_me.config)
    config["data"]["directory"] = directory
    config["general"]["show_graph"] = False
    config["validation"]["expect"] = expect
    # Written for one row per file on staircase synthetic drift; the per-scan layout has its
    # own tests with continuous drift (test_thz_ellipsometry_phase1_workflow.py, 'scan rows').
    config["acquisition"]["rows"] = "acquisition"
    return config


def _write_dataset(directory, sample_name, *, incidence_angle_deg=INCIDENCE_ANGLE_DEG,
                   **options):
    return thz_ellipsometry_run_me.build_simulated_dataset(
        str(directory), sample_name, incidence_angle_deg=incidence_angle_deg,
        emitter_angles_deg=EMITTER_ANGLES_DEG, cycles=CYCLES, **options)


def _primary_probe():
    return thz_ellipsometry_run_me.config["detection"]["probe_azimuth_deg"]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("hr_silicon_pol=15.acc", 15.0),
    ("ref-gold_pol=0.acc", 0.0),
    ("sample_pol=m30_temp=300K.acc", -30.0),
    ("sample_temp=300K_pol=112.5.acc", 112.5),
    ("sample_no_token.acc", None),
])
def test_polarisation_angle_parsed_from_the_repo_filename_grammar(filename, expected):
    assert loader.polarisation_angle_deg_from_filename(filename) == expected


def test_accumulation_file_round_trips_through_the_writer_and_reader(tmp_path):
    paths = synthetic_files.write_accumulation_files(
        str(tmp_path), index_sample_function=materials.high_resistivity_silicon_index,
        emitter_angles_rad=np.deg2rad([0.0, 45.0]), sample_name="probe", scans_per_angle=3)
    assert len(paths) == 2
    parsed = loader.read_accumulation_file(paths[0])
    assert parsed.scans.shape[0] == 3
    assert parsed.time_ps.size == parsed.scans.shape[1]
    assert np.all(np.diff(parsed.time_ps) > 0)


def test_loader_splits_sample_from_reference(tmp_path):
    _write_dataset(tmp_path, "hr_silicon")
    series = loader.load_measurement(str(tmp_path), magnet_calibration=thz_ellipsometry_run_me.config["geometry"][
                                          "magnet_calibration"],
                                      default_probe_azimuth_deg=_primary_probe())
    sample = series[("sample", _primary_probe())]
    reference = series[("channel_reference", _primary_probe())]
    assert len(sample) == len(reference) == len(EMITTER_ANGLES_DEG) * CYCLES
    assert np.allclose(np.unique(sample.angles_deg), EMITTER_ANGLES_DEG)


def test_loader_explains_a_missing_polarisation_token(tmp_path):
    synthetic_files.write_accumulation_files(
        str(tmp_path), index_sample_function=materials.high_resistivity_silicon_index,
        emitter_angles_rad=np.deg2rad([0.0]), sample_name="probe", angle_token="angle")
    with pytest.raises(ValueError, match="angle_token"):
        loader.load_polarisation_series(str(tmp_path), angle_token="pol")


def test_loader_reports_a_missing_directory_and_an_empty_one(tmp_path):
    with pytest.raises(NotADirectoryError):
        loader.load_polarisation_series(str(tmp_path / "absent"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match=".acc"):
        loader.load_polarisation_series(str(tmp_path / "empty"))


def test_spectra_use_a_common_window_centre_so_relative_timing_survives(tmp_path):
    """Windowing each angle at its own peak would erase the drift the model must measure."""
    _write_dataset(tmp_path, "hr_silicon", relative_noise=0.0, drift_span_fs=200.0)
    sample = loader.load_measurement(
        str(tmp_path), default_probe_azimuth_deg=_primary_probe())[("sample", _primary_probe())]
    frequencies, spectra = preprocess.spectra_from_traces(
        sample.time_ps, sample.traces, window_half_width_ps=3.0)
    band = (frequencies > 0.8e12) & (frequencies < 2.0e12)
    phase_first = np.unwrap(np.angle(spectra[0][band]))
    phase_last = np.unwrap(np.angle(spectra[-1][band]))
    assert np.ptp(phase_last - phase_first) > 0.05


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_stage_registry_is_populated_and_self_describing():
    assert set(stage_names()) >= {"load_series", "transform_to_spectra", "analyse", "validate"}
    for name in stage_names():
        assert ELLIPSOMETRY_STAGES[name].summary, f"{name} has no help text"


# ---------------------------------------------------------------------------
# V5: the full chain
# ---------------------------------------------------------------------------

def test_workflow_recovers_hr_silicon_from_files_on_disk(tmp_path):
    _write_dataset(tmp_path, "hr_silicon")
    outcome = run_ellipsometry(_base_config(str(tmp_path)))

    assert outcome.report is not None
    assert outcome.report.passed, str(outcome.report)
    assert np.median(outcome.result.inversion.refractive_index) == pytest.approx(3.4175, abs=0.02)
    assert outcome.result.calibration.is_flat


def test_workflow_recovers_doped_silicon_and_sees_its_absorption(tmp_path):
    """The doped wafer is where k is actually measurable -- that is the point of the pair."""
    _write_dataset(tmp_path, "doped_silicon")
    config = _base_config(str(tmp_path), expect=None)
    config["validation"]["branch_reference_index"] = None
    outcome = run_ellipsometry(config)

    truth = materials.doped_silicon_index(outcome.result.inversion.frequencies_hz, 1.0)
    recovered = outcome.result.inversion
    assert np.median(np.abs(recovered.refractive_index - truth.real)) < 0.03
    assert np.median(np.abs(recovered.extinction - (-truth.imag))) < 0.03
    low_band = recovered.frequencies_hz < 1.2e12
    assert np.median(recovered.extinction[low_band]) > 0.05, \
        "doped silicon must show real absorption where it has it (k ~ 0.1 below 1.2 THz)"
    # Passive within the noise everywhere; positive conductivity wherever k is resolved. Near
    # 3 THz the true k (~0.006) is below the noise bar, so a bin there may fall below zero.
    bar = outcome.result.index_standard_error
    assert np.all(recovered.extinction > -3.0 * bar)
    resolved = -truth.imag > 3.0 * bar
    assert np.all(recovered.conductivity_real_si[resolved] > 0)


def test_workflow_recovers_the_injected_drift(tmp_path):
    _write_dataset(tmp_path, "hr_silicon", relative_noise=0.0, drift_span_fs=60.0)
    outcome = run_ellipsometry(_base_config(str(tmp_path)))
    assert outcome.result.sample_fit.fitted_drift_fs == pytest.approx(60.0, abs=5.0)
    # Noise-free repeats carry no measured noise: the fit must fall back, not divide by zero.
    assert outcome.noise[("sample", _primary_probe())].spectral_variance is None


def test_workflow_cross_checks_a_deliberately_wrong_incidence_angle(tmp_path):
    """Mount the sample 2 deg off what the analysis is told, and the cross-check must notice."""
    actual = INCIDENCE_ANGLE_DEG + 2.0
    _write_dataset(tmp_path, "hr_silicon", incidence_angle_deg=actual, relative_noise=0.0)
    outcome = run_ellipsometry(_base_config(str(tmp_path)))
    assert outcome.result.incidence_angle_fit.angle_deg == pytest.approx(actual, abs=0.1)
    assert not outcome.report.passed
    failing = [check.name for check in outcome.report.checks if not check.passed]
    assert any("incidence angle" in name for name in failing)


def test_a_degenerate_primary_probe_stops_the_run_and_says_why(tmp_path):
    _write_dataset(tmp_path, "hr_silicon", probe_azimuth_deg=0.0)
    config = _base_config(str(tmp_path))
    config["detection"]["probe_azimuth_deg"] = 0.0
    with pytest.raises(ValueError, match="DEGENERATE"):
        run_ellipsometry(config)


def test_a_degenerate_secondary_probe_is_flagged_by_its_diagnostic(tmp_path):
    _write_dataset(tmp_path, "hr_silicon", second_probe_azimuth_deg=45.0)
    outcome = run_ellipsometry(_base_config(str(tmp_path)))
    flagged = [finding for finding in outcome.findings if finding.diagnostic == "probe_degeneracy"]
    assert flagged and "45" in flagged[0].message


def test_workflow_warns_when_too_few_angles_were_measured(tmp_path):
    thz_ellipsometry_run_me.build_simulated_dataset(
        str(tmp_path), "hr_silicon", incidence_angle_deg=INCIDENCE_ANGLE_DEG,
        emitter_angles_deg=np.array([0.0, 90.0]), cycles=3)
    config = _base_config(str(tmp_path))
    config["acquisition"]["background_term"] = False
    outcome = run_ellipsometry(config)
    assert any("distinct polarisation angles" in warning for warning in outcome.warnings)


def test_two_angles_cannot_carry_a_background_term_and_the_error_says_so(tmp_path):
    thz_ellipsometry_run_me.build_simulated_dataset(
        str(tmp_path), "hr_silicon", incidence_angle_deg=INCIDENCE_ANGLE_DEG,
        emitter_angles_deg=np.array([0.0, 90.0]), cycles=3)
    with pytest.raises(ValueError, match="background"):
        run_ellipsometry(_base_config(str(tmp_path)))


def test_run_report_and_figure_render_headlessly(tmp_path):
    _write_dataset(tmp_path, "hr_silicon")
    outcome = run_ellipsometry(_base_config(str(tmp_path)))
    text = format_run_report(outcome)
    assert "[run_ellipsometry] summary" in text
    assert "quality flags" in text
    figure_path = tmp_path / "run.png"
    plot_run(outcome, show=False, save_path=str(figure_path))
    assert figure_path.exists() and figure_path.stat().st_size > 0


def test_driver_main_runs_in_simulation_mode_and_reports_success():
    assert thz_ellipsometry_run_me.main(["--simulate", "hr_silicon", "--no-graph"]) == 0


def test_driver_main_reports_a_missing_directory_instead_of_crashing(tmp_path):
    assert thz_ellipsometry_run_me.main(
        ["--directory", str(tmp_path / "absent"), "--no-graph"]) == 2
