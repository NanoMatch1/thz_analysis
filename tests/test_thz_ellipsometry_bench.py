"""Tests for the bench-setup tools: wire-grid null, purge settling, half-wave-plate walk.

Unit tests drive ``thz_ellipsometry.core.bench`` with known answers; workflow tests write real
``.acc`` files and drive ``thz_ellipsometry_bench_run_me.py`` exactly as on the bench.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import thz_ellipsometry_bench_run_me
from thz_ellipsometry.adapters.bench_tools import BENCH_TOOLS, run_bench_tool
from thz_ellipsometry.adapters.synthetic_files import write_accumulation_files
from thz_ellipsometry.core import bench, materials, simulate

FREQUENCIES = np.linspace(0.1e12, 4.0e12, 300)
BAND = (FREQUENCIES > 0.8e12) & (FREQUENCIES < 3.0e12)
PULSE = simulate.default_emitted_spectrum(FREQUENCIES) * np.exp(
    -2j * np.pi * FREQUENCIES * 5e-12)


def _noise(generator, scale):
    return scale * (generator.normal(size=FREQUENCIES.size)
                    + 1j * generator.normal(size=FREQUENCIES.size))


# ---------------------------------------------------------------------------
# Unit: core.bench
# ---------------------------------------------------------------------------

def test_relative_delay_is_exact_and_positive_for_a_later_pulse():
    later = PULSE * np.exp(-2j * np.pi * FREQUENCIES * 17e-15) * 0.95
    delay, amplitude = bench.relative_delay_and_amplitude(later, PULSE, FREQUENCIES, BAND)
    assert delay * 1e15 == pytest.approx(17.0, abs=1e-6)
    assert amplitude == pytest.approx(0.95, rel=1e-9)


@pytest.mark.parametrize("true_null", [86.3, 3.7, 271.2])
def test_wire_grid_null_is_found_through_the_sign_change(true_null):
    generator = np.random.default_rng(1)
    readings = true_null + np.array([-20.0, -8.0, -4.0, -2.0, 0.5, 2.0, 4.0, 8.0, 20.0])
    spectra = np.array([np.sin(np.deg2rad(reading - true_null)) * PULSE + _noise(generator, 0.003)
                        for reading in readings])
    amplitudes = bench.signed_projection(spectra, spectra[0], BAND)
    result = bench.fit_wire_grid_null(readings, amplitudes)
    assert result.null_deg % 360.0 == pytest.approx(true_null % 360.0,
                                                    abs=4 * result.null_standard_error_deg)
    assert result.null_standard_error_deg < 0.5


def test_a_background_shifts_a_lone_null_and_the_reversed_sweep_removes_it():
    """Over +/-20 deg a constant background looks like a moved null; reversing M separates them."""
    readings_p = np.array([66.0, 78.0, 84.0, 86.0, 88.0, 94.0, 106.0])
    readings_minus_p = readings_p + 180.0
    background = 0.02
    sweep_p = np.sin(np.deg2rad(readings_p - 86.0)) + background
    sweep_minus_p = np.sin(np.deg2rad(readings_minus_p - 86.0)) + background
    lone = bench.fit_wire_grid_null(readings_p, sweep_p)
    paired = bench.fit_wire_grid_nulls([(readings_p, sweep_p),
                                        (readings_minus_p, sweep_minus_p)], fit_background=True)
    assert lone.null_deg - 86.0 == pytest.approx(-np.rad2deg(background), rel=0.05)
    assert paired[0].null_deg == pytest.approx(86.0, abs=1e-6)
    assert paired[1].null_deg == pytest.approx(266.0, abs=1e-6)
    assert paired[0].leakage == pytest.approx(background, abs=1e-9)


def test_a_pair_is_fitted_alone_by_default_and_keeps_a_scale_error_visible():
    """Nulls 181 deg apart (scale error), no background: each sweep must land on its own null.

    Regression: the old pair model (own A and null per sweep + shared constant) let the constant
    trade against opposite null shifts -- on bench data it invented 4-11% background and moved
    the nulls 3-7 deg.
    """
    generator = np.random.default_rng(4)
    offsets = np.array([-22.0, -10.0, -6.0, -4.0, -2.0, 0.0, 2.0, 6.0, 18.0])
    first_null, second_null = 156.2, 337.2
    sweeps = [(null + offsets,
               sign * np.sin(np.deg2rad(offsets)) + 0.005 * generator.normal(size=offsets.size))
              for null, sign in ((first_null, 1.0), (second_null, -1.0))]
    first, second = bench.fit_wire_grid_nulls(sweeps)
    assert first.leakage == 0.0 and second.leakage == 0.0
    assert first.null_deg == pytest.approx(first_null, abs=0.2)
    assert second.null_deg == pytest.approx(second_null, abs=0.2)
    assert first.null_standard_error_deg < 0.2


def test_the_shared_sinusoid_forces_the_nulls_180_apart():
    readings = np.array([66.0, 78.0, 84.0, 86.0, 88.0, 94.0, 106.0])
    sweeps = [(readings, np.sin(np.deg2rad(readings - 86.0)) + 0.01),
              (readings + 180.0, np.sin(np.deg2rad(readings + 180.0 - 86.0)) + 0.01)]
    first, second = bench.fit_wire_grid_nulls(sweeps, fit_background=True)
    assert second.null_deg - first.null_deg == pytest.approx(180.0, abs=1e-9)
    assert first.null_standard_error_deg == second.null_standard_error_deg
    np.testing.assert_allclose(first.residuals, 0.0, atol=1e-12)
    np.testing.assert_allclose(second.residuals, 0.0, atol=1e-12)


def test_wire_grid_null_model_reproduces_the_fitted_points():
    readings = np.array([66.0, 78.0, 84.0, 86.0, 88.0, 94.0, 106.0])
    values = -0.7 * np.sin(np.deg2rad(readings - 86.4)) + 0.01
    fit = bench.fit_wire_grid_null(readings, values, fit_background=True)
    np.testing.assert_allclose(fit.model(readings), values, atol=1e-12)
    np.testing.assert_allclose(fit.residuals, 0.0, atol=1e-12)
    assert fit.model([fit.null_deg])[0] == pytest.approx(fit.leakage, abs=1e-12)


def test_wire_grid_null_needs_enough_readings():
    with pytest.raises(ValueError, match="at least four"):
        bench.fit_wire_grid_null([80.0, 90.0, 100.0], [1.0, 0.0, -1.0])


def test_settling_trend_follows_a_decaying_transient_and_calls_it_settled_late():
    minutes = np.arange(12) * 2.0
    delays = 40e-15 * (1.0 - np.exp(-minutes / 5.0))
    spectra = np.array([PULSE * np.exp(-2j * np.pi * FREQUENCIES * delay) for delay in delays])
    early = bench.settling_trend(spectra[:4], minutes[:4] * 60.0, FREQUENCIES, BAND)
    late = bench.settling_trend(spectra, minutes * 60.0, FREQUENCIES, BAND)
    assert np.allclose(late.delays_fs, delays * 1e15, atol=1e-6)
    assert not early.is_settled(1.0)
    assert late.is_settled(1.0)


def _plate_spectra(plate_angles_deg, wedge_per_thz=0.0, wedge_direction_deg=30.0, seed=0):
    """Gold at several HWP angles: a 90-deg polarisation factor and a 360-deg wedge roll-off."""
    generator = np.random.default_rng(seed)
    frequencies_thz = FREQUENCIES / 1e12
    spectra = []
    for plate in plate_angles_deg:
        polarisation = 1.0 + 0.3 * np.cos(np.deg2rad(4.0 * plate))
        walk = wedge_per_thz * (np.cos(np.deg2rad(plate - wedge_direction_deg))
                                - np.cos(np.deg2rad(-wedge_direction_deg)))
        spectra.append(polarisation * np.exp(-walk * frequencies_thz) * PULSE
                       + _noise(generator, 1e-4))
    return np.array(spectra)


PLATES = np.array([0.0, 22.5, 45.0, 90.0, 180.0, 270.0])


def test_pure_polarisation_shows_no_walk():
    result = bench.analyse_half_wave_plate(PLATES, _plate_spectra(PLATES), FREQUENCIES, BAND)
    assert result.same_state_slope_spread < 1e-3
    assert result.wedge_slope_amplitude < 1e-3
    assert np.allclose(np.abs(result.constant[[0, 3, 4, 5]]), 1.0, atol=1e-3)


def test_a_wedged_plate_is_detected_and_separated_from_polarisation():
    result = bench.analyse_half_wave_plate(PLATES, _plate_spectra(PLATES, wedge_per_thz=0.05),
                                           FREQUENCIES, BAND)
    assert result.same_state_slope_spread > 0.05
    assert result.wedge_slope_amplitude == pytest.approx(0.05, rel=0.05)
    # The roll-off is -w cos(plate - 30): its LEAST-attenuating direction is 30 + 180 deg.
    wrapped_difference = (result.wedge_direction_deg - 210.0 + 180.0) % 360.0 - 180.0
    assert abs(wrapped_difference) < 3.0


# ---------------------------------------------------------------------------
# Workflow: real files through the bench driver
# ---------------------------------------------------------------------------

def test_bench_tools_are_registered_with_help_text():
    assert {"null", "live", "hwp"} <= set(BENCH_TOOLS)
    for entry in BENCH_TOOLS.values():
        assert entry.summary


def test_null_tool_finds_the_magnet_zero_from_files(tmp_path):
    readings = np.array([66.0, 78.0, 82.0, 84.0, 86.0, 88.0, 90.0, 94.0, 106.0])
    true_null = 86.3
    schedule = simulate.AcquisitionSchedule(np.deg2rad(readings - true_null),
                                            np.arange(readings.size) * 60.0,
                                            np.ones(readings.size, dtype=int))
    # Probe at 0 deg with [001] along p: the crystal sees E_s only -- a perfect grid passing s.
    write_accumulation_files(str(tmp_path), index_sample_function=materials.gold_index,
                             schedule=schedule, sample_name="wgp_grid=s", scans_per_angle=3,
                             angle_token="mag", magnet_calibration={0.0: true_null},
                             probe_azimuth_deg=0.0, crystal_orientation_rad=0.0,
                             incidence_angle_rad=np.deg2rad(45.0), relative_noise=0.003)
    results, report = run_bench_tool("null", str(tmp_path), thz_ellipsometry_bench_run_me.config)
    result = results[None]
    assert result.null_deg == pytest.approx(true_null, abs=4 * result.null_standard_error_deg)
    assert "magnet_calibration" in report


def test_null_tool_builds_the_four_state_calibration_table(tmp_path):
    """Grid passing s nulls p and -p, passing p nulls s and -s: four sweeps -> one table."""
    true_table = {0.0: 86.3, 90.0: 176.8, 180.0: 266.0, 270.0: 356.9}
    for polarisation, crystal_deg in ((0.0, 0.0), (180.0, 0.0), (90.0, 90.0), (270.0, 90.0)):
        # The crystal sees only one component: [001] along p at probe 0 sees E_s (a grid
        # passing s); rotated 90 deg it sees E_p (a grid passing p).
        offsets = np.array([-20.0, -8.0, -4.0, -2.0, 2.0, 4.0, 8.0, 20.0])
        schedule = simulate.AcquisitionSchedule(
            np.deg2rad(polarisation + offsets), np.arange(offsets.size) * 60.0,
            np.ones(offsets.size, dtype=int))
        write_accumulation_files(
            str(tmp_path), index_sample_function=materials.gold_index, schedule=schedule,
            sample_name=f"wgp_null={polarisation:g}", scans_per_angle=3, angle_token="mag",
            magnet_calibration=true_table, probe_azimuth_deg=0.0,
            crystal_orientation_rad=np.deg2rad(crystal_deg),
            incidence_angle_rad=np.deg2rad(45.0), relative_noise=0.003)
    results, report = run_bench_tool("null", str(tmp_path), thz_ellipsometry_bench_run_me.config)
    for polarisation, reading in true_table.items():
        fitted = results[polarisation]
        assert fitted.null_deg == pytest.approx(reading,
                                                abs=4 * fitted.null_standard_error_deg + 0.05)
    assert "magnet_calibration'] = {0.0:" in report


def _write_paired_null_sweeps(directory):
    true_table = {0.0: 86.3, 180.0: 266.0}
    offsets = np.array([-20.0, -8.0, -4.0, -2.0, 2.0, 4.0, 8.0, 20.0])
    for polarisation in true_table:
        schedule = simulate.AcquisitionSchedule(
            np.deg2rad(polarisation + offsets), np.arange(offsets.size) * 60.0,
            np.ones(offsets.size, dtype=int))
        write_accumulation_files(
            directory, index_sample_function=materials.gold_index, schedule=schedule,
            sample_name=f"wgp_grid=s_null={polarisation:g}", scans_per_angle=2,
            angle_token="mag", magnet_calibration=true_table, probe_azimuth_deg=0.0,
            crystal_orientation_rad=0.0, incidence_angle_rad=np.deg2rad(45.0),
            relative_noise=0.003)
    return true_table


def test_null_sweeps_carry_both_fits_and_the_traces(tmp_path):
    from thz_ellipsometry.adapters.bench_tools import collect_null_sweeps
    true_table = _write_paired_null_sweeps(str(tmp_path))
    collection = collect_null_sweeps(str(tmp_path), thz_ellipsometry_bench_run_me.config)
    assert [sweep.polarisation for sweep in collection.sweeps] == [0.0, 180.0]
    for sweep in collection.sweeps:
        assert sweep.paired and not sweep.background_fitted
        assert sweep.fit.leakage == 0.0 and sweep.lone_fit.leakage == 0.0
        assert sweep.name == f"wgp_grid=s_null={sweep.polarisation:g}"
        assert sweep.traces.shape == (8, sweep.time_ps.size)
        assert sweep.lone_fit.null_deg == pytest.approx(
            true_table[sweep.polarisation], abs=4 * sweep.lone_fit.null_standard_error_deg + 0.05)
        np.testing.assert_allclose(sweep.readings_deg, sweep.lone_fit.magnet_angles_deg)


def test_null_report_checks_the_180_degree_separation(tmp_path):
    _write_paired_null_sweeps(str(tmp_path))
    _, report = run_bench_tool("null", str(tmp_path), thz_ellipsometry_bench_run_me.config)
    assert "fitted alone, no background" in report
    assert "nulls 0 and 180:" in report and "180 expected" in report
    configuration = {**thz_ellipsometry_bench_run_me.config,
                     "null": {**thz_ellipsometry_bench_run_me.config["null"],
                              "fit_background": True}}
    results, report = run_bench_tool("null", str(tmp_path), configuration)
    assert "one sinusoid" in report and "180 expected" not in report
    assert results[180.0].null_deg - results[0.0].null_deg == pytest.approx(180.0, abs=1e-9)


def test_only_the_null_tool_registers_figures():
    assert BENCH_TOOLS["null"].figures is not None
    assert BENCH_TOOLS["live"].figures is None and BENCH_TOOLS["hwp"].figures is None


def test_bench_driver_saves_the_null_figures_headlessly(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from thz_ellipsometry.adapters.bench_tools import make_bench_figures
    _write_paired_null_sweeps(str(tmp_path))
    assert thz_ellipsometry_bench_run_me.main(["null", str(tmp_path), "--no-graph"]) == 0
    for name in ("null_fits.png", "null_traces.png"):
        assert (tmp_path / "bench_analysis" / name).is_file()
    figures = make_bench_figures("null", str(tmp_path), thz_ellipsometry_bench_run_me.config)
    fits = figures["null_fits.png"]
    assert len([axis for axis in fits.axes if axis.get_xlabel() == "magnet reading [deg]"]) == 2
    import matplotlib.pyplot as plt
    for figure in figures.values():
        plt.close(figure)


def test_live_tool_tracks_settling_per_magnet_state(tmp_path):
    elapsed = np.arange(8) * 130.0
    delays = -40e-15 * np.exp(-elapsed / 300.0)
    schedule = simulate.AcquisitionSchedule(np.zeros(8), elapsed, np.arange(1, 9))
    write_accumulation_files(str(tmp_path), index_sample_function=materials.gold_index,
                             schedule=schedule, sample_name="ref-gold", scans_per_angle=2,
                             angle_token="mag", incidence_angle_rad=np.deg2rad(45.0),
                             relative_noise=0.002, drift_span_s=delays, seconds_per_scan=60.0)
    results, report = run_bench_tool("live", str(tmp_path), thz_ellipsometry_bench_run_me.config)
    trend = results[0.0]
    assert trend.delays_fs[-1] == pytest.approx(-(delays[-1] - delays[0]) * 1e15, abs=2.0)
    assert "SETTLED" in report


def _write_plate_files(directory, plates, wedge_per_thz):
    time_ps = np.arange(512) * 0.05
    frequencies = np.fft.rfftfreq(512, d=0.05e-12)
    offset = (time_ps - 5.0) / 0.11
    pulse = np.fft.rfft(-offset * np.exp(-0.5 * offset**2))
    generator = np.random.default_rng(3)
    for plate in plates:
        walk = wedge_per_thz * (np.cos(np.deg2rad(plate - 30.0)) - np.cos(np.deg2rad(-30.0)))
        trace = np.fft.irfft(pulse * np.exp(-walk * frequencies / 1e12), n=512)
        lines = []
        for scan in range(3):
            if scan:
                lines.append("%%")
            lines += [f"%title hwp-gold_hwp={plate:g}_mag=045 acc {scan + 1}", "%Created 3",
                      "%type 0", "%Parameters Parameters",
                      f"%param Date and time,2026-10-07 10:{int(plate) % 60:02d}:{scan:02d}.000000"]
            noisy = trace + 1e-4 * np.abs(trace).max() * generator.normal(size=512)
            lines += [f"{t:.18e} {value:.18e}" for t, value in zip(time_ps, noisy)]
        with open(os.path.join(directory, f"hwp-gold_hwp={plate:g}_mag=045.acc"), "w") as handle:
            handle.write("\n".join(lines) + "\n")


def test_hwp_tool_reports_walk_from_files(tmp_path):
    _write_plate_files(str(tmp_path), PLATES, wedge_per_thz=0.05)
    result, report = run_bench_tool("hwp", str(tmp_path), thz_ellipsometry_bench_run_me.config)
    assert result.same_state_slope_spread > 0.05
    assert "beam walk" in report


def test_bench_driver_runs_each_tool_headlessly_and_reports_a_bad_directory(tmp_path):
    assert thz_ellipsometry_bench_run_me.main(["null", str(tmp_path / "absent")]) == 2
    _write_plate_files(str(tmp_path), PLATES, wedge_per_thz=0.0)
    assert thz_ellipsometry_bench_run_me.main(["hwp", str(tmp_path)]) == 0
