"""Tests for the knife-edge diagnostic: unit tests on the pure functions with known answers, and a
workflow test that writes real ``.acc`` files and drives ``diagnostics/knife_edge_run_me.py``."""

from __future__ import annotations

import os

import numpy as np
import pytest

from acquisition_editor import save_acc
from diagnostics import knife_edge, knife_edge_run_me

PLANTED_RADIUS_MM = 2.5
PLANTED_CENTRE_MM = 6.0
TIME_PS = 100.0 + 0.05 * np.arange(240)


def _pulse(time_ps, centre_ps=104.0, width_ps=0.25):
    """Single-cycle THz-like pulse: derivative of a Gaussian."""
    offset = time_ps - centre_ps
    return -offset / width_ps * np.exp(-0.5 * (offset / width_ps) ** 2)


def _transmitted_fraction(position_mm):
    """Blade entering the beam as position increases: 1 at small position, 0 at large."""
    return knife_edge.edge_model(position_mm, PLANTED_CENTRE_MM, PLANTED_RADIUS_MM, -1.0, 1.0)


def _write_scan_series(directory, positions_mm, *, scans_per_file=4, noise=1e-4, seed=0,
                       extra_names=()):
    generator = np.random.default_rng(seed)
    for position_mm in positions_mm:
        clean = _transmitted_fraction(position_mm) * _pulse(TIME_PS)
        scans = [clean + noise * generator.normal(size=TIME_PS.size)
                 for _ in range(scans_per_file)]
        name = f"sample_knife-edge_{position_mm:g}mm"
        save_acc({"data": np.column_stack([TIME_PS, *scans]),
                  "header": [f"title {name} acc 1", "param Date and time,2026-10-06 10:00:00"]},
                 os.path.join(directory, f"{name}.acc"))
    for name in extra_names:
        save_acc({"data": np.column_stack([TIME_PS, _pulse(TIME_PS)]),
                  "header": [f"title {name} acc 1"]}, os.path.join(directory, f"{name}.acc"))


# ---------------------------------------------------------------------------
# Position parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected", [
    ("sample_si-HR_knife-edge_7mm.acc", 7.0),
    ("sample_si-HR_knife-edge_0mm_purging.acc", 0.0),
    ("blade_12.5mm_x.acc", 12.5),
    ("blade_-1.5mm.acc", -1.5),
    ("reference_gold.acc", None),
    ("blade_10mmx.acc", None),         # the token must end at a non-alphanumeric character
])
def test_position_from_filename(filename, expected):
    assert knife_edge.position_from_filename(filename) == expected


def test_two_position_tokens_are_ambiguous():
    with pytest.raises(ValueError, match="ambiguous"):
        knife_edge.position_from_filename("blade_3mm_aperture_5mm.acc")


# ---------------------------------------------------------------------------
# Edge fit
# ---------------------------------------------------------------------------

def test_fit_recovers_planted_radius_and_centre():
    positions = np.linspace(0.0, 12.0, 25)
    values = 0.8 * _transmitted_fraction(positions) + 0.05
    edge_fit = knife_edge.fit_knife_edge(positions, values, np.full(positions.size, 1e-3))
    assert edge_fit.beam_radius_mm == pytest.approx(PLANTED_RADIUS_MM, rel=1e-4)
    assert edge_fit.edge_centre_mm == pytest.approx(PLANTED_CENTRE_MM, abs=1e-4)
    assert edge_fit.step_height == pytest.approx(-0.8, rel=1e-4)
    assert edge_fit.covered_fraction > 0.99
    assert edge_fit.trusted


def test_fit_flags_a_scan_that_stops_mid_edge():
    positions = np.linspace(0.0, PLANTED_CENTRE_MM + 0.5, 12)
    edge_fit = knife_edge.fit_knife_edge(positions, _transmitted_fraction(positions))
    assert edge_fit.covered_fraction < 0.8
    assert any("covers only" in warning for warning in edge_fit.warnings)


def test_fit_inflates_errors_and_warns_when_the_erf_misses_the_points():
    positions = np.linspace(0.0, 12.0, 25)
    ripple = 0.05 * np.sin(3.0 * positions)          # edge-diffraction-like, far above noise
    values = _transmitted_fraction(positions) + ripple
    noise_only = knife_edge.fit_knife_edge(positions, _transmitted_fraction(positions),
                                           np.full(positions.size, 1e-3))
    rippled = knife_edge.fit_knife_edge(positions, values, np.full(positions.size, 1e-3))
    assert rippled.reduced_chi_square > 10.0
    assert rippled.beam_radius_error_mm > 10.0 * noise_only.beam_radius_error_mm
    assert any("misses the points" in warning for warning in rippled.warnings)


# ---------------------------------------------------------------------------
# Spectral table
# ---------------------------------------------------------------------------

def test_table_amplitude_follows_the_transmitted_fraction_at_every_frequency():
    positions = np.arange(0.0, 13.0, 1.0)
    files = [knife_edge.KnifeEdgeFile(f"blade_{position:g}mm.acc", position, TIME_PS,
                                      np.atleast_2d(_transmitted_fraction(position)
                                                    * _pulse(TIME_PS)))
             for position in positions]
    table = knife_edge.build_knife_edge_table(files, (0.3, 1.0, 2.0))
    expected = _transmitted_fraction(positions)
    for column_index in range(3):
        normalised = table.amplitude[:, column_index] / table.amplitude[0, column_index]
        np.testing.assert_allclose(normalised, expected / expected[0], rtol=1e-9)
    np.testing.assert_allclose(table.intensity, table.amplitude ** 2)
    assert np.all(np.isnan(table.amplitude_error))          # one scan per file: no error
    assert table.resolution_thz == pytest.approx(1.0 / (TIME_PS[-1] - TIME_PS[0]))
    columns = table.as_columns()
    assert list(columns)[0] == "position_mm"
    assert len(columns) == 1 + 2 * 2 * 3                    # (value, error) x quantities x freqs


def test_table_rejects_mismatched_time_axes():
    files = [knife_edge.KnifeEdgeFile("a_0mm.acc", 0.0, TIME_PS, np.atleast_2d(_pulse(TIME_PS))),
             knife_edge.KnifeEdgeFile("a_1mm.acc", 1.0, TIME_PS + 1.0,
                                      np.atleast_2d(_pulse(TIME_PS)))]
    with pytest.raises(ValueError, match="different time axis"):
        knife_edge.build_knife_edge_table(files, (1.0,))


# ---------------------------------------------------------------------------
# Workflow: real files through the run_me
# ---------------------------------------------------------------------------

def test_run_me_end_to_end(tmp_path, capsys):
    positions = np.arange(0.0, 13.0, 1.0)
    _write_scan_series(str(tmp_path), positions,
                       extra_names=("sample_knife-edge_0mm_purging", "reference_gold"))

    assert knife_edge_run_me.main([str(tmp_path), "--no-graph"]) == 0
    printed = capsys.readouterr().out
    assert "skipped sample_knife-edge_0mm_purging.acc: excluded by config" in printed
    assert "skipped reference_gold.acc: no '_<pos>mm' token" in printed

    output_directory = tmp_path / "knife_edge_analysis"
    assert (output_directory / "knife_edge.png").is_file()
    written = np.genfromtxt(output_directory / "knife_edge_table.csv", delimiter=",", names=True)
    np.testing.assert_allclose(written["position_mm"], positions)

    table, edge_fits = knife_edge_run_me.run_knife_edge(
        {**knife_edge_run_me.config, "data_directory": str(tmp_path)})
    assert len(edge_fits) == len(knife_edge_run_me.config["frequencies_thz"])
    for edge_fit in edge_fits:
        assert edge_fit.beam_radius_mm == pytest.approx(PLANTED_RADIUS_MM, rel=0.02)
        assert edge_fit.edge_centre_mm == pytest.approx(PLANTED_CENTRE_MM, abs=0.05)


def test_run_me_reports_a_missing_directory(tmp_path):
    assert knife_edge_run_me.main([str(tmp_path / "absent"), "--no-graph"]) == 2
