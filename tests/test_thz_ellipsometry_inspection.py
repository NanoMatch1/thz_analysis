"""Step-by-step inspection: the record, its round trip, the figures, the stepwise observer, the
bundle and the viewer -- driven headlessly as a user would."""

from __future__ import annotations

import copy
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import thz_ellipsometry_run_me
import thz_ellipsometry_view
from thz_ellipsometry.adapters import inspection
from thz_ellipsometry.adapters.export import save_run
from thz_ellipsometry.adapters.stages import CHECKPOINTS, run_ellipsometry


@pytest.fixture(scope="module")
def inspected_run(tmp_path_factory):
    """A simulated HR-Si run (gold + sample, per-scan rows), watched by a stepwise observer."""
    data = tmp_path_factory.mktemp("data")
    thz_ellipsometry_run_me.build_simulated_dataset(
        str(data), "hr_silicon", incidence_angle_deg=45.0,
        emitter_angles_deg=(0.0, 90.0, 180.0, 270.0, 180.0, 90.0, 0.0), cycles=1,
        drift_within_acquisition=True)
    config = copy.deepcopy(thz_ellipsometry_run_me.config)
    config["data"]["directory"] = str(data)
    config["general"]["show_graph"] = False
    config["validation"].update(expect="hr_silicon", cross_check_material=None)
    observer = inspection.StepwiseInspector("all", show=False)
    outcome = run_ellipsometry(config, observer=observer)
    return outcome, observer, data


def test_every_checkpoint_has_a_figure_and_the_observer_saw_them_in_order(inspected_run):
    _, observer, _ = inspected_run
    for checkpoint in CHECKPOINTS:
        assert inspection.figures_at(checkpoint), checkpoint
    seen = [name.split("__")[0] for name in observer.shown]
    assert seen == sorted(seen, key=CHECKPOINTS.index)
    assert set(seen) == set(CHECKPOINTS)


def test_the_record_round_trips_through_its_file(inspected_run, tmp_path):
    outcome, _, _ = inspected_run
    record = inspection.record_from_outcome(outcome)
    back = inspection.load_inspection(inspection.save_inspection(record,
                                                                 str(tmp_path / "i.npz")))
    assert [entry.key for entry in back.series] == [entry.key for entry in record.series]
    assert back.series[0].role == "channel_reference"
    for original, restored in zip(record.series, back.series):
        np.testing.assert_array_equal(restored.row_traces, original.row_traces)
        np.testing.assert_allclose(restored.predicted_spectra, original.predicted_spectra)
        assert restored.window_half_width_samples == original.window_half_width_samples
        assert restored.drift_model == original.drift_model
        np.testing.assert_array_equal(restored.nuisance_names, original.nuisance_names)
    np.testing.assert_allclose(back.index, record.index)
    assert back.calibration_applied == pytest.approx(record.calibration_applied)
    assert set(back.calibration_by_material) == {"gold reference", "sample as hr_silicon"}
    assert back.reference_material == "hr_silicon" and "summary" in back.run_report


def test_the_model_meets_the_rows_at_the_noise(inspected_run):
    """The residual the 4b figure plots is the fit's own: ~noise on simulated data."""
    outcome, _, _ = inspected_run
    entry = inspection.record_from_outcome(outcome).series[1]
    band = outcome.result.band
    normalised = (np.abs(entry.row_spectra[:, band] - entry.predicted_spectra[:, band]) ** 2
                  / entry.row_variance[:, band])
    assert 0.3 < normalised.mean() < 3.0


def test_unknown_checkpoints_are_refused():
    with pytest.raises(ValueError, match="unknown inspection checkpoint"):
        inspection.resolve_checkpoints(["windowing", "nope"])
    assert inspection.resolve_checkpoints(None) == ()
    assert inspection.resolve_checkpoints("all") == CHECKPOINTS


def test_the_bundle_keeps_the_steps_and_the_viewer_redraws_them(inspected_run, monkeypatch,
                                                                  tmp_path, capsys):
    outcome, observer, _ = inspected_run
    monkeypatch.setenv("THZ_CATALOG_ROOT", str(tmp_path))
    bundle = save_run(outcome, directory=str(tmp_path), notes_mode="none", producer="test",
                      output_function=lambda *_: None)
    figures = sorted(os.listdir(os.path.join(bundle, "figures")))
    assert figures == sorted(observer.shown)
    assert os.path.isfile(os.path.join(bundle, "inspection.npz"))

    redrawn = tmp_path / "redrawn"
    assert thz_ellipsometry_view.main([bundle, "--step", "windowing", "--step", "result",
                                       "--save", str(redrawn)]) == 0
    assert sorted(os.listdir(redrawn)) == sorted(
        name for name in figures if name.split("__")[0] in ("windowing", "result"))
    assert thz_ellipsometry_view.main([bundle, "--list"]) == 0
    assert "harmonic_drift_and_residuals" in capsys.readouterr().out
    assert thz_ellipsometry_view.main([str(tmp_path / "absent")]) == 2


def test_the_window_sweep_runs_headlessly_on_simulated_data(tmp_path, capsys):
    import thz_ellipsometry_window_sweep
    figure_path = tmp_path / "sweep.png"
    assert thz_ellipsometry_window_sweep.main(
        ["--simulate", "hr_silicon", "--rows", "acquisition", "--half-widths", "2", "3",
         "--save", str(figure_path)]) == 0
    assert figure_path.is_file()
    table = capsys.readouterr().out
    assert "half-width 2 ps" in table and "half-width 3 ps" in table
