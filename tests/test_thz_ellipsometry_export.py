"""Saving a run at the end: notes collected safely, a catalogue-readable bundle, provenance.

Covers the generic notes helper (dataset_core.adapters.run_notes), the ellipsometry bundle
writer and loader, catalogue indexing and annotation, and the run_me save path -- all headless.
"""

from __future__ import annotations

import copy
import io
import json
import os

import numpy as np
import pytest

import thz_ellipsometry_run_me
from dataset_core.adapters.catalog import Catalog
from dataset_core.adapters.run_notes import collect_run_notes
from thz_ellipsometry.adapters.export import load_saved_run, save_run
from thz_ellipsometry.adapters.stages import run_ellipsometry


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def _answers(*lines):
    queue = list(lines)

    def answer(prompt=""):
        if not queue:
            raise EOFError
        return queue.pop(0)
    return answer


# ---------------------------------------------------------------------------
# Notes helper
# ---------------------------------------------------------------------------

def test_notes_are_asked_on_a_terminal_and_kept_multi_line():
    notes = collect_run_notes("summary", input_stream=_Terminal(), environment={},
                              input_function=_answers("first line", "second line", ""),
                              output_function=lambda *_: None)
    assert notes.text == "first line\nsecond line"
    assert notes.source == "prompt" and notes.prefilled_summary == "summary"


def test_no_terminal_never_prompts_and_falls_back_to_config_notes():
    def refuse(prompt=""):
        raise AssertionError("must not prompt without a terminal")
    notes = collect_run_notes("summary", config_notes="from config", input_stream=io.StringIO(),
                              environment={}, input_function=refuse)
    assert (notes.text, notes.source) == ("from config", "config")


def test_environment_overrides_the_mode_and_skip_and_keep_work():
    assert collect_run_notes("s", input_stream=_Terminal(),
                             environment={"THZ_NOTES_MODE": "none"}).source == "none"
    skipped = collect_run_notes("s", input_stream=_Terminal(), environment={},
                                input_function=_answers("-"), output_function=lambda *_: None)
    assert (skipped.text, skipped.source) == ("", "none")
    kept = collect_run_notes("s", config_notes="old", input_stream=_Terminal(), environment={},
                             input_function=_answers(""), output_function=lambda *_: None)
    assert (kept.text, kept.source) == ("old", "config")
    with pytest.raises(ValueError, match="unknown notes mode"):
        collect_run_notes("s", mode="sometimes", environment={})


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------

@pytest.fixture
def finished_run(tmp_path):
    data = tmp_path / "data"
    thz_ellipsometry_run_me.build_simulated_dataset(
        str(data), "doped_silicon", incidence_angle_deg=45.0,
        emitter_angles_deg=(0.0, 90.0, 180.0, 270.0, 270.0, 180.0, 90.0, 0.0), cycles=1)
    config = copy.deepcopy(thz_ellipsometry_run_me.config)
    config["data"]["directory"] = str(data)
    config["general"]["show_graph"] = False
    config["acquisition"]["rows"] = "acquisition"     # staircase synthetic drift
    config["validation"].update(expect=None, cross_check_material=None,
                                branch_reference_index=None)
    return run_ellipsometry(config), data


def test_saved_bundle_round_trips_and_carries_its_provenance(finished_run, monkeypatch, tmp_path):
    outcome, data = finished_run
    monkeypatch.setenv("THZ_CATALOG_ROOT", str(tmp_path))
    bundle = save_run(outcome, notes_mode="config", config_notes="doped wafer, bench day 1",
                      producer="test", output_function=lambda *_: None)
    assert bundle.endswith(".thzbundle") and os.path.dirname(bundle) == str(data)
    assert "doped_silicon" in os.path.basename(bundle)
    for name in ("recipe.json", "report.md", "results.csv", "arrays.npz", "figure.png"):
        assert os.path.exists(os.path.join(bundle, name)), name

    saved = load_saved_run(bundle)
    band = outcome.result.inversion
    assert np.allclose(saved.results["n"], band.refractive_index, rtol=1e-8)
    assert np.allclose(saved.results["k"], band.extinction, rtol=1e-8, atol=1e-12)
    assert np.allclose(saved.arrays["ratio"], outcome.result.ratio)

    recipe = saved.recipe
    assert recipe["producer"] == "test" and recipe["notes"] == "doped wafer, bench day 1"
    assert recipe["notes_record"]["source"] == "config"
    assert recipe["code_versions"]["thz_analysis"]["commit"]
    inputs = recipe["inputs"]
    assert len(inputs) == sum(len(series) for series in outcome.series.values())
    assert all(len(entry["sha256"]) == 64 for entry in inputs)
    json.dumps(recipe)          # fully serialisable, complex config values included


def test_the_bundle_is_indexed_findable_and_annotatable(finished_run, monkeypatch, tmp_path):
    outcome, _ = finished_run
    monkeypatch.setenv("THZ_CATALOG_ROOT", str(tmp_path))
    bundle = save_run(outcome, notes_mode="none", producer="test",
                      output_function=lambda *_: None)
    catalog = Catalog(root=str(tmp_path))
    records = catalog.find(measurement_type="ellipsometry")
    assert len(records) == 1
    record = records[0]
    assert record.producer == "test" and record.bundle_kind == "thz_ellipsometry_run"
    assert set(record.quantities) >= {"n", "k"}

    updated = catalog.annotate(record.bundle_id, "properly described afterwards")
    assert updated.notes == "properly described afterwards"
    recipe = json.load(open(os.path.join(bundle, "recipe.json"), encoding="utf-8"))
    assert recipe["notes_history"][0]["previous"] == ""
    with pytest.raises(ValueError, match="report.md"):
        catalog.open(record.bundle_id)


def test_findings_reach_the_catalogue_as_flags(tmp_path, monkeypatch):
    data = tmp_path / "data"
    thz_ellipsometry_run_me.build_simulated_dataset(str(data), "doped_silicon",
                                                   incidence_angle_deg=45.0,
                                                   background_relative=4.0)
    config = copy.deepcopy(thz_ellipsometry_run_me.config)
    config["data"]["directory"] = str(data)
    config["general"]["show_graph"] = False
    config["acquisition"]["rows"] = "acquisition"     # staircase synthetic drift
    config["validation"].update(expect=None, cross_check_material=None,
                                branch_reference_index=None)
    outcome = run_ellipsometry(config)
    monkeypatch.setenv("THZ_CATALOG_ROOT", str(tmp_path))
    save_run(outcome, notes_mode="none", output_function=lambda *_: None, figure=False)
    record = Catalog(root=str(tmp_path)).find(measurement_type="ellipsometry")[0]
    assert "background_is_small" in record.flags_raised


def test_driver_saves_with_notes_from_the_command_line(monkeypatch, tmp_path):
    monkeypatch.setenv("THZ_CATALOG_ROOT", str(tmp_path))
    monkeypatch.setattr("tempfile.mkdtemp", lambda prefix="": str(tmp_path / "simulated"))
    os.makedirs(tmp_path / "simulated")
    assert thz_ellipsometry_run_me.main(["--simulate", "hr_silicon", "--no-graph",
                                         "--notes", "headless smoke run"]) == 0
    bundles = [name for name in os.listdir(tmp_path / "simulated")
               if name.endswith(".thzbundle")]
    assert len(bundles) == 1
    recipe = json.load(open(tmp_path / "simulated" / bundles[0] / "recipe.json",
                            encoding="utf-8"))
    assert recipe["notes"] == "headless smoke run"
    assert recipe["producer"] == "thz_ellipsometry_run_me.py"
