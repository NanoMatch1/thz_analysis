"""The diagnostics registry: single source of truth, and never fatal.

Two properties are worth pinning hard, because losing either brings back the failure
mode the module exists to prevent:

  1. the ledger document is DERIVED from the registry, so a documented assumption and
     a checked assumption cannot drift apart;
  2. a diagnostic never halts a run — including a diagnostic that is itself broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset_core.adapters import diagnostics
from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters.diagnostics import registry as registry_module
from dataset_core.data_structures.thz import BaseTHzData, THzData


@pytest.fixture
def isolated_registry():
    """Swap in an empty registry so tests can register without polluting the real one."""
    original = dict(registry_module.DIAGNOSTIC_REGISTRY)
    registry_module.DIAGNOSTIC_REGISTRY.clear()
    yield registry_module.DIAGNOSTIC_REGISTRY
    registry_module.DIAGNOSTIC_REGISTRY.clear()
    registry_module.DIAGNOSTIC_REGISTRY.update(original)


class FakeDataSet:
    def __init__(self, objects=None, config=None):
        self.data = objects or {}
        self.config = config or {}


def make_thz_data(filename="sample_test.acc", n_scans=12, seed=0):
    rng = np.random.default_rng(seed)
    time_ps = np.arange(120) * 0.05 + 100.0
    argument = (time_ps - (time_ps[0] + 0.3 * (time_ps[-1] - time_ps[0]))) / 0.2
    pulse = -argument * np.exp(-0.5 * argument ** 2)
    scans = [
        BaseTHzData(data=np.column_stack((time_ps, pulse * (1 + 0.01 * index)
                                          + rng.standard_normal(120) * 1e-3)),
                    headers=[f"scan {index}"])
        for index in range(n_scans)
    ]
    return THzData(data=scans, header=None, filename=filename, data_type="acc")


# ─────────────────────────────────────────────────────────────────────────────
# the registry is the single source of truth
# ─────────────────────────────────────────────────────────────────────────────


def test_registering_a_diagnostic_needs_exactly_one_declaration(isolated_registry):
    @diagnostics.diagnostic(
        stage="demo", assumption="the trace is not empty",
        why="an empty trace produces numbers that look like zeros",
        remedy="check the loader",
    )
    def trace_is_not_empty(dataset, config):
        return []

    assert list(isolated_registry) == ["trace_is_not_empty"]
    entry = isolated_registry["trace_is_not_empty"]
    assert entry.stage == "demo"
    assert entry.severity is diagnostics.Severity.WARN
    # The same declaration drives the document...
    ledger = diagnostics.render_assumptions_ledger()
    assert "the trace is not empty" in ledger
    assert "an empty trace produces numbers that look like zeros" in ledger
    assert "check the loader" in ledger
    assert "`demo`" in ledger


def test_the_ledger_covers_every_registered_assumption():
    """No entry can exist in the registry without appearing in the document."""
    ledger = diagnostics.render_assumptions_ledger()
    for entry in diagnostics.DIAGNOSTIC_REGISTRY.values():
        assert entry.assumption in ledger, f"{entry.name} missing from the ledger"
        assert entry.remedy in ledger
    assert "Do not edit by hand" in ledger


def test_a_check_cannot_describe_itself_inconsistently(isolated_registry):
    """Findings take their metadata from the registration, not from the check."""
    @diagnostics.diagnostic(
        stage="demo", assumption="registered assumption text",
        why="because", remedy="registered remedy text",
        severity=diagnostics.Severity.FAIL,
    )
    def always_complains(dataset, config):
        yield diagnostics.Problem(message="something is off", filename="a.acc")

    findings = diagnostics.run_diagnostics(FakeDataSet())
    assert len(findings) == 1
    finding = findings[0]
    assert finding.assumption == "registered assumption text"
    assert finding.remedy == "registered remedy text"
    assert finding.severity is diagnostics.Severity.FAIL
    assert finding.diagnostic == "always_complains"


# ─────────────────────────────────────────────────────────────────────────────
# never fatal
# ─────────────────────────────────────────────────────────────────────────────


def test_a_broken_diagnostic_is_reported_and_does_not_halt(isolated_registry):
    @diagnostics.diagnostic(stage="demo", assumption="a", why="b", remedy="c")
    def explodes(dataset, config):
        raise RuntimeError("this check is broken")

    @diagnostics.diagnostic(stage="demo", assumption="d", why="e", remedy="f")
    def works_fine(dataset, config):
        yield diagnostics.Problem(message="a real finding")

    findings = diagnostics.run_diagnostics(FakeDataSet())
    names = {finding.diagnostic for finding in findings}
    assert names == {"explodes", "works_fine"}

    broken = next(f for f in findings if f.diagnostic == "explodes")
    assert broken.severity is diagnostics.Severity.FAIL
    assert "RuntimeError" in broken.message
    assert "says nothing about your data" in broken.remedy


def test_findings_are_sorted_worst_first(isolated_registry):
    for severity, name in ((diagnostics.Severity.INFO, "quiet"),
                           (diagnostics.Severity.FAIL, "loud"),
                           (diagnostics.Severity.WARN, "middling")):
        def make(severity=severity):
            def check(dataset, config):
                yield diagnostics.Problem(message="x")
            return check
        diagnostics.diagnostic(stage="demo", assumption=name, why="w", remedy="r",
                               severity=severity, name=name)(make())

    findings = diagnostics.run_diagnostics(FakeDataSet())
    assert [f.severity for f in findings] == [diagnostics.Severity.FAIL,
                                              diagnostics.Severity.WARN,
                                              diagnostics.Severity.INFO]


def test_stage_filter_runs_only_that_stage(isolated_registry):
    diagnostics.diagnostic(stage="early", assumption="a", why="w", remedy="r",
                           name="early_check")(lambda dataset, config: [
                               diagnostics.Problem(message="early")])
    diagnostics.diagnostic(stage="late", assumption="b", why="w", remedy="r",
                           name="late_check")(lambda dataset, config: [
                               diagnostics.Problem(message="late")])

    findings = diagnostics.run_diagnostics(FakeDataSet(), stage="late")
    assert [finding.diagnostic for finding in findings] == ["late_check"]


# ─────────────────────────────────────────────────────────────────────────────
# the run log
# ─────────────────────────────────────────────────────────────────────────────


def test_run_log_collects_notes_and_findings(isolated_registry):
    diagnostics.diagnostic(stage="demo", assumption="a", why="w", remedy="r",
                           name="noisy")(lambda dataset, config: [
                               diagnostics.Problem(message="found something")])
    dataset = FakeDataSet()
    log = diagnostics.get_run_log(dataset)
    log.note("window", "applied a 201-sample hann window", filename="a.acc")

    diagnostics.check(dataset)
    assert len(log.notes) == 1
    assert len(log.findings) == 1
    assert log.worst is diagnostics.Severity.WARN


def test_run_report_says_nothing_was_halted(isolated_registry, capsys):
    diagnostics.diagnostic(stage="demo", assumption="a", why="w", remedy="do the thing",
                           severity=diagnostics.Severity.FAIL,
                           name="broken_thing")(lambda dataset, config: [
                               diagnostics.Problem(message="it broke", filename="a.acc")])
    diagnostics.run_report(FakeDataSet())
    output = capsys.readouterr().out
    assert "RUN REPORT" in output
    assert "Nothing was halted." in output
    assert "do the thing" in output
    assert "a.acc" in output


def test_a_clean_run_does_not_claim_the_measurement_is_good(isolated_registry, capsys):
    diagnostics.run_report(FakeDataSet())
    output = capsys.readouterr().out
    assert "No assumptions were found broken" in output
    assert "not a guarantee of a" in output


def test_run_log_serialises_to_plain_data(isolated_registry):
    diagnostics.diagnostic(stage="demo", assumption="a", why="w", remedy="r",
                           name="noisy")(lambda dataset, config: [
                               diagnostics.Problem(message="m", detail={"value": 3})])
    dataset = FakeDataSet()
    diagnostics.check(dataset)
    payload = diagnostics.get_run_log(dataset).to_dict()
    assert payload["findings"][0]["severity"] == "warn"
    assert payload["findings"][0]["detail"] == {"value": 3}


# ─────────────────────────────────────────────────────────────────────────────
# the built-in checks, against real pipeline state
# ─────────────────────────────────────────────────────────────────────────────


def test_lost_scan_data_is_reported_as_broken():
    data_obj = make_thz_data()
    dataset = FakeDataSet({data_obj.filename: data_obj})
    thz._ensure_scan_matrix(data_obj)
    data_obj.data = np.asarray(data_obj.data)[:-10]      # a step that drops rows
    thz._ensure_scan_matrix(data_obj)

    findings = diagnostics.run_diagnostics(dataset, stage="preprocessing")
    names = [finding.diagnostic for finding in findings]
    assert "per_scan_data_survived" in names
    finding = next(f for f in findings if f.diagnostic == "per_scan_data_survived")
    assert finding.severity is diagnostics.Severity.FAIL


def test_intact_scan_data_raises_nothing():
    data_obj = make_thz_data()
    dataset = FakeDataSet({data_obj.filename: data_obj})
    thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)
    findings = diagnostics.run_diagnostics(dataset, stage="preprocessing")
    assert findings == []


def test_too_few_repeats_is_flagged():
    data_obj = make_thz_data(n_scans=4)
    dataset = FakeDataSet({data_obj.filename: data_obj})
    thz._ensure_scan_matrix(data_obj)
    findings = diagnostics.run_diagnostics(dataset, stage="noise")
    finding = next(f for f in findings if f.diagnostic == "enough_repeats_for_noise")
    assert "only 4 repeats" in finding.message


def test_clipped_window_is_flagged():
    data_obj = make_thz_data()
    dataset = FakeDataSet({data_obj.filename: data_obj},
                          config={"window": {"type": "hann", "alpha": 1.0}})
    thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)
    thz.window_single_pulse_fixed_width(dataset, half_width_ps=50.0, show_graph=False)

    findings = diagnostics.run_diagnostics(dataset, stage="window")
    assert any(f.diagnostic == "window_fits_the_record" for f in findings)


def test_oversampling_is_noted_but_only_as_info():
    dataset = FakeDataSet(config={"resolution": {"decimation_factor": 15,
                                                 "df_resolution_hz": 1.5e11}})
    findings = diagnostics.run_diagnostics(dataset, stage="resolution")
    finding = next(f for f in findings if f.diagnostic == "spectra_are_oversampled")
    assert finding.severity is diagnostics.Severity.INFO
    assert "15x oversampled" in finding.message


def test_applied_resolution_clears_the_oversampling_note():
    dataset = FakeDataSet(config={"resolution": {"decimation_factor": 15,
                                                 "resolution_applied": True}})
    findings = diagnostics.run_diagnostics(dataset, stage="resolution")
    assert findings == []


def test_purge_check_fires_on_a_frequency_dependent_drift():
    """Water removal lifts the high band more than the low band; laser drift does not."""
    rng = np.random.default_rng(3)
    n_scans, n_samples = 24, 160
    time_ps = np.arange(n_samples) * 0.05 + 100.0
    argument = (time_ps - (time_ps[0] + 0.3 * (time_ps[-1] - time_ps[0]))) / 0.2
    pulse = -argument * np.exp(-0.5 * argument ** 2)

    def dataset_with(gain_is_frequency_dependent):
        scans = []
        for index in range(n_scans):
            fraction = index / (n_scans - 1)
            trace = pulse.copy()
            if gain_is_frequency_dependent:
                # Sharpen the pulse as the run proceeds: narrower in time is broader in
                # frequency, so the high band gains more than the low — water's signature.
                narrowed = (time_ps - (time_ps[0] + 0.3 * (time_ps[-1] - time_ps[0]))) \
                    / (0.2 * (1 - 0.15 * fraction))
                trace = -narrowed * np.exp(-0.5 * narrowed ** 2)
            else:
                trace = pulse * (1 + 0.15 * fraction)      # uniform: laser power drift
            scans.append(BaseTHzData(
                data=np.column_stack((time_ps,
                                      trace + rng.standard_normal(n_samples) * 1e-4)),
                headers=[f"scan {index}"]))
        obj = THzData(data=scans, header=None, filename="sample_test.acc",
                      data_type="acc")
        dataset = FakeDataSet({obj.filename: obj})
        thz._ensure_scan_matrix(obj)
        return dataset

    def fired(dataset):
        findings = diagnostics.run_diagnostics(dataset, stage="acquisition")
        return any(f.diagnostic == "purge_still_equilibrating" for f in findings)

    assert fired(dataset_with(True)), "frequency-dependent drift should be flagged"
    assert not fired(dataset_with(False)), (
        "a uniform gain is laser drift, not water, and must not be flagged as purge")
