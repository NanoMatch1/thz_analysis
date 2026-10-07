"""Save an ellipsometry run as a catalogue bundle, with notes asked for at the end.

The bundle is a ``<created>_<series>.thzbundle`` directory written next to the data, in the same
format the catalogue already indexes (``recipe.json`` + ``report.md``), so
``catalog_browse.py --find`` finds it like any other analysis:

    recipe.json   identity, provenance, config, inputs (with SHA-256), notes, summary, findings
    report.md     notes, the run report, the diagnostic findings, and how to load it back
    results.csv   the band: frequency, n, k, their error, eps, sigma_1, rho
    arrays.npz    everything numeric: full-axis rho and its variance, calibration per frequency,
                  every series' fitted P, Q, background, delays, amplitudes, tilts
    figure.png    the run figure

Load it back with :func:`load_saved_run`. Nothing here recomputes anything: the bundle is a
record of what this run produced, with enough provenance (code commits and whether the tree was
dirty, input file hashes, full config) to reproduce it.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from dataset_core.adapters.run_notes import collect_run_notes
from dataset_core.services.provenance import code_versions, file_sha256

from .report import format_run_report, plot_run

__all__ = ["BUNDLE_KIND", "SavedRun", "build_prefilled_summary", "load_saved_run", "save_run"]

BUNDLE_KIND = "thz_ellipsometry_run"
SCHEMA_VERSION = "1.0"
BUNDLE_SUFFIX = ".thzbundle"


def _plain(value):
    """JSON-safe copy of a config/summary value: complex -> {real, imag}, arrays -> lists."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _series_name(outcome, delimiter="_"):
    """The sample's filename without its key=value tokens: doped-si_mag=090_cyc=03 -> doped-si."""
    sample = outcome.sample_series
    if not sample.paths:
        return "sample"
    stem = os.path.splitext(os.path.basename(sample.paths[0]))[0]
    words = [token for token in stem.split(delimiter) if token and "=" not in token]
    return delimiter.join(words) or "sample"


def _summary(outcome):
    result = outcome.result
    inversion = result.inversion
    summary = {
        "median_n": float(np.median(inversion.refractive_index)),
        "median_k": float(np.median(inversion.extinction)),
        "median_index_standard_error": (None if result.index_standard_error is None
                                        else float(np.median(result.index_standard_error))),
        "band_thz": [float(inversion.frequencies_hz.min() / 1e12),
                     float(inversion.frequencies_hz.max() / 1e12)],
        "incidence_angle_deg": float(np.rad2deg(inversion.incidence_angle_rad)),
        "incidence_angle_source": result.incidence_angle_source,
        "channel_ratio": result.calibration.ratio,
        "calibration_source": outcome.calibration_source,
        "quality_flags": result.quality_flags,
    }
    if result.tilt_fit is not None:
        summary["tilt_fit"] = {"tilt_deg": result.tilt_fit.tilt_deg,
                               "tilt_standard_error_deg": result.tilt_fit.tilt_standard_error_deg,
                               "model": result.tilt_fit.model_name,
                               "parameters": result.tilt_fit.model_parameters}
    return summary


def build_prefilled_summary(outcome):
    """What the run knows about itself, shown at the notes prompt and stored with the notes."""
    lines = [f"script      : {os.path.basename(sys.argv[0]) or 'interactive'}",
             f"data        : {outcome.config['data']['directory']}"]
    for (role, probe), series in sorted(outcome.series.items(), key=lambda item: str(item[0])):
        lines.append(f"{role:<12}: {len(series)} acquisitions at probe {probe} deg, "
                     f"angles {sorted(set(np.round(series.angles_deg % 360.0, 1)))}")
    summary = _summary(outcome)
    error = summary["median_index_standard_error"]
    lines.append(f"result      : n = {summary['median_n']:.4f}, k = {summary['median_k']:.4f}"
                 + ("" if error is None else f" (+/- {error:.4f})")
                 + f" over {summary['band_thz'][0]:.2f}-{summary['band_thz'][1]:.2f} THz")
    lines.append(f"calibration : {summary['calibration_source']}; angle "
                 f"{summary['incidence_angle_deg']:.2f} deg ({summary['incidence_angle_source']})")
    findings = outcome.findings or []
    lines.append("findings    : " + (", ".join(sorted({finding.diagnostic for finding in findings}))
                                     if findings else "none"))
    return "\n".join(lines)


def _inputs(outcome):
    entries = []
    for (role, probe), series in outcome.series.items():
        for path, angle, scans in zip(series.paths, series.angles_deg, series.scan_counts):
            entries.append({"path": os.path.abspath(path), "role": role,
                            "probe_azimuth_deg": probe, "polarization_angle_deg": float(angle),
                            "scans": int(scans), "sha256": file_sha256(path)})
    return entries


def _findings(outcome):
    return [{"diagnostic": finding.diagnostic, "stage": finding.stage,
             "severity": finding.severity.label, "message": finding.message,
             "remedy": finding.remedy} for finding in (outcome.findings or [])]


def _arrays(outcome):
    result = outcome.result
    arrays = {
        "frequencies_hz": outcome.frequencies_hz,
        "band": result.band,
        "ratio": result.ratio,
        "calibration_ratio_per_frequency": result.calibration.ratio_per_frequency,
        "band_frequencies_hz": result.inversion.frequencies_hz,
        "index": result.inversion.index,
    }
    if result.ratio_variance is not None:
        arrays["ratio_variance"] = result.ratio_variance
    if result.index_standard_error is not None:
        arrays["index_standard_error"] = result.index_standard_error
    for (role, probe), fit in outcome.fits.items():
        prefix = f"{role}__probe_{probe}__"
        arrays[prefix + "channel_p"] = fit.channel_p
        arrays[prefix + "channel_s"] = fit.channel_s
        arrays[prefix + "delays_s"] = fit.delays_s
        arrays[prefix + "angles_rad"] = fit.emitter_angles_rad
        if fit.background is not None:
            arrays[prefix + "background"] = fit.background
        if fit.amplitudes is not None:
            arrays[prefix + "amplitudes_at_1thz"] = fit.amplitudes
        if fit.amplitude_tilts_per_thz is not None:
            arrays[prefix + "tilts_per_thz"] = fit.amplitude_tilts_per_thz
        if fit.elapsed_seconds is not None:
            arrays[prefix + "elapsed_seconds"] = fit.elapsed_seconds
        if fit.segment_ids is not None:
            arrays[prefix + "segment_ids"] = fit.segment_ids
        if fit.nuisance_parameters is not None:
            arrays[prefix + "nuisance_parameters"] = fit.nuisance_parameters
            arrays[prefix + "nuisance_parameter_errors"] = fit.nuisance_parameter_errors
            arrays[prefix + "nuisance_parameter_names"] = np.array(fit.nuisance_parameter_names)
    return arrays


RESULT_COLUMNS = ("frequency_thz", "n", "k", "index_standard_error", "eps_real", "eps_imag",
                  "sigma1_s_per_m", "rho_real", "rho_imag")


def _write_results_csv(path, outcome):
    result = outcome.result
    inversion = result.inversion
    permittivity = inversion.permittivity
    error = (result.index_standard_error if result.index_standard_error is not None
             else np.full(inversion.index.shape, np.nan))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULT_COLUMNS)
        for row in zip(inversion.frequencies_hz / 1e12, inversion.refractive_index,
                       inversion.extinction, error, permittivity.real, permittivity.imag,
                       inversion.conductivity_real_si, inversion.ratio.real,
                       inversion.ratio.imag):
            writer.writerow([f"{value:.10g}" for value in row])


def _report_markdown(outcome, notes, bundle_name):
    lines = [f"# {bundle_name}", "", "## Notes", "",
             notes.text or "_(none given)_", "", f"*Source of the notes: {notes.source}.*", "",
             "## What the run knew", "", "```", notes.prefilled_summary, "```", "",
             "## Run report", "", "```", format_run_report(outcome), "```", ""]
    findings = outcome.findings or []
    lines += ["## Diagnostic findings", ""]
    if findings:
        # 'flag **name**' is the pattern the catalogue indexes as flags_raised.
        lines += [f"- {finding.severity.label}: flag **{finding.diagnostic}** — {finding.message}"
                  for finding in findings]
    else:
        lines.append("None of the registered assumptions was found broken.")
    lines += ["", "## Load it back", "", "```python",
              "from thz_ellipsometry.adapters.export import load_saved_run",
              f"run = load_saved_run(r\"{bundle_name}\")   # path to this bundle directory",
              "run.results['n'], run.results['k']     # band arrays; run.arrays has the rest",
              "```", ""]
    return "\n".join(lines)


def save_run(outcome, *, directory=None, notes_mode="auto", config_notes="", figure=True,
             producer=None, input_function=input, output_function=print, input_stream=None):
    """Write the bundle for a finished run; ask for notes first (see ``run_notes``).

    ``directory`` defaults to the run's data directory, so results sit beside the raw files.
    Returns the bundle path.
    """
    notes = collect_run_notes(build_prefilled_summary(outcome), mode=notes_mode,
                              config_notes=config_notes, input_function=input_function,
                              output_function=output_function, input_stream=input_stream)
    created = datetime.now(timezone.utc)
    series_name = _series_name(outcome)
    bundle_name = f"{created.strftime('%Y-%m-%d_%H%M%S')}_{series_name}_ellipsometry{BUNDLE_SUFFIX}"
    target = directory or outcome.config["data"]["directory"]
    bundle_dir = os.path.join(target, bundle_name)
    os.makedirs(bundle_dir, exist_ok=False)

    versions = code_versions()
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "bundle_kind": BUNDLE_KIND,
        "bundle_id": str(uuid.uuid4()),
        "created": created.isoformat(),
        "producer": producer or os.path.basename(sys.argv[0]) or "interactive",
        "series_name": series_name,
        "source_dir": os.path.abspath(outcome.config["data"]["directory"]),
        "git_sha": (versions.get("thz_analysis") or {}).get("commit"),
        "code_versions": versions,
        "measurement_type": "ellipsometry",
        "quantities": ["n", "k", "sigma"],
        "notes": notes.text,
        "notes_record": notes.as_record(),
        "n_files": sum(len(series) for series in outcome.series.values()),
        "metadata": {"mode": outcome.config.get("mode", "isotropic"),
                     "calibration_source": outcome.calibration_source},
        "config": _plain(outcome.config),
        "inputs": _inputs(outcome),
        "summary": _plain(_summary(outcome)),
        "findings": _findings(outcome),
        "validation": (None if outcome.report is None else str(outcome.report)),
        "steps": [],
    }
    with open(os.path.join(bundle_dir, "recipe.json"), "w", encoding="utf-8") as handle:
        json.dump(recipe, handle, indent=2)
    with open(os.path.join(bundle_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(_report_markdown(outcome, notes, bundle_name))
    _write_results_csv(os.path.join(bundle_dir, "results.csv"), outcome)
    np.savez_compressed(os.path.join(bundle_dir, "arrays.npz"), **_arrays(outcome))
    if figure:
        plot_run(outcome, show=False, save_path=os.path.join(bundle_dir, "figure.png"))
    output_function(f"[save_run] wrote {bundle_dir}"
                    + ("" if notes.text else "  (no notes: add later with catalog_browse.py "
                                             "--annotate)"))
    _index_in_catalog(bundle_dir, output_function)
    return bundle_dir


def _index_in_catalog(bundle_dir, output_function):
    """Index the new bundle now if it sits under the catalogue root; otherwise say how to."""
    try:
        from dataset_core.adapters.catalog import Catalog
        catalog = Catalog()
        relative = os.path.relpath(os.path.abspath(bundle_dir), os.path.abspath(catalog.root))
        if relative.startswith(".."):
            output_function(f"[save_run] not under the catalogue root ({catalog.root}); index it "
                            "with: catalog_browse.py --root <its data root> --rebuild")
            return
        record = catalog.update(bundle_dir)
        output_function(f"[save_run] indexed in the catalogue as {record.bundle_id[:8]}")
    except Exception as error:  # noqa: BLE001 -- saving succeeded; indexing is a convenience
        output_function(f"[save_run] saved, but catalogue indexing failed: {error}")


@dataclass(frozen=True)
class SavedRun:
    path: str
    recipe: dict
    results: dict          #: column -> array, from results.csv
    arrays: dict           #: name -> array, from arrays.npz

    @property
    def notes(self):
        return self.recipe.get("notes", "")


def load_saved_run(bundle_dir):
    """Read a saved ellipsometry bundle back: recipe, the band table and every stored array."""
    with open(os.path.join(bundle_dir, "recipe.json"), encoding="utf-8") as handle:
        recipe = json.load(handle)
    if recipe.get("bundle_kind") != BUNDLE_KIND:
        raise ValueError(f"{bundle_dir} is a {recipe.get('bundle_kind', 'session')!r} bundle, not "
                         f"an ellipsometry run")
    with open(os.path.join(bundle_dir, "results.csv"), encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, values = rows[0], np.array(rows[1:], dtype=float)
    results = {name: values[:, column] for column, name in enumerate(header)}
    with np.load(os.path.join(bundle_dir, "arrays.npz")) as stored:
        arrays = {name: stored[name] for name in stored.files}
    return SavedRun(path=bundle_dir, recipe=recipe, results=results, arrays=arrays)
