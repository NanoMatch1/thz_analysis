"""Stage registry and the end-to-end driver.

The registry is the single source of truth for what the package can do: a stage registers itself
at definition time, and the dispatch table, the replay list and the help text all read from the
same structure. Adding a stage therefore means editing exactly one place -- deliberately not the
pattern used by ``pipeline_registry.CORE_STAGE_NAMES``, which keeps a hand-maintained tuple
alongside the function definitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import fit_instrument_from_references
from .loader import load_polarisation_series, spectra_from_traces
from .materials import reference_index
from .model import is_degenerate_azimuth
from .pipeline import analyse_polarisation_series
from .validation import validate_index_against_reference

__all__ = [
    "ELLIPSOMETRY_STAGES",
    "RunOutcome",
    "ellipsometry_stage",
    "run_ellipsometry",
    "stage_names",
]

#: name -> StageEntry. The one place a stage exists.
ELLIPSOMETRY_STAGES = {}


@dataclass(frozen=True)
class StageEntry:
    name: str
    function: object
    summary: str


def ellipsometry_stage(function):
    """Register a pipeline stage. The docstring's first line becomes its help text."""
    summary = (function.__doc__ or "").strip().splitlines()
    ELLIPSOMETRY_STAGES[function.__name__] = StageEntry(
        name=function.__name__, function=function,
        summary=summary[0] if summary else "")
    return function


def stage_names():
    return sorted(ELLIPSOMETRY_STAGES)


@dataclass
class RunOutcome:
    config: dict
    sample_series: object
    reference_series: object
    frequencies_hz: np.ndarray
    result: object
    report: object = None
    warnings: list = None


@ellipsometry_stage
def load_series(config):
    """Read the .acc polarisation series for the sample and its reference."""
    polarization = config["polarization"]
    reference = config.get("reference", {})
    return load_polarisation_series(
        config["data"]["directory"],
        angle_token=polarization.get("angle_token", "pol"),
        delimiter=polarization.get("filename_delimiter", "_"),
        reference_vocabulary=tuple(reference.get("vocabulary", ("gold", "mirror"))),
    )


@ellipsometry_stage
def transform_to_spectra(config, series):
    """Baseline, window and FFT a polarisation series onto a common frequency axis."""
    preprocess = config.get("preprocess", {})
    return spectra_from_traces(
        series.time_ps, series.traces,
        window_half_width_ps=preprocess.get("window_half_width_ps"),
        baseline_fraction=preprocess.get("baseline_fraction", 0.1),
        pad_factor=preprocess.get("pad_factor", 4),
        window_centre_ps=preprocess.get("window_centre_ps"),
    )


def _collect_warnings(config, sample_series, reference_series):
    warnings = []
    azimuth_deg = config["polarization"].get("probe_azimuth_deg")
    if azimuth_deg is not None and is_degenerate_azimuth(np.deg2rad(azimuth_deg)):
        warnings.append(
            f"GaP probe azimuth {azimuth_deg} deg is near a DEGENERATE orientation: one "
            "polarisation channel is blind and rho cannot be measured. Rotate to ~31.7 deg.")
    if len(sample_series) < 4:
        warnings.append(
            f"only {len(sample_series)} emitter angles; 6-12 are recommended so the drift "
            "nuisance is identifiable and the harmonic residual means something.")
    if reference_series is None:
        warnings.append(
            "no reference file matched the reference vocabulary; the channel ratio must then "
            "be supplied in config['reference']['channel_ratio'].")
    spacing = np.diff(np.sort(sample_series.angles_deg))
    if spacing.size and np.ptp(spacing) > 0.25 * np.mean(spacing):
        warnings.append("emitter angles are unevenly spaced; the harmonic fit copes, but "
                        "check that no acquisition is missing.")
    return warnings


@ellipsometry_stage
def analyse(config, sample_spectra, reference_spectra, frequencies_hz):
    """Harmonic fit, channel calibration and inversion for n and k."""
    geometry = config["geometry"]
    blur = config.get("blur", {})
    band = config.get("band", {})
    reference = config.get("reference", {})
    incidence_angle = np.deg2rad(geometry["incidence_angle_deg"])

    reference_name = reference.get("index", "gold")
    reference_values = (reference_index(reference_name, frequencies_hz)
                        if reference_spectra is not None else None)

    angular_spread = (np.deg2rad(blur["angular_spread_deg"])
                      if blur.get("enabled") and blur.get("angular_spread_deg") else 0.0)

    return analyse_polarisation_series(
        frequencies_hz=frequencies_hz,
        sample_spectra=sample_spectra["spectra"],
        sample_emitter_angles_rad=sample_spectra["angles_rad"],
        reference_spectra=None if reference_spectra is None else reference_spectra["spectra"],
        reference_emitter_angles_rad=(None if reference_spectra is None
                                      else reference_spectra["angles_rad"]),
        reference_index=reference_values,
        reference_name=reference_name,
        channel_ratio=reference.get("channel_ratio"),
        emitter_offset_rad=np.deg2rad(geometry.get("emitter_offset_deg", 0.0)),
        incidence_angle_rad=incidence_angle,
        index_incident=geometry.get("index_incident", 1.0),
        drift_model=config["polarization"].get("drift_model", "linear_ramp"),
        angular_spread_rad=angular_spread,
        frequency_min_hz=band.get("frequency_min_thz", 0.0) * 1e12 or None,
        frequency_max_hz=(band["frequency_max_thz"] * 1e12
                          if band.get("frequency_max_thz") else None),
        minimum_relative_amplitude=band.get("minimum_relative_amplitude", 0.02),
        branch_reference_index=config.get("validation", {}).get("branch_reference_index"),
        cross_check_index=_cross_check_index(config, frequencies_hz),
    )


def _cross_check_index(config, frequencies_hz):
    name = config.get("validation", {}).get("cross_check_material")
    return None if name is None else reference_index(name, frequencies_hz)


@ellipsometry_stage
def validate(config, result, frequencies_hz):
    """Compare the recovered index against a known one, with explicit pass criteria."""
    settings = config.get("validation", {})
    expected_name = settings.get("expect")
    if expected_name is None:
        return None
    expected = reference_index(expected_name, frequencies_hz)[result.band]
    return validate_index_against_reference(
        result, expected, label=expected_name,
        tolerance_n=settings.get("tolerance_n", 0.02),
        tolerance_k=settings.get("tolerance_k", 0.05),
        expected_angle_deg=(config["geometry"]["incidence_angle_deg"]
                            if settings.get("cross_check_material") else None),
        tolerance_angle_deg=settings.get("tolerance_angle_deg", 0.1),
    )


def run_ellipsometry(config):
    """Drive the whole chain from a config dict. Returns a RunOutcome."""
    sample_series, reference_series = load_series(config)
    warnings = _collect_warnings(config, sample_series, reference_series)

    frequencies_hz, sample_spectra = transform_to_spectra(config, sample_series)
    sample_block = {"spectra": sample_spectra, "angles_rad": sample_series.angles_rad}

    reference_block = None
    if reference_series is not None:
        reference_frequencies, reference_spectra = transform_to_spectra(
            config, reference_series)
        if not np.allclose(reference_frequencies, frequencies_hz):
            raise ValueError("sample and reference were transformed onto different frequency "
                             "axes; they must share a time axis and padding")
        reference_block = {"spectra": reference_spectra,
                           "angles_rad": reference_series.angles_rad}

    result = analyse(config, sample_block, reference_block, frequencies_hz)
    report = validate(config, result, frequencies_hz)

    return RunOutcome(config=config, sample_series=sample_series,
                      reference_series=reference_series, frequencies_hz=frequencies_hz,
                      result=result, report=report, warnings=warnings)
