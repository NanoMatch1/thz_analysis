"""Stage and acquisition-mode registries, and the end-to-end driver.

Two registries, each the single source of truth for what it lists:

``@ellipsometry_stage``   one step of a run. The dispatch, the replay list and the help text
                          all read from ELLIPSOMETRY_STAGES -- deliberately not the pattern of
                          ``pipeline_registry.CORE_STAGE_NAMES``, which keeps a hand-maintained
                          tuple alongside the function definitions.
``@acquisition_mode``     a measurement scheme (plan sec. 1): which stages it runs in what
                          order. ``config["mode"]`` picks one. Phase 1 has ``isotropic``; the
                          probe-polarisation generalised mode is added the same way later.

The calibration sources live in ``core.calibration_sources`` (they are pure); the stage here only
assembles their inputs from the loaded data and dispatches by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.calibration_sources import CalibrationInputs, compute_channel_calibration
from ..core.calibration import ellipsometric_ratio_from_channels, fit_incidence_angle
from ..core.detection import is_degenerate_azimuth
from ..core.harmonic import fit_emitter_harmonic
from ..core.materials import reference_index
from ..core.pipeline import analyse_calibrated_series, band_mask
from ..core.preprocess import transform_traces
from ..core.validation import validate_index_against_reference
from .loader import SAMPLE_ROLE, load_measurement
from .noise import estimate_series_noise

__all__ = [
    "ACQUISITION_MODES",
    "ELLIPSOMETRY_STAGES",
    "RunOutcome",
    "acquisition_mode",
    "ellipsometry_stage",
    "mode_names",
    "run_ellipsometry",
    "stage_names",
]

CHANNEL_REFERENCE_ROLE = "channel_reference"
ANGLE_REFERENCE_ROLE = "angle_reference"


@dataclass(frozen=True)
class RegistryEntry:
    name: str
    function: object
    summary: str


#: name -> RegistryEntry. The one place a stage exists.
ELLIPSOMETRY_STAGES: dict[str, RegistryEntry] = {}
#: name -> RegistryEntry. The one place an acquisition mode exists.
ACQUISITION_MODES: dict[str, RegistryEntry] = {}


def _register(registry, function):
    summary = (function.__doc__ or "").strip().splitlines()
    registry[function.__name__] = RegistryEntry(name=function.__name__, function=function,
                                                summary=summary[0] if summary else "")
    return function


def ellipsometry_stage(function):
    """Register a pipeline stage. The docstring's first line becomes its help text."""
    return _register(ELLIPSOMETRY_STAGES, function)


def acquisition_mode(function):
    """Register an acquisition mode. The docstring's first line becomes its help text."""
    return _register(ACQUISITION_MODES, function)


def stage_names():
    return sorted(ELLIPSOMETRY_STAGES)


def mode_names():
    return sorted(ACQUISITION_MODES)


@dataclass
class RunOutcome:
    """Everything a run produced, kept so diagnostics and reports can inspect any of it."""

    config: dict
    series: dict                       #: (role, probe_deg) -> PolarisationSeries
    frequencies_hz: np.ndarray
    result: object
    fits: dict = field(default_factory=dict)          #: (role, probe_deg) -> HarmonicFit
    transformed: dict = field(default_factory=dict)   #: (role, probe_deg) -> TransformedSeries
    noise: dict = field(default_factory=dict)         #: (role, probe_deg) -> SeriesNoise
    primary_probe_deg: float | None = None
    calibration_source: str = ""
    report: object = None
    warnings: list = field(default_factory=list)
    findings: list = field(default_factory=list)

    @property
    def sample_series(self):
        return self.series.get((SAMPLE_ROLE, self.primary_probe_deg))

    @property
    def reference_series(self):
        return self.series.get((CHANNEL_REFERENCE_ROLE, self.primary_probe_deg))


# ---------------------------------------------------------------------------
# Config access, with the defaults in one place
# ---------------------------------------------------------------------------

def _section(config, name):
    return config.get(name) or {}


def _primary_probe_deg(config):
    return _section(config, "detection").get("probe_azimuth_deg")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

@ellipsometry_stage
def load_series(config):
    """Read every .acc file, grouped by role (sample, references) and probe setting."""
    acquisition = _section(config, "acquisition")
    return load_measurement(
        config["data"]["directory"],
        angle_token=acquisition.get("angle_token", "mag"),
        probe_token=acquisition.get("probe_token", "probe"),
        delimiter=acquisition.get("filename_delimiter", "_"),
        role_vocabulary={role: tuple(words) for role, words in
                         acquisition.get("roles", {}).items()} or None,
        magnet_calibration=_section(config, "geometry").get("magnet_calibration"),
        default_probe_azimuth_deg=_primary_probe_deg(config),
    )


@ellipsometry_stage
def transform_to_spectra(config, series):
    """Baseline, window and FFT a polarisation series onto a common frequency axis."""
    preprocess = _section(config, "preprocess")
    return transform_traces(
        series.time_ps, series.traces,
        window_half_width_ps=preprocess.get("window_half_width_ps"),
        baseline_fraction=preprocess.get("baseline_fraction", 0.1),
        pad_factor=preprocess.get("pad_factor", 4),
        window_centre_ps=preprocess.get("window_centre_ps"),
        window_shape=preprocess.get("window_shape", "tukey"),
        taper_fraction=preprocess.get("taper_fraction", 0.5),
    )


@ellipsometry_stage
def estimate_noise(config, series, transformed):
    """Per-acquisition spectral noise from the repeat scans (None when too few repeats)."""
    settings = _section(config, "noise")
    if not settings.get("enabled", True):
        from .noise import SeriesNoise
        return SeriesNoise(None, (), "disabled in config['noise']")
    return estimate_series_noise(series, transformed,
                                 minimum_scans=settings.get("minimum_scans", 3))


@ellipsometry_stage
def fit_harmonic(config, series, transformed, noise):
    """Fit P, Q (and background, drift) to one series, weighted by its noise when known."""
    acquisition = _section(config, "acquisition")
    return fit_emitter_harmonic(
        series.angles_rad, transformed.spectra, transformed.frequencies_hz,
        drift_model=acquisition.get("drift_model", "linear_ramp"),
        elapsed_seconds=series.elapsed_seconds,
        background_term=acquisition.get("background_term", True),
        spectral_variance=noise.spectral_variance,
        amplitude_model=acquisition.get("amplitude_model", "linear_ramp"))


@ellipsometry_stage
def select_band(config, fits, transformed, primary_probe_deg):
    """The trusted band: config limits, an amplitude floor and the channel signal-to-noise."""
    band = _section(config, "band")
    keys = [key for key in ((SAMPLE_ROLE, primary_probe_deg),
                            (CHANNEL_REFERENCE_ROLE, primary_probe_deg)) if key in fits]
    frequencies = transformed[keys[0]].frequencies_hz
    signal_to_noise = [fits[key].channel_signal_to_noise for key in keys
                       if fits[key].channel_signal_to_noise is not None
                       and fits[key].reduced_chi_square is not None]
    minimum_thz = band.get("frequency_min_thz")
    maximum_thz = band.get("frequency_max_thz")
    mask = band_mask(
        frequencies,
        frequency_min_hz=minimum_thz * 1e12 if minimum_thz else None,
        frequency_max_hz=maximum_thz * 1e12 if maximum_thz else None,
        spectra=np.concatenate([transformed[key].spectra for key in keys]),
        minimum_relative_amplitude=band.get("minimum_relative_amplitude", 0.02),
        channel_signal_to_noise=signal_to_noise or None,
        minimum_signal_to_noise=band.get("minimum_signal_to_noise", 0.0))
    if not mask.any():
        hint = ("config['band'] and the signal-to-noise threshold")
        if primary_probe_deg is not None and is_degenerate_azimuth(np.deg2rad(primary_probe_deg)):
            hint = (f"the probe setting: {primary_probe_deg:g} deg is a DEGENERATE azimuth of the "
                    "EO crystal, where one channel is blind. Rotate to ~31.7 deg from [001]")
        raise ValueError(
            f"the requested band selects no frequencies; data spans "
            f"{frequencies.min()/1e12:.2f}-{frequencies.max()/1e12:.2f} THz. Check {hint}.")
    return mask


def _incidence_angle_rad(config):
    return np.deg2rad(config["geometry"]["incidence_angle_deg"])


@ellipsometry_stage
def calibrate(config, fits, frequencies_hz, band, primary_probe_deg):
    """The channel ratio C = d_p/d_s, from the calibration source named in the config."""
    calibration = _section(config, "calibration")
    detection = _section(config, "detection")
    geometry = config["geometry"]
    reference_key = (CHANNEL_REFERENCE_ROLE, primary_probe_deg)
    reference_material = calibration.get("reference_material", "gold")
    reference_fit = fits.get(reference_key)
    inputs = CalibrationInputs(
        frequencies_hz=frequencies_hz, band=band,
        incidence_angle_rad=_incidence_angle_rad(config),
        index_incident=geometry.get("index_incident", 1.0),
        # A KNOWN emitter offset is already removed in the loader (magnet_calibration),
        # which is exactly equivalent to undoing its Moebius mixing; nothing is left here.
        emitter_offset_rad=0.0,
        channel_reference_fit=reference_fit,
        channel_reference_index=(None if reference_fit is None
                                 else reference_index(reference_material, frequencies_hz)),
        channel_reference_name=reference_material,
        sample_fits_by_probe_deg={probe: fit for (role, probe), fit in fits.items()
                                  if role == SAMPLE_ROLE and probe is not None},
        primary_probe_deg=primary_probe_deg,
        crystal_orientation_rad=np.deg2rad(detection.get("crystal_001_from_p_deg", 0.0)),
        fit_probe_offset=detection.get("fit_probe_offset", False),
        stored_channel_ratio=calibration.get("stored_channel_ratio"),
    )
    return compute_channel_calibration(calibration.get("channel", "gold_reference"), inputs)


@ellipsometry_stage
def determine_incidence_angle(config, fits, channel_calibration, frequencies_hz, band,
                              primary_probe_deg):
    """The incidence angle used: set mechanically, or fitted on a known angle reference."""
    calibration = _section(config, "calibration")
    nominal = _incidence_angle_rad(config)
    source = calibration.get("incidence_angle", "mechanical")
    if source == "mechanical":
        return nominal, "mechanical", None
    if source != "fit_from_reference":
        raise ValueError(f"calibration.incidence_angle must be 'mechanical' or "
                         f"'fit_from_reference', not {source!r}")
    key = (ANGLE_REFERENCE_ROLE, primary_probe_deg)
    if key not in fits:
        raise ValueError("calibration.incidence_angle = 'fit_from_reference' needs angle-"
                         "reference files (see config['acquisition']['roles'])")
    material = calibration.get("angle_reference_material", "hr_silicon")
    ratio = ellipsometric_ratio_from_channels(fits[key].channel_ratio, channel_calibration)
    angle_fit = fit_incidence_angle(ratio[band], reference_index(material, frequencies_hz)[band],
                                    nominal, index_incident=config["geometry"].get(
                                        "index_incident", 1.0))
    return angle_fit.angle_rad, f"fit_from_reference({material})", angle_fit


@ellipsometry_stage
def analyse(config, sample_fit, reference_fit, channel_calibration, frequencies_hz, band,
            incidence_angle_rad, incidence_angle_source):
    """Calibrated ratio to n and k, with propagated noise and the angle systematic."""
    inversion = _section(config, "inversion")
    blur = inversion.get("blur") or {}
    geometry = config["geometry"]
    cross_check = _section(config, "validation").get("cross_check_material")
    return analyse_calibrated_series(
        frequencies_hz=frequencies_hz, sample_fit=sample_fit, calibration=channel_calibration,
        incidence_angle_rad=incidence_angle_rad, band=band,
        index_incident=geometry.get("index_incident", 1.0),
        angular_spread_rad=(np.deg2rad(blur["angular_spread_deg"])
                            if blur.get("enabled") and blur.get("angular_spread_deg") else 0.0),
        branch_reference_index=_section(config, "validation").get("branch_reference_index"),
        cross_check_index=(None if cross_check is None
                           else reference_index(cross_check, frequencies_hz)),
        reference_fit=reference_fit,
        fit_out_of_plane_tilt=inversion.get("fit_out_of_plane_tilt", False),
        tilt_model=inversion.get("tilt_model", "drude"),
        tilt_fixed_parameters=inversion.get("tilt_fixed_parameters") or None,
        incidence_angle_uncertainty_deg=geometry.get("incidence_angle_uncertainty_deg", 0.0),
        incidence_angle_source=incidence_angle_source,
    )


@ellipsometry_stage
def validate(config, result, frequencies_hz):
    """Compare the recovered index against a known one, with explicit pass criteria."""
    settings = _section(config, "validation")
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


@ellipsometry_stage
def check_assumptions(outcome):
    """Run every registered ellipsometry diagnostic against the finished run (never halts)."""
    from .diagnostics import run_ellipsometry_diagnostics
    return run_ellipsometry_diagnostics(outcome)


def _collect_warnings(config, series, primary_probe_deg):
    """Setup problems worth stating before any numbers are believed."""
    warnings = []
    if primary_probe_deg is None:
        warnings.append("config['detection']['probe_azimuth_deg'] is not set; the probe "
                        "setting of every file must then come from its filename.")
    sample = series.get((SAMPLE_ROLE, primary_probe_deg))
    if sample is not None and len(np.unique(np.round(sample.angles_deg % 360.0, 6))) < 3:
        warnings.append(
            f"only {len(np.unique(sample.angles_deg % 360.0))} distinct polarisation angles; "
            "four (magnet 0/90/180/270) are needed for the background and drift terms.")
    if (CHANNEL_REFERENCE_ROLE, primary_probe_deg) not in series and (
            _section(config, "calibration").get("channel", "gold_reference")
            == "gold_reference"):
        warnings.append("no file matched the channel-reference vocabulary; the "
                        "'gold_reference' calibration source cannot run.")
    return warnings


# ---------------------------------------------------------------------------
# Acquisition modes
# ---------------------------------------------------------------------------

@acquisition_mode
def isotropic(config):
    """Magnet states at one probe setting (plus optional probe-rotation pairs): rho -> n, k."""
    primary_probe_deg = _primary_probe_deg(config)
    series = load_series(config)
    warnings = _collect_warnings(config, series, primary_probe_deg)

    transformed, noise, fits = {}, {}, {}
    for key, entry in series.items():
        transformed[key] = transform_to_spectra(config, entry)
        noise[key] = estimate_noise(config, entry, transformed[key])
        fits[key] = fit_harmonic(config, entry, transformed[key], noise[key])

    axes = [item.frequencies_hz for item in transformed.values()]
    if any(axis.shape != axes[0].shape or not np.allclose(axis, axes[0]) for axis in axes):
        raise ValueError("the series were transformed onto different frequency axes; they "
                         "must share a time axis and padding")
    frequencies_hz = axes[0]

    sample_key = (SAMPLE_ROLE, primary_probe_deg)
    if sample_key not in fits:
        raise ValueError(f"no sample series at the primary probe setting {primary_probe_deg}; "
                         f"loaded: {sorted(series)}")
    band = select_band(config, fits, transformed, primary_probe_deg)
    channel_calibration = calibrate(config, fits, frequencies_hz, band, primary_probe_deg)
    angle_rad, angle_source, angle_fit = determine_incidence_angle(
        config, fits, channel_calibration, frequencies_hz, band, primary_probe_deg)
    result = analyse(config, fits[sample_key],
                     fits.get((CHANNEL_REFERENCE_ROLE, primary_probe_deg)),
                     channel_calibration, frequencies_hz, band, angle_rad, angle_source)
    if angle_fit is not None and result.incidence_angle_fit is None:
        from dataclasses import replace
        result = replace(result, incidence_angle_fit=angle_fit)
    report = validate(config, result, frequencies_hz)

    outcome = RunOutcome(
        config=config, series=series, frequencies_hz=frequencies_hz, result=result,
        fits=fits, transformed=transformed, noise=noise, primary_probe_deg=primary_probe_deg,
        calibration_source=_section(config, "calibration").get("channel", "gold_reference"),
        report=report, warnings=warnings)
    outcome.findings = check_assumptions(outcome)
    return outcome


def run_ellipsometry(config):
    """Drive the whole chain from a config dict, in the mode it names. Returns a RunOutcome."""
    mode = config.get("mode", "isotropic")
    try:
        entry = ACQUISITION_MODES[mode]
    except KeyError:
        raise KeyError(f"unknown acquisition mode {mode!r}; registered: {mode_names()}") \
            from None
    return entry.function(config)
