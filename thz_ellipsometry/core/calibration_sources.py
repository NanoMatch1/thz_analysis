"""Where the instrument's channel ratio C = d_p/d_s comes from: a registry of sources.

Isotropic mode needs exactly one instrument constant between the harmonic fit and rho (plan
sec. 5.1). There is more than one honest way to get it, and they cross-check each other, so each
is a registered source that declares what data it needs. Adding a source means defining it here
with ``@calibration_source`` and nothing else: the driver dispatches on the registry, and the
help text and the "what does this need" error messages are read from the same entries.

Every source receives the same :class:`CalibrationInputs` and returns a
:class:`~thz_ellipsometry.core.calibration.ChannelCalibration` holding the C(f) it measured; the
calibration MODEL named in the inputs (``calibration_models``: constant, constant + delay, ...)
is then fitted to that, the same way for every source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibration import (
    ChannelCalibration,
    apply_calibration_model,
    channel_ratio_from_reference,
)
from .detection import channel_ratio_for_probe, crystal_orientation_from_probe_settings
from .harmonic import HarmonicFit

__all__ = [
    "CALIBRATION_SOURCES",
    "CalibrationInputs",
    "CalibrationSource",
    "calibration_source",
    "calibration_source_names",
    "compute_channel_calibration",
]


@dataclass
class CalibrationInputs:
    """Everything any calibration source might need. Sources take what they declare."""

    frequencies_hz: np.ndarray
    band: np.ndarray                                  #: trusted-frequency mask
    incidence_angle_rad: float
    index_incident: float = 1.0
    emitter_offset_rad: float = 0.0
    #: A reference of known index (normally gold) measured in the same geometry.
    channel_reference_fit: HarmonicFit | None = None
    channel_reference_index: np.ndarray | None = None
    channel_reference_name: str = "gold"
    #: The sample's harmonic fit at each probe setting, keyed by probe azimuth in degrees
    #: (from the crystal [001] axis). The primary setting is the one the inversion uses.
    sample_fits_by_probe_deg: dict = field(default_factory=dict)
    primary_probe_deg: float | None = None
    crystal_orientation_rad: float = 0.0
    fit_probe_offset: bool = False
    #: A previously measured channel ratio, for replay.
    stored_channel_ratio: complex | None = None
    #: The registered calibration model fitted to the source's C(f) (calibration_models).
    model: str = "constant"


@dataclass(frozen=True)
class CalibrationSource:
    name: str
    function: object
    requires: tuple
    summary: str


#: name -> CalibrationSource. The one place a source exists.
CALIBRATION_SOURCES: dict[str, CalibrationSource] = {}


def calibration_source(*, requires):
    """Register a channel-ratio source. ``requires`` names the CalibrationInputs it reads."""
    def register(function):
        summary = (function.__doc__ or "").strip().splitlines()
        CALIBRATION_SOURCES[function.__name__] = CalibrationSource(
            name=function.__name__, function=function, requires=tuple(requires),
            summary=summary[0] if summary else "")
        return function
    return register


def calibration_source_names():
    return sorted(CALIBRATION_SOURCES)


def _missing_requirements(entry, inputs):
    missing = []
    for requirement in entry.requires:
        value = getattr(inputs, requirement)
        if requirement == "sample_fits_by_probe_deg":
            if len(value) < 2:
                missing.append("the sample measured at two or more probe settings")
        elif value is None:
            missing.append(requirement)
    return missing


def compute_channel_calibration(name, inputs):
    """Dispatch to a registered source, after checking it has what it declares it needs."""
    try:
        entry = CALIBRATION_SOURCES[name]
    except KeyError:
        raise KeyError(f"unknown calibration source {name!r}; registered: "
                       f"{calibration_source_names()}") from None
    missing = _missing_requirements(entry, inputs)
    if missing:
        raise ValueError(f"calibration source {name!r} ({entry.summary}) cannot run: missing "
                         f"{', '.join(missing)}")
    return apply_calibration_model(entry.function(inputs), inputs.model, inputs.frequencies_hz,
                                   inputs.band)


@calibration_source(requires=("channel_reference_fit", "channel_reference_index"))
def gold_reference(inputs):
    """A known reflector (gold) swapped into the focus: C = (P/Q)_ref / rho_ref."""
    return channel_ratio_from_reference(
        inputs.channel_reference_fit.channel_ratio, inputs.channel_reference_index,
        inputs.incidence_angle_rad, index_incident=inputs.index_incident,
        reference_name=inputs.channel_reference_name, frequency_mask=inputs.band,
        emitter_offset_rad=inputs.emitter_offset_rad,
        reference_channel_ratio_variance=inputs.channel_reference_fit.channel_ratio_variance)


@calibration_source(requires=("stored_channel_ratio",))
def stored(inputs):
    """A previously measured channel ratio, replayed."""
    value = complex(inputs.stored_channel_ratio)
    return ChannelCalibration(
        ratio_per_frequency=np.full(np.shape(inputs.frequencies_hz), value), ratio=value,
        relative_scatter=0.0, phase_scatter_rad=0.0, reference_name="stored",
        emitter_offset_rad=float(inputs.emitter_offset_rad))


@calibration_source(requires=("sample_fits_by_probe_deg", "primary_probe_deg"))
def probe_rotation(inputs):
    """The sample itself at two probe settings: the sample cancels, nothing in the THz path moves.

    m_i = C(phi_i, chi) rho for the same rho, so ratios between settings fix the crystal
    orientation chi, and with it C at the primary setting. Assumes the emitter zero is on the
    plane of incidence (an offset mixes channels and the sample no longer cancels exactly), and
    a real detection vector -- the imaginary part of the measured ratio-of-ratios, reported as
    ``imaginary_fraction``, says how far that holds.
    """
    probes_deg = sorted(inputs.sample_fits_by_probe_deg)
    primary = float(inputs.primary_probe_deg)
    if primary not in inputs.sample_fits_by_probe_deg:
        raise ValueError(f"primary probe setting {primary} deg was not measured; measured "
                         f"settings are {probes_deg}")
    ordered = [primary] + [probe for probe in probes_deg if probe != primary]
    band = np.asarray(inputs.band, dtype=bool)
    ratios = [inputs.sample_fits_by_probe_deg[probe].channel_ratio[band] for probe in ordered]
    solution = crystal_orientation_from_probe_settings(
        ratios, np.deg2rad(ordered), nominal_orientation_rad=inputs.crystal_orientation_rad,
        fit_probe_offset=inputs.fit_probe_offset)
    value = complex(channel_ratio_for_probe(np.deg2rad(primary) + solution["probe_offset_rad"],
                                            solution["orientation_rad"]))
    # Per-frequency scatter of the measured ratio-of-ratios is the honest flatness diagnostic.
    per_frequency = ratios[1] / ratios[0]
    mean = np.mean(per_frequency)
    return ChannelCalibration(
        ratio_per_frequency=np.full(np.shape(inputs.frequencies_hz), value), ratio=value,
        relative_scatter=float(np.std(np.abs(per_frequency)) / max(abs(mean), 1e-30)),
        phase_scatter_rad=float(np.std(np.unwrap(np.angle(per_frequency)))),
        reference_name=(f"probe_rotation(chi={np.rad2deg(solution['orientation_rad']):.3f} deg,"
                        f" imag fraction {solution['imaginary_fraction']:.2g})"),
        emitter_offset_rad=0.0)
