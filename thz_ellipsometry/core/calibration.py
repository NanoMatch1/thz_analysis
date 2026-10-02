"""Calibration: the detection channel ratio, and the incidence angle cross-check.

The harmonic fit returns ``P/Q = (d_p/d_s) * rho``.  One complex, frequency-flat instrument
constant stands between that and the physics, and a gold mirror measured in the same geometry
supplies it.

Why swapping gold in is safe, when swapping a reflectance reference in is not: what we take from
gold is a polarisation RATIO, which is itself immune to the sample-placement error that poisons
an amplitude reference.  A micron of height or a femtosecond of drift multiplies both
polarisations identically and divides out of the reference just as it divides out of the sample.

The incidence angle is NOT obtained here for the silicon validation -- it is set mechanically,
because polished silicon reflects visible light and can be aligned with a camera.  The angle fit
below is a CROSS-CHECK against that mechanical value: agreement validates both, disagreement
localises the problem to the geometry rather than to the analysis.

One instrument term does NOT cancel against a single reference, and it is worth being explicit
about because an earlier draft of this design assumed it did.  If the emitter's angular zero is
offset from the plane of incidence by delta, the measured channel ratio is not a scaled rho but
a Moebius transform of it:

    m = (C rho + t) / (1 - C t rho),      C = d_p/d_s,   t = tan(delta)

because the commanded "p" setting then contains sin(delta) of s.  Gold alone cannot separate C
from t -- its rho is essentially -1 at every frequency, so it supplies one complex constraint
for three real unknowns.  Two options: measure the emitter zero once during setup (a wire grid
aligned to the plane of incidence gives a sharp, sign-sensitive null in coherent detection), or
supply a SECOND known reference whose rho differs from gold's and fit (C, t) jointly.

The tolerance, if the offset is simply left uncorrected: |dN| is about 0.009 per 0.1 degrees at
70 degrees incidence and 0.002 per 0.1 degrees at 45, essentially independent of the sample. So
0.2 degrees keeps it inside the 0.02 validation tolerance at 70 degrees, and a degree is fine at
45. It is a mechanical specification, not a blocker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .model import ellipsometric_ratio

__all__ = [
    "ChannelCalibration",
    "IncidenceAngleFit",
    "apply_instrument_to_ratio",
    "channel_ratio_from_reference",
    "ellipsometric_ratio_from_channels",
    "fit_incidence_angle",
    "fit_instrument_from_references",
    "ratio_derivative_wrt_measured",
    "remove_emitter_offset",
]


def apply_instrument_to_ratio(ratio, channel_ratio, emitter_offset_rad=0.0):
    """Forward instrument model: rho -> the channel ratio the harmonic fit actually returns."""
    tangent = np.tan(emitter_offset_rad)
    ratio = np.asarray(ratio, dtype=complex)
    return (channel_ratio * ratio + tangent) / (1.0 - channel_ratio * tangent * ratio)


def _recover_ratio(measured, channel_ratio, emitter_offset_rad=0.0):
    """Inverse of apply_instrument_to_ratio."""
    return remove_emitter_offset(measured, emitter_offset_rad) / channel_ratio


def remove_emitter_offset(measured, emitter_offset_rad=0.0):
    """Undo the Moebius mixing of a known emitter offset, leaving C * rho."""
    tangent = np.tan(emitter_offset_rad)
    measured = np.asarray(measured, dtype=complex)
    return (measured - tangent) / (1.0 + measured * tangent)


def ratio_derivative_wrt_measured(measured, channel_ratio, emitter_offset_rad=0.0):
    """d rho / d(P/Q): how noise on the measured channel ratio reaches rho."""
    tangent = np.tan(emitter_offset_rad)
    measured = np.asarray(measured, dtype=complex)
    return (1.0 + tangent**2) / ((1.0 + measured * tangent) ** 2 * channel_ratio)


@dataclass(frozen=True)
class ChannelCalibration:
    """The detection channel ratio d_p/d_s, with its own quality diagnostics."""

    ratio_per_frequency: np.ndarray   #: (n_frequencies,) complex
    ratio: complex                    #: the frequency-averaged value actually applied
    relative_scatter: float           #: std/|mean| across frequency -- should be small
    phase_scatter_rad: float          #: std of arg across frequency
    reference_name: str
    emitter_offset_rad: float = 0.0   #: emitter angular zero vs the plane of incidence
    #: |standard error| of the applied ratio. It is ONE number for the whole run, so its error
    #: is fully correlated across frequency: a systematic of the run, reported as a flag and
    #: deliberately not folded into the per-frequency noise bars.
    standard_error: float = 0.0

    @property
    def emitter_offset_deg(self):
        return float(np.rad2deg(self.emitter_offset_rad))

    @property
    def is_flat(self):
        """The detection vector is real and frequency-independent by crystal symmetry.

        Structure in the measured ratio therefore means something ELSE is wrong -- beam
        astigmatism, a mis-set crystal, polarisation-dependent optics -- so flatness is a
        genuine diagnostic rather than a formality.
        """
        return self.relative_scatter < 0.05 and self.phase_scatter_rad < 0.1


def channel_ratio_from_reference(reference_channel_ratio, reference_index,
                                 incidence_angle_rad, index_incident=1.0,
                                 reference_name="gold", frequency_mask=None,
                                 emitter_offset_rad=0.0):
    """d_p/d_s from a reference sample of known index.

    Parameters
    ----------
    reference_channel_ratio : (n_frequencies,) complex
        ``P/Q`` from the harmonic fit of the reference measurement.
    reference_index : complex or (n_frequencies,) complex
        Known index of the reference. For gold, rho is -1 to about 0.1% at any angle.
    frequency_mask : (n_frequencies,) bool, optional
        Restrict the average to a trusted band.
    """
    reference_channel_ratio = np.asarray(reference_channel_ratio, dtype=complex)
    expected = ellipsometric_ratio(reference_index, incidence_angle_rad, index_incident)
    expected = np.broadcast_to(np.asarray(expected, dtype=complex),
                               reference_channel_ratio.shape)
    tangent = np.tan(emitter_offset_rad)
    # Undo the known emitter offset before dividing, so what remains is purely d_p/d_s.
    deoffset = (reference_channel_ratio - tangent) / (1.0 + reference_channel_ratio * tangent)
    ratio_per_frequency = deoffset / expected

    selected = (ratio_per_frequency if frequency_mask is None
                else ratio_per_frequency[np.asarray(frequency_mask, dtype=bool)])
    if selected.size == 0:
        raise ValueError("frequency_mask selected no frequencies for the channel calibration")

    mean_ratio = complex(np.mean(selected))
    magnitude = np.abs(mean_ratio)
    relative_scatter = float(np.std(np.abs(selected)) / magnitude) if magnitude else np.inf
    phase_scatter = float(np.std(np.unwrap(np.angle(selected))))

    return ChannelCalibration(
        ratio_per_frequency=ratio_per_frequency,
        ratio=mean_ratio,
        relative_scatter=relative_scatter,
        phase_scatter_rad=phase_scatter,
        reference_name=reference_name,
        emitter_offset_rad=float(emitter_offset_rad),
        standard_error=float(np.std(selected) / np.sqrt(selected.size)),
    )


def ellipsometric_ratio_from_channels(channel_ratio, calibration):
    """Undo the instrument: (P/Q) -> rho, including any known emitter angular offset."""
    if isinstance(calibration, ChannelCalibration):
        return _recover_ratio(channel_ratio, calibration.ratio, calibration.emitter_offset_rad)
    return _recover_ratio(channel_ratio, calibration, 0.0)


def fit_instrument_from_references(measured_channel_ratios, reference_indices,
                                   incidence_angle_rad, index_incident=1.0,
                                   frequency_mask=None, reference_names=None,
                                   maximum_offset_deg=10.0):
    """Fit (d_p/d_s, emitter offset) jointly from TWO OR MORE known reference samples.

    Gold alone cannot do this -- its rho is essentially -1 at every frequency, so it gives one
    complex constraint for three real unknowns. A second reference whose rho differs strongly,
    such as an HR-Si wafer (rho ~ -0.12 at 70 degrees), closes it and over-determines it across
    frequency.

    Use this when the emitter zero has not been established independently, or to check a value
    that has. Note that it consumes each reference as a CALIBRATOR, so a sample used here is no
    longer an independent validation.
    """
    measured_channel_ratios = [np.asarray(entry, dtype=complex)
                               for entry in measured_channel_ratios]
    if len(measured_channel_ratios) < 2:
        raise ValueError(
            "at least two references with different rho are needed to separate the channel "
            "ratio from the emitter offset; gold alone is degenerate")

    expected = []
    for index_entry, measured in zip(reference_indices, measured_channel_ratios):
        value = ellipsometric_ratio(index_entry, incidence_angle_rad, index_incident)
        expected.append(np.broadcast_to(np.asarray(value, dtype=complex), measured.shape))

    if frequency_mask is not None:
        mask = np.asarray(frequency_mask, dtype=bool)
        measured_channel_ratios = [entry[mask] for entry in measured_channel_ratios]
        expected = [entry[mask] for entry in expected]

    def residual(parameters):
        channel = parameters[0] + 1j * parameters[1]
        offset = np.deg2rad(parameters[2])
        pieces = []
        for measured, truth in zip(measured_channel_ratios, expected):
            difference = apply_instrument_to_ratio(truth, channel, offset) - measured
            pieces.append(np.concatenate([difference.real, difference.imag]))
        return np.concatenate(pieces)

    start = measured_channel_ratios[0][0] / expected[0][0]
    result = least_squares(
        residual, [start.real, start.imag, 0.0],
        bounds=([-np.inf, -np.inf, -maximum_offset_deg],
                [np.inf, np.inf, maximum_offset_deg]),
        xtol=1e-14, ftol=1e-14)

    channel = complex(result.x[0] + 1j * result.x[1])
    offset = float(np.deg2rad(result.x[2]))
    per_frequency = np.full(measured_channel_ratios[0].shape, channel)
    names = "+".join(reference_names) if reference_names else "multi-reference"
    return ChannelCalibration(
        ratio_per_frequency=per_frequency,
        ratio=channel,
        relative_scatter=0.0,
        phase_scatter_rad=0.0,
        reference_name=names,
        emitter_offset_rad=offset,
    )


# ---------------------------------------------------------------------------
# Incidence angle: a cross-check, not a calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IncidenceAngleFit:
    angle_rad: float
    angle_deg: float
    residual_norm: float
    nominal_angle_deg: float

    @property
    def discrepancy_deg(self):
        return self.angle_deg - self.nominal_angle_deg


def fit_incidence_angle(measured_ratio, known_index, nominal_angle_rad, index_incident=1.0,
                        search_half_width_deg=8.0, frequency_mask=None):
    """Fit the single parameter theta to a sample whose index is known.

    Used on HR-Si, whose index is 3.4175 and flat, to check the mechanically set angle. One
    parameter against many frequencies, so it is well determined -- expect about 0.01 degrees
    at a 0.5% measurement error.
    """
    measured_ratio = np.asarray(measured_ratio, dtype=complex)
    if frequency_mask is not None:
        mask = np.asarray(frequency_mask, dtype=bool)
        measured_ratio = measured_ratio[mask]
        known_index = (known_index[mask] if np.ndim(known_index) else known_index)
    if measured_ratio.size == 0:
        raise ValueError("no frequencies selected for the incidence-angle fit")

    def residual(parameters):
        modelled = ellipsometric_ratio(known_index, parameters[0], index_incident)
        difference = modelled - measured_ratio
        return np.concatenate([difference.real, difference.imag])

    half_width = np.deg2rad(search_half_width_deg)
    result = least_squares(
        residual, [float(nominal_angle_rad)],
        bounds=([max(nominal_angle_rad - half_width, np.deg2rad(1.0))],
                [min(nominal_angle_rad + half_width, np.deg2rad(89.0))]),
        xtol=1e-14, ftol=1e-14)

    scale = np.linalg.norm(np.concatenate([measured_ratio.real, measured_ratio.imag]))
    return IncidenceAngleFit(
        angle_rad=float(result.x[0]),
        angle_deg=float(np.rad2deg(result.x[0])),
        residual_norm=float(np.linalg.norm(result.fun) / (scale if scale else 1.0)),
        nominal_angle_deg=float(np.rad2deg(nominal_angle_rad)),
    )
