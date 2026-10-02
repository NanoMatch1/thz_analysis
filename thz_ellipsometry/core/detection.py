"""Detection: what a <110> electro-optic crystal projects, and how to calibrate it.

The EO crystal is the polarisation analyser of this instrument. For a fixed crystal and probe
polarisation the sampled signal is LINEAR in the THz field, S = d_p E_p + d_s E_s, with a
detection vector (d_p, d_s) that is real and frequency-independent: the crystal is cubic, so
both THz components see the same index, absorption and phase matching, and only the geometric
projection through the chi(2) tensor differs (plan sec. 3.2).

Two angles fix the detection vector:

    phi   the probe polarisation, measured from the crystal [001] axis
    chi   the orientation of the crystal [001] axis, measured from the p direction

The plan mounts the crystal with [001] along s (chi = 90 deg). The MVP code and its tests used
chi = 0; both are just values of the same parameter, and an overall sign of the detection
vector is absorbed by every calibration.

The channel ratio C = d_p/d_s is the one instrument constant isotropic mode needs. Gold supplies
it by swapping a reference into the focus; ``channel_ratio_from_probe_rotation`` supplies it
with nothing in the THz path moving at all, because measuring the SAME sample at two probe
settings and dividing cancels the sample:

    m_1 / m_2 = C(phi_1, chi) rho / (C(phi_2, chi) rho) = C(phi_1, chi) / C(phi_2, chi)

which leaves chi as the only unknown (``explorations/thz_ellipsometry/
probe_rotation_self_calibration.py``, commit c4817b6). That ratio is REAL in the ideal
instrument, so its imaginary part is a free diagnostic of strain birefringence or probe walk.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

__all__ = [
    "BALANCED_PROBE_AZIMUTH_RAD",
    "channel_ratio_for_probe",
    "crystal_orientation_from_probe_settings",
    "detection_channel_ratio",
    "detection_vector_in_sample_frame",
    "electro_optic_detection_vector",
    "is_degenerate_azimuth",
]

#: Probe azimuth from the crystal [001] axis giving equal sensitivity to both THz components,
#: i.e. tan(2*phi) = 2.  At this azimuth the detected signal never nulls as the emitter
#: polarisation sweeps a full 90 degrees.
BALANCED_PROBE_AZIMUTH_RAD = 0.5 * np.arctan(2.0)


def electro_optic_detection_vector(probe_azimuth_rad):
    """Sensitivity of a <110> zincblende crystal to the THz field along its two in-plane axes.

    Planken et al., JOSA B 18, 313 (2001): the sampled signal is

        S ~ E_[001] * sin(2 phi) + 2 * E_perp * cos(2 phi)

    with phi the probe polarisation angle from the crystal [001] axis.

    Returns ``(sensitivity_along_001, sensitivity_perpendicular)`` in the CRYSTAL frame. With
    the crystal's [001] axis along p (chi = 0) this is directly (d_p, d_s); otherwise rotate it
    with :func:`detection_vector_in_sample_frame`.

    At probe_azimuth = 0 or 45 degrees one entry vanishes: those are the DEGENERATE
    orientations, where one polarisation channel is blind and rho cannot be measured at all.
    """
    azimuth = np.asarray(probe_azimuth_rad, dtype=float)
    return np.stack([np.sin(2.0 * azimuth), 2.0 * np.cos(2.0 * azimuth)], axis=-1)


def detection_vector_in_sample_frame(probe_azimuth_rad, crystal_orientation_rad=0.0):
    """(d_p, d_s): the crystal's detection vector rotated into the sample's p/s basis."""
    along, across = np.moveaxis(electro_optic_detection_vector(probe_azimuth_rad), -1, 0)
    cosine, sine = np.cos(crystal_orientation_rad), np.sin(crystal_orientation_rad)
    return np.stack([along * cosine - across * sine, along * sine + across * cosine], axis=-1)


def channel_ratio_for_probe(probe_azimuth_rad, crystal_orientation_rad=0.0):
    """C = d_p / d_s implied by the probe and crystal angles alone."""
    vector = detection_vector_in_sample_frame(probe_azimuth_rad, crystal_orientation_rad)
    return vector[..., 0] / vector[..., 1]


def detection_channel_ratio(probe_azimuth_rad):
    """d_p / d_s for a crystal with [001] along p, = tan(2 phi) / 2.

    Useful as a sanity bound on a measured calibration, not as a replacement for it: 0.1
    degrees of azimuth error is already ~0.9% in this ratio.
    """
    return channel_ratio_for_probe(probe_azimuth_rad, 0.0)


def is_degenerate_azimuth(probe_azimuth_rad, tolerance_deg=2.0):
    """True when the probe is near an orientation that blinds one crystal-frame channel."""
    azimuth_deg = np.rad2deg(np.asarray(probe_azimuth_rad)) % 90.0
    return bool(np.any((azimuth_deg < tolerance_deg)
                       | (np.abs(azimuth_deg - 45.0) < tolerance_deg)
                       | (azimuth_deg > 90.0 - tolerance_deg)))


def crystal_orientation_from_probe_settings(channel_ratios, probe_azimuths_rad, *,
                                            nominal_orientation_rad=0.0,
                                            search_half_width_deg=40.0,
                                            fit_probe_offset=False):
    """Solve for the crystal orientation chi from one sample measured at several probe settings.

    Parameters
    ----------
    channel_ratios : sequence of (n_frequencies,) complex
        The harmonic-fit P/Q for each probe setting, restricted to a trusted band. Each equals
        C(phi_i, chi) * rho for the SAME rho, so ratios between settings cancel the sample.
    probe_azimuths_rad : sequence of float
        The commanded probe angles from the crystal [001] axis.
    fit_probe_offset : bool
        Also fit a common offset of the commanded probe angles. Needs at least three settings:
        two settings give one real equation, enough for chi alone.

    Returns
    -------
    dict with ``orientation_rad``, ``probe_offset_rad``, ``ratio_of_ratios`` (per pair, the
    measured mean), ``imaginary_fraction`` (|Im|/|Re| of the measured ratio-of-ratios: zero for
    an ideal instrument) and ``residual_norm``.
    """
    channel_ratios = [np.asarray(entry, dtype=complex) for entry in channel_ratios]
    azimuths = np.asarray(probe_azimuths_rad, dtype=float)
    if len(channel_ratios) < 2 or len(channel_ratios) != azimuths.size:
        raise ValueError("need the same sample measured at two or more probe settings, with "
                         "one probe azimuth per channel-ratio array")
    if fit_probe_offset and azimuths.size < 3:
        raise ValueError("fitting a probe-angle offset as well as the crystal orientation "
                         "needs at least three probe settings")

    measured = np.array([np.mean(entry / channel_ratios[0]) for entry in channel_ratios[1:]])

    def residual(parameters):
        orientation = parameters[0]
        offset = parameters[1] if fit_probe_offset else 0.0
        reference = channel_ratio_for_probe(azimuths[0] + offset, orientation)
        predicted = np.array([channel_ratio_for_probe(azimuth + offset, orientation) / reference
                              for azimuth in azimuths[1:]])
        difference = predicted - measured
        return np.concatenate([difference.real, difference.imag])

    half_width = np.deg2rad(search_half_width_deg)
    start = [float(nominal_orientation_rad)] + ([0.0] if fit_probe_offset else [])
    lower = [nominal_orientation_rad - half_width] + ([-np.deg2rad(10.0)] if fit_probe_offset else [])
    upper = [nominal_orientation_rad + half_width] + ([np.deg2rad(10.0)] if fit_probe_offset else [])
    result = least_squares(residual, start, bounds=(lower, upper), xtol=1e-14, ftol=1e-14)
    imaginary_fraction = float(np.max(np.abs(measured.imag) / np.maximum(np.abs(measured.real),
                                                                          1e-30)))
    scale = float(np.linalg.norm(measured)) or 1.0
    return {
        "orientation_rad": float(result.x[0]),
        "probe_offset_rad": float(result.x[1]) if fit_probe_offset else 0.0,
        "ratio_of_ratios": measured,
        "imaginary_fraction": imaginary_fraction,
        "residual_norm": float(np.linalg.norm(result.fun) / scale),
    }
