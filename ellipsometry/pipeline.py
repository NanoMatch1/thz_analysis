"""The analysis chain, start to finish, with no file I/O and no DataSet.

Everything the measurement needs, expressed as one pure function of arrays. The adapter layer
loads data and calls this; the tests and the simulator call exactly the same thing, so there is
no path through the analysis that only runs on real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import (
    ChannelCalibration,
    channel_ratio_from_reference,
    ellipsometric_ratio_from_channels,
    fit_incidence_angle,
)
from .harmonic import HarmonicFit, fit_emitter_harmonic
from .inversion import InversionResult, invert_ratio

__all__ = ["EllipsometryResult", "analyse_polarisation_series", "band_mask"]


def band_mask(frequencies_hz, frequency_min_hz=None, frequency_max_hz=None,
              spectra=None, minimum_relative_amplitude=0.0):
    """Boolean mask selecting a trusted frequency band.

    Beyond the requested limits this also drops bins whose amplitude has fallen below a
    fraction of the series peak. Without that floor the dead top end of the spectrum reaches
    the channel calibration and the inversion as pure noise, which shows up as a large
    apparent scatter in a quantity that is supposed to be frequency-flat.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    mask = np.ones(frequencies_hz.shape, dtype=bool)
    if frequency_min_hz is not None:
        mask &= frequencies_hz >= frequency_min_hz
    if frequency_max_hz is not None:
        mask &= frequencies_hz <= frequency_max_hz
    if spectra is not None and minimum_relative_amplitude > 0.0:
        amplitude = np.max(np.abs(np.atleast_2d(spectra)), axis=0)
        peak = amplitude.max()
        if peak > 0:
            mask &= amplitude >= minimum_relative_amplitude * peak
    return mask


@dataclass(frozen=True)
class EllipsometryResult:
    frequencies_hz: np.ndarray
    sample_fit: HarmonicFit
    reference_fit: HarmonicFit | None
    calibration: ChannelCalibration | None
    ratio: np.ndarray
    inversion: InversionResult
    band: np.ndarray
    incidence_angle_fit: object | None = None

    @property
    def index(self):
        return self.inversion.index

    @property
    def quality_flags(self):
        """Everything a run should be judged on before its numbers are believed."""
        flags = {
            "sample_harmonic_residual": self.sample_fit.residual_norm,
            "sample_fitted_drift_fs": float(np.ptp(self.sample_fit.delays_s) * 1e15),
        }
        if self.reference_fit is not None:
            flags["reference_harmonic_residual"] = self.reference_fit.residual_norm
        if self.calibration is not None:
            flags["channel_ratio_scatter"] = self.calibration.relative_scatter
            flags["channel_ratio_phase_scatter_rad"] = self.calibration.phase_scatter_rad
            flags["channel_ratio_is_flat"] = self.calibration.is_flat
        if self.incidence_angle_fit is not None:
            flags["incidence_angle_discrepancy_deg"] = (
                self.incidence_angle_fit.discrepancy_deg)
        return flags


def analyse_polarisation_series(
    *,
    frequencies_hz,
    sample_spectra,
    sample_emitter_angles_rad,
    incidence_angle_rad,
    reference_spectra=None,
    reference_emitter_angles_rad=None,
    reference_index=None,
    reference_name="gold",
    channel_ratio=None,
    emitter_offset_rad=0.0,
    index_incident=1.0,
    drift_model="linear_ramp",
    angular_spread_rad=0.0,
    frequency_min_hz=None,
    frequency_max_hz=None,
    minimum_relative_amplitude=0.02,
    branch_reference_index=None,
    cross_check_index=None,
):
    """Run the whole chain: harmonic fit -> channel calibration -> rho -> n, k.

    Exactly one of ``reference_spectra`` (with ``reference_index``) or ``channel_ratio`` must be
    supplied. The first is the normal path -- a gold mirror measured in the same geometry. The
    second exists for replaying a stored calibration, and for tests.

    ``cross_check_index``, if given, triggers a one-parameter incidence-angle fit against a
    sample of known index. This is a CROSS-CHECK on the mechanically set angle, not a
    calibration: the returned angle is reported, never silently applied.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    amplitude_source = (sample_spectra if reference_spectra is None
                        else np.concatenate([np.atleast_2d(sample_spectra),
                                             np.atleast_2d(reference_spectra)]))
    mask = band_mask(frequencies_hz, frequency_min_hz, frequency_max_hz,
                     spectra=amplitude_source,
                     minimum_relative_amplitude=minimum_relative_amplitude)
    if not mask.any():
        raise ValueError(
            f"the requested band selects no frequencies; data spans "
            f"{frequencies_hz.min()/1e12:.2f}-{frequencies_hz.max()/1e12:.2f} THz and the "
            f"amplitude floor is {minimum_relative_amplitude:.3g} of the peak")

    sample_fit = fit_emitter_harmonic(sample_emitter_angles_rad, sample_spectra,
                                      frequencies_hz, drift_model=drift_model)

    reference_fit = None
    calibration = None
    if reference_spectra is not None:
        if reference_index is None:
            raise ValueError("reference_spectra requires reference_index")
        if reference_emitter_angles_rad is None:
            reference_emitter_angles_rad = sample_emitter_angles_rad
        reference_fit = fit_emitter_harmonic(reference_emitter_angles_rad, reference_spectra,
                                             frequencies_hz, drift_model=drift_model)
        calibration = channel_ratio_from_reference(
            reference_fit.channel_ratio, reference_index, incidence_angle_rad,
            index_incident=index_incident, reference_name=reference_name,
            frequency_mask=mask, emitter_offset_rad=emitter_offset_rad)
        ratio = ellipsometric_ratio_from_channels(sample_fit.channel_ratio, calibration)
    elif channel_ratio is not None:
        calibration = (channel_ratio if isinstance(channel_ratio, ChannelCalibration)
                       else ChannelCalibration(
                           ratio_per_frequency=np.full(frequencies_hz.shape,
                                                       complex(channel_ratio)),
                           ratio=complex(channel_ratio), relative_scatter=0.0,
                           phase_scatter_rad=0.0, reference_name="supplied",
                           emitter_offset_rad=float(emitter_offset_rad)))
        ratio = ellipsometric_ratio_from_channels(sample_fit.channel_ratio, calibration)
    else:
        raise ValueError("supply either reference_spectra + reference_index, or channel_ratio")

    inversion = invert_ratio(ratio[mask], frequencies_hz[mask], incidence_angle_rad,
                             index_incident=index_incident,
                             angular_spread_rad=angular_spread_rad,
                             reference_index=branch_reference_index)

    angle_fit = None
    if cross_check_index is not None:
        known = np.asarray(cross_check_index, dtype=complex)
        known = known[mask] if known.ndim else known
        angle_fit = fit_incidence_angle(ratio[mask], known, incidence_angle_rad,
                                        index_incident=index_incident)

    return EllipsometryResult(
        frequencies_hz=frequencies_hz,
        sample_fit=sample_fit,
        reference_fit=reference_fit,
        calibration=calibration,
        ratio=ratio,
        inversion=inversion,
        band=mask,
        incidence_angle_fit=angle_fit,
    )
