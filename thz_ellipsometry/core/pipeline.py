"""The analysis chain, start to finish, with no file I/O and no DataSet.

Everything the measurement needs, expressed as pure functions of arrays. There is exactly ONE
inversion path, :func:`analyse_calibrated_series`: the stage driver calls it after choosing a
calibration source from the registry, and :func:`analyse_polarisation_series` -- the
arrays-in convenience wrapper the tests and the simulator use -- calls it after a gold
calibration. So there is no path through the analysis that only runs on real data, and no second
copy of the inversion to drift out of step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import (
    ChannelCalibration,
    channel_ratio_from_reference,
    ellipsometric_ratio_from_channels,
    fit_incidence_angle,
    ratio_derivative_wrt_measured,
    remove_emitter_offset,
)
from .harmonic import HarmonicFit, fit_emitter_harmonic
from .inversion import InversionResult, invert_ratio
from .sensitivity import (
    index_derivative_wrt_angle,
    index_standard_error_from_ratio,
)
from .tilt import TiltFit, fit_tilt_with_dispersion_model, invert_with_known_tilt

__all__ = [
    "EllipsometryResult",
    "analyse_calibrated_series",
    "analyse_polarisation_series",
    "band_mask",
]


def band_mask(frequencies_hz, frequency_min_hz=None, frequency_max_hz=None,
              spectra=None, minimum_relative_amplitude=0.0, channel_signal_to_noise=None,
              minimum_signal_to_noise=0.0):
    """Boolean mask selecting a trusted frequency band.

    Beyond the requested limits this drops two kinds of bin. Bins whose amplitude has fallen
    below a fraction of the series peak: without that floor the dead top end of the spectrum
    reaches the calibration as pure noise and a frequency-flat quantity scatters by tens of %.
    And, when a noise model is available, bins where either channel is below a signal-to-noise
    threshold (plan sec. 6.2: about ten).
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
    if channel_signal_to_noise is not None and minimum_signal_to_noise > 0.0:
        for entry in np.atleast_2d(channel_signal_to_noise):
            mask &= entry >= minimum_signal_to_noise
    return mask


@dataclass(frozen=True)
class EllipsometryResult:
    frequencies_hz: np.ndarray
    sample_fit: HarmonicFit
    reference_fit: HarmonicFit | None
    calibration: ChannelCalibration | None
    ratio: np.ndarray                     #: (n_frequencies,) rho, full axis
    inversion: InversionResult            #: band only
    band: np.ndarray
    incidence_angle_fit: object | None = None
    #: (n_frequencies,) Var(rho) from the measurement noise, full axis; None if unavailable.
    ratio_variance: np.ndarray | None = None
    #: (n_band,) standard error of n, equal to that of k (isotropic precision).
    index_standard_error: np.ndarray | None = None
    #: (n_band,) complex shift of N per +1 degree of incidence-angle error: a SYSTEMATIC.
    index_shift_per_degree: np.ndarray | None = None
    incidence_angle_uncertainty_deg: float = 0.0
    tilt_fit: TiltFit | None = None
    incidence_angle_source: str = "mechanical"

    @property
    def index(self):
        return self.inversion.index

    @property
    def index_angle_systematic(self):
        """|dN| for the stated incidence-angle uncertainty, per band frequency, or None."""
        if self.index_shift_per_degree is None or not self.incidence_angle_uncertainty_deg:
            return None
        return np.abs(self.index_shift_per_degree) * self.incidence_angle_uncertainty_deg

    @property
    def quality_flags(self):
        """Everything a run should be judged on before its numbers are believed."""
        flags = {
            "sample_harmonic_residual": self.sample_fit.residual_norm,
            "sample_fitted_drift_fs": self.sample_fit.fitted_drift_fs,
        }
        if self.sample_fit.reduced_chi_square is not None:
            flags["sample_reduced_chi_square"] = self.sample_fit.reduced_chi_square
        if self.reference_fit is not None:
            flags["reference_harmonic_residual"] = self.reference_fit.residual_norm
        if self.calibration is not None:
            flags["channel_ratio_scatter"] = self.calibration.relative_scatter
            flags["channel_ratio_phase_scatter_rad"] = self.calibration.phase_scatter_rad
            flags["channel_ratio_is_flat"] = self.calibration.is_flat
        if self.incidence_angle_fit is not None:
            flags["incidence_angle_discrepancy_deg"] = (
                self.incidence_angle_fit.discrepancy_deg)
        if self.index_standard_error is not None:
            flags["median_index_standard_error"] = float(np.median(self.index_standard_error))
        if self.tilt_fit is not None:
            flags["fitted_out_of_plane_tilt_deg"] = self.tilt_fit.tilt_deg
            flags["fitted_tilt_standard_error_deg"] = self.tilt_fit.tilt_standard_error_deg
        return flags


def _numerical_index_error_with_tilt(measured, variance, channel_ratio, tilt_rad, angle_rad,
                                     index_incident, index):
    """sigma_n (= sigma_k) for the tilt-corrected inversion, by a complex finite difference."""
    step = 1e-7 * np.maximum(np.abs(measured), 1e-12)
    shifted = invert_with_known_tilt(measured + step, channel_ratio, tilt_rad, angle_rad,
                                     index_incident=index_incident, reference_index=None)
    derivative = (shifted - index) / step
    return np.abs(derivative) * np.sqrt(np.maximum(variance, 0.0) / 2.0)


def analyse_calibrated_series(*, frequencies_hz, sample_fit, calibration, incidence_angle_rad,
                              band, index_incident=1.0, angular_spread_rad=0.0,
                              branch_reference_index=None, cross_check_index=None,
                              reference_fit=None, fit_out_of_plane_tilt=False,
                              tilt_model="drude", tilt_fixed_parameters=None,
                              tilt_initial_parameters=None,
                              incidence_angle_uncertainty_deg=0.0,
                              incidence_angle_source="mechanical"):
    """Calibrated harmonic fit -> rho -> n, k, with propagated noise and the angle systematic.

    ``cross_check_index``, if given, triggers a one-parameter incidence-angle fit against a
    sample of known index. That is a CROSS-CHECK on the angle actually used, reported and never
    silently applied.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    mask = np.asarray(band, dtype=bool)
    if not mask.any():
        raise ValueError("the band selects no frequencies")
    if fit_out_of_plane_tilt and angular_spread_rad > 0.0:
        raise ValueError("the tilt fit and the blur correction are not combined yet; enable one")

    measured = sample_fit.channel_ratio
    ratio = ellipsometric_ratio_from_channels(measured, calibration)
    measured_variance = sample_fit.channel_ratio_variance
    ratio_variance = None
    if measured_variance is not None:
        derivative = ratio_derivative_wrt_measured(measured, calibration.ratio,
                                                   calibration.emitter_offset_rad)
        ratio_variance = np.abs(derivative) ** 2 * measured_variance

    tilt_fit = None
    if fit_out_of_plane_tilt:
        offset_removed = remove_emitter_offset(measured, calibration.emitter_offset_rad)
        offset_variance = (None if measured_variance is None
                           else measured_variance[mask])
        tilt_fit = fit_tilt_with_dispersion_model(
            offset_removed[mask], calibration.ratio, frequencies_hz[mask], incidence_angle_rad,
            model_name=tilt_model, fixed=tilt_fixed_parameters,
            initial=tilt_initial_parameters, index_incident=index_incident,
            ratio_variance=offset_variance)
        index = invert_with_known_tilt(offset_removed[mask], calibration.ratio,
                                       tilt_fit.tilt_rad, incidence_angle_rad,
                                       index_incident=index_incident,
                                       reference_index=branch_reference_index)
        inversion = InversionResult(frequencies_hz=frequencies_hz[mask], ratio=ratio[mask],
                                    index=index, incidence_angle_rad=float(incidence_angle_rad),
                                    angular_spread_rad=0.0)
        index_standard_error = (
            None if offset_variance is None else _numerical_index_error_with_tilt(
                offset_removed[mask], offset_variance, calibration.ratio, tilt_fit.tilt_rad,
                incidence_angle_rad, index_incident, index))
    else:
        inversion = invert_ratio(ratio[mask], frequencies_hz[mask], incidence_angle_rad,
                                 index_incident=index_incident,
                                 angular_spread_rad=angular_spread_rad,
                                 reference_index=branch_reference_index)
        index_standard_error = (
            None if ratio_variance is None else index_standard_error_from_ratio(
                ratio[mask], ratio_variance[mask], incidence_angle_rad, index_incident,
                index=inversion.index))

    index_shift_per_degree = np.deg2rad(1.0) * index_derivative_wrt_angle(
        ratio[mask], incidence_angle_rad, index_incident,
        reference_index=inversion.index)

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
        ratio_variance=ratio_variance,
        index_standard_error=index_standard_error,
        index_shift_per_degree=index_shift_per_degree,
        incidence_angle_uncertainty_deg=float(incidence_angle_uncertainty_deg),
        tilt_fit=tilt_fit,
        incidence_angle_source=incidence_angle_source,
    )


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
    minimum_signal_to_noise=0.0,
    branch_reference_index=None,
    cross_check_index=None,
    sample_elapsed_seconds=None,
    reference_elapsed_seconds=None,
    background_term=False,
    sample_spectral_variance=None,
    reference_spectral_variance=None,
    fit_out_of_plane_tilt=False,
    tilt_model="drude",
    tilt_fixed_parameters=None,
    incidence_angle_uncertainty_deg=0.0,
):
    """Run the whole chain on arrays: harmonic fit -> channel calibration -> rho -> n, k.

    Exactly one of ``reference_spectra`` (with ``reference_index``) or ``channel_ratio`` must be
    supplied. The first is the normal path -- a gold mirror measured in the same geometry. The
    second exists for replaying a stored calibration, and for tests.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    fit_options = dict(drift_model=drift_model, background_term=background_term)
    sample_fit = fit_emitter_harmonic(sample_emitter_angles_rad, sample_spectra, frequencies_hz,
                                      elapsed_seconds=sample_elapsed_seconds,
                                      spectral_variance=sample_spectral_variance,
                                      **fit_options)

    reference_fit = None
    if reference_spectra is not None:
        if reference_index is None:
            raise ValueError("reference_spectra requires reference_index")
        if reference_emitter_angles_rad is None:
            reference_emitter_angles_rad = sample_emitter_angles_rad
        reference_fit = fit_emitter_harmonic(reference_emitter_angles_rad, reference_spectra,
                                             frequencies_hz,
                                             elapsed_seconds=reference_elapsed_seconds,
                                             spectral_variance=reference_spectral_variance,
                                             **fit_options)
    elif channel_ratio is None:
        raise ValueError("supply either reference_spectra + reference_index, or channel_ratio")

    amplitude_source = (sample_spectra if reference_spectra is None
                        else np.concatenate([np.atleast_2d(sample_spectra),
                                             np.atleast_2d(reference_spectra)]))
    signal_to_noise = [fit.channel_signal_to_noise for fit in (sample_fit, reference_fit)
                       if fit is not None and fit.channel_signal_to_noise is not None]
    mask = band_mask(frequencies_hz, frequency_min_hz, frequency_max_hz,
                     spectra=amplitude_source,
                     minimum_relative_amplitude=minimum_relative_amplitude,
                     channel_signal_to_noise=signal_to_noise or None,
                     minimum_signal_to_noise=minimum_signal_to_noise)
    if not mask.any():
        raise ValueError(
            f"the requested band selects no frequencies; data spans "
            f"{frequencies_hz.min()/1e12:.2f}-{frequencies_hz.max()/1e12:.2f} THz and the "
            f"amplitude floor is {minimum_relative_amplitude:.3g} of the peak")

    if reference_fit is not None:
        calibration = channel_ratio_from_reference(
            reference_fit.channel_ratio, reference_index, incidence_angle_rad,
            index_incident=index_incident, reference_name=reference_name,
            frequency_mask=mask, emitter_offset_rad=emitter_offset_rad)
    else:
        calibration = (channel_ratio if isinstance(channel_ratio, ChannelCalibration)
                       else ChannelCalibration(
                           ratio_per_frequency=np.full(frequencies_hz.shape,
                                                       complex(channel_ratio)),
                           ratio=complex(channel_ratio), relative_scatter=0.0,
                           phase_scatter_rad=0.0, reference_name="supplied",
                           emitter_offset_rad=float(emitter_offset_rad)))

    return analyse_calibrated_series(
        frequencies_hz=frequencies_hz, sample_fit=sample_fit, calibration=calibration,
        incidence_angle_rad=incidence_angle_rad, band=mask, index_incident=index_incident,
        angular_spread_rad=angular_spread_rad, branch_reference_index=branch_reference_index,
        cross_check_index=cross_check_index, reference_fit=reference_fit,
        fit_out_of_plane_tilt=fit_out_of_plane_tilt, tilt_model=tilt_model,
        tilt_fixed_parameters=tilt_fixed_parameters,
        incidence_angle_uncertainty_deg=incidence_angle_uncertainty_deg)
