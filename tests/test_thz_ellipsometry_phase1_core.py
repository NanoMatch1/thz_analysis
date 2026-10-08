"""Unit tests for the Phase 1 core of thz_ellipsometry (docs/THZ_ELLIPSOMETRY_IMPLEMENTATION_PLAN.md).

Each test is against a known answer: a closed form, a planted parameter, an equivalence that must
hold exactly, or a Monte Carlo that the propagated error bars must match.
"""

from __future__ import annotations

import numpy as np
import pytest

from thz_ellipsometry.core import (
    calibration,
    calibration_sources,
    detection,
    harmonic,
    materials,
    model,
    pipeline,
    preprocess,
    sensitivity,
    simulate,
    tilt,
)

FREQUENCIES = np.linspace(0.8e12, 3.0e12, 60)
INCIDENCE_ANGLE = np.deg2rad(45.0)
MAGNET_STATES = np.deg2rad(np.tile([0.0, 90.0, 180.0, 270.0], 3))
ELAPSED = np.arange(MAGNET_STATES.size) * 60.0
GOLD = materials.gold_index(FREQUENCIES)
DOPED = materials.doped_silicon_index(FREQUENCIES, 1.0)


def _series(index_sample, **options):
    options.setdefault("emitter_angles_rad", MAGNET_STATES)
    options.setdefault("incidence_angle_rad", INCIDENCE_ANGLE)
    return simulate.synthesize_spectra(index_sample, FREQUENCIES, **options)


# ---------------------------------------------------------------------------
# Harmonic fit: background, timestamps, weights
# ---------------------------------------------------------------------------

def test_background_column_equals_explicit_magnet_reversal_differencing():
    """For 0/90/180/270 the fitted background is EXACTLY the plan's +/-M differencing."""
    angles = np.deg2rad([0.0, 90.0, 180.0, 270.0])
    measured = _series(DOPED, emitter_angles_rad=angles, background_relative=0.3)
    fit = harmonic.fit_emitter_harmonic(angles, measured.spectra, FREQUENCIES,
                                        drift_model="none", background_term=True)
    channel_p = 0.5 * (measured.spectra[0] - measured.spectra[2])
    channel_s = 0.5 * (measured.spectra[1] - measured.spectra[3])
    assert np.allclose(fit.channel_p, channel_p, rtol=1e-12, atol=1e-14)
    assert np.allclose(fit.channel_s, channel_s, rtol=1e-12, atol=1e-14)


def test_background_is_removed_from_the_ratio_and_recovered_itself():
    clean = _series(DOPED)
    contaminated = _series(DOPED, background_relative=0.3)
    with_term = harmonic.fit_emitter_harmonic(MAGNET_STATES, contaminated.spectra, FREQUENCIES,
                                              drift_model="none", background_term=True)
    reference = harmonic.fit_emitter_harmonic(MAGNET_STATES, clean.spectra, FREQUENCIES,
                                              drift_model="none")
    assert np.allclose(with_term.channel_ratio, reference.channel_ratio, rtol=1e-10)
    expected_background = 0.3 * np.abs(clean.spectra).max() * simulate.default_emitted_spectrum(
        FREQUENCIES)
    assert np.allclose(with_term.background, expected_background, rtol=1e-10)


def test_two_angles_with_a_background_are_rejected_as_degenerate():
    angles = np.deg2rad([0.0, 90.0, 0.0, 90.0])
    measured = _series(DOPED, emitter_angles_rad=angles)
    with pytest.raises(ValueError, match="background"):
        harmonic.fit_emitter_harmonic(angles, measured.spectra, FREQUENCIES,
                                      drift_model="none", background_term=True)


def test_padded_bins_do_not_shrink_the_drift_error():
    """Neighbouring padded bins carry the same information: the shared drift's error must
    not fall by sqrt(oversampling) just because the FFT interpolated between them."""
    elapsed = np.arange(12) * 60.0
    measured = _series(DOPED, elapsed_seconds=elapsed, drift_span_s=20e-15, relative_noise=0.01,
                       seed=3)
    as_independent = harmonic.fit_emitter_harmonic(MAGNET_STATES, measured.spectra, FREQUENCIES,
                                                   elapsed_seconds=elapsed)
    oversampled = harmonic.fit_emitter_harmonic(MAGNET_STATES, measured.spectra, FREQUENCIES,
                                                elapsed_seconds=elapsed,
                                                frequency_oversampling=4.0)
    assert oversampled.fitted_drift_fs == pytest.approx(as_independent.fitted_drift_fs)
    np.testing.assert_allclose(oversampled.nuisance_parameter_errors,
                               2.0 * as_independent.nuisance_parameter_errors)
    assert oversampled.independent_degrees_of_freedom == pytest.approx(
        as_independent.degrees_of_freedom / 4.0)


def test_drift_ramp_runs_over_real_elapsed_time_not_acquisition_order():
    """An acquisition with a long pause: the ramp in time is right, the ramp in order is not."""
    elapsed = np.concatenate([np.arange(6) * 60.0, 3600.0 + np.arange(6) * 60.0])
    measured = _series(DOPED, elapsed_seconds=elapsed, drift_span_s=80e-15,
                       background_relative=0.05)
    in_time = harmonic.fit_emitter_harmonic(MAGNET_STATES, measured.spectra, FREQUENCIES,
                                            elapsed_seconds=elapsed, background_term=True)
    in_order = harmonic.fit_emitter_harmonic(MAGNET_STATES, measured.spectra, FREQUENCIES,
                                             background_term=True)
    assert in_time.fitted_drift_fs == pytest.approx(80.0, abs=0.01)
    assert np.allclose(in_time.delays_s, measured.true_delays_s, atol=1e-19)
    assert in_order.residual_norm > 100 * in_time.residual_norm


def _monte_carlo_ratio(weighted, trials=60, noise=0.003):
    ratios, predicted, chi_squares = [], [], []
    for seed in range(trials):
        measured = _series(DOPED, relative_noise=noise, elapsed_seconds=ELAPSED,
                           background_relative=0.05, drift_span_s=20e-15, seed=seed)
        variance = (np.full(measured.spectra.shape, (noise * np.abs(measured.spectra).max()) ** 2)
                    if weighted else None)
        fit = harmonic.fit_emitter_harmonic(MAGNET_STATES, measured.spectra, FREQUENCIES,
                                            elapsed_seconds=ELAPSED, background_term=True,
                                            spectral_variance=variance)
        ratios.append(fit.channel_ratio)
        predicted.append(fit.channel_ratio_variance)
        chi_squares.append(fit.reduced_chi_square)
    return np.array(ratios), np.array(predicted), chi_squares


def test_weighted_fit_covariance_matches_the_monte_carlo_scatter():
    ratios, predicted, chi_squares = _monte_carlo_ratio(weighted=True)
    observed = np.var(ratios, axis=0)
    assert np.median(observed / np.median(predicted, axis=0)) == pytest.approx(1.0, abs=0.2)
    assert np.mean(chi_squares) == pytest.approx(1.0, abs=0.1)


def test_unweighted_fit_estimates_its_covariance_from_the_residual():
    ratios, predicted, chi_squares = _monte_carlo_ratio(weighted=False)
    observed = np.var(ratios, axis=0)
    assert np.median(observed / np.median(predicted, axis=0)) == pytest.approx(1.0, abs=0.3)
    assert all(value is None for value in chi_squares)


def test_zero_spectral_variance_is_refused_rather_than_weighted_infinitely():
    measured = _series(DOPED)
    with pytest.raises(ValueError, match="zero everywhere"):
        harmonic.fit_emitter_harmonic(MAGNET_STATES, measured.spectra, FREQUENCIES,
                                      spectral_variance=np.zeros(measured.spectra.shape))


# ---------------------------------------------------------------------------
# Magnet offset: removing it from the angles IS the Moebius correction
# ---------------------------------------------------------------------------

def test_shifting_the_angles_by_a_known_offset_is_exactly_the_moebius_correction():
    offset = np.deg2rad(6.0)
    measured = _series(DOPED, emitter_angle_offset_rad=offset)
    shifted = harmonic.fit_emitter_harmonic(MAGNET_STATES + offset, measured.spectra,
                                            FREQUENCIES, drift_model="none")
    unshifted = harmonic.fit_emitter_harmonic(MAGNET_STATES, measured.spectra, FREQUENCIES,
                                              drift_model="none")
    assert np.allclose(calibration.remove_emitter_offset(unshifted.channel_ratio, offset),
                       shifted.channel_ratio, rtol=1e-10)


# ---------------------------------------------------------------------------
# Detection and the calibration-source registry
# ---------------------------------------------------------------------------

def test_detection_vector_in_sample_frame_reduces_to_the_crystal_frame_at_zero():
    azimuths = np.deg2rad([10.0, 31.72, 60.0])
    assert np.allclose(detection.detection_vector_in_sample_frame(azimuths, 0.0),
                       detection.electro_optic_detection_vector(azimuths))


def test_plan_mounting_detects_only_p_at_zero_probe_and_only_s_at_45():
    """Plan sec. 7.3: with [001] along s, probe at 0 deg sees E_p only, at 45 deg E_s only."""
    at_zero = detection.detection_vector_in_sample_frame(0.0, np.deg2rad(90.0))
    at_45 = detection.detection_vector_in_sample_frame(np.deg2rad(45.0), np.deg2rad(90.0))
    assert abs(at_zero[1]) < 1e-12 and abs(at_zero[0]) == pytest.approx(2.0)
    assert abs(at_45[0]) < 1e-12 and abs(at_45[1]) == pytest.approx(1.0)


@pytest.mark.parametrize("orientation_deg", [80.0, 90.0, 97.0])
def test_crystal_orientation_recovered_from_two_probe_settings(orientation_deg):
    orientation = np.deg2rad(orientation_deg)
    ratio_rho = model.ellipsometric_ratio(DOPED, INCIDENCE_ANGLE)
    probes = np.deg2rad([31.72, 15.0])
    measured = [detection.channel_ratio_for_probe(probe, orientation) * ratio_rho
                for probe in probes]
    solution = detection.crystal_orientation_from_probe_settings(
        measured, probes, nominal_orientation_rad=np.deg2rad(90.0))
    assert np.rad2deg(solution["orientation_rad"]) == pytest.approx(orientation_deg, abs=1e-6)
    assert solution["imaginary_fraction"] < 1e-10


def test_calibration_registry_lists_the_sources_with_their_help_text():
    assert {"gold_reference", "probe_rotation", "stored"} <= set(
        calibration_sources.calibration_source_names())
    for entry in calibration_sources.CALIBRATION_SOURCES.values():
        assert entry.summary and entry.requires


def _fits_at_probe(probe_deg, orientation_rad, seed=0):
    measured = _series(DOPED, probe_azimuth_rad=np.deg2rad(probe_deg),
                       crystal_orientation_rad=orientation_rad, elapsed_seconds=ELAPSED,
                       relative_noise=0.001, seed=seed)
    return harmonic.fit_emitter_harmonic(MAGNET_STATES, measured.spectra, FREQUENCIES,
                                         elapsed_seconds=ELAPSED), measured


def test_probe_rotation_and_gold_give_the_same_channel_ratio():
    orientation = np.deg2rad(92.0)
    primary, measured = _fits_at_probe(31.72, orientation)
    secondary, _ = _fits_at_probe(15.0, orientation, seed=1)
    gold_measured = _series(GOLD, probe_azimuth_rad=np.deg2rad(31.72),
                            crystal_orientation_rad=orientation, seed=2, relative_noise=0.001)
    gold_fit = harmonic.fit_emitter_harmonic(MAGNET_STATES, gold_measured.spectra, FREQUENCIES)
    inputs = calibration_sources.CalibrationInputs(
        frequencies_hz=FREQUENCIES, band=np.ones(FREQUENCIES.shape, dtype=bool),
        incidence_angle_rad=INCIDENCE_ANGLE, channel_reference_fit=gold_fit,
        channel_reference_index=GOLD,
        sample_fits_by_probe_deg={31.72: primary, 15.0: secondary}, primary_probe_deg=31.72,
        crystal_orientation_rad=np.deg2rad(90.0))
    from_gold = calibration_sources.compute_channel_calibration("gold_reference", inputs)
    from_probe = calibration_sources.compute_channel_calibration("probe_rotation", inputs)
    assert from_probe.ratio == pytest.approx(measured.true_channel_ratio, rel=2e-3)
    assert from_gold.ratio == pytest.approx(from_probe.ratio, rel=3e-3)


def test_a_calibration_source_names_what_it_is_missing():
    inputs = calibration_sources.CalibrationInputs(
        frequencies_hz=FREQUENCIES, band=np.ones(FREQUENCIES.shape, dtype=bool),
        incidence_angle_rad=INCIDENCE_ANGLE)
    with pytest.raises(ValueError, match="two or more probe settings"):
        calibration_sources.compute_channel_calibration("probe_rotation", inputs)
    with pytest.raises(ValueError, match="channel_reference_fit"):
        calibration_sources.compute_channel_calibration("gold_reference", inputs)
    with pytest.raises(KeyError, match="registered"):
        calibration_sources.compute_channel_calibration("no_such_source", inputs)


# ---------------------------------------------------------------------------
# Tilt: fittable on a dispersive sample, degenerate on a flat one
# ---------------------------------------------------------------------------

def test_known_tilt_inversion_is_exact_without_noise():
    channel = -0.97 + 0.05j
    measured = tilt.tilted_channel_ratio(DOPED, np.deg2rad(0.8), channel, INCIDENCE_ANGLE)
    recovered = tilt.invert_with_known_tilt(measured, channel, np.deg2rad(0.8), INCIDENCE_ANGLE)
    assert np.allclose(recovered, DOPED, atol=1e-9)


def test_tilt_and_resistivity_recovered_from_a_dispersive_sample_alone():
    channel = -1.0 + 0.0j
    generator = np.random.default_rng(3)
    measured = tilt.tilted_channel_ratio(DOPED, np.deg2rad(0.5), channel, INCIDENCE_ANGLE)
    measured = measured + 0.003 * np.abs(measured).mean() * (
        generator.normal(size=measured.size) + 1j * generator.normal(size=measured.size))
    fit = tilt.fit_tilt_with_dispersion_model(measured, channel, FREQUENCIES, INCIDENCE_ANGLE,
                                              model_name="drude", fixed={"eps_inf": 11.68})
    assert fit.tilt_deg == pytest.approx(0.5, abs=4 * fit.tilt_standard_error_deg)
    assert fit.tilt_standard_error_deg < 0.05
    assert 10.0 ** fit.model_parameters["log10_resistivity_ohm_cm"] == pytest.approx(1.0,
                                                                                     rel=0.1)


def test_tilt_on_a_flat_sample_is_reported_as_undetermined_not_as_tight():
    """The trap the pseudo-inverse fell into: a rank-deficient fit must say so."""
    flat = materials.high_resistivity_silicon_index(FREQUENCIES)
    generator = np.random.default_rng(3)
    measured = tilt.tilted_channel_ratio(flat, np.deg2rad(0.5), -1.0, INCIDENCE_ANGLE)
    measured = measured + 0.003 * (generator.normal(size=measured.size)
                                   + 1j * generator.normal(size=measured.size))
    fit = tilt.fit_tilt_with_dispersion_model(measured, -1.0, FREQUENCIES, INCIDENCE_ANGLE,
                                              model_name="constant")
    assert not np.isfinite(fit.tilt_standard_error_deg)


def test_unknown_dispersion_model_or_parameter_is_explained():
    with pytest.raises(KeyError, match="registered"):
        tilt.fit_tilt_with_dispersion_model(np.ones(3), 1.0, FREQUENCIES[:3], INCIDENCE_ANGLE,
                                            model_name="lorentz")
    with pytest.raises(ValueError, match="cannot fix"):
        tilt.fit_tilt_with_dispersion_model(np.ones(3), 1.0, FREQUENCIES[:3], INCIDENCE_ANGLE,
                                            fixed={"plasma": 1.0})


# ---------------------------------------------------------------------------
# Sensitivity (plan sec. 8)
# ---------------------------------------------------------------------------

def test_angle_error_affine_map_reproduces_the_plan_example():
    scale, offset = sensitivity.angle_error_affine_coefficients(np.deg2rad(42.0),
                                                                np.deg2rad(45.0))
    assert scale == pytest.approx(1.38, abs=0.005)
    assert offset == pytest.approx(-0.12, abs=0.005)
    assert scale * 11.68 + offset == pytest.approx(15.97, abs=0.01)


def test_angle_error_affine_map_is_exact_for_any_material():
    true_angle, assumed_angle = np.deg2rad(43.3), np.deg2rad(45.0)
    scale, offset = sensitivity.angle_error_affine_coefficients(true_angle, assumed_angle)
    ratio = model.ellipsometric_ratio(DOPED, true_angle)
    misread = model.index_from_ellipsometric_ratio(ratio, assumed_angle) ** 2
    assert np.allclose(misread, scale * DOPED**2 + offset, rtol=1e-10)


def test_permittivity_derivative_matches_a_finite_difference():
    ratio = model.ellipsometric_ratio(DOPED, INCIDENCE_ANGLE)
    step = 1e-7
    numeric = (model.index_from_ellipsometric_ratio(ratio + step, INCIDENCE_ANGLE) ** 2
               - model.index_from_ellipsometric_ratio(ratio, INCIDENCE_ANGLE) ** 2) / step
    analytic = sensitivity.permittivity_derivative_wrt_ratio(ratio, INCIDENCE_ANGLE)
    assert np.allclose(analytic, numeric, rtol=1e-5)


def test_index_error_bars_match_the_monte_carlo_scatter_end_to_end():
    """The headline Phase 1 claim: the bars on n and k are the measured scatter, and equal."""
    noise = 0.003
    indices, bars = [], []
    for seed in range(40):
        sample = _series(DOPED, relative_noise=noise, elapsed_seconds=ELAPSED,
                         background_relative=0.05, drift_span_s=20e-15, seed=seed)
        gold = _series(GOLD, elapsed_seconds=ELAPSED, seed=100 + seed)
        result = pipeline.analyse_polarisation_series(
            frequencies_hz=FREQUENCIES, sample_spectra=sample.spectra,
            sample_emitter_angles_rad=MAGNET_STATES, incidence_angle_rad=INCIDENCE_ANGLE,
            reference_spectra=gold.spectra, reference_index=GOLD,
            sample_elapsed_seconds=ELAPSED, reference_elapsed_seconds=ELAPSED,
            background_term=True, minimum_relative_amplitude=0.0, branch_reference_index=DOPED,
            sample_spectral_variance=np.full(sample.spectra.shape,
                                             (noise * np.abs(sample.spectra).max()) ** 2))
        indices.append(result.index)
        bars.append(result.index_standard_error)
    indices, bar = np.array(indices), np.median(bars, axis=0)
    assert np.median(indices.real.std(axis=0) / bar) == pytest.approx(1.0, abs=0.15)
    assert np.median(indices.imag.std(axis=0) / bar) == pytest.approx(1.0, abs=0.15)


def test_tilt_route_through_the_pipeline_recovers_the_index():
    sample = _series(DOPED, out_of_plane_tilt_rad=np.deg2rad(1.0), elapsed_seconds=ELAPSED,
                     relative_noise=0.001, seed=4)
    gold = _series(GOLD, elapsed_seconds=ELAPSED, seed=5)
    common = dict(frequencies_hz=FREQUENCIES, sample_spectra=sample.spectra,
                  sample_emitter_angles_rad=MAGNET_STATES, incidence_angle_rad=INCIDENCE_ANGLE,
                  reference_spectra=gold.spectra, reference_index=GOLD,
                  sample_elapsed_seconds=ELAPSED, reference_elapsed_seconds=ELAPSED,
                  minimum_relative_amplitude=0.0, branch_reference_index=DOPED)
    ignored = pipeline.analyse_polarisation_series(**common)
    fitted = pipeline.analyse_polarisation_series(
        **common, fit_out_of_plane_tilt=True, tilt_fixed_parameters={"eps_inf": 11.68})
    assert np.median(np.abs(ignored.index - DOPED)) > 0.1
    assert np.median(np.abs(fitted.index - DOPED)) < 0.01
    assert fitted.tilt_fit.tilt_deg == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# Simulator and preprocessing
# ---------------------------------------------------------------------------

def test_interleaved_schedule_cycles_every_state_in_order():
    schedule = simulate.interleaved_schedule([0.0, 90.0, 180.0, 270.0], cycles=3,
                                             seconds_per_acquisition=30.0)
    assert len(schedule) == 12
    assert np.allclose(np.rad2deg(schedule.polarization_angles_rad[:4]), [0, 90, 180, 270])
    assert np.allclose(schedule.cycle_numbers, np.repeat([1, 2, 3], 4))
    assert np.allclose(np.diff(schedule.elapsed_seconds), 30.0)


def test_acquisitions_carry_independent_noise_per_scan():
    acquisitions = simulate.synthesize_acquisitions(
        materials.high_resistivity_silicon_index, polarization_angles_rad=np.deg2rad([0.0, 90.0]),
        scans_per_acquisition=5, relative_noise=0.01, seed=1)
    assert acquisitions.scans.shape[:2] == (2, 5)
    assert not np.allclose(acquisitions.scans[0, 0], acquisitions.scans[0, 1])


def test_preprocessing_returns_the_window_it_applied():
    time_ps = np.arange(512) * 0.05
    traces = np.exp(-((time_ps - 5.0) / 0.2) ** 2)[None, :]
    transformed = preprocess.transform_traces(time_ps, traces, window_half_width_ps=2.0)
    assert transformed.window.shape == time_ps.shape
    assert transformed.window.max() == pytest.approx(1.0, abs=0.01)
    assert transformed.window[0] == 0.0
    assert transformed.padded_length == 4 * 512
    assert not transformed.window_placement.clipped
    # A ~4 ps window in a 4 x 25.6 ps FFT: one independent point per ~26 bins.
    assert transformed.resolution.t_resolution_s == pytest.approx(4.0e-12, rel=0.03)
    assert transformed.resolution.oversampling == pytest.approx(
        4 * 25.6e-12 / transformed.resolution.t_resolution_s)


def test_a_window_past_the_record_start_is_truncated_not_resized():
    """Short pre-pulse record (the bench default): the flat top must stay on the pulse."""
    short_record = np.arange(141) * 0.05                # 7 ps, pulse 2.35 ps in
    long_record = np.arange(201) * 0.05                 # 10 ps, pulse 4.35 ps in
    clipped = preprocess.transform_traces(
        short_record, np.exp(-((short_record - 2.35) / 0.15) ** 2)[None, :],
        window_half_width_ps=3.0, window_centre_ps=2.35)
    roomy = preprocess.transform_traces(
        long_record, np.exp(-((long_record - 4.35) / 0.15) ** 2)[None, :],
        window_half_width_ps=3.0, window_centre_ps=4.35)
    assert clipped.window_placement.clipped_before == 13
    assert not roomy.window_placement.clipped
    # The same weights relative to the centre (index 47 vs 87), past the edge taper...
    offset = 87 - 47
    np.testing.assert_allclose(clipped.window[10:131], roomy.window[10 + offset:131 + offset])
    # ...the flat top on the pulse, and the record's own start ramped to zero.
    assert clipped.window[47] == pytest.approx(1.0)
    assert clipped.window[0] == 0.0
    # Shorter support -> coarser true resolution than the unclipped window.
    assert clipped.resolution.df_resolution_hz > roomy.resolution.df_resolution_hz


# ---------------------------------------------------------------------------
# Emitter calibration and the amplitude-drift nuisance
# ---------------------------------------------------------------------------

def test_reading_and_polarisation_maps_are_inverse_and_periodic():
    from thz_ellipsometry.core import emitter
    table = {0.0: 86.3, 90.0: 176.8, 180.0: 266.0, 270.0: 356.9}
    polarisations = np.array([0.0, 37.0, 90.0, 180.0, 300.0, -30.0, 400.0])
    readings = emitter.reading_from_polarization(polarisations, table)
    assert np.allclose(emitter.polarization_from_reading(readings, table), polarisations)
    assert np.allclose(emitter.polarization_from_reading([86.3, 176.8, 266.0, 356.9], table),
                       [0.0, 90.0, 180.0, 270.0])
    assert np.allclose(emitter.polarization_from_reading([90.0], {0.0: 86.3}), [3.7])


def test_a_table_whose_readings_run_backwards_is_rejected():
    from thz_ellipsometry.core import emitter
    with pytest.raises(ValueError, match="rotation sense"):
        emitter.polarization_from_reading([0.0], {0.0: 90.0, 90.0: 0.0, 180.0: 270.0})


def test_amplitude_ramp_is_fitted_and_without_it_the_ratio_is_biased():
    measured = _series(DOPED, elapsed_seconds=ELAPSED, amplitude_drift=0.03,
                       background_relative=0.05, relative_noise=0.0005, seed=7)
    clean = _series(DOPED, elapsed_seconds=ELAPSED)
    truth = harmonic.fit_emitter_harmonic(MAGNET_STATES, clean.spectra, FREQUENCIES,
                                          elapsed_seconds=ELAPSED, drift_model="none")
    fitted = harmonic.fit_emitter_harmonic(MAGNET_STATES, measured.spectra, FREQUENCIES,
                                           elapsed_seconds=ELAPSED, background_term=True,
                                           amplitude_model="linear_ramp")
    ignored = harmonic.fit_emitter_harmonic(MAGNET_STATES, measured.spectra, FREQUENCIES,
                                            elapsed_seconds=ELAPSED, background_term=True)
    assert fitted.fitted_amplitude_change == pytest.approx(0.03, abs=0.002)
    error_fitted = np.median(np.abs(fitted.channel_ratio / truth.channel_ratio - 1.0))
    error_ignored = np.median(np.abs(ignored.channel_ratio / truth.channel_ratio - 1.0))
    assert error_fitted < 0.002 < error_ignored


def test_spectral_tilt_ramp_is_recovered_and_a_flat_ramp_cannot_absorb_it():
    """The purge's water signature on a short record is a log-amplitude TILT (F40)."""
    measured = _series(DOPED, elapsed_seconds=ELAPSED, tilt_drift_per_thz=0.04,
                       background_relative=0.05, relative_noise=0.0005, seed=8)
    clean = _series(DOPED, elapsed_seconds=ELAPSED)
    truth = harmonic.fit_emitter_harmonic(MAGNET_STATES, clean.spectra, FREQUENCIES,
                                          elapsed_seconds=ELAPSED, drift_model="none")
    tilted = harmonic.fit_emitter_harmonic(MAGNET_STATES, measured.spectra, FREQUENCIES,
                                           elapsed_seconds=ELAPSED, background_term=True,
                                           amplitude_model="tilt_ramp")
    flat = harmonic.fit_emitter_harmonic(MAGNET_STATES, measured.spectra, FREQUENCIES,
                                         elapsed_seconds=ELAPSED, background_term=True,
                                         amplitude_model="linear_ramp")
    assert tilted.fitted_tilt_change_per_thz == pytest.approx(0.04, abs=0.003)
    error_tilted = np.median(np.abs(tilted.channel_ratio / truth.channel_ratio - 1.0))
    error_flat = np.median(np.abs(flat.channel_ratio / truth.channel_ratio - 1.0))
    assert error_tilted < 0.002 < error_flat


def test_auto_nuisance_models_are_linear_for_one_pass_and_curved_for_a_palindrome():
    from thz_ellipsometry.adapters.stages import resolve_nuisance_models
    assert resolve_nuisance_models("auto", "auto", 4) == ("linear_ramp", "tilt_ramp")
    assert resolve_nuisance_models("auto", "auto", 8) == ("settling", "tilt_settling")
    assert resolve_nuisance_models("none", "linear_ramp", 8) == ("none", "linear_ramp")


def test_settling_profile_spans_zero_to_one_and_is_the_exact_exponential():
    abscissa = np.linspace(0.0, 1.0, 9)
    assert np.allclose(harmonic.settling_profile(abscissa, 0.0), abscissa)
    for rate in (-2.0, 0.5, 3.0):
        profile = harmonic.settling_profile(abscissa, rate)
        assert profile[0] == pytest.approx(0.0) and profile[-1] == pytest.approx(1.0)
        exponential = 1.0 - np.exp(-rate * abscissa)
        assert np.allclose(profile, exponential / exponential[-1])


# ---------------------------------------------------------------------------
# Per-scan rows: drift measured inside each acquisition (segment_settling)
# ---------------------------------------------------------------------------

SCAN_SECONDS = 48.0
SCANS_PER_STATE = 12
SINGLE_PASS_DEG = (270.0, 180.0, 90.0, 0.0)


def _single_pass_scan_rows(gap_seconds=120.0):
    """One pass through the four states, a scan every 48 s, a gap (box open) between states."""
    angles, elapsed, segments = [], [], []
    start = 0.0
    for segment, angle in enumerate(SINGLE_PASS_DEG):
        times = start + SCAN_SECONDS * np.arange(SCANS_PER_STATE)
        angles += [np.deg2rad(angle)] * SCANS_PER_STATE
        elapsed += list(times)
        segments += [segment] * SCANS_PER_STATE
        start = times[-1] + SCAN_SECONDS + gap_seconds
    return np.array(angles), np.array(elapsed), np.array(segments)


def _bench_like_delays(elapsed, segments):
    """The 2026-10-07 shape: a purge settling over the block plus a short settle per opening."""
    trend = -25e-15 * harmonic.settling_profile(elapsed / np.ptp(elapsed), 2.0)
    transients = harmonic.segment_transients([-6.0, -5.0, -8.0, -10.0], 0.5, elapsed, segments)
    return trend + transients


def test_segment_settling_is_exact_where_a_file_average_is_not():
    """Single pass, per-scan rows: the in-file drift pins the model; per-file averages cannot."""
    angles, elapsed, segments = _single_pass_scan_rows()
    delays = _bench_like_delays(elapsed, segments)
    measured = _series(DOPED, emitter_angles_rad=angles, elapsed_seconds=elapsed,
                       drift_span_s=delays, background_relative=0.1)
    clean = _series(DOPED, emitter_angles_rad=angles)
    truth = harmonic.fit_emitter_harmonic(angles, clean.spectra, FREQUENCIES,
                                          drift_model="none").channel_ratio

    per_scan = harmonic.fit_emitter_harmonic(angles, measured.spectra, FREQUENCIES,
                                             drift_model="segment_settling",
                                             elapsed_seconds=elapsed, segment_ids=segments,
                                             background_term=True)
    averaged = np.vstack([measured.spectra[segments == k].mean(axis=0) for k in range(4)])
    file_times = np.array([elapsed[segments == k].mean() for k in range(4)])
    per_file = harmonic.fit_emitter_harmonic(np.deg2rad(SINGLE_PASS_DEG), averaged, FREQUENCIES,
                                             drift_model="linear_ramp",
                                             elapsed_seconds=file_times, background_term=True)

    assert np.max(np.abs(per_scan.channel_ratio / truth - 1.0)) < 1e-8
    assert np.max(np.abs(per_file.channel_ratio / truth - 1.0)) > 1e-3
    assert np.array_equal(per_scan.segment_ids, segments)


def test_segment_settling_recovers_the_planted_drift_through_noise():
    angles, elapsed, segments = _single_pass_scan_rows()
    measured = _series(DOPED, emitter_angles_rad=angles, elapsed_seconds=elapsed,
                       drift_span_s=_bench_like_delays(elapsed, segments),
                       relative_noise=1e-3, background_relative=0.1)
    fit = harmonic.fit_emitter_harmonic(angles, measured.spectra, FREQUENCIES,
                                        drift_model="segment_settling", elapsed_seconds=elapsed,
                                        segment_ids=segments, background_term=True)
    values = dict(zip(fit.nuisance_parameter_names, fit.nuisance_parameters))
    assert values["delay_span_fs"] == pytest.approx(-25.0, abs=0.2)
    assert values["settling_rate"] == pytest.approx(2.0, abs=0.1)
    assert values["transient_decay_minutes"] == pytest.approx(0.5, abs=0.05)
    assert [values[f"transient_{k}_fs"] for k in range(4)] == pytest.approx(
        [-6.0, -5.0, -8.0, -10.0], abs=0.2)


def test_segment_settling_needs_segments_and_times():
    angles, elapsed, segments = _single_pass_scan_rows()
    measured = _series(DOPED, emitter_angles_rad=angles)
    for missing in ({"elapsed_seconds": elapsed}, {"segment_ids": segments}):
        with pytest.raises(ValueError, match="per-scan rows"):
            harmonic.fit_emitter_harmonic(angles, measured.spectra, FREQUENCIES,
                                          drift_model="segment_settling", **missing)


def test_segment_ids_must_be_contiguous_numbers():
    angles, elapsed, segments = _single_pass_scan_rows()
    measured = _series(DOPED, emitter_angles_rad=angles)
    with pytest.raises(ValueError, match="0..n-1"):
        harmonic.fit_emitter_harmonic(angles, measured.spectra, FREQUENCIES,
                                      drift_model="segment_settling", elapsed_seconds=elapsed,
                                      segment_ids=segments * 2)


def test_revisit_lever_arm_ignores_repeats_inside_one_acquisition():
    """Scans of one file repeat each other trivially; only a return visit pins a jump."""
    # 0 and 90 deg on doped Si are opposite in sign (rho ~ -0.6), so never "the same signal".
    angles = np.deg2rad([0.0, 0.0, 90.0, 90.0])
    elapsed = np.arange(angles.size) * 60.0
    measured = _series(DOPED, emitter_angles_rad=angles)
    fits = {layout: harmonic.fit_emitter_harmonic(angles, measured.spectra, FREQUENCIES,
                                                  drift_model="none", elapsed_seconds=elapsed,
                                                  segment_ids=ids)
            for layout, ids in (("rows", None), ("scans", np.arange(angles.size) // 2))}
    assert fits["rows"].segment_ids is None
    assert harmonic.revisit_lever_arm(fits["rows"]) == pytest.approx(1.0 / 3.0)
    assert harmonic.revisit_lever_arm(fits["scans"]) == 0.0


def test_existing_drift_models_still_take_the_row_count_alone():
    """The registry gained a segment count; the models without segments ignore it."""
    for name, names_for in harmonic.DRIFT_MODELS.items():
        if name not in harmonic.SEGMENTED_DRIFT_MODELS:
            assert names_for(5) == names_for(5, 3)
