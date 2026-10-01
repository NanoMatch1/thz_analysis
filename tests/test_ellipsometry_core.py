"""Unit tests for the pure half of the ellipsometry package.

Known-answer wherever possible: Fresnel round trips, the perfect-conductor limit, the Brewster
angle, common-mode cancellation, planted-parameter recovery. Gates V1-V4 of the MVP plan.
"""

from __future__ import annotations

import numpy as np
import pytest

import ellipsometry as ell
from ellipsometry import calibration, harmonic, inversion, materials, model, pipeline, simulate


FREQUENCIES = np.linspace(0.8e12, 3.0e12, 40)
INCIDENCE_ANGLE = np.deg2rad(70.0)
EMITTER_ANGLES = np.linspace(0.0, np.pi, 12, endpoint=False)


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("index_sample", [3.4175 - 0j, 6.96 - 8.11j, 1.5 - 0.02j])
@pytest.mark.parametrize("angle_deg", [15.0, 45.0, 70.0, 85.0])
def test_ratio_inversion_round_trips(index_sample, angle_deg):
    angle = np.deg2rad(angle_deg)
    ratio = model.ellipsometric_ratio(index_sample, angle)
    recovered = model.index_from_ellipsometric_ratio(ratio, angle)
    assert abs(recovered - index_sample) < 1e-10


def test_common_mode_factor_cancels_exactly():
    """A scalar gain/timing factor multiplies r_p and r_s identically and divides out."""
    index_sample = 6.96 - 8.11j
    reflection_p, reflection_s = model.reflection_coefficients(index_sample, INCIDENCE_ANGLE)
    factor = 0.37 * np.exp(1j * 1.9)
    assert abs((reflection_p * factor) / (reflection_s * factor)
               - model.ellipsometric_ratio(index_sample, INCIDENCE_ANGLE)) < 1e-14


def test_perfect_conductor_ratio_approaches_minus_one():
    assert abs(model.ellipsometric_ratio(1e6 - 1e6j, INCIDENCE_ANGLE) + 1.0) < 1e-4


def test_brewster_minimum_is_at_arctan_n():
    angles = np.deg2rad(np.arange(60.0, 85.0, 0.01))
    magnitudes = np.abs(model.ellipsometric_ratio(3.4175 - 0j, angles))
    brewster = np.rad2deg(np.arctan(3.4175))
    assert abs(np.rad2deg(angles[int(np.argmin(magnitudes))]) - brewster) < 0.02


def test_detection_vector_is_balanced_at_the_documented_azimuth():
    vector = model.electro_optic_detection_vector(model.BALANCED_PROBE_AZIMUTH_RAD)
    assert vector[0] == pytest.approx(vector[1], rel=1e-12)
    assert vector[0] == pytest.approx(2.0 / np.sqrt(5.0), rel=1e-12)


@pytest.mark.parametrize("azimuth_deg,expected", [(0.0, True), (45.0, True), (90.0, True),
                                                  (31.72, False), (20.0, False)])
def test_degenerate_azimuths_are_flagged(azimuth_deg, expected):
    assert model.is_degenerate_azimuth(np.deg2rad(azimuth_deg)) is expected


def test_nearest_branch_avoids_the_sign_flip_for_a_lossless_sample():
    """Noise pushing k slightly negative must not flip the index to -N."""
    angle = np.deg2rad(70.0)
    index_sample = 3.4175 - 1e-5j
    ratio = model.ellipsometric_ratio(index_sample, angle)
    flipped_somewhere = False
    for nudge in (-2e-3, 2e-3, -5e-3, 5e-3):
        nudged = ratio * np.exp(1j * nudge)
        passive = model.index_from_ellipsometric_ratio(nudged, angle)
        nearest = model.index_from_ellipsometric_ratio(nudged, angle,
                                                       reference_index=index_sample)
        assert abs(nearest.real - 3.4175) < 0.05, f"nearest branch failed for nudge {nudge}"
        flipped_somewhere |= passive.real < 0
    assert flipped_somewhere, "the passive rule should flip sign for at least one nudge"


# ---------------------------------------------------------------------------
# materials.py
# ---------------------------------------------------------------------------

def test_absorption_and_extinction_are_inverse():
    for absorption, frequency in ((0.05, 1e12), (2.5, 3e12)):
        extinction = materials.extinction_from_power_absorption(absorption, frequency)
        assert materials.power_absorption_from_extinction(extinction, frequency) == pytest.approx(
            absorption, rel=1e-12)


def test_high_resistivity_silicon_k_is_far_below_any_measurement_floor():
    """The honest baseline: k ~ 1e-4, which no reflection measurement resolves."""
    index_sample = materials.high_resistivity_silicon_index(1e12)
    assert index_sample.real == pytest.approx(3.4175, rel=1e-12)
    assert 1e-5 < -index_sample.imag < 1e-3
    floor = ell.validation.extinction_measurement_floor(0.005, INCIDENCE_ANGLE, index_sample)
    assert floor > 10 * (-index_sample.imag)


def test_doped_silicon_approaches_the_undoped_limit():
    lightly_doped = materials.doped_silicon_index(1e12, 1e6)
    assert lightly_doped.real == pytest.approx(np.sqrt(materials.SILICON_STATIC_PERMITTIVITY),
                                               abs=1e-3)


@pytest.mark.parametrize("resistivity", [0.1, 1.0, 10.0])
def test_doped_silicon_is_passive_and_more_absorbing_when_more_doped(resistivity):
    index_sample = materials.doped_silicon_index(1e12, resistivity)
    assert index_sample.imag <= 0
    more_doped = materials.doped_silicon_index(1e12, resistivity / 10.0)
    assert -more_doped.imag > -index_sample.imag


def test_gold_is_a_near_perfect_mirror():
    """rho_gold is -1 to about 0.6% at 70 degrees -- and, more to the point, COMPUTABLE.

    What the calibration needs is not that rho_gold equals -1 but that we know it. The
    deviation from -1 is ~0.006 and only the ~20% uncertainty in the gold conductivity
    propagates, so rho_gold is known to roughly 0.1%.
    """
    index_gold = materials.gold_index(1e12)
    ratio = model.ellipsometric_ratio(index_gold, INCIDENCE_ANGLE)
    assert 1e-3 < abs(ratio + 1.0) < 1e-2
    shifted = model.ellipsometric_ratio(
        materials.gold_index(1e12, dc_conductivity_si=1.2 * 4.1e7), INCIDENCE_ANGLE)
    assert abs(shifted - ratio) < 2e-3


def test_unknown_reference_material_names_the_valid_options():
    with pytest.raises(KeyError, match="gold"):
        materials.reference_index("platinum", 1e12)


# ---------------------------------------------------------------------------
# harmonic.py  (V2)
# ---------------------------------------------------------------------------

def _planted_series(delays_s=None, relative_noise=0.0, seed=0, index_sample=None):
    index_sample = (materials.doped_silicon_index(FREQUENCIES, 1.0)
                    if index_sample is None else index_sample)
    return simulate.synthesize_spectra(
        index_sample, FREQUENCIES, emitter_angles_rad=EMITTER_ANGLES,
        incidence_angle_rad=INCIDENCE_ANGLE, relative_noise=relative_noise,
        drift_span_s=0.0 if delays_s is None else delays_s, seed=seed)


def test_harmonic_fit_is_exact_without_noise():
    measurement = _planted_series()
    fit = harmonic.fit_emitter_harmonic(EMITTER_ANGLES, measurement.spectra, FREQUENCIES,
                                        drift_model="none")
    assert fit.residual_norm < 1e-12
    expected = model.ellipsometric_ratio(measurement.true_index, INCIDENCE_ANGLE)
    recovered = fit.channel_ratio / measurement.true_channel_ratio
    assert np.max(np.abs(recovered - expected)) < 1e-10


def test_harmonic_residual_sees_unmodelled_drift():
    quiet = harmonic.harmonic_residual_norm(EMITTER_ANGLES, _planted_series().spectra,
                                            FREQUENCIES)
    drifting = harmonic.harmonic_residual_norm(
        EMITTER_ANGLES, _planted_series(delays_s=100e-15).spectra, FREQUENCIES)
    assert quiet < 1e-12
    assert drifting > 1e-3


def test_linear_ramp_removes_a_drift_the_plain_fit_cannot():
    measurement = _planted_series(delays_s=100e-15)
    expected = model.ellipsometric_ratio(measurement.true_index, INCIDENCE_ANGLE)

    uncorrected = harmonic.fit_emitter_harmonic(EMITTER_ANGLES, measurement.spectra,
                                                FREQUENCIES, drift_model="none")
    corrected = harmonic.fit_emitter_harmonic(EMITTER_ANGLES, measurement.spectra,
                                              FREQUENCIES, drift_model="linear_ramp")
    uncorrected_error = np.max(np.abs(
        uncorrected.channel_ratio / measurement.true_channel_ratio - expected))
    corrected_error = np.max(np.abs(
        corrected.channel_ratio / measurement.true_channel_ratio - expected))
    assert corrected_error < uncorrected_error / 50
    assert np.ptp(corrected.delays_s) == pytest.approx(100e-15, rel=0.05)


def test_common_delay_is_degenerate_and_does_not_disturb_the_ratio():
    """Gauge check: only differences between acquisitions matter."""
    measurement = _planted_series()
    shifted = measurement.spectra * np.exp(
        1j * 2.0 * np.pi * FREQUENCIES[None, :] * 7e-15)
    plain = harmonic.fit_emitter_harmonic(EMITTER_ANGLES, measurement.spectra, FREQUENCIES,
                                          drift_model="none")
    delayed = harmonic.fit_emitter_harmonic(EMITTER_ANGLES, shifted, FREQUENCIES,
                                            drift_model="none")
    assert np.max(np.abs(plain.channel_ratio - delayed.channel_ratio)) < 1e-10


def test_harmonic_fit_rejects_degenerate_and_malformed_input():
    measurement = _planted_series()
    with pytest.raises(ValueError, match="degenerate"):
        harmonic.fit_emitter_harmonic(np.zeros(12), measurement.spectra, FREQUENCIES)
    with pytest.raises(ValueError, match="at least two"):
        harmonic.fit_emitter_harmonic(EMITTER_ANGLES[:1], measurement.spectra[:1], FREQUENCIES)
    with pytest.raises(ValueError, match="rows"):
        harmonic.fit_emitter_harmonic(EMITTER_ANGLES[:5], measurement.spectra, FREQUENCIES)


def test_per_acquisition_drift_model_needs_enough_angles():
    measurement = simulate.synthesize_spectra(
        materials.high_resistivity_silicon_index(FREQUENCIES), FREQUENCIES,
        emitter_angles_rad=np.linspace(0.0, np.pi, 4, endpoint=False))
    with pytest.raises(ValueError, match="nuisance parameters"):
        harmonic.fit_emitter_harmonic(measurement.emitter_angles_rad, measurement.spectra,
                                      FREQUENCIES, drift_model="per_acquisition")


# ---------------------------------------------------------------------------
# calibration.py  (V3, V4)
# ---------------------------------------------------------------------------

def test_channel_ratio_recovered_from_a_gold_reference():
    gold = materials.gold_index(FREQUENCIES)
    reference = simulate.synthesize_spectra(gold, FREQUENCIES, emitter_angles_rad=EMITTER_ANGLES,
                                            incidence_angle_rad=INCIDENCE_ANGLE, seed=3)
    fit = harmonic.fit_emitter_harmonic(EMITTER_ANGLES, reference.spectra, FREQUENCIES,
                                        drift_model="none")
    result = calibration.channel_ratio_from_reference(fit.channel_ratio, gold, INCIDENCE_ANGLE)
    assert abs(result.ratio - reference.true_channel_ratio) < 5e-3
    assert result.is_flat


def test_channel_ratio_flatness_flags_a_frequency_dependent_fault():
    gold = materials.gold_index(FREQUENCIES)
    reference = simulate.synthesize_spectra(gold, FREQUENCIES, emitter_angles_rad=EMITTER_ANGLES,
                                            incidence_angle_rad=INCIDENCE_ANGLE, seed=4)
    fit = harmonic.fit_emitter_harmonic(EMITTER_ANGLES, reference.spectra, FREQUENCIES,
                                        drift_model="none")
    tilted = fit.channel_ratio * np.linspace(0.7, 1.3, FREQUENCIES.size)
    result = calibration.channel_ratio_from_reference(tilted, gold, INCIDENCE_ANGLE)
    assert not result.is_flat


def test_incidence_angle_fit_recovers_a_planted_angle():
    for planted_deg in (65.0, 70.0, 75.0):
        planted = np.deg2rad(planted_deg)
        index_sample = materials.high_resistivity_silicon_index(FREQUENCIES)
        ratio = model.ellipsometric_ratio(index_sample, planted)
        fit = calibration.fit_incidence_angle(ratio, index_sample, np.deg2rad(70.0))
        assert fit.angle_deg == pytest.approx(planted_deg, abs=0.01)


def test_incidence_angle_fit_is_precise_under_realistic_noise():
    planted = np.deg2rad(70.0)
    index_sample = materials.high_resistivity_silicon_index(FREQUENCIES)
    ratio = model.ellipsometric_ratio(index_sample, planted)
    generator = np.random.default_rng(11)
    recovered = []
    for _ in range(20):
        noisy = ratio * (1.0 + 0.005 * generator.normal(size=ratio.shape))
        recovered.append(calibration.fit_incidence_angle(noisy, index_sample, planted).angle_deg)
    assert np.std(recovered) < 0.05
    assert np.mean(recovered) == pytest.approx(70.0, abs=0.05)


# ---------------------------------------------------------------------------
# inversion.py
# ---------------------------------------------------------------------------

def test_zero_divergence_reproduces_the_plane_wave_answer():
    index_sample = 6.96 - 8.11j
    jones = inversion.effective_jones_with_divergence(index_sample, INCIDENCE_ANGLE, 0.0)
    reflection_p, reflection_s = model.reflection_coefficients(index_sample, INCIDENCE_ANGLE)
    assert abs(jones[0, 0] - reflection_p) < 1e-12
    assert abs(jones[1, 1] - reflection_s) < 1e-12
    assert abs(jones[0, 1]) < 1e-14


def test_symmetric_divergence_generates_no_cross_polarisation():
    """Parity: the off-diagonal terms are odd in the out-of-plane angle."""
    jones = inversion.effective_jones_with_divergence(6.96 - 8.11j, INCIDENCE_ANGLE,
                                                      np.deg2rad(6.0))
    assert abs(jones[0, 1] / jones[1, 1]) < 1e-10


def test_blur_aware_inversion_recovers_the_index_of_a_blurred_measurement():
    index_sample = materials.doped_silicon_index(FREQUENCIES, 1.0)
    spread = np.deg2rad(3.0)
    blurred = np.array([
        (lambda jones: jones[0, 0] / jones[1, 1])(
            inversion.effective_jones_with_divergence(single, INCIDENCE_ANGLE, spread))
        for single in index_sample])
    naive = inversion.invert_ratio(blurred, FREQUENCIES, INCIDENCE_ANGLE)
    corrected = inversion.invert_ratio(blurred, FREQUENCIES, INCIDENCE_ANGLE,
                                       angular_spread_rad=spread)
    assert np.max(np.abs(corrected.index - index_sample)) < 1e-6
    assert np.max(np.abs(naive.index - index_sample)) > 1e-3


def test_conductivity_is_positive_for_a_passive_sample():
    index_sample = materials.doped_silicon_index(FREQUENCIES, 1.0)
    result = inversion.invert_ratio(model.ellipsometric_ratio(index_sample, INCIDENCE_ANGLE),
                                    FREQUENCIES, INCIDENCE_ANGLE)
    assert np.all(result.conductivity_real_si > 0)


def test_invert_ratio_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must match"):
        inversion.invert_ratio(np.ones(5, dtype=complex), FREQUENCIES, INCIDENCE_ANGLE)


# ---------------------------------------------------------------------------
# pipeline.py  (V1 end to end, pure)
# ---------------------------------------------------------------------------

def _run_pipeline(index_sample, branch_reference=None, **kwargs):
    gold = materials.gold_index(FREQUENCIES)
    sample = simulate.synthesize_spectra(index_sample, FREQUENCIES,
                                         emitter_angles_rad=EMITTER_ANGLES,
                                         incidence_angle_rad=INCIDENCE_ANGLE, seed=5, **kwargs)
    reference = simulate.synthesize_spectra(gold, FREQUENCIES, emitter_angles_rad=EMITTER_ANGLES,
                                            incidence_angle_rad=INCIDENCE_ANGLE, seed=6,
                                            **kwargs)
    return pipeline.analyse_polarisation_series(
        frequencies_hz=FREQUENCIES,
        sample_spectra=sample.spectra, sample_emitter_angles_rad=EMITTER_ANGLES,
        reference_spectra=reference.spectra, reference_emitter_angles_rad=EMITTER_ANGLES,
        reference_index=gold, incidence_angle_rad=INCIDENCE_ANGLE,
        branch_reference_index=branch_reference)


def test_pipeline_round_trips_silicon_exactly():
    index_sample = materials.high_resistivity_silicon_index(FREQUENCIES)
    result = _run_pipeline(index_sample, branch_reference=3.4175)
    assert np.max(np.abs(result.index.real - 3.4175)) < 1e-9
    assert result.calibration.is_flat


def test_pipeline_round_trips_doped_silicon_exactly():
    index_sample = materials.doped_silicon_index(FREQUENCIES, 1.0)
    result = _run_pipeline(index_sample)
    assert np.max(np.abs(result.index - index_sample)) < 1e-9


@pytest.mark.parametrize("offset_deg,tolerance", [(0.1, 0.01), (0.2, 0.02)])
def test_emitter_offset_within_spec_stays_inside_tolerance(offset_deg, tolerance):
    """The mechanical specification: the emitter zero within ~0.2 deg of the incidence plane."""
    index_sample = materials.doped_silicon_index(FREQUENCIES, 1.0)
    result = _run_pipeline(index_sample, emitter_angle_offset_rad=np.deg2rad(offset_deg))
    assert np.median(np.abs(result.index.real - index_sample.real)) < tolerance


def test_a_large_emitter_offset_is_not_absorbed_by_a_single_reference():
    """Documented limitation: it is a Moebius transform, not a scale factor.

    An earlier draft of the design assumed a constant emitter offset would cancel against the
    gold reference. It does not -- the commanded 'p' setting contains sin(delta) of s, which
    mixes P and Q rather than scaling them.
    """
    index_sample = materials.doped_silicon_index(FREQUENCIES, 1.0)
    result = _run_pipeline(index_sample, emitter_angle_offset_rad=np.deg2rad(3.0))
    assert np.median(np.abs(result.index.real - index_sample.real)) > 0.1


def test_two_references_recover_the_emitter_offset():
    """Gold alone is degenerate; gold plus silicon separates the channel ratio from the offset."""
    offset = np.deg2rad(1.5)
    gold = materials.gold_index(FREQUENCIES)
    silicon = materials.high_resistivity_silicon_index(FREQUENCIES)
    measured = []
    for index_entry in (gold, silicon):
        series = simulate.synthesize_spectra(
            index_entry, FREQUENCIES, emitter_angles_rad=EMITTER_ANGLES,
            incidence_angle_rad=INCIDENCE_ANGLE, emitter_angle_offset_rad=offset)
        fit = harmonic.fit_emitter_harmonic(EMITTER_ANGLES, series.spectra, FREQUENCIES,
                                            drift_model="none")
        measured.append(fit.channel_ratio)

    recovered = calibration.fit_instrument_from_references(
        measured, [gold, silicon], INCIDENCE_ANGLE,
        reference_names=["gold", "hr_silicon"])
    assert recovered.emitter_offset_deg == pytest.approx(1.5, abs=0.02)
    assert abs(recovered.ratio - 1.0) < 1e-3


def test_single_reference_is_rejected_for_the_joint_instrument_fit():
    gold = materials.gold_index(FREQUENCIES)
    with pytest.raises(ValueError, match="at least two references"):
        calibration.fit_instrument_from_references([np.ones(FREQUENCIES.size, dtype=complex)],
                                                   [gold], INCIDENCE_ANGLE)


def test_a_known_emitter_offset_is_corrected_when_supplied():
    """If the offset is measured during setup, supplying it removes the error entirely."""
    offset = np.deg2rad(2.0)
    index_sample = materials.doped_silicon_index(FREQUENCIES, 1.0)
    gold = materials.gold_index(FREQUENCIES)
    sample = simulate.synthesize_spectra(index_sample, FREQUENCIES,
                                         emitter_angles_rad=EMITTER_ANGLES,
                                         incidence_angle_rad=INCIDENCE_ANGLE,
                                         emitter_angle_offset_rad=offset)
    reference = simulate.synthesize_spectra(gold, FREQUENCIES, emitter_angles_rad=EMITTER_ANGLES,
                                            incidence_angle_rad=INCIDENCE_ANGLE,
                                            emitter_angle_offset_rad=offset)
    result = pipeline.analyse_polarisation_series(
        frequencies_hz=FREQUENCIES,
        sample_spectra=sample.spectra, sample_emitter_angles_rad=EMITTER_ANGLES,
        reference_spectra=reference.spectra, reference_emitter_angles_rad=EMITTER_ANGLES,
        reference_index=gold, incidence_angle_rad=INCIDENCE_ANGLE,
        emitter_offset_rad=offset)
    assert np.max(np.abs(result.index - index_sample)) < 1e-9


def test_pipeline_survives_drift_with_the_ramp_model():
    index_sample = materials.doped_silicon_index(FREQUENCIES, 1.0)
    result = _run_pipeline(index_sample, drift_span_s=80e-15)
    assert np.max(np.abs(result.index - index_sample)) < 1e-6


def test_pipeline_reports_an_angle_cross_check():
    index_sample = materials.high_resistivity_silicon_index(FREQUENCIES)
    gold = materials.gold_index(FREQUENCIES)
    sample = simulate.synthesize_spectra(index_sample, FREQUENCIES,
                                         emitter_angles_rad=EMITTER_ANGLES,
                                         incidence_angle_rad=np.deg2rad(71.0), seed=7)
    reference = simulate.synthesize_spectra(gold, FREQUENCIES, emitter_angles_rad=EMITTER_ANGLES,
                                            incidence_angle_rad=np.deg2rad(71.0), seed=8)
    result = pipeline.analyse_polarisation_series(
        frequencies_hz=FREQUENCIES,
        sample_spectra=sample.spectra, sample_emitter_angles_rad=EMITTER_ANGLES,
        reference_spectra=reference.spectra, reference_emitter_angles_rad=EMITTER_ANGLES,
        reference_index=gold, incidence_angle_rad=np.deg2rad(70.0),
        cross_check_index=index_sample, branch_reference_index=3.4175)
    assert result.incidence_angle_fit.angle_deg == pytest.approx(71.0, abs=0.05)
    assert result.quality_flags["incidence_angle_discrepancy_deg"] == pytest.approx(1.0, abs=0.05)


def test_pipeline_requires_a_calibration_source():
    with pytest.raises(ValueError, match="channel_ratio"):
        pipeline.analyse_polarisation_series(
            frequencies_hz=FREQUENCIES, sample_spectra=np.zeros((12, 40), dtype=complex),
            sample_emitter_angles_rad=EMITTER_ANGLES, incidence_angle_rad=INCIDENCE_ANGLE)


def test_empty_band_is_an_explicit_error():
    index_sample = materials.high_resistivity_silicon_index(FREQUENCIES)
    gold = materials.gold_index(FREQUENCIES)
    sample = simulate.synthesize_spectra(index_sample, FREQUENCIES,
                                         emitter_angles_rad=EMITTER_ANGLES)
    with pytest.raises(ValueError, match="selects no frequencies"):
        pipeline.analyse_polarisation_series(
            frequencies_hz=FREQUENCIES, sample_spectra=sample.spectra,
            sample_emitter_angles_rad=EMITTER_ANGLES, incidence_angle_rad=INCIDENCE_ANGLE,
            channel_ratio=1.0, frequency_min_hz=9e12)


# ---------------------------------------------------------------------------
# validation.py  (V5)
# ---------------------------------------------------------------------------

def test_validation_passes_a_realistic_noisy_silicon_measurement():
    index_sample = materials.high_resistivity_silicon_index(FREQUENCIES)
    result = _run_pipeline(index_sample, branch_reference=3.4175, relative_noise=0.002)
    report = ell.validate_index_against_reference(
        result, index_sample, label="HR-Si V5", tolerance_n=0.02, tolerance_k=0.05)
    assert report.passed, str(report)


def test_validation_fails_loudly_on_a_wrong_answer():
    index_sample = materials.doped_silicon_index(FREQUENCIES, 1.0)
    result = _run_pipeline(index_sample)
    report = ell.validate_index_against_reference(result, 3.4175 - 0j, label="deliberate")
    assert not report.passed
    assert "FAIL" in str(report)
