"""Ellipsometry assumptions, registered in the repo's ``@diagnostic`` registry.

Each check is registered ONCE, at its definition, with the assumption it guards, why it matters
and what to do -- so it appears in the generated ``docs/assumptions_ledger.md`` alongside the
rest of the pipeline's assumptions, with no second list to maintain. Diagnostics never halt a
run; they report.

The registry is shared with the DataSet pipeline, so the stage names are prefixed
``ellipsometry.`` and an ellipsometry run evaluates only those stages. Every check also returns
immediately when handed anything other than an ellipsometry ``RunOutcome``, so a DataSet run
that happens to have imported this module is unaffected.
"""

from __future__ import annotations

import numpy as np

from dataset_core.adapters.diagnostics import (
    Problem,
    Severity,
    diagnostic,
    registered_stages,
    run_diagnostics,
)

from ..core.detection import is_degenerate_azimuth
from ..core.materials import reference_index
from ..core.model import index_from_ellipsometric_ratio
from ..core.sensitivity import permittivity_derivative_wrt_ratio
from .loader import SAMPLE_ROLE

__all__ = ["ELLIPSOMETRY_STAGE_PREFIX", "run_ellipsometry_diagnostics"]

ELLIPSOMETRY_STAGE_PREFIX = "ellipsometry."
ACQUISITION = "ellipsometry.acquisition"
CALIBRATION = "ellipsometry.calibration"
INVERSION = "ellipsometry.inversion"


def _is_outcome(candidate):
    return hasattr(candidate, "fits") and hasattr(candidate, "primary_probe_deg") and hasattr(
        candidate, "result")


def run_ellipsometry_diagnostics(outcome):
    """Every registered ellipsometry diagnostic, run against a finished RunOutcome."""
    findings = []
    for stage in registered_stages():
        if stage.startswith(ELLIPSOMETRY_STAGE_PREFIX):
            findings.extend(run_diagnostics(outcome, outcome.config, stage=stage))
    return findings


def _band(outcome):
    return outcome.result.band


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

@diagnostic(
    stage=ACQUISITION,
    assumption="the probe polarisation is not at a degenerate azimuth of the EO crystal",
    why="At 0 or 45 deg from [001] one THz component produces no signal at all, so P/Q is a "
        "ratio with a zero in it and rho cannot be measured, however long you average.",
    remedy="Rotate the probe (or the crystal, once) to about 31.7 deg from [001], where both "
           "channels are equally sensitive.",
    severity=Severity.FAIL)
def probe_degeneracy(outcome, config):
    if not _is_outcome(outcome):
        return
    for (role, probe) in outcome.series:
        if probe is not None and is_degenerate_azimuth(np.deg2rad(probe)):
            yield Problem(f"{role} measured at probe {probe:g} deg, a degenerate azimuth")


@diagnostic(
    stage=ACQUISITION,
    assumption="every acquisition is a pure first harmonic in polarisation angle plus a "
               "constant background, up to the fitted drift",
    why="Anything outside that form -- a magnet that does not saturate, a drift the ramp "
        "cannot describe, a misread filename -- is not cancelled by the ratio and goes straight "
        "into rho.",
    remedy="Inspect the per-frequency harmonic residual in the run figure; check magnet "
           "saturation and the acquisition order; try drift_model='per_acquisition' if the "
           "drift is not monotonic.")
def harmonic_model_holds(outcome, config):
    if not _is_outcome(outcome):
        return
    band = _band(outcome)
    for key, fit in outcome.fits.items():
        label = f"{key[0]} @ probe {key[1]}"
        if fit.reduced_chi_square is not None:
            if fit.reduced_chi_square > 3.0:
                yield Problem(f"{label}: reduced chi-square {fit.reduced_chi_square:.2f} "
                              "against the measured noise (should be ~1)",
                              detail={"reduced_chi_square": fit.reduced_chi_square})
        else:
            residual = float(np.median(fit.residual_per_frequency[band]))
            if residual > 0.05:
                yield Problem(f"{label}: median relative harmonic residual {residual:.3f} in "
                              "the band (no noise model, so judged against 5%)",
                              detail={"median_residual": residual})


@diagnostic(
    stage=ACQUISITION,
    assumption="the non-magnetic background is small compared with the magnetic signal",
    why="The background term removes it exactly only if it is truly independent of the magnet "
        "state. A large background makes any magnet-dependent part of it (stray field reaching "
        "the sample, a substrate signal that depends on M) a first-order error.",
    remedy="Find the source: optical rectification in the emitter substrate, pump leakage onto "
           "the detector, or electronic pickup. Block the pump to separate the last two.",
    severity=Severity.WARN)
def background_is_small(outcome, config):
    if not _is_outcome(outcome):
        return
    band = _band(outcome)
    for key, fit in outcome.fits.items():
        if fit.background is None:
            continue
        signal = np.maximum(np.abs(fit.channel_p), np.abs(fit.channel_s))[band]
        fraction = float(np.median(np.abs(fit.background[band]) / np.maximum(signal, 1e-300)))
        if fraction > 0.1:
            yield Problem(f"{key[0]} @ probe {key[1]}: background is {fraction:.0%} of the "
                          "magnetic signal (median over the band)", detail={"fraction": fraction})


@diagnostic(
    stage=ACQUISITION,
    assumption="the fitted timing drift is well inside the range the drift model searches",
    why="A drift that reaches the search bound has been clipped, and the part that was not "
        "fitted remains as a phase error in rho (1.5 fs is ~0.9% at 1 THz).",
    remedy="Shorten the acquisition or cycle the states faster; check the lab temperature and "
           "the purge; raise maximum_drift_fs only if the drift is real.",
    severity=Severity.WARN)
def drift_within_range(outcome, config):
    if not _is_outcome(outcome):
        return
    for key, fit in outcome.fits.items():
        if fit.fitted_drift_fs > 250.0:
            yield Problem(f"{key[0]} @ probe {key[1]}: fitted drift {fit.fitted_drift_fs:.0f} "
                          "fs across the series", detail={"drift_fs": fit.fitted_drift_fs})


#: How well each nuisance term must be determined for its drift to be told apart from the
#: channel ratio. ~0.3 fs of delay or ~0.3% of gain is ~0.2% in rho at 1 THz.
SEPARABILITY_LIMITS = {"delay_span_fs": 0.3, "gain": 0.003, "tilt_per_thz": 0.005}


@diagnostic(
    stage=ACQUISITION,
    assumption="the acquisition order lets the drift be told apart from the channel ratio",
    why="A drift is measured by seeing the SAME polarisation state at different times. In a "
        "single pass 0/90/180/270 on gold with C ~ -1 the p and s channels are equal, so 0 and "
        "90 (and 180 and 270) look alike and a drift between them is, to first order, a change "
        "of P/Q -- exactly the calibration. The fit then returns a number, but a biased one "
        "(F40: index error 0.026 -> 0.14 with no purge at all).",
    remedy="Record the block as a palindrome (0,90,180,270,270,180,90,0): it revisits every "
           "state and separates drift from C at any channel ratio.",
    severity=Severity.WARN)
def drift_separable_from_ratio(outcome, config):
    if not _is_outcome(outcome):
        return
    from ..core.harmonic import revisit_lever_arm
    for key, fit in outcome.fits.items():
        if fit.nuisance_parameter_errors is None:
            continue
        lever = revisit_lever_arm(fit, outcome.result.band)
        if 0.0 < lever < 0.5:
            caveat = ("drift trades with P/Q -- a palindrome order fixes this"
                      if fit.segment_ids is None else
                      "per-scan rows remove the drift that runs THROUGH the files, but a lasting "
                      "step at an opening of the box is invisible and goes into P/Q -- only a "
                      "return visit (palindrome) can check that")
            yield Problem(f"{key[0]} @ probe {key[1]}: the same signal was only seen again "
                          f"{lever:.0%} of the block later (neighbouring acquisitions), so "
                          f"{caveat}", detail={"revisit_lever_arm": lever})
            continue
        poorly = {name: float(error) for name, error in zip(fit.nuisance_parameter_names,
                                                             fit.nuisance_parameter_errors)
                  if name in SEPARABILITY_LIMITS and not error <= SEPARABILITY_LIMITS[name]}
        if poorly:
            described = ", ".join(f"{name} +/- {value:.3g}" for name, value in poorly.items())
            yield Problem(f"{key[0]} @ probe {key[1]}: drift terms poorly determined "
                          f"({described}); {len(fit.delays_s)} acquisitions", detail=poorly)


@diagnostic(
    stage=ACQUISITION,
    assumption="the fitted amplitude drift across a block is a few percent at most",
    why="The amplitude ramp removes a smooth scale change between magnet states. A large one "
        "means the laser or the purge changed a lot inside the block, and anything not smooth "
        "(a step when the box was opened) is not removed and leaks into rho.",
    remedy="Wait longer after closing the box (bench_run_me live), and check the laser power "
           "log; repeat the block if it coincides with a disturbance.",
    severity=Severity.WARN)
def amplitude_drift_small(outcome, config):
    if not _is_outcome(outcome):
        return
    for key, fit in outcome.fits.items():
        if fit.fitted_amplitude_change > 0.05:
            yield Problem(f"{key[0]} @ probe {key[1]}: amplitude changed "
                          f"{fit.fitted_amplitude_change:.1%} across the series",
                          detail={"amplitude_change": fit.fitted_amplitude_change})
        if fit.fitted_tilt_change_per_thz > 0.05:
            yield Problem(f"{key[0]} @ probe {key[1]}: spectral tilt changed "
                          f"{fit.fitted_tilt_change_per_thz:.1%}/THz across the series -- the "
                          "purge was still drying the box (the fit models it; this is advisory)",
                          detail={"tilt_change_per_thz": fit.fitted_tilt_change_per_thz})


@diagnostic(
    stage=ACQUISITION,
    assumption="each acquisition has enough repeat scans for the noise model",
    why="Without the noise model the fit is unweighted and its error bars come from the "
        "residual scatter, which is fine for the values but makes the bars themselves rough "
        "and loses the chi-square test of the harmonic model.",
    remedy="Record at least three scans per acquisition (config['noise']['minimum_scans']).",
    severity=Severity.INFO)
def noise_model_available(outcome, config):
    if not _is_outcome(outcome):
        return
    for key, entry in outcome.noise.items():
        if entry.spectral_variance is None:
            yield Problem(f"{key[0]} @ probe {key[1]}: {entry.reason}")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@diagnostic(
    stage=CALIBRATION,
    assumption="the channel ratio C = d_p/d_s is frequency-flat",
    why="By crystal symmetry the detection vector is real and frequency-independent, so "
        "structure in C means something else polarisation-dependent is in the beam: probe walk, "
        "astigmatism, a reference smaller or larger than the beam, polarising optics.",
    remedy="Look at the channel-ratio panel of the run figure. A slope suggests clipping or a "
           "reference/sample size mismatch; ripple suggests an echo inside the window.")
def channel_ratio_is_flat(outcome, config):
    if not _is_outcome(outcome):
        return
    calibration = outcome.result.calibration
    if calibration is not None and not calibration.is_flat:
        yield Problem(f"channel ratio scatter {calibration.relative_scatter:.3f} (magnitude) and "
                      f"{calibration.phase_scatter_rad:.3f} rad (phase) across the band; "
                      "limits are 0.05 and 0.1",
                      detail={"relative_scatter": calibration.relative_scatter,
                              "phase_scatter_rad": calibration.phase_scatter_rad})


@diagnostic(
    stage=CALIBRATION,
    assumption="two independent calibration sources agree on the channel ratio when both "
               "can run",
    why="Gold and probe rotation reach C by physically different routes -- one swaps an object "
        "into the focus, the other moves nothing in the THz path. Agreement validates both; "
        "disagreement localises an error to the geometry (gold) or the probe optics (rotation).",
    remedy="If they disagree, check the gold's tilt and size against the sample's, and the "
           "half-wave plate for wedge (plan sec. 7.8).")
def calibration_sources_agree(outcome, config):
    if not _is_outcome(outcome):
        return
    from .stages import calibrate
    probes = [probe for role, probe in outcome.fits if role == SAMPLE_ROLE]
    has_gold = any(role == "channel_reference" for role, _ in outcome.fits)
    if len(probes) < 2 or not has_gold:
        return
    values = {}
    for source in ("gold_reference", "probe_rotation"):
        trial = dict(outcome.config)
        trial["calibration"] = dict(outcome.config.get("calibration") or {}, channel=source)
        values[source] = calibrate(trial, outcome.fits, outcome.frequencies_hz,
                                   outcome.result.band, outcome.primary_probe_deg).ratio
    difference = abs(values["gold_reference"] - values["probe_rotation"]) / abs(
        values["gold_reference"])
    if difference > 0.02:
        yield Problem(f"gold gives C = {values['gold_reference']:.4f}, probe rotation "
                      f"{values['probe_rotation']:.4f} ({difference:.1%} apart)",
                      detail={"relative_difference": difference})


# ---------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------

@diagnostic(
    stage=INVERSION,
    assumption="the recovered index is passive (k >= 0) within its error bars",
    why="Under the package convention N = n - ik an absorbing sample has k >= 0. A k that is "
        "significantly negative is not physics: it is a sign-convention or calibration error, "
        "not a branch problem, because N follows linearly from (1 - rho)/(1 + rho).",
    remedy="Check the calibration source and its sign, and the emitter rotation sense; then "
           "the incidence angle.",
    severity=Severity.FAIL)
def index_is_passive(outcome, config):
    if not _is_outcome(outcome):
        return
    result = outcome.result
    extinction = result.inversion.extinction
    error = (result.index_standard_error if result.index_standard_error is not None
             else np.zeros_like(extinction))
    significant = extinction < -3.0 * error - 1e-3
    fraction = float(np.mean(significant))
    if fraction > 0.1:
        yield Problem(f"k is significantly negative in {fraction:.0%} of the band",
                      detail={"fraction": fraction})


@diagnostic(
    stage=INVERSION,
    assumption="the sign convention is consistent end to end (no rho -> -rho flip)",
    why="Mixing the Fresnel sign convention between the calibration and the inversion "
        "replaces rho with -rho. On silicon at 45 deg that returns eps ~ 0.52 instead of 11.7 "
        "(plan sec. 6.5) -- a number that looks like a material, not like an error.",
    remedy="Every model in the package uses thz_core's Fresnel coefficients; check any "
           "externally supplied (stored) channel ratio for its sign.",
    severity=Severity.FAIL)
def sign_convention_consistent(outcome, config):
    # Only decidable against a known answer: a conductor genuinely has Re(eps) < 1, so the
    # flipped-sign signature is tested by asking which of rho and -rho explains the EXPECTED
    # material better.
    if not _is_outcome(outcome):
        return
    expected_name = (config.get("validation") or {}).get("expect")
    if expected_name is None:
        return
    result = outcome.result
    band_frequencies = result.inversion.frequencies_hz
    expected = float(np.median(np.real(reference_index(expected_name, band_frequencies) ** 2)))
    measured = float(np.median(result.inversion.permittivity.real))
    flipped = float(np.median(np.real(index_from_ellipsometric_ratio(
        -result.ratio[result.band], result.inversion.incidence_angle_rad) ** 2)))
    if abs(flipped - expected) < abs(measured - expected):
        yield Problem(f"Re(eps) = {measured:.3f}, but -rho would give {flipped:.3f}, closer to "
                      f"the expected {expected_name} value {expected:.3f}: the ratio's sign is "
                      "flipped somewhere")


@diagnostic(
    stage=INVERSION,
    assumption="the inversion is well conditioned across the band",
    why="d eps/d rho grows as (1 + rho)^-3; for good conductors rho -> -1 and small errors in rho "
        "become large errors in eps (plan sec. 8.2). Bins there are noisy by physics, not by a "
        "fault, and should be read with their error bars.",
    remedy="Expected for metallic samples at 45 deg. A higher incidence angle improves it.",
    severity=Severity.INFO)
def inversion_well_conditioned(outcome, config):
    if not _is_outcome(outcome):
        return
    result = outcome.result
    derivative = permittivity_derivative_wrt_ratio(result.ratio[result.band],
                                                   result.inversion.incidence_angle_rad)
    fraction = float(np.mean(np.abs(derivative) > 100.0))
    if fraction > 0.0:
        yield Problem(f"|d eps/d rho| > 100 in {fraction:.0%} of the band (near-mirror sample)",
                      detail={"fraction": fraction})


@diagnostic(
    stage=INVERSION,
    assumption="the fitted out-of-plane tilt is small and determined",
    why="A tilt of more than about a degree means the mount is far off; a tilt with an error "
        "bar as large as itself means the sample is too flat in frequency to separate tilt from "
        "the channel ratio, and the fit should not be trusted (HR-Si is the textbook case).",
    remedy="Re-seat the sample against the THz peak (plan sec. 4.5), or turn the tilt fit off "
           "for dispersionless samples.",
    severity=Severity.WARN)
def tilt_is_plausible(outcome, config):
    if not _is_outcome(outcome) or outcome.result.tilt_fit is None:
        return
    tilt = outcome.result.tilt_fit
    if abs(tilt.tilt_deg) > 1.0:
        yield Problem(f"fitted tilt {tilt.tilt_deg:+.3f} deg")
    if tilt.tilt_standard_error_deg > max(0.5 * abs(tilt.tilt_deg), 0.05):
        yield Problem(f"tilt {tilt.tilt_deg:+.3f} +/- {tilt.tilt_standard_error_deg:.3f} deg is "
                      "poorly determined (sample too flat in frequency?)")
