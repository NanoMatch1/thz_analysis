"""Maximum-Entropy-Method (MEM) phase retrieval for THz reflection de-embedding.

WHY THIS EXISTS
---------------
The air-gap de-embed needs the *material* reflection phase `arg(r_back)` reconstructed
from the robust magnitude `|r_back|` (the measured phase is corrupted by the gap's linear
timing phase and by alignment).  Our first implementation
(`deembed_air_gap_iterative.minimum_phase_from_magnitude`) used the Hilbert transform of
ln|r|.  That is the *minimum-phase* solution computed as an integral transform — and
integral transforms need the WHOLE spectrum, so truncating to our measurement band injects
the worst error at the band EDGES, which is exactly the sub-1-THz region (carrier dynamics)
we care about most.

MEM computes the SAME minimum-phase answer a different way: it fits an autoregressive
(all-pole) model to the in-band reflectivity and ANALYTICALLY CONTINUES, instead of
integrating across a truncated spectrum.  That is the documented reason MEM beats
(subtractive-)Kramers-Kronig and Hilbert for conductors and near the spectrum endpoints
(Lucarini et al., Appl. Opt. 45, 6519).  Same family (causal / minimum-phase), better
numerical conditioning at the edges.

WHAT IT IS / ISN'T
------------------
- It is window-AGNOSTIC: it operates only on a magnitude array `|r|` sampled on a uniform
  frequency grid.  SiO2-window and HR-Si-window de-embeds both feed it the same way; only
  the surrounding geometry (front reflection, internal angle) differs, not this step.
- Like Hilbert, bare MEM returns the MINIMUM-PHASE solution.  If the true `r_back` is
  minimum-phase, MEM is correct (and beats Hilbert at the edges).  If it is NOT
  minimum-phase (e.g. an embedded all-pass), neither is exact without ANCHOR points; an
  optional anchor (known phase at one or more frequencies) removes a residual offset/slope.

VALIDATION (run this file)
--------------------------
The synthetic test builds a KNOWN minimum-phase reflection as a rational function on the
unit circle, so both |r| AND arg(r) are known exactly and analytically.  We hand only |r|
to MEM and to Hilbert and compare the recovered phase to ground truth — biased toward
NEITHER method.  A pole is placed near the unit circle to stress the band edge, where we
expect MEM to win.

Run:  PYTHONPATH=. ../.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/mem_phase_retrieval.py
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import hilbert

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Core: MEM (autoregressive) minimum-phase reconstruction ───────────────────


def autoregressive_coefficients_from_reflectivity(reflectivity_power, num_coefficients):
    """Fit an all-pole (AR) model to a reflectivity POWER spectrum R(nu) = |r|^2.

    MEM models the power spectrum as
        R(nu) ~ error_power / |1 + sum_m a_m exp(-i 2 pi m nu)|^2 ,
    i.e. the squared magnitude of an all-pole transfer function.  The coefficients
    {a_1..a_M} and the prediction-error power follow from the autocorrelation of R via the
    Yule-Walker normal equations (same machinery as Burg/AR spectral estimation).

    Parameters
    ----------
    reflectivity_power : real array on a uniform frequency grid (one band, monotonic).
        Should be |r|^2 (power).  Must be >= 0.
    num_coefficients : AR model order M.  Higher M -> sharper spectral features but more
        risk of fitting noise; typical 10-40 for smooth THz reflectivity.

    Returns
    -------
    ar_coefficients : complex array [a_1, ..., a_M] (a_0 = 1 implied).
    error_power : float, the AR prediction-error power (sets the overall scale).
    """
    reflectivity_power = np.asarray(reflectivity_power, dtype=float)
    num_samples = reflectivity_power.size
    if num_coefficients >= num_samples:
        raise ValueError("num_coefficients must be smaller than the number of samples")

    # Autocorrelation of the power spectrum = inverse DFT of R (Wiener-Khinchin).  R is real,
    # so the autocorrelation lags are (numerically) real; keep the real part.
    autocorrelation = np.fft.ifft(reflectivity_power)
    autocorrelation = np.real(autocorrelation)

    # Yule-Walker normal equations: Toeplitz(r_0..r_{M-1}) @ a = -[r_1..r_M].
    toeplitz_first_column = autocorrelation[:num_coefficients]
    right_hand_side = -autocorrelation[1 : num_coefficients + 1]
    ar_coefficients = solve_toeplitz(
        (toeplitz_first_column, toeplitz_first_column), right_hand_side
    )

    # Prediction-error power: P = r_0 + sum_m a_m r_m.
    error_power = autocorrelation[0] + np.dot(
        ar_coefficients, autocorrelation[1 : num_coefficients + 1]
    )
    return ar_coefficients.astype(complex), float(error_power)


def maximum_entropy_phase(reflectivity_magnitude, num_coefficients):
    """Minimum-phase reflection phase consistent with |r|, via MEM (AR all-pole model).

    The minimum-phase reflection is  r_min(nu) = sqrt(P) / D(nu)  with
        D(nu) = 1 + sum_m a_m exp(-i 2 pi m nu)   (the AR prediction-error filter),
    so  arg(r_min) = -arg(D(nu)).  D is minimum-phase (stable all-pole), hence so is r_min.

    Parameters
    ----------
    reflectivity_magnitude : real array |r| on a uniform frequency grid (one band).
    num_coefficients : AR order M.

    Returns
    -------
    phase : real array, the reconstructed (minimum) phase, same length as the input.
    """
    reflectivity_magnitude = np.asarray(reflectivity_magnitude, dtype=float)
    reflectivity_power = reflectivity_magnitude**2

    ar_coefficients, _ = autoregressive_coefficients_from_reflectivity(
        reflectivity_power, num_coefficients
    )

    num_samples = reflectivity_magnitude.size
    normalized_frequency = np.arange(num_samples) / num_samples  # nu in [0, 1)
    orders = np.arange(1, num_coefficients + 1)
    # D(nu) = 1 + sum_m a_m exp(-i 2 pi m nu), evaluated per sample.
    phase_factors = np.exp(
        -1j * 2.0 * np.pi * np.outer(normalized_frequency, orders)
    )
    prediction_error_filter = 1.0 + phase_factors @ ar_coefficients
    return -np.angle(prediction_error_filter)


def hilbert_minimum_phase(reflectivity_magnitude):
    """Minimum phase via Hilbert(ln|r|) — the original engine, kept for comparison."""
    reflectivity_magnitude = np.asarray(reflectivity_magnitude, dtype=float)
    log_magnitude = np.log(np.maximum(reflectivity_magnitude, 1e-12))
    return -np.imag(hilbert(log_magnitude))


def apply_phase_anchor(frequency_hz, reconstructed_phase, anchor_frequencies_hz,
                       anchor_phases_rad):
    """Pin a minimum-phase reconstruction to known phase values (offset + slope).

    Bare MEM/Hilbert fix the phase only up to the freedom of a true all-pass component.
    Given >=1 anchor (a frequency where arg(r) is known, e.g. HR-Si), remove a constant
    offset (1 anchor) or an affine offset+slope (>=2 anchors) so the reconstruction matches.
    This is the pragmatic anchor; a full Blaschke-product correction is a later refinement.
    """
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    anchor_frequencies_hz = np.atleast_1d(anchor_frequencies_hz).astype(float)
    anchor_phases_rad = np.atleast_1d(anchor_phases_rad).astype(float)

    indices = [int(np.argmin(np.abs(frequency_hz - f))) for f in anchor_frequencies_hz]
    # Snap to the grid frequencies actually present, so the affine correction evaluated on
    # the grid reproduces the anchor exactly (no requested-vs-grid frequency mismatch).
    grid_anchor_frequencies = frequency_hz[indices]
    reconstructed_at_anchors = reconstructed_phase[indices]
    residual = anchor_phases_rad - reconstructed_at_anchors

    if grid_anchor_frequencies.size == 1:
        return reconstructed_phase + residual[0]
    slope, intercept = np.polyfit(grid_anchor_frequencies, residual, 1)
    return reconstructed_phase + slope * frequency_hz + intercept


# ── Known-minimum-phase synthetic (exact magnitude AND phase, biased to neither) ──


def known_minimum_phase_reflection(normalized_frequency, zeros, poles, gain=1.0):
    """A minimum-phase rational reflection on the unit circle z = exp(i 2 pi nu).

        H(z) = gain * prod_k (1 - q_k z^{-1}) / prod_l (1 - p_l z^{-1}),  |q_k|, |p_l| < 1.

    With all zeros and poles strictly inside the unit circle, H is minimum-phase, so BOTH
    |H| and arg(H) are determined and known exactly — the honest ground truth for the test.
    """
    z = np.exp(1j * 2.0 * np.pi * normalized_frequency)
    z_inverse = 1.0 / z
    numerator = np.full(z.shape, gain, dtype=complex)
    for zero in zeros:
        numerator *= 1.0 - zero * z_inverse
    denominator = np.ones(z.shape, dtype=complex)
    for pole in poles:
        denominator *= 1.0 - pole * z_inverse
    return numerator / denominator


def synthetic_validation(num_coefficients=30):
    """Compare MEM vs Hilbert phase recovery against a KNOWN minimum-phase ground truth.

    The honest part: the physical reflectivity exists over the FULL period (DC -> Nyquist),
    but a real THz measurement only sees a finite BAND.  We give both retrievers ONLY the
    in-band magnitude slice (no DC, no high-frequency tail) and score the recovered phase
    against the exact analytic phase IN that band.  Band truncation is exactly what makes
    Hilbert(ln|r|) ring at the endpoints; MEM's AR model continues past the band instead.
    The pole at radius 0.96 puts a sharp feature near the LOW edge to stress that region.
    """
    num_samples = 2048
    normalized_frequency_full = np.arange(num_samples) / num_samples

    # Minimum-phase test system: zeros/poles strictly inside the unit circle (so the system
    # is minimum-phase and arg(H) is known exactly), with one pole near the edge (r=0.96).
    zeros = [0.6 * np.exp(1j * 2 * np.pi * 0.12), 0.6 * np.exp(-1j * 2 * np.pi * 0.12)]
    poles = [0.96 * np.exp(1j * 2 * np.pi * 0.05), 0.96 * np.exp(-1j * 2 * np.pi * 0.05),
             0.5 * np.exp(1j * 2 * np.pi * 0.30), 0.5 * np.exp(-1j * 2 * np.pi * 0.30)]
    reflection_full = known_minimum_phase_reflection(
        normalized_frequency_full, zeros, poles, gain=0.4)

    # Extract the measured BAND: a contiguous low-side slice (mimics DC-cutoff + finite span).
    band_start_fraction, band_stop_fraction = 0.015, 0.42
    band_mask = (normalized_frequency_full >= band_start_fraction) & (
        normalized_frequency_full <= band_stop_fraction)
    frequency_band = normalized_frequency_full[band_mask]
    magnitude_band = np.abs(reflection_full[band_mask])
    phase_true_band = np.unwrap(np.angle(reflection_full[band_mask]))

    # Both retrievers see ONLY the band slice (as if it were the whole spectrum they have).
    phase_mem = maximum_entropy_phase(magnitude_band, num_coefficients)
    phase_hilbert = hilbert_minimum_phase(magnitude_band)
    phase_mem = np.unwrap(phase_mem)
    phase_hilbert = np.unwrap(phase_hilbert)

    # Minimum phase is defined up to an additive constant the gap-slope fit later absorbs;
    # also up to a linear (delay) term.  Score SHAPE: detrend (remove best affine fit of the
    # residual) over the central band so neither method is charged for offset/slope.
    central = (frequency_band > 0.10) & (frequency_band < 0.35)

    def remove_affine_against_truth(phase):
        residual = phase - phase_true_band
        slope, intercept = np.polyfit(frequency_band[central], residual[central], 1)
        return phase - (slope * frequency_band + intercept)

    phase_mem_aligned = remove_affine_against_truth(phase_mem)
    phase_hilbert_aligned = remove_affine_against_truth(phase_hilbert)

    # Score in the lower band (region of interest) and at the very low edge.
    low_band = frequency_band <= 0.12
    edge_band = frequency_band <= 0.05

    def rms(phase_aligned, mask):
        return float(np.sqrt(np.mean((phase_aligned - phase_true_band)[mask] ** 2)))

    mem_low = rms(phase_mem_aligned, low_band)
    hilbert_low = rms(phase_hilbert_aligned, low_band)
    mem_edge = rms(phase_mem_aligned, edge_band)
    hilbert_edge = rms(phase_hilbert_aligned, edge_band)

    print("=== MEM vs Hilbert phase retrieval (known min-phase, finite BAND) ===")
    print(f"  AR order M = {num_coefficients}, band samples = {frequency_band.size} "
          f"(nu {band_start_fraction}-{band_stop_fraction})")
    print(f"  low-band phase RMS error:  MEM {mem_low:.4f} rad   Hilbert {hilbert_low:.4f} rad")
    print(f"  band-EDGE phase RMS error: MEM {mem_edge:.4f} rad   Hilbert {hilbert_edge:.4f} rad")
    better = "MEM" if mem_edge < hilbert_edge else "Hilbert"
    print(f"  -> at the LOW edge, {better} is better "
          f"(MEM/Hilbert ratio {mem_edge / max(hilbert_edge, 1e-9):.2f})")

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(frequency_band, magnitude_band, color="black")
    axes[0].set_ylabel("|r|")
    axes[0].set_title("Measured band of a known minimum-phase reflection (pole near low edge)")
    axes[1].plot(frequency_band, phase_true_band, color="black", lw=2, label="true phase")
    axes[1].plot(frequency_band, phase_mem_aligned, "--", color="tab:green", label="MEM")
    axes[1].plot(frequency_band, phase_hilbert_aligned, ":", color="tab:red", label="Hilbert")
    axes[1].axvspan(frequency_band.min(), 0.05, color="tab:orange", alpha=0.15, label="low edge")
    axes[1].set_xlabel("normalized frequency nu")
    axes[1].set_ylabel("phase (rad, affine-aligned)")
    axes[1].legend(loc="best", fontsize=8)
    figure.tight_layout()
    output_path = "explorations/air_gap_cnt_reflection/mem_phase_validation.png"
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    print(f"  figure -> {output_path}")

    return dict(mem_low=mem_low, hilbert_low=hilbert_low,
                mem_edge=mem_edge, hilbert_edge=hilbert_edge)


if __name__ == "__main__":
    synthetic_validation()
