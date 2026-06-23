"""Fabry-Perot air-gap de-embedding for window-coupled reflection (scoping demo).

Context (ANALYSIS_NOTES §9 / §9b): a CNT mat pressed on the back of the SiO2 window
does not contact perfectly. A thin AIR GAP sits between the window and the sample,
turning the single SiO2->CNT interface we *assume* into a 3-medium stack

        SiO2  |   air gap (thickness d)   |  CNT
       (n=1.95)        (n=1)               (n - i k)
              ^ r_front                    ^ r_back
              the wave bounces in the gap: a Fabry-Perot etalon.

The measured reflection is therefore the etalon response, not r_{SiO2->CNT}. The
gap adds (a) a round-trip phase ramp 2*beta(omega) = a pure time delay, which in the
inversion looks like n falling and crossing 1 at high f, and (b) multiple-bounce
ripple. This script shows, in isolation and with readable math:

  1. FORWARD  : how a known CNT + known gap produce the distorted measured r.
  2. DE-EMBED : the exact closed-form that recovers the gap->sample reflection
                r_back = r_{air->CNT}, then inverts it back to the planted n, k.
  3. ROUGHNESS: a distribution of gap thickness (a rough mat) damps the high-f
                recovery (a coherence-bandwidth limit) -- the real-world ceiling.

Exact relation (single lossless gap between known medium 0 and unknown medium 2):

    r_meas = (r_front + r_back * e^{-i 2beta}) / (1 + r_front * r_back * e^{-i 2beta})

  =>  x := r_back * e^{-i 2beta} = (r_meas - r_front) / (1 - r_front * r_meas)

so |x| = |r_back| exactly (lossless gap) and arg(x) = arg(r_back) - 2beta, where
2beta = omega/c * 2 d cos(theta_gap) is a pure linear phase removed by estimating d.

NOTHING here touches the pipeline; it only *reads* two thz_core Fresnel primitives.
Sign convention matches thz_core: n_hat = n - i k, exp(-i omega t).
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thz_core.thz_core as core

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
REFRACTIVE_INDEX_SIO2 = 1.95
REFRACTIVE_INDEX_AIR = 1.0
EXTERNAL_ANGLE_RAD = np.deg2rad(45.0)


# ---------------------------------------------------------------------------
# Geometry: external 45 deg in air -> refract into SiO2 -> the gap is air again,
# so by Snell the angle inside the gap returns to 45 deg.
# ---------------------------------------------------------------------------

def internal_angles():
    """Return (theta_in_SiO2, theta_in_gap) in radians for a 45 deg external beam."""
    theta_sio2 = np.arcsin(np.sin(EXTERNAL_ANGLE_RAD) / REFRACTIVE_INDEX_SIO2)
    theta_gap = np.arcsin(REFRACTIVE_INDEX_SIO2 * np.sin(theta_sio2) / REFRACTIVE_INDEX_AIR)
    return float(theta_sio2), float(theta_gap)


# ---------------------------------------------------------------------------
# Readable physics helpers (s-polarisation throughout)
# ---------------------------------------------------------------------------

def gap_round_trip_phase(frequency_hz, gap_thickness_m, gap_angle_rad):
    """2*beta(omega) = (omega/c) * 2 * d * cos(theta_gap): the etalon round-trip phase."""
    omega = 2.0 * np.pi * frequency_hz
    return omega / SPEED_OF_LIGHT_M_PER_S * 2.0 * gap_thickness_m * np.cos(gap_angle_rad)


def fabry_perot_reflection(reflection_front, reflection_back, round_trip_phase):
    """FORWARD model: measured reflection of a single gap layer (medium0|gap|medium2).

    r_meas = (r_front + r_back e^{-i 2beta}) / (1 + r_front r_back e^{-i 2beta})
    """
    bounce = np.exp(-1j * round_trip_phase)
    return (reflection_front + reflection_back * bounce) / (
        1.0 + reflection_front * reflection_back * bounce
    )


def deembed_gap_layer(reflection_measured, reflection_front):
    """INVERSE model: solve the FP relation exactly for x = r_back * e^{-i 2beta}.

    x = (r_meas - r_front) / (1 - r_front r_meas).  Well conditioned: the
    denominator stays well away from 0 because |r_front| < 1.
    """
    return (reflection_measured - reflection_front) / (
        1.0 - reflection_front * reflection_measured
    )


def estimate_gap_thickness(frequency_hz, x_complex, gap_angle_rad, fit_band_hz):
    """Recover the gap thickness d from the linear phase slope of arg(x).

    arg(x) = arg(r_back) - 2beta = arg(r_back) - (omega/c) 2 d cos(theta).  For a
    weakly dispersive r_back the dominant slope vs omega is -(2 d cos theta / c),
    so d = -slope * c / (2 cos theta).  Fit only the trusted band.
    """
    omega = 2.0 * np.pi * frequency_hz
    in_band = (frequency_hz >= fit_band_hz[0]) & (frequency_hz <= fit_band_hz[1]) & np.isfinite(x_complex)
    phase = np.unwrap(np.angle(x_complex[in_band]))
    slope_per_rad_per_s, _ = np.polyfit(omega[in_band], phase, 1)
    return float(-slope_per_rad_per_s * SPEED_OF_LIGHT_M_PER_S / (2.0 * np.cos(gap_angle_rad)))


def remove_round_trip_phase(x_complex, frequency_hz, gap_thickness_m, gap_angle_rad):
    """Strip the 2beta linear phase from x to recover r_back = r_{air->sample}."""
    return x_complex * np.exp(1j * gap_round_trip_phase(frequency_hz, gap_thickness_m, gap_angle_rad))


# ---------------------------------------------------------------------------
# Planted ground truth: a Drude-like conductive CNT index (n - i k)
# ---------------------------------------------------------------------------

def planted_cnt_index(frequency_hz):
    """A smooth conductive index n_hat = n - i k from a simple Drude permittivity."""
    omega = 2.0 * np.pi * frequency_hz
    eps_inf = 4.0
    plasma_omega = 2.0 * np.pi * 6e12      # 6 THz plasma frequency
    scattering_rate = 2.0 * np.pi * 3e12   # 3 THz damping
    eps = eps_inf - plasma_omega**2 / (omega**2 + 1j * scattering_rate * omega)
    n_hat = np.sqrt(eps)                    # principal branch: Re > 0
    # thz_core convention is n - i k with k > 0; np.sqrt(eps) gives n + i k here, conjugate.
    return np.conj(n_hat)


# ---------------------------------------------------------------------------
# Demo 1 -- smooth (single-valued) gap: exact recovery
# ---------------------------------------------------------------------------

def demo_smooth_gap(frequency_hz, theta_sio2, theta_gap, gap_thickness_m, fit_band_hz):
    n_cnt_true = planted_cnt_index(frequency_hz)

    # The two interfaces of the gap (s-pol):
    reflection_front = core.fresnel_reflection_s(REFRACTIVE_INDEX_SIO2, REFRACTIVE_INDEX_AIR, theta_sio2)
    reflection_back_true = core.fresnel_reflection_s(REFRACTIVE_INDEX_AIR, n_cnt_true, theta_gap)

    # FORWARD: what the instrument actually measures through the gap.
    round_trip = gap_round_trip_phase(frequency_hz, gap_thickness_m, theta_gap)
    reflection_measured = fabry_perot_reflection(reflection_front, reflection_back_true, round_trip)

    # NAIVE (no de-embed): invert r_meas as if it were r_{SiO2->CNT} (the current pipeline).
    full_mask = np.ones(frequency_hz.size, dtype=bool)
    n_naive, k_naive, _ = core.invert_nk_reflection(
        frequency_hz, reflection_measured, full_mask, theta_rad=theta_sio2, n_incident=REFRACTIVE_INDEX_SIO2)

    # DE-EMBED step 1: the exact algebra. x = r_back * e^{-i 2beta}; |x| == |r_back|.
    x = deembed_gap_layer(reflection_measured, reflection_front)

    # DE-EMBED step 2: strip the 2beta linear phase. Two ways to get the gap d:
    #   (a) KNOWN d  -- in practice the pulse round-trip delay measured by the T0 /
    #       cross-correlation step (a clean, sample-independent number). Here we use
    #       the planted d to show the algebra recovers n,k EXACTLY when d is right.
    #   (b) FITTED d -- from the slope of arg(x). Convenient but BIASED, because the
    #       sample's own dispersion contributes phase slope too (see the gap estimate).
    reflection_back_known = remove_round_trip_phase(x, frequency_hz, gap_thickness_m, theta_gap)
    n_known, k_known, _ = core.invert_nk_reflection(
        frequency_hz, reflection_back_known, full_mask, theta_rad=theta_gap, n_incident=REFRACTIVE_INDEX_AIR)

    gap_estimate_m = estimate_gap_thickness(frequency_hz, x, theta_gap, fit_band_hz)
    reflection_back_fitted = remove_round_trip_phase(x, frequency_hz, gap_estimate_m, theta_gap)
    n_fitted, k_fitted, _ = core.invert_nk_reflection(
        frequency_hz, reflection_back_fitted, full_mask, theta_rad=theta_gap, n_incident=REFRACTIVE_INDEX_AIR)

    band = (frequency_hz >= fit_band_hz[0]) & (frequency_hz <= fit_band_hz[1])
    print("--- Demo 1: smooth gap ---")
    print(f"  planted gap d        = {gap_thickness_m*1e6:.2f} um")
    print(f"  fitted gap d (arg x) = {gap_estimate_m*1e6:.2f} um  "
          f"(biased by sample dispersion)")
    print(f"  |x| == |r_back| (lossless gap), max abs err in band = "
          f"{np.max(np.abs(np.abs(x[band]) - np.abs(reflection_back_true[band]))):.2e}")
    print(f"  n recovery (KNOWN d):  max |dn| in band = "
          f"{np.max(np.abs(n_known[band] - n_cnt_true.real[band])):.2e}  <- exact")
    print(f"  n recovery (FITTED d): max |dn| in band = "
          f"{np.max(np.abs(n_fitted[band] - n_cnt_true.real[band])):.3f}  <- gap-estimate limited")
    print(f"  naive (no de-embed) drops below 1: n@{fit_band_hz[1]*1e-12:.1f}THz = "
          f"{np.interp(fit_band_hz[1], frequency_hz, n_naive):.2f}")

    return {
        "n_true": n_cnt_true.real, "k_true": -n_cnt_true.imag,
        "n_naive": n_naive, "n_known": n_known, "n_fitted": n_fitted,
        "r_meas": reflection_measured, "r_back_true": reflection_back_true, "x": x,
        "gap_estimate_m": gap_estimate_m,
    }


# ---------------------------------------------------------------------------
# Demo 2 -- rough gap: a distribution of d damps the high-f recovery
# ---------------------------------------------------------------------------

def demo_rough_gap(frequency_hz, theta_sio2, theta_gap, mean_gap_m, roughness_sigma_m, fit_band_hz):
    n_cnt_true = planted_cnt_index(frequency_hz)
    reflection_front = core.fresnel_reflection_s(REFRACTIVE_INDEX_SIO2, REFRACTIVE_INDEX_AIR, theta_sio2)
    reflection_back_true = core.fresnel_reflection_s(REFRACTIVE_INDEX_AIR, n_cnt_true, theta_gap)

    # The beam spot averages COHERENTLY over a distribution of gap thickness.
    rng = np.random.default_rng(0)
    gap_samples = mean_gap_m + roughness_sigma_m * rng.standard_normal(4000)
    measured_accumulator = np.zeros(frequency_hz.size, dtype=complex)
    for gap_thickness_m in gap_samples:
        round_trip = gap_round_trip_phase(frequency_hz, gap_thickness_m, theta_gap)
        measured_accumulator += fabry_perot_reflection(reflection_front, reflection_back_true, round_trip)
    reflection_measured_rough = measured_accumulator / gap_samples.size

    x_rough = deembed_gap_layer(reflection_measured_rough, reflection_front)

    # |x| should equal |r_back| for a smooth gap; roughness makes |x| roll off at high f.
    coherence_ratio = np.abs(x_rough) / np.abs(reflection_back_true)
    # where the recovered |r_back| has lost 50% of its amplitude:
    below = np.where(coherence_ratio < 0.5)[0]
    coherence_edge_thz = frequency_hz[below[0]] * 1e-12 if below.size else np.nan

    print("\n--- Demo 2: rough gap ---")
    print(f"  mean gap = {mean_gap_m*1e6:.1f} um, roughness sigma = {roughness_sigma_m*1e6:.1f} um")
    print(f"  |x|/|r_back| drops below 0.5 at ~{coherence_edge_thz:.2f} THz "
          f"(coherence-bandwidth ceiling)" if np.isfinite(coherence_edge_thz)
          else "  |x|/|r_back| stays above 0.5 across the band")
    return {"coherence_ratio": coherence_ratio, "x_rough": x_rough,
            "r_back_true": reflection_back_true}


# ---------------------------------------------------------------------------

def main():
    frequency_hz = np.linspace(0.05e12, 4.0e12, 800)
    theta_sio2, theta_gap = internal_angles()
    fit_band_hz = (0.4e12, 3.0e12)
    print(f"angles: SiO2 internal {np.rad2deg(theta_sio2):.2f} deg, "
          f"gap {np.rad2deg(theta_gap):.2f} deg (cos={np.cos(theta_gap):.3f})\n")

    smooth = demo_smooth_gap(frequency_hz, theta_sio2, theta_gap, gap_thickness_m=30e-6, fit_band_hz=fit_band_hz)
    rough = demo_rough_gap(frequency_hz, theta_sio2, theta_gap,
                           mean_gap_m=30e-6, roughness_sigma_m=12e-6, fit_band_hz=fit_band_hz)

    f_thz = frequency_hz * 1e-12
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")

    ax = axes[0, 0]
    ax.plot(f_thz, np.abs(smooth["r_meas"]), label="|r_meas| (through gap)")
    ax.plot(f_thz, np.abs(smooth["r_back_true"]), "--", label="|r_back| true (air->CNT)")
    ax.plot(f_thz, np.abs(smooth["x"]), ":", label="|x| de-embedded")
    ax.set_title("Reflection magnitude: gap distortion vs de-embedded")
    ax.set_xlabel("THz"); ax.set_ylabel("|r|"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(f_thz, smooth["n_true"], "k", lw=2, label="n planted")
    ax.plot(f_thz, smooth["n_naive"], label="n naive (no de-embed)")
    ax.plot(f_thz, smooth["n_known"], "--", label="n de-embed (known d, exact)")
    ax.plot(f_thz, smooth["n_fitted"], ":", label="n de-embed (fitted d)")
    ax.axhline(1.0, color="crimson", lw=0.6, linestyle="dashed")
    ax.set_ylim(0, max(8, np.nanmax(smooth["n_true"]) + 1))
    ax.set_title("Refractive index: naive drops <1, de-embed (known d) recovers truth")
    ax.set_xlabel("THz"); ax.set_ylabel("n"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(f_thz, np.unwrap(np.angle(smooth["x"])), label="arg(x) = arg(r_back) - 2beta")
    ax.plot(f_thz, np.unwrap(np.angle(smooth["r_back_true"])), "--", label="arg(r_back) true")
    ax.set_title(f"Phase: 2beta linear ramp = the gap "
                 f"(est d = {smooth['gap_estimate_m']*1e6:.1f} um)")
    ax.set_xlabel("THz"); ax.set_ylabel("rad"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(f_thz, rough["coherence_ratio"])
    ax.axhline(0.5, color="crimson", lw=0.6, linestyle="dashed")
    ax.set_title("Roughness ceiling: |x|/|r_back| roll-off (sigma_d = 12 um)")
    ax.set_xlabel("THz"); ax.set_ylabel("recovered / true |r_back|"); ax.set_ylim(0, 1.2)

    out = os.path.join(os.path.dirname(__file__), "explore_air_gap_deembedding.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
