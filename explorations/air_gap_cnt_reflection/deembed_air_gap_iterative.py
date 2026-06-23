"""Iterative gap-thickness (d) refinement for the air-gap de-embed — a TESTBED.

Read this before trusting any number it prints.

The core difficulty (state it plainly): after the exact de-embed
    x = (r_meas - r_front)/(1 - r_front r_meas) = r_back * e^{-i 2beta(d)}
we must split arg(x) into the material phase arg(r_back) and the gap's linear phase
2beta(d) = (omega/c) 2 d cos(theta_gap).  A pure linear fit to arg(x) attributes ALL of
its linear trend to the gap — but a conductor's r_back ALSO has a roughly linear phase
trend, so the fit is BIASED (over-estimates d, over-lifts n).  This is a genuine
degeneracy; breaking it REQUIRES an assumption.  This script implements two, side by
side, and tests them against synthetic ground truth so we can SEE which assumption holds:

  A) ITERATIVE smooth-material anchor (the literal "iterate" idea).
     Assumption: the material n(omega), k(omega) are smooth (low-order polynomial).
     Loop: invert with current d -> smooth n,k -> model arg(r_back) -> residual slope -> d.
     WARNING: if the true material phase is itself smooth, smoothing returns it unchanged
     and the iteration barely moves off its (biased) start — i.e. this assumption does NOT
     reliably break the degeneracy.  Included precisely so we can watch it fail/succeed.

  B) MINIMUM-PHASE (causality) anchor.
     Assumption: r_back is minimum-phase, so its phase is fixed by its magnitude
     (|x| = |r_back| exactly for a lossless gap) via the Hilbert transform of ln|x|.
     This is determined by the MAGNITUDE, independent of the measured phase, so it CAN
     break the degeneracy in one shot.  Whether the assumption is valid for r_{air->CNT}
     is the open question — the synthetic test checks it directly.

A third, assumption-light route (NOT phase-based) is d from the second-reflection pulse
round-trip DELAY (cross-correlation) — the cleanest, deferred to the pipeline-data step.

Nothing here is a settled method.  It is scaffolding for tomorrow's plan.

Run:  PYTHONPATH=. ../.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/deembed_air_gap_iterative.py
"""

from __future__ import annotations

import os

import numpy as np
from scipy.signal import hilbert

import thz_core.thz_core as core

# Reuse the validated prototype primitives and the real-data pipeline runner.
from explore_air_gap_deembedding import (
    SPEED_OF_LIGHT_M_PER_S, REFRACTIVE_INDEX_SIO2, REFRACTIVE_INDEX_AIR,
    internal_angles, gap_round_trip_phase, fabry_perot_reflection,
    deembed_gap_layer, planted_cnt_index,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _band_mask(frequency_hz, mask, band_hz):
    return mask & np.isfinite(frequency_hz) & (frequency_hz >= band_hz[0]) & (frequency_hz <= band_hz[1])


def gap_from_linear_slope(frequency_hz, residual_phase, in_band, gap_angle_rad):
    """d from the linear slope of a (supposedly pure -2beta) residual phase vs omega."""
    omega = 2.0 * np.pi * frequency_hz[in_band]
    phase = residual_phase[in_band]
    slope, _ = np.polyfit(omega, phase, 1)
    return float(-slope * SPEED_OF_LIGHT_M_PER_S / (2.0 * np.cos(gap_angle_rad)))


# ── Method A: iterative smooth-material anchor ──────────────────────────────


def estimate_gap_iterative_smooth(
    frequency_hz, x_complex, mask, gap_angle_rad, band_hz,
    smooth_order=3, max_iterations=20, tolerance_m=2e-8,
):
    """Iteratively refine d assuming the material n,k are smooth (low-order poly).

    Returns (d, history, diagnostics).  See module docstring for the assumption and
    its known weakness (smoothing a smooth phase is near-identity -> weak refinement).
    """
    in_band = _band_mask(frequency_hz, mask, band_hz)
    omega = 2.0 * np.pi * frequency_hz
    arg_x = np.unwrap(np.angle(x_complex))

    # Initial d: the raw (biased) linear-slope of arg(x) itself.
    gap_thickness_m = gap_from_linear_slope(frequency_hz, arg_x, in_band, gap_angle_rad)
    history = [gap_thickness_m]

    for _ in range(max_iterations):
        reflection_back = x_complex * np.exp(1j * gap_round_trip_phase(frequency_hz, gap_thickness_m, gap_angle_rad))
        n, k, _ = core.invert_nk_reflection(
            frequency_hz, reflection_back, mask, theta_rad=gap_angle_rad, n_incident=REFRACTIVE_INDEX_AIR)

        # Smooth the recovered n,k over the band (the assumption made explicit).
        freq_band = frequency_hz[in_band]
        n_poly = np.polyval(np.polyfit(freq_band, n[in_band], smooth_order), frequency_hz)
        k_poly = np.polyval(np.polyfit(freq_band, k[in_band], smooth_order), frequency_hz)
        model_reflection = core.fresnel_reflection_s(
            REFRACTIVE_INDEX_AIR, n_poly - 1j * k_poly, gap_angle_rad)

        # Residual phase = arg(x) - arg(model material).  If d were right this is -2beta.
        residual = arg_x - np.unwrap(np.angle(model_reflection))
        new_gap_thickness_m = gap_from_linear_slope(frequency_hz, residual, in_band, gap_angle_rad)
        history.append(new_gap_thickness_m)
        if abs(new_gap_thickness_m - gap_thickness_m) < tolerance_m:
            gap_thickness_m = new_gap_thickness_m
            break
        gap_thickness_m = new_gap_thickness_m

    return gap_thickness_m, np.array(history), {}


# ── Method B: minimum-phase (causality) anchor ──────────────────────────────


def minimum_phase_from_magnitude(magnitude):
    """Minimum-phase phase consistent with a magnitude spectrum (Hilbert of ln|.|).

    Approximate: uses scipy's analytic-signal Hilbert on the band-limited ln|magnitude|,
    so it carries edge effects.  Good enough to test the idea; a cepstral version with
    out-of-band extrapolation would be the production form.
    """
    log_magnitude = np.log(np.maximum(magnitude, 1e-12))
    return -np.imag(hilbert(log_magnitude))


def estimate_gap_minimum_phase(frequency_hz, x_complex, mask, gap_angle_rad, band_hz):
    """d from the excess phase arg(x) - phi_minphase(|x|), which should be pure -2beta."""
    in_band = _band_mask(frequency_hz, mask, band_hz)
    material_phase = minimum_phase_from_magnitude(np.abs(x_complex))
    excess = np.unwrap(np.angle(x_complex)) - material_phase
    gap_thickness_m = gap_from_linear_slope(frequency_hz, excess, in_band, gap_angle_rad)
    return gap_thickness_m, dict(material_phase=material_phase, excess=excess, in_band=in_band)


# ── Synthetic ground-truth test (does each method recover a KNOWN d?) ─────────


def synthetic_test():
    frequency_hz = np.linspace(0.1e12, 4.0e12, 1000)
    theta_sio2, theta_gap = internal_angles()
    band_hz = (0.4e12, 3.0e12)
    mask = np.ones(frequency_hz.size, dtype=bool)
    true_gap_m = 20e-6

    n_cnt = planted_cnt_index(frequency_hz)
    reflection_front = core.fresnel_reflection_s(REFRACTIVE_INDEX_SIO2, REFRACTIVE_INDEX_AIR, theta_sio2)
    reflection_back_true = core.fresnel_reflection_s(REFRACTIVE_INDEX_AIR, n_cnt, theta_gap)
    round_trip = gap_round_trip_phase(frequency_hz, true_gap_m, theta_gap)
    reflection_measured = fabry_perot_reflection(reflection_front, reflection_back_true, round_trip)

    x = deembed_gap_layer(reflection_measured, reflection_front)

    in_band = _band_mask(frequency_hz, mask, band_hz)
    d_argx = gap_from_linear_slope(frequency_hz, np.unwrap(np.angle(x)), in_band, theta_gap)
    d_iter, history, _ = estimate_gap_iterative_smooth(frequency_hz, x, mask, theta_gap, band_hz)
    d_minphase, mp_diag = estimate_gap_minimum_phase(frequency_hz, x, mask, theta_gap, band_hz)

    print("=== SYNTHETIC TEST (planted Drude CNT, known gap) ===")
    print(f"  true gap d            = {true_gap_m*1e6:.2f} um")
    print(f"  d (arg-x slope)       = {d_argx*1e6:.2f} um   (biased baseline)")
    print(f"  d (iterative smooth)  = {d_iter*1e6:.2f} um   (converged from {history[0]*1e6:.2f}, "
          f"{len(history)-1} steps)")
    print(f"  d (minimum-phase)     = {d_minphase*1e6:.2f} um")
    print(f"  -> errors: arg-x {(d_argx-true_gap_m)*1e6:+.2f} um, "
          f"iter {(d_iter-true_gap_m)*1e6:+.2f} um, minphase {(d_minphase-true_gap_m)*1e6:+.2f} um")
    return dict(frequency_hz=frequency_hz, x=x, theta_gap=theta_gap, band_hz=band_hz,
                true_gap_m=true_gap_m, d_argx=d_argx, d_iter=d_iter, d_minphase=d_minphase,
                history=history, mp_diag=mp_diag, n_true=n_cnt.real, k_true=-n_cnt.imag, mask=mask)


# ── Real-data application ────────────────────────────────────────────────────


def real_data_test():
    from compare_reflection_pathways import run_shared_axis, PIPELINE_CONFIG, short_name
    theta_sio2, theta_gap = internal_angles()
    band_hz = (0.4e12, 2.5e12)
    dataset = run_shared_axis(PIPELINE_CONFIG, eps_infinity=1.0)

    results = []
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        processing = data_obj.processing_dict
        if processing.get("reflection_r") is None:
            continue
        frequency_hz = np.asarray(processing["fft_freq"], dtype=float)
        mask = np.asarray(processing["transfer_mask"], dtype=bool)
        r_meas = np.asarray(processing["reflection_r"], dtype=complex)
        r_front = complex(processing["r_reference"])
        x = deembed_gap_layer(r_meas, r_front)

        in_band = _band_mask(frequency_hz, mask, band_hz)
        d_argx = gap_from_linear_slope(frequency_hz, np.unwrap(np.angle(x)), in_band, theta_gap)
        d_iter, history, _ = estimate_gap_iterative_smooth(frequency_hz, x, mask, theta_gap, band_hz)
        d_minphase, mp_diag = estimate_gap_minimum_phase(frequency_hz, x, mask, theta_gap, band_hz)

        n_versions = {}
        for label, gap_m in (("argx", d_argx), ("iter", d_iter), ("minphase", d_minphase)):
            r_back = x * np.exp(1j * gap_round_trip_phase(frequency_hz, gap_m, theta_gap))
            n, k, _ = core.invert_nk_reflection(
                frequency_hz, r_back, mask, theta_rad=theta_gap, n_incident=REFRACTIVE_INDEX_AIR)
            n_versions[label] = (n, k, gap_m)

        name = short_name(filename)
        print(f"\n{name}: d  arg-x {d_argx*1e6:.1f}  iter {d_iter*1e6:.1f}  "
              f"minphase {d_minphase*1e6:.1f} um")
        results.append((name, frequency_hz, mask, x, n_versions, mp_diag, band_hz))
    return results


def make_figure(synthetic, real_results):
    n_real = len(real_results)
    figure, axes = plt.subplots(1 + n_real, 3, figsize=(16, 4.4 * (1 + n_real)), squeeze=False)
    figure.suptitle("Iterative / minimum-phase gap-d TESTBED — synthetic validation (top) + real data", fontsize=12)

    # Synthetic row: d-convergence, excess-phase linearity, recovered n.
    f_thz = synthetic["frequency_hz"] * 1e-12
    ax = axes[0][0]
    ax.plot(synthetic["history"] * 1e6, "o-", label="iterative d")
    ax.axhline(synthetic["true_gap_m"] * 1e6, color="k", ls="--", label="true d")
    ax.axhline(synthetic["d_minphase"] * 1e6, color="C2", ls=":", label="minphase d")
    ax.set_title("SYNTHETIC: d convergence"); ax.set_xlabel("iteration"); ax.set_ylabel("d (um)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0][1]
    diag = synthetic["mp_diag"]
    ib = diag["in_band"]
    ax.plot(f_thz[ib], np.unwrap(np.angle(synthetic["x"]))[ib], label="arg(x)")
    ax.plot(f_thz[ib], diag["material_phase"][ib], label="minphase material")
    ax.plot(f_thz[ib], diag["excess"][ib], label="excess = -2beta (should be linear)")
    ax.set_title("SYNTHETIC: phase split"); ax.set_xlabel("THz"); ax.set_ylabel("rad")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0][2]
    for label, gap_m in (("argx", synthetic["d_argx"]), ("iter", synthetic["d_iter"]), ("minphase", synthetic["d_minphase"])):
        r_back = synthetic["x"] * np.exp(1j * gap_round_trip_phase(synthetic["frequency_hz"], gap_m, synthetic["theta_gap"]))
        n, _, _ = core.invert_nk_reflection(
            synthetic["frequency_hz"], r_back, synthetic["mask"], theta_rad=synthetic["theta_gap"], n_incident=REFRACTIVE_INDEX_AIR)
        ax.plot(f_thz, n, label=f"{label} (d={gap_m*1e6:.1f})")
    ax.plot(f_thz, synthetic["n_true"], "k--", lw=2, label="true n")
    ax.set_ylim(0, 8); ax.set_title("SYNTHETIC: recovered n"); ax.set_xlabel("THz"); ax.set_ylabel("n")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Real rows: n for each d method, excess-phase linearity, d bar.
    for row, (name, frequency_hz, mask, x, n_versions, mp_diag, band_hz) in enumerate(real_results, start=1):
        f_thz = frequency_hz * 1e-12
        plot_band = _band_mask(frequency_hz, mask, (0.3e12, 2.5e12))
        ax = axes[row][0]
        for label, (n, k, gap_m) in n_versions.items():
            ax.plot(f_thz[plot_band], n[plot_band], label=f"{label} d={gap_m*1e6:.1f}")
        ax.axhline(1.0, color="0.5", ls=":", lw=1.0)
        ax.set_ylim(0, 6); ax.set_title(f"{name}: n by d-method"); ax.set_xlabel("THz"); ax.set_ylabel("n")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row][1]
        ib = mp_diag["in_band"]
        ax.plot(f_thz[ib], np.unwrap(np.angle(x))[ib], label="arg(x)")
        ax.plot(f_thz[ib], mp_diag["material_phase"][ib], label="minphase material")
        ax.plot(f_thz[ib], mp_diag["excess"][ib], label="excess (should be linear)")
        ax.set_title(f"{name}: phase split"); ax.set_xlabel("THz"); ax.set_ylabel("rad")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row][2]
        labels = list(n_versions)
        ax.bar(labels, [n_versions[l][2] * 1e6 for l in labels], color=["C0", "C1", "C2"])
        ax.set_title(f"{name}: gap d estimate"); ax.set_ylabel("d (um)")
        ax.grid(alpha=0.3, axis="y")

    output_path = os.path.join(os.path.dirname(__file__), "deembed_air_gap_iterative.png")
    figure.savefig(output_path, dpi=130, bbox_inches="tight")
    return output_path


def main():
    synthetic = synthetic_test()
    real_results = real_data_test()
    output_path = make_figure(synthetic, real_results)
    print(f"\nSaved figure: {output_path}")
    print("\nReminder: every d here rests on a phase assumption (smoothness or minimum-phase).")
    print("The assumption-light check is d from the 2nd-reflection pulse delay — next.")


if __name__ == "__main__":
    main()
