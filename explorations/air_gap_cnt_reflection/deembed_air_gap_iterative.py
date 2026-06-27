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


def reconstruct_material_phase(magnitude, phase_engine="hilbert", num_coefficients=30):
    """Causal (minimum-phase) reconstruction of arg(r_back) from |r_back|.

    phase_engine: 'hilbert' (legacy Hilbert(ln|.|), edge-prone) or 'mem' (maximum-entropy
    AR model, far better at the low-frequency band EDGE — see mem_phase_retrieval.py and the
    de-embed research report).  Both return the minimum-phase solution; they differ only in
    numerical conditioning near the spectrum endpoints.
    """
    if phase_engine == "hilbert":
        return minimum_phase_from_magnitude(magnitude)
    if phase_engine == "mem":
        from mem_phase_retrieval import maximum_entropy_phase
        return maximum_entropy_phase(magnitude, num_coefficients)
    raise ValueError(f"unknown phase_engine {phase_engine!r} (use 'hilbert' or 'mem')")


def estimate_gap_minimum_phase(frequency_hz, x_complex, mask, gap_angle_rad, band_hz,
                               phase_engine="hilbert", num_coefficients=30):
    """d from the excess phase arg(x) - phi_minphase(|x|), which should be pure -2beta.

    phase_engine selects the causal phase reconstruction ('hilbert' or 'mem').
    """
    in_band = _band_mask(frequency_hz, mask, band_hz)
    material_phase = reconstruct_material_phase(
        np.abs(x_complex), phase_engine=phase_engine, num_coefficients=num_coefficients)
    excess = np.unwrap(np.angle(x_complex)) - material_phase
    gap_thickness_m = gap_from_linear_slope(frequency_hz, excess, in_band, gap_angle_rad)
    return gap_thickness_m, dict(material_phase=material_phase, excess=excess, in_band=in_band)


# ── Method C: amplitude-KK gap-shift (the arXiv 2412.18662 idea, adapted) ─────


def _coarse_to_fine_minimum(cost_function, search_values, refine_steps=2, refine_span=2):
    """Grid-minimise cost_function over search_values, then refine around the best point.

    refine_span = how many grid steps either side of the current best to re-grid into;
    refine_steps = how many refinement passes.  Returns (best_value, best_cost, trace).
    """
    values = np.asarray(search_values, dtype=float)
    trace = []
    for _ in range(refine_steps + 1):
        costs = np.array([cost_function(v) for v in values])
        best_index = int(np.argmin(costs))
        trace.append((values.copy(), costs.copy()))
        step = values[1] - values[0] if values.size > 1 else 0.0
        if step == 0.0:
            break
        low = values[max(best_index - refine_span, 0)]
        high = values[min(best_index + refine_span, values.size - 1)]
        values = np.linspace(low, high, values.size)
    return float(trace[-1][0][int(np.argmin(trace[-1][1]))]), float(np.min(trace[-1][1])), trace


def estimate_gap_amplitude_kk(
    frequency_hz, reflection_measured, reflection_front, mask, gap_angle_rad, band_hz,
    gap_search_m=None, phase_engine="mem", num_coefficients=30, residual="amplitude",
    material_sign=None,
):
    """Estimate the gap d by matching a CAUSAL forward model to the measured reflection.

    This adapts the robust amplitude-minimisation idea of arXiv 2412.18662 ("Robust phase
    correction for THz reflection") to our Fabry-Perot CONTACT-GAP geometry.  Unlike the
    phase-SLOPE estimator (estimate_gap_minimum_phase), the gap here is NOT read off a
    measured-phase slope (degenerate — the material's own dispersion adds slope).  Instead:

      1. De-embed the (known) front interface:  x = (r_meas - r_front)/(1 - r_front r_meas).
         This is EXACT and INDEPENDENT of d; algebraically x = r_back * e^{-i 2 beta(d)}, so
         |x| = |r_back| (a lossless gap does not change the magnitude).
      2. Reconstruct the material reflection by CAUSALITY from its magnitude:
         r_back_causal = |x| * exp(i * phi),  phi = MEM/Hilbert minimum phase of |x|.
         This is fully determined — it does NOT depend on d.
      3. Forward-model the measured reflection for a TRIAL gap d and compare to what was
         actually measured.  The d that best reproduces the measurement wins:
            residual='amplitude' :  min_d  sum_band  ( |r_model(d)| - |r_meas| )^2   (phase-free,
                                    robust — uses only the trustworthy magnitude; faithful to the
                                    arXiv amplitude-minimisation; the gap enters |r_model| via the
                                    Fabry-Perot fringes).
            residual='complex'   :  min_d  sum_band  |r_model(d) - r_meas|^2  (more sensitive for
                                    SMALL gaps where the amplitude barely moves, but uses the
                                    measured phase, so less robust to alignment error).

    TWO FAILURE MODES this method has and the phase-slope estimator does NOT (both found by
    the synthetic test, both real):
      * CONDUCTOR SIGN AMBIGUITY.  A metal-like back reflection carries a ~pi sign flip
        (r_back ~ -1) that minimum-phase-from-magnitude does NOT reproduce.  The forward FP
        model needs the correct sign, so we try material_sign in {+1, -1} and keep the lower
        cost (set material_sign explicitly to disable the search).  The phase-SLOPE estimator
        is immune because a constant pi does not change a slope.
      * SUB-FRINGE DEGENERACY.  The gap enters |r_meas| only through Fabry-Perot fringes of
        period Delta_f = c/(2 d cos theta).  If the band spans much less than one fringe
        (small gap), |r_meas| is nearly flat in d and the amplitude cost is degenerate.  We
        warn when the recovered gap is sub-fringe for the band; trust the phase-slope/MEM
        estimator there instead.  -> This method is for LARGER residual gaps (super-fringe)
        and as a misplacement cross-check, NOT the small good-contact gap.

    Returns (gap_thickness_m, diagnostics).
    """
    in_band = _band_mask(frequency_hz, mask, band_hz)

    x = deembed_gap_layer(reflection_measured, reflection_front)
    magnitude_back = np.abs(x)
    material_phase = reconstruct_material_phase(
        magnitude_back, phase_engine=phase_engine, num_coefficients=num_coefficients)

    measured_in_band = reflection_measured[in_band]
    measured_magnitude_in_band = np.abs(measured_in_band)

    def cost_for_sign(material_sign):
        reflection_back_causal = material_sign * magnitude_back * np.exp(1j * material_phase)

        def cost(gap_thickness_m):
            round_trip = gap_round_trip_phase(frequency_hz, gap_thickness_m, gap_angle_rad)
            model = fabry_perot_reflection(reflection_front, reflection_back_causal, round_trip)
            model_in_band = model[in_band]
            if residual == "amplitude":
                difference = np.abs(model_in_band) - measured_magnitude_in_band
            elif residual == "complex":
                difference = model_in_band - measured_in_band
            else:
                raise ValueError("residual must be 'amplitude' or 'complex'")
            return float(np.mean(np.abs(difference) ** 2))

        return cost, reflection_back_causal

    if gap_search_m is None:
        gap_search_m = np.linspace(0.0, 150e-6, 151)

    signs_to_try = (1.0, -1.0) if material_sign is None else (material_sign,)
    best = None
    for sign in signs_to_try:
        cost, reflection_back_causal = cost_for_sign(sign)
        gap_thickness_m, best_cost, trace = _coarse_to_fine_minimum(cost, gap_search_m)
        if best is None or best_cost < best["best_cost"]:
            best = dict(gap_thickness_m=gap_thickness_m, best_cost=best_cost, trace=trace,
                        material_sign=sign, reflection_back_causal=reflection_back_causal)

    # Fringe-resolvability: is the recovered gap super-fringe over this band?
    band_span_hz = float(frequency_hz[in_band].max() - frequency_hz[in_band].min())
    one_fringe_gap_m = SPEED_OF_LIGHT_M_PER_S / (
        2.0 * np.cos(gap_angle_rad) * max(band_span_hz, 1.0))
    fringes_across_band = best["gap_thickness_m"] / one_fringe_gap_m
    sub_fringe = fringes_across_band < 0.5
    if sub_fringe:
        print(f"  [amplitude-KK WARNING] recovered gap {best['gap_thickness_m']*1e6:.1f} um is "
              f"SUB-FRINGE ({fringes_across_band:.2f} fringes across the band; one fringe needs "
              f"{one_fringe_gap_m*1e6:.0f} um). Amplitude is weakly constraining here — prefer the "
              f"phase-slope/MEM estimator.")

    return best["gap_thickness_m"], dict(
        material_phase=material_phase,
        reflection_back_causal=best["reflection_back_causal"],
        material_sign=best["material_sign"], best_cost=best["best_cost"],
        trace=best["trace"], in_band=in_band, residual=residual,
        fringes_across_band=fringes_across_band, sub_fringe=sub_fringe,
        one_fringe_gap_m=one_fringe_gap_m)


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
    d_minphase, mp_diag = estimate_gap_minimum_phase(
        frequency_hz, x, mask, theta_gap, band_hz, phase_engine="hilbert")
    d_mem, _ = estimate_gap_minimum_phase(
        frequency_hz, x, mask, theta_gap, band_hz, phase_engine="mem")
    d_amp_kk, amp_diag = estimate_gap_amplitude_kk(
        frequency_hz, reflection_measured, reflection_front, mask, theta_gap, band_hz,
        phase_engine="mem", residual="amplitude")
    d_complex_kk, _ = estimate_gap_amplitude_kk(
        frequency_hz, reflection_measured, reflection_front, mask, theta_gap, band_hz,
        phase_engine="mem", residual="complex")

    print("=== SYNTHETIC TEST (planted Drude CNT, known gap) ===")
    print(f"  true gap d            = {true_gap_m*1e6:.2f} um")
    print(f"  d (arg-x slope)       = {d_argx*1e6:.2f} um   (biased baseline)")
    print(f"  d (iterative smooth)  = {d_iter*1e6:.2f} um   (converged from {history[0]*1e6:.2f}, "
          f"{len(history)-1} steps)")
    print(f"  d (minphase, Hilbert) = {d_minphase*1e6:.2f} um   (phase-slope)")
    print(f"  d (minphase, MEM)     = {d_mem*1e6:.2f} um   (phase-slope)")
    print(f"  d (amplitude-KK, MEM) = {d_amp_kk*1e6:.2f} um   (magnitude match, robust)")
    print(f"  d (complex-KK,   MEM) = {d_complex_kk*1e6:.2f} um   (complex match, sensitive)")
    print(f"  -> errors: arg-x {(d_argx-true_gap_m)*1e6:+.2f}, "
          f"iter {(d_iter-true_gap_m)*1e6:+.2f}, "
          f"Hilbert {(d_minphase-true_gap_m)*1e6:+.2f}, MEM {(d_mem-true_gap_m)*1e6:+.2f}, "
          f"ampKK {(d_amp_kk-true_gap_m)*1e6:+.2f}, cplxKK {(d_complex_kk-true_gap_m)*1e6:+.2f} um")
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
