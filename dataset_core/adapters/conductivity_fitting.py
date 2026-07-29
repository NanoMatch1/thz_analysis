"""Headless, pipeline-agnostic Drude fitting + KK-consistent geometry optimisation.

Why this module exists
----------------------
The imaginary optical conductivity sigma_2 = omega*eps0*(eps_inf - n^2 + k^2) is a
small *difference of two large numbers*, so it is hypersensitive to the sample
thickness d (through n^2) and to eps_inf.  sigma_1 = omega*eps0*2nk is robust.  A
common failure mode: you can Drude-fit sigma_1 OR sigma_2 alone, but not both with one
(sigma_dc, tau) — a causality/KK inconsistency that signals the wrong d (or eps_inf).

The physically correct thickness is the single value at which sigma_1 AND sigma_2 obey
*one* Drude.  That is information the Fabry-Perot fringe method does not use (and cannot
use here — gating removes the fringes, and Delta_f = c/2nd is circular because it needs n).

This module provides three things, all reusable across the transmission,
single-reflection, and window-reflection pipelines:

1. ``joint_drude_fit``        - the pure primitive: one Drude fitted to the COMPLEX
                                sigma (sigma_1 and sigma_2 together).  Deterministic,
                                testable, no dataset dependency.
2. ``fit_conductivity``       - headless pipeline stage: pull sigma off every sample,
                                joint-fit, store the result, print a ``[fit_conductivity]``
                                reporter.  Replaces the per-sample blocking FitGUI loop.
3. ``optimize_geometry_parameter`` / ``optimize_thickness``
                              - vary a geometric parameter (thickness for transmission,
                                or anything via an injected ``apply_value`` callback for
                                reflection) and pick the value that MAXIMISES the joint
                                Drude R^2.  Fast, because ``transfer_H`` does not depend on
                                the geometry: only invert -> derive -> fit re-runs.
4. ``thickness_slider``       - optional GUI modality: a d slider with a live joint fit and
                                an "optimise" button.  Headless-guarded.

Design
------
Dependency injection throughout (per the project style): the optimiser and the slider
take an ``apply_value(dataset, value)`` callback that applies the trial geometry and
repopulates ``processing_dict['sigma']``.  The transmission default is provided; a
reflection pipeline passes its own (e.g. an air-gap re-inversion).  Nothing here is
hard-wired to one geometry.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from thz_core.thz_core.fitting import fit_model, FitResult
from dataset_core.adapters import thz_adapter as thz

_EPS0 = 8.854187817e-12
_HZ_TO_THZ = 1e-12


def _iter_selected(dataset, samples):
    """Yield (filename, data_obj) for the selected non-reference samples.

    ``samples`` selects which wafers vote: ``None`` = all; a substring, or a list of
    substrings, keeps only matching filenames.  Thickness is a PER-WAFER property and an
    intrinsic (non-Drude) wafer must not be averaged into a Drude objective, so the
    optimiser is normally pointed at a single conductive sample.
    """
    if isinstance(samples, str):
        samples = [samples]
    for filename, data_obj in thz._sample_items(dataset):
        if samples is None or any(token.lower() in filename.lower() for token in samples):
            yield filename, data_obj


# ── Pure primitive: one Drude to sigma_1 AND sigma_2 ───────────────────────────


def joint_drude_fit(
    freq_hz: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    fit_band_thz: tuple[float, float] | None = None,
    model: str = "drude_conductivity",
    initial: dict | None = None,
    bounds: dict | None = None,
    fixed: dict | None = None,
    tau_min: float | None = 1e-14,
    max_nfev: int = 20000,
) -> tuple[FitResult, dict]:
    """Fit ONE Drude model to the complex conductivity (real + imag jointly).

    This is the KK-consistency primitive: ``fit_component='both'`` weights sigma_1 and
    sigma_2 equally, so a good fit means one (sigma_dc, tau) explains BOTH — which only
    happens at the correct geometry.  A high R^2 here is the degeneracy-breaker.

    ``tau_min`` (default ~10 fs) sets a physical lower bound on the scattering time. The engine
    default (1e-16 s) is effectively zero, so at a WRONG thickness the joint fit can rail
    ``tau -> 0`` — which flattens sigma_1 to a constant (no roll-off) and reports a spuriously
    "converged" but meaningless fit. The floor is a robustness guard (a genuinely sub-10-fs tau
    is unphysical for these doped semiconductors and usually signals fit collapse), NOT a
    substitute for the correct geometry — the joint R^2 is still the KK-consistency signal. Pass
    ``tau_min=None`` (or your own ``bounds['tau']``) to opt out.

    Parameters
    ----------
    freq_hz : np.ndarray
        Frequency axis (Hz).
    sigma : np.ndarray
        Complex conductivity (S/m), same length as ``freq_hz``.
    mask : np.ndarray or None
        Trusted-bin boolean mask (e.g. the SNR mask).  Combined with ``fit_band_thz``.
    fit_band_thz : (lo, hi) or None
        Restrict the fit to this THz band (converted to Hz).  ``None`` uses the mask /
        all finite bins.
    model : str
        Any conductivity model in the fitting registry (``'drude_conductivity'``,
        ``'drude_smith_conductivity'``, ...).
    initial, bounds, fixed : dict or None
        Optional per-parameter initial guesses / (lo, hi) bounds / fixed values.
    max_nfev : int
        Optimiser iteration cap.

    Returns
    -------
    (FitResult, metrics_dict)
    """
    fit_cfg: dict = {"model": model, "fit_component": "both", "max_nfev": int(max_nfev)}
    if fit_band_thz is not None:
        fit_cfg["fit_range_hz"] = (
            float(fit_band_thz[0]) / _HZ_TO_THZ,
            float(fit_band_thz[1]) / _HZ_TO_THZ,
        )

    # tau lower-bound guard for the tau-bearing models (unless the caller overrides tau's bound).
    fit_bounds = {k: [float(v[0]), float(v[1])] for k, v in bounds.items()} if bounds else {}
    model_has_tau = model in ("drude_conductivity", "drude_smith_conductivity")
    if tau_min and model_has_tau and "tau" not in fit_bounds:
        fit_bounds["tau"] = [float(tau_min), 1e-10]

    fit_initial = dict(initial) if initial else None
    if fit_initial and tau_min and model_has_tau and "tau" in fit_initial:
        # curve_fit requires the initial guess to lie within the bounds.
        fit_initial["tau"] = max(float(fit_initial["tau"]), float(tau_min))

    if fit_initial:
        fit_cfg["initial_params"] = fit_initial
    if fit_bounds:
        fit_cfg["bounds"] = fit_bounds
    if fixed:
        fit_cfg["fixed_params"] = dict(fixed)

    return fit_model(freq_hz, sigma, model, {"fitting": fit_cfg}, mask=mask)


def _seed_drude_from_sigma(
    freq_hz: np.ndarray, sigma: np.ndarray, mask: np.ndarray | None
) -> dict:
    """Data-driven initial (sigma_dc, tau) so the optimiser starts near the answer.

    Uses sigma_1 (robust) only: sigma_dc ~ the low-frequency real conductivity, and tau
    from the roll-off (sigma_1 halves near omega*tau = 1).  Rough but keeps curve_fit's
    parameter scaling well-conditioned across a thickness sweep.
    """
    finite = np.isfinite(sigma) & (freq_hz > 0)
    if mask is not None:
        finite = finite & np.asarray(mask, dtype=bool)
    if np.count_nonzero(finite) < 3:
        return {"sigma_dc": 100.0, "tau": 1e-13}
    f = freq_hz[finite]
    s1 = np.real(sigma)[finite]
    order = np.argsort(f)
    f, s1 = f[order], s1[order]
    sigma_dc = float(np.clip(np.median(s1[: max(2, len(s1) // 5)]), 1e-3, None))
    # frequency where sigma_1 drops to half its low-f value -> omega*tau ~ 1
    half = sigma_dc / 2.0
    below = np.where(s1 <= half)[0]
    if below.size:
        f_half = f[below[0]]
        tau = 1.0 / (2.0 * np.pi * f_half) if f_half > 0 else 1e-13
    else:
        tau = 1e-13
    return {"sigma_dc": sigma_dc, "tau": float(np.clip(tau, 1e-15, 5e-12))}


# ── Headless pipeline stage ────────────────────────────────────────────────────


def fit_conductivity(
    dataset,
    *,
    samples=None,
    model: str = "drude_conductivity",
    fit_band_thz: tuple[float, float] | None = None,
    initial: dict | None = None,
    bounds: dict | None = None,
    fixed: dict | None = None,
    store: bool = True,
    verbose: bool = True,
) -> dict[str, FitResult]:
    """Joint-Drude fit every selected sample's conductivity in place (headless).

    Reads ``processing_dict['sigma']`` + ``['fft_freq']`` + ``['transfer_mask']`` (as set
    by ``derive_eps_sigma``) and stores the ``FitResult`` under
    ``processing_dict['fit_result']``.  Prints a ``[fit_conductivity]`` reporter per
    sample with sigma_dc, tau, the plasma frequency and the joint R^2.

    ``samples`` (None / substring / list of substrings) restricts which wafers are fitted.

    Returns ``{filename: FitResult}``.
    """
    results: dict[str, FitResult] = {}
    for filename, data_obj in _iter_selected(dataset, samples):
        processing = data_obj.processing_dict
        freq = processing.get('fft_freq')
        sigma = processing.get('sigma')
        if freq is None or sigma is None:
            if verbose:
                print(f"[fit_conductivity] '{filename}': no sigma; skipping.")
            continue
        mask = processing.get('transfer_mask')
        seed = initial or _seed_drude_from_sigma(freq, sigma, mask)
        result, _metrics = joint_drude_fit(
            freq, sigma, mask,
            fit_band_thz=fit_band_thz, model=model,
            initial=seed, bounds=bounds, fixed=fixed,
        )
        if store:
            processing['fit_result'] = result
        results[filename] = result
        if verbose:
            print(_format_fit_report(filename, result))
    return results


def _format_fit_report(filename: str, result: FitResult) -> str:
    """One-line ``[fit_conductivity]`` reporter with the physical readouts."""
    if not result.success:
        return f"[fit_conductivity] '{filename}': FIT FAILED — {result.message}"
    params = {**result.param_values, **result.fixed_params}
    sigma_dc = params.get('sigma_dc', float('nan'))
    tau = params.get('tau', float('nan'))
    plasma_thz = crossover_thz = float('nan')
    if np.isfinite(sigma_dc) and np.isfinite(tau) and tau > 0:
        plasma_thz = np.sqrt(sigma_dc / (_EPS0 * tau)) / (2.0 * np.pi) * _HZ_TO_THZ
        crossover_thz = 1.0 / (2.0 * np.pi * tau) * _HZ_TO_THZ
    extra = ""
    if 'c' in params:
        extra = f", c={params['c']:+.3f}"
    return (
        f"[fit_conductivity] '{filename}': sigma_DC={sigma_dc:.1f} S/m, "
        f"tau={tau * 1e15:.0f} fs{extra}  ->  f_plasma={plasma_thz:.3f} THz, "
        f"sigma1=sigma2 crossover={crossover_thz:.3f} THz;  joint R^2={result.r_squared:.4f} "
        f"(sigma1 & sigma2 fitted together)"
    )


# ── Geometry re-inversion callbacks (dependency injection) ─────────────────────


def transmission_reinvert(thickness_key: str = 'thickness_m') -> Callable:
    """Return an ``apply_value(dataset, thickness_m)`` for the transmission pipeline.

    Re-inverts n,k from the (thickness-independent) stored ``transfer_H`` at the trial
    thickness, then re-derives eps/sigma.  This is the cheap inner loop the optimiser and
    slider call — no FFT, no windowing, no reload.
    """
    def apply_value(dataset, thickness_m: float) -> None:
        thz.invert_nk(dataset, thickness_m=float(thickness_m))
        thz.derive_eps_sigma(dataset)
    return apply_value


# ── KK-consistent geometry optimisation ────────────────────────────────────────


def _joint_badness(
    dataset,
    *,
    samples,
    model: str,
    fit_band_thz: tuple[float, float] | None,
    initial: dict | None,
    bounds: dict | None,
    fixed: dict | None,
) -> tuple[float, dict[str, FitResult]]:
    """Mean (1 - joint_R^2) across the selected samples — the objective to minimise.

    R^2 is scale-invariant, so it stays comparable as the geometry rescales sigma.  A
    value near 0 means one Drude cleanly explains both sigma_1 and sigma_2 (KK-consistent).
    """
    badness = []
    per_sample: dict[str, FitResult] = {}
    for filename, data_obj in _iter_selected(dataset, samples):
        processing = data_obj.processing_dict
        freq = processing.get('fft_freq')
        sigma = processing.get('sigma')
        if freq is None or sigma is None:
            continue
        mask = processing.get('transfer_mask')
        seed = initial or _seed_drude_from_sigma(freq, sigma, mask)
        result, _ = joint_drude_fit(
            freq, sigma, mask,
            fit_band_thz=fit_band_thz, model=model,
            initial=seed, bounds=bounds, fixed=fixed,
        )
        per_sample[filename] = result
        r2 = result.r_squared if (result.success and np.isfinite(result.r_squared)) else -1.0
        badness.append(1.0 - r2)
    mean_badness = float(np.mean(badness)) if badness else np.inf
    return mean_badness, per_sample


def optimize_geometry_parameter(
    dataset,
    apply_value: Callable,
    bounds: tuple[float, float],
    *,
    samples=None,
    param_label: str = "parameter",
    display_scale: float = 1.0,
    display_unit: str = "",
    model: str = "drude_conductivity",
    fit_band_thz: tuple[float, float] | None = None,
    fit_initial: dict | None = None,
    fit_bounds: dict | None = None,
    fit_fixed: dict | None = None,
    n_scan: int = 25,
    refine: bool = True,
    verbose: bool = True,
) -> dict:
    """Pick the geometric value that makes sigma_1 & sigma_2 obey ONE Drude.

    Generic optimiser: ``apply_value(dataset, value)`` applies a trial geometry (thickness,
    air-gap, ...) and repopulates sigma; this maximises the joint Drude R^2 over ``bounds``.
    Works for any pipeline via the injected callback.

    Parameters
    ----------
    apply_value : callable
        ``apply_value(dataset, value) -> None``.  Must leave ``processing_dict['sigma']``
        updated for every sample (e.g. :func:`transmission_reinvert`).
    bounds : (lo, hi)
        Search interval for the geometric value (SI units the callback expects).
    param_label, display_scale, display_unit :
        Reporting only — e.g. ``'thickness', 1e6, 'um'`` prints micrometres.
    n_scan : int
        Coarse grid points across ``bounds`` first (robust against local minima).  The
        refine step then polishes the best grid cell with a bounded scalar minimise.
    refine : bool
        Whether to run the bounded local refinement after the scan.

    Returns
    -------
    dict with keys: ``best_value``, ``best_badness``, ``best_joint_r2``,
    ``scan_values``, ``scan_badness``, ``per_sample`` (FitResults at the optimum).
    The dataset is LEFT at the optimum (sigma re-derived there).
    """
    from scipy.optimize import minimize_scalar

    lo, hi = float(bounds[0]), float(bounds[1])

    def objective(value: float) -> float:
        apply_value(dataset, value)
        badness, _ = _joint_badness(
            dataset, samples=samples, model=model, fit_band_thz=fit_band_thz,
            initial=fit_initial, bounds=fit_bounds, fixed=fit_fixed,
        )
        return badness

    # Coarse scan (robust bracket)
    scan_values = np.linspace(lo, hi, max(2, int(n_scan)))
    scan_badness = np.array([objective(v) for v in scan_values])
    best_idx = int(np.argmin(scan_badness))
    best_value = float(scan_values[best_idx])
    best_badness = float(scan_badness[best_idx])

    # Local refinement inside the neighbouring grid cells
    if refine and scan_values.size >= 3:
        left = scan_values[max(0, best_idx - 1)]
        right = scan_values[min(scan_values.size - 1, best_idx + 1)]
        if right > left:
            result = minimize_scalar(
                objective, bounds=(left, right), method='bounded',
                options={'xatol': (hi - lo) * 1e-4},
            )
            if result.fun <= best_badness:
                best_value = float(result.x)
                best_badness = float(result.fun)

    # Leave the dataset at the optimum and collect the fits there
    apply_value(dataset, best_value)
    _, per_sample = _joint_badness(
        dataset, samples=samples, model=model, fit_band_thz=fit_band_thz,
        initial=fit_initial, bounds=fit_bounds, fixed=fit_fixed,
    )
    best_joint_r2 = 1.0 - best_badness

    if verbose:
        shown = best_value * display_scale
        print(
            f"[optimize_{param_label}] best {param_label} = {shown:.3f} {display_unit} "
            f"(joint sigma1+sigma2 Drude R^2 = {best_joint_r2:.4f}; "
            f"scanned {scan_values.size} pts over "
            f"[{lo * display_scale:.2f}, {hi * display_scale:.2f}] {display_unit})"
        )
        for filename, result in per_sample.items():
            print("  " + _format_fit_report(filename, result))

    return {
        "best_value": best_value,
        "best_badness": best_badness,
        "best_joint_r2": best_joint_r2,
        "scan_values": scan_values,
        "scan_badness": scan_badness,
        "per_sample": per_sample,
    }


def optimize_thickness(
    dataset,
    bounds_m: tuple[float, float],
    *,
    samples=None,
    reinvert: Callable | None = None,
    model: str = "drude_conductivity",
    fit_band_thz: tuple[float, float] | None = None,
    n_scan: int = 25,
    refine: bool = True,
    verbose: bool = True,
    **fit_overrides,
) -> dict:
    """Transmission convenience wrapper of :func:`optimize_geometry_parameter`.

    Finds the wafer thickness (metres) at which one Drude fits both sigma_1 and sigma_2.
    ``reinvert`` defaults to :func:`transmission_reinvert`; pass your own for reflection
    (e.g. an air-gap re-inversion) to optimise that parameter instead.

    ``samples`` should name the conductive wafer to calibrate (thickness is per-wafer, so
    do NOT let an intrinsic/non-Drude wafer vote — see :func:`_iter_selected`).
    """
    apply_value = reinvert or transmission_reinvert()
    return optimize_geometry_parameter(
        dataset, apply_value, bounds_m, samples=samples,
        param_label="thickness", display_scale=1e6, display_unit="um",
        model=model, fit_band_thz=fit_band_thz,
        fit_initial=fit_overrides.get('fit_initial'),
        fit_bounds=fit_overrides.get('fit_bounds'),
        fit_fixed=fit_overrides.get('fit_fixed'),
        n_scan=n_scan, refine=refine, verbose=verbose,
    )


# ── Optional GUI modality: a thickness slider with a live joint fit ────────────


def thickness_slider(
    dataset,
    bounds_m: tuple[float, float],
    *,
    reinvert: Callable | None = None,
    init_value_m: float | None = None,
    model: str = "drude_conductivity",
    fit_band_thz: tuple[float, float] | None = None,
    sample: str | None = None,
    block: bool = True,
    show: bool = True,
):
    """Interactive d slider: live re-inversion + joint Drude fit + overlay.

    Drag the slider to set the thickness; sigma is re-inverted from the stored transfer
    function and a joint Drude is refitted live, so you watch sigma_2 slide up/down and the
    joint R^2 respond.  The "Optimise" button jumps to the KK-consistent thickness
    (:func:`optimize_thickness`).

    Headless-safe: on a non-interactive backend (e.g. Agg) it builds the figure and
    returns the objects without blocking, so a smoke test can drive it.  ``block=False``
    also returns immediately with the widget handles kept alive on the returned dict.

    Returns a dict with the matplotlib handles (figure, slider, button) so callers/tests
    can introspect or drive them.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    from thz_core.thz_core.fitting import drude_conductivity

    apply_value = reinvert or transmission_reinvert()

    filename, data_obj = thz._select_sample(dataset, sample)
    lo, hi = float(bounds_m[0]), float(bounds_m[1])
    init_value = float(init_value_m) if init_value_m is not None else 0.5 * (lo + hi)

    figure, (sigma_axis, info_axis) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 1]},
    )
    plt.subplots_adjust(bottom=0.22)
    info_axis.axis('off')
    slider_axis = figure.add_axes([0.12, 0.08, 0.6, 0.04])
    button_axis = figure.add_axes([0.80, 0.06, 0.12, 0.06])
    thickness_slider_widget = Slider(
        slider_axis, 'd (um)', lo * 1e6, hi * 1e6, valinit=init_value * 1e6,
    )
    optimise_button = Button(button_axis, 'Optimise')

    state: dict = {}

    def redraw(thickness_m: float) -> None:
        apply_value(dataset, thickness_m)
        processing = data_obj.processing_dict
        freq = processing.get('fft_freq')
        sigma = processing.get('sigma')
        mask = processing.get('transfer_mask')
        seed = _seed_drude_from_sigma(freq, sigma, mask)
        result, _ = joint_drude_fit(
            freq, sigma, mask, fit_band_thz=fit_band_thz, model=model, initial=seed,
        )
        state['result'] = result

        f_thz = freq * _HZ_TO_THZ
        show_band = np.isfinite(sigma)
        if mask is not None:
            show_band = show_band & mask
        sigma_axis.clear()
        sigma_axis.plot(f_thz[show_band], np.real(sigma)[show_band], 'o', ms=3,
                        color='C0', label=r'$\sigma_1$ data')
        sigma_axis.plot(f_thz[show_band], np.imag(sigma)[show_band], 's', ms=3,
                        color='C1', label=r'$\sigma_2$ data')
        if result.success:
            omega = 2.0 * np.pi * freq
            params = {**result.param_values, **result.fixed_params}
            if model == 'drude_conductivity':
                model_sigma = drude_conductivity(omega, params['sigma_dc'], params['tau'])
                sigma_axis.plot(f_thz, np.real(model_sigma), '-', color='C0', lw=1.2,
                                label=r'Drude $\sigma_1$')
                sigma_axis.plot(f_thz, np.imag(model_sigma), '-', color='C1', lw=1.2,
                                label=r'Drude $\sigma_2$')
        sigma_axis.axhline(0, color='gray', lw=0.5)
        sigma_axis.set_xlabel('Frequency (THz)')
        sigma_axis.set_ylabel('Conductivity (S/m)')
        if fit_band_thz is not None:
            sigma_axis.set_xlim(0, fit_band_thz[1] * 1.3)
        sigma_axis.legend(fontsize=8)
        sigma_axis.set_title(f'{filename}')

        # readout panel
        info_axis.clear(); info_axis.axis('off')
        lines = [f"d = {thickness_m * 1e6:.2f} um", ""]
        if result.success:
            params = {**result.param_values, **result.fixed_params}
            sigma_dc, tau = params['sigma_dc'], params['tau']
            plasma = np.sqrt(sigma_dc / (_EPS0 * tau)) / (2 * np.pi) * _HZ_TO_THZ
            cross = 1.0 / (2 * np.pi * tau) * _HZ_TO_THZ
            lines += [
                f"sigma_DC = {sigma_dc:.1f} S/m",
                f"tau = {tau * 1e15:.0f} fs",
                f"f_plasma = {plasma:.3f} THz",
                f"crossover = {cross:.3f} THz",
                "",
                f"joint R^2 = {result.r_squared:.4f}",
            ]
        else:
            lines += ["FIT FAILED"]
        info_axis.text(0.0, 0.95, "\n".join(lines), va='top', ha='left',
                       family='monospace', fontsize=10, transform=info_axis.transAxes)
        sigma_axis.axvline(cross, color='gray', lw=0.5, ls='--', label='sigma1=sigma2 crossover')
        figure.canvas.draw_idle()

    def on_slider(_value) -> None:
        redraw(thickness_slider_widget.val * 1e-6)

    def on_optimise(_event) -> None:
        result = optimize_thickness(
            dataset, (lo, hi), reinvert=apply_value, model=model,
            fit_band_thz=fit_band_thz, verbose=True,
        )
        thickness_slider_widget.set_val(result['best_value'] * 1e6)  # triggers redraw

    thickness_slider_widget.on_changed(on_slider)
    optimise_button.on_clicked(on_optimise)
    redraw(init_value)

    # Keep the widgets alive for the window's whole lifetime. matplotlib holds only a WEAK
    # reference to Slider/Button callbacks, so if the only strong refs are this function's
    # locals they are GC'd the moment the caller (which usually discards the return value)
    # lets the frame go — leaving a visible but DEAD (non-responsive) slider. Pinning them
    # to the figure ties their lifetime to the window, so drag events keep firing even when
    # show() does not block (interactive mode) and the caller keeps no handle.
    figure._thickness_slider_widgets = (thickness_slider_widget, optimise_button)

    handles = {
        'figure': figure, 'slider': thickness_slider_widget,
        'button': optimise_button, 'redraw': redraw, 'sample': filename,
    }

    interactive = 'agg' not in matplotlib.get_backend().lower()
    if show and interactive:
        # Force a genuinely blocking event loop. With matplotlib interactive mode ON,
        # plt.show(block=True) can return immediately, so the pipeline script races ahead
        # (to save_database / result_viewer) and the slider never gets its own loop. Drop
        # to non-interactive for the blocking show, then restore the caller's mode.
        was_interactive = plt.isinteractive()
        try:
            if block:
                plt.ioff()
            plt.show(block=block)
        finally:
            if was_interactive:
                plt.ion()
    return handles
