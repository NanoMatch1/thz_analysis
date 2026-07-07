"""Monte-Carlo uncertainty propagation from the transfer function to n / k / eps / sigma.

The transfer function H carries a per-bin uncertainty (``transfer_H_sigma`` / ``transfer_phase_sigma``
from :func:`compute_transfer_uncertainty` — the additive spectral noise floor). The inversion
H -> n,k and the derivation n,k -> eps,sigma are nonlinear, so instead of hand-deriving Jacobians
we push the uncertainty through the *exact same* inversion the pipeline used, by Monte Carlo:

    1. draw N noisy copies of H:  |H| + N(0, sigma_|H|),  arg H + N(0, sigma_phase)
    2. re-run the pipeline's own invert + derive on each draw (dependency-injected ``reinvert``)
    3. the spread of n / k / eps / sigma across the draws IS their uncertainty

This works for transmission and both reflection geometries with no per-geometry formulas, captures
the r -> -1 near-singularity behaviour, and includes the n/k correlation for free.

CAVEAT (why this is a first step, not the final word): the INPUT uncertainty here is only the
additive white-noise floor — it misses signal-correlated noise (timing jitter, amplitude drift)
that inflates the on-peak scatter. So the bars are meaningful but OPTIMISTIC (a lower bound). The
upgrade is to feed a better input sigma (propagate the measured per-timepoint repeat scatter
sigma_t(t) through the windowed DFT); the MC machinery and the viewer wiring stay identical.

Spread estimator: a robust 1-sigma = half the 15.9-84.1 percentile interval, so an occasional
2*pi phase-unwrap flip in a single draw cannot blow up the reported error.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters.thz_adapter import _sample_items

# processing_dict quantities we propagate; complex ones are split into _real/_imag sigmas.
_PROPAGATED = ("n", "k", "eps", "sigma")


def _robust_sigma(stack: np.ndarray) -> np.ndarray:
    """Robust per-bin 1-sigma across draws: half the 15.9-84.1 percentile width."""
    lo, hi = np.nanpercentile(stack, [15.865, 84.135], axis=0)
    return 0.5 * (hi - lo)


def _perturbed_transfer(H0: np.ndarray, sigma_H: np.ndarray, sigma_phase: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """One noisy draw of H: independent Gaussian noise on |H| and arg H."""
    magnitude = np.abs(H0)
    phase = np.angle(H0)
    magnitude_draw = magnitude + rng.standard_normal(magnitude.shape) * np.nan_to_num(sigma_H)
    magnitude_draw = np.clip(magnitude_draw, 0.0, None)          # amplitude is non-negative
    phase_draw = phase + rng.standard_normal(phase.shape) * np.nan_to_num(sigma_phase)
    return magnitude_draw * np.exp(1j * phase_draw)


# ── re-inversion callbacks (dependency injection; match the pipeline's own call) ─


def make_transmission_reinvert(thickness_m: float) -> Callable:
    """``reinvert(dataset)`` = invert_nk(thickness) + derive_eps_sigma (transmission)."""
    def reinvert(dataset):
        thz.invert_nk(dataset, thickness_m=float(thickness_m))
        thz.derive_eps_sigma(dataset)
    return reinvert


def make_reflection_reinvert(**invert_kwargs) -> Callable:
    """``reinvert(dataset)`` = invert_nk_reflection(**kwargs) + derive_eps_sigma."""
    def reinvert(dataset):
        thz.invert_nk_reflection(dataset, **invert_kwargs)
        thz.derive_eps_sigma(dataset)
    return reinvert


# ── the Monte-Carlo stage ────────────────────────────────────────────────────────


def propagate_uncertainty(
    dataset,
    reinvert: Callable,
    *,
    n_draws: int = 200,
    seed: int = 0,
    verbose: bool = True,
):
    """Monte-Carlo-propagate the transfer-function uncertainty to n / k / eps / sigma.

    Writes per-sample ``processing_dict`` keys ``n_sigma``, ``k_sigma``,
    ``eps_real_sigma``, ``eps_imag_sigma``, ``sigma_real_sigma``, ``sigma_imag_sigma``
    (the quantity registry exposes these as the viewer's error bars). Needs
    ``compute_transfer_uncertainty`` to have run first.

    Parameters
    ----------
    reinvert : callable
        ``reinvert(dataset)`` re-runs the pipeline's invert + derive reading the current
        ``transfer_H`` (see :func:`make_transmission_reinvert` / :func:`make_reflection_reinvert`).
    n_draws : int
        Number of Monte-Carlo draws (200 is plenty for a stable spread; cost is N inversions).
    seed : int
        RNG seed for reproducibility (the draws are recorded implicitly via the seed).
    """
    rng = np.random.default_rng(seed)

    samples = [
        (filename, data_obj)
        for filename, data_obj in _sample_items(dataset)
        if data_obj.processing_dict.get("transfer_H") is not None
        and data_obj.processing_dict.get("transfer_H_sigma") is not None
    ]
    if not samples:
        if verbose:
            print("[propagate_uncertainty] no transfer_H_sigma found "
                  "(run compute_transfer_uncertainty first); skipping.")
        return dataset

    # Snapshot originals so we can restore the dataset exactly after resampling.
    originals = {
        filename: {key: data_obj.processing_dict.get(key)
                   for key in ("transfer_H", *_PROPAGATED)}
        for filename, data_obj in samples
    }
    accum = {filename: {q: [] for q in _PROPAGATED} for filename, _ in samples}

    # Don't record the 200 inner invert/derive calls into the replayable recipe.
    previous_recording = getattr(dataset, "_recipe_recording", True)
    dataset._recipe_recording = False
    try:
        for _ in range(int(n_draws)):
            for filename, data_obj in samples:
                processing = data_obj.processing_dict
                processing["transfer_H"] = _perturbed_transfer(
                    originals[filename]["transfer_H"],
                    processing["transfer_H_sigma"],
                    processing.get("transfer_phase_sigma"),
                    rng,
                )
            reinvert(dataset)
            for filename, data_obj in samples:
                processing = data_obj.processing_dict
                for quantity in _PROPAGATED:
                    value = processing.get(quantity)
                    accum[filename][quantity].append(
                        None if value is None else np.asarray(value).copy())
    finally:
        for filename, data_obj in samples:
            data_obj.processing_dict.update(originals[filename])
        dataset._recipe_recording = previous_recording

    for filename, data_obj in samples:
        processing = data_obj.processing_dict
        written = []
        for quantity in _PROPAGATED:
            draws = [a for a in accum[filename][quantity] if a is not None]
            if not draws:
                continue
            stack = np.stack(draws)
            if np.iscomplexobj(stack):
                processing[f"{quantity}_real_sigma"] = _robust_sigma(stack.real)
                processing[f"{quantity}_imag_sigma"] = _robust_sigma(stack.imag)
                written += [f"{quantity}_real_sigma", f"{quantity}_imag_sigma"]
            else:
                processing[f"{quantity}_sigma"] = _robust_sigma(stack)
                written.append(f"{quantity}_sigma")
        if verbose:
            # advertise a representative in-band bar so the scale is visible in the log
            n_sigma = processing.get("n_sigma")
            median_bar = (float(np.nanmedian(n_sigma)) if n_sigma is not None
                          and np.isfinite(np.nanmedian(n_sigma)) else float("nan"))
            print(f"[propagate_uncertainty] '{filename}': {int(n_draws)} draws -> "
                  f"{len(written)} error arrays (median sigma_n = {median_bar:.4f}). "
                  f"** additive-noise-only: optimistic lower bound **")
    return dataset
