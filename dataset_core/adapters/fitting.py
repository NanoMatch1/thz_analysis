"""Interactive model fitting wired into the pipeline + bundle.

``fit_interactive(dataset)`` pulls each sample's derived quantity (sigma / eps / n) straight out of
``processing_dict``, launches ONE multi-file ``FitGUI`` (prev/next navigation) with the pipeline's
Monte-Carlo error bars, and writes each resulting ``FitResult`` back into
``processing_dict['fit_result']``. Because that key is part of the pickled snapshot, the fits then
travel in the ``.thzbundle`` automatically and reappear as the Drude overlay in the registry
``ResultsViewer``.

Replaces the standalone ``fitting_run_me.py`` loop (which opened one window per sample off a saved
database). The headless auto-Drude path (``conductivity_fitting.fit_conductivity``) is unchanged and
complementary.

DESIGN NOTE — interactive fits are NOT replayable. A hand-tuned fit can't be re-derived from config,
so it is stored as a *snapshot artifact* (in ``processing_dict``) and recorded in the recipe as
*provenance only* (model + final params), never as a replayable step.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from dataset_core.adapters.conductivity_fitting import _iter_selected
from dataset_core.services.provenance import _jsonable

_HZ_TO_THZ = 1e-12

# quantity -> (processing_dict data key, real-part error key, imag-part error key or None).
# The error keys are the Monte-Carlo sigmas written by uncertainty.propagate_uncertainty.
_FIT_QUANTITIES = {
    "sigma": ("sigma", "sigma_real_sigma", "sigma_imag_sigma"),
    "eps": ("eps", "eps_real_sigma", "eps_imag_sigma"),
    "n": ("n", "n_sigma", None),
}


def _build_error(processing_dict: dict, real_key: str, imag_key: str | None):
    """Complex error array (real=Re-trace sigma, imag=Im-trace sigma), or None if absent."""
    real_sigma = processing_dict.get(real_key)
    imag_sigma = processing_dict.get(imag_key) if imag_key else None
    if real_sigma is None and imag_sigma is None:
        return None
    length = len(processing_dict["fft_freq"])
    real_part = np.asarray(real_sigma, dtype=float) if real_sigma is not None else np.zeros(length)
    imag_part = np.asarray(imag_sigma, dtype=float) if imag_sigma is not None else np.zeros(length)
    return real_part + 1j * imag_part


def _report(filename: str, model: str, result) -> None:
    if result is None or not getattr(result, "success", False):
        print(f"[fit_interactive] '{filename}': closed without a successful fit.")
        return
    params = {**result.param_values, **result.fixed_params}
    param_str = ", ".join(f"{name}={value:.4g}" for name, value in params.items())
    print(f"[fit_interactive] '{filename}': {model}  {param_str}  (R^2={result.r_squared:.4f})")


def fit_interactive(
    dataset,
    *,
    quantity: str = "sigma",
    model: str = "drude_conductivity",
    samples=None,
    fit_band_thz: tuple[float, float] | None = None,
    store: bool = True,
    _gui_launcher=None,
):
    """Launch the interactive fit GUI on a dataset's results; write fits back into it.

    Parameters
    ----------
    quantity : {'sigma', 'eps', 'n'}
        Which derived quantity to fit. ``sigma``/``eps`` are complex; ``n`` is real.
    model : str
        Initial model in the GUI (any key in the fitting MODEL_REGISTRY).
    samples : None | str | list[str]
        Restrict to matching sample filenames (substring). None = all samples.
    fit_band_thz : (lo, hi) or None
        Preset the GUI's fit range (THz).
    store : bool
        Write each ``FitResult`` back into ``processing_dict['fit_result']`` (default True).
    _gui_launcher : callable or None
        Injection seam for tests (defaults to ``thz_core.fitting.launch_fit_gui``).

    Returns
    -------
    dict[str, FitResult | None]
        Per-sample fit results (also stored on the dataset when ``store``).
    """
    if quantity not in _FIT_QUANTITIES:
        raise ValueError(f"quantity must be one of {list(_FIT_QUANTITIES)}, got {quantity!r}.")
    data_key, real_error_key, imag_error_key = _FIT_QUANTITIES[quantity]

    # Collect the samples that actually carry this quantity + a shared frequency axis.
    entries = []  # (filename, data_obj, freq, data, mask, error)
    reference_freq = None
    for filename, data_obj in _iter_selected(dataset, samples):
        processing = data_obj.processing_dict
        freq = processing.get("fft_freq")
        data = processing.get(data_key)
        if freq is None or data is None:
            continue
        freq = np.asarray(freq, dtype=float)
        if reference_freq is None:
            reference_freq = freq
        elif freq.shape != reference_freq.shape:
            raise ValueError(
                f"fit_interactive needs a shared frequency axis, but '{filename}' has a different "
                f"length ({freq.size} vs {reference_freq.size}). Fit mismatched grids separately "
                f"with the `samples=` filter (common when merging runs at different resolutions)."
            )
        entries.append((
            filename, data_obj, freq, np.asarray(data),
            processing.get("transfer_mask"),
            _build_error(processing, real_error_key, imag_error_key),
        ))

    if not entries:
        print(f"[fit_interactive] no samples carry '{data_key}'; nothing to fit.")
        return {}

    data_dict = {name: data for name, _, _, data, _, _ in entries}
    mask_dict = {name: mask for name, _, _, _, mask, _ in entries if mask is not None}
    error_dict = {name: error for name, _, _, _, _, error in entries if error is not None}

    # Build the GUI's fitting config (model + optional preset fit range).
    from thz_core.thz_core.fitting import default_fit_config
    fit_config = default_fit_config(model)
    if fit_band_thz is not None:
        fit_config["fit_range_hz"] = (fit_band_thz[0] / _HZ_TO_THZ, fit_band_thz[1] / _HZ_TO_THZ)

    launcher = _gui_launcher
    if launcher is None:
        from thz_core.thz_core.fitting import launch_fit_gui as launcher

    try:
        results = launcher(
            reference_freq, data_dict=data_dict, mask_dict=mask_dict,
            error_dict=error_dict or None, initial_model=model,
            config={"fitting": fit_config},
        )
    except Exception as exc:  # e.g. tkinter TclError with no display
        print(f"[fit_interactive] GUI unavailable ({type(exc).__name__}: {exc}); skipping.")
        return {}

    results = results or {}
    for filename, data_obj, *_ in entries:
        result = results.get(filename)
        if store and result is not None:
            data_obj.processing_dict["fit_result"] = result
        _report(filename, model, result)

    _record_fit_provenance(dataset, quantity, model, fit_band_thz, results)
    return results


def _record_fit_provenance(dataset, quantity, model, fit_band_thz, results) -> None:
    """Append a PROVENANCE-only recipe entry (interactive fits are not replayable)."""
    recipe = getattr(dataset, "recipe", None)
    if recipe is None:
        return
    fit_params = {
        filename: _jsonable({**result.param_values, **result.fixed_params,
                             "r_squared": result.r_squared})
        for filename, result in results.items()
        if result is not None and getattr(result, "success", False)
    }
    recipe.append({
        "stage": "fit_interactive",
        "kwargs": {"quantity": quantity, "model": model, "fit_band_thz": _jsonable(fit_band_thz)},
        "provenance_only": True,   # cannot be replayed; records the fit that was performed
        "results": fit_params,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
