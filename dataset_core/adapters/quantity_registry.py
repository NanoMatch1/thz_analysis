"""Registry of displayable / exportable THz-TDS quantities — the single source of truth.

Every quantity a viewer can plot or an exporter can write is registered here ONCE, with how to
pull it out of a sample's ``processing_dict``. Adding a new quantity (a new derived array, a new
fit overlay) is a single ``register(...)`` call — the new results viewer and the CSV export both
pick it up automatically, so display and export can never drift apart.

A quantity is a scalar series vs. frequency. Complex results (H, eps, sigma, the FFT spectrum) are
registered as two real quantities (real/imag or mag/phase) so display and export stay uniform —
one quantity = one line = one CSV column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

_HZ_TO_THZ = 1e-12


@dataclass
class Quantity:
    """One displayable/exportable series vs frequency.

    ``extract(processing_dict) -> (freq_hz, y)`` or ``None`` when the inputs are missing.
    ``overlay`` optionally returns a model curve ``(freq_hz, y)`` (e.g. a Drude fit) to draw on
    top. ``mask_key`` / ``error_key`` name the ``processing_dict`` entries the viewer uses for the
    trusted-band shading and error bars.
    """
    name: str
    label: str
    group: str
    y_label: str
    extract: Callable[[dict], tuple | None]
    mask_key: str | None = "transfer_mask"
    error_key: str | None = None
    yscale: str = "linear"
    include_references: bool = False
    export_header: str | None = None      # CSV column name; None => not exported
    overlay: Callable[[dict], tuple | None] | None = None


QUANTITY_REGISTRY: dict[str, Quantity] = {}


def register(quantity: Quantity) -> Quantity:
    """Register a quantity (idempotent by name). Returns it for convenience."""
    QUANTITY_REGISTRY[quantity.name] = quantity
    return quantity


def display_quantities() -> list[Quantity]:
    """All registered quantities, in registration order (for the viewer's selector)."""
    return list(QUANTITY_REGISTRY.values())


def export_quantities_list() -> list[Quantity]:
    """Quantities that define a CSV column (``export_header`` set), in registration order."""
    return [q for q in QUANTITY_REGISTRY.values() if q.export_header]


def available_names(processing_dict: dict) -> list[str]:
    """Names of quantities whose ``extract`` yields data for this sample."""
    names = []
    for name, quantity in QUANTITY_REGISTRY.items():
        try:
            if quantity.extract(processing_dict) is not None:
                names.append(name)
        except Exception:
            continue
    return names


# ── extraction helpers ──────────────────────────────────────────────────────────


def _get(processing_dict: dict, key: str):
    return processing_dict.get(key)


def _freq(processing_dict: dict):
    return processing_dict.get("fft_freq")


def _real_series(value_key: str, transform: Callable = lambda a: a) -> Callable:
    """Extractor for a real array stored under ``value_key`` (vs fft_freq)."""
    def extract(processing_dict: dict):
        freq = _freq(processing_dict)
        value = processing_dict.get(value_key)
        if freq is None or value is None:
            return None
        return np.asarray(freq), transform(np.asarray(value))
    return extract


def _complex_component(value_key: str, component: str) -> Callable:
    """Extractor for real/imag/abs/angle of a complex array under ``value_key``."""
    func = {"real": np.real, "imag": np.imag, "abs": np.abs, "angle": np.angle}[component]

    def extract(processing_dict: dict):
        freq = _freq(processing_dict)
        value = processing_dict.get(value_key)
        if freq is None or value is None:
            return None
        return np.asarray(freq), func(np.asarray(value))
    return extract


# ── fit overlays (registry-driven: a new fit model plugs its overlay in here) ────


def _evaluate_conductivity_fit(processing_dict: dict):
    """Evaluate a stored conductivity FitResult over the frequency axis -> complex sigma.

    Supports the flat-parameter conductivity models (drude, drude-smith). Oscillator-list
    models (lorentz) are skipped. Returns ``(freq_hz, complex_sigma)`` or ``None``.
    """
    fit_result = processing_dict.get("fit_result")
    freq = _freq(processing_dict)
    if fit_result is None or freq is None or not getattr(fit_result, "success", False):
        return None
    try:
        from thz_core.thz_core.fitting import MODEL_REGISTRY, get_param_names
    except Exception:
        return None
    model_name = getattr(fit_result, "model_name", None)
    info = MODEL_REGISTRY.get(model_name)
    if info is None or info.get("target") != "sigma":
        return None
    param_names = get_param_names(model_name, getattr(fit_result, "n_oscillators", None))
    params = {**getattr(fit_result, "param_values", {}), **getattr(fit_result, "fixed_params", {})}
    try:
        ordered = [float(params[name]) for name in param_names]
    except (KeyError, TypeError, ValueError):
        return None
    omega = 2.0 * np.pi * np.asarray(freq, dtype=float)
    try:
        sigma = info["function"](omega, *ordered)
    except Exception:
        return None
    return np.asarray(freq), np.asarray(sigma)


def _conductivity_fit_component(component: str) -> Callable:
    func = {"real": np.real, "imag": np.imag}[component]

    def overlay(processing_dict: dict):
        result = _evaluate_conductivity_fit(processing_dict)
        if result is None:
            return None
        freq, sigma = result
        return freq, func(sigma)
    return overlay


# ── built-in quantities (single source of truth) ─────────────────────────────────

# fft_sigma / fft_phase_sigma are written by thz_adapter.compute_spectrum_uncertainty.
register(Quantity(
    name="fft_mag", label="FFT magnitude", group="Spectrum", y_label="|FFT|",
    extract=_complex_component("fft_spectrum", "abs"),
    mask_key="snr_mask", error_key="fft_sigma", yscale="log", include_references=True,
    export_header="fft_mag",
))
register(Quantity(
    name="fft_phase", label="FFT phase", group="Spectrum", y_label="arg FFT (rad)",
    extract=lambda pd: (None if pd.get("fft_spectrum") is None or _freq(pd) is None
                        else (np.asarray(_freq(pd)), np.unwrap(np.angle(pd["fft_spectrum"])))),
    mask_key="snr_mask", error_key="fft_phase_sigma", include_references=True,
    export_header="fft_phase",
))
register(Quantity(
    name="transfer_mag", label="Transfer |H|", group="Transfer function", y_label="|H|",
    extract=_complex_component("transfer_H", "abs"),
    mask_key="transfer_mask", error_key="transfer_H_sigma", export_header="transfer_mag",
))
register(Quantity(
    name="transfer_phase", label="Transfer phase", group="Transfer function", y_label="arg H (rad)",
    extract=lambda pd: (None if pd.get("transfer_H") is None or _freq(pd) is None
                        else (np.asarray(_freq(pd)), np.unwrap(np.angle(pd["transfer_H"])))),
    mask_key="transfer_mask", error_key="transfer_phase_sigma", export_header="transfer_phase",
))
# error_key entries name the Monte-Carlo sigmas written by uncertainty.propagate_uncertainty.
register(Quantity(
    name="n", label="n (refractive index)", group="Optical constants", y_label="n",
    extract=_real_series("n"), error_key="n_sigma", export_header="n",
))
register(Quantity(
    name="k", label="k (extinction)", group="Optical constants", y_label="k",
    extract=_real_series("k"), error_key="k_sigma", export_header="k",
))
register(Quantity(
    name="eps_real", label="epsilon_1 (real permittivity)", group="Permittivity", y_label="eps_1",
    extract=_complex_component("eps", "real"), error_key="eps_real_sigma", export_header="eps_real",
))
register(Quantity(
    name="eps_imag", label="epsilon_2 (imag permittivity)", group="Permittivity", y_label="eps_2",
    extract=_complex_component("eps", "imag"), error_key="eps_imag_sigma", export_header="eps_imag",
))
register(Quantity(
    name="sigma_real", label="sigma_1 (real conductivity)", group="Conductivity",
    y_label="sigma_1 (S/m)", extract=_complex_component("sigma", "real"),
    error_key="sigma_real_sigma", export_header="sigma_real",
    overlay=_conductivity_fit_component("real"),
))
register(Quantity(
    name="sigma_imag", label="sigma_2 (imag conductivity)", group="Conductivity",
    y_label="sigma_2 (S/m)", extract=_complex_component("sigma", "imag"),
    error_key="sigma_imag_sigma", export_header="sigma_imag",
    overlay=_conductivity_fit_component("imag"),
))
