"""Transfer function and trusted-band masking utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .metrics import compute_metrics


def _validate_1d_complex(name: str, value: np.ndarray) -> np.ndarray:
    """Validate and return a 1D complex array."""
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if array.size < 4:
        raise ValueError(f"{name} must have at least 4 elements.")
    return array.astype(np.complex128)


def _phase_continuity_score(unwrapped_phase: np.ndarray) -> float:
    """Compute an advisory phase continuity score (lower is smoother)."""
    if unwrapped_phase.size < 4:
        return np.nan
    dphi = np.diff(unwrapped_phase)
    mad = np.median(np.abs(dphi - np.median(dphi)))
    scale = np.median(np.abs(dphi)) + 1e-15
    return float(mad / scale)


def _noise_floor(magnitude: np.ndarray, tail_fraction: float) -> float:
    """Estimate a robust noise floor from the high-index tail of spectrum."""
    n_points = magnitude.size
    tail_points = max(8, int(np.ceil(n_points * tail_fraction)))
    tail = magnitude[-tail_points:]
    floor = float(np.median(np.abs(tail)))
    return max(floor, 1e-30)


def _remove_short_segments(mask: np.ndarray, min_bins: int) -> np.ndarray:
    """Remove true segments shorter than min_bins."""
    cleaned = np.zeros_like(mask, dtype=bool)
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if not value and start is not None:
            if index - start >= min_bins:
                cleaned[start:index] = True
            start = None
    if start is not None and mask.size - start >= min_bins:
        cleaned[start:] = True
    return cleaned


def transfer_function(
    f: np.ndarray,
    Y_samp: np.ndarray,
    Y_ref: np.ndarray,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Compute transmission transfer function and initial validity mask.

    Parameters
    ----------
    f : np.ndarray
        Frequency axis in Hz.
    Y_samp, Y_ref : np.ndarray
        Sample and reference complex spectra.
    config : dict
        Configuration dictionary with optional ``transfer`` keys:
        ``min_ref_amp_rel``, ``regularization_eps``, and ``unwrap_phase``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        Complex transfer function ``H``, initial validity mask, and metrics.
    """
    frequency = np.asarray(f, dtype=float)
    sample_spec = _validate_1d_complex("Y_samp", Y_samp)
    reference_spec = _validate_1d_complex("Y_ref", Y_ref)
    if frequency.ndim != 1:
        raise ValueError("f must be a 1D array.")
    if not (frequency.size == sample_spec.size == reference_spec.size):
        raise ValueError("f, Y_samp, and Y_ref must have same length.")

    transfer_cfg = config.get("transfer", {})
    min_ref_amp_rel = float(transfer_cfg.get("min_ref_amp_rel", 1e-4))
    if min_ref_amp_rel <= 0:
        raise ValueError("transfer.min_ref_amp_rel must be positive.")
    regularization_eps = float(transfer_cfg.get("regularization_eps", 1e-30))
    unwrap_phase = bool(transfer_cfg.get("unwrap_phase", True))

    ref_mag = np.abs(reference_spec)
    ref_peak = float(np.max(ref_mag))
    ref_floor = max(ref_peak * min_ref_amp_rel, regularization_eps)
    valid_ref = ref_mag >= ref_floor

    transfer = np.full_like(sample_spec, np.nan + 1j * np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        transfer[valid_ref] = sample_spec[valid_ref] / reference_spec[valid_ref]

    if unwrap_phase:
        valid_indices = np.where(np.isfinite(transfer))[0]
        if valid_indices.size > 1:
            phase = np.angle(transfer[valid_indices])
            phase_unwrapped = np.unwrap(phase)
            magnitude = np.abs(transfer[valid_indices])
            transfer[valid_indices] = magnitude * np.exp(1j * phase_unwrapped)

    finite_mask = np.isfinite(np.real(transfer)) & np.isfinite(np.imag(transfer))
    phase_continuity = np.nan
    if np.count_nonzero(finite_mask) > 4:
        phase_continuity = _phase_continuity_score(np.angle(transfer[finite_mask]))

    invalid_ref_fraction = 1.0 - float(np.mean(valid_ref))

    metrics = compute_metrics(
        stage="transfer",
        inputs={"size": frequency.size},
        outputs={
            "values": {
                "min_ref_amp_rel": min_ref_amp_rel,
                "regularization_eps": regularization_eps,
            },
            "phase_continuity_score": phase_continuity,
            "invalid_ref_fraction": invalid_ref_fraction,
        },
        config=config,
    )
    return transfer, finite_mask, metrics


def trusted_band_mask(
    f: np.ndarray,
    Y_ref: np.ndarray,
    Y_samp: np.ndarray,
    H: np.ndarray,
    config: dict,
) -> Tuple[np.ndarray, dict]:
    """Build advisory trusted-band mask from DR/SNR style thresholds.

    Parameters
    ----------
    f : np.ndarray
        Frequency axis in Hz.
    Y_ref, Y_samp : np.ndarray
        Reference and sample complex spectra.
    H : np.ndarray
        Transfer function array.
    config : dict
        Configuration dictionary with optional ``mask`` keys:
        ``snr_thresh_db``, ``tail_fraction``, ``min_contiguous_bins``.

    Returns
    -------
    tuple[np.ndarray, dict]
        Boolean trusted mask and stage metrics.
    """
    frequency = np.asarray(f, dtype=float)
    reference_spec = _validate_1d_complex("Y_ref", Y_ref)
    sample_spec = _validate_1d_complex("Y_samp", Y_samp)
    transfer = _validate_1d_complex("H", H)
    if not (frequency.size == reference_spec.size == sample_spec.size == transfer.size):
        raise ValueError("f, Y_ref, Y_samp, and H must have same length.")

    mask_cfg = config.get("mask", {})
    snr_thresh_db = float(mask_cfg.get("snr_thresh_db", 20.0))
    tail_fraction = float(mask_cfg.get("tail_fraction", 0.2))
    min_contiguous_bins = int(mask_cfg.get("min_contiguous_bins", 8))

    if not (0.0 < tail_fraction < 1.0):
        raise ValueError("mask.tail_fraction must be between 0 and 1.")
    if min_contiguous_bins < 1:
        raise ValueError("mask.min_contiguous_bins must be >= 1.")

    ref_mag = np.abs(reference_spec)
    samp_mag = np.abs(sample_spec)

    ref_floor = _noise_floor(ref_mag, tail_fraction)
    samp_floor = _noise_floor(samp_mag, tail_fraction)

    ref_snr_db = 20.0 * np.log10(np.maximum(ref_mag, ref_floor) / ref_floor)
    samp_snr_db = 20.0 * np.log10(np.maximum(samp_mag, samp_floor) / samp_floor)

    finite_h = np.isfinite(np.real(transfer)) & np.isfinite(np.imag(transfer))
    base_mask = (ref_snr_db >= snr_thresh_db) & (samp_snr_db >= snr_thresh_db) & finite_h

    clean_mask = _remove_short_segments(base_mask, min_contiguous_bins)
    coverage = float(np.mean(clean_mask))

    metrics = compute_metrics(
        stage="mask",
        inputs={"size": frequency.size},
        outputs={
            "values": {
                "snr_thresh_db": snr_thresh_db,
                "ref_noise_floor": ref_floor,
                "samp_noise_floor": samp_floor,
            },
            "trusted_coverage_fraction": coverage,
        },
        config=config,
    )
    metrics["values"]["ref_snr_db"] = ref_snr_db
    metrics["values"]["samp_snr_db"] = samp_snr_db
    return clean_mask, metrics
