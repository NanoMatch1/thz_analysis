"""Advisory metrics utilities for THz-TDS processing stages."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import numpy as np


def _snapshot_config(config: dict) -> dict:
    """Return a deep-copied config snapshot for traceability."""
    return deepcopy(config)


def _safe_float(value: float | np.ndarray) -> float:
    """Convert scalar-like values to a plain float safely."""
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).item())
    return float(value)


def compute_metrics(
    stage: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    config: dict,
) -> dict:
    """Create a standardized advisory metrics dictionary for a stage.

    Parameters
    ----------
    stage : str
        Stage name such as ``alignment`` or ``transfer``.
    inputs : Dict[str, Any]
        Input data summary for this stage.
    outputs : Dict[str, Any]
        Output data summary and pre-computed metrics from this stage.
    config : dict
        Plain configuration dictionary.

    Returns
    -------
    dict
        Standardized metrics payload with values, flags, warnings, and a
        config snapshot.
    """
    warnings: list[str] = []
    flags: dict[str, bool] = {}
    values: dict[str, Any] = {}

    values.update(outputs.get("values", {}))

    if stage == "alignment":
        peak = outputs.get("corr_peak", np.nan)
        sharpness = outputs.get("corr_sharpness", np.nan)
        values["corr_peak"] = _safe_float(peak)
        values["corr_sharpness"] = _safe_float(sharpness)
        flags["low_alignment_confidence"] = bool(
            np.isfinite(peak) and peak < 0.3
        )
        if flags["low_alignment_confidence"]:
            warnings.append("Low cross-correlation peak during alignment.")

    if stage == "window":
        retention = outputs.get("energy_retention", np.nan)
        values["energy_retention"] = _safe_float(retention)
        flags["low_energy_retention"] = bool(
            np.isfinite(retention) and retention < 0.2
        )
        if flags["low_energy_retention"]:
            warnings.append("Window retained less than 20% signal energy.")

    if stage == "fft":
        snr_db = outputs.get("snr_db", np.nan)
        values["time_domain_snr_db"] = _safe_float(snr_db)
        flags["low_time_snr"] = bool(np.isfinite(snr_db) and snr_db < 10.0)
        if flags["low_time_snr"]:
            warnings.append("Low estimated time-domain SNR.")

    if stage == "transfer":
        phase_score = outputs.get("phase_continuity_score", np.nan)
        invalid_fraction = outputs.get("invalid_ref_fraction", np.nan)
        values["phase_continuity_score"] = _safe_float(phase_score)
        values["invalid_ref_fraction"] = _safe_float(invalid_fraction)
        flags["phase_discontinuity"] = bool(
            np.isfinite(phase_score) and phase_score > 1.5
        )
        flags["excess_invalid_ref"] = bool(
            np.isfinite(invalid_fraction) and invalid_fraction > 0.2
        )
        if flags["phase_discontinuity"]:
            warnings.append("Large phase derivative variation in transfer data.")
        if flags["excess_invalid_ref"]:
            warnings.append("Reference spectrum has many low-amplitude bins.")

    if stage == "mask":
        coverage = outputs.get("trusted_coverage_fraction", np.nan)
        values["trusted_coverage_fraction"] = _safe_float(coverage)
        flags["small_trusted_band"] = bool(
            np.isfinite(coverage) and coverage < 0.1
        )
        if flags["small_trusted_band"]:
            warnings.append("Trusted band is very narrow.")

    return {
        "stage": stage,
        "values": values,
        "flags": flags,
        "warnings": warnings,
        "config_snapshot": _snapshot_config(config),
        "inputs_summary": {
            "keys": sorted(inputs.keys()),
        },
    }
