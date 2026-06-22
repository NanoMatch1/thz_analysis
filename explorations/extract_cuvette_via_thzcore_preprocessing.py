"""Cuvette n extraction through thz_core's OWN preprocessing (the correct front-end).

Context (ANALYSIS_NOTES §19, steps 2/2b/2c).  Validating the substrate-sandwich
inversion, both the MATLAB replication (asymmetric zero-padding) and the legacy-thz
front-end (`centerpad`) inflated the *substrate* index to ~2.2-2.5, when fused silica
should be ~1.96.  The raw time-domain peak delay (5.3 ps) already pointed to ~1.9; the
inflation was a preprocessing artifact (a hardcoded ~3 ps padding shift adds spurious
group delay).

This script runs the SAME 200 K triplet through thz_core's real transmission
preprocessing — each pulse windowed in place on the shared time axis (no asymmetric
padding), FFT, transfer function, analytic inversion — to confirm the front-end is the
difference.  Result: n_sub ~ 1.95 (fused silica), n_sample ~ 2.15.

Geometry note: the empty-cuvette substrate measurement crosses BOTH windows
(air|sub|gap|sub|air), so the effective single-slab path is 2 x the window thickness.

Run:  PYTHONPATH=. ../.venv/Scripts/python.exe explorations/extract_cuvette_via_thzcore_preprocessing.py
"""

from __future__ import annotations

import os

import numpy as np

import thz_core.thz_core as core

try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass


DATA_DIRECTORY = r"C:\Users\Samuel\Data\THz\Vasilis_Data\data"
AIR_REFERENCE_FILE = "reference_200K_tr.dat"
SUBSTRATE_FILE = "Substrate_200K_tr.dat"
SAMPLE_FILE = "Sample_200K_tr.dat"

WINDOW_THICKNESS_M = 0.9e-3                 # one fused-silica window
SUBSTRATE_TOTAL_PATH_M = 2 * WINDOW_THICKNESS_M   # two windows in the beam
SAMPLE_LAYER_THICKNESS_M = 0.13e-3

REPORT_FREQUENCY_MIN_HZ = 0.8e12
REPORT_FREQUENCY_MAX_HZ = 2.0e12

WINDOW_CONFIG = {"window": {"type": "tukey", "alpha": 1}}
TRANSFER_CONFIG = {
    "transfer": {"apply_snr_mask": False},
    "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
}
INVERT_CONFIG = {"invert": {"max_iterations": 20, "convergence_tol": 1e-10}}


def load_time_field_seconds(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a 2-column [time_ps, field] trace, returning time in SI seconds."""
    data = np.loadtxt(os.path.join(DATA_DIRECTORY, filename))
    return data[:, 0] * 1e-12, data[:, 1]


def subtract_leading_baseline(field: np.ndarray, leading_points: int = 10) -> np.ndarray:
    """Remove the DC offset estimated from the leading samples."""
    return field - np.mean(field[:leading_points])


def windowed_spectrum(
    time_seconds: np.ndarray, field: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Window the pulse in place (shared time axis) and FFT — thz_core front-end.

    Crucially, the pulse keeps its true position on the common time axis, so the
    transfer-function phase carries the real inter-pulse delay (no padding ramp).
    """
    windowed_field, _, _ = core.window_time(time_seconds, field, WINDOW_CONFIG)
    frequency, spectrum, _ = core.fft_spectrum(time_seconds, windowed_field, {})
    return frequency, spectrum


def analytic_index(
    frequency: np.ndarray,
    transmitted_spectrum: np.ndarray,
    reference_spectrum: np.ndarray,
    thickness_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transfer function + analytic (Duvillaret) free-standing inversion."""
    transfer, _, _ = core.transfer_function(
        frequency, transmitted_spectrum, reference_spectrum, TRANSFER_CONFIG
    )
    mask, _ = core.trusted_band_mask(
        frequency, reference_spectrum, transmitted_spectrum, transfer, TRANSFER_CONFIG
    )
    n, k, _ = core.invert_nk(frequency, transfer, thickness_m, mask, INVERT_CONFIG)
    return n, k


def report_band_statistics(label: str, frequency: np.ndarray, n: np.ndarray, k: np.ndarray) -> None:
    in_band = (frequency >= REPORT_FREQUENCY_MIN_HZ) & (frequency <= REPORT_FREQUENCY_MAX_HZ)
    n_band = n[in_band][np.isfinite(n[in_band])]
    k_band = k[in_band][np.isfinite(k[in_band])]
    print(
        f"  {label:<28} n median {np.median(n_band):.3f} "
        f"[{n_band.min():.3f}, {n_band.max():.3f}]   k median {np.median(k_band):.3f}"
    )


def main() -> None:
    time_seconds, air_field = load_time_field_seconds(AIR_REFERENCE_FILE)
    _, substrate_field = load_time_field_seconds(SUBSTRATE_FILE)
    _, sample_field = load_time_field_seconds(SAMPLE_FILE)

    air_field = subtract_leading_baseline(air_field)
    substrate_field = subtract_leading_baseline(substrate_field)
    sample_field = subtract_leading_baseline(sample_field)

    frequency, air_spectrum = windowed_spectrum(time_seconds, air_field)
    _, substrate_spectrum = windowed_spectrum(time_seconds, substrate_field)
    _, sample_spectrum = windowed_spectrum(time_seconds, sample_field)

    print("Cuvette extraction through thz_core preprocessing (200 K)")

    # Step 1: substrate vs air (two windows in the beam -> 2-slab path).
    n_sub, k_sub = analytic_index(
        frequency, substrate_spectrum, air_spectrum, SUBSTRATE_TOTAL_PATH_M
    )
    print("\n[Step 1] substrate (vs air, 2-window path):")
    report_band_statistics("fused silica expected ~1.96", frequency, n_sub, k_sub)

    # Step 2: sample vs substrate (the empty cuvette is the reference).
    n_sample, k_sample = analytic_index(
        frequency, sample_spectrum, substrate_spectrum, SAMPLE_LAYER_THICKNESS_M
    )
    print("\n[Step 2] sample (vs empty cuvette):")
    report_band_statistics("sample", frequency, n_sample, k_sample)

    print(
        "\nContrast: the same data through the MATLAB asymmetric padding / legacy "
        "centerpad gave n_sub ~ 2.2-2.5 (a preprocessing artifact).  thz_core's "
        "in-place windowing on the shared time axis recovers fused silica (~1.95)."
    )


if __name__ == "__main__":
    main()
