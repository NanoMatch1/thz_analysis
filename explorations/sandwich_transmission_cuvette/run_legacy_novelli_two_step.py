"""Two-step Novelli (legacy-thz) extraction on the 200 K triplet, for cross-comparison.

The group's legacy Python (`matchbook/legacy-thz`, M. Ballabio) implements the Novelli
analytic sandwich inversion (https://doi.org/10.1364/OE.510393) but only as a single
sample step that needs the substrate index supplied, wired to one ref/sample pair.

This script drives the SAME legacy helpers (no edits to `main_TDS.py`) in the two-step
workflow so it can be compared against thz_core and the MATLAB port
(`validate_sandwich_against_matlab.py`):

    Step 1 (substrate):  reference = air,        sample = substrate,  ns = 1
                         -> free-standing slab inversion -> n_sub(omega)
    Step 2 (sample):     reference = substrate,  sample = sample,     ns = n_sub(omega)
                         -> Novelli sandwich inversion -> n_sample(omega)

Why this is the right two-step reduction: the Novelli phase term is referenced to the
empty (air) gap in BOTH steps — n = 1 + Δφ·c/(ω·d) — so with ns = 1 the Fresnel factor
(n+ns)²/((1+ns)²·n) collapses to the free-standing (n+1)²/(4n), i.e. exactly the
substrate-in-air slab.  Only the amplitude/Fresnel factor uses ns; feeding n_sub(omega)
into step 2 is the sole coupling.

Fabry-Pérot is OFF: the Novelli method is the analytic, echo-free closed form.  That is
deliberate — it makes the legacy result an independent *analytic* baseline to compare
against the grid solvers (an iterative FP pass can be layered on later if wanted).

Run:  PYTHONPATH=. ../.venv/Scripts/python.exe explorations/run_legacy_novelli_two_step.py
(run from the repo root; legacy-thz is added to sys.path below).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import constants as physical_constants

# The legacy helpers use bare absolute imports (e.g. `from phase_interpolation import
# phaseex`), so their own directory must be on sys.path.
LEGACY_DIRECTORY = r"C:\Users\Samuel\matchbook\legacy-thz"
sys.path.insert(0, LEGACY_DIRECTORY)

import dataimport as legacy_dataimport          # noqa: E402
import fft_err as legacy_fft                     # noqa: E402
import padding as legacy_padding                 # noqa: E402
import phase_interpolation as legacy_phase       # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass


SPEED_OF_LIGHT_M_PER_S = physical_constants.c

DATA_DIRECTORY = r"C:\Users\Samuel\Data\THz\Vasilis_Data\data"
AIR_REFERENCE_FILE = "reference_200K_tr.dat"
SUBSTRATE_FILE = "Substrate_200K_tr.dat"
SAMPLE_FILE = "Sample_200K_tr.dat"

# Measured physical parameters (Samuel, 2026-06-19).
SUBSTRATE_SLAB_THICKNESS_M = 0.9e-3
SAMPLE_LAYER_THICKNESS_M = 0.13e-3

# The substrate measurement is the EMPTY CUVETTE (air | sub | gap | sub | air), so the
# beam crosses BOTH windows.  The free-standing step-1 inversion therefore uses the
# total material path = 2 slabs (the thin air gap adds a negligible n_air·gap phase).
SUBSTRATE_NUMBER_OF_SLABS = 2
SUBSTRATE_TOTAL_PATH_M = SUBSTRATE_NUMBER_OF_SLABS * SUBSTRATE_SLAB_THICKNESS_M

# Reporting band (matches the MATLAB / thz_core comparison).
REPORT_FREQUENCY_MIN_THZ = 0.8
REPORT_FREQUENCY_MAX_THZ = 2.0


def novelli_extract(
    reference_dataframe: pd.DataFrame,
    sample_dataframe: pd.DataFrame,
    thickness_m: float,
    substrate_index,
) -> pd.DataFrame:
    """Run one Novelli analytic extraction step using the legacy helpers.

    Replicates the active analysis path of ``main_TDS.py`` (centre-pad + Hann window →
    FFT with informed unwrap → phase-offset/phaseex → Novelli n,k), parameterised by
    thickness and the surrounding substrate index ``substrate_index`` (``ns``).

    Parameters
    ----------
    reference_dataframe, sample_dataframe : pd.DataFrame
        Time-domain traces from ``dataimport`` (columns Time (ps), Mean, std error).
    thickness_m : float
        Layer thickness for this step (substrate slab, then sample layer).
    substrate_index : float or np.ndarray
        ``ns`` in the Novelli Fresnel factor.  Scalar 1.0 for the substrate step;
        the complex/real n_sub(omega) array for the sample step.

    Returns
    -------
    pd.DataFrame
        Columns: Frequency (THz), n, k (full rfft frequency axis).
    """
    # centerpad mutates its input's Mean column (offset removal); copy to stay clean.
    reference_centered = legacy_padding.centerpad(reference_dataframe.copy())
    sample_centered = legacy_padding.centerpad(sample_dataframe.copy())

    reference_spectrum = legacy_fft.fft_err(reference_centered)
    sample_spectrum = legacy_fft.fft_err(sample_centered)

    frequency_thz = reference_spectrum["Frequency (THz)"]
    angular_frequency = 2.0 * np.pi * frequency_thz * 1e12

    phase_offset = legacy_phase.phaseoffset(reference_centered, sample_centered)
    phase_difference = legacy_phase.phaseex(reference_spectrum, sample_spectrum)

    amplitude_ratio = sample_spectrum["Amplitude"] / reference_spectrum["Amplitude"]

    # Novelli analytic inversion (sandwich; ns=1 reduces to free-standing slab).
    with np.errstate(divide="ignore", invalid="ignore"):
        n_real = 1.0 + (
            (phase_difference - phase_offset) * SPEED_OF_LIGHT_M_PER_S
        ) / (angular_frequency * thickness_m)

        fresnel_factor = (
            (n_real + substrate_index) ** 2
            / (1.0 + substrate_index) ** 2
            / n_real
        )
        k_real = -SPEED_OF_LIGHT_M_PER_S / (angular_frequency * thickness_m) * np.log(
            np.real(fresnel_factor) * amplitude_ratio
        )

    return pd.DataFrame(
        {
            "Frequency (THz)": frequency_thz.values,
            "n": np.asarray(n_real, dtype=float),
            "k": np.asarray(k_real, dtype=float),
        }
    )


def crop_to_report_band(result: pd.DataFrame) -> pd.DataFrame:
    """Restrict a result frame to the 0.8-2.0 THz reporting band."""
    in_band = result["Frequency (THz)"].between(
        REPORT_FREQUENCY_MIN_THZ, REPORT_FREQUENCY_MAX_THZ
    )
    return result.loc[in_band].reset_index(drop=True)


def main() -> None:
    air_reference = legacy_dataimport.dataimport(DATA_DIRECTORY, AIR_REFERENCE_FILE)
    substrate = legacy_dataimport.dataimport(DATA_DIRECTORY, SUBSTRATE_FILE)
    sample = legacy_dataimport.dataimport(DATA_DIRECTORY, SAMPLE_FILE)

    print("Legacy Novelli two-step extraction on the 200 K triplet")
    print(f"  data: {DATA_DIRECTORY}")

    # --- Step 1: substrate vs air (ns = 1 -> free-standing slab, 2-slab path) ---
    substrate_result = novelli_extract(
        air_reference, substrate, SUBSTRATE_TOTAL_PATH_M, substrate_index=1.0
    )
    substrate_band = crop_to_report_band(substrate_result)
    print("\n[Step 1] substrate (air ref, ns=1):")
    print(f"   n_sub median = {np.nanmedian(substrate_band['n']):.4f}, "
          f"k_sub median = {np.nanmedian(substrate_band['k']):.4f}, "
          f"n range [{np.nanmin(substrate_band['n']):.3f}, {np.nanmax(substrate_band['n']):.3f}]")

    # --- Step 2: sample vs substrate (ns = n_sub(omega) from step 1) ---
    # Both steps share one rfft frequency grid (identical padded length), so the step-1
    # n_sub array aligns bin-for-bin with step 2.  The real n_sub feeds the (real)
    # Novelli Fresnel factor; the substrate is low-loss so this is well justified.
    substrate_index_array = substrate_result["n"].values

    sample_result = novelli_extract(
        substrate, sample, SAMPLE_LAYER_THICKNESS_M,
        substrate_index=substrate_index_array,
    )
    if sample_result.shape[0] != substrate_result.shape[0]:
        raise ValueError(
            "Step 1 and step 2 frequency grids differ "
            f"({substrate_result.shape[0]} vs {sample_result.shape[0]}); the "
            "n_sub(omega) array cannot be fed in bin-for-bin."
        )
    sample_band = crop_to_report_band(sample_result)
    print("\n[Step 2] sample (substrate ref, ns=n_sub(omega)):")
    print(f"   n_sample median = {np.nanmedian(sample_band['n']):.4f}, "
          f"k_sample median = {np.nanmedian(sample_band['k']):.4f}, "
          f"n range [{np.nanmin(sample_band['n']):.3f}, {np.nanmax(sample_band['n']):.3f}]")

    _try_plot(substrate_band, sample_band)
    print("\nDone.  Compare against validate_sandwich_against_matlab.png "
          "(thz_core / MATLAB).  Novelli is analytic, no Fabry-Perot.")
    print(
        "\nCAVEAT (diagnosed 2026-06-21): the sample n above is NON-PHYSICAL because\n"
        "the legacy `centerpad`/`phaseoffset` TIME-DOMAIN preprocessing mishandles the\n"
        "inter-pulse delay on this thick-substrate cuvette data.  The Novelli FORMULA\n"
        "itself is correct: fed a properly preprocessed sample H (MATLAB-style padding)\n"
        "the same formula returns n ~ 2.15, matching thz_core/MATLAB.  So the legacy\n"
        "front-end — not the inversion — is the weak link here."
    )


def _try_plot(substrate_band: pd.DataFrame, sample_band: pd.DataFrame) -> None:
    """Save an n/k overlay for the substrate and sample steps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exception:  # pragma: no cover
        print(f"(plot skipped: {exception})")
        return

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    figure.suptitle("Legacy Novelli analytic two-step (no Fabry-Perot) — 200 K")

    axes[0, 0].plot(substrate_band["Frequency (THz)"], substrate_band["n"], "C0-")
    axes[0, 0].set_title("Substrate n"); axes[0, 0].set_ylabel("n")
    axes[1, 0].plot(substrate_band["Frequency (THz)"], substrate_band["k"], "C0-")
    axes[1, 0].set_title("Substrate k"); axes[1, 0].set_xlabel("Frequency (THz)"); axes[1, 0].set_ylabel("k")

    axes[0, 1].plot(sample_band["Frequency (THz)"], sample_band["n"], "C3-")
    axes[0, 1].set_title("Sample n"); axes[0, 1].set_ylabel("n")
    axes[1, 1].plot(sample_band["Frequency (THz)"], sample_band["k"], "C3-")
    axes[1, 1].set_title("Sample k"); axes[1, 1].set_xlabel("Frequency (THz)"); axes[1, 1].set_ylabel("k")

    output_path = os.path.join(os.path.dirname(__file__), "run_legacy_novelli_two_step.png")
    figure.savefig(output_path, dpi=130)
    print(f"\nSaved figure: {output_path}")


if __name__ == "__main__":
    main()
