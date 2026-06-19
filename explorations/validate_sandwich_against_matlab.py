"""Validate thz_core's substrate-sandwich inversion against the MATLAB reference.

Step 2 of the sandwich build plan (see docs/sandwich_extraction_explained.md and
ANALYSIS_NOTES.md §19).  It runs the full two-step workflow on the one-temperature
200 K triplet in ``C:/Users/Samuel/Data/THz/Vasilis_Data/data`` and checks that
``thz_core`` reproduces the MATLAB result.

What it does
------------
1. Faithfully replicates the MATLAB time-domain preprocessing (pad, peak-centred
   Hamming window, zero-padded FFT, frequency crop) so both solvers see the SAME
   measured transfer function H(omega) = E_trans(omega) / E_ref(omega).  This isolates
   the inversion (the part we changed) from the preprocessing.
2. Runs a direct Python PORT of each MATLAB grid solver — Ratio_calc evaluated with
   the MATLAB's exact Fresnel/Fabry-Pérot expressions (including their sign), the same
   grids, and the same ±N-cell continuity tracking.  This is the reference answer.
3. Runs thz_core's ``invert_nk_grid`` through the same H, with the new FP / medium /
   continuity options configured to match the MATLAB physics.
4. Compares, for both the substrate (ASASA) and sample (ASMSA) steps, with and without
   Fabry-Pérot, and prints median / max |Δn|, |Δk|.  Also an isolated algebraic check
   that the no-FP models are numerically identical.

The expected finding (documented in ANALYSIS_NOTES §19): the two MATLAB files use
INCONSISTENT FP signs.  The sample code's etalon is the textbook 1/(1 − r²P²) — which
thz_core matches.  The substrate code's etalon evaluates to 1/(1 + r0²P²) — the
opposite sign, with no physical resonance — which thz_core does NOT match (by design;
thz_core is internally consistent and physically correct).  This script quantifies
both.

Run:  ../.venv/Scripts/python.exe explorations/validate_sandwich_against_matlab.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# The script prints maths symbols (Δ, −, omega); force UTF-8 so a cp1252 Windows
# console does not choke on them.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

import thz_core.thz_core as core
from thz_core.thz_core.multilayer import asasa_transfer_function


# ── Fixed parameters (read straight from the MATLAB codes) ──────────────────

SPEED_OF_LIGHT_M_PER_S = 2.997925e8
AMBIENT_MEDIUM_INDEX = 1.00055           # n_air in both MATLAB files
FFT_LENGTH = 2 ** 12                     # NFourier
FREQUENCY_MIN_HZ = 0.8e12
FREQUENCY_MAX_HZ = 2.0e12
FREQUENCY_TARGET_COUNT = 200             # nfreq_target
HAMMING_WINDOW_DURATION_S = 4e-12        # 4 ps Hamming, NHam = round(4ps/dt)

# Physical parameters.  NOTE: updated from the MATLAB defaults to Samuel's measured
# values (2026-06-19): substrate n ≈ 1.95, slab 0.9 mm, inter-substrate gap 60–120 µm.
# The MATLAB shipped with d_sub = 1 mm and a substrate search window of 1.71–1.80 that
# EXCLUDED the true n ≈ 1.95 — the cause of the sawtooth (alias-hopping).  The gap is
# irrelevant to the substrate index in the gated (FP-off) limit (it cancels against the
# reference), so its 60–120 µm uncertainty does not affect the substrate result.
SUBSTRATE_SLAB_THICKNESS_M = 0.9e-3      # measured slab thickness (was MATLAB L2=L4=1 mm)
SUBSTRATE_GAP_THICKNESS_M = 0.09e-3      # 90 µm midpoint of 60–120 µm (FP-off: unused)
SAMPLE_LAYER_THICKNESS_M = 0.13e-3       # L3 in sample_code (sample/MOF layer)

# Substrate (ASASA) search grid — RE-CENTRED on the measured n ≈ 1.95 (was 1.71–1.80).
SUBSTRATE_N_MIN, SUBSTRATE_N_MAX, SUBSTRATE_N_COUNT = 1.88, 2.02, 500
SUBSTRATE_K_MIN_MATLAB, SUBSTRATE_K_MAX_MATLAB = 0.0, -0.03   # imag(n) sign
SUBSTRATE_K_COUNT = 500
SUBSTRATE_CONTINUITY_HALF_WIDTH = 20

# Sample (ASMSA) search grid — sample_code nmin/nmax/kmin/kmax.
SAMPLE_N_MIN, SAMPLE_N_MAX, SAMPLE_N_COUNT = 1.0, 2.5, 700
SAMPLE_K_MIN_MATLAB, SAMPLE_K_MAX_MATLAB = 0.0, -0.2
SAMPLE_K_COUNT = 500
SAMPLE_CONTINUITY_HALF_WIDTH = 15

DATA_DIRECTORY = r"C:\Users\Samuel\Data\THz\Vasilis_Data\data"
REFERENCE_AIR_FILE = "reference_200K_tr.dat"   # = Air_200K (the substrate-step ref)
SUBSTRATE_FILE = "Substrate_200K_tr.dat"
SAMPLE_FILE = "Sample_200K_tr.dat"


# ── Data loading and MATLAB-faithful preprocessing ──────────────────────────


def load_time_field_picoseconds(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a 2-column [time_ps, field] .dat trace."""
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def peak_centred_hamming(field: np.ndarray, samples_in_window: int) -> np.ndarray:
    """Replicate the MATLAB peak-centred Hamming window (clamped at array edges)."""
    window = np.zeros_like(field)
    peak_index = int(np.argmax(np.abs(field)))
    start = peak_index - int(np.ceil(samples_in_window / 2))
    stop = start + samples_in_window - 1
    start_clamped = max(0, start)
    stop_clamped = min(len(field) - 1, stop)
    hamming = np.hamming(samples_in_window)
    count = stop_clamped - start_clamped + 1
    window[start_clamped:stop_clamped + 1] = hamming[:count]
    return field * window


def matlab_transfer_function(
    reference_field: np.ndarray,
    transmitted_field: np.ndarray,
    timestep_s: float,
    reference_pad: tuple[int, int],
    transmitted_pad: tuple[int, int],
    remove_offset: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the MATLAB pad → window → FFT → crop chain, returning (freq, H).

    H(omega) = E_trans(omega) / E_ref(omega), the same quantity the MATLAB solver
    matches (its Ratio_meas = DeltaE/Eref = E_trans/E_ref − 1; the −1 is folded into
    Ratio_calc, so the matched transfer function is E_trans/E_ref).
    """
    reference = np.concatenate(
        [np.zeros(reference_pad[0]), reference_field, np.zeros(reference_pad[1])]
    )
    transmitted = np.concatenate(
        [np.zeros(transmitted_pad[0]), transmitted_field, np.zeros(transmitted_pad[1])]
    )

    if remove_offset:
        reference = reference - np.mean(reference[:5])
        transmitted = transmitted - np.mean(transmitted[:5])

    samples_in_window = round(HAMMING_WINDOW_DURATION_S / timestep_s)
    reference = peak_centred_hamming(reference, samples_in_window)
    transmitted = peak_centred_hamming(transmitted, samples_in_window)

    reference_spectrum = np.fft.fft(reference, FFT_LENGTH)
    transmitted_spectrum = np.fft.fft(transmitted, FFT_LENGTH)

    frequency_step = 1.0 / timestep_s / FFT_LENGTH
    frequency_full = (1.0 / timestep_s) * np.arange(FFT_LENGTH) / FFT_LENGTH

    index_min = max(1, int(np.floor(FREQUENCY_MIN_HZ / frequency_step)))
    index_max = int(np.floor(FREQUENCY_MAX_HZ / frequency_step))
    decimation = max(1, int(np.floor((index_max - index_min) / FREQUENCY_TARGET_COUNT)))

    # MATLAB 1-based inclusive slice Nfmin:Nfreqstep:Nfmax → 0-based here.
    selection = np.arange(index_min - 1, index_max, decimation)
    frequency = frequency_full[selection]
    transfer = transmitted_spectrum[selection] / reference_spectrum[selection]
    return frequency, transfer


# ── Direct Python port of the MATLAB grid solvers (the reference answer) ─────


def _matlab_complex_grid(
    n_min: float, n_max: float, n_count: int,
    k_min: float, k_max: float, k_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the MATLAB ns = n_axis + 1i*k_axis grid (k_axis can be negative)."""
    n_axis = np.linspace(n_min, n_max, n_count)
    k_axis = np.linspace(k_min, k_max, k_count)
    n_hat_grid = n_axis[:, None] + 1j * k_axis[None, :]
    return n_axis, k_axis, n_hat_grid


def _continuity_minimise(
    difference: np.ndarray,
    n_axis: np.ndarray,
    k_axis: np.ndarray,
    previous_value: complex | None,
    half_width: int,
) -> complex:
    """MATLAB branch-tracking: global min on the first bin, then a ±half-window box."""
    if previous_value is None:
        flat_index = int(np.argmin(difference))
        row, col = np.unravel_index(flat_index, difference.shape)
        return complex(n_axis[row] + 1j * k_axis[col])

    center_row = int(np.argmin(np.abs(n_axis - np.real(previous_value))))
    center_col = int(np.argmin(np.abs(k_axis - np.imag(previous_value))))
    row_lo, row_hi = max(0, center_row - half_width), min(len(n_axis), center_row + half_width + 1)
    col_lo, col_hi = max(0, center_col - half_width), min(len(k_axis), center_col + half_width + 1)
    local = difference[row_lo:row_hi, col_lo:col_hi]
    local_index = int(np.argmin(local))
    local_row, local_col = np.unravel_index(local_index, local.shape)
    return complex(n_axis[row_lo + local_row] + 1j * k_axis[col_lo + local_col])


def matlab_substrate_model(
    n_hat_grid: np.ndarray, frequency_hz: float, fabry_perot: bool = True
) -> np.ndarray:
    """Cuvette_code Ratio_calc + 1 = E_trans/E_ref model (note the substrate FP sign).

    n_hat_grid follows the MATLAB n + i*k convention with k ≤ 0; that is numerically
    identical to thz_core's n − i*k with k ≥ 0, so the same array feeds both.
    """
    n_air = AMBIENT_MEDIUM_INDEX
    n_sub = n_hat_grid
    omega = 2.0 * np.pi * frequency_hz
    c = SPEED_OF_LIGHT_M_PER_S

    t12 = 2 * n_air / (n_air + n_sub)
    t23 = 2 * n_sub / (n_sub + n_air)
    t34 = 2 * n_air / (n_air + n_sub)
    t45 = 2 * n_sub / (n_sub + n_air)
    r23 = (n_air - n_sub) / (n_sub + n_air)
    r34 = (n_sub - n_air) / (n_air + n_sub)

    phase2 = np.exp(-1j * n_sub * omega * SUBSTRATE_SLAB_THICKNESS_M / c)
    phase3 = np.exp(-1j * n_air * omega * SUBSTRATE_GAP_THICKNESS_M / c)
    phase4 = np.exp(-1j * n_sub * omega * SUBSTRATE_SLAB_THICKNESS_M / c)
    phase_ref = np.exp(
        -1j * n_air * omega
        * (2 * SUBSTRATE_SLAB_THICKNESS_M + SUBSTRATE_GAP_THICKNESS_M) / c
    )

    # MATLAB: (...) / ((1 - r23*r34*phase3^2)*phase_ref)   [substrate FP sign]
    etalon = (1 - r23 * r34 * phase3 ** 2) if fabry_perot else 1.0
    return (phase2 * phase3 * phase4 * t12 * t23 * t34 * t45) / (etalon * phase_ref)


def matlab_sample_model(
    n_hat_grid: np.ndarray, n_substrate: complex, frequency_hz: float,
    fabry_perot: bool = True,
) -> np.ndarray:
    """sample_code Ratio_calc + 1 = E_trans/E_ref model (textbook FP sign)."""
    n_air = AMBIENT_MEDIUM_INDEX
    n_sub = n_substrate
    n_samp = n_hat_grid
    omega = 2.0 * np.pi * frequency_hz
    c = SPEED_OF_LIGHT_M_PER_S

    t23 = 2 * n_sub / (n_sub + n_samp)
    t34 = 2 * n_samp / (n_samp + n_sub)
    t23ref = 2 * n_sub / (n_sub + n_air)
    t34ref = 2 * n_air / (n_air + n_sub)
    r23 = (n_samp - n_sub) / (n_sub + n_samp)
    r34 = (n_sub - n_samp) / (n_samp + n_sub)
    r23ref = (n_air - n_sub) / (n_sub + n_air)
    r34ref = (n_sub - n_air) / (n_air + n_sub)

    phase3 = np.exp(-1j * n_samp * omega * SAMPLE_LAYER_THICKNESS_M / c)
    phase3ref = np.exp(-1j * n_air * omega * SAMPLE_LAYER_THICKNESS_M / c)

    sample_etalon = (1 + r23 * r34 * phase3 ** 2) if fabry_perot else 1.0
    reference_etalon = (1 + r23ref * r34ref * phase3ref ** 2) if fabry_perot else 1.0
    numerator = phase3 * t23 * t34 / sample_etalon
    denominator = phase3ref * t23ref * t34ref / reference_etalon
    return numerator / denominator


def run_matlab_port_solver(
    frequency: np.ndarray,
    transfer: np.ndarray,
    model_function,
    grid_spec: tuple,
    half_width: int,
) -> np.ndarray:
    """Generic grid + continuity solver matching the MATLAB minimisation exactly."""
    n_axis, k_axis, n_hat_grid = _matlab_complex_grid(*grid_spec)
    result = np.zeros(frequency.size, dtype=complex)
    previous = None
    for index, frequency_hz in enumerate(frequency):
        model = model_function(n_hat_grid, float(frequency_hz))
        measured = transfer[index]
        difference = np.abs(np.real(model - measured)) + np.abs(np.imag(model - measured))
        previous = _continuity_minimise(difference, n_axis, k_axis, previous, half_width)
        result[index] = previous
    return result


# ── thz_core solver via the public invert_nk_grid ───────────────────────────


def run_thzcore_substrate(
    frequency: np.ndarray, transfer: np.ndarray, fabry_perot: bool, grid_step: float
) -> np.ndarray:
    """thz_core substrate_only inversion (returns MATLAB-convention n + i*k)."""
    mask = np.ones(frequency.size, dtype=bool)
    config = {
        "invert_grid": {
            "geometry": "substrate_only",
            "thickness_sub_m": SUBSTRATE_SLAB_THICKNESS_M,
            "thickness_gap_m": SUBSTRATE_GAP_THICKNESS_M,
            "medium_index": AMBIENT_MEDIUM_INDEX,
            "fabry_perot": fabry_perot,
            "n_real_bounds": [SUBSTRATE_N_MIN, SUBSTRATE_N_MAX],
            "k_bounds": [0.0, abs(SUBSTRATE_K_MAX_MATLAB)],
            "grid_step": grid_step,
            "continuity_tracking": {
                "enabled": True,
                "search_half_width_cells": SUBSTRATE_CONTINUITY_HALF_WIDTH,
            },
        }
    }
    n, k, _ = core.invert_nk_grid(frequency, transfer, mask, config)
    return n + 1j * (-k)   # thz_core reports k ≥ 0 as -Im; back to MATLAB convention


def run_thzcore_sample(
    frequency: np.ndarray,
    transfer: np.ndarray,
    n_substrate: np.ndarray,
    fabry_perot: bool,
    grid_step: float,
) -> np.ndarray:
    """thz_core substrate_sandwich inversion (returns MATLAB-convention n + i*k)."""
    mask = np.ones(frequency.size, dtype=bool)
    config = {
        "invert_grid": {
            "geometry": "substrate_sandwich",
            "thickness_sample_m": SAMPLE_LAYER_THICKNESS_M,
            "thickness_ref_gap_m": SAMPLE_LAYER_THICKNESS_M,
            "n_substrate": np.real(n_substrate) - 1j * np.imag(n_substrate),
            "medium_index": AMBIENT_MEDIUM_INDEX,
            "fabry_perot": fabry_perot,
            "n_real_bounds": [SAMPLE_N_MIN, SAMPLE_N_MAX],
            "k_bounds": [0.0, abs(SAMPLE_K_MAX_MATLAB)],
            "grid_step": grid_step,
            "continuity_tracking": {
                "enabled": True,
                "search_half_width_cells": SAMPLE_CONTINUITY_HALF_WIDTH,
            },
        }
    }
    n, k, _ = core.invert_nk_grid(frequency, transfer, mask, config)
    return n + 1j * (-k)


# ── Reporting ────────────────────────────────────────────────────────────────


def summarise_difference(label: str, reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Print and return median / max |Δn| and |Δk| between two complex n(omega)."""
    delta_n = np.abs(np.real(candidate) - np.real(reference))
    delta_k = np.abs(np.imag(candidate) - np.imag(reference))
    stats = {
        "median_dn": float(np.median(delta_n)),
        "max_dn": float(np.max(delta_n)),
        "median_dk": float(np.median(delta_k)),
        "max_dk": float(np.max(delta_k)),
    }
    print(
        f"  {label:<42}  |Δn| med {stats['median_dn']:.2e} max {stats['max_dn']:.2e}"
        f"   |Δk| med {stats['median_dk']:.2e} max {stats['max_dk']:.2e}"
    )
    return stats


def check_models_identical_without_fabry_perot(frequency: np.ndarray) -> None:
    """Algebraic check: the no-FP substrate models are numerically identical.

    Confirms thz_core's asasa_transfer_function equals the MATLAB substrate model in
    the gated limit, independent of any data — the cleanest equivalence proof.
    """
    _, _, n_hat_grid = _matlab_complex_grid(
        SUBSTRATE_N_MIN, SUBSTRATE_N_MAX, 40,
        SUBSTRATE_K_MIN_MATLAB, SUBSTRATE_K_MAX_MATLAB, 30,
    )
    # MATLAB n + i*k (k ≤ 0) is numerically the same array as thz_core n − i*k (k ≥ 0).
    worst = 0.0
    for frequency_hz in frequency[:: max(1, frequency.size // 10)]:
        omega = 2.0 * np.pi * float(frequency_hz)
        matlab_h = matlab_substrate_model(n_hat_grid, float(frequency_hz), fabry_perot=False)
        thzcore_h = asasa_transfer_function(
            n_hat_grid, omega, SUBSTRATE_SLAB_THICKNESS_M,
            medium_index=AMBIENT_MEDIUM_INDEX, fabry_perot=False,
        )
        worst = max(worst, float(np.max(np.abs(matlab_h - thzcore_h))))
    print(f"\n[model equivalence, no FP] max |H_matlab - H_thzcore| over grid = {worst:.3e}")
    # The tiny residual is the speed-of-light constant difference (MATLAB 2.997925e8
    # vs thz_core 299792458.0, ~1.4e-7 relative) acting on the ~150 rad substrate
    # phase — NOT a model difference. The two no-FP models are otherwise identical.
    assert worst < 1e-4, "no-FP substrate models should match up to the c constant"
    print("  PASS: thz_core asasa == MATLAB substrate model in the gated limit "
          "(residual = speed-of-light constant only).")


def main() -> None:
    reference_path = os.path.join(DATA_DIRECTORY, REFERENCE_AIR_FILE)
    substrate_path = os.path.join(DATA_DIRECTORY, SUBSTRATE_FILE)
    sample_path = os.path.join(DATA_DIRECTORY, SAMPLE_FILE)

    time_reference, field_reference = load_time_field_picoseconds(reference_path)
    _, field_substrate = load_time_field_picoseconds(substrate_path)
    _, field_sample = load_time_field_picoseconds(sample_path)
    timestep_s = float(np.mean(np.diff(time_reference))) * 1e-12

    print(f"Loaded 200 K triplet from {DATA_DIRECTORY}")
    print(f"  samples = {field_reference.size}, dt = {timestep_s * 1e12:.4f} ps")

    # --- Substrate step (ASASA): reference = air, transmitted = substrate ---
    substrate_frequency, substrate_transfer = matlab_transfer_function(
        field_reference, field_substrate, timestep_s,
        reference_pad=(150, 210), transmitted_pad=(210, 150), remove_offset=False,
    )
    substrate_grid_spec = (
        SUBSTRATE_N_MIN, SUBSTRATE_N_MAX, SUBSTRATE_N_COUNT,
        SUBSTRATE_K_MIN_MATLAB, SUBSTRATE_K_MAX_MATLAB, SUBSTRATE_K_COUNT,
    )

    check_models_identical_without_fabry_perot(substrate_frequency)

    print("\n=== Substrate step (ASASA), n(omega) over 0.8-2.0 THz ===")
    matlab_substrate_no_fp = run_matlab_port_solver(
        substrate_frequency, substrate_transfer,
        lambda grid, f: matlab_substrate_model(grid, f, fabry_perot=False),
        substrate_grid_spec, SUBSTRATE_CONTINUITY_HALF_WIDTH,
    )
    matlab_substrate_fp = run_matlab_port_solver(
        substrate_frequency, substrate_transfer,
        lambda grid, f: matlab_substrate_model(grid, f, fabry_perot=True),  # as-shipped sign
        substrate_grid_spec, SUBSTRATE_CONTINUITY_HALF_WIDTH,
    )
    # thz_core grid_step: fine enough to resolve the narrow substrate band.
    substrate_step = 0.0004
    thzcore_substrate_no_fp = run_thzcore_substrate(
        substrate_frequency, substrate_transfer, fabry_perot=False, grid_step=substrate_step
    )
    thzcore_substrate_fp = run_thzcore_substrate(
        substrate_frequency, substrate_transfer, fabry_perot=True, grid_step=substrate_step
    )

    print(f"   n_sub median (MATLAB FP off) = {np.median(np.real(matlab_substrate_no_fp)):.4f}, "
          f"k = {np.median(-np.imag(matlab_substrate_no_fp)):.4f}")
    print(" Gated-limit equivalence (should AGREE to grid resolution):")
    summarise_difference("substrate FP OFF: thz_core vs MATLAB", matlab_substrate_no_fp, thzcore_substrate_no_fp)
    print(" Fabry-Perot sign discrepancy (substrate code; EXPECTED to differ):")
    summarise_difference("substrate FP ON : thz_core(textbook) vs MATLAB(as-shipped)", matlab_substrate_fp, thzcore_substrate_fp)
    print(" Size of the FP correction itself (thz_core on vs off):")
    summarise_difference("substrate FP on vs off (thz_core)", thzcore_substrate_no_fp, thzcore_substrate_fp)

    # Use the thz_core FP-OFF substrate index to feed the sample step: these 1 mm slabs
    # have well-separated echoes (gatable), so the gated substrate index is the
    # physically appropriate input and avoids the substrate FP-sign question.
    n_substrate_for_sample = thzcore_substrate_no_fp

    # --- Sample step (ASMSA): reference = substrate, transmitted = sample ---
    sample_frequency, sample_transfer = matlab_transfer_function(
        field_substrate, field_sample, timestep_s,
        reference_pad=(150, 150), transmitted_pad=(150, 150), remove_offset=True,
    )
    sample_grid_spec = (
        SAMPLE_N_MIN, SAMPLE_N_MAX, SAMPLE_N_COUNT,
        SAMPLE_K_MIN_MATLAB, SAMPLE_K_MAX_MATLAB, SAMPLE_K_COUNT,
    )

    # The MATLAB port and thz_core must use the SAME substrate index to be comparable;
    # feed both the thz_core FP-on n_sub (resampled implicitly — same frequency grid).
    print("\n=== Sample step (ASMSA), n(omega) over 0.8–2.0 THz ===")
    matlab_sample_fp = run_matlab_port_solver(
        sample_frequency, sample_transfer,
        lambda grid, f, ns=n_substrate_for_sample, freq=sample_frequency:
            matlab_sample_model(grid, complex(ns[int(np.argmin(np.abs(freq - f)))]), f),
        sample_grid_spec, SAMPLE_CONTINUITY_HALF_WIDTH,
    )
    sample_step = 0.002
    thzcore_sample_fp = run_thzcore_sample(
        sample_frequency, sample_transfer, n_substrate_for_sample,
        fabry_perot=True, grid_step=sample_step,
    )
    thzcore_sample_no_fp = run_thzcore_sample(
        sample_frequency, sample_transfer, n_substrate_for_sample,
        fabry_perot=False, grid_step=sample_step,
    )

    print(f"   sample n median (thz_core, FP on) = {np.median(np.real(thzcore_sample_fp)):.4f}")
    print(" thz_core vs MATLAB-port (same n_sub, textbook FP sign — expected to AGREE):")
    summarise_difference("sample   FP ON  vs MATLAB", matlab_sample_fp, thzcore_sample_fp)
    summarise_difference("sample   FP OFF vs MATLAB(FP on)", matlab_sample_fp, thzcore_sample_no_fp)

    _try_plot(
        substrate_frequency, matlab_substrate_no_fp, thzcore_substrate_no_fp, thzcore_substrate_fp,
        sample_frequency, matlab_sample_fp, thzcore_sample_fp,
    )

    print("\nDone.  See ANALYSIS_NOTES.md §19 for interpretation of the substrate FP-sign gap.")


def _try_plot(
    substrate_frequency, matlab_substrate, thzcore_substrate_no_fp, thzcore_substrate_fp,
    sample_frequency, matlab_sample, thzcore_sample_fp,
) -> None:
    """Save an overlay figure; skip silently if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exception:  # pragma: no cover
        print(f"(plot skipped: {exception})")
        return

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    f_sub = substrate_frequency / 1e12
    axes[0, 0].plot(f_sub, np.real(matlab_substrate), "k-", label="MATLAB port")
    axes[0, 0].plot(f_sub, np.real(thzcore_substrate_no_fp), "C0--", label="thz_core FP off")
    axes[0, 0].plot(f_sub, np.real(thzcore_substrate_fp), "C3:", label="thz_core FP on")
    axes[0, 0].set_title("Substrate n")
    axes[0, 0].set_ylabel("n"); axes[0, 0].legend(fontsize=7)
    axes[1, 0].plot(f_sub, -np.imag(matlab_substrate), "k-")
    axes[1, 0].plot(f_sub, -np.imag(thzcore_substrate_no_fp), "C0--")
    axes[1, 0].plot(f_sub, -np.imag(thzcore_substrate_fp), "C3:")
    axes[1, 0].set_title("Substrate k"); axes[1, 0].set_xlabel("Frequency (THz)"); axes[1, 0].set_ylabel("k")

    f_samp = sample_frequency / 1e12
    axes[0, 1].plot(f_samp, np.real(matlab_sample), "k-", label="MATLAB port")
    axes[0, 1].plot(f_samp, np.real(thzcore_sample_fp), "C3:", label="thz_core FP on")
    axes[0, 1].set_title("Sample n"); axes[0, 1].set_ylabel("n"); axes[0, 1].legend(fontsize=7)
    axes[1, 1].plot(f_samp, -np.imag(matlab_sample), "k-")
    axes[1, 1].plot(f_samp, -np.imag(thzcore_sample_fp), "C3:")
    axes[1, 1].set_title("Sample k"); axes[1, 1].set_xlabel("Frequency (THz)"); axes[1, 1].set_ylabel("k")

    output_path = os.path.join(os.path.dirname(__file__), "validate_sandwich_against_matlab.png")
    figure.savefig(output_path, dpi=130)
    print(f"\nSaved overlay figure: {output_path}")


if __name__ == "__main__":
    main()
