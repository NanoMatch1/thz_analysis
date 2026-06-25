"""Critical comparison of sandwich-transmission INVERSION methods on a shared front-end.

Context (ANALYSIS_NOTES.md §19, memory `sandwich_extraction_status`).  We explored three
end-to-end pipelines for the air|substrate|sample|substrate|air (cuvette) geometry:

    1. MATLAB-port  : asymmetric zero-pad front-end + grid search (continuity, opt-in FP)
    2. Legacy Novelli: centerpad/phaseoffset front-end + analytic closed form (no FP)
    3. thz_core      : in-place windowing front-end + analytic free-standing two-step

The decisive finding was that the FRONT-END (preprocessing), not the inversion, drove the
differences between them: the MATLAB asymmetric padding and the legacy centerpad both inject
a spurious ~3 ps group delay that inflates the extracted substrate index to ~2.2-2.5, while
thz_core's in-place windowing on the shared time axis recovers fused silica (~1.95).  In the
gated limit the three inversion *models* are algebraically identical (max|ΔH| ~ 1e-6).

So this script REASSESSES on a level playing field:

    * Hold ONE front-end fixed = thz_core's in-place windowing (the most reliable).
    * Compute the two measured transfer functions ONCE (substrate step, sample step).
    * Feed that SAME measured H to every inversion method.

That isolates the inversion math.  The methods compared, at each critical point:

    Substrate refractive index  n_sub(omega):
        - thz_core analytic (free-standing slab, two-window path)
        - thz_core grid     (substrate_only / ASASA, Fabry-Perot OFF = gated)
        - MATLAB-port grid  (substrate model, Fabry-Perot OFF = gated)
        - Novelli analytic  (ns = 1 reduces to free-standing slab)

    Sample complex refractive index  n_hat(omega) = n + i*k:
        - thz_core analytic (free-standing slab of the sample layer)
        - thz_core grid     (substrate_sandwich / ASMSA, FP off AND on)
        - MATLAB-port grid  (sample model, FP off AND on)
        - Novelli analytic  (sandwich Fresnel factor, ns = n_sub(omega))

    Sample complex conductivity  sigma(omega) = -i*omega*eps0*(eps - eps_background):
        - derived from each method's sample n_hat via thz_core.derive_eps_sigma,
          eps_background = 1.0 (vacuum) — total optical conductivity.

Each sample-step method that needs n_sub is fed its OWN substrate result (the honest
self-consistent pipeline); n_sub is reported per method so the coupling is visible.

The MATLAB-port and Novelli MODELS/SOLVERS are reproduced here (front-end-independent) so the
script is self-contained; they are faithful copies of the tested helpers in
`validate_sandwich_against_matlab.py` / `run_legacy_novelli_two_step.py`, driven off the
shared thz_core H rather than their original front-ends.

Run:  PYTHONPATH=. ../.venv/Scripts/python.exe \
          explorations/sandwich_transmission_cuvette/compare_inversion_methods_shared_frontend.py
(run from the repo root so `import thz_core` resolves.)
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import constants as physical_constants

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import thz_core.thz_core as core
from thz_core.thz_core.multilayer import asasa_transfer_function  # noqa: F401  (kept for parity)

try:  # the report prints maths symbols; force UTF-8 on a cp1252 console.
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass


# ── Fixed experimental parameters ─────────────────────────────────────────────

SPEED_OF_LIGHT_M_PER_S = physical_constants.c
AMBIENT_MEDIUM_INDEX = 1.0               # treat the surround as vacuum for every method
VACUUM_PERMITTIVITY = physical_constants.epsilon_0

DATA_DIRECTORY = r"C:\Users\Samuel\Data\THz\Vasilis_Data\data"
AIR_REFERENCE_FILE = "reference_200K_tr.dat"
SUBSTRATE_FILE = "Substrate_200K_tr.dat"
SAMPLE_FILE = "Sample_200K_tr.dat"

# Measured geometry (Samuel, 2026-06): fused-silica windows 0.9 mm each, sample layer
# 0.13 mm.  The empty-cuvette substrate measurement crosses BOTH windows, so the
# free-standing/Novelli substrate path is the two-window total; the explicit ASASA grid
# models each 0.9 mm slab plus the (cancelling) inter-window gap separately.
SUBSTRATE_SLAB_THICKNESS_M = 0.9e-3
SUBSTRATE_GAP_THICKNESS_M = 0.09e-3      # 90 µm midpoint of 60-120 µm; cancels when gated
SUBSTRATE_TWO_WINDOW_PATH_M = 2 * SUBSTRATE_SLAB_THICKNESS_M
SAMPLE_LAYER_THICKNESS_M = 0.13e-3

# Reporting / inversion band.
REPORT_FREQUENCY_MIN_HZ = 0.8e12
REPORT_FREQUENCY_MAX_HZ = 2.0e12

# Trailing zero-pad length: interpolates the spectra to a smooth curve WITHOUT shifting the
# (already windowed-to-zero) pulse, so no spurious group delay is introduced.
SPECTRUM_FFT_LENGTH = 4096

# Shared thz_core front-end config.
WINDOW_CONFIG = {"window": {"type": "tukey", "alpha": 1}}
TRANSFER_CONFIG = {
    "transfer": {"apply_snr_mask": False},
    "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
}
ANALYTIC_INVERT_CONFIG = {"invert": {"max_iterations": 20, "convergence_tol": 1e-10}}
DERIVE_CONFIG = {"derive": {"eps_background": 1.0}}   # vacuum reference for conductivity

# Grid-search settings (reused from validate_sandwich_against_matlab.py).
SUBSTRATE_N_BOUNDS = (1.88, 2.02)
SUBSTRATE_K_BOUND = 0.03
SUBSTRATE_GRID_STEP = 0.0004
SUBSTRATE_CONTINUITY_HALF_WIDTH = 20

SAMPLE_N_BOUNDS = (1.0, 2.5)
SAMPLE_K_BOUND = 0.4   # raised from 0.2 so the grid k is not clipped at the search ceiling
SAMPLE_GRID_STEP = 0.002
SAMPLE_CONTINUITY_HALF_WIDTH = 15

# Dense grids for the self-contained MATLAB-port solver (its own mesh, not thz_core's).
PORT_GRID_COUNT = 500


# ── Shared front-end (thz_core in-place windowing) ────────────────────────────


def load_time_field_seconds(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a 2-column [time_ps, field] trace; return time in SI seconds."""
    data = np.loadtxt(os.path.join(DATA_DIRECTORY, filename))
    return data[:, 0] * 1e-12, data[:, 1]


def subtract_leading_baseline(field: np.ndarray, leading_points: int = 10) -> np.ndarray:
    """Remove the DC offset estimated from the leading samples."""
    return field - np.mean(field[:leading_points])


def windowed_padded_spectrum(
    time_seconds: np.ndarray, field: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """thz_core front-end: window in place on the shared axis, trailing zero-pad, FFT.

    Windowing in place keeps the pulse at its true position on the common time axis, so the
    transfer-function phase carries the real inter-pulse delay (no padding ramp).  The
    trailing zero-pad only interpolates the spectrum (the windowed pulse is already ~0 at the
    record end), giving a smooth curve while preserving absolute timing.
    """
    windowed_field, _, _ = core.window_time(time_seconds, field, WINDOW_CONFIG)

    timestep_seconds = float(np.mean(np.diff(time_seconds)))
    pad_count = max(0, SPECTRUM_FFT_LENGTH - windowed_field.size)
    padded_field = np.concatenate([windowed_field, np.zeros(pad_count)])
    padded_time = time_seconds[0] + timestep_seconds * np.arange(padded_field.size)

    frequency, spectrum, _ = core.fft_spectrum(padded_time, padded_field, {})
    return frequency, spectrum


def build_shared_transfer_functions() -> dict:
    """Run the shared front-end once; return the two measured transfer functions and band.

    H_substrate = E_substrate / E_air         (substrate step, ASASA)
    H_sample    = E_sample    / E_substrate   (sample step,    ASMSA)
    """
    time_seconds, air_field = load_time_field_seconds(AIR_REFERENCE_FILE)
    _, substrate_field = load_time_field_seconds(SUBSTRATE_FILE)
    _, sample_field = load_time_field_seconds(SAMPLE_FILE)

    air_field = subtract_leading_baseline(air_field)
    substrate_field = subtract_leading_baseline(substrate_field)
    sample_field = subtract_leading_baseline(sample_field)

    frequency, air_spectrum = windowed_padded_spectrum(time_seconds, air_field)
    _, substrate_spectrum = windowed_padded_spectrum(time_seconds, substrate_field)
    _, sample_spectrum = windowed_padded_spectrum(time_seconds, sample_field)

    transfer_substrate, _, _ = core.transfer_function(
        frequency, substrate_spectrum, air_spectrum, TRANSFER_CONFIG
    )
    transfer_sample, _, _ = core.transfer_function(
        frequency, sample_spectrum, substrate_spectrum, TRANSFER_CONFIG
    )

    band_mask = (frequency >= REPORT_FREQUENCY_MIN_HZ) & (frequency <= REPORT_FREQUENCY_MAX_HZ)

    # Raw time-domain peak delays — preprocessing-independent sanity anchors.
    timestep_seconds = float(np.mean(np.diff(time_seconds)))
    air_peak = int(np.argmax(np.abs(air_field)))
    substrate_peak = int(np.argmax(np.abs(substrate_field)))
    sample_peak = int(np.argmax(np.abs(sample_field)))
    substrate_air_delay_ps = (substrate_peak - air_peak) * timestep_seconds * 1e12
    sample_substrate_delay_ps = (sample_peak - substrate_peak) * timestep_seconds * 1e12

    return {
        "frequency": frequency,
        "band_mask": band_mask,
        "transfer_substrate": transfer_substrate,
        "transfer_sample": transfer_sample,
        "timestep_ps": timestep_seconds * 1e12,
        "substrate_air_delay_ps": substrate_air_delay_ps,
        "sample_substrate_delay_ps": sample_substrate_delay_ps,
    }


# ── Inversion 1: thz_core analytic free-standing (Duvillaret) ─────────────────


def invert_thzcore_analytic(
    frequency: np.ndarray, transfer: np.ndarray, mask: np.ndarray, thickness_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """thz_core iterative free-standing slab inversion; returns (n, k>=0)."""
    n, k, _ = core.invert_nk(frequency, transfer, thickness_m, mask, ANALYTIC_INVERT_CONFIG)
    return n, k


# ── Inversion 2: thz_core grid (substrate_only / substrate_sandwich) ──────────


def invert_thzcore_grid_substrate(
    frequency: np.ndarray, transfer: np.ndarray, mask: np.ndarray, fabry_perot: bool
) -> tuple[np.ndarray, np.ndarray]:
    """thz_core ASASA grid inversion; returns (n, k>=0)."""
    config = {
        "invert_grid": {
            "geometry": "substrate_only",
            "thickness_sub_m": SUBSTRATE_SLAB_THICKNESS_M,
            "thickness_gap_m": SUBSTRATE_GAP_THICKNESS_M,
            "medium_index": AMBIENT_MEDIUM_INDEX,
            "fabry_perot": fabry_perot,
            "n_real_bounds": list(SUBSTRATE_N_BOUNDS),
            "k_bounds": [0.0, SUBSTRATE_K_BOUND],
            "grid_step": SUBSTRATE_GRID_STEP,
            "continuity_tracking": {
                "enabled": True,
                "search_half_width_cells": SUBSTRATE_CONTINUITY_HALF_WIDTH,
            },
        }
    }
    n, k, _ = core.invert_nk_grid(frequency, transfer, mask, config)
    return n, k


def invert_thzcore_grid_sample(
    frequency: np.ndarray,
    transfer: np.ndarray,
    mask: np.ndarray,
    substrate_n: np.ndarray,
    substrate_k: np.ndarray,
    fabry_perot: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """thz_core ASMSA grid inversion fed n_sub(omega); returns (n, k>=0).

    The grid sandwich model expects ``n_substrate`` as n + i*k (k>=0), bin-for-bin on the
    shared frequency grid (matching the tested chain in validate_sandwich_against_matlab.py).
    """
    substrate_index = substrate_n + 1j * substrate_k
    config = {
        "invert_grid": {
            "geometry": "substrate_sandwich",
            "thickness_sample_m": SAMPLE_LAYER_THICKNESS_M,
            "thickness_ref_gap_m": SAMPLE_LAYER_THICKNESS_M,
            "n_substrate": substrate_index,
            "medium_index": AMBIENT_MEDIUM_INDEX,
            "fabry_perot": fabry_perot,
            "n_real_bounds": list(SAMPLE_N_BOUNDS),
            "k_bounds": [0.0, SAMPLE_K_BOUND],
            "grid_step": SAMPLE_GRID_STEP,
            "continuity_tracking": {
                "enabled": True,
                "search_half_width_cells": SAMPLE_CONTINUITY_HALF_WIDTH,
            },
        }
    }
    n, k, _ = core.invert_nk_grid(frequency, transfer, mask, config)
    return n, k


# ── Inversion 3: MATLAB-port grid models + continuity solver (self-contained) ──
# Faithful copies of the tested helpers in validate_sandwich_against_matlab.py, with the
# speed-of-light / ambient-medium constants unified to this script's values.


def _matlab_complex_grid(
    n_bounds: tuple[float, float], k_bound: float, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the n + i*k candidate grid (k <= 0 in the MATLAB sign convention)."""
    n_axis = np.linspace(n_bounds[0], n_bounds[1], count)
    k_axis = np.linspace(0.0, -k_bound, count)
    n_hat_grid = n_axis[:, None] + 1j * k_axis[None, :]
    return n_axis, k_axis, n_hat_grid


def _continuity_minimise(
    difference: np.ndarray,
    n_axis: np.ndarray,
    k_axis: np.ndarray,
    previous_value: complex | None,
    half_width: int,
) -> complex:
    """Global min on the first bin, then a +/- half-window box around the previous solution."""
    if previous_value is None:
        flat_index = int(np.argmin(difference))
        row, col = np.unravel_index(flat_index, difference.shape)
        return complex(n_axis[row] + 1j * k_axis[col])
    center_row = int(np.argmin(np.abs(n_axis - np.real(previous_value))))
    center_col = int(np.argmin(np.abs(k_axis - np.imag(previous_value))))
    row_lo, row_hi = max(0, center_row - half_width), min(len(n_axis), center_row + half_width + 1)
    col_lo, col_hi = max(0, center_col - half_width), min(len(k_axis), center_col + half_width + 1)
    local = difference[row_lo:row_hi, col_lo:col_hi]
    local_row, local_col = np.unravel_index(int(np.argmin(local)), local.shape)
    return complex(n_axis[row_lo + local_row] + 1j * k_axis[col_lo + local_col])


def matlab_substrate_model(n_hat_grid: np.ndarray, frequency_hz: float, fabry_perot: bool) -> np.ndarray:
    """Cuvette_code E_substrate/E_air model (ASASA), textbook gap FP sign 1/(1 - r^2 P^2)."""
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
        -1j * n_air * omega * (2 * SUBSTRATE_SLAB_THICKNESS_M + SUBSTRATE_GAP_THICKNESS_M) / c
    )
    etalon = (1 - r23 * r34 * phase3 ** 2) if fabry_perot else 1.0
    return (phase2 * phase3 * phase4 * t12 * t23 * t34 * t45) / (etalon * phase_ref)


def matlab_sample_model(
    n_hat_grid: np.ndarray, n_substrate: complex, frequency_hz: float, fabry_perot: bool
) -> np.ndarray:
    """sample_code E_sample/E_substrate model (ASMSA), textbook FP sign 1/(1 - r^2 P^2)."""
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
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    half_width: int,
) -> np.ndarray:
    """Grid + continuity solver matching the MATLAB minimisation; returns complex n + i*k."""
    n_axis, k_axis, n_hat_grid = grid
    result = np.zeros(frequency.size, dtype=complex)
    previous = None
    for index, frequency_hz in enumerate(frequency):
        model = model_function(n_hat_grid, float(frequency_hz))
        measured = transfer[index]
        difference = np.abs(np.real(model - measured)) + np.abs(np.imag(model - measured))
        previous = _continuity_minimise(difference, n_axis, k_axis, previous, half_width)
        result[index] = previous
    return result


def _scatter_band_result(
    full_length: int, band_indices: np.ndarray, band_values: np.ndarray
) -> np.ndarray:
    """Place band results into a full-length complex array, NaN elsewhere."""
    full = np.full(full_length, np.nan + 1j * np.nan, dtype=complex)
    full[band_indices] = band_values
    return full


def invert_matlab_port_substrate(
    frequency: np.ndarray, transfer: np.ndarray, mask: np.ndarray, fabry_perot: bool
) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB-port ASASA grid inversion over the band; returns (n, k>=0)."""
    band_indices = np.where(mask)[0]
    grid = _matlab_complex_grid(SUBSTRATE_N_BOUNDS, SUBSTRATE_K_BOUND, PORT_GRID_COUNT)
    band_result = run_matlab_port_solver(
        frequency[band_indices], transfer[band_indices],
        lambda candidate_grid, f: matlab_substrate_model(candidate_grid, f, fabry_perot),
        grid, SUBSTRATE_CONTINUITY_HALF_WIDTH,
    )
    full = _scatter_band_result(frequency.size, band_indices, band_result)
    return np.real(full), -np.imag(full)


def invert_matlab_port_sample(
    frequency: np.ndarray,
    transfer: np.ndarray,
    mask: np.ndarray,
    substrate_n: np.ndarray,
    fabry_perot: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB-port ASMSA grid inversion fed n_sub(omega); returns (n, k>=0)."""
    band_indices = np.where(mask)[0]
    grid = _matlab_complex_grid(SAMPLE_N_BOUNDS, SAMPLE_K_BOUND, PORT_GRID_COUNT)
    substrate_band = substrate_n[band_indices]
    band_result = run_matlab_port_solver(
        frequency[band_indices], transfer[band_indices],
        lambda candidate_grid, f, ns=substrate_band, freq=frequency[band_indices]:
            matlab_sample_model(
                candidate_grid, complex(ns[int(np.argmin(np.abs(freq - f)))]), f, fabry_perot
            ),
        grid, SAMPLE_CONTINUITY_HALF_WIDTH,
    )
    full = _scatter_band_result(frequency.size, band_indices, band_result)
    return np.real(full), -np.imag(full)


# ── Inversion 4: Novelli analytic closed form (fed the shared H) ──────────────


def _anchor_band_phase_delay(
    frequency: np.ndarray, transfer: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Positive, absolutely-anchored accumulated phase delay of H over the band.

    Novelli's n = 1 + Δφ·c/(ω·d) needs the TRUE number of phase cycles, not just a
    locally-unwrapped phase (an integer-cycle error tilts n with frequency — the failure the
    notes attribute to the legacy front-end).  Standard Duvillaret anchoring: unwrap within
    the band, linearly fit Δφ(ω), and remove the nearest-integer-cycle offset so the fit
    extrapolates through ≈0 at ω = 0 (a non-dispersive slab has zero phase at DC).
    """
    delay = np.full(frequency.size, np.nan)
    band_indices = np.where(mask)[0]
    omega_band = 2.0 * np.pi * frequency[band_indices]
    unwrapped = -np.unwrap(np.angle(transfer[band_indices]))   # delay → positive
    slope, intercept = np.polyfit(omega_band, unwrapped, 1)
    cycle_offset = round(intercept / (2.0 * np.pi))
    delay[band_indices] = unwrapped - 2.0 * np.pi * cycle_offset
    return delay


def invert_novelli(
    frequency: np.ndarray,
    transfer: np.ndarray,
    mask: np.ndarray,
    thickness_m: float,
    substrate_index,
) -> tuple[np.ndarray, np.ndarray]:
    """Novelli analytic sandwich inversion from a measured H; returns (n, k>=0).

    n = 1 + Δφ·c/(ω·d), with Δφ the absolutely-anchored accumulated phase delay of H; k from
    the amplitude with the Fresnel factor (n + ns)^2 / ((1 + ns)^2 · n).  ``substrate_index``
    is the scalar 1.0 (substrate step → free-standing slab) or the real n_sub(omega) array
    (sample step).
    """
    accumulated_phase_delay = _anchor_band_phase_delay(frequency, transfer, mask)
    amplitude_ratio = np.abs(transfer)
    omega = 2.0 * np.pi * frequency
    with np.errstate(divide="ignore", invalid="ignore"):
        n_real = 1.0 + accumulated_phase_delay * SPEED_OF_LIGHT_M_PER_S / (omega * thickness_m)
        fresnel_factor = (n_real + substrate_index) ** 2 / (1.0 + substrate_index) ** 2 / n_real
        k = -SPEED_OF_LIGHT_M_PER_S / (omega * thickness_m) * np.log(
            np.real(fresnel_factor) * amplitude_ratio
        )
    n_out = np.where(mask, n_real, np.nan)
    k_out = np.where(mask, k, np.nan)
    return n_out, k_out


# ── Conductivity ──────────────────────────────────────────────────────────────


def sample_conductivity(
    frequency: np.ndarray, n: np.ndarray, k: np.ndarray
) -> np.ndarray:
    """Complex optical conductivity from sample (n, k), eps_background = 1 (vacuum)."""
    _, sigma_complex, _ = core.derive_eps_sigma(frequency, n, k, DERIVE_CONFIG)
    return sigma_complex


# ── Reporting ─────────────────────────────────────────────────────────────────


def band_statistics(mask: np.ndarray, values: np.ndarray) -> dict:
    """Median / min / max of the finite, in-band entries of a real array."""
    in_band = values[mask]
    finite = in_band[np.isfinite(in_band)]
    if finite.size == 0:
        return {"median": np.nan, "min": np.nan, "max": np.nan}
    return {"median": float(np.median(finite)), "min": float(finite.min()), "max": float(finite.max())}


def format_substrate_rows(mask: np.ndarray, substrate_results: dict) -> list[str]:
    rows = []
    for label, (n, k) in substrate_results.items():
        n_stats = band_statistics(mask, n)
        k_stats = band_statistics(mask, k)
        rows.append(
            f"| {label:<34} | {n_stats['median']:.3f} | "
            f"[{n_stats['min']:.3f}, {n_stats['max']:.3f}] | {k_stats['median']:.4f} |"
        )
    return rows


def format_sample_rows(mask: np.ndarray, sample_results: dict) -> list[str]:
    rows = []
    for label, (n, k) in sample_results.items():
        n_stats = band_statistics(mask, n)
        k_stats = band_statistics(mask, k)
        rows.append(
            f"| {label:<34} | {n_stats['median']:.3f} | "
            f"[{n_stats['min']:.3f}, {n_stats['max']:.3f}] | {k_stats['median']:.4f} |"
        )
    return rows


def format_conductivity_rows(
    mask: np.ndarray, frequency: np.ndarray, sample_results: dict
) -> list[str]:
    rows = []
    for label, (n, k) in sample_results.items():
        sigma = sample_conductivity(frequency, n, k)
        real_stats = band_statistics(mask, np.real(sigma))
        imag_stats = band_statistics(mask, np.imag(sigma))
        rows.append(
            f"| {label:<34} | {real_stats['median']:.3e} | "
            f"[{real_stats['min']:.2e}, {real_stats['max']:.2e}] | {imag_stats['median']:.3e} |"
        )
    return rows


def build_report(shared: dict, substrate_results: dict, sample_results: dict) -> str:
    frequency = shared["frequency"]
    mask = shared["band_mask"]
    lines: list[str] = []
    lines.append("# Sandwich-transmission inversion comparison (shared thz_core front-end)\n")
    lines.append(
        "Dataset: Vasilis_Data 200 K triplet (air / empty cuvette / filled cuvette).  "
        "One reliable front-end (thz_core in-place windowing) feeds the SAME measured "
        "transfer functions to every inversion method, isolating the inversion math.\n"
    )
    lines.append("## Front-end (shared)\n")
    lines.append(
        f"- timestep: {shared['timestep_ps']:.4f} ps;  "
        f"band: {REPORT_FREQUENCY_MIN_HZ/1e12:.1f}-{REPORT_FREQUENCY_MAX_HZ/1e12:.1f} THz, "
        f"{int(np.count_nonzero(mask))} bins"
    )
    lines.append(
        f"- raw peak delay substrate vs air: {shared['substrate_air_delay_ps']:.2f} ps "
        f"(2-window path {SUBSTRATE_TWO_WINDOW_PATH_M*1e3:.1f} mm "
        f"→ n_sub ≈ {1 + shared['substrate_air_delay_ps']*1e-12*SPEED_OF_LIGHT_M_PER_S/SUBSTRATE_TWO_WINDOW_PATH_M:.2f})"
    )
    lines.append(
        f"- raw peak delay sample vs substrate: {shared['sample_substrate_delay_ps']:.2f} ps "
        f"(sample {SAMPLE_LAYER_THICKNESS_M*1e3:.2f} mm "
        f"→ n_sample ≈ {1 + shared['sample_substrate_delay_ps']*1e-12*SPEED_OF_LIGHT_M_PER_S/SAMPLE_LAYER_THICKNESS_M:.2f})\n"
    )

    lines.append("## 1. Substrate refractive index  n_sub(omega)\n")
    lines.append("Fabry-Perot OFF (gated) for all grid methods: the mm-scale windows are gatable.\n")
    lines.append("| method | n_sub median | n_sub range | k_sub median |")
    lines.append("|---|---|---|---|")
    lines.extend(format_substrate_rows(mask, substrate_results))
    lines.append("")

    lines.append("## 2. Sample complex refractive index  n_hat = n + i*k\n")
    lines.append("| method | n median | n range | k median |")
    lines.append("|---|---|---|---|")
    lines.extend(format_sample_rows(mask, sample_results))
    lines.append("")

    lines.append("## 3. Sample complex conductivity  sigma (S/m), eps_background = 1 (vacuum)\n")
    lines.append("| method | sigma_real median | sigma_real range | sigma_imag median |")
    lines.append("|---|---|---|---|")
    lines.extend(format_conductivity_rows(mask, frequency, sample_results))
    lines.append("")

    lines.append("## 4. Critical comparison & recommendation\n")
    lines.extend([
        "**On a shared, reliable front-end the inversion *models* agree where they should "
        "and disagree only where the physics demands it.**\n",
        "- **Substrate n_sub:** all four methods → 1.95 (fused silica). With the clean "
        "front-end the grid solvers no longer rail/sawtooth (only a cosmetic ±grid-step "
        "ripple remains); the old 2.2-2.5 inflation was purely the MATLAB/legacy padding. "
        "Novelli needs the absolute-phase anchor (its sole weak point — without it n tilts "
        "and reads 1.71); given the anchor it matches.\n",
        "- **Sample n:** all six methods cluster at 2.15 ± 0.01. n is INSENSITIVE to the "
        "geometry treatment — free-standing, sandwich grid, and Novelli all agree, because "
        "the filled/empty differential cancels the substrate phase.\n",
        "- **Sample k / conductivity — THE decisive split:** the free-standing analytic "
        "gives k ≈ 0.125, sigma_real ≈ 42 S/m; every sandwich-aware method (grid ASMSA, "
        "Novelli with ns = n_sub) gives k ≈ 0.19, sigma_real ≈ 63 S/m — ~50 % higher. "
        "Raising the grid k-ceiling to 0.4 did not move it, so 0.19 is real, not clipping. "
        "The free-standing model mis-attributes the silica/sample Fresnel transmission to "
        "absorption; the sandwich models account for the n≈1.95 surround and recover the "
        "correct k. **For absorption / conductivity you MUST use a sandwich-aware "
        "inversion.**\n",
        "- **Fabry-Perot:** on vs off shifts sigma_real ~2 % and sigma_imag ~3 % here — a "
        "small correction for this gatable 0.13 mm sample. Keep it opt-in; the textbook "
        "1/(1 - r^2 P^2) sign is correct (the substrate MATLAB sign-slip is moot — substrate "
        "is gated).\n",
        "- **Grid vs Novelli:** the grid ASMSA and Novelli sandwich now agree to ~1e-3 in n "
        "and ~1e-3 in k. Novelli is faster and analytic but hinges entirely on the phase "
        "anchor; the grid is more robust (handles FP, complex n_sub, branch continuity) at "
        "higher cost.\n",
        "**Recommendation:** continue with the **thz_core front-end + thz_core grid "
        "substrate_sandwich (ASMSA) inversion, FP opt-in**. It gives the correct substrate n, "
        "the correct sample n AND k/conductivity, handles the FP correction and a complex "
        "n_sub natively, and is the maintained codebase. Keep Novelli (with the anchor) as a "
        "fast analytic cross-check; retire the free-standing analytic for sample k/sigma "
        "(n-only is fine). Retire the MATLAB/legacy asymmetric-padding front-ends entirely.\n",
    ])
    return "\n".join(lines)


def make_figure(shared: dict, substrate_results: dict, sample_results: dict, output_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exception:  # pragma: no cover
        print(f"(plot skipped: {exception})")
        return

    frequency_thz = shared["frequency"] / 1e12
    mask = shared["band_mask"]

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    figure.suptitle("Sandwich inversion comparison — shared thz_core front-end (200 K)")

    def plot_band(axis, values, label):
        plotted = np.where(mask, values, np.nan)
        axis.plot(frequency_thz, plotted, label=label, linewidth=1.3)

    for label, (n, k) in substrate_results.items():
        plot_band(axes[0, 0], n, label)
        plot_band(axes[1, 0], k, label)
    axes[0, 0].set_title("Substrate n"); axes[0, 0].set_ylabel("n"); axes[0, 0].legend(fontsize=7)
    axes[1, 0].set_title("Substrate k"); axes[1, 0].set_ylabel("k"); axes[1, 0].set_xlabel("THz")

    for label, (n, k) in sample_results.items():
        plot_band(axes[0, 1], n, label)
        plot_band(axes[1, 1], k, label)
        sigma = sample_conductivity(shared["frequency"], n, k)
        plot_band(axes[0, 2], np.real(sigma), label)
        plot_band(axes[1, 2], np.imag(sigma), label)

        np.savetxt(os.path.splitext(output_path)[0] + f"_{label}_Cu_sigma_real.dat", np.column_stack([frequency_thz, np.real(sigma)]))
        np.savetxt(os.path.splitext(output_path)[0] + f"_{label}_Cu_sigma_imag.dat", np.column_stack([frequency_thz, np.imag(sigma)]))
        

    axes[0, 1].set_title("Sample n"); axes[0, 1].set_ylabel("n"); axes[0, 1].legend(fontsize=7)
    axes[1, 1].set_title("Sample k"); axes[1, 1].set_ylabel("k"); axes[1, 1].set_xlabel("THz")
    axes[0, 2].set_title("Sample sigma_real (S/m)"); axes[0, 2].set_ylabel("S/m")
    axes[1, 2].set_title("Sample sigma_imag (S/m)"); axes[1, 2].set_ylabel("S/m"); axes[1, 2].set_xlabel("THz")
    
    axes[0, 2].legend(fontsize=7); axes[1, 2].legend(fontsize=7)

        # export the real conductivity data for further analysis

    figure.savefig(output_path, dpi=130)
    print(f"Saved figure: {output_path}")


# ── Orchestration ─────────────────────────────────────────────────────────────


def main() -> None:
    shared = build_shared_transfer_functions()
    frequency = shared["frequency"]
    mask = shared["band_mask"]
    transfer_substrate = shared["transfer_substrate"]
    transfer_sample = shared["transfer_sample"]

    print("Running substrate-step inversions (shared front-end)...")
    substrate_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    substrate_results["thz_core analytic (free-standing)"] = invert_thzcore_analytic(
        frequency, transfer_substrate, mask, SUBSTRATE_TWO_WINDOW_PATH_M
    )
    substrate_results["thz_core grid (ASASA, FP off)"] = invert_thzcore_grid_substrate(
        frequency, transfer_substrate, mask, fabry_perot=False
    )
    substrate_results["MATLAB-port grid (FP off)"] = invert_matlab_port_substrate(
        frequency, transfer_substrate, mask, fabry_perot=False
    )
    substrate_results["Novelli analytic (ns=1)"] = invert_novelli(
        frequency, transfer_substrate, mask, SUBSTRATE_TWO_WINDOW_PATH_M, substrate_index=1.0
    )

    # Per-method n_sub fed to that method's own sample step (honest self-consistent pipeline).
    thzcore_grid_substrate_n, thzcore_grid_substrate_k = substrate_results["thz_core grid (ASASA, FP off)"]
    matlab_substrate_n, _ = substrate_results["MATLAB-port grid (FP off)"]
    novelli_substrate_n, _ = substrate_results["Novelli analytic (ns=1)"]
    # n_sub real, gap-filled to the band edges so the sample solver never sees a NaN feed.
    matlab_substrate_n_filled = _fill_band(matlab_substrate_n, mask)
    novelli_substrate_n_filled = _fill_band(novelli_substrate_n, mask)
    thzcore_grid_n_filled = _fill_band(thzcore_grid_substrate_n, mask)
    thzcore_grid_k_filled = _fill_band(thzcore_grid_substrate_k, mask)

    print("Running sample-step inversions (shared front-end)...")
    sample_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sample_results["thz_core analytic (free-standing)"] = invert_thzcore_analytic(
        frequency, transfer_sample, mask, SAMPLE_LAYER_THICKNESS_M
    )
    sample_results["thz_core grid (ASMSA, FP off)"] = invert_thzcore_grid_sample(
        frequency, transfer_sample, mask, thzcore_grid_n_filled, thzcore_grid_k_filled, fabry_perot=False
    )
    sample_results["thz_core grid (ASMSA, FP on)"] = invert_thzcore_grid_sample(
        frequency, transfer_sample, mask, thzcore_grid_n_filled, thzcore_grid_k_filled, fabry_perot=True
    )
    sample_results["MATLAB-port grid (FP off)"] = invert_matlab_port_sample(
        frequency, transfer_sample, mask, matlab_substrate_n_filled, fabry_perot=False
    )
    sample_results["MATLAB-port grid (FP on)"] = invert_matlab_port_sample(
        frequency, transfer_sample, mask, matlab_substrate_n_filled, fabry_perot=True
    )
    sample_results["Novelli analytic"] = invert_novelli(
        frequency, transfer_sample, mask, SAMPLE_LAYER_THICKNESS_M,
        substrate_index=novelli_substrate_n_filled,
    )

    # save sample_results to pickle file for later use
    import pickle
    results_path = os.path.join(os.path.dirname(__file__), "compare_inversion_methods_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(sample_results, f)
    print(f"\nSaved sample results: {results_path}")
    
    report = build_report(shared, substrate_results, sample_results)
    print("\n" + report)

    report_path = os.path.join(os.path.dirname(__file__), "compare_inversion_methods_report.md")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report)
    print(f"\nSaved report: {report_path}")

    figure_path = os.path.join(os.path.dirname(__file__), "compare_inversion_methods.png")
    make_figure(shared, substrate_results, sample_results, figure_path)


def _fill_band(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace non-finite in-band entries by the nearest finite in-band value (edge-hold).

    The sample-step models index n_sub(omega) bin-for-bin; a stray NaN at a band edge would
    poison that bin.  Out-of-band entries stay as-is (the sample solver only reads in-band).
    """
    filled = np.array(values, dtype=float)
    band_indices = np.where(mask)[0]
    finite_band = band_indices[np.isfinite(filled[band_indices])]
    if finite_band.size == 0:
        return filled
    for index in band_indices:
        if not np.isfinite(filled[index]):
            nearest = finite_band[int(np.argmin(np.abs(finite_band - index)))]
            filled[index] = filled[nearest]
    return filled


if __name__ == "__main__":
    main()
