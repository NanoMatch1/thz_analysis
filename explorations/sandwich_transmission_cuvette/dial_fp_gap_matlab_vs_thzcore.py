"""Dial the empty-cuvette air gap and compare the two FP-ON sample inversions.

Focused companion to ``compare_inversion_methods_shared_frontend.py``.  Here we keep ONLY
the two Fabry-Perot-ON sandwich-aware sample inversions:

    * MATLAB-port grid  (sample_code model, FP on)
    * thz_core grid     (substrate_sandwich / ASMSA, FP on)

Both share one front-end (thz_core in-place windowing) and one substrate extraction
(thz_core ASASA, gated), so any difference between the two curves is purely the sample
inversion math — not preprocessing and not n_sub.

WHY THIS SCRIPT EXISTS — the FP dial
------------------------------------
The empty-cuvette reference is  air | sub | AIR GAP | sub | air.  That thin air gap is the
dominant Fabry-Perot cavity in the whole experiment:

    * It is NOT gatable: a ~0.1 mm air gap has a round-trip echo of only ~0.7 ps, buried
      under the main pulse — you cannot window it out, so FP-OFF is the wrong assumption.
    * It is strong: r at the sub|air interface is (1.95-1.0)/(1.95+1.0) ~ 0.32, so
      r^2 ~ 0.10 — a ~10 % etalon ripple on the reference (denominator of H_sample).
    * The sample's OWN FP is negligible: the sample (n~2.15) is nearly index-matched to the
      silica (n~1.95), so r^2 ~ 0.002.  The gap dominates the FP correction.

The ripple period (free spectral range) is set by the gap thickness:

    FSR = c / (2 * n_air * d_gap)

so the air gap is the one parameter you dial to make the modelled FP correction line up
with the ripple actually present in the measured |H_sample|.  This script exposes that gap
as ``REFERENCE_GAP_THICKNESS_M`` (overridable on the command line) and overlays the FSR
comb on the measured |H| so you can read off the right value by eye, then confirm it
flattens the residual ripple in n / k / sigma.

Run (default gap):
    PYTHONPATH=. ../.venv/Scripts/python.exe \
        explorations/sandwich_transmission_cuvette/dial_fp_gap_matlab_vs_thzcore.py

Run with a specific gap in micrometres (e.g. 90 um), for quick dialing:
    PYTHONPATH=. ../.venv/Scripts/python.exe \
        explorations/sandwich_transmission_cuvette/dial_fp_gap_matlab_vs_thzcore.py --gap-um 90
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy import constants as physical_constants

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import thz_core.thz_core as core

try:  # the report prints maths symbols; force UTF-8 on a cp1252 console.
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass


# ════════════════════════════════════════════════════════════════════════════
#  THE FP DIAL — empty-cuvette air-gap thickness (the dominant FP cavity)
# ════════════════════════════════════════════════════════════════════════════
# Round-trip echo ~ 2*d_gap/c (air); FSR = c/(2*d_gap).  Same spacer sets the sample
# thickness, so the true gap is ~ the sample thickness, but dial it freely here to match
# the ripple seen in the measured |H_sample| panel.  Override with --gap-um.
REFERENCE_GAP_THICKNESS_M = 0.13e-3      # <-- DIAL ME  (60-130 um is the plausible range)

# Sample's own thickness (the layer that REPLACES the air gap in the filled cuvette).
SAMPLE_LAYER_THICKNESS_M = 0.13e-3


# ── Fixed experimental parameters ─────────────────────────────────────────────

SPEED_OF_LIGHT_M_PER_S = physical_constants.c
AMBIENT_MEDIUM_INDEX = 1.0               # surround + empty gap treated as vacuum
VACUUM_PERMITTIVITY = physical_constants.epsilon_0

DATA_DIRECTORY = r"C:\Users\Samuel\Data\THz\Vasilis_Data\data"
AIR_REFERENCE_FILE = "reference_200K_tr.dat"
SUBSTRATE_FILE = "Substrate_200K_tr.dat"
SAMPLE_FILE = "Sample_200K_tr.dat"

# Fused-silica windows, 0.9 mm each; empty-cuvette path crosses BOTH windows.
SUBSTRATE_SLAB_THICKNESS_M = 0.9e-3
SUBSTRATE_GAP_THICKNESS_M = 0.09e-3      # inter-window gap for the ASASA substrate step
SUBSTRATE_TWO_WINDOW_PATH_M = 2 * SUBSTRATE_SLAB_THICKNESS_M

REPORT_FREQUENCY_MIN_HZ = 0.8e12
REPORT_FREQUENCY_MAX_HZ = 2.0e12

# Trailing zero-pad: smooths the spectra WITHOUT shifting the windowed pulse (no group delay).
SPECTRUM_FFT_LENGTH = 4096

WINDOW_CONFIG = {"window": {"type": "tukey", "alpha": 1}}
TRANSFER_CONFIG = {
    "transfer": {"apply_snr_mask": False},
    "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
}
DERIVE_CONFIG = {"derive": {"eps_background": 1.0}}   # vacuum reference for conductivity

# Substrate (ASASA) grid — gated, just to provide n_sub(omega) for both sample methods.
SUBSTRATE_N_BOUNDS = (1.88, 2.02)
SUBSTRATE_K_BOUND = 0.03
SUBSTRATE_GRID_STEP = 0.0004
SUBSTRATE_CONTINUITY_HALF_WIDTH = 20

# Sample (ASMSA) grid.
SAMPLE_N_BOUNDS = (1.0, 2.5)
SAMPLE_K_BOUND = 0.4
SAMPLE_GRID_STEP = 0.002
SAMPLE_CONTINUITY_HALF_WIDTH = 15

# Dense mesh for the self-contained MATLAB-port solver.
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
    """Window in place on the shared axis, trailing zero-pad, FFT (preserves absolute timing)."""
    windowed_field, _, _ = core.window_time(time_seconds, field, WINDOW_CONFIG)
    timestep_seconds = float(np.mean(np.diff(time_seconds)))
    pad_count = max(0, SPECTRUM_FFT_LENGTH - windowed_field.size)
    padded_field = np.concatenate([windowed_field, np.zeros(pad_count)])
    padded_time = time_seconds[0] + timestep_seconds * np.arange(padded_field.size)
    frequency, spectrum, _ = core.fft_spectrum(padded_time, padded_field, {})
    return frequency, spectrum


def build_shared_transfer_functions() -> dict:
    """Run the shared front-end once; return the substrate-step and sample-step H + band."""
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
    return {
        "frequency": frequency,
        "band_mask": band_mask,
        "transfer_substrate": transfer_substrate,
        "transfer_sample": transfer_sample,
    }


# ── Substrate step (thz_core ASASA, gated) → n_sub feed ───────────────────────


def invert_thzcore_grid_substrate(
    frequency: np.ndarray, transfer: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """thz_core ASASA grid inversion (FP off / gated); returns (n, k>=0)."""
    config = {
        "invert_grid": {
            "geometry": "substrate_only",
            "thickness_sub_m": SUBSTRATE_SLAB_THICKNESS_M,
            "thickness_gap_m": SUBSTRATE_GAP_THICKNESS_M,
            "medium_index": AMBIENT_MEDIUM_INDEX,
            "fabry_perot": False,
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


# ── Sample step A: thz_core ASMSA, FP on ──────────────────────────────────────


def invert_thzcore_grid_sample_fp_on(
    frequency: np.ndarray,
    transfer: np.ndarray,
    mask: np.ndarray,
    substrate_n: np.ndarray,
    substrate_k: np.ndarray,
    reference_gap_thickness_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """thz_core ASMSA grid inversion, Fabry-Perot ON, fed n_sub(omega); returns (n, k>=0)."""
    substrate_index = substrate_n + 1j * substrate_k
    config = {
        "invert_grid": {
            "geometry": "substrate_sandwich",
            "thickness_sample_m": SAMPLE_LAYER_THICKNESS_M,
            "thickness_ref_gap_m": reference_gap_thickness_m,
            "n_substrate": substrate_index,
            "medium_index": AMBIENT_MEDIUM_INDEX,
            "fabry_perot": True,
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


# ── Sample step B: MATLAB-port (sample_code) model + solver, FP on ────────────


def matlab_sample_model_fp_on(
    n_hat_grid: np.ndarray,
    n_substrate: complex,
    frequency_hz: float,
    reference_gap_thickness_m: float,
) -> np.ndarray:
    """sample_code E_sample/E_substrate model (ASMSA) with both FP etalons ON.

    Sample etalon rides on the sample layer (d_sample); reference etalon rides on the empty
    air gap (d_gap = ``reference_gap_thickness_m``) — the dominant, dialable FP term.
    Textbook sign 1/(1 - r^2 P^2), here written 1/(1 + r23*r34*P^2) with r34 = -r23.
    """
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
    phase3ref = np.exp(-1j * n_air * omega * reference_gap_thickness_m / c)
    sample_etalon = 1 + r23 * r34 * phase3 ** 2
    reference_etalon = 1 + r23ref * r34ref * phase3ref ** 2
    numerator = phase3 * t23 * t34 / sample_etalon
    denominator = phase3ref * t23ref * t34ref / reference_etalon
    return numerator / denominator


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


def invert_matlab_port_sample_fp_on(
    frequency: np.ndarray,
    transfer: np.ndarray,
    mask: np.ndarray,
    substrate_n: np.ndarray,
    reference_gap_thickness_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB-port ASMSA grid inversion, FP on, fed n_sub(omega); returns (n, k>=0)."""
    band_indices = np.where(mask)[0]
    n_axis, k_axis, n_hat_grid = _matlab_complex_grid(SAMPLE_N_BOUNDS, SAMPLE_K_BOUND, PORT_GRID_COUNT)
    substrate_band = substrate_n[band_indices]
    band_frequency = frequency[band_indices]
    band_transfer = transfer[band_indices]

    result = np.zeros(band_indices.size, dtype=complex)
    previous = None
    for position, frequency_hz in enumerate(band_frequency):
        n_sub_here = complex(substrate_band[position])
        model = matlab_sample_model_fp_on(
            n_hat_grid, n_sub_here, float(frequency_hz), reference_gap_thickness_m
        )
        measured = band_transfer[position]
        difference = np.abs(np.real(model - measured)) + np.abs(np.imag(model - measured))
        previous = _continuity_minimise(difference, n_axis, k_axis, previous, SAMPLE_CONTINUITY_HALF_WIDTH)
        result[position] = previous

    full = np.full(frequency.size, np.nan + 1j * np.nan, dtype=complex)
    full[band_indices] = result
    return np.real(full), -np.imag(full)


# ── Conductivity / helpers ─────────────────────────────────────────────────────


def sample_conductivity(frequency: np.ndarray, n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Complex optical conductivity from sample (n, k), eps_background = 1 (vacuum)."""
    _, sigma_complex, _ = core.derive_eps_sigma(frequency, n, k, DERIVE_CONFIG)
    return sigma_complex


def _fill_band(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Edge-hold non-finite in-band entries so the sample solver never reads a NaN feed."""
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


def band_statistics(mask: np.ndarray, values: np.ndarray) -> dict:
    """Median / min / max of the finite, in-band entries of a real array."""
    finite = values[mask][np.isfinite(values[mask])]
    if finite.size == 0:
        return {"median": np.nan, "min": np.nan, "max": np.nan}
    return {"median": float(np.median(finite)), "min": float(finite.min()), "max": float(finite.max())}


def gap_free_spectral_range_hz(gap_thickness_m: float) -> float:
    """FSR of the empty air-gap etalon: c / (2 * n_air * d_gap)."""
    return SPEED_OF_LIGHT_M_PER_S / (2.0 * AMBIENT_MEDIUM_INDEX * gap_thickness_m)


# ── Reporting / figure ─────────────────────────────────────────────────────────


def print_report(
    shared: dict, gap_thickness_m: float, sample_results: dict, substrate_n: np.ndarray
) -> None:
    mask = shared["band_mask"]
    frequency = shared["frequency"]
    fsr_thz = gap_free_spectral_range_hz(gap_thickness_m) / 1e12
    round_trip_ps = 2.0 * gap_thickness_m / SPEED_OF_LIGHT_M_PER_S * 1e12

    n_sub_stats = band_statistics(mask, substrate_n)
    print("\n" + "=" * 78)
    print("FP-ON sample inversion: MATLAB-port vs thz_core ASMSA  (shared front-end)")
    print("=" * 78)
    print(f"  air gap dialed:   {gap_thickness_m*1e6:7.1f} um   "
          f"(round-trip {round_trip_ps:.2f} ps, etalon FSR {fsr_thz:.3f} THz)")
    print(f"  sample thickness: {SAMPLE_LAYER_THICKNESS_M*1e6:7.1f} um")
    print(f"  n_sub (shared input, gated ASASA): median {n_sub_stats['median']:.3f} "
          f"[{n_sub_stats['min']:.3f}, {n_sub_stats['max']:.3f}]")
    print(f"  band {REPORT_FREQUENCY_MIN_HZ/1e12:.1f}-{REPORT_FREQUENCY_MAX_HZ/1e12:.1f} THz, "
          f"{int(np.count_nonzero(mask))} bins")
    print("-" * 78)
    header = f"  {'method':<26} {'n med':>8} {'k med':>9} {'sigma_re med':>14} {'sigma_im med':>14}"
    print(header)
    for label, (n, k) in sample_results.items():
        sigma = sample_conductivity(frequency, n, k)
        n_med = band_statistics(mask, n)["median"]
        k_med = band_statistics(mask, k)["median"]
        sr_med = band_statistics(mask, np.real(sigma))["median"]
        si_med = band_statistics(mask, np.imag(sigma))["median"]
        print(f"  {label:<26} {n_med:8.3f} {k_med:9.4f} {sr_med:14.3e} {si_med:14.3e}")
    print("=" * 78)


def make_figure(
    shared: dict, gap_thickness_m: float, sample_results: dict,
    substrate_n: np.ndarray, output_path: str
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exception:  # pragma: no cover
        print(f"(plot skipped: {exception})")
        return

    frequency = shared["frequency"]
    frequency_thz = frequency / 1e12
    mask = shared["band_mask"]
    fsr_thz = gap_free_spectral_range_hz(gap_thickness_m) / 1e12

    def plot_band(axis, values, label, **kwargs):
        axis.plot(frequency_thz, np.where(mask, values, np.nan), label=label, linewidth=1.4, **kwargs)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    figure.suptitle(
        f"FP-ON sample inversion — air gap {gap_thickness_m*1e6:.0f} um "
        f"(etalon FSR {fsr_thz:.3f} THz), MATLAB-port vs thz_core ASMSA (200 K)"
    )

    # (0,0) measured |H_sample| with the gap-FSR comb — the dialing aid.
    measured_amplitude = np.abs(shared["transfer_sample"])
    plot_band(axes[0, 0], measured_amplitude, "|H_sample| measured", color="black")
    band_lo, band_hi = REPORT_FREQUENCY_MIN_HZ / 1e12, REPORT_FREQUENCY_MAX_HZ / 1e12
    comb = np.arange(np.ceil(band_lo / fsr_thz), band_hi / fsr_thz + 1) * fsr_thz
    for line_position in comb:
        axes[0, 0].axvline(line_position, color="tab:red", linestyle=":", linewidth=0.9)
    axes[0, 0].set_title("measured |H| + gap-FSR comb (dial gap to match ripple)")
    axes[0, 0].set_ylabel("|H_sample|"); axes[0, 0].legend(fontsize=7)

    # (0,1) n_sub shared input.
    plot_band(axes[0, 1], substrate_n, "n_sub (gated ASASA)", color="tab:green")
    axes[0, 1].set_title("substrate n (shared input)"); axes[0, 1].set_ylabel("n_sub")
    axes[0, 1].legend(fontsize=7)

    # remaining panels: the two FP-ON methods overlaid.
    for label, (n, k) in sample_results.items():
        sigma = sample_conductivity(frequency, n, k)
        plot_band(axes[0, 2], n, label)
        plot_band(axes[1, 0], k, label)
        plot_band(axes[1, 1], np.real(sigma), label)
        plot_band(axes[1, 2], np.imag(sigma), label)

    axes[0, 2].set_title("sample n"); axes[0, 2].set_ylabel("n"); axes[0, 2].legend(fontsize=7)
    axes[1, 0].set_title("sample k"); axes[1, 0].set_ylabel("k"); axes[1, 0].set_xlabel("THz")
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].set_title("sigma_real (S/m)"); axes[1, 1].set_ylabel("S/m"); axes[1, 1].set_xlabel("THz")
    axes[1, 1].legend(fontsize=7)
    axes[1, 2].set_title("sigma_imag (S/m)"); axes[1, 2].set_ylabel("S/m"); axes[1, 2].set_xlabel("THz")
    axes[1, 2].legend(fontsize=7)

    figure.savefig(output_path, dpi=130)
    print(f"Saved figure: {output_path}")


# ── Orchestration ─────────────────────────────────────────────────────────────


def parse_gap_thickness_m() -> float:
    """Read the air-gap dial from the command line (--gap-um), defaulting to the constant."""
    parser = argparse.ArgumentParser(description="Dial the empty-cuvette air gap for the FP correction.")
    parser.add_argument(
        "--gap-um", type=float, default=REFERENCE_GAP_THICKNESS_M * 1e6,
        help="Empty-cuvette air-gap thickness in micrometres (the FP cavity to dial).",
    )
    arguments, _ = parser.parse_known_args()
    return arguments.gap_um * 1e-6


def main() -> None:
    gap_thickness_m = parse_gap_thickness_m()

    shared = build_shared_transfer_functions()
    frequency = shared["frequency"]
    mask = shared["band_mask"]

    # One gated substrate extraction → n_sub fed identically to both sample methods.
    substrate_n, substrate_k = invert_thzcore_grid_substrate(
        frequency, shared["transfer_substrate"], mask
    )
    substrate_n_filled = _fill_band(substrate_n, mask)
    substrate_k_filled = _fill_band(substrate_k, mask)

    sample_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sample_results["thz_core ASMSA (FP on)"] = invert_thzcore_grid_sample_fp_on(
        frequency, shared["transfer_sample"], mask,
        substrate_n_filled, substrate_k_filled, gap_thickness_m,
    )
    sample_results["MATLAB-port (FP on)"] = invert_matlab_port_sample_fp_on(
        frequency, shared["transfer_sample"], mask, substrate_n_filled, gap_thickness_m,
    )

    print_report(shared, gap_thickness_m, sample_results, substrate_n_filled)

    gap_tag = f"{gap_thickness_m*1e6:.0f}um"
    figure_path = os.path.join(
        os.path.dirname(__file__), f"dial_fp_gap_{gap_tag}.png"
    )
    make_figure(shared, gap_thickness_m, sample_results, substrate_n_filled, figure_path)


if __name__ == "__main__":
    main()
