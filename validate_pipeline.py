"""Visual validation of the full thz_core pipeline.

Runs reference (N2 gas) and sample (silicon, ~310-320 um)
through every processing stage and generates diagnostic plots.

Pipeline flow
~~~~~~~~~~~~~
1. ``subtract_baseline`` — remove DC offset from each trace
   by averaging the first N leading samples.
2. ``align_on_peak`` — center all traces on their main
   pulse peak, then crop to the global overlap so every
   trace shares the same array length.
3. ``window_time`` — apply the same Tukey taper to each.
4. ``zero_pad`` — build a shared time grid spanning the
   union of all time ranges (+ optional trailing zeros
   for frequency resolution).
5. ``fft_spectrum`` -> ``transfer_function`` ->
   ``trusted_band_mask`` -> ``invert_nk`` ->
   ``derive_eps_sigma``.

Usage:
    python examples/validate_pipeline.py

Requires:
    matplotlib (pip install matplotlib)
"""

from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(
    str(pathlib.Path(__file__).resolve().parent.parent)
)

from thz_core import (
    align_on_peak,
    derive_eps_sigma,
    fft_spectrum,
    invert_nk,
    subtract_baseline,
    transfer_function,
    trusted_band_mask,
    window_time,
    zero_pad,
)

# ── Data paths ──────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent
REF_PATH = r"C:\Users\Samuel\Data\Chris\reference_RT.txt"
SAMP_PATH = r"C:\Users\Samuel\Data\Chris\sample_RT.txt"

# ── Physical parameters ─────────────────────────────────
THICKNESS_M = 315e-6  # metres (Silicon, 310-320 um)
PS_TO_S = 1e-12

# ── Configuration ───────────────────────────────────────
CONFIG: dict = {
    "window": {
        "type": "tukey",
        "alpha": 0.3,
        "zero_tail": True,
    },
    "pad": {
        "extend_factor": 4.0,
    },
    "fft": {
        "norm": "backward",
        "amplitude_scale": 1.0,
    },
    "transfer": {
        "min_ref_amp_rel": 1e-3,
        "regularization_eps": 1e-30,
        "unwrap_phase": True,
    },
    "mask": {
        "snr_thresh_db": 6.0,
        "tail_fraction": 0.25,
        "min_contiguous_bins": 3,
    },
    "invert": {
        "max_iterations": 30,
        "convergence_tol": 1e-12,
    },
}

# auto_range for batch mode (index range for peak search)
# Set to None for interactive SpanSelector UI.
AUTO_RANGE = (47, 57)


# ═══════════════════════════════════════════════════════
#  PIPELINE STEP FUNCTIONS
# ═══════════════════════════════════════════════════════

def load_raw_data() -> dict:
    """Load reference and sample from text files.

    Returns
    -------
    dict
        ``data_dict`` mapping labels to (N, 2) arrays
        with columns [time_s, amplitude].
    """
    ref = np.loadtxt(REF_PATH)
    samp = np.loadtxt(SAMP_PATH)
    data_dict = {
        "reference": np.column_stack((
            ref[:, 0] * PS_TO_S, ref[:, 1],
        )),
        "sample": np.column_stack((
            samp[:, 0] * PS_TO_S, samp[:, 1],
        )),
    }
    for label, arr in data_dict.items():
        time_col = arr[:, 0]
        dt_ps = np.median(np.diff(time_col)) * 1e12
        print(
            f"Loaded {label:12s}: {time_col.size} pts, "
            f"dt = {dt_ps:.4f} ps, "
            f"range = [{time_col[0]*1e12:.2f}, "
            f"{time_col[-1]*1e12:.2f}] ps"
        )
    return data_dict


def step_baseline(
    data_dict: dict,
    config: dict = CONFIG,
) -> dict:
    """Subtract DC baseline from every trace.

    Returns a new ``data_dict`` with the baseline removed
    from each amplitude column.
    """
    corrected, bl_metrics = subtract_baseline(
        data_dict, config,
    )
    print_metrics("Baseline subtraction", bl_metrics)
    return corrected


def step_align(
    data_dict: dict,
    auto_range: tuple | None = AUTO_RANGE,
) -> dict:
    """Center all traces on their main pulse peak.

    Returns the cropped ``data_dict`` where all arrays
    share the same length and peaks are aligned.
    """
    return align_on_peak(data_dict, auto_range=auto_range)


def step_window(
    data_dict: dict,
    config: dict = CONFIG,
) -> dict:
    """Apply temporal windowing to every trace.

    Returns a new ``data_dict`` with windowed amplitudes
    (time columns preserved).
    """
    windowed_dict: dict = {}
    for label, arr in data_dict.items():
        time_col = arr[:, 0]
        amp_col = arr[:, 1]
        windowed_amp, win_metrics = window_time(
            time_col, amp_col, config,
        )
        windowed_dict[label] = np.column_stack((
            time_col, windowed_amp,
        ))
        print_metrics(f"Window ({label})", win_metrics)
    return windowed_dict


def step_zero_pad(
    data_dict: dict,
    config: dict = CONFIG,
) -> Tuple[np.ndarray, dict]:
    """Zero-pad all traces onto a shared time grid.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(t_common, padded_dict)`` where
        ``padded_dict`` maps labels to padded amplitude
        arrays.
    """
    t_common, padded_dict, pad_metrics = zero_pad(
        data_dict, config,
    )
    print_metrics("Zero pad", pad_metrics)
    return t_common, padded_dict


def step_fft(
    t_common: np.ndarray,
    padded_dict: dict,
    config: dict = CONFIG,
) -> Tuple[np.ndarray, dict]:
    """FFT each padded trace.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(freq, spectra_dict)`` where ``spectra_dict``
        maps labels to complex spectra.
    """
    spectra: dict = {}
    freq = None
    for label, padded_amp in padded_dict.items():
        freq_arr, spectrum, fft_metrics = fft_spectrum(
            t_common, padded_amp, config,
        )
        freq = freq_arr
        spectra[label] = spectrum
        print_metrics(f"FFT ({label})", fft_metrics)
    return freq, spectra


def step_transfer(
    freq: np.ndarray,
    spectra: dict,
    ref_key: str = "reference",
    samp_key: str = "sample",
    config: dict = CONFIG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute transfer function H = samp / ref.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(h_complex, mask)`` — complex transfer function
        and trusted-band boolean mask.
    """
    y_ref_f = spectra[ref_key]
    y_samp_f = spectra[samp_key]

    h_complex, finite_mask, tf_metrics = (
        transfer_function(
            freq, y_samp_f, y_ref_f, config,
        )
    )
    print_metrics("Transfer function", tf_metrics)

    mask, mask_metrics = trusted_band_mask(
        freq, y_ref_f, y_samp_f, h_complex, config,
    )
    print_metrics("Trusted-band mask", mask_metrics)

    return h_complex, mask


def step_invert(
    freq: np.ndarray,
    h_complex: np.ndarray,
    mask: np.ndarray,
    thickness_m: float = THICKNESS_M,
    config: dict = CONFIG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract optical constants n, k from H(w).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(n_out, k_out)``
    """
    n_out, k_out, inv_metrics = invert_nk(
        freq, h_complex, thickness_m, mask, config,
    )
    print_metrics("Inversion", inv_metrics)
    return n_out, k_out


def step_derive(
    freq: np.ndarray,
    n_out: np.ndarray,
    k_out: np.ndarray,
    config: dict = CONFIG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive eps and sigma from n, k.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(eps, sigma)``
    """
    eps, sigma, der_metrics = derive_eps_sigma(
        freq, n_out, k_out, config,
    )
    print_metrics("Derive (eps, sigma)", der_metrics)
    return eps, sigma


# ═══════════════════════════════════════════════════════
#  PRINTING / SUMMARY HELPERS
# ═══════════════════════════════════════════════════════

def print_metrics(label: str, metrics: dict) -> None:
    """Print stage metrics to console."""
    print(f"\n{'=' * 54}")
    print(f"  {label}")
    print(f"{'=' * 54}")
    vals = metrics.get("values", {})
    for key, val in vals.items():
        if isinstance(val, np.ndarray):
            print(
                f"  {key:30s} : array({val.shape})"
            )
        elif isinstance(val, dict):
            print(f"  {key:30s} :")
            for sub_key, sub_val in val.items():
                print(
                    f"    {sub_key:28s} : {sub_val}"
                )
        else:
            print(f"  {key:30s} : {val}")
    flags = metrics.get("flags", {})
    if flags:
        for key, val in flags.items():
            flag_str = "!! FLAGGED" if val else "   ok"
            print(f"  {key:30s} : {flag_str}")


def print_summary(
    freq: np.ndarray,
    n_out: np.ndarray,
    k_out: np.ndarray,
    eps: np.ndarray,
    mask: np.ndarray,
    raw_data: dict,
) -> dict:
    """Print numerical summary and return the band mask.

    Returns
    -------
    dict
        Contains ``band``, ``n_band``, ``freq_thz``.
    """
    freq_thz = freq * 1e-12
    valid = mask & np.isfinite(n_out) & (freq > 0)
    band_lo_thz, band_hi_thz = 0.2, 4.5
    band = (
        valid
        & (freq_thz >= band_lo_thz)
        & (freq_thz <= band_hi_thz)
    )
    n_band = np.count_nonzero(band)
    print(
        f"\nTrusted-band bins in "
        f"[{band_lo_thz}-{band_hi_thz}] THz: {n_band}"
    )
    if n_band > 0:
        n_med = float(np.median(n_out[band]))
        k_med = float(np.median(k_out[band]))
        eps_med = float(np.median(np.real(eps[band])))
        print(
            f"  n median = {n_med:.3f}  "
            f"(literature ~3.42)"
        )
        print(
            f"  k median = {k_med:.5f}  "
            f"(literature ~0)"
        )
        print(
            f"  eps' median = {eps_med:.2f}  "
            f"(literature ~11.7)"
        )

    ref_arr = raw_data["reference"]
    samp_arr = raw_data["sample"]
    print(
        "\n--- Data notes ---\n"
        "Pipeline: subtract_baseline -> align_on_peak -> "
        "window_time -> zero_pad -> fft_spectrum\n"
        f"  ref:  {ref_arr[0,0]*1e12:.2f} - "
        f"{ref_arr[-1,0]*1e12:.2f} ps\n"
        f"  samp: {samp_arr[0,0]*1e12:.2f} - "
        f"{samp_arr[-1,0]*1e12:.2f} ps\n"
        "All traces centered on peak, windowed, then\n"
        "placed on a shared zero-padded grid.\n"
        "Material delay is encoded in the offset\n"
        "between trace positions on the grid."
    )
    return {
        "band": band,
        "n_band": n_band,
        "freq_thz": freq_thz,
    }


# ═══════════════════════════════════════════════════════
#  PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════

def plot_raw_waveforms(
    ax: plt.Axes,
    raw_data: dict,
) -> None:
    """Panel (a): raw waveforms on original time axes."""
    for label, arr in raw_data.items():
        ax.plot(
            arr[:, 0] * 1e12, arr[:, 1],
            label=label, linewidth=0.8,
        )
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title("(a) Raw waveforms")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_padded_waveforms(
    ax: plt.Axes,
    t_common: np.ndarray,
    padded_dict: dict,
) -> None:
    """Panel (b): windowed waveforms on shared grid."""
    time_ps = t_common * 1e12
    for label, padded in padded_dict.items():
        ax.plot(
            time_ps, padded,
            label=f"{label} (padded)",
            linewidth=0.6, alpha=0.6,
        )
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title("(b) Common grid & windowed waveforms")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_spectra_magnitude(
    ax: plt.Axes,
    freq: np.ndarray,
    spectra: dict,
) -> None:
    """Panel (c): frequency spectra magnitude in dB."""
    freq_thz = freq * 1e-12
    all_peaks = [
        np.max(np.abs(spec)) for spec in spectra.values()
    ]
    ref_peak = max(all_peaks) + 1e-30
    for label, spec in spectra.items():
        mag_db = 20.0 * np.log10(
            np.abs(spec) / ref_peak + 1e-30,
        )
        ax.plot(
            freq_thz, mag_db,
            label=label, linewidth=0.8,
        )
    ax.axhspan(
        -80, -60, alpha=0.08, color="red",
        label="Noise floor region",
    )
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Magnitude (dB rel. peak)")
    ax.set_title("(c) Frequency spectra")
    ax.set_xlim(0, 6)
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_phase(
    ax: plt.Axes,
    freq: np.ndarray,
    spectra: dict,
    h_complex: np.ndarray,
) -> None:
    """Panel (d): unwrapped phase spectra."""
    freq_thz = freq * 1e-12
    ref_spec = list(spectra.values())[0]
    ref_peak = np.max(np.abs(ref_spec)) + 1e-30

    for label, spec in spectra.items():
        ax.plot(
            freq_thz, np.unwrap(np.angle(spec)),
            label=f"{label} phase",
            linewidth=0.8, alpha=0.5,
        )

    valid_f = np.abs(ref_spec) > ref_peak * 1e-3
    phase_h = np.full(freq.size, np.nan)
    phase_h[valid_f] = np.unwrap(
        np.angle(h_complex[valid_f]),
    )
    ax.plot(
        freq_thz, phase_h,
        label="H(\u03c9) phase",
        linewidth=1.2, color="black",
    )
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Phase (rad)")
    ax.set_title("(d) Unwrapped phase spectra")
    ax.set_xlim(0, 6)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_transfer_magnitude(
    ax: plt.Axes,
    freq: np.ndarray,
    h_complex: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Panel (e): transfer function magnitude."""
    freq_thz = freq * 1e-12
    h_mag = np.abs(h_complex)
    ax.plot(freq_thz, h_mag, linewidth=0.8, color="C2")
    ax.fill_between(
        freq_thz, 0, h_mag,
        where=mask, alpha=0.15, color="green",
        label="Trusted band",
    )
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("|H(\u03c9)|")
    ax.set_title("(e) Transfer function magnitude")
    ax.set_xlim(0, 6)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_refractive_index(
    ax: plt.Axes,
    freq_thz: np.ndarray,
    n_out: np.ndarray,
    band: np.ndarray,
) -> None:
    """Panel (f): refractive index in trusted band."""
    if np.count_nonzero(band) > 0:
        ax.plot(
            freq_thz[band], n_out[band],
            linewidth=1.0, color="C0",
            label="n(\u03c9)",
        )
        ax.plot(
            freq_thz[band], n_out[band],
            linewidth=1.0, color="C0",
            label="n(\u03c9)",
        )
    ax.axhline(
        3.42, color="red", linestyle="--",
        linewidth=0.8, label="Si literature (3.42)",
    )
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Refractive index n")
    ax.set_title(
        "(f) Refractive index \u2014 trusted band"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_extinction(
    ax: plt.Axes,
    freq_thz: np.ndarray,
    k_out: np.ndarray,
    band: np.ndarray,
) -> None:
    """Panel (g): extinction coefficient in trusted band."""
    if np.count_nonzero(band) > 0:
        ax.plot(
            freq_thz[band], k_out[band],
            linewidth=1.0, color="C1",
            label="k(\u03c9)",
        )
    ax.axhline(
        0.0, color="gray", linestyle="--",
        linewidth=0.6,
    )
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Extinction coefficient k")
    ax.set_title(
        "(g) Extinction coefficient \u2014 trusted band"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_permittivity(
    ax: plt.Axes,
    freq_thz: np.ndarray,
    eps: np.ndarray,
    band: np.ndarray,
) -> None:
    """Panel (h): real permittivity in trusted band."""
    if np.count_nonzero(band) > 0:
        ax.plot(
            freq_thz[band], np.real(eps[band]),
            linewidth=1.0, color="C3",
            label="\u03b5'(\u03c9) (real part)",
        )        
        ax.plot(
            freq_thz[band], np.imag(eps[band]),
            linewidth=1.0, color="C4",
            label="\u03b5''(\u03c9) (imaginary part)",
        )
    ax.axhline(
        11.7, color="red", linestyle="--",
        linewidth=0.8,
        label="Si literature (11.7)",
    )
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("\u03b5' (permittivity)")
    ax.set_title(
        "(h) Permittivity \u2014 trusted band"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_conductivity(
    ax: plt.Axes,
    freq_thz: np.ndarray,
    eps: np.ndarray,
    band: np.ndarray,
) -> None:
    """Panel (h): conductivity in trusted band."""
    if np.count_nonzero(band) > 0:
        ax.plot(
            freq_thz[band], np.real(eps[band]) * 2 * np.pi * freq_thz[band] * 8.854e-12,
            linewidth=1.0, color="C5",
            label="\u03c3(\u03c9) (real conductivity)",
        )        
        ax.plot(
            freq_thz[band], np.imag(eps[band]) * 2 * np.pi * freq_thz[band] * 8.854e-12,
            linewidth=1.0, color="C6",
            label="\u03c3(\u03c9) (imaginary conductivity)",
        )
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("\u03c3 (conductivity, S/m)")
    ax.set_title(
        "(h) Conductivity \u2014 trusted band"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def make_diagnostic_figure(
    raw_data: dict,
    t_common: np.ndarray,
    padded_dict: dict,
    freq: np.ndarray,
    spectra: dict,
    h_complex: np.ndarray,
    mask: np.ndarray,
    n_out: np.ndarray,
    k_out: np.ndarray,
    eps: np.ndarray,
    summary: dict,
) -> plt.Figure:
    """Create the 4x2 diagnostic figure.

    Returns the figure object for saving / display.
    """
    band = summary["band"]
    freq_thz = summary["freq_thz"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(
        "THz-TDS Pipeline Validation  \u2014  Silicon "
        f"({THICKNESS_M*1e6:.0f} \u00b5m)",
        fontsize=14, fontweight="bold",
    )

    plot_raw_waveforms(axes[0, 0], raw_data)
    plot_padded_waveforms(
        axes[0, 1], t_common, padded_dict,
    )
    plot_spectra_magnitude(axes[1, 0], freq, spectra)
    plot_phase(axes[1, 1], freq, spectra, h_complex)
    plot_transfer_magnitude(
        axes[2, 0], freq, h_complex, mask,
    )
    plot_refractive_index(
        axes[2, 1], freq_thz, n_out, band,
    )
    plot_extinction(axes[2, 1], freq_thz, k_out, band)
    plot_permittivity(axes[3, 0], freq_thz, eps, band)
    plot_conductivity(axes[3, 1], freq_thz, eps, band)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ═══════════════════════════════════════════════════════
#  EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════

def _side_by_side_tsv(
    column_groups: list[list[np.ndarray]],
    header_rows: list[list[str]],
) -> str:
    """Build a TSV string from column groups of potentially different lengths.

    Parameters
    ----------
    column_groups : list[list[np.ndarray]]
        Each group is a list of 1-D arrays (columns).
        Groups may have different row counts.
    header_rows : list[list[str]]
        Each element is a flat list of header cells for
        one header row, spanning all groups.
    """
    max_rows = max(
        col.size for group in column_groups for col in group
    )
    lines = ["\t".join(row) for row in header_rows]
    for i in range(max_rows):
        cells: list[str] = []
        for group in column_groups:
            for col in group:
                if i < col.size:
                    val = col[i]
                    if isinstance(val, (float, np.floating)):
                        cells.append(str(val))
                    else:
                        cells.append(str(val))
                else:
                    cells.append("")
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def export_time_domain(
    path: pathlib.Path,
    raw_data: dict,
    windowed_data: dict,
    ref_key: str = "reference",
    samp_key: str = "sample",
) -> None:
    """Export raw + windowed time-domain data to a TSV file.

    Columns per trace: Time (ps), Mean, std error.
    """
    def _cols(arr: np.ndarray) -> list[np.ndarray]:
        time_ps = arr[:, 0] * 1e12
        amp = arr[:, 1]
        err = np.full(amp.size, np.nan)
        return [time_ps, amp, err]

    raw_ref = _cols(raw_data[ref_key])
    win_ref = _cols(windowed_data[ref_key])
    raw_samp = _cols(raw_data[samp_key])
    win_samp = _cols(windowed_data[samp_key])

    groups = [raw_ref, win_ref, raw_samp, win_samp]

    label_row = (
        ["Reference"] * 3
        + ["windowed Reference"] * 3
        + ["Sample"] * 3
        + ["windowed Sample"] * 3
    )
    col_row = ["Time (ps)", "Mean", "std error"] * 4

    text = _side_by_side_tsv(groups, [label_row, col_row])
    # Replace nan with empty string for optional columns
    text = text.replace("nan", "")
    path.write_text(text, encoding="utf-8")
    print(f"Exported time-domain data to {path}")


def export_fft(
    path: pathlib.Path,
    freq: np.ndarray,
    spectra: dict,
    ref_key: str = "reference",
    samp_key: str = "sample",
) -> None:
    """Export FFT results to a TSV file.

    Columns per trace: Frequency (THz), Amplitude,
    Δ(Amplitude), Phase, Δ(Phase).
    """
    freq_thz = freq * 1e-12

    def _cols(spectrum: np.ndarray) -> list[np.ndarray]:
        amp = np.abs(spectrum)
        amp_err = np.full(amp.size, np.nan)
        phase = np.unwrap(np.angle(spectrum))
        phase_err = np.full(amp.size, np.nan)
        return [freq_thz, amp, amp_err, phase, phase_err]

    ref_cols = _cols(spectra[ref_key])
    samp_cols = _cols(spectra[samp_key])

    groups = [ref_cols, samp_cols]

    label_row = ["Reference"] * 5 + ["Sample"] * 5
    col_row = [
        "Frequency (THz)", "Amplitude",
        "\u0394(Amplitude)", "Phase", "\u0394(Phase)",
    ] * 2

    text = _side_by_side_tsv(groups, [label_row, col_row])
    text = text.replace("nan", "")
    path.write_text(text, encoding="utf-8")
    print(f"Exported FFT data to {path}")


def export_optical_constants(
    path: pathlib.Path,
    freq: np.ndarray,
    n_out: np.ndarray,
    k_out: np.ndarray,
    eps: np.ndarray,
    sigma: np.ndarray,
) -> None:
    """Export optical constants to a TSV file.

    Columns: Frequency, n, k, ε1, ε2, σRe, σIm.
    """
    freq_thz = freq * 1e-12
    eps_real = np.real(eps)
    eps_imag = np.imag(eps)
    sigma_real = np.real(sigma)
    sigma_imag = np.imag(sigma)

    header = "Frequency\tn\tk\t\u03b51\t\u03b52\t\u03c3Re\t\u03c3Im"
    cols = np.column_stack((
        freq_thz, n_out, k_out,
        eps_real, eps_imag, sigma_real, sigma_imag,
    ))

    lines = [header]
    for row in cols:
        cells = []
        for val in row:
            if np.isfinite(val):
                cells.append(str(val))
            else:
                cells.append("")
        lines.append("\t".join(cells))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Exported optical constants to {path}")


def export_all(
    output_dir: pathlib.Path,
    raw_data: dict,
    windowed_data: dict,
    freq: np.ndarray,
    spectra: dict,
    n_out: np.ndarray,
    k_out: np.ndarray,
    eps: np.ndarray,
    sigma: np.ndarray,
    ref_key: str = "reference",
    samp_key: str = "sample",
) -> None:
    """Export all pipeline results to TSV files.

    Creates three files in *output_dir*:
    - ``time-domain.txt``
    - ``fft-out.txt``
    - ``optical-constants.txt``
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    export_time_domain(
        output_dir / "sam_time-domain.txt",
        raw_data, windowed_data, ref_key, samp_key,
    )
    export_fft(
        output_dir / "sam_fft-out.txt",
        freq, spectra, ref_key, samp_key,
    )
    export_optical_constants(
        output_dir / "sam_optical-constants.txt",
        freq, n_out, k_out, eps, sigma,
    )


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def main() -> None:
    """Execute full pipeline and display plots.

    Comment out individual step_* calls below to examine
    the effect of skipping that processing stage.
    """
    # 1. Load
    raw_data = load_raw_data()

    # 2. Baseline subtraction (comment out to skip)
    baselined = step_baseline(raw_data)

    # 3. Align on peak (comment out to skip centering)
    aligned = step_align(baselined, auto_range=(40, 60)) # Set auto_range=None to enable interactive peak selection.
    # aligned = step_align(baselined, auto_range=AUTO_RANGE)

    # 4. Window (comment out to skip windowing)
    windowed = step_window(aligned)

    # 5. Zero-pad (comment out to skip padding)
    t_common, padded_dict = step_zero_pad(windowed)

    # 6. FFT
    freq, spectra = step_fft(t_common, padded_dict)

    # 7. Transfer function + trusted-band mask
    h_complex, mask = step_transfer(freq, spectra)

    # 8. Inversion
    n_out, k_out = step_invert(freq, h_complex, mask)

    # 9. Derived properties
    eps, sigma = step_derive(freq, n_out, k_out)

    # 10. Summary
    summary = print_summary(
        freq, n_out, k_out, eps, mask, raw_data,
    )

    # 11. Plots
    fig = make_diagnostic_figure(
        raw_data, t_common, padded_dict,
        freq, spectra, h_complex, mask,
        n_out, k_out, eps, summary,
    )
    out_path = HERE / "pipeline_validation.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to {out_path}")

    ## 12. Export data files
    # export_all(
    #     output_dir=HERE / "output",
    #     raw_data=raw_data,
    #     windowed_data=windowed,
    #     freq=freq,
    #     spectra=spectra,
    #     n_out=n_out,
    #     k_out=k_out,
    #     eps=eps,
    #     sigma=sigma,
    # )

    plt.show()


if __name__ == "__main__":
    main()
