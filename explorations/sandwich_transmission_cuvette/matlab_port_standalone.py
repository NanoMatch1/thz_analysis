"""Standalone MATLAB-port sandwich-transmission inversion (user-friendly edition).

WHAT THIS SCRIPT DOES
=====================
It takes THz time-domain traces from a "sandwich" (cuvette) transmission experiment and
extracts, for the filled sample layer, the complex refractive index (n + i*k) and the
complex optical conductivity sigma(omega).  The geometry is:

        air | window | sample | window | air        (the filled cuvette)
        air | window |  gap   | window | air        (the empty cuvette, the reference)

Three traces are needed per measurement:
    * air       : beam through nothing (free space)            -> reference_*.dat
    * substrate : beam through the EMPTY cuvette (two windows)  -> Substrate_*.dat
    * sample    : beam through the FILLED cuvette               -> Sample_*.dat

The inversion is the faithful MATLAB port (the "cuvette_code" / "sample_code" models),
solved with a grid + continuity search.  It runs in two steps:
    Step 1 (substrate): from air -> substrate, recover the window index n_sub(omega).
    Step 2 (sample):    from substrate -> sample, recover the sample n + i*k, using n_sub.

The signal front-end (windowing, FFT, transfer function) reuses the validated `thz_core`
routines so the numbers match the maintained pipeline; only the *inversion math* is the
MATLAB port.

HOW TO USE IT (no Python knowledge required)
============================================
1. Edit the CONFIG block below.  Every knob you might touch lives there, with a comment.
2. To process MANY samples or MANY temperatures, just add more rows to
   CONFIG["measurement_sets"] -- one row per air/substrate/sample triplet.
3. To turn the Fabry-Perot (etalon) correction on or off, set
   CONFIG["fabry_perot"]["enabled"] = True or False.
4. Run it:
       PYTHONPATH=. ../.venv/Scripts/python.exe \
           explorations/sandwich_transmission_cuvette/matlab_port_standalone.py
   (run from the repo root so `import thz_core` resolves).

It prints a short table per measurement set and saves, next to this script, a figure and a
results file (.pkl) per set.
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass

import numpy as np
from scipy import constants as physical_constants

# Make `import thz_core` work when run from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import thz_core.thz_core as core

try:  # the report prints maths symbols; force UTF-8 on a cp1252 (Windows) console.
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  --  EDIT THIS BLOCK.  Everything you might change lives here.
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Where the data lives ──────────────────────────────────────────────────
    "data_directory": r"C:\Users\Samuel\Data\THz\Sam\13-11-25_Co-HHTP",

    # ── The measurements to process ───────────────────────────────────────────
    # One ROW per air/substrate/sample triplet.  Add as many rows as you like to
    # process several samples or several temperatures in a single run.  "label" is
    # just a name used in the printout, figure title, and output filenames.
    # Prefer the ".acc" files: they hold every raw acquisition, so the script keeps
    # the statistical power of the repeats (a ".dat" is just the average, one scan).
    "measurement_sets": [
        {
            "label": "150K",
            "air":       "reference_air.acc",
            "substrate": "reference_substrate_150K.acc",
            "sample":    "sample_Co-HHTP_150K.acc",
        },
        # Add more temperatures the same way, e.g.:
        # {
        #     "label": "225K",
        #     "air":       "reference_air.acc",
        #     "substrate": "reference_substrate_225K.acc",
        #     "sample":    "sample_Co-HHTP_225K.acc",
        # },
    ],

    # ── Geometry (the physical thicknesses of the cuvette) ────────────────────
    # Each fused-silica window, in micrometres.  The empty-cuvette beam crosses
    # BOTH windows, so the substrate step models two of these.
    "window_thickness_um": 900.0,        # 0.9 mm per window
    # The inter-window air gap seen in the EMPTY cuvette substrate step, in um.
    # This cancels when the substrate is gated (Fabry-Perot OFF), so its exact
    # value barely matters for n_sub.
    "substrate_gap_um": 90.0,
    # The sample layer thickness = the cuvette spacer, in micrometres.  This sets
    # the sample propagation phase AND (when Fabry-Perot is on) the etalon spacing.
    "sample_layer_thickness_um": 130.0,
    # Refractive index of the surrounding medium and the empty gap (vacuum/air = 1).
    "ambient_index": 1.0,

    # ── Fabry-Perot (etalon) correction  --  THE ON/OFF TOGGLE ────────────────
    # The thin air gap in the empty cuvette is a real, NON-gatable etalon (~10%
    # ripple), so turning this ON is usually the physically correct choice for the
    # SAMPLE step.  The etalon spacing is set by "sample_layer_thickness_um" above
    # (one physical spacer fills the cuvette everywhere).
    "fabry_perot": {
        "enabled": True,
    },

    # ── Frequency band to report / invert over (Hz) ───────────────────────────
    "report_frequency_min_hz": 0.8e12,
    "report_frequency_max_hz": 2.0e12,

    # ── FFT zero-padding ──────────────────────────────────────────────────────
    # Trailing zeros only SMOOTH (interpolate) the spectrum; because the pulse is
    # already windowed to ~0 at the record end, no spurious time shift is added.
    "fft_length": 4096,

    # ── Substrate-step grid search (recovers the window index n_sub) ──────────
    # Fused silica is ~1.95, so the search brackets that.  Substrate is GATED
    # (Fabry-Perot off) here because the mm-scale windows are gatable.
    "substrate_n_bounds": (1.88, 2.02),
    "substrate_k_bound": 0.03,
    "substrate_continuity_half_width": 20,

    # ── Sample-step grid search (recovers the sample n + i*k) ─────────────────
    "sample_n_bounds": (1.0, 2.5),
    "sample_k_bound": 0.4,               # ceiling on |k|; raise if k rails against it
    "sample_continuity_half_width": 15,

    # Number of candidate points along each axis of the n/k search grid.  Higher =
    # finer resolution but slower.  500 x 500 is a good default.
    "grid_point_count": 500,

    # ── Conductivity reference ────────────────────────────────────────────────
    # eps_background subtracted before computing sigma; 1.0 = vacuum = total
    # optical conductivity.
    "eps_background": 1.0,

    # ── Output ────────────────────────────────────────────────────────────────
    "save_figure": True,
    "save_results_pickle": True,
    "save_conductivity_dat": True,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Below here is the machinery.  You shouldn't need to edit it to change settings
#  -- but it's written step-by-step so you can SEE what each stage does.
# ══════════════════════════════════════════════════════════════════════════════

SPEED_OF_LIGHT_M_PER_S = physical_constants.c


# ── Step 0: load the raw traces into a lightweight container ───────────────────
# The instrument writes two files per measurement:
#   * a .acc file = EVERY individual acquisition (scan), one after another, each
#     separated by a "%%" line.  This is the raw, unaveraged data.
#   * a .dat file = the single AVERAGE of those acquisitions (one trace).
# A .acc file therefore contains both: the raw scans AND (by averaging them) the
# mean.  We load the .acc when we can, so we keep the statistical power of the
# repeats; a .dat works too but then there is only one (already-averaged) scan.


@dataclass
class AcquisitionTrace:
    """One measured trace: the time axis + every raw acquisition + their average.

    - .time_seconds      : 1D time axis, SI seconds.
    - .raw_acquisitions  : 2D array, shape (n_time, n_acquisitions) -- the raw scans.
    - .averaged          : 1D mean across acquisitions.  USE THIS for the inversion.
    - .standard_error    : 1D uncertainty on .averaged (scatter across scans / sqrt N).

    Rule of thumb: use ``.averaged`` for all normal processing; reach into
    ``.raw_acquisitions`` only when you want statistics (e.g. error bars from the
    spread across the repeated scans).
    """

    time_seconds: np.ndarray
    raw_acquisitions: np.ndarray   # (n_time, n_acquisitions)

    @property
    def averaged(self) -> np.ndarray:
        """Mean field across all acquisitions -- the working trace."""
        return np.mean(self.raw_acquisitions, axis=1)

    @property
    def n_acquisitions(self) -> int:
        """How many raw scans were averaged."""
        return self.raw_acquisitions.shape[1]

    @property
    def standard_error(self) -> np.ndarray:
        """Uncertainty on .averaged: scatter across scans / sqrt(number of scans)."""
        if self.n_acquisitions < 2:
            return np.zeros(self.time_seconds.size)
        spread = np.std(self.raw_acquisitions, axis=1, ddof=1)
        return spread / np.sqrt(self.n_acquisitions)


@dataclass
class MeasurementSet:
    """One air/substrate/sample triplet, loaded and held in memory.

    Lightweight stand-in for the heavy DataSet class: it just holds the three
    measured traces.  Each trace is an AcquisitionTrace (raw scans + average), so
    the statistical power of the repeats is preserved.  The three traces may have
    their own time axes; the front-end pads them to a common FFT length, so they
    only need to share the same time STEP (which they do for one instrument).
    """

    label: str
    air: AcquisitionTrace
    substrate: AcquisitionTrace
    sample: AcquisitionTrace


def _parse_two_column_blocks(file_text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse instrument text into (time_seconds, raw_acquisitions).

    The file is split on "%%" into blocks (one per acquisition).  Header lines
    start with "%" and are ignored; data lines are "time_ps  field".  A .dat file
    has no "%%" so it parses as a single block (one acquisition).
    """
    acquisition_columns: list[np.ndarray] = []
    time_picoseconds: np.ndarray | None = None

    for block in file_text.split("%%"):
        rows = []
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("%"):
                continue                       # skip blank + header lines
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue                       # skip any unparseable row
        if not rows:
            continue
        block_array = np.array(rows)
        if time_picoseconds is None:
            time_picoseconds = block_array[:, 0]
        acquisition_columns.append(block_array[:, 1])

    if time_picoseconds is None or not acquisition_columns:
        raise ValueError("No numeric data rows found in file.")

    # Guard against a truncated final scan: trim every column to the shortest length.
    shortest = min(time_picoseconds.size, min(column.size for column in acquisition_columns))
    time_seconds = time_picoseconds[:shortest] * 1e-12     # files store picoseconds
    raw_acquisitions = np.column_stack([column[:shortest] for column in acquisition_columns])
    return time_seconds, raw_acquisitions


def load_acquisition_trace(directory: str, filename: str) -> AcquisitionTrace:
    """Load one .acc (many scans) or .dat (one averaged scan) into an AcquisitionTrace.

    A per-acquisition DC baseline is removed (each scan minus its own leading
    samples).  Because averaging is linear, this leaves ``.averaged`` identical to
    baselining the average, while also baseline-correcting the raw scans for stats.
    """
    with open(os.path.join(directory, filename), "r") as data_file:
        file_text = data_file.read()
    time_seconds, raw_acquisitions = _parse_two_column_blocks(file_text)

    leading_points = 10
    baseline_per_scan = np.mean(raw_acquisitions[:leading_points, :], axis=0)
    raw_acquisitions = raw_acquisitions - baseline_per_scan[None, :]

    return AcquisitionTrace(time_seconds=time_seconds, raw_acquisitions=raw_acquisitions)


def load_all_measurement_sets(config: dict) -> list[MeasurementSet]:
    """Read every triplet listed in config["measurement_sets"] into memory.

    Returns a list of MeasurementSet objects -- one per row you put in the config.
    This is how the script handles MANY samples / temperatures: it just loops.
    """
    directory = config["data_directory"]
    loaded_sets: list[MeasurementSet] = []
    for row in config["measurement_sets"]:
        loaded_sets.append(
            MeasurementSet(
                label=row["label"],
                air=load_acquisition_trace(directory, row["air"]),
                substrate=load_acquisition_trace(directory, row["substrate"]),
                sample=load_acquisition_trace(directory, row["sample"]),
            )
        )
    return loaded_sets


# ── Step 1: turn time-domain traces into the two measured transfer functions ───
# This is the "front-end".  It reuses thz_core so the numbers match the validated
# pipeline.  The three sub-steps are: window the pulse, zero-pad + FFT, then ratio.


def _windowed_padded_spectrum(
    time_seconds: np.ndarray, field: np.ndarray, config: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Window the pulse in place, trailing zero-pad, FFT.  Returns (frequency, spectrum).

    Windowing IN PLACE keeps the pulse at its true position on the time axis, so the
    transfer-function phase carries the real inter-pulse delay (no padding ramp).
    """
    # (a) apply a Tukey (tapered cosine) window so the spectrum is clean.
    window_config = {"window": {"type": "tukey", "alpha": 1}}
    windowed_field, _, _ = core.window_time(time_seconds, field, window_config)

    # (b) append trailing zeros up to the requested FFT length (smooths the spectrum).
    timestep_seconds = float(np.mean(np.diff(time_seconds)))
    pad_count = max(0, config["fft_length"] - windowed_field.size)
    padded_field = np.concatenate([windowed_field, np.zeros(pad_count)])
    padded_time = time_seconds[0] + timestep_seconds * np.arange(padded_field.size)

    # (c) FFT to the frequency domain.
    frequency, spectrum, _ = core.fft_spectrum(padded_time, padded_field, {})
    return frequency, spectrum


def compute_transfer_functions(measurement: MeasurementSet, config: dict) -> dict:
    """Build the two measured transfer functions for one measurement set.

        H_substrate = E_substrate / E_air        (substrate step)
        H_sample    = E_sample    / E_substrate  (sample step)

    Returns a dict with the frequency axis, both transfer functions, and a boolean
    band mask selecting the report band.
    """
    # Use each trace's AVERAGED field (the mean across its raw acquisitions).
    frequency, air_spectrum = _windowed_padded_spectrum(
        measurement.air.time_seconds, measurement.air.averaged, config
    )
    _, substrate_spectrum = _windowed_padded_spectrum(
        measurement.substrate.time_seconds, measurement.substrate.averaged, config
    )
    _, sample_spectrum = _windowed_padded_spectrum(
        measurement.sample.time_seconds, measurement.sample.averaged, config
    )

    # SNR masking is left off here; we restrict to the report band explicitly below.
    transfer_config = {"transfer": {"apply_snr_mask": False}}
    transfer_substrate, _, _ = core.transfer_function(
        frequency, substrate_spectrum, air_spectrum, transfer_config
    )
    transfer_sample, _, _ = core.transfer_function(
        frequency, sample_spectrum, substrate_spectrum, transfer_config
    )

    band_mask = (
        (frequency >= config["report_frequency_min_hz"])
        & (frequency <= config["report_frequency_max_hz"])
    )
    return {
        "frequency": frequency,
        "band_mask": band_mask,
        "transfer_substrate": transfer_substrate,
        "transfer_sample": transfer_sample,
    }


# ── Step 2: the MATLAB-port inversion models (the physics) ─────────────────────
# These are the ported "cuvette_code" (substrate) and "sample_code" (sample)
# transfer-function MODELS.  Given a trial complex index, each predicts what the
# measured H should be; the solver (below) then searches for the index that best
# matches the data, frequency by frequency.


def matlab_substrate_model(
    trial_index_grid: np.ndarray, frequency_hz: float, config: dict, fabry_perot: bool
) -> np.ndarray:
    """Predict H_substrate (empty cuvette / air) for a grid of trial window indices.

    Geometry: air | window | gap | window | air, normalised by the air path.
    """
    ambient_index = config["ambient_index"]
    window_thickness_m = config["window_thickness_um"] * 1e-6
    gap_thickness_m = config["substrate_gap_um"] * 1e-6

    n_air = ambient_index
    n_sub = trial_index_grid
    omega = 2.0 * np.pi * frequency_hz
    c = SPEED_OF_LIGHT_M_PER_S

    # Fresnel transmission coefficients at each interface the pulse crosses.
    t12 = 2 * n_air / (n_air + n_sub)
    t23 = 2 * n_sub / (n_sub + n_air)
    t34 = 2 * n_air / (n_air + n_sub)
    t45 = 2 * n_sub / (n_sub + n_air)
    # Reflection coefficients bounding the inter-window gap (for the FP etalon).
    r23 = (n_air - n_sub) / (n_sub + n_air)
    r34 = (n_sub - n_air) / (n_air + n_sub)

    # Propagation phases through each layer.
    phase_window_in = np.exp(-1j * n_sub * omega * window_thickness_m / c)
    phase_gap = np.exp(-1j * n_air * omega * gap_thickness_m / c)
    phase_window_out = np.exp(-1j * n_sub * omega * window_thickness_m / c)
    # The air reference path (same physical length, no windows).
    phase_air_reference = np.exp(
        -1j * n_air * omega * (2 * window_thickness_m + gap_thickness_m) / c
    )

    # The etalon term: 1 when FP is off (gated), the textbook 1/(1 - r^2 P^2) when on.
    etalon = (1 - r23 * r34 * phase_gap ** 2) if fabry_perot else 1.0

    numerator = phase_window_in * phase_gap * phase_window_out * t12 * t23 * t34 * t45
    return numerator / (etalon * phase_air_reference)


def matlab_sample_model(
    trial_index_grid: np.ndarray,
    substrate_index: complex,
    frequency_hz: float,
    config: dict,
    fabry_perot: bool,
) -> np.ndarray:
    """Predict H_sample (filled / empty cuvette) for a grid of trial sample indices.

    The sample layer and the empty reference gap share one physical spacer
    thickness, so both use config["sample_layer_thickness_um"].  When FP is on,
    BOTH the sample layer and the air-gap reference carry an etalon term; the air
    gap's is the dominant correction.
    """
    spacer_thickness_m = config["sample_layer_thickness_um"] * 1e-6
    n_air = config["ambient_index"]
    n_sub = substrate_index
    n_sample = trial_index_grid
    omega = 2.0 * np.pi * frequency_hz
    c = SPEED_OF_LIGHT_M_PER_S

    # Filled-cuvette interfaces: substrate -> sample -> substrate.
    t23 = 2 * n_sub / (n_sub + n_sample)
    t34 = 2 * n_sample / (n_sample + n_sub)
    r23 = (n_sample - n_sub) / (n_sub + n_sample)
    r34 = (n_sub - n_sample) / (n_sample + n_sub)
    # Empty-cuvette reference interfaces: substrate -> air -> substrate.
    t23_reference = 2 * n_sub / (n_sub + n_air)
    t34_reference = 2 * n_air / (n_air + n_sub)
    r23_reference = (n_air - n_sub) / (n_sub + n_air)
    r34_reference = (n_sub - n_air) / (n_air + n_sub)

    # Propagation through the spacer: sample-filled vs air-filled.
    phase_sample = np.exp(-1j * n_sample * omega * spacer_thickness_m / c)
    phase_reference = np.exp(-1j * n_air * omega * spacer_thickness_m / c)

    # Etalon terms (textbook sign, written 1/(1 + r23*r34*P^2) with r34 = -r23).
    sample_etalon = (1 + r23 * r34 * phase_sample ** 2) if fabry_perot else 1.0
    reference_etalon = (
        (1 + r23_reference * r34_reference * phase_reference ** 2) if fabry_perot else 1.0
    )

    numerator = phase_sample * t23 * t34 / sample_etalon
    denominator = phase_reference * t23_reference * t34_reference / reference_etalon
    return numerator / denominator


# ── Step 2b: the grid + continuity solver (the search) ─────────────────────────


def _build_complex_index_grid(
    n_bounds: tuple[float, float], k_bound: float, point_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the n + i*k candidate grid (k <= 0, the MATLAB sign convention)."""
    n_axis = np.linspace(n_bounds[0], n_bounds[1], point_count)
    k_axis = np.linspace(0.0, -k_bound, point_count)
    index_grid = n_axis[:, None] + 1j * k_axis[None, :]
    return n_axis, k_axis, index_grid


def _pick_best_index_with_continuity(
    mismatch: np.ndarray,
    n_axis: np.ndarray,
    k_axis: np.ndarray,
    previous_index: complex | None,
    half_width: int,
) -> complex:
    """Pick the grid index that best matches the data at one frequency.

    First frequency: take the global best over the whole grid.  After that: only
    look in a small box around the previous frequency's answer, which keeps the
    solution smooth (continuous) and avoids jumping to a far-away false match.
    """
    if previous_index is None:
        flat_index = int(np.argmin(mismatch))
        row, col = np.unravel_index(flat_index, mismatch.shape)
        return complex(n_axis[row] + 1j * k_axis[col])

    center_row = int(np.argmin(np.abs(n_axis - np.real(previous_index))))
    center_col = int(np.argmin(np.abs(k_axis - np.imag(previous_index))))
    row_lo, row_hi = max(0, center_row - half_width), min(len(n_axis), center_row + half_width + 1)
    col_lo, col_hi = max(0, center_col - half_width), min(len(k_axis), center_col + half_width + 1)
    local = mismatch[row_lo:row_hi, col_lo:col_hi]
    local_row, local_col = np.unravel_index(int(np.argmin(local)), local.shape)
    return complex(n_axis[row_lo + local_row] + 1j * k_axis[col_lo + local_col])


def _solve_over_band(
    frequency: np.ndarray,
    transfer: np.ndarray,
    band_mask: np.ndarray,
    index_grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    half_width: int,
    model_at_frequency,
) -> np.ndarray:
    """Run the grid + continuity search across the report band.

    model_at_frequency(trial_index_grid, frequency_hz, band_position) must return the
    predicted H on the trial grid.  Returns a full-length complex array (n + i*k),
    NaN outside the band.
    """
    n_axis, k_axis, trial_index_grid = index_grid
    band_indices = np.where(band_mask)[0]

    result = np.full(frequency.size, np.nan + 1j * np.nan, dtype=complex)
    previous_index = None
    for band_position, full_index in enumerate(band_indices):
        frequency_hz = float(frequency[full_index])
        predicted = model_at_frequency(trial_index_grid, frequency_hz, band_position)
        measured = transfer[full_index]
        # Match on real AND imaginary parts (L1 mismatch, as in the MATLAB code).
        mismatch = np.abs(np.real(predicted - measured)) + np.abs(np.imag(predicted - measured))
        previous_index = _pick_best_index_with_continuity(
            mismatch, n_axis, k_axis, previous_index, half_width
        )
        result[full_index] = previous_index
    return result


def invert_substrate(transfer_functions: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Step 1: recover the window index n_sub from the substrate transfer function.

    Substrate is always GATED (Fabry-Perot off): the mm-scale windows are gatable,
    so the etalon term cancels.  Returns (n_sub, k_sub>=0), NaN outside the band.
    """
    frequency = transfer_functions["frequency"]
    grid = _build_complex_index_grid(
        config["substrate_n_bounds"], config["substrate_k_bound"], config["grid_point_count"]
    )

    def model_at_frequency(trial_index_grid, frequency_hz, _band_position):
        return matlab_substrate_model(trial_index_grid, frequency_hz, config, fabry_perot=False)

    complex_index = _solve_over_band(
        frequency,
        transfer_functions["transfer_substrate"],
        transfer_functions["band_mask"],
        grid,
        config["substrate_continuity_half_width"],
        model_at_frequency,
    )
    # k stored as a positive number (the MATLAB grid uses k <= 0 internally).
    return np.real(complex_index), -np.imag(complex_index)


def invert_sample(
    transfer_functions: dict, substrate_n: np.ndarray, config: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Step 2: recover the sample n + i*k, fed the substrate index per frequency.

    Honours the Fabry-Perot toggle in config["fabry_perot"]["enabled"].  Returns
    (n, k>=0), NaN outside the band.
    """
    frequency = transfer_functions["frequency"]
    band_mask = transfer_functions["band_mask"]
    fabry_perot = bool(config["fabry_perot"]["enabled"])
    grid = _build_complex_index_grid(
        config["sample_n_bounds"], config["sample_k_bound"], config["grid_point_count"]
    )

    # The sample model needs n_sub at each band frequency; build that lookup once.
    band_indices = np.where(band_mask)[0]
    substrate_n_in_band = substrate_n[band_indices]

    def model_at_frequency(trial_index_grid, frequency_hz, band_position):
        substrate_index_here = complex(substrate_n_in_band[band_position])
        return matlab_sample_model(
            trial_index_grid, substrate_index_here, frequency_hz, config, fabry_perot
        )

    complex_index = _solve_over_band(
        frequency,
        transfer_functions["transfer_sample"],
        band_mask,
        grid,
        config["sample_continuity_half_width"],
        model_at_frequency,
    )
    return np.real(complex_index), -np.imag(complex_index)


# ── Step 3: conductivity from the sample (n, k) ────────────────────────────────


def compute_conductivity(
    frequency: np.ndarray, sample_n: np.ndarray, sample_k: np.ndarray, config: dict
) -> np.ndarray:
    """Complex optical conductivity sigma(omega) from sample (n, k)."""
    derive_config = {"derive": {"eps_background": config["eps_background"]}}
    _, sigma_complex, _ = core.derive_eps_sigma(frequency, sample_n, sample_k, derive_config)
    return sigma_complex


def _fill_band_edges(values: np.ndarray, band_mask: np.ndarray) -> np.ndarray:
    """Edge-hold any non-finite in-band entries so the sample solver never reads a NaN.

    The sample step indexes n_sub frequency by frequency; a stray NaN at a band edge
    would poison that bin.  Out-of-band entries are left as-is.
    """
    filled = np.array(values, dtype=float)
    band_indices = np.where(band_mask)[0]
    finite_band = band_indices[np.isfinite(filled[band_indices])]
    if finite_band.size == 0:
        return filled
    for index in band_indices:
        if not np.isfinite(filled[index]):
            nearest = finite_band[int(np.argmin(np.abs(finite_band - index)))]
            filled[index] = filled[nearest]
    return filled


# ── Reporting / output ─────────────────────────────────────────────────────────


def _band_median(band_mask: np.ndarray, values: np.ndarray) -> float:
    """Median of the finite, in-band entries (NaN if none)."""
    in_band = values[band_mask]
    finite = in_band[np.isfinite(in_band)]
    return float(np.median(finite)) if finite.size else float("nan")


def print_report(label: str, results: dict, config: dict) -> None:
    """Print a short summary table for one measurement set."""
    band_mask = results["band_mask"]
    frequency = results["frequency"]
    fp_state = "ON" if config["fabry_perot"]["enabled"] else "OFF"

    sigma = results["sample_sigma"]
    print("\n" + "=" * 72)
    print(f"  Measurement set: {label}    (Fabry-Perot {fp_state})")
    print("=" * 72)
    print(f"  spacer thickness : {config['sample_layer_thickness_um']:.1f} um")
    print(f"  n_sub (window)   : {_band_median(band_mask, results['substrate_n']):.3f}")
    print(f"  sample n         : {_band_median(band_mask, results['sample_n']):.3f}")
    print(f"  sample k         : {_band_median(band_mask, results['sample_k']):.4f}")
    print(f"  sigma_real (S/m) : {_band_median(band_mask, np.real(sigma)):.3e}")
    print(f"  sigma_imag (S/m) : {_band_median(band_mask, np.imag(sigma)):.3e}")
    print(f"  band             : {config['report_frequency_min_hz']/1e12:.1f}"
          f"-{config['report_frequency_max_hz']/1e12:.1f} THz, "
          f"{int(np.count_nonzero(band_mask))} bins")
    print("=" * 72)


def save_figure(label: str, results: dict, output_path: str) -> None:
    """Save a 2x3 panel figure (substrate n; sample n, k; sigma real/imag)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exception:  # pragma: no cover
        print(f"(plot skipped: {exception})")
        return

    frequency_thz = results["frequency"] / 1e12
    band_mask = results["band_mask"]
    sigma = results["sample_sigma"]

    def plot_band(axis, values):
        axis.plot(frequency_thz, np.where(band_mask, values, np.nan), linewidth=1.4)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    figure.suptitle(f"MATLAB-port sandwich inversion -- {label}")

    plot_band(axes[0, 0], results["substrate_n"]); axes[0, 0].set_title("substrate n_sub")
    axes[0, 0].set_ylabel("n_sub")
    plot_band(axes[0, 1], results["sample_n"]); axes[0, 1].set_title("sample n")
    axes[0, 1].set_ylabel("n")
    plot_band(axes[0, 2], results["sample_k"]); axes[0, 2].set_title("sample k")
    axes[0, 2].set_ylabel("k")
    plot_band(axes[1, 0], np.real(sigma)); axes[1, 0].set_title("sigma_real (S/m)")
    axes[1, 0].set_ylabel("S/m"); axes[1, 0].set_xlabel("THz")
    plot_band(axes[1, 1], np.imag(sigma)); axes[1, 1].set_title("sigma_imag (S/m)")
    axes[1, 1].set_ylabel("S/m"); axes[1, 1].set_xlabel("THz")
    axes[1, 2].axis("off")

    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    print(f"  saved figure : {output_path}")


# ── Orchestration: run the whole chain for every measurement set ───────────────


def process_measurement_set(measurement: MeasurementSet, config: dict) -> dict:
    """Run the full chain for one set; return all the extracted arrays in a dict."""
    transfer_functions = compute_transfer_functions(measurement, config)
    band_mask = transfer_functions["band_mask"]

    # Step 1: substrate window index.
    substrate_n, substrate_k = invert_substrate(transfer_functions, config)
    substrate_n_filled = _fill_band_edges(substrate_n, band_mask)

    # Step 2: sample index, fed the substrate index per frequency.
    sample_n, sample_k = invert_sample(transfer_functions, substrate_n_filled, config)

    # Step 3: conductivity.
    sample_sigma = compute_conductivity(
        transfer_functions["frequency"], sample_n, sample_k, config
    )

    return {
        "label": measurement.label,
        "frequency": transfer_functions["frequency"],
        "band_mask": band_mask,
        "substrate_n": substrate_n,
        "substrate_k": substrate_k,
        "sample_n": sample_n,
        "sample_k": sample_k,
        "sample_sigma": sample_sigma,
    }


def main() -> None:
    output_directory = os.path.dirname(__file__)
    measurement_sets = load_all_measurement_sets(CONFIG)
    print(f"Loaded {len(measurement_sets)} measurement set(s).")

    all_results: dict[str, dict] = {}
    for measurement in measurement_sets:
        print(f"\nProcessing '{measurement.label}' ...")
        results = process_measurement_set(measurement, CONFIG)
        all_results[measurement.label] = results

        print_report(measurement.label, results, CONFIG)

        if CONFIG["save_figure"]:
            figure_path = os.path.join(output_directory, f"matlab_port_{measurement.label}.png")
            save_figure(measurement.label, results, figure_path)

        if CONFIG["save_conductivity_dat"]:
            frequency_thz = results["frequency"] / 1e12
            sigma = results["sample_sigma"]
            base = os.path.join(output_directory, f"matlab_port_{measurement.label}")
            np.savetxt(base + "_sigma_real.dat", np.column_stack([frequency_thz, np.real(sigma)]))
            np.savetxt(base + "_sigma_imag.dat", np.column_stack([frequency_thz, np.imag(sigma)]))

    if CONFIG["save_results_pickle"]:
        results_path = os.path.join(output_directory, "matlab_port_results.pkl")
        with open(results_path, "wb") as results_file:
            pickle.dump(all_results, results_file)
        print(f"\nSaved all results: {results_path}")


if __name__ == "__main__":
    main()
