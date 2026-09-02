"""A walkthrough of the THz-TDS noise model, on real data.

Run this to regenerate every figure and number in the accompanying report. Each step
below answers one question, and each produces one figure:

  1  What are we actually working with?          the individual scans vs their mean
  2  Does the noise have structure?              sigma(t) under the pulse
  3  What is that structure made of?             the three fitted terms, separately
  4  Is the instrument steady?                   per-scan amplitude and delay vs clock time
  5  How much of the "noise" was drift?          raw vs drift-corrected sigma(t)
  6  What does it mean in the frequency domain?  propagated bars vs the tail-median floor
  7  Which estimator is measuring noise?         the 1/sqrt(M) test
  8  What actually limits this measurement?      within-scan vs drift vs floor, per frequency

Figures are written twice, for light and dark presentation surfaces.

    python explorations/noise_model_walkthrough/noise_model_walkthrough.py
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from thz_core.thz_core.noise import (
    block_spectral_scatter,
    drift_corrected_scatter,
    fit_noise_parameters,
    noise_amplitude,
    spectral_noise_moments,
    time_derivative,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_ROOT = "/home/match/data/CNTs"
FEATURED = f"{DATA_ROOT}/2026-08-20_CNT-paper-doped_2/export/sample_CNT_doped-12mm.acc"
SURVEY = {
    "CNT doped 12 mm (2026-08-20)":
        f"{DATA_ROOT}/2026-08-20_CNT-paper-doped_2/export/sample_CNT_doped-12mm.acc",
    "CNT bare reference (2026-08-20)":
        f"{DATA_ROOT}/2026-08-20_CNT-paper-doped_2/export/reference_CNT_undoped-19mm.acc",
    "CNT doped 7 mm (2026-08-20)":
        f"{DATA_ROOT}/2026-08-20_CNT-paper-doped_2/export/sample_CNT_doped-7mm.acc",
    "CNT bare reference (2026-08-21)":
        f"{DATA_ROOT}/2026-08-21_CNT-paper-doped_3/comparison/reference_CNT-2_bare-5mm.acc",
    "CNT doped 15 mm (2026-08-21)":
        f"{DATA_ROOT}/2026-08-21_CNT-paper-doped_3/comparison/sample_CNT-2_doped-15mm.acc",
}

# Validated categorical slots 1-3 (all-pairs safe in both modes) plus a neutral.
LIGHT = dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#d8d7d2",
             s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#4a3aa7")
DARK = dict(surface="#1a1a19", text="#ffffff", muted="#c3c2b7", grid="#3a3a38",
            s1="#3987e5", s2="#d95926", s3="#199e70", s4="#9085e9")

_PICOSECOND = 1e-12
_HZ_TO_THZ = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# loading
# ─────────────────────────────────────────────────────────────────────────────


def load_acquisition(filepath: str) -> dict:
    """Per-scan waveforms plus each scan's wall-clock timestamp, straight from the file.

    Parsed here rather than through the pipeline loader so the walkthrough shows the
    raw material, with no processing choices already baked in.
    """
    with open(filepath) as handle:
        text = handle.read()

    timestamps, waveforms, time_axis = [], [], None
    for chunk in text.split("%%"):
        stamp = re.search(r"Date and time,([0-9\-: .]+)", chunk)
        rows = []
        for line in chunk.splitlines():
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        if len(rows) < 8:
            continue
        array = np.asarray(rows)
        if time_axis is None:
            time_axis = array[:, 0] * _PICOSECOND
        if array.shape[0] != time_axis.size:
            continue
        waveforms.append(array[:, 1])
        timestamps.append(datetime.fromisoformat(stamp.group(1).strip())
                          if stamp else None)

    waveforms = np.asarray(waveforms)
    if timestamps and all(stamp is not None for stamp in timestamps):
        origin = timestamps[0]
        elapsed_minutes = np.asarray(
            [(stamp - origin).total_seconds() / 60.0 for stamp in timestamps])
    else:
        elapsed_minutes = np.arange(waveforms.shape[0], dtype=float)

    return dict(
        filepath=filepath,
        time_seconds=time_axis,
        waveforms=waveforms,
        elapsed_minutes=elapsed_minutes,
        dt=float(np.median(np.diff(time_axis))),
    )


# ─────────────────────────────────────────────────────────────────────────────
# figure plumbing
# ─────────────────────────────────────────────────────────────────────────────


def style(mode: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": mode["surface"], "axes.facecolor": mode["surface"],
        "savefig.facecolor": mode["surface"],
        "text.color": mode["text"], "axes.labelcolor": mode["text"],
        "axes.edgecolor": mode["grid"], "xtick.color": mode["muted"],
        "ytick.color": mode["muted"], "grid.color": mode["grid"],
        "axes.titlecolor": mode["text"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.5, "grid.linewidth": 0.6,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "600",
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.8, "figure.dpi": 110,
    })


def emit(figure, name: str, mode_name: str) -> None:
    path = os.path.join(OUT_DIR, f"{name}_{mode_name}.png")
    figure.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(figure)


def build(name: str, builder) -> None:
    """Render one figure twice, once per presentation surface."""
    for mode_name, mode in (("light", LIGHT), ("dark", DARK)):
        style(mode)
        figure = builder(mode)
        emit(figure, name, mode_name)
    print(f"  wrote {name}_[light|dark].png")


# ─────────────────────────────────────────────────────────────────────────────
# the walkthrough
# ─────────────────────────────────────────────────────────────────────────────

print("Loading featured acquisition...")
featured = load_acquisition(FEATURED)
waveforms = featured["waveforms"]
dt = featured["dt"]
time_ps = featured["time_seconds"] / _PICOSECOND
n_scans, n_samples = waveforms.shape
print(f"  {os.path.basename(FEATURED)}: {n_scans} scans x {n_samples} samples, "
      f"dt = {dt * 1e15:.0f} fs, acquisition spanned "
      f"{featured['elapsed_minutes'][-1]:.0f} min")

print("Fitting the noise model...")
drift = drift_corrected_scatter(waveforms, dt)
parameters, _ = fit_noise_parameters(waveforms, dt, drift=drift)
mean_waveform = drift.mean_waveform
derivative = time_derivative(mean_waveform, dt)
peak_amplitude = float(np.max(np.abs(mean_waveform)))
fractions = parameters.term_contributions(mean_waveform, dt)

window = np.hanning(n_samples)
spectra = np.fft.rfft(waveforms * window, n=n_samples, axis=1)
mean_spectrum = np.fft.rfft(mean_waveform * window, n=n_samples)
frequency_thz = np.fft.rfftfreq(n_samples, dt) * _HZ_TO_THZ
moments = spectral_noise_moments(drift.sigma_t, window=window, n_fft=n_samples,
                                 spectrum=mean_spectrum, n_averaged=n_scans)
blocked = block_spectral_scatter(spectra, n_blocks=8)


def tail_median_floor(spectrum, tail_fraction=0.25):
    magnitude = np.abs(spectrum)
    tail = max(8, int(np.ceil(magnitude.size * tail_fraction)))
    return float(np.median(magnitude[-tail:]))


floor = tail_median_floor(mean_spectrum)

print("Rendering figures...")


# 1 ─ the raw material -------------------------------------------------------
def figure_scans(mode):
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.4),
                                         gridspec_kw={"width_ratios": [1.6, 1]})
    for row in waveforms:
        left.plot(time_ps, row, color=mode["s1"], alpha=0.10, lw=0.7)
    left.plot(time_ps, mean_waveform, color=mode["text"], lw=1.6)
    left.set_xlabel("delay (ps)")
    left.set_ylabel("detector signal (a.u.)")
    left.set_title(f"{n_scans} individual scans (faint) and their mean (dark)")

    peak_index = int(np.argmax(np.abs(mean_waveform)))
    zoom = slice(max(0, peak_index - 6), peak_index + 7)
    for row in waveforms:
        right.plot(time_ps[zoom], row[zoom], color=mode["s1"], alpha=0.18, lw=0.7)
    right.plot(time_ps[zoom], mean_waveform[zoom], color=mode["text"], lw=1.8,
               marker="o", ms=4)
    right.set_xlabel("delay (ps)")
    right.set_title("zoom on the peak — the spread is the signal we want")
    figure.tight_layout()
    return figure


build("01_scans", figure_scans)


# 2 ─ noise has structure ----------------------------------------------------
def figure_sigma_structure(mode):
    figure, axis = plt.subplots(figsize=(9.5, 3.6))
    twin = axis.twinx()
    twin.plot(time_ps, mean_waveform, color=mode["grid"], lw=1.4, zorder=1)
    twin.set_ylabel("pulse (a.u.)", color=mode["muted"])
    twin.tick_params(axis="y", colors=mode["muted"])
    twin.grid(False)
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color(mode["grid"])

    axis.plot(time_ps, drift.sigma_t, color=mode["s2"], lw=1.9, zorder=3,
              label=r"measured $\sigma(t)$ across scans")
    axis.set_zorder(twin.get_zorder() + 1)
    axis.patch.set_visible(False)
    axis.set_xlabel("delay (ps)")
    axis.set_ylabel(r"noise amplitude $\sigma(t)$ (a.u.)", color=mode["s2"])
    axis.tick_params(axis="y", colors=mode["s2"])
    axis.set_title("Noise is not flat in time — it has structure tied to the pulse")
    axis.legend(loc="upper left")
    figure.tight_layout()
    return figure


build("02_sigma_structure", figure_sigma_structure)


# 3 ─ the decomposition ------------------------------------------------------
def figure_decomposition(mode):
    figure, axis = plt.subplots(figsize=(9.5, 3.8))
    additive = np.full_like(mean_waveform, parameters.sigma_alpha)
    multiplicative = np.abs(parameters.sigma_beta * mean_waveform)
    jitter = np.abs(parameters.sigma_tau * derivative)
    total = noise_amplitude(mean_waveform, dt, parameters.sigma_alpha,
                            parameters.sigma_beta, parameters.sigma_tau)

    axis.plot(time_ps, drift.sigma_t, color=mode["muted"], lw=1.2, alpha=0.8,
              label=r"measured $\sigma(t)$")
    axis.plot(time_ps, additive, color=mode["s1"], lw=1.7, ls="--",
              label=rf"additive  $\sigma_\alpha$ = {parameters.sigma_alpha:.2e}")
    axis.plot(time_ps, multiplicative, color=mode["s2"], lw=1.7, ls="-.",
              label=rf"multiplicative  $\sigma_\beta$ = {parameters.sigma_beta * 100:.2f}%")
    axis.plot(time_ps, jitter, color=mode["s3"], lw=1.7, ls=":",
              label=rf"jitter  $\sigma_\tau$ = {parameters.sigma_tau * 1e15:.2f} fs")
    axis.plot(time_ps, total, color=mode["text"], lw=1.8, alpha=0.9,
              label="model total")
    axis.set_xlabel("delay (ps)")
    axis.set_ylabel(r"$\sigma(t)$ (a.u.)")
    axis.set_title("The three sources explain different parts of the record")
    axis.legend(ncol=2, loc="upper left")
    figure.tight_layout()
    return figure


build("03_decomposition", figure_decomposition)


# 4 ─ drift over the acquisition --------------------------------------------
def figure_drift(mode):
    figure, (upper, lower) = plt.subplots(2, 1, figsize=(9.5, 4.6), sharex=True)
    minutes = featured["elapsed_minutes"]
    upper.plot(minutes, drift.amplitudes, color=mode["s1"], lw=1.4, marker="o", ms=3)
    upper.axhline(1.0, color=mode["grid"], lw=1.0)
    upper.set_ylabel("relative amplitude $A_l$")
    upper.set_title("Each scan's amplitude and arrival time, against wall-clock time")

    lower.plot(minutes, drift.delays * 1e15, color=mode["s2"], lw=1.4, marker="o", ms=3)
    lower.axhline(0.0, color=mode["grid"], lw=1.0)
    lower.set_ylabel(r"delay $\eta_l$ (fs)")
    lower.set_xlabel("elapsed time (minutes)")
    figure.tight_layout()
    return figure


build("04_drift", figure_drift)


# 5 ─ drift is not noise -----------------------------------------------------
def figure_drift_removed(mode):
    figure, axis = plt.subplots(figsize=(9.5, 3.4))
    axis.plot(time_ps, drift.raw_sigma_t, color=mode["s2"], lw=1.7,
              label="raw scatter across scans")
    axis.plot(time_ps, drift.sigma_t, color=mode["s1"], lw=1.7,
              label="after removing per-scan amplitude and delay")
    axis.fill_between(time_ps, drift.sigma_t, drift.raw_sigma_t,
                      color=mode["s2"], alpha=0.12, lw=0)
    axis.set_xlabel("delay (ps)")
    axis.set_ylabel(r"$\sigma(t)$ (a.u.)")
    axis.set_title(f"Drift counted as noise inflates the estimate "
                   f"{drift.drift_inflation:.2f}x (shaded)")
    axis.legend(loc="upper left")
    figure.tight_layout()
    return figure


build("05_drift_removed", figure_drift_removed)


# 6 ─ into the frequency domain ---------------------------------------------
def figure_spectrum(mode):
    figure, axis = plt.subplots(figsize=(9.5, 3.9))
    magnitude = np.abs(mean_spectrum)
    stride = 3
    axis.semilogy(frequency_thz, magnitude, color=mode["s1"], lw=1.4,
                  label="mean spectrum")
    axis.errorbar(frequency_thz[::stride], magnitude[::stride],
                  yerr=moments.sigma_magnitude[::stride], fmt="none",
                  ecolor=mode["s1"], elinewidth=1.0, capsize=2, alpha=0.7)
    axis.axhline(floor, color=mode["s2"], lw=1.7, ls="--",
                 label=f"tail-median “noise floor” = {floor:.2e}")
    axis.axhline(float(np.median(moments.sigma_magnitude)), color=mode["s3"],
                 lw=1.7, ls="-",
                 label=f"measured uncertainty = {np.median(moments.sigma_magnitude):.2e}")
    axis.set_xlabel("frequency (THz)")
    axis.set_ylabel("|Y(f)| (a.u.)")
    axis.set_ylim(1e-5, None)
    ratio = floor / float(np.median(moments.sigma_magnitude))
    axis.set_title(f"The conventional floor sits {ratio:.0f}x above the measured uncertainty")
    axis.legend(loc="lower left")
    figure.tight_layout()
    return figure


build("06_spectrum", figure_spectrum)


# 7 ─ the 1/sqrt(M) test -----------------------------------------------------
counts = [count for count in (8, 16, 32, 64, n_scans) if count <= n_scans]
floor_series, sigma_series = [], []
for count in counts:
    subset = waveforms[:count]
    subset_spectrum = np.fft.rfft(subset.mean(axis=0) * window, n=n_samples)
    subset_drift = drift_corrected_scatter(subset, dt)
    subset_moments = spectral_noise_moments(subset_drift.sigma_t, window=window,
                                            n_fft=n_samples, spectrum=subset_spectrum,
                                            n_averaged=count)
    floor_series.append(tail_median_floor(subset_spectrum))
    sigma_series.append(float(np.median(subset_moments.sigma_magnitude)))
floor_series = np.asarray(floor_series)
sigma_series = np.asarray(sigma_series)
counts_array = np.asarray(counts, dtype=float)


def figure_averaging(mode):
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.5))
    ideal = sigma_series[0] * np.sqrt(counts_array[0] / counts_array)
    left.loglog(counts_array, sigma_series, color=mode["s3"], marker="o", ms=6,
                label="measured uncertainty")
    left.loglog(counts_array, floor_series, color=mode["s2"], marker="s", ms=6,
                label="tail-median floor")
    left.loglog(counts_array, ideal, color=mode["muted"], lw=1.2, ls="--",
                label=r"ideal $1/\sqrt{M}$")
    left.set_xlabel("scans averaged, M")
    left.set_ylabel("uncertainty (a.u.)")
    left.set_title("More averaging should mean less uncertainty")
    left.legend()

    right.semilogx(counts_array, sigma_series * np.sqrt(counts_array)
                   / (sigma_series[0] * np.sqrt(counts_array[0])),
                   color=mode["s3"], marker="o", ms=6, label="measured uncertainty")
    right.semilogx(counts_array, floor_series * np.sqrt(counts_array)
                   / (floor_series[0] * np.sqrt(counts_array[0])),
                   color=mode["s2"], marker="s", ms=6, label="tail-median floor")
    right.axhline(1.0, color=mode["muted"], lw=1.2, ls="--")
    right.set_xlabel("scans averaged, M")
    right.set_ylabel(r"$\sigma\sqrt{M}$, normalised")
    right.set_title("A noise estimate holds flat here; a signal estimate does not")
    right.legend()
    figure.tight_layout()
    return figure


build("07_averaging", figure_averaging)


# 8 ─ what actually limits the measurement -----------------------------------
def figure_budget(mode):
    figure, axis = plt.subplots(figsize=(9.5, 3.7))
    band = (frequency_thz > 0.2) & (frequency_thz < 6.0)
    axis.semilogy(frequency_thz[band], moments.sigma_magnitude[band], color=mode["s1"],
                  lw=1.8, label="within-scan noise (drift removed)")
    axis.semilogy(frequency_thz[band], blocked[band], color=mode["s2"], lw=1.8,
                  label="including acquisition drift (batch means)")
    axis.axhline(floor, color=mode["s3"], lw=1.7, ls="--",
                 label="tail-median floor (frequency-independent by construction)")
    axis.set_xlabel("frequency (THz)")
    axis.set_ylabel("uncertainty on the mean spectrum (a.u.)")
    axis.set_title("This measurement is drift-limited in the mid-band, not noise-limited")
    axis.legend(loc="lower left")
    figure.tight_layout()
    return figure


build("08_budget", figure_budget)


# ─────────────────────────────────────────────────────────────────────────────
# the numbers
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 94)
print("NOISE BUDGET — featured acquisition")
print("=" * 94)
print(f"  file            : {os.path.basename(FEATURED)}")
print(f"  scans           : {n_scans} over {featured['elapsed_minutes'][-1]:.0f} minutes")
print(f"  pulse peak      : {peak_amplitude:.4e} a.u.")
print(f"  sigma_alpha     : {parameters.sigma_alpha:.4e} a.u. "
      f"({parameters.sigma_alpha / peak_amplitude * 100:.3f}% of peak)  additive / electronics")
print(f"  sigma_beta      : {parameters.sigma_beta * 100:.3f}%                    "
      f"multiplicative / laser power")
print(f"  sigma_tau       : {parameters.sigma_tau * 1e15:.2f} fs                    "
      f"timing jitter / delay line")
print(f"  variance share  : additive {fractions['additive']:.1%}, "
      f"multiplicative {fractions['multiplicative']:.1%}, jitter {fractions['jitter']:.1%}")
print(f"  fit             : method={parameters.method}, "
      f"profile_error={parameters.profile_error:.3f}, trusted={parameters.converged}")
print(f"  drift           : amplitude {np.ptp(drift.amplitudes) * 100:.2f}% p-p, "
      f"delay {np.ptp(drift.delays) * 1e15:.1f} fs p-p, "
      f"inflation {drift.drift_inflation:.2f}x")
print(f"  tail floor      : {floor:.4e}  ({floor / np.median(moments.sigma_magnitude):.1f}x "
      f"the measured uncertainty)")

print()
print("=" * 94)
print("SURVEY — every acquisition in the two CNT sessions")
print("=" * 94)
print(f"  {'acquisition':<32} {'M':>4} {'sigma_alpha/peak':>17} {'sigma_beta':>11} "
      f"{'sigma_tau':>10} {'drift':>7} {'dominant':>15}")
for label, filepath in SURVEY.items():
    if not os.path.exists(filepath):
        print(f"  {label:<32}  (missing)")
        continue
    entry = load_acquisition(filepath)
    entry_drift = drift_corrected_scatter(entry["waveforms"], entry["dt"])
    entry_parameters, _ = fit_noise_parameters(entry["waveforms"], entry["dt"],
                                               drift=entry_drift)
    entry_peak = float(np.max(np.abs(entry_drift.mean_waveform)))
    entry_fractions = entry_parameters.term_contributions(entry_drift.mean_waveform,
                                                          entry["dt"])
    dominant = max(entry_fractions, key=entry_fractions.get)
    print(f"  {label:<32} {entry['waveforms'].shape[0]:4d} "
          f"{entry_parameters.sigma_alpha / entry_peak * 100:16.3f}% "
          f"{entry_parameters.sigma_beta * 100:10.3f}% "
          f"{entry_parameters.sigma_tau * 1e15:9.2f}fs "
          f"{entry_drift.drift_inflation:6.2f}x {dominant:>15}")

print()
print(f"Figures written to {OUT_DIR}")
