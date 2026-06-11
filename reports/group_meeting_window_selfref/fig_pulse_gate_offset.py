"""Figure: how pulse position within the time gate affects self-referencing.

Story for the group meeting
---------------------------
Front-pulse self-referencing forms W = Y2/Y1, a *spectral* ratio. In the time
domain that is a deconvolution, and a deconvolution through a finite, tapered
gate is only exact when the gate weighs each pulse's surroundings identically.
A real instrument drift acts like a convolution kernel g(t) that smears small
echoes around every pulse (here +/- tau). If a pulse sits off-centre in its
Hann gate, the taper clips/attenuates those echoes asymmetrically, so the
front-pulse correction no longer perfectly matches the second-pulse
contamination and a residual survives.

This script sweeps the pulse position relative to the gate centre and plots the
residual self-referenced error, for a few drift echo delays tau. It also shows
the time-domain picture (pulse + drift echoes under the Hann gate) at a centred
and an off-centre position so the mechanism is visible.

Reuses the production code paths:
- thz_core.window_transfer_model builds the true second pulse from the first;
- thz_adapter._absolute_time_spectrum is the exact gating+FFT used in the
  pipeline's self-referencing.

Run:
    .venv/Scripts/python.exe reports/group_meeting_window_selfref/fig_pulse_gate_offset.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import thz_core.thz_core as core  # noqa: E402
from dataset_core.adapters import thz_adapter as thz  # noqa: E402

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- synthetic geometry (mirrors the CNT-13/D window, 0.9 mm fused silica) ----
N_WINDOW = 1.96 - 0.006j
THICKNESS_M = 0.9e-3
THETA_EXT_DEG = 45.0
H_TRUE = -1.5                      # flat sign-flipped sample response (CNT-like)

DT_PS = 0.05
PULSE_WIDTH_PS = 0.25
GATE_WIDTH_PS = 6.0               # matches the real second-reflection gate span
DRIFT_AMPLITUDE = 0.15           # 15% spectral ripple
ECHO_DELAYS_PS = (0.3, 0.6, 1.0)  # time-domain spread of the drift kernel
ASSESS_BAND_THZ = (0.4, 2.0)
FFT_LENGTH = 8192

# Long enough to hold both pulses plus their gates with margin.
TIME_PS = np.arange(140.0, 185.0, DT_PS)
FRONT_CENTRE_PS = 154.8
SECOND_CENTRE_PS = None          # filled from the model delay below


def _full_trace_freq_hz():
    return np.fft.rfftfreq(TIME_PS.size, DT_PS * 1e-12)


def _gaussian(centre_ps):
    return np.exp(-((TIME_PS - centre_ps) ** 2) / (2 * PULSE_WIDTH_PS**2))


def _build_two_pulse_trace(extra_second_filter):
    """Front pulse + second pulse (front filtered by the window model W)."""
    freq_hz = _full_trace_freq_hz()
    w_model = core.window_transfer_model(
        freq_hz, N_WINDOW, THICKNESS_M, np.deg2rad(THETA_EXT_DEG)
    )
    front = _gaussian(FRONT_CENTRE_PS)
    second = np.fft.irfft(
        np.fft.rfft(front) * w_model * extra_second_filter, n=front.size
    )
    return front + second


def _drift_kernel(echo_delay_ps):
    """Real, zero-phase drift filter: G(f) = 1 + a*cos(2*pi*f*tau).

    In time domain this is delta(t) + (a/2)[delta(t-tau)+delta(t+tau)], i.e. it
    plants echo replicas of every pulse at +/- tau — a clean, interpretable
    stand-in for alignment drift.
    """
    freq_hz = _full_trace_freq_hz()
    return 1.0 + DRIFT_AMPLITUDE * np.cos(2 * np.pi * freq_hz * echo_delay_ps * 1e-12)


def _measured_delay_s():
    beta = np.sqrt(N_WINDOW.real**2 - np.sin(np.deg2rad(THETA_EXT_DEG)) ** 2)
    return 2.0 * THICKNESS_M * beta / 299_792_458.0


def _gate_segment(trace, gate_centre_ps):
    """Crop the trace to a GATE_WIDTH_PS boxcar centred at gate_centre_ps.

    Returns (time_s, amplitude) on the original sampling — the segment that the
    pipeline would Hann-window and FFT.
    """
    lo = gate_centre_ps - GATE_WIDTH_PS / 2
    hi = gate_centre_ps + GATE_WIDTH_PS / 2
    mask = (TIME_PS >= lo) & (TIME_PS <= hi)
    return TIME_PS[mask] * 1e-12, trace[mask]


def _selfref_H(reference_trace, sample_trace, gate_offset_ps):
    """Self-referenced H over the assessment band for a given gate offset.

    gate_offset_ps is the pulse position relative to the gate centre: the gate
    is placed at pulse_centre - offset, so positive offset = pulse right of the
    gate centre.
    """
    first_gate = FRONT_CENTRE_PS - gate_offset_ps
    second_gate = SECOND_CENTRE_PS - gate_offset_ps

    def spectrum(trace, gate_centre):
        time_s, amplitude = _gate_segment(trace, gate_centre)
        freq, spec = thz._absolute_time_spectrum(time_s, amplitude, FFT_LENGTH)
        return freq, spec

    freq, y1_samp = spectrum(sample_trace, first_gate)
    _, y2_samp = spectrum(sample_trace, second_gate)
    _, y1_ref = spectrum(reference_trace, first_gate)
    _, y2_ref = spectrum(reference_trace, second_gate)

    with np.errstate(divide="ignore", invalid="ignore"):
        h_old = y2_samp / y2_ref
        h_new = h_old * (y1_ref / y1_samp)

    f_thz = freq * 1e-12
    band = (f_thz >= ASSESS_BAND_THZ[0]) & (f_thz <= ASSESS_BAND_THZ[1])
    return h_old[band], h_new[band]


def _rms_error(h_band):
    return float(np.sqrt(np.mean(np.abs(h_band / H_TRUE - 1.0) ** 2)))


def main():
    global SECOND_CENTRE_PS
    delay_s = _measured_delay_s()
    SECOND_CENTRE_PS = FRONT_CENTRE_PS + delay_s * 1e12

    reference_trace = _build_two_pulse_trace(1.0)

    offsets_ps = np.linspace(-2.4, 2.4, 49)

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0])
    ax_err = fig.add_subplot(grid[0, :])
    ax_centre = fig.add_subplot(grid[1, 0])
    ax_offset = fig.add_subplot(grid[1, 1])

    summary = {}
    for echo_delay_ps in ECHO_DELAYS_PS:
        sample_trace = np.fft.irfft(
            np.fft.rfft(_build_two_pulse_trace(H_TRUE)) * _drift_kernel(echo_delay_ps),
            n=TIME_PS.size,
        )
        errors_new = []
        errors_old = []
        for offset in offsets_ps:
            h_old, h_new = _selfref_H(reference_trace, sample_trace, offset)
            errors_new.append(_rms_error(h_new) * 100)
            errors_old.append(_rms_error(h_old) * 100)
        errors_new = np.array(errors_new)
        line, = ax_err.plot(offsets_ps, errors_new, lw=1.8,
                            label=f"self-ref, drift echo tau = {echo_delay_ps} ps")
        summary[echo_delay_ps] = (errors_new, np.array(errors_old))
        centre_error = errors_new[np.argmin(np.abs(offsets_ps))]
        ax_err.scatter([0], [centre_error], color=line.get_color(), zorder=5)

    # Conventional reference error (drift-limited, ~offset-independent) shown for
    # the middle echo delay as the "do nothing" baseline.
    baseline = summary[ECHO_DELAYS_PS[1]][1]
    ax_err.plot(offsets_ps, baseline, color="0.5", lw=1.4, ls=":",
                label="conventional reference (drift-limited)")

    # Shade the "keep the pulse here" region (within +/-1 ps of centre).
    ax_err.axvspan(-1.0, 1.0, color="tab:green", alpha=0.07,
                   label="recommended (+/-1 ps)")
    ax_err.axvline(0.0, color="0.7", lw=0.8)
    ax_err.set_yscale("log")
    ax_err.set_xlabel("Pulse position relative to gate centre (ps)")
    ax_err.set_ylabel("Residual error in H (rms %, log)")
    ax_err.set_title("Self-referencing accuracy vs pulse position in the time gate "
                     f"(gate width {GATE_WIDTH_PS:.0f} ps, drift {DRIFT_AMPLITUDE*100:.0f}%)")
    ax_err.legend(fontsize=8, ncol=2)
    ax_err.grid(True, which="both", alpha=0.2)

    # ---- time-domain mechanism panels (echo delay = 1.0 ps, the worst case) --
    sample_trace = np.fft.irfft(
        np.fft.rfft(_build_two_pulse_trace(H_TRUE)) * _drift_kernel(ECHO_DELAYS_PS[-1]),
        n=TIME_PS.size,
    )
    for ax, offset, title in [
        (ax_centre, 0.0, "Pulse centred in gate — echoes weighted symmetrically"),
        (ax_offset, 2.0, "Pulse near gate edge — trailing echo clipped"),
    ]:
        gate_centre = SECOND_CENTRE_PS - offset
        time_s, amplitude = _gate_segment(sample_trace, gate_centre)
        time_ps = time_s * 1e12
        hann = np.hanning(amplitude.size)
        ax.plot(time_ps, amplitude, color="0.3", lw=1.0, label="gated signal")
        ax.plot(time_ps, hann * np.max(np.abs(amplitude)), color="tab:red", lw=1.2,
                ls="--", label="Hann taper")
        ax.axvline(SECOND_CENTRE_PS, color="tab:blue", lw=0.8, alpha=0.6)
        for sign in (-1, 1):
            ax.axvline(SECOND_CENTRE_PS + sign * ECHO_DELAYS_PS[-1], color="tab:green",
                       lw=0.8, ls=":", alpha=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time (ps)")
        ax.set_xlim(gate_centre - GATE_WIDTH_PS / 2, gate_centre + GATE_WIDTH_PS / 2)
    ax_centre.set_ylabel("Amplitude")
    ax_centre.legend(fontsize=8)

    out_path = os.path.join(OUTPUT_DIR, "fig_pulse_gate_offset.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    # ---- printed summary for the report ------------------------------------
    print("\nResidual self-ref error (rms %) by drift echo delay:")
    print(f"  {'tau (ps)':>9} | {'centred':>8} | {'+/-1 ps':>8} | {'+/-2 ps':>8}")
    for echo_delay_ps in ECHO_DELAYS_PS:
        errors_new, _ = summary[echo_delay_ps]
        def at(off):
            return errors_new[np.argmin(np.abs(offsets_ps - off))]
        worst_1 = max(at(-1.0), at(1.0))
        worst_2 = max(at(-2.0), at(2.0))
        print(f"  {echo_delay_ps:>9.1f} | {at(0.0):>7.2f}% | {worst_1:>7.2f}% | {worst_2:>7.2f}%")
    print(f"\nConventional reference baseline (drift-limited): "
          f"{np.mean(baseline):.1f}% rms")


if __name__ == "__main__":
    main()
