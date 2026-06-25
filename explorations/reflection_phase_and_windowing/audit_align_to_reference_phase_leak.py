"""AUDIT 1: does align_to_reference inject a spurious phase into the self-referenced H?

Hypothesis (derived analytically): in the shared-axis self-reference path,
    H = (Y2_s/Y1_s) / (Y2_r/Y1_r),
and fft_spectrum multiplies the FIRST reflection spectrum by exp(-2πi f t0) (absolute-time
phase reference) but NOT the second. So

    H_measured = H_physical * exp(+2πi f (t0_sample - t0_reference)).

align_to_reference shifts ONLY the sample's time axis by `integer_shift` (the reference is the
untouched T0 anchor). That makes t0_sample - t0_reference = integer_shift, so a pure axis
relabel — which should be physically irrelevant after self-referencing — leaks in as a linear
phase exp(2πi f * integer_shift). A 1-sample re-align is a big high-frequency phase error.

This script reproduces the mechanism on SYNTHETIC two-pulse traces using the SAME math the
adapter uses (rfft + the first-reflection exp(-2πi f t0) factor + the self-ref ratio), with NO
GUI and NO real data dependency, and checks the leaked phase equals the prediction to ~1e-12.

Run:  PYTHONPATH=. ../.venv/Scripts/python.exe explorations/reflection_phase_and_windowing/audit_align_to_reference_phase_leak.py
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPEED_OF_LIGHT = 299_792_458.0


def gaussian_pulse(time_s, centre_s, width_s, amplitude):
    return amplitude * np.exp(-0.5 * ((time_s - centre_s) / width_s) ** 2)


def make_trace(time_s, front_centre_s, back_centre_s, back_amp, back_width_s):
    """A two-pulse reflection trace: front (window) pulse + back (sample) pulse."""
    front = gaussian_pulse(time_s, front_centre_s, 0.30e-12, 1.0)
    back = gaussian_pulse(time_s, back_centre_s, back_width_s, back_amp)
    return front + back


def segment_spectrum(time_s, amplitude, n_fft, apply_absolute_time_factor):
    """rfft of a segment; optionally apply the first-reflection exp(-2πi f t0) factor."""
    spectrum = np.fft.rfft(amplitude, n=n_fft)
    freq = np.fft.rfftfreq(n_fft, float(np.median(np.diff(time_s))))
    if apply_absolute_time_factor:
        spectrum = spectrum * np.exp(-2j * np.pi * freq * time_s[0])
    return freq, spectrum


def windowed(time_s, amplitude, centre_s, half_width_s):
    """Isolate one pulse with a Hann window centred at `centre_s` (full length kept)."""
    mask = np.abs(time_s - centre_s) <= half_width_s
    out = np.zeros_like(amplitude)
    idx = np.where(mask)[0]
    if idx.size:
        win = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(idx.size)) / (idx.size - 1)))
        out[idx] = amplitude[idx] * win
    return out


def self_referenced_H(time_s, ref_trace, samp_trace, regions, n_fft):
    """Reproduce the shared-axis self-ref H exactly as the adapter forms it."""
    (front_c, front_hw), (back_c, back_hw) = regions
    # second reflection: NO absolute-time factor; first reflection: WITH factor.
    _, Y2_r = segment_spectrum(time_s, windowed(time_s, ref_trace, back_c, back_hw), n_fft, False)
    _, Y2_s = segment_spectrum(time_s, windowed(time_s, samp_trace, back_c, back_hw), n_fft, False)
    freq, Y1_r = segment_spectrum(time_s, windowed(time_s, ref_trace, front_c, front_hw), n_fft, True)
    _, Y1_s = segment_spectrum(time_s, windowed(time_s, samp_trace, front_c, front_hw), n_fft, True)
    H = (Y2_s / Y2_r) * (Y1_r / Y1_s)   # = (Y2_s/Y1_s)/(Y2_r/Y1_r)
    return freq, H


def main():
    dt = 0.05e-12
    time_s = np.arange(0.0, 40e-12, dt)
    n_fft = 4096

    # Front pulses coincident (window reflection is reference-invariant); sample back pulse
    # is a slightly broadened, reduced echo (stands in for a real CNT/Si reflection).
    front_c = 8.0e-12
    back_c = 20.0e-12
    ref_trace = make_trace(time_s, front_c, back_c, back_amp=0.45, back_width_s=0.30e-12)
    samp_trace = make_trace(time_s, front_c, back_c, back_amp=0.62, back_width_s=0.42e-12)

    regions = ((front_c, 2.0e-12), (back_c, 2.0e-12))

    # Baseline H with sample and reference on the SAME axis (align OFF).
    freq, H_aligned_off = self_referenced_H(time_s, ref_trace, samp_trace, regions, n_fft)

    print("=== AUDIT 1: align_to_reference phase leak (synthetic, adapter math) ===")
    print(f"dt = {dt*1e12:.3f} ps, n_fft = {n_fft}")

    # Now emulate align_to_reference: shift ONLY the sample's time AXIS by integer samples
    # (a relabel, exactly what the adapter does: data[:,0] += integer_shift). The reference
    # is untouched. Compare H for several integer shifts.
    band = (freq >= 0.3e12) & (freq <= 3.0e12)
    print("\nInteger-sample re-align -> phase change in H (should be ZERO if align were safe):")
    max_leak = {}
    for n_shift in (-2, -1, 0, 1, 2):
        shifted_time = time_s + n_shift * dt
        # sample uses shifted axis; reference uses original axis
        (front_c0, front_hw), (back_c0, back_hw) = regions
        _, Y2_r = segment_spectrum(time_s, windowed(time_s, ref_trace, back_c0, back_hw), n_fft, False)
        _, Y2_s = segment_spectrum(shifted_time, windowed(shifted_time, samp_trace, back_c0 + n_shift*dt, back_hw), n_fft, False)
        f2, Y1_r = segment_spectrum(time_s, windowed(time_s, ref_trace, front_c0, front_hw), n_fft, True)
        _, Y1_s = segment_spectrum(shifted_time, windowed(shifted_time, samp_trace, front_c0 + n_shift*dt, front_hw), n_fft, True)
        H_shift = (Y2_s / Y2_r) * (Y1_r / Y1_s)

        leaked_phase = np.angle(H_shift[band] / H_aligned_off[band])
        predicted = np.angle(np.exp(2j * np.pi * freq[band] * (n_shift * dt)))
        residual = np.angle(np.exp(1j * (leaked_phase - predicted)))
        max_leak[n_shift] = np.max(np.abs(leaked_phase))
        print(f"  shift {n_shift:+d} samp ({n_shift*dt*1e12:+.3f} ps): "
              f"max |Δφ(H)| in 0.3-3 THz = {np.rad2deg(np.max(np.abs(leaked_phase))):6.1f} deg ; "
              f"matches exp(2πi f Δ) prediction to {np.max(np.abs(residual)):.1e} rad")

    one_sample_deg = np.rad2deg(max_leak[1])
    print(f"\nVERDICT: a SINGLE-sample re-align injects up to {one_sample_deg:.0f} deg of phase "
          f"error into H across the band — purely from relabelling the sample's time axis.")
    print("This is the 'minor realignments -> huge discrepancies' the audit flagged, and it is")
    print("ZERO when align is off (sample & reference share t0). The leak = exp(2πi f * shift),")
    print("i.e. the asymmetric first-reflection exp(-2πi f t0) factor not cancelling once the")
    print("sample axis is moved relative to the (unshifted) reference.")

    # Figure: phase of H for each shift.
    figure, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    f_thz = freq[band] * 1e-12
    for n_shift in (-2, -1, 0, 1, 2):
        shifted_time = time_s + n_shift * dt
        (front_c0, front_hw), (back_c0, back_hw) = regions
        _, Y2_r = segment_spectrum(time_s, windowed(time_s, ref_trace, back_c0, back_hw), n_fft, False)
        _, Y2_s = segment_spectrum(shifted_time, windowed(shifted_time, samp_trace, back_c0 + n_shift*dt, back_hw), n_fft, False)
        _, Y1_r = segment_spectrum(time_s, windowed(time_s, ref_trace, front_c0, front_hw), n_fft, True)
        _, Y1_s = segment_spectrum(shifted_time, windowed(shifted_time, samp_trace, front_c0 + n_shift*dt, front_hw), n_fft, True)
        H_shift = (Y2_s / Y2_r) * (Y1_r / Y1_s)
        axis.plot(f_thz, np.unwrap(np.angle(H_shift[band])), label=f"align shift {n_shift:+d} samp")
    axis.set_xlabel("Frequency (THz)")
    axis.set_ylabel("phase of H (rad)")
    axis.set_title("Self-referenced H phase vs integer re-align — leak from the first-reflection t0 factor")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "audit_align_to_reference_phase_leak.png")
    figure.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
