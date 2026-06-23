"""Evaluate front-reflection self-referencing on the CNT-13/D dataset.

Question being evaluated
------------------------
Can the first reflection (air->SiO2 front face) of a window-coupled reflection
trace be used to *predict* the bare-window second reflection (SiO2->air back
face), so that a synthetic reference can be constructed from the sample trace
itself instead of relying on a separately-acquired bare-window measurement that
suffers mount-to-mount drift?

Key quantities
--------------
W(omega)  = Y_second / Y_first of the BARE window trace.
            A property of the window alone (Fresnel factors + internal
            propagation); the source spectrum, detector response and shared air
            path cancel in the intra-trace ratio.
R(omega)  = Y_second / Y_first of any trace (sample or bare).
H_new     = R_sample / W   — drift-immune transfer function.
H_old     = Y2_sample / Y2_reference — the current approach.
D(omega)  = Y1_sample / Y1_reference — mount-to-mount drift of the front pulse.
            Identity D == 1 would mean self-referencing adds nothing;
            structure in D is the fake-feature contamination of H_old.
            Note H_new = H_old / D.

Outputs: PNG figures + printed metrics into explorations/output/.

Run with the repo venv from the repo root:
    .venv/Scripts/python.exe explorations/window_selfref_evaluation.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.io.loaders.acc_loader import ACCLoader  # noqa: E402

# ---------------------------------------------------------------- constants
DATA_ROOT = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\D\segmented"
OUTPUT_DIR = os.path.join(REPO_ROOT, "explorations", "output")

SPEED_OF_LIGHT = 299_792_458.0          # m/s
WINDOW_THICKNESS_M = 0.9e-3             # nominal SiO2 window thickness
THETA_EXTERNAL_RAD = np.deg2rad(45.0)   # external angle of incidence
N_WINDOW_NOMINAL = 1.95

FFT_LENGTH = 8192                       # zero-padded length (dt=0.05 ps -> df~2.4 GHz)
TRUSTED_BAND_THZ = (0.25, 2.75)         # band used for metrics / inversion
PS_TO_S = 1e-12
HZ_TO_THZ = 1e-12


# ---------------------------------------------------------------- data access

def load_segment_scans(segment_name: str, filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_seconds, scan_matrix) for one segmented .acc file.

    scan_matrix has shape (n_time, n_scans) — one column per acquisition.
    """
    filepath = os.path.join(DATA_ROOT, segment_name, filename)
    thz_data = ACCLoader(filepath).load()
    raw = np.asarray(thz_data.raw_data, dtype=float)
    time_seconds = raw[:, 0] * PS_TO_S
    return time_seconds, raw[:, 1:]


def list_acc_filenames() -> list[str]:
    """Filenames present in BOTH segment folders, references first."""
    first_dir = os.path.join(DATA_ROOT, "first_reflection")
    second_dir = os.path.join(DATA_ROOT, "second_reflection")
    common = sorted(
        set(os.listdir(first_dir)) & set(os.listdir(second_dir))
    )
    return [f for f in common if f.endswith(".acc")]


# ---------------------------------------------------------------- spectra

def segment_spectrum(time_seconds: np.ndarray, amplitude: np.ndarray,
                     fft_length: int = FFT_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed, zero-padded spectrum referenced to ABSOLUTE time.

    The phase factor exp(-i*2*pi*f*t0) refers the FFT (whose implicit origin is
    the first sample) back to the absolute experiment time axis, so the
    second/first ratio carries the true inter-pulse delay as a linear phase.
    """
    dt = float(np.median(np.diff(time_seconds)))
    window = np.hanning(amplitude.size)
    spectrum = np.fft.rfft(amplitude * window, n=fft_length) * dt
    freq_hz = np.fft.rfftfreq(fft_length, dt)
    absolute_time_phase = np.exp(-2j * np.pi * freq_hz * time_seconds[0])
    return freq_hz, spectrum * absolute_time_phase


def band_mask(freq_hz: np.ndarray) -> np.ndarray:
    lo, hi = TRUSTED_BAND_THZ
    f_thz = freq_hz * HZ_TO_THZ
    return (f_thz >= lo) & (f_thz <= hi)


def envelope_peak_time(time_seconds: np.ndarray, amplitude: np.ndarray) -> float:
    """Peak time of |amplitude| with parabolic sub-sample refinement."""
    magnitude = np.abs(amplitude)
    i = int(np.argmax(magnitude))
    if 0 < i < magnitude.size - 1:
        y0, y1, y2 = magnitude[i - 1: i + 2]
        denom = (y0 - 2 * y1 + y2)
        offset = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        offset = 0.0
    dt = float(np.median(np.diff(time_seconds)))
    return time_seconds[i] + offset * dt


# ---------------------------------------------------------------- window model

def window_fresnel_factor(n_window_complex, theta_external_rad: float):
    """Fresnel amplitude factor of W: t_in * r_back * t_out / r_front (s-pol).

    r_front : air->window reflection at the external angle.
    t_in    : air->window transmission at the external angle.
    r_back  : window->air reflection at the internal angle.
    t_out   : window->air transmission back through the front face.
    """
    cos_e = np.cos(theta_external_rad)
    sin_e = np.sin(theta_external_rad)
    # window-side cosine via branch-safe sqrt (handles complex n)
    n_cos_internal = np.sqrt(n_window_complex**2 - sin_e**2)

    r_front = (cos_e - n_cos_internal) / (cos_e + n_cos_internal)
    t_in = 2.0 * cos_e / (cos_e + n_cos_internal)
    r_back = (n_cos_internal - cos_e) / (n_cos_internal + cos_e)
    t_out = 2.0 * n_cos_internal / (n_cos_internal + cos_e)
    return t_in * r_back * t_out / r_front


def invert_window_index(freq_hz: np.ndarray, w_measured: np.ndarray,
                        mask: np.ndarray, thickness_m: float,
                        theta_external_rad: float,
                        delay_estimate_s: float,
                        n_initial: float = N_WINDOW_NOMINAL,
                        n_iterations: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Extract complex n_window(omega) = n - i*k from measured W(omega).

    Model: W = F(n) * exp(-2i*(omega/c)*d*beta),  beta = sqrt(n^2 - sin^2(theta_e)).
    Per iteration: divide out F computed with the current n, then read beta from
    the remaining log; phase branch is anchored with the measured pulse delay.
    """
    sin_e = np.sin(theta_external_rad)
    omega = 2.0 * np.pi * freq_hz
    idx = np.where(mask)[0]

    n_complex = np.full(freq_hz.size, n_initial, dtype=complex)
    phase_w = np.full(freq_hz.size, np.nan)
    phase_w[idx] = np.unwrap(np.angle(w_measured[idx]))

    for _ in range(n_iterations):
        fresnel = window_fresnel_factor(n_complex[idx], theta_external_rad)
        fresnel_phase = np.unwrap(np.angle(fresnel))

        # Anchor the absolute 2*pi branch of the W phase using the measured
        # envelope delay: expected propagation phase at the anchor bin is
        # -omega*delay (group ~= phase delay for low-dispersion silica).
        anchor = 0
        expected_phase = fresnel_phase[anchor] - omega[idx][anchor] * delay_estimate_s
        branch = np.round((expected_phase - phase_w[idx][anchor]) / (2.0 * np.pi))
        phase_anchored = phase_w[idx] + 2.0 * np.pi * branch

        propagation_phase = phase_anchored - fresnel_phase
        log_magnitude = np.log(
            np.maximum(np.abs(w_measured[idx]) / np.maximum(np.abs(fresnel), 1e-30), 1e-30)
        )
        # exp(-2i*(omega/c)*d*beta) = |W/F| * exp(i*prop_phase)
        #   Re(beta) = -c*prop_phase / (2*omega*d)
        #   Im(beta) =  c*log|W/F| / (2*omega*d)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = (
                -SPEED_OF_LIGHT * propagation_phase / (2.0 * omega[idx] * thickness_m)
                + 1j * SPEED_OF_LIGHT * log_magnitude / (2.0 * omega[idx] * thickness_m)
            )
        n_new = np.sqrt(beta**2 + sin_e**2)
        n_new = np.where(n_new.real < 0, -n_new, n_new)
        n_complex[idx] = n_new

    n_out = np.full(freq_hz.size, np.nan)
    k_out = np.full(freq_hz.size, np.nan)
    n_out[idx] = n_complex[idx].real
    k_out[idx] = -n_complex[idx].imag  # n_hat = n - i*k convention
    return n_out, k_out


# ---------------------------------------------------------------- evaluation

def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.abs(values) ** 2)))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filenames = list_acc_filenames()
    reference_names = [f for f in filenames if f.startswith("reference")]
    sample_names = [f for f in filenames if f.startswith("sample")]
    print(f"Files: reference={reference_names}, samples={sample_names}\n")
    reference_name = reference_names[0]

    # ---- load everything -------------------------------------------------
    traces: dict[str, dict] = {}
    for name in filenames:
        t1, scans1 = load_segment_scans("first_reflection", name)
        t2, scans2 = load_segment_scans("second_reflection", name)
        freq, spec1 = segment_spectrum(t1, scans1.mean(axis=1))
        _, spec2 = segment_spectrum(t2, scans2.mean(axis=1))
        traces[name] = {
            "t1": t1, "scans1": scans1, "t2": t2, "scans2": scans2,
            "freq": freq, "Y1": spec1, "Y2": spec2,
            "R": spec2 / spec1,
        }
        print(f"{name}: {scans1.shape[1]} scans, "
              f"first {t1[0]*1e12:.2f}-{t1[-1]*1e12:.2f} ps, "
              f"second {t2[0]*1e12:.2f}-{t2[-1]*1e12:.2f} ps")

    freq = traces[reference_name]["freq"]
    f_thz = freq * HZ_TO_THZ
    mask = band_mask(freq)

    # ---- pulse delay & predicted delay -----------------------------------
    ref = traces[reference_name]
    t_peak_1 = envelope_peak_time(ref["t1"], ref["scans1"].mean(axis=1))
    t_peak_2 = envelope_peak_time(ref["t2"], ref["scans2"].mean(axis=1))
    measured_delay = t_peak_2 - t_peak_1
    cos_internal = np.sqrt(1 - (np.sin(THETA_EXTERNAL_RAD) / N_WINDOW_NOMINAL) ** 2)
    predicted_delay = 2 * N_WINDOW_NOMINAL * WINDOW_THICKNESS_M * cos_internal / SPEED_OF_LIGHT
    print(f"\nBare-window pulse delay: measured {measured_delay*1e12:.3f} ps, "
          f"model (n={N_WINDOW_NOMINAL}, d={WINDOW_THICKNESS_M*1e3} mm) "
          f"{predicted_delay*1e12:.3f} ps")
    implied_nd = measured_delay * SPEED_OF_LIGHT / 2.0  # = n*d*cos(theta_i) approx
    print(f"Implied n*d*cos(theta_i) = {implied_nd*1e3:.4f} mm "
          f"(nominal {N_WINDOW_NOMINAL * WINDOW_THICKNESS_M * cos_internal * 1e3:.4f} mm)")

    w_window = ref["R"]

    # ---- 1) leave-one-scan-out cross-validation on the bare window -------
    print("\n--- Leave-one-scan-out CV (bare window) ---")
    n_scans = ref["scans1"].shape[1]
    cv_residuals = []
    baseline_residuals = []
    for held_out in range(n_scans):
        train = [i for i in range(n_scans) if i != held_out]
        _, y1_train = segment_spectrum(ref["t1"], ref["scans1"][:, train].mean(axis=1))
        _, y2_train = segment_spectrum(ref["t2"], ref["scans2"][:, train].mean(axis=1))
        w_train = y2_train / y1_train
        _, y1_held = segment_spectrum(ref["t1"], ref["scans1"][:, held_out])
        _, y2_held = segment_spectrum(ref["t2"], ref["scans2"][:, held_out])
        y2_predicted = w_train * y1_held
        cv_residuals.append(
            rms((y2_predicted[mask] - y2_held[mask])) / rms(y2_held[mask]))
        # baseline: conventional approach = "use the other scans' Y2 directly"
        baseline_residuals.append(
            rms((y2_train[mask] - y2_held[mask])) / rms(y2_held[mask]))
    for i, (cv, base) in enumerate(zip(cv_residuals, baseline_residuals)):
        print(f"  scan {i}: predicted-from-Y1 residual {cv:.4f}   "
              f"direct-Y2-substitution residual {base:.4f}")
    print(f"  mean: predicted {np.mean(cv_residuals):.4f}, "
          f"direct {np.mean(baseline_residuals):.4f}")

    # ---- 2) mount-to-mount drift of the front pulse -----------------------
    print("\n--- Front-pulse drift D = Y1_sample / Y1_reference ---")
    fig_d, (ax_dm, ax_dp) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                         layout="constrained")
    drift_summary = {}
    for name in sample_names:
        drift = traces[name]["Y1"] / ref["Y1"]
        mag_dev = rms(np.abs(drift[mask]) - 1.0)
        # remove best-fit linear phase (pure timing) before quoting phase drift
        phase = np.unwrap(np.angle(drift[mask]))
        slope = np.polyfit(freq[mask], phase, 1)
        phase_nonlinear = phase - np.polyval(slope, freq[mask])
        drift_summary[name] = (mag_dev, rms(phase_nonlinear), slope[0])
        print(f"  {name}: |D|-1 rms {mag_dev:.4f}, nonlinear phase rms "
              f"{rms(phase_nonlinear):.4f} rad, timing part "
              f"{-slope[0]/(2*np.pi)*1e12*1e3:+.1f} fs")
        ax_dm.plot(f_thz[mask], np.abs(drift[mask]), label=name)
        ax_dp.plot(f_thz[mask], phase_nonlinear, label=name)
    ax_dm.axhline(1.0, color="0.5", lw=0.8)
    ax_dm.set_ylabel("|D|")
    ax_dm.set_title("Front-pulse drift D = Y1_sample / Y1_reference "
                    "(structure here = fake features in H_old)")
    ax_dp.axhline(0.0, color="0.5", lw=0.8)
    ax_dp.set_ylabel("arg(D) minus linear fit (rad)")
    ax_dp.set_xlabel("Frequency (THz)")
    ax_dm.legend(fontsize=8)
    fig_d.savefig(os.path.join(OUTPUT_DIR, "front_pulse_drift.png"), dpi=150)

    # ---- 3) H_old vs H_new ------------------------------------------------
    fig_h, axes_h = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 layout="constrained")
    for name in sample_names:
        h_old = traces[name]["Y2"] / ref["Y2"]
        h_new = traces[name]["R"] / w_window
        line, = axes_h[0].plot(f_thz[mask], np.abs(h_old[mask]), lw=1.0,
                               label=f"{name} H_old")
        axes_h[0].plot(f_thz[mask], np.abs(h_new[mask]), lw=1.0, ls="--",
                       color=line.get_color(), label=f"{name} H_new")
        phase_old = np.unwrap(np.angle(h_old[mask]))
        phase_new = np.unwrap(np.angle(h_new[mask]))
        axes_h[1].plot(f_thz[mask], phase_old, lw=1.0, color=line.get_color())
        axes_h[1].plot(f_thz[mask], phase_new, lw=1.0, ls="--",
                       color=line.get_color())
    axes_h[0].set_ylabel("|H|")
    axes_h[0].set_title("H_old = Y2s/Y2r (solid)  vs  H_new = R_s/W (dashed)")
    axes_h[1].set_ylabel("arg(H) (rad)")
    axes_h[1].set_xlabel("Frequency (THz)")
    axes_h[0].legend(fontsize=7, ncol=2)
    fig_h.savefig(os.path.join(OUTPUT_DIR, "transfer_old_vs_new.png"), dpi=150)

    # ---- 4) n_SiO2 extraction from W --------------------------------------
    print("\n--- Window index extraction from W ---")
    n_sio2, k_sio2 = invert_window_index(
        freq, w_window, mask, WINDOW_THICKNESS_M, THETA_EXTERNAL_RAD,
        delay_estimate_s=measured_delay,
    )
    inner = mask & (f_thz >= 0.4) & (f_thz <= 2.0)
    print(f"  n_SiO2 median (0.4-2.0 THz): {np.nanmedian(n_sio2[inner]):.4f} "
          f"(nominal {N_WINDOW_NOMINAL})")
    print(f"  k_SiO2 median (0.4-2.0 THz): {np.nanmedian(k_sio2[inner]):.4f}")

    fig_n, (ax_n, ax_k) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                       layout="constrained")
    ax_n.plot(f_thz[mask], n_sio2[mask])
    ax_n.axhline(N_WINDOW_NOMINAL, color="0.5", lw=0.8, ls=":",
                 label=f"nominal {N_WINDOW_NOMINAL}")
    ax_n.set_ylabel("n")
    ax_n.set_title(f"SiO2 window index from W (d = {WINDOW_THICKNESS_M*1e3} mm assumed)")
    ax_n.legend()
    ax_k.plot(f_thz[mask], k_sio2[mask])
    ax_k.set_ylabel("k")
    ax_k.set_xlabel("Frequency (THz)")
    fig_n.savefig(os.path.join(OUTPUT_DIR, "window_index.png"), dpi=150)

    # ---- 5) time-domain prediction check (bare window, per scan) ----------
    fig_t, ax_t = plt.subplots(figsize=(10, 5), layout="constrained")
    dt = float(np.median(np.diff(ref["t2"])))
    for held_out in range(n_scans):
        train = [i for i in range(n_scans) if i != held_out]
        _, y1_train = segment_spectrum(ref["t1"], ref["scans1"][:, train].mean(axis=1))
        _, y2_train = segment_spectrum(ref["t2"], ref["scans2"][:, train].mean(axis=1))
        _, y1_held = segment_spectrum(ref["t1"], ref["scans1"][:, held_out])
        y2_predicted_spec = (y2_train / y1_train) * y1_held
        # back to time on the second-reflection axis
        t_grid = ref["t2"][0] + dt * np.arange(FFT_LENGTH)
        time_phase = np.exp(2j * np.pi * freq * ref["t2"][0])
        y2_predicted_time = np.fft.irfft(y2_predicted_spec * time_phase / dt,
                                         n=FFT_LENGTH)
        n_pts = ref["t2"].size
        window = np.hanning(n_pts)
        measured = ref["scans2"][:, held_out] * window
        ax_t.plot(ref["t2"] * 1e12, measured, color="0.3", lw=1.0,
                  label="measured (held-out scan)" if held_out == 0 else None)
        ax_t.plot(t_grid[:n_pts] * 1e12, y2_predicted_time[:n_pts], ls="--", lw=1.0,
                  label="predicted from first reflection" if held_out == 0 else None)
    ax_t.set_xlabel("Time (ps)")
    ax_t.set_ylabel("Amplitude (windowed)")
    ax_t.set_title("Bare window: second reflection, measured vs predicted (per scan)")
    ax_t.legend()
    fig_t.savefig(os.path.join(OUTPUT_DIR, "time_domain_prediction.png"), dpi=150)

    print(f"\nFigures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
