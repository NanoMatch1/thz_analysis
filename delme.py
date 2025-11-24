import numpy as np
import matplotlib.pyplot as plt


def generate_gaussian_thz_pulse(t, f0_thz=1.5, tau_ps=0.3, phase=0.0):
    """
    Generate a single-cycle (or few-cycle) THz pulse:
    Gaussian envelope × cosine carrier.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds.
    f0_thz : float
        Carrier frequency in THz (centre of the broadband).
    tau_ps : float
        RMS width of the Gaussian envelope in picoseconds.
        Smaller tau -> shorter pulse -> broader spectrum.
    phase : float
        Phase of the carrier in radians.

    Returns
    -------
    E : np.ndarray
        Time-domain electric field (arb. units).
    """
    t_ps = t * 1e12  # convert to ps
    envelope = np.exp(-0.5 * (t_ps / tau_ps) ** 2)  # Gaussian envelope
    carrier = np.cos(2 * np.pi * f0_thz * 1e12 * t + phase)
    return envelope * carrier


def compute_spectrum(t, s):
    """
    Compute one-sided amplitude spectrum of a real signal.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds, uniformly spaced.
    s : np.ndarray
        Signal array.

    Returns
    -------
    freqs_thz : np.ndarray
        Frequency axis in THz (one-sided).
    amp : np.ndarray
        One-sided amplitude spectrum (arb. units).
    """
    dt = t[1] - t[0]
    N = len(t)

    S = np.fft.fft(s)
    freqs = np.fft.fftfreq(N, d=dt)  # Hz

    mask = freqs >= 0
    freqs_pos = freqs[mask]
    S_pos = S[mask]

    amp = np.abs(S_pos) / N
    return freqs_pos / 1e12, amp  # THz


def simulate_for_dt(dt_fs, t_window_ps=10.0, f0_thz=1.5, tau_ps=0.3):
    """
    Generate a THz pulse and its spectrum for a given sampling interval.

    Parameters
    ----------
    dt_fs : float
        Sampling interval in femtoseconds.
    t_window_ps : float
        Total time window in picoseconds (0 .. t_window_ps).
    f0_thz : float
        Carrier frequency in THz.
    tau_ps : float
        RMS width of the Gaussian envelope in ps.

    Returns
    -------
    t : np.ndarray
        Time axis in seconds.
    E : np.ndarray
        Time-domain signal.
    f_thz : np.ndarray
        Frequency axis in THz.
    amp : np.ndarray
        Amplitude spectrum.
    """
    dt = dt_fs * 1e-15          # fs -> s
    t_max = t_window_ps * 1e-12 # ps -> s

    t = np.arange(0, t_max, dt)
    E = generate_gaussian_thz_pulse(t, f0_thz=f0_thz, tau_ps=tau_ps)

    f_thz, amp = compute_spectrum(t, E)
    return t, E, f_thz, amp


def main():
    # --- Define a "realistic" THz pulse ---
    # Centre ~1.5 THz, bandwidth ~0.5–3 THz (tunable via tau_ps)
    f0_thz = 1.5
    tau_ps = 0.3        # try 0.2–0.4 ps to see bandwidth changes
    t_window_ps = 10.0  # time window

    # Very fine sampling for reference (almost continuous)
    dt_ref_fs = 1.0
    t_ref, E_ref, f_ref_thz, amp_ref = simulate_for_dt(
        dt_ref_fs, t_window_ps, f0_thz, tau_ps
    )

    # Sampling intervals to test (fs)
    sampling_dts_fs = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0]

    # --- Time-domain comparison (zoom around the pulse) ---
    plt.figure(figsize=(10, 6))

    t_ref_ps = t_ref * 1e12
    # zoom to central region (e.g. 2–8 ps to catch the pulse)
    t_min_zoom, t_max_zoom = 2.0, 8.0
    mask_zoom_ref = (t_ref_ps >= t_min_zoom) & (t_ref_ps <= t_max_zoom)

    plt.plot(t_ref_ps[mask_zoom_ref], E_ref[mask_zoom_ref],
             label=f"Reference ({dt_ref_fs:.1f} fs)", linewidth=2)

    colors = plt.cm.viridis(np.linspace(0, 1, len(sampling_dts_fs)))

    for dt_fs, c in zip(sampling_dts_fs, colors):
        t, E, _, _ = simulate_for_dt(dt_fs, t_window_ps, f0_thz, tau_ps)
        t_ps = t * 1e12
        mask_zoom = (t_ps >= t_min_zoom) & (t_ps <= t_max_zoom)

        dt = dt_fs * 1e-15
        f_nyq_thz = 1.0 / (2 * dt) / 1e12

        plt.plot(t_ps[mask_zoom], E[mask_zoom], ".-", markersize=3,
                 color=c, alpha=0.8,
                 label=f"{dt_fs:.0f} fs (Nyquist ~ {f_nyq_thz:.1f} THz)")

    plt.xlabel("Time (ps)")
    plt.ylabel("E(t) (arb. units)")
    plt.title("Gaussian-envelope THz pulse sampled at different intervals")
    plt.legend()
    plt.grid(True)

    # --- Frequency-domain comparison ---
    plt.figure(figsize=(10, 6))
    plt.plot(f_ref_thz, amp_ref,
             label=f"Reference ({dt_ref_fs:.1f} fs)", linewidth=2)

    for dt_fs, c in zip(sampling_dts_fs, colors):
        t, E, f_thz, amp = simulate_for_dt(dt_fs, t_window_ps, f0_thz, tau_ps)

        dt = dt_fs * 1e-15
        f_nyq_thz = 1.0 / (2 * dt) / 1e12

        plt.plot(f_thz, amp, "-", color=c, alpha=0.8,
                 label=f"{dt_fs:.0f} fs (Nyquist ~ {f_nyq_thz:.1f} THz)")
        plt.axvline(f_nyq_thz, color=c, linestyle="--", alpha=0.4)

    plt.xlim(0, 6)  # focus on 0–6 THz
    plt.xlabel("Frequency (THz)")
    plt.ylabel("|E(f)| (arb. units)")
    plt.title("Spectrum of THz pulse vs sampling interval")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
