"""DEMO: front-reflection window EXTENT & SYMMETRY when pre-pulse is limited.

CNT-17 has only ~2.3 ps of pre-pulse before the first reflection but ~11 ps to the
second reflection. A symmetric peak-centred window of MAXIMUM symmetric extent is
therefore only ~4.6 ps wide and discards ~9 ps of usable post-pulse oscillation.

Question (Samuel): if the first reflection gets a SHORTER time window, do we get a
different distribution of frequencies, or does it modify the PHASE? And does that
force us to keep the prepend+taper centering as an optional tool?

Three front-pulse window strategies, all built from the same half-Hann taper logic:
    narrow_sym  : symmetric, half-width = available pre-pulse (~2.3 ps). Drops tail.
    asym        : 2.3 ps pre + ~8 ps post (keeps tail, ASYMMETRIC about the peak).
    prepend_sym : prepend zeros to make pre = post = ~8 ps, symmetric (keeps tail).

We measure, for each:
  - |Y1(f)|  -> frequency distribution / resolution / leakage  (panel B)
  - phase(Y1) vs the prepend_sym reference -> phase distortion  (panel C)
  - pulse CENTROID time -> the group-delay shift an asymmetric window induces
  - phase(H) with the SAME strategy applied to sample AND reference first pulses,
    second reflection held at a fixed symmetric window -> does the front-window
    choice propagate to the MEASUREMENT, or cancel via self-referencing? (panel D)

Run:
    .venv/Scripts/python.exe explorations/demo_window_extent.py
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.dataset import DataSet  # noqa: E402

ROOT_DIR = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-17"
OUTPUT_DIR = os.path.join(REPO_ROOT, "explorations", "output")
DT = 50e-15
N_FFT = 2048
PHASE_BAND_THZ = (0.1, 4.5)
_S_TO_PS = 1e12

first_reflection_range = (152, 158.5)  # ps
second_reflection_range = (161, 168)  # ps

def load_trace(ds, key):
    for fn, obj in ds.data.items():
        if key in fn.lower():
            raw = np.asarray(obj.raw_data, float)
            t_ps = raw[:, 0]
            y = raw[:, 1:].mean(axis=1) if raw.shape[1] > 2 else raw[:, 1]
            y = y - y[:20].mean()  # baseline off the genuine pre-pulse region
            return fn, t_ps * 1e-12, y
    raise SystemExit(f"no trace matching {key!r}")


def peak_index(t, y, lo_ps, hi_ps):
    m = (t * _S_TO_PS >= lo_ps) & (t * _S_TO_PS <= hi_ps)
    idx = np.where(m)[0]
    return idx[int(np.argmax(np.abs(y[idx])))]


def windowed(t, y, peak_idx, h_pre, h_post):
    """Half-Hann rise over h_pre samples, fall over h_post; zeros outside.

    Prepends/appends zeros (linear time extension) if the window runs off the array.
    Returns (t_ext, y_win, t_peak) — y_win is full-length on the (maybe extended) axis.
    """
    t = t.copy(); y = y.copy()
    start = peak_idx - h_pre
    if start < 0:
        pad = -start
        t = np.concatenate([t[0] - DT * np.arange(pad, 0, -1), t])
        y = np.concatenate([np.zeros(pad), y])
        peak_idx += pad
        start = peak_idx - h_pre
    end = peak_idx + h_post + 1
    if end > len(y):
        pad = end - len(y)
        t = np.concatenate([t, t[-1] + DT * np.arange(1, pad + 1)])
        y = np.concatenate([y, np.zeros(pad)])
    win = np.zeros(len(y))
    win[start:peak_idx] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, h_pre, endpoint=False)))
    win[peak_idx:peak_idx + h_post + 1] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, h_post + 1)))
    return t, y * win, t[peak_idx]


def fft_abs(t, y):
    """rfft referenced to absolute time 0 (so phases are comparable across axes)."""
    freq = np.fft.rfftfreq(N_FFT, DT)
    Y = np.fft.rfft(y, n=N_FFT) * np.exp(-2j * np.pi * np.fft.rfftfreq(N_FFT, DT) * t[0])
    return freq, Y


def centroid_ps(t, y):
    w = np.abs(y) ** 2
    return float(np.sum(t * w) / np.sum(w)) * _S_TO_PS


def safe_ratio(num, den, floor_rel=1e-3):
    # Floor on the AC content (exclude the DC bin): an asymmetric window injects a
    # large DC term that would otherwise inflate the floor and null the whole band.
    floor = np.abs(den)[1:].max() * floor_rel
    out = np.full_like(num, np.nan + 1j * np.nan)
    good = np.abs(den) >= floor
    out[good] = num[good] / den[good]
    return out

def crop_array(t, y, lo_ps, hi_ps):
    """Crop t, y to the given time range (ps)."""
    m = (t * _S_TO_PS >= lo_ps) & (t * _S_TO_PS <= hi_ps)
    return t[m], y[m]

def window_symmetric(t, y, peak_idx, half_width):
    """Half-Hann rise/fall over half_width samples, symmetric about peak_idx."""
    t_windowed, y_windowed, t_peak = windowed(t, y, peak_idx, half_width, half_width)
    plt.plot(t_windowed * _S_TO_PS, y_windowed, lw=1.5, label=f"half-width {half_width} samp")
    plt.plot(t * _S_TO_PS, y, color="0.6", lw=1, label="raw")
    plt.legend()
    plt.show()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds = DataSet(ROOT_DIR)
    ds.load_all_data(case_insensitive=True, explicit_dir=True)
    s_name, ts, ys = load_trace(ds, "sample_a-40.0_s-0")
    r_name, tr, yr = load_trace(ds, "reference_a-40.0_s-0")

    i1s = peak_index(ts, ys, *first_reflection_range);  i2s = peak_index(ts, ys, *second_reflection_range)
    i1r = peak_index(tr, yr, *first_reflection_range);  i2r = peak_index(tr, yr, *second_reflection_range)

    sample_cropped = np.column_stack(crop_array(ts, ys, *second_reflection_range))
    reference_cropped = np.column_stack(crop_array(tr, yr, *first_reflection_range))

    plt.plot(sample_cropped[:, 0] * _S_TO_PS, sample_cropped[:, 1], color="0.6", lw=1, label="sample")
    plt.plot(reference_cropped[:, 0] * _S_TO_PS, reference_cropped[:, 1], color="0.3", lw=1, label="reference")
    # plt.show()



    h_pre_avail = i1s                      # samples of pre-pulse on sample front
    h_post_full = 160                      # 8 ps post
    h_second = 100                         # symmetric 5 ps each side for back pulse

    print("=== front-window extent demo — CNT-17 s-0 ===")
    print(f"sample={s_name}\nref={r_name}")
    print(f"front peak {ts[i1s]*_S_TO_PS:.2f} ps, back peak {ts[i2s]*_S_TO_PS:.2f} ps")
    print(f"pre-pulse available: {h_pre_avail*DT*_S_TO_PS:.2f} ps "
          f"({h_pre_avail} samp); post used: {h_post_full*DT*_S_TO_PS:.2f} ps")

    strategies = {
        "narrow_sym":  dict(h_pre=h_pre_avail, h_post=h_pre_avail),
        "asym":        dict(h_pre=h_pre_avail, h_post=h_post_full),
        "prepend_sym": dict(h_pre=h_post_full, h_post=h_post_full),
    }

    # fixed symmetric back-pulse spectra (same for every front strategy)
    _, y2s, _ = windowed(ts, ys, i2s, h_second, h_second)
    _, y2r, _ = windowed(tr, yr, i2r, h_second, h_second)
    t2s_axis, y2s_w, _ = windowed(ts, ys, i2s, h_second, h_second)
    t2r_axis, y2r_w, _ = windowed(tr, yr, i2r, h_second, h_second)
    freq, Y2s = fft_abs(t2s_axis, y2s_w)
    _,    Y2r = fft_abs(t2r_axis, y2r_w)
    f_thz = freq * 1e-12
    pb = (f_thz >= PHASE_BAND_THZ[0]) & (f_thz <= PHASE_BAND_THZ[1])

    results = {}
    for name, hw in strategies.items():
        t1s_axis, y1s_w, tpk_s = windowed(ts, ys, i1s, hw["h_pre"], hw["h_post"])
        t1r_axis, y1r_w, tpk_r = windowed(tr, yr, i1r, hw["h_pre"], hw["h_post"])
        _, Y1s = fft_abs(t1s_axis, y1s_w)
        _, Y1r = fft_abs(t1r_axis, y1r_w)
        H = safe_ratio(safe_ratio(Y2s, Y1s), safe_ratio(Y2r, Y1r))
        results[name] = dict(
            t1=t1s_axis, y1=y1s_w, Y1s=Y1s, H=H,
            centroid=centroid_ps(t1s_axis, y1s_w), tpk=tpk_s * _S_TO_PS,
        )

    ref = results["prepend_sym"]   # symmetric + full info = the cleanest reference
    print("\nfront-pulse centroid vs peak (asymmetry pulls centroid off the peak):")
    for name, r in results.items():
        print(f"  {name:12s} centroid {r['centroid']:8.4f} ps  "
              f"(peak {r['tpk']:.4f}, shift {r['centroid']-r['tpk']:+.4f} ps)")

    print("\n|Y1| spectral RMS difference vs prepend_sym (normalised):")
    for name, r in results.items():
        a = np.abs(r["Y1s"][pb]); b = np.abs(ref["Y1s"][pb])
        a = a / a.max(); b = b / b.max()
        print(f"  {name:12s} {np.sqrt(np.mean((a-b)**2)):.4f}")

    trusted = (f_thz >= 0.5) & (f_thz <= 4.5)
    print("\nphase(H) error vs prepend_sym -> does front-window choice reach the MEASUREMENT:")
    print(f"  {'strategy':12s} {'rms full (0.1-4.5)':>20s} {'rms trusted (0.5-4.5)':>22s}")
    for name, r in results.items():
        d_full = np.angle(r["H"][pb] * np.conj(ref["H"][pb]))
        d_tr = np.angle(r["H"][trusted] * np.conj(ref["H"][trusted]))
        print(f"  {name:12s} {np.sqrt(np.nanmean(d_full**2)):>20.4f} "
              f"{np.sqrt(np.nanmean(d_tr**2)):>22.4f}")

    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    fig.suptitle("Front-reflection window extent & symmetry — CNT-17 s-0", fontsize=12)

    ax = axes[0, 0]
    ax.plot(ts * _S_TO_PS, ys, color="0.6", lw=1, label="raw (baseline-subtracted)")
    for name, r in results.items():
        ax.plot(r["t1"] * _S_TO_PS, r["y1"], lw=1.2, label=name)
    ax.axvline(ts[i2s] * _S_TO_PS, color="k", ls=":", lw=1, label="2nd reflection")
    ax.set_xlim(150, 168)
    ax.set_title("(A) front pulse — what each window keeps")
    ax.set_xlabel("time (ps)"); ax.set_ylabel("amplitude"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name, r in results.items():
        m = np.abs(r["Y1s"]); m = m / m[pb].max()
        ax.semilogy(f_thz[pb], m[pb], lw=1.2, label=name)
    ax.set_title("(B) |Y1(f)| (normalised) — frequency distribution")
    ax.set_xlabel("frequency (THz)"); ax.set_ylabel("|Y1| (norm, log)"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name, r in results.items():
        if name == "prepend_sym":
            continue
        d = np.unwrap(np.angle(r["Y1s"][pb] * np.conj(ref["Y1s"][pb])))
        ax.plot(f_thz[pb], d, lw=1.2, label=f"{name} - prepend_sym")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_title("(C) phase(Y1) distortion vs symmetric-full reference")
    ax.set_xlabel("frequency (THz)"); ax.set_ylabel("Δ phase (rad)"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    for name, r in results.items():
        if name == "prepend_sym":
            continue
        d = np.unwrap(np.angle(r["H"][pb] * np.conj(ref["H"][pb])))
        ax.plot(f_thz[pb], d, lw=1.2, label=f"{name} - prepend_sym")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_title("(D) phase(H) ERROR vs symmetric-full — the n-relevant bias")
    ax.set_xlabel("frequency (THz)"); ax.set_ylabel("Δ phase(H) (rad)"); ax.legend(fontsize=8)

    out = os.path.join(OUTPUT_DIR, "demo_window_extent.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
    # plt.show()
