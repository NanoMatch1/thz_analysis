"""DEMO: does the FFT time-origin (T0) reference change the self-referenced H?

Audit for ANALYSIS_NOTES §11/§14. Self-referencing forms
    H = (Y2_samp/Y1_samp) / (Y2_ref/Y1_ref)
from the first (front) and second (sample) reflection FFTs. An FFT references its
phase to the FIRST SAMPLE of its array (the segment's own T0). To reference it to
absolute experiment time instead, multiply by exp(-2*pi*i*f*t[0]).

The production code currently does this ASYMMETRICALLY: the first segment gets the
absolute-time factor, the second does not. This script computes H three ways on
ONE real sample + reference (CNT-17, s-0) after the real per-segment processing,
to show what that choice does to the phase:

    own  : both segments referenced to their own array T0 (plain rfft)
    abs  : both segments referenced to absolute time   (rfft * exp(-2pi i f t0))
    prod : first=abs, second=own  (what transfer_function currently builds)

|H| is invariant under any phase ramp, so all three share one magnitude curve —
the referencing is a PHASE-only effect (it moves n / group delay, not |r|).

Run:
    .venv/Scripts/python.exe explorations/demo_phase_referencing.py
"""

import os
import sys

import numpy as np
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.dataset import DataSet  # noqa: E402
from dataset_core.adapters import thz_adapter as thz  # noqa: E402

ROOT_DIR = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-17"
OUTPUT_DIR = os.path.join(REPO_ROOT, "explorations", "output")
SAMPLE_KEY = "s-0"          # pick the s-0 orientation
BAND_THZ = (0.0, 5.0)
PHASE_BAND_THZ = (0.1, 4.5)  # where |Y1| is meaningful for phase display
_S_TO_PS = 1e12


N_FFT = 4096  # shared FFT length so all four segments share one frequency grid


def fft_two_ways(t, y, n_fft=N_FFT):
    """Return (freq, Y_own, Y_abs, t0) for a time-domain segment (t, y).

    Y_own : rfft(y, n=n_fft) — phase origin = the segment's own array sample 0
            (i.e. its own T0 = absolute time t[0], with zeros appended to n_fft).
    Y_abs : Y_own * exp(-2*pi*i*f*t[0]) — phase origin = absolute time 0.

    The own-vs-abs distinction only survives if segments have DIFFERENT t[0].
    """
    t = np.asarray(t)
    y = np.asarray(y)
    dt = float(np.median(np.diff(t)))
    freq = np.fft.rfftfreq(n_fft, dt)
    y_own = np.fft.rfft(y, n=n_fft)
    y_abs = y_own * np.exp(-2j * np.pi * freq * t[0])
    return freq, y_own, y_abs, t[0]


def safe_ratio(num, den, floor_rel=1e-3):
    den_mag = np.abs(den)
    floor = den_mag.max() * floor_rel
    out = np.full_like(num, np.nan + 1j * np.nan)
    good = den_mag >= floor
    out[good] = num[good] / den[good]
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- load + real per-segment processing (mirrors run_me_reflection.py) ---
    dataset = DataSet(ROOT_DIR)
    dataset.load_all_data(case_insensitive=True)  # auto-detect reflection layout
    dataset.group_files(keywords=["type"])

    # Front-pulse timing calibration (sample remounted): integer shift to both
    # segments, sub-sample residual stored on the outer object.
    thz.align_to_reference(dataset, timing_segment="first_reflection")

    centering = {"centering": {"peak_mode": "auto", "taper_ps": 1}}
    window = {"window": {"type": "Hann", "alpha": 1}}
    pad = {"pad": {"n_samples": N_FFT}}

    # Process up to and including window_time, then SNAPSHOT the time-domain
    # segments (this is BEFORE zero_pad's common-grid re-referencing). Then run
    # zero_pad and snapshot again (production point, just before fft_spectrum).
    # NOTE: pre_window_align_peak omitted — windowing hygiene (preserves absolute
    # time), stale vs center_pulse, and needs interaction; irrelevant to this demo.
    for seg in ("second_reflection", "first_reflection"):
        thz.subtract_baseline(dataset, segment=seg)
        thz.global_truncate(dataset, segment=seg)
        thz.center_pulse(dataset, segment=seg, config=centering)
        thz.window_time(dataset, segment=seg, config=window, show_graph=False)


    # --- pick one sample + its reference ---
    sample_obj = None
    for filename, obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        if SAMPLE_KEY in filename.lower():
            sample_obj = obj
            sample_name = filename
            break
    if sample_obj is None:
        raise SystemExit(f"No sample matching '{SAMPLE_KEY}' found.")
    ref_obj = dataset.get_reference(sample_name, ref_type="reference")
    ref_name = ref_obj.filename

    def snapshot(obj):
        """Return (t, y) for the (first, second) segments of obj at the current step."""
        return ((obj.first_segment.data[:, 0].copy(), obj.first_segment.data[:, 1].copy()),
                (obj.data[:, 0].copy(), obj.data[:, 1].copy()))

    pre = {"sample": snapshot(sample_obj), "ref": snapshot(ref_obj)}

    for seg in ("second_reflection", "first_reflection"):
        thz.zero_pad(dataset, segment=seg, config=pad)
    post = {"sample": snapshot(sample_obj), "ref": snapshot(ref_obj)}

    # ---------------------------------------------------------------------
    # Build H three ways for a given snapshot.
    # ---------------------------------------------------------------------
    def transfer_three_ways(snap):
        (t1s, y1s), (t2s, y2s) = snap["sample"]
        (t1r, y1r), (t2r, y2r) = snap["ref"]
        freq, Y1s_own, Y1s_abs, a1s = fft_two_ways(t1s, y1s)
        _,    Y2s_own, Y2s_abs, a2s = fft_two_ways(t2s, y2s)
        _,    Y1r_own, Y1r_abs, a1r = fft_two_ways(t1r, y1r)
        _,    Y2r_own, Y2r_abs, a2r = fft_two_ways(t2r, y2r)

        H_own = safe_ratio(safe_ratio(Y2s_own, Y1s_own), safe_ratio(Y2r_own, Y1r_own))
        H_abs = safe_ratio(safe_ratio(Y2s_abs, Y1s_abs), safe_ratio(Y2r_abs, Y1r_abs))
        # production: first=abs, second=own
        H_prod = safe_ratio(safe_ratio(Y2s_own, Y1s_abs), safe_ratio(Y2r_own, Y1r_abs))
        t0s = dict(t1s=a1s, t2s=a2s, t1r=a1r, t2r=a2r)
        return freq, H_own, H_abs, H_prod, t0s

    freq, H_own_pre, H_abs_pre, H_prod_pre, t0_pre = transfer_three_ways(pre)
    _,    H_own_post, H_abs_post, H_prod_post, t0_post = transfer_three_ways(post)
    f_thz = freq * 1e-12
    pb = (f_thz >= PHASE_BAND_THZ[0]) & (f_thz <= PHASE_BAND_THZ[1])

    def report(tag, t0s, H_own, H_abs):
        d = ((t0s["t2s"] - t0s["t1s"]) - (t0s["t2r"] - t0s["t1r"])) * _S_TO_PS
        print(f"\n--- {tag} ---")
        print(f"  array t[0] (ps): first  samp {t0s['t1s']*_S_TO_PS:9.4f}  "
              f"ref {t0s['t1r']*_S_TO_PS:9.4f}  (d {(t0s['t1s']-t0s['t1r'])*_S_TO_PS:+.4f})")
        print(f"  array t[0] (ps): second samp {t0s['t2s']*_S_TO_PS:9.4f}  "
              f"ref {t0s['t2r']*_S_TO_PS:9.4f}  (d {(t0s['t2s']-t0s['t2r'])*_S_TO_PS:+.4f})")
        print(f"  Delta = (t2s-t1s)-(t2r-t1r) = {d:+.4f} ps")
        ratio = H_abs * np.conj(H_own)
        ph = np.unwrap(np.angle(ratio[pb]))
        slope = np.polyfit(freq[pb], ph, 1)[0]
        dt_equiv = -slope / (2 * np.pi) * _S_TO_PS
        mag = np.nanmax(np.abs(np.abs(H_own[pb]) - np.abs(H_abs[pb])))
        print(f"  |H| own-vs-abs max diff: {mag:.2e}  (phase-ramp invariant -> ~0)")
        print(f"  phase(H_abs/H_own) slope -> equivalent {dt_equiv:+.4f} ps of phase error")

    print("\n=== T0 reference demo — CNT-17 ===")
    print(f"sample = {sample_name}\nref    = {ref_name}")
    report("PRE-pad  (FFT each segment at its own array origin)", t0_pre, H_own_pre, H_abs_pre)
    report("POST-pad (production: zero_pad common-grid, then FFT)", t0_post, H_own_post, H_abs_post)
    print("\nTakeaway: zero_pad's pad_to_common_grid puts every trace of a segment on a")
    print("shared absolute origin, so own==abs==prod after padding. The own-T0 error is")
    print("only present if you FFT BEFORE the common-grid step.")

    # ---------------------------------------------------------------------
    # figure: PRE vs POST, phase(H) and |H|
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    fig.suptitle(f"FFT T0 reference (own vs absolute) — CNT-17 {sample_name}", fontsize=12)

    ax = axes[0, 0]
    ax.plot(f_thz[pb], np.unwrap(np.angle(H_own_pre[pb])), label="own", lw=1.4)
    ax.plot(f_thz[pb], np.unwrap(np.angle(H_abs_pre[pb])), "--", label="abs", lw=1.4)
    ax.plot(f_thz[pb], np.unwrap(np.angle(H_prod_pre[pb])), ":", label="prod (current)", lw=1.4)
    ax.set_title("(A) phase(H) PRE-pad — own vs abs DIVERGE")
    ax.set_xlabel("Frequency (THz)"); ax.set_ylabel("unwrapped phase (rad)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(f_thz[pb], np.unwrap(np.angle(H_own_post[pb])), label="own", lw=1.4)
    ax.plot(f_thz[pb], np.unwrap(np.angle(H_abs_post[pb])), "--", label="abs", lw=1.4)
    ax.plot(f_thz[pb], np.unwrap(np.angle(H_prod_post[pb])), ":", label="prod (current)", lw=1.4)
    ax.set_title("(B) phase(H) POST-pad (production) — methods COLLAPSE")
    ax.set_xlabel("Frequency (THz)"); ax.set_ylabel("unwrapped phase (rad)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(f_thz[pb], np.angle(H_own_pre[pb] * np.conj(H_abs_pre[pb])), label="pre-pad", lw=1.4)
    ax.plot(f_thz[pb], np.angle(H_own_post[pb] * np.conj(H_abs_post[pb])), "--", label="post-pad", lw=1.4)
    ax.set_title("(C) phase(H_own) - phase(H_abs): the residual the choice introduces")
    ax.set_xlabel("Frequency (THz)"); ax.set_ylabel("phase difference (rad)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(f_thz[pb], np.abs(H_own_post[pb]), label="|H| own", lw=1.4)
    ax.plot(f_thz[pb], np.abs(H_abs_post[pb]), "--", label="|H| abs", lw=1.4)
    ax.set_title("(D) |H| (post-pad) — phase-only effect, magnitude untouched")
    ax.set_xlabel("Frequency (THz)"); ax.set_ylabel("|H|")
    ax.legend(fontsize=8)

    out_path = os.path.join(OUTPUT_DIR, "demo_phase_referencing.png")
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved figure: {out_path}")


if __name__ == "__main__":
    main()
    plt.show()
