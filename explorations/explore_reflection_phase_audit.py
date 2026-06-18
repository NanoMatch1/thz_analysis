"""Audit the reflection-pipeline phase treatment on real CNT-17 data.

Samuel's concern: CNT reflection n drops below 1 at higher frequency and won't fit
a model; he suspects instrumental (air gap, ANALYSIS_NOTES §9) but wants to rule
out a code-side phase bug analogous to the transmission one (the complex-H
round-trip + invert_nk re-unwrap that lost integer 2*pi cycles).

What this checks, headless on CNT-17 (window-coupled, shared-axis path):
  1. Is the reflection inversion immune to the integer-cycle / unwrap-branch issue?
     (It should be: `invert_nk_reflection` is closed-form on the COMPLEX r, with no
     unwrap and no DC anchor — exp(i*2*pi*m) is invisible to it.)
  2. Is there a residual LINEAR phase in phi(H)? A constant group delay Δt shows up
     as a straight slope; in reflection that is the air-gap signature (2*gap/c),
     i.e. instrumental, NOT a unit-cell bug. Reported as an equivalent delay in ps.
  3. Is there a constant phase offset (intercept) near a multiple of pi? That would
     hint at a sign / branch-convention problem in the code.
  4. Where does n cross 1, and does removing the fitted linear phase (a pure timing
     de-embed) lift n back above 1? If yes -> the drop is a delay (instrumental),
     not the inversion.

No pipeline code is modified.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
ROOT_DIR = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-17"
SECOND_REFLECTION_DIR = os.path.join(ROOT_DIR, "second_reflection")
TRUSTED_BAND_HZ = (0.5e12, 2.5e12)

CONFIG = {
    "theta_external_deg": 45.0,
    "polarization": "s",
    "n_sio2": 1.95,
    "center_mode": "crop",
    "n_fft": 4096,
    "regions": {
        "first_reflection": (152.0, 158.5),
        "second_reflection": (161.0, 168.0),
    },
    "window": {"type": "hann", "alpha": 1.0},
    "pad_start": {"enabled": False},
}


def build_dataset():
    # Point at the pre-segmented second_reflection subdir; transfer_function with
    # self_reference=True finds the sibling first_reflection/ folder. This is the
    # original Phase-2 reflection workflow (readme), the representative path.
    dataset = DataSet(SECOND_REFLECTION_DIR)
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)
    return dataset


def run_segmented(dataset):
    thz.subtract_baseline(dataset, show_graph=False)
    dataset.group_files(keywords=["type"])
    thz.global_truncate(dataset)
    thz.window_time(dataset, config={"window": CONFIG["window"]}, show_graph=False)
    thz.zero_pad(dataset, config={"pad": {"extend_factor": 3.0}}, show_graph=False)
    thz.fft_spectrum(dataset)
    thz.transfer_function(dataset, config={"transfer": {"self_reference": True,
                          "apply_snr_mask": True},
                          "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25,
                                   "min_contiguous_bins": 3}}, ref_type="reference")
    thz.invert_nk_reflection(dataset, geometry="window",
                             theta_deg=CONFIG["theta_external_deg"],
                             polarization=CONFIG["polarization"],
                             n_window=CONFIG["n_sio2"])
    return dataset


def line_fit(freq, phase, mask):
    sel = mask & np.isfinite(phase)
    if np.count_nonzero(sel) < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(freq[sel], phase[sel], 1)
    return slope, intercept


def main():
    dataset = run_segmented(build_dataset())

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), layout="constrained")
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        processing = data_obj.processing_dict
        freq = processing.get("fft_freq")
        H = processing.get("transfer_H")
        r = processing.get("reflection_r")
        n = processing.get("n")
        k = processing.get("k")
        mask = processing.get("transfer_mask")
        if freq is None or H is None or n is None:
            continue
        band = (freq >= TRUSTED_BAND_HZ[0]) & (freq <= TRUSTED_BAND_HZ[1])
        if mask is not None:
            band = band & mask

        H_phase = np.full(freq.size, np.nan)
        fin = np.isfinite(H)
        H_phase[fin] = np.unwrap(np.angle(H[fin]))
        slope, intercept = line_fit(freq, H_phase, band)
        equiv_delay_ps = -slope / (2 * np.pi) * 1e12
        intercept_mod = (intercept + np.pi) % (2 * np.pi) - np.pi
        f_thz = freq * 1e-12

        # De-embed ONLY the fitted linear phase (a pure timing / air-gap removal):
        # multiply r by exp(-i*slope*f). Then re-invert with the SAME geometry the
        # pipeline used and see whether n stops dropping below 1.
        theta_internal_rad = processing.get("theta_internal_rad", 0.0)
        r_deembed = r * np.exp(-1j * slope * freq)
        n_deembed, k_deembed, _ = thz.core.invert_nk_reflection(
            freq, r_deembed, mask, theta_rad=theta_internal_rad,
            polarization=CONFIG["polarization"], n_incident=CONFIG["n_sio2"],
        )

        good = np.isfinite(n) & band
        def at(arr, t):
            return np.interp(t, f_thz[good], arr[good]) if good.any() else np.nan
        below = np.where((n < 1.0) & good)[0]
        cross = f_thz[below[0]] if below.size else np.nan

        print(f"\n=== {filename} ===")
        print(f"  phi(H) over {TRUSTED_BAND_HZ[0]*1e-12}-{TRUSTED_BAND_HZ[1]*1e-12} THz:")
        print(f"    linear slope -> equivalent delay = {equiv_delay_ps:+.4f} ps "
              f"(air gap = c*dt/2 ~ {abs(equiv_delay_ps)*1e-12*SPEED_OF_LIGHT_M_PER_S/2*1e6:.1f} um)")
        print(f"    intercept = {intercept:+.3f} rad ({intercept/np.pi:+.2f} pi); "
              f"mod 2pi = {intercept_mod:+.3f} rad ({intercept_mod/np.pi:+.2f} pi)")
        print(f"    n crosses 1 at ~{cross:.3f} THz" if np.isfinite(cross)
              else "    n stays >= 1 in band")
        print(f"    n  raw     @0.7={at(n,0.7):.2f} @1.5={at(n,1.5):.2f} @2.0={at(n,2.0):.2f}")
        print(f"    n  deembed @0.7={at(n_deembed,0.7):.2f} @1.5={at(n_deembed,1.5):.2f} "
              f"@2.0={at(n_deembed,2.0):.2f}  (linear phase removed)")

        axes[0].plot(f_thz, H_phase, label=filename)
        axes[1].plot(f_thz, n, label=f"{filename} raw")
        axes[1].plot(f_thz, n_deembed, "--", lw=1, label=f"{filename} de-embed")
        axes[2].plot(f_thz, k, label=f"{filename} k")

    for ax in axes:
        ax.set_xlim(0.2, 3.0)
        ax.axvspan(TRUSTED_BAND_HZ[0]*1e-12, TRUSTED_BAND_HZ[1]*1e-12, alpha=0.06, color="green")
        ax.legend(fontsize=7)
    axes[0].set_title("phi(H) unwrapped (slope = residual timing / air gap)")
    axes[0].set_ylabel("rad"); axes[0].axhline(0, color="gray", lw=0.5)
    axes[1].set_title("n"); axes[1].axhline(1.0, color="crimson", lw=0.6, linestyle="dashed")
    axes[1].set_ylabel("n")
    axes[2].set_title("k"); axes[2].set_ylabel("k"); axes[2].set_xlabel("THz")

    out = os.path.join(os.path.dirname(__file__), "explore_reflection_phase_audit.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
