"""Compare the three silicon calibration datasets: today / chris / denis.

Question (Samuel): chris & denis look conductive (doped) Si; today's could be HR float-zone
or low-doping. Classify each from its THz response.

  today  : TRANSMISSION (air reference, Si sample). amp ratio ~0.70 (clean slab).
  chris  : TRANSMISSION (reference + sample, .txt). amp ratio ~0.064 (very lossy).
  denis  : REFLECTION (GOLD reference!) — gold does not transmit THz, so this is a
           single-bounce reflection, processed with geometry='gold'.

Signatures: HR/intrinsic Si -> flat n~3.4, k~0, sigma~0, transmits well. Doped/conductive
Si -> free-carrier (Drude) absorption strongest at LOW frequency -> |T| rolls off to DC,
k and Re(sigma) large and rising toward DC.

Run:  PYTHONPATH=. .venv/Scripts/python.exe explorations/transmission_validation/compare_silicon_calibration_sets.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, REPO)

from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz

ROOT = r"C:\Users\Samuel\Data\THz\calibration\silicon"
THICKNESS_M = 290e-6   # assumed for the transmission sets (Samuel's wafer); chris unknown -> same guess
HZ = 1e-12


def load_xy(path):
    t, a = [], []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                t.append(float(parts[0])); a.append(float(parts[1]))
            except ValueError:
                pass
    return np.array(t), np.array(a)


def write_acc(folder, name, time_ps, amplitude):
    os.makedirs(folder, exist_ok=True)
    lines = [f"%title {name} acc 1", "%type 0"]
    lines += [f"{t:.6f} {a:.10e}" for t, a in zip(time_ps, amplitude)]
    with open(os.path.join(folder, name + ".acc"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _config():
    return {
        "general": {"show_graph": False},
        "geometry": {"theta_external_deg": 45.0, "polarization": "s", "r_reference": -1.0},
        "window": {"type": "hann", "alpha": 1.0},
        "pad": {"extend_factor": 1.0},
        "transfer": {"self_reference": False, "apply_snr_mask": True, "min_ref_amp_rel": 1e-3,
                     "regularization_eps": 1e-30, "unwrap_phase": True},
        "mask": {"snr_thresh_db": 15, "tail_fraction": 0.25, "min_contiguous_bins": 3},
        "derive": {"eps_background": 1.0},
    }


def dataset_from_dir(folder):
    """Load multi-scan .acc directly (DataSet averages the scans)."""
    dataset = DataSet(folder, config=_config())
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)
    dataset.group_files(keywords=["type"])
    return dataset


def dataset_from_txt_pair(ref_txt, sample_txt):
    """Stage a single-scan .txt (time, amp) pair as .acc and load."""
    tmp = tempfile.mkdtemp(prefix="si_cal_")
    write_acc(tmp, "reference_x", *load_xy(ref_txt))
    write_acc(tmp, "sample_x", *load_xy(sample_txt))
    dataset = DataSet(tmp, config=_config())
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)
    dataset.group_files(keywords=["type"])
    return dataset


def process_transmission(dataset, thickness_m, half_width_ps=3.0):
    thz.subtract_baseline(dataset)
    thz.window_time_fixed_width(dataset, half_width_ps=half_width_ps, center_in_trace=True,
                                show_graph=False)
    thz.zero_pad(dataset, show_graph=False)
    thz.fft_spectrum(dataset)
    thz.transfer_function(dataset, ref_type="reference")
    thz.invert_nk(dataset, thickness_m=thickness_m)
    thz.derive_eps_sigma(dataset)
    return _collect(dataset)


def process_reflection(dataset, half_width_ps=3.0):
    thz.subtract_baseline(dataset)
    # remove the gold<->Si positioning offset (a height misregistration, ~2.8 ps) so the
    # reflection phase is the material's; for a decent reflector the material group delay
    # is small, so aligning the surface peaks is the right zero. THEN window in place.
    thz.recenter_peaks_to_common_t0(dataset, target_t0_ps=None, subsample=True, show_graph=False)
    thz.window_time_fixed_width(dataset, half_width_ps=half_width_ps, center_in_trace=True,
                                show_graph=False)
    thz.zero_pad(dataset, show_graph=False)
    thz.fft_spectrum(dataset)
    thz.transfer_function(dataset, ref_type="reference")
    thz.invert_nk_reflection(dataset, geometry="gold", theta_deg=45.0, polarization="s",
                             r_reference=-1.0)
    thz.derive_eps_sigma(dataset)
    return _collect(dataset)


def _collect(dataset):
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        p = data_obj.processing_dict
        return dict(freq=p["fft_freq"], transfer=p["transfer_H"], n=p.get("n"), k=p.get("k"),
                    sigma=p.get("sigma"), mask=p.get("transfer_mask"))
    return None


def band_stats(result, lo=0.3, hi=1.5):
    f = result["freq"] * HZ
    b = result["mask"] & (f >= lo) & (f <= hi) & np.isfinite(result["n"])
    return (float(np.nanmedian(result["n"][b])), float(np.nanmedian(result["k"][b])),
            float(np.nanmedian(np.real(result["sigma"])[b])))


def main():
    today = process_transmission(
        dataset_from_dir(os.path.join(ROOT, "2025-06-25_silicon_transmission")), THICKNESS_M)
    chris = process_transmission(
        dataset_from_txt_pair(os.path.join(ROOT, "chris", "reference_RT.txt"),
                              os.path.join(ROOT, "chris", "sample_RT.txt")), THICKNESS_M)
    denis = process_reflection(
        dataset_from_dir(os.path.join(ROOT, "denis")))

    print(f"\n{'dataset':22s} {'geometry':12s} {'n':>7} {'k':>8} {'Re sigma S/m':>13}")
    for name, geom, res in (("today", "transmission", today), ("chris", "transmission", chris),
                            ("denis", "reflection", denis)):
        n, k, s = band_stats(res)
        print(f"{name:22s} {geom:12s} {n:7.2f} {k:8.3f} {s:13.1f}")
    print("\n(transmission n assumes d=290 um; chris thickness unknown so its n is indicative. "
          "denis is reflection -> n,k from |r|, thickness-free.)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle("Silicon calibration sets: today (HR?) vs chris & denis (conductive?)")
    colors = {"today": "C2", "chris": "C3", "denis": "C0"}
    for name, res in (("today", today), ("chris", chris), ("denis", denis)):
        f = res["freq"] * HZ
        b = res["mask"] & (f >= 0.2) & (f <= 2.5) & np.isfinite(res["n"])
        axes[0][0].plot(f[b], np.abs(res["transfer"])[b], color=colors[name], label=name)
        axes[0][1].plot(f[b], res["n"][b], color=colors[name], label=name)
        axes[1][0].plot(f[b], res["k"][b], color=colors[name], label=name)
        axes[1][1].plot(f[b], np.real(res["sigma"])[b], color=colors[name], label=name)
    axes[0][0].set_title("|transfer|  (|T| trans / |r_Si/r_gold| refl)"); axes[0][0].set_yscale("log")
    axes[0][1].set_title("n"); axes[0][1].axhline(3.4175, color="0.6", ls=":", label="HR-Si 3.4175")
    axes[1][0].set_title("k (extinction)")
    axes[1][1].set_title("Re sigma (S/m)")
    for ax in axes.flat:
        ax.set_xlabel("Frequency (THz)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    out = os.path.join(HERE, "compare_silicon_calibration_sets.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
