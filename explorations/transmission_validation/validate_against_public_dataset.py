"""Validate the transmission pipeline against a PUBLIC known-answer dataset.

Uses the phoeniks (puls-lab) example 01_Basic_Extraction: a SYNTHETIC THz-TDS pair
(reference + 1 mm sample) generated from a KNOWN n(f), k(f) (provided in
Artifical_n_k_alpha.txt). Running our real pipeline on it and recovering that planted
n, k is a clean, material-independent test of the pipeline MECHANICS — separate from any
question about Samuel's silicon measurements or thickness.

It downloads the three files once (cached next to this script), converts the time traces
to our .acc format (time in ps), runs the exact run_me_transmission flow
(subtract_baseline -> window_time_fixed_width -> zero_pad -> fft_spectrum ->
transfer_function -> invert_nk(thickness=1 mm)), and overlays recovered vs true n, k.

Run:  PYTHONPATH=. .venv/Scripts/python.exe explorations/transmission_validation/validate_against_public_dataset.py
"""
from __future__ import annotations

import os
import sys
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, REPO)

from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz

BASE_URL = ("https://raw.githubusercontent.com/puls-lab/phoeniks/main/"
            "examples/01_Basic_Extraction")
FILES = ("Artifical_Reference.txt", "Artifical_Sample_1mm.txt", "Artifical_n_k_alpha.txt")
THICKNESS_M = 1.0e-3
HALF_WIDTH_PS = 8.0

DATA_DIR = os.path.join(HERE, "phoeniks_data")
ACC_DIR = os.path.join(HERE, "phoeniks_acc")


def download_if_missing():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in FILES:
        dest = os.path.join(DATA_DIR, name)
        if not os.path.exists(dest):
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", dest)


def write_acc(name, time_seconds, amplitude):
    """Write a 2-column .acc (time in PICOSECONDS, the format DataSet expects)."""
    os.makedirs(ACC_DIR, exist_ok=True)
    lines = [f"%title {name} acc 1", "%type 0"]
    lines += [f"{t * 1e12:.8f} {a:.10e}" for t, a in zip(time_seconds, amplitude)]
    with open(os.path.join(ACC_DIR, name + ".acc"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    download_if_missing()
    reference = np.loadtxt(os.path.join(DATA_DIR, "Artifical_Reference.txt"))
    sample = np.loadtxt(os.path.join(DATA_DIR, "Artifical_Sample_1mm.txt"))
    ground_truth = np.loadtxt(os.path.join(DATA_DIR, "Artifical_n_k_alpha.txt"))
    gt_freq_hz, gt_n, gt_k = ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2]

    write_acc("reference_phoeniks", reference[:, 0], reference[:, 1])
    write_acc("sample_phoeniks", sample[:, 0], sample[:, 1])
    dt_ps = np.median(np.diff(reference[:, 0])) * 1e12
    print(f"dt={dt_ps:.4f} ps, span={reference[-1,0]*1e12:.1f} ps, N={reference.shape[0]}; "
          f"ground-truth n {gt_n.min():.3f}-{gt_n.max():.3f}, k {gt_k.min():.4f}-{gt_k.max():.4f}")

    config = {
        "general": {"show_graph": False},
        "window": {"type": "hann", "alpha": 1.0},
        "pad": {"extend_factor": 1.0},
        "transfer": {"self_reference": False, "apply_snr_mask": True, "min_ref_amp_rel": 1e-3,
                     "regularization_eps": 1e-30, "unwrap_phase": True},
        "mask": {"snr_thresh_db": 20, "tail_fraction": 0.25, "min_contiguous_bins": 3},
        "derive": {"eps_background": 1.0},
    }
    dataset = DataSet(ACC_DIR, config=config)
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)
    dataset.group_files(keywords=["type"])
    thz.subtract_baseline(dataset)
    thz.window_time_fixed_width(dataset, half_width_ps=HALF_WIDTH_PS, center_in_trace=True,
                                show_graph=False)
    thz.zero_pad(dataset, show_graph=False)
    thz.fft_spectrum(dataset)
    thz.transfer_function(dataset, ref_type="reference")
    thz.invert_nk(dataset, thickness_m=THICKNESS_M)

    fig, (ax_n, ax_k) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    fig.suptitle("Pipeline vs ground truth — phoeniks synthetic 1 mm sample")
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        p = data_obj.processing_dict
        freq = p["fft_freq"]; n = p["n"]; k = p["k"]; mask = p["transfer_mask"]
        f_thz = freq * 1e-12
        band = mask & (f_thz >= 0.1) & (f_thz <= 3.0) & np.isfinite(n)
        gt_n_on_grid = np.interp(freq, gt_freq_hz, gt_n)
        gt_k_on_grid = np.interp(freq, gt_freq_hz, gt_k)

        print(f"\n--- {filename}: recovered vs true (n, k) ---")
        print(f"{'freq THz':>9} {'n_pipe':>8} {'n_true':>8} {'dn':>8} {'k_pipe':>8} {'k_true':>8}")
        for f_test in (0.2, 0.5, 1.0, 1.5, 2.0, 2.5):
            i = int(np.argmin(np.abs(f_thz - f_test)))
            print(f"{f_test:9.1f} {n[i]:8.3f} {gt_n_on_grid[i]:8.3f} {n[i]-gt_n_on_grid[i]:8.3f} "
                  f"{k[i]:8.4f} {gt_k_on_grid[i]:8.4f}")
        dn = np.abs(n[band] - gt_n_on_grid[band])
        dk = np.abs(k[band] - gt_k_on_grid[band])
        print(f"  band 0.1-3 THz: mean |dn| = {np.nanmean(dn):.4f}, mean |dk| = {np.nanmean(dk):.4f}")

        ax_n.plot(f_thz[band], n[band], lw=1.6, label="pipeline")
        ax_n.plot(f_thz[band], gt_n_on_grid[band], "k--", lw=1.2, label="ground truth")
        ax_k.plot(f_thz[band], k[band], lw=1.6, label="pipeline")
        ax_k.plot(f_thz[band], gt_k_on_grid[band], "k--", lw=1.2, label="ground truth")
    ax_n.set_title("n"); ax_n.set_xlabel("THz"); ax_n.set_ylabel("n"); ax_n.legend(); ax_n.grid(alpha=0.3)
    ax_k.set_title("k"); ax_k.set_xlabel("THz"); ax_k.set_ylabel("k"); ax_k.legend(); ax_k.grid(alpha=0.3)
    out = os.path.join(HERE, "validate_against_public_dataset.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
