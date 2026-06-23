"""End-to-end validation of front-pulse self-referencing on CNT-13/D.

Runs the REAL phase-2 pipeline (mirroring run_me_dataset_core.py) twice on the
segmented second-reflection folder — once conventionally, once with
``transfer_function(self_reference=True)`` — through to n, k and sigma, and
compares:

- n/k and sigma curves per sample (old solid vs self-referenced dashed);
- the pairwise consistency of sigma across the three s-orientation repeats
  (the P-orientation file is genuinely different and is excluded from the
  consistency metric).

Run:
    .venv/Scripts/python.exe explorations/validate_selfref_real_data.py
"""

import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.dataset import DataSet  # noqa: E402
from dataset_core.adapters import thz_adapter as thz  # noqa: E402

DATA_DIR = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\D\segmented\second_reflection"
OUTPUT_DIR = os.path.join(REPO_ROOT, "explorations", "output")
BAND_THZ = (0.4, 2.5)
REPEAT_SAMPLES = ["sample_a-45.acc", "sample_a-45_1.acc", "sample_a-45_2.acc"]


def run_pipeline(self_reference: bool) -> DataSet:
    dataset = DataSet(DATA_DIR)
    dataset.load_all_data(case_insensitive=True)
    thz.subtract_baseline(dataset, show_graph=False)
    dataset.group_files(keywords=["type"])
    thz.global_truncate(dataset)
    thz.window_time(dataset, config={"window": {"type": "hann"}}, show_graph=False)
    thz.zero_pad(dataset, config={"pad": {"extend_factor": 3.0}})
    thz.fft_spectrum(dataset)
    thz.transfer_function(
        dataset,
        config={
            "transfer": {"apply_snr_mask": True, "self_reference": self_reference},
            "mask": {"snr_thresh_db": 20, "tail_fraction": 0.25, "min_contiguous_bins": 3},
        },
        ref_type="reference",
    )
    thz.invert_nk_reflection(
        dataset, geometry="window", theta_deg=45, polarization="s", n_window=1.95,
    )
    thz.derive_eps_sigma(dataset)
    return dataset


def collect(dataset: DataSet) -> dict:
    results = {}
    for filename, data_obj in thz._sample_items(dataset):
        processing = data_obj.processing_dict
        freq_thz = processing["fft_freq"] * 1e-12
        band = (freq_thz >= BAND_THZ[0]) & (freq_thz <= BAND_THZ[1]) & processing["transfer_mask"]
        results[filename] = {
            "f": freq_thz, "band": band,
            "n": processing["n"], "k": processing["k"],
            "sigma": processing["sigma"],
        }
    return results


def pairwise_spread(results: dict, names: list, key) -> list:
    spreads = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = results[names[i]], results[names[j]]
            band = a["band"] & b["band"]
            va, vb = key(a)[band], key(b)[band]
            scale = np.sqrt(np.nanmean(np.abs((va + vb) / 2) ** 2))
            spreads.append(float(np.sqrt(np.nanmean(np.abs(va - vb) ** 2)) / scale))
    return spreads


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== conventional reference ===")
    old = collect(run_pipeline(self_reference=False))
    print("\n=== front-pulse self-referenced ===")
    new = collect(run_pipeline(self_reference=True))

    # --- consistency metrics over the three s-orientation repeats ----------
    print("\n--- pairwise rms spread across s-orientation repeats (0.4-2.5 THz) ---")
    for label, key in [("n", lambda r: r["n"]),
                       ("sigma_1", lambda r: r["sigma"].real),
                       ("|sigma|", lambda r: np.abs(r["sigma"]))]:
        spread_old = np.mean(pairwise_spread(old, REPEAT_SAMPLES, key))
        spread_new = np.mean(pairwise_spread(new, REPEAT_SAMPLES, key))
        print(f"  {label:8s}: conventional {spread_old:.4f}  ->  self-referenced {spread_new:.4f}")

    # --- figures ------------------------------------------------------------
    cmap = plt.get_cmap("tab10")
    fig_nk, (ax_n, ax_k) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                        layout="constrained")
    fig_s, (ax_s1, ax_s2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                         layout="constrained")
    for index, name in enumerate(old):
        colour = cmap(index)
        for results, style, tag in [(old, "-", "old"), (new, "--", "self-ref")]:
            r = results[name]
            band = r["band"]
            label = f"{name.replace('.acc', '')} ({tag})"
            ax_n.plot(r["f"][band], r["n"][band], style, color=colour, lw=1.1, label=label)
            ax_k.plot(r["f"][band], r["k"][band], style, color=colour, lw=1.1)
            ax_s1.plot(r["f"][band], r["sigma"].real[band] / 100, style, color=colour,
                       lw=1.1, label=label)
            ax_s2.plot(r["f"][band], r["sigma"].imag[band] / 100, style, color=colour, lw=1.1)

    ax_n.set_ylabel("n")
    ax_k.set_ylabel("k")
    ax_k.set_xlabel("Frequency (THz)")
    ax_n.set_title("CNT-13/D: n, k — conventional (solid) vs self-referenced (dashed)")
    ax_n.legend(fontsize=7, ncol=2)
    fig_nk.savefig(os.path.join(OUTPUT_DIR, "real_data_nk_old_vs_new.png"), dpi=150)

    ax_s1.set_ylabel("sigma_1 (S/cm)")
    ax_s2.set_ylabel("sigma_2 (S/cm)")
    ax_s2.set_xlabel("Frequency (THz)")
    ax_s1.set_title("CNT-13/D: conductivity — conventional (solid) vs self-referenced (dashed)")
    ax_s1.legend(fontsize=7, ncol=2)
    fig_s.savefig(os.path.join(OUTPUT_DIR, "real_data_sigma_old_vs_new.png"), dpi=150)

    print(f"\nFigures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
