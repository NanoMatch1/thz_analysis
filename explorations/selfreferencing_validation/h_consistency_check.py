"""Quantify whether front-pulse self-referencing improves sample-to-sample
consistency of the transfer function H on CNT-13/D.

H_old = Y2_sample / Y2_reference (current approach)
H_new = (Y2_sample / Y1_sample) / W,  W = Y2_ref / Y1_ref  (self-referenced)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import window_selfref_evaluation as ev


def relative_spread(h_matrix: np.ndarray) -> float:
    mean_h = h_matrix.mean(axis=0)
    return float(
        np.sqrt(np.mean(np.abs(h_matrix - mean_h) ** 2))
        / np.sqrt(np.mean(np.abs(mean_h) ** 2))
    )


def main() -> None:
    names = ev.list_acc_filenames()
    reference_name = [f for f in names if f.startswith("reference")][0]
    sample_names = [f for f in names if f.startswith("sample")]

    spectra = {}
    freq = None
    for name in names:
        t1, scans1 = ev.load_segment_scans("first_reflection", name)
        t2, scans2 = ev.load_segment_scans("second_reflection", name)
        freq, y1 = ev.segment_spectrum(t1, scans1.mean(axis=1))
        _, y2 = ev.segment_spectrum(t2, scans2.mean(axis=1))
        spectra[name] = (y1, y2)

    mask = ev.band_mask(freq)
    y1_ref, y2_ref = spectra[reference_name]
    w_window = y2_ref / y1_ref

    h_old = np.array([spectra[s][1][mask] / y2_ref[mask] for s in sample_names])
    h_new = np.array(
        [(spectra[s][1][mask] / spectra[s][0][mask]) / w_window[mask]
         for s in sample_names]
    )

    print(f"relative rms spread across {len(sample_names)} samples "
          f"({ev.TRUSTED_BAND_THZ[0]}-{ev.TRUSTED_BAND_THZ[1]} THz):")
    print(f"  H_old: {relative_spread(h_old):.4f}")
    print(f"  H_new: {relative_spread(h_new):.4f}")

    print("\npairwise rms |dH| / rms|H|  (old / new):")
    for i in range(len(sample_names)):
        for j in range(i + 1, len(sample_names)):
            pair_mean_old = h_old[[i, j]].mean(axis=0)
            pair_mean_new = h_new[[i, j]].mean(axis=0)
            diff_old = (np.sqrt(np.mean(np.abs(h_old[i] - h_old[j]) ** 2))
                        / np.sqrt(np.mean(np.abs(pair_mean_old) ** 2)))
            diff_new = (np.sqrt(np.mean(np.abs(h_new[i] - h_new[j]) ** 2))
                        / np.sqrt(np.mean(np.abs(pair_mean_new) ** 2)))
            print(f"  {sample_names[i][:24]:26s} vs {sample_names[j][:24]:26s}: "
                  f"{diff_old:.4f} / {diff_new:.4f}")


if __name__ == "__main__":
    main()
