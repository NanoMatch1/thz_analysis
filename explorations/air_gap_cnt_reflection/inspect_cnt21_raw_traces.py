"""Raw-trace inspection of the CNT-21 polarization dataset (corrected channel pairing).

For every acquisition: time span, scan count, the two reflection pulse positions, and the
second-pulse POLARITY of each sample vs its own per-channel reference (ANALYSIS_NOTES par.1
sign convention: opposite polarity = seeing the high-index sample through the window; same
polarity = the parasitic SiO2->air front term dominates = air gap).

Run from repo root:
  ./.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/inspect_cnt21_raw_traces.py
"""

from __future__ import annotations

import os

import numpy as np

ROOT = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-21\polarization"

# channel -> (sample relative path, reference relative path)
CHANNELS = {
    "0-0   (S, fibers ||)": ("s-pol/sample_0-0_CNT-0-0_cam-c.acc",
                             "s-pol/reference_0-0_sio2-0-0_SS_cam-colinear.acc"),
    "0-90  (S, fibers _|_)": ("s-pol/sample_0-90_CNT-0-90_cam-c.acc",
                              "s-pol/reference_0-90_sio2-0-90_SP_cam-colinear.acc"),
    "90-0  (P, fibers _|_)": ("p-pol/sample_90-0_CNT-90-0_cam-c.acc",
                              "p-pol/reference_90-0_sio2-90-0_PS_cam-colinear.acc"),
    "90-0R (P, realign)": ("p-pol/sample_90-0_CNT-90-0_cam-c_realign.acc",
                           "p-pol/reference_90-0_sio2-90-0_PS_cam-colinear.acc"),
    "90-90 (P, fibers ||)": ("p-pol/sample_90-90_CNT-90-90_cam-c.acc",
                             "p-pol/reference_90-90_sio2-90-90_PP_cam-colinear.acc"),
    "Si-p  (P, FZ silicon)": ("silicon/p-pol/sample_90-0_silicon-FZ-90-0_PS_cam-colinear.acc",
                              "silicon/p-pol/reference_90-0_sio2-90-0_PS_cam-colinear.acc"),
    "Si-s  (S, old silicon)": ("silicon/s-pol/sample_0-0_si_old.acc",
                               "silicon/s-pol/reference_0-0_sio2_old.acc"),
}


def load_acc(path):
    """Load one .acc as (time_ps, mean_field, n_scans) using the repo loader."""
    from dataset_core.dataset import DataSet
    folder = os.path.dirname(path)
    dataset = DataSet(folder)
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)
    for filename, data_obj in dataset.data.items():
        if os.path.basename(filename).lower() == os.path.basename(path).lower():
            raw = np.asarray(data_obj.raw_data, dtype=float)
            time_ps = raw[:, 0]
            scans = raw[:, 1:]
            return time_ps, scans.mean(axis=1), scans.shape[1]
    raise FileNotFoundError(path)


def find_two_pulses(time_ps, field, min_separation_ps=8.0):
    """Return (first_peak_ps, second_peak_ps, signed peak amplitude at each)."""
    envelope = np.abs(field)
    index_a = int(np.argmax(envelope))
    masked = envelope.copy()
    masked[np.abs(time_ps - time_ps[index_a]) < min_separation_ps] = 0.0
    index_b = int(np.argmax(masked))
    (first_index, second_index) = sorted((index_a, index_b))
    return (time_ps[first_index], time_ps[second_index],
            field[first_index], field[second_index])


def signed_extremum(time_ps, field, center_ps, half_width_ps=2.0):
    """The signed value of the largest |field| sample within +/-half_width of center."""
    window = np.abs(time_ps - center_ps) <= half_width_ps
    segment = field[window]
    return float(segment[np.argmax(np.abs(segment))])


def main():
    cache = {}

    def get(path):
        if path not in cache:
            cache[path] = load_acc(path)
        return cache[path]

    print(f"{'channel':24s} {'file':12s} {'span_ps':>18s} {'scans':>5s} "
          f"{'pulse1_ps':>9s} {'pulse2_ps':>9s} {'peak2/peak1':>11s}")
    results = {}
    for channel, (sample_rel, reference_rel) in CHANNELS.items():
        for role, rel in (("sample", sample_rel), ("reference", reference_rel)):
            time_ps, field, n_scans = get(os.path.join(ROOT, rel))
            first_ps, second_ps, amp1, amp2 = find_two_pulses(time_ps, field)
            results[(channel, role)] = (time_ps, field, first_ps, second_ps)
            print(f"{channel:24s} {role:12s} "
                  f"[{time_ps[0]:8.2f},{time_ps[-1]:8.2f}] {n_scans:>5d} "
                  f"{first_ps:>9.2f} {second_ps:>9.2f} {amp2/amp1:>11.3f}")

    print("\n--- second-pulse polarity: sample vs its reference (opposite = good contact) ---")
    for channel, (sample_rel, reference_rel) in CHANNELS.items():
        t_s, y_s, _, p2_s = results[(channel, "sample")]
        t_r, y_r, _, p2_r = results[(channel, "reference")]
        val_s = signed_extremum(t_s, y_s, p2_s)
        val_r = signed_extremum(t_r, y_r, p2_r)
        relation = "OPPOSITE (contact)" if val_s * val_r < 0 else "SAME (gap-dominant?)"
        print(f"  {channel:24s} sample {val_s:+9.4f}  ref {val_r:+9.4f}  "
              f"dt2={p2_s - p2_r:+6.3f} ps  -> {relation}")


if __name__ == "__main__":
    main()
