"""Smoke test for the shared-axis reflection path (build_full_trace_reflection ->
define_reflection_regions -> isolate_and_window -> FFT -> transfer -> invert).

Runs headless on CNT-17 in both center modes and checks:
  - both reflections end on ONE shared axis (same time array, same length),
  - the full self-referenced pipeline runs through to n, k,
  - prints band-mean n, k for a sanity check vs the existing path.

Run:
    .venv/Scripts/python.exe explorations/smoke_isolate_and_window.py
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz

ROOT_DIRECTORY = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-17"
FIRST_REGION_PS = (152.0, 158.5)
SECOND_REGION_PS = (161.0, 168.0)
BAND_THZ = (0.5, 3.0)


def run_new_path(center_mode):
    dataset = DataSet(ROOT_DIRECTORY)
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)

    thz.build_full_trace_reflection(dataset)
    dataset.group_files(keywords=["type"])
    thz.align_to_reference(dataset, timing_segment="first_reflection", roi=FIRST_REGION_PS)

    thz.subtract_baseline(dataset, segment="second_reflection")
    thz.subtract_baseline(dataset, segment="first_reflection")
    thz.global_truncate(dataset, segment="second_reflection")
    thz.global_truncate(dataset, segment="first_reflection")

    thz.define_reflection_regions(
        dataset,
        config={"regions": {"first_reflection": FIRST_REGION_PS,
                            "second_reflection": SECOND_REGION_PS}},
    )
    thz.isolate_and_window(
        dataset,
        config={"window": {"type": "hann", "alpha": 1.0}},
        center_mode=center_mode,
        show_graph=False,
    )

    # shared-axis check
    any_obj = next(o for f, o in dataset.data.items() if not dataset.data.is_reference(f))
    first_axis = any_obj.first_segment.data[:, 0]
    second_axis = any_obj.data[:, 0]
    assert first_axis.shape == second_axis.shape, "first/second axes differ in length"
    assert np.allclose(first_axis, second_axis), "first/second axes are not identical"

    thz.fft_spectrum(dataset, segment="second_reflection")
    first_freq = next(iter(dataset.data.values())).processing_dict.get("fft_freq")
    shared_n_fft = 2 * (len(first_freq) - 1) if first_freq is not None else None
    thz.fft_spectrum(dataset, segment="first_reflection", n_fft=shared_n_fft)

    thz.transfer_function(
        dataset,
        config={"transfer": {"self_reference": True}},
        ref_type="reference",
    )
    thz.invert_nk_reflection(
        dataset, geometry="window", theta_deg=45, polarization="s", n_window=1.95,
    )
    thz.derive_eps_sigma(dataset)
    return dataset


def report(dataset, center_mode):
    print(f"\n--- new path, center_mode={center_mode} ---")
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        processing = data_obj.processing_dict
        frequency_hz = processing.get("fft_freq")
        refractive_index = processing.get("n")
        extinction = processing.get("k")
        if frequency_hz is None or refractive_index is None:
            print(f"  {filename}: no inversion result")
            continue
        frequency_thz = frequency_hz * 1e-12
        band = ((frequency_thz >= BAND_THZ[0]) & (frequency_thz <= BAND_THZ[1])
                & np.isfinite(refractive_index))
        if band.any():
            print(f"  {filename}: n={np.nanmean(refractive_index[band]):.3f} "
                  f"k={np.nanmean(extinction[band]):.3f} "
                  f"({band.sum()} bins in {BAND_THZ} THz)")
        else:
            print(f"  {filename}: no finite n in band")


def main():
    for center_mode in ("crop", "pad"):
        dataset = run_new_path(center_mode)
        report(dataset, center_mode)
    print("\nshared-axis assertions passed for both modes.")


if __name__ == "__main__":
    main()
