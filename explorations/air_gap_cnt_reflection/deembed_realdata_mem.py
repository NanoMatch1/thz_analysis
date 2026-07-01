"""Apply the NEW de-embed methods (MEM / Hilbert phase-excess + amplitude-KK) to REAL data.

Runs the window self-referenced reflection pipeline (run_shared_axis) on a dataset, then for
each sample de-embeds the contact gap three ways and overlays n, k:
  * naive            -- invert r_meas directly (current pipeline, no de-embed);
  * phase-excess Hilbert / MEM -- estimate the gap d from arg(x)-phi_minphase(|x|), strip it,
                        re-invert with AIR incidence (the production small-gap estimator, F18);
  * amplitude-KK     -- magnitude-match cross-check (expected to warn SUB-FRINGE for small gaps).

This is a LOOK, not a verdict: per F19 the inversion is hypersensitive, so read the SHAPE and
the gap numbers, not absolute n. Run on CNT (0/90 + ref) and Si (the well-conditioned control).

Run:
  PYTHONPATH=. ../.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/deembed_realdata_mem.py cnt
  PYTHONPATH=. ../.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/deembed_realdata_mem.py si
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thz_core.thz_core as core
import compare_reflection_pathways as pathways
from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz
from explore_air_gap_deembedding import (
    deembed_gap_layer, remove_round_trip_phase, gap_round_trip_phase,
)
from deembed_air_gap_iterative import estimate_gap_minimum_phase, estimate_gap_amplitude_kk

FIT_BAND_HZ = (0.4e12, 2.5e12)
PROBE_BAND_HZ = (1.0e12, 2.0e12)
CROP_BAND_HZ = (0.15e12, 3.5e12)   # restrict before MEM so out-of-band noise can't poison the AR fit

DATASETS = {
    "cnt": dict(
        root_dir=r"C:\Users\Samuel\Data\THz\calibration\reflection\2026_06_19_CNT\test1\export",
        regions={"first_reflection": (153.5, 159.0), "second_reflection": (178.5, 183.5)},
        title="CNT 0/90-deg (2026_06_19)",
    ),
    "si": dict(
        root_dir=r"C:\Users\Samuel\Data\THz\Sam\reflection_testing\2026-06-23_refl_CNT\export\silicon",
        regions=None,   # auto-detected from the trace below
        title="Silicon (2026-06-23, firmly pressed)",
    ),
}


def internal_and_gap_angles(n_sio2=1.95, external_deg=45.0):
    external = np.deg2rad(external_deg)
    theta_sio2 = np.arcsin(np.sin(external) / n_sio2)
    theta_gap = np.arcsin(n_sio2 * np.sin(theta_sio2) / 1.0)
    return float(theta_sio2), float(theta_gap)


def detect_reflection_regions(root_dir):
    """Find the two reflection pulse windows from the reference trace (for a new dataset)."""
    from dataset_core.dataset import DataSet
    from dataset_core.adapters import thz_adapter as thz
    dataset = DataSet(root_dir)
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)
    thz.build_full_trace_reflection(dataset)
    # Use the reference trace; find the two largest peaks separated in time.
    for filename, data_obj in dataset.data.items():
        time_ps = np.asarray(data_obj.data[:, 0], dtype=float) * 1e12
        field = np.asarray(data_obj.data[:, 1], dtype=float)
        envelope = np.abs(field)
        first_index = int(np.argmax(envelope))
        first_ps = time_ps[first_index]
        # Mask out +/-8 ps around the first peak, find the next.
        masked = envelope.copy()
        masked[np.abs(time_ps - first_ps) < 8.0] = 0.0
        second_ps = time_ps[int(np.argmax(masked))]
        low, high = sorted((first_ps, second_ps))
        print(f"  detected pulses at {low:.1f} and {high:.1f} ps (from {os.path.basename(filename)})")
        return {"first_reflection": (low - 2.5, low + 3.0),
                "second_reflection": (high - 2.5, high + 3.0)}
    raise RuntimeError("no traces to detect regions from")


def run_shared_axis_current(config, eps_infinity=1.0):
    """Window self-referenced reflection pipeline, current adapter API (mirrors the shared
    harness but with global_truncate_segments, which replaced global_truncate(segment=...))."""
    dataset = DataSet(config["root_dir"])
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)
    thz.build_full_trace_reflection(dataset)
    dataset.group_files(keywords=["type"])
    thz.align_to_reference(
        dataset, timing_segment="first_reflection",
        roi=config["regions"]["first_reflection"],
        subsample_correction=True, show_graph=False)
    thz.subtract_baseline(dataset)
    thz.global_truncate_segments(dataset, segment="second_reflection")
    thz.global_truncate_segments(dataset, segment="first_reflection")
    thz.define_reflection_regions(dataset, config)
    thz.isolate_and_window(
        dataset, config={"window": config["window"]},
        center_mode=config["center_mode"], show_graph=False)
    thz.fft_spectrum(dataset, segment="second_reflection", n_fft=config["n_fft"])
    thz.fft_spectrum(dataset, segment="first_reflection", n_fft=config["n_fft"])
    thz.transfer_function(
        dataset, config={"transfer": {"self_reference": True}}, ref_type="reference")
    thz.invert_nk_reflection(
        dataset, geometry="window", theta_deg=config["theta_external_deg"],
        polarization=config["polarization"], n_window=config["n_sio2"])
    thz.derive_eps_sigma(dataset, config={"derive": {"eps_background": eps_infinity}})
    return dataset


def build_config(root_dir, regions):
    config = dict(pathways.PIPELINE_CONFIG)
    config["root_dir"] = root_dir
    config["regions"] = regions
    config["gates"] = regions
    return config


def run(dataset_key):
    spec = DATASETS[dataset_key]
    root_dir = spec["root_dir"]
    regions = spec["regions"] or detect_reflection_regions(root_dir)

    # Point the shared harness at this dataset (it reads module globals for the dir + regions).
    pathways.ROOT_DIR = root_dir
    pathways.FIRST_REGION_PS = regions["first_reflection"]
    pathways.SECOND_REGION_PS = regions["second_reflection"]
    config = build_config(root_dir, regions)
    pathways.PIPELINE_CONFIG = config

    dataset = run_shared_axis_current(config, eps_infinity=1.0)
    theta_sio2, theta_gap = internal_and_gap_angles(n_sio2=config["n_sio2"],
                                                    external_deg=config["theta_external_deg"])

    samples = []
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        processing = data_obj.processing_dict
        if processing.get("reflection_r") is None:
            continue
        samples.append((filename, processing))

    figure, axes = plt.subplots(len(samples), 2, figsize=(13, 4.6 * len(samples)), squeeze=False)
    figure.suptitle(f"De-embed methods on real data — {spec['title']}", fontsize=12)

    for row, (filename, processing) in enumerate(samples):
        name = pathways.short_name(filename)
        frequency = np.asarray(processing["fft_freq"], dtype=float)
        mask = np.asarray(processing["transfer_mask"], dtype=bool)
        r_meas = np.asarray(processing["reflection_r"], dtype=complex)
        r_front = complex(processing["r_reference"])
        n_naive = np.asarray(processing["n"], dtype=float)
        k_naive = np.asarray(processing["k"], dtype=float)

        # Crop to a trustworthy band so the MEM autoregressive fit isn't poisoned by noise.
        crop = (frequency >= CROP_BAND_HZ[0]) & (frequency <= CROP_BAND_HZ[1])
        freq_c = frequency[crop]
        mask_c = mask[crop]
        r_meas_c = r_meas[crop]
        x_c = deembed_gap_layer(r_meas_c, r_front)

        results = {}
        for engine in ("hilbert", "mem"):
            d_est, _ = estimate_gap_minimum_phase(
                freq_c, x_c, mask_c, theta_gap, FIT_BAND_HZ, phase_engine=engine)
            r_back = remove_round_trip_phase(x_c, freq_c, d_est, theta_gap)
            n_de, k_de, _ = core.invert_nk_reflection(
                freq_c, r_back, mask_c, theta_rad=theta_gap, n_incident=1.0)
            results[engine] = dict(d=d_est, n=n_de, k=k_de)

        d_amp, amp_diag = estimate_gap_amplitude_kk(
            freq_c, r_meas_c, r_front, mask_c, theta_gap, FIT_BAND_HZ, phase_engine="mem")

        def band_min(freq, values, m):
            inb = (freq >= PROBE_BAND_HZ[0]) & (freq <= PROBE_BAND_HZ[1]) & m & np.isfinite(values)
            return float(np.nanmin(values[inb])) if inb.any() else np.nan

        print(f"\n{name}:")
        print(f"  r_front (SiO2->air) = {r_front:.3f}")
        print(f"  gap d: Hilbert {results['hilbert']['d']*1e6:+.1f} um, "
              f"MEM {results['mem']['d']*1e6:+.1f} um, "
              f"amplitude-KK {d_amp*1e6:.1f} um (sub_fringe={amp_diag['sub_fringe']})")
        print(f"  min n in {PROBE_BAND_HZ[0]/1e12:.0f}-{PROBE_BAND_HZ[1]/1e12:.0f} THz: "
              f"naive {band_min(frequency, n_naive, mask):+.2f} -> "
              f"Hilbert {band_min(freq_c, results['hilbert']['n'], mask_c):+.2f}, "
              f"MEM {band_min(freq_c, results['mem']['n'], mask_c):+.2f}")

        f_thz = frequency * 1e-12
        fc_thz = freq_c * 1e-12
        plot_naive = (f_thz >= 0.3) & (f_thz <= 2.5) & mask
        plot_c = (fc_thz >= 0.3) & (fc_thz <= 2.5) & mask_c

        axn = axes[row][0]
        axn.plot(f_thz[plot_naive], n_naive[plot_naive], "0.5", ls="--", lw=1.4, label="naive")
        axn.plot(fc_thz[plot_c], results["hilbert"]["n"][plot_c], "C3", lw=1.4,
                 label=f"Hilbert (d={results['hilbert']['d']*1e6:.1f}um)")
        axn.plot(fc_thz[plot_c], results["mem"]["n"][plot_c], "C0", lw=1.6,
                 label=f"MEM (d={results['mem']['d']*1e6:.1f}um)")
        axn.axhline(1.0, color="0.7", ls=":", lw=1.0)
        axn.set_title(f"{name}: n", fontsize=10)
        axn.set_xlabel("THz"); axn.set_ylabel("n"); axn.legend(fontsize=8); axn.grid(alpha=0.3)

        axk = axes[row][1]
        axk.plot(f_thz[plot_naive], k_naive[plot_naive], "0.5", ls="--", lw=1.4, label="naive")
        axk.plot(fc_thz[plot_c], results["hilbert"]["k"][plot_c], "C3", lw=1.4, label="Hilbert")
        axk.plot(fc_thz[plot_c], results["mem"]["k"][plot_c], "C0", lw=1.6, label="MEM")
        axk.set_title(f"{name}: k", fontsize=10)
        axk.set_xlabel("THz"); axk.set_ylabel("k"); axk.legend(fontsize=8); axk.grid(alpha=0.3)

    output_path = os.path.join(os.path.dirname(__file__), f"deembed_realdata_mem_{dataset_key}.png")
    figure.savefig(output_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure: {output_path}")
    return output_path


if __name__ == "__main__":
    keys = sys.argv[1:] or ["cnt", "si"]
    for key in keys:
        print(f"\n===== {key.upper()} =====")
        run(key)
