"""Reflection-mode THz-TDS pipeline (window-coupled internal reflection).

Geometry: a sample pressed against the flat back face of a SiO2 window, probed
at 45 deg external incidence, s-polarised. One acquisition contains the front
(air->SiO2) first reflection and the back (SiO2->sample) second reflection. The
SiO2-only trace is the reference: its second reflection is r_{SiO2->air}, which
cancels the window path when we ratio, then restores absolute scale via the
computed Fresnel coefficient.

Two phases:
  1. segment(): load the raw traces and crop the first + second reflections into
     separate .acc files under <dir>/segmented/<component>/ (no zero-pad, no
     taper). This is the only reflection-specific step.
  2. process(): point a normal DataSet at segmented/second_reflection/ and run
     the standard chain -> group -> zero_pad -> FFT -> H = second_s/second_ref
     -> r_sample = r_{SiO2->air} * H -> invert -> derive eps/sigma.

Set INTERACTIVE = True to pick the first/second-reflection gates by dragging;
otherwise the headless GATES below are used (repeatable analysis).
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz





def segment(dataset, config):
    """Phase 1: crop the first + second reflections to separate .acc files.

    Returns the segmented root dir; the second reflections land in
    ``<root>/second_reflection/`` with the original filenames.
    """
    # dataset = DataSet(file_dir)
    # dataset.load_all_data(case_insensitive=True)
    show_graphs = config.get("show_graphs", False)
    interactive = config.get("interactive", False)
    gates = config.get("gates", None)

    segments = None if interactive else gates
    thz.segment_reflections(
        dataset, segments=segments, show_graph=show_graphs,
    )
    return os.path.join(config["file_dir"], "segmented")


def process(dataset, config, show_graphs=False):
    """Phase 2: the normal transmission-style chain on the second reflections."""

    # Pair sample <-> SiO2 reference. keywords=['type'] keeps the differing
    # descriptor tokens (sio2 vs CNT-s-1) out of the match criteria so the two
    # files pair on their complementary types.
    dataset.group_files(keywords=["type"])
    dataset.grouping.show_matches()

    # Calibrate: shift each sample to its reference T0 (removes the instrumental
    # timing offset; cross-correlation, sub-sample, handles the sign flip).
    thz.align_to_reference(dataset, ref_type="reference")
    # thz.pre_window_align_peak(dataset, show_graph=show_graphs, recalibrate=True)  #, auto_range=(40,60))

    thz.window_time(dataset, config={"window": {"type": "Hann", "alpha": 0.1}}, show_graph=show_graphs)
    thz.zero_pad(dataset, config={"pad": {"extend_factor": 2.0}}, show_graph=show_graphs)
    thz.fft_spectrum(dataset)
    if show_graphs:
        thz.plot_fft(dataset, normalise=False, scale="")

    # H = second_reflection_sample / second_reflection_reference (SiO2-only).
    # ref_type='reference' matches the bare-'reference' SiO2 file.
    thz.transfer_function(dataset, ref_type="reference")
    # thz.phase_correction(dataset, source="transfer")

    # Window geometry: r_sample = r_{SiO2->air} * H, invert inside the SiO2.
    thz.invert_nk_reflection(
        dataset,
        geometry="window",
        theta_deg=config["theta_external_deg"],
        polarization=config["polarization"],
        n_window=config["n_sio2"],
    )
    thz.derive_eps_sigma(dataset)
    return dataset


def report(dataset, band_thz=(0.5, 3.0)):
    """Print a quick numeric summary of n,k over a band for each sample."""
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        pd_ = data_obj.processing_dict
        freq = pd_.get("fft_freq")
        n = pd_.get("n")
        k = pd_.get("k")
        r = pd_.get("reflection_r")
        if freq is None or n is None:
            print(f"{filename}: no inversion result.")
            continue
        f_thz = freq * 1e-12
        band = (f_thz >= band_thz[0]) & (f_thz <= band_thz[1]) & np.isfinite(n)
        theta_int = pd_.get("theta_internal_rad")
        r_ref = pd_.get("r_reference")
        print(f"\n=== {filename} ===")
        print(f"  internal angle = {np.rad2deg(theta_int):.2f} deg, "
              f"r_(SiO2->air) = {r_ref:.4f}")
        if band.any():
            print(f"  band {band_thz[0]}-{band_thz[1]} THz ({band.sum()} bins):")
            print(f"    n  mean = {np.nanmean(n[band]):.3f}  "
                  f"(min {np.nanmin(n[band]):.3f}, max {np.nanmax(n[band]):.3f})")
            print(f"    k  mean = {np.nanmean(k[band]):.3f}  "
                  f"(min {np.nanmin(k[band]):.3f}, max {np.nanmax(k[band]):.3f})")
            print(f"    |r_sample| mean = {np.nanmean(np.abs(r[band])):.3f}")
        else:
            print("  no finite n in band.")


if __name__ == "__main__":

    # --- CONFIG ---
    config = {
        "file_dir": r"C:\Users\Samuel\Data\THz\Sam\2026-06-03_CNT-paper\testing\segmented\second_reflection",
        # "file_dir": r"C:\Users\Samuel\Data\THz\Sam\2026-06-08_CNT-paper\export",
        # "file_dir": r"C:\Users\Samuel\Data\THz\Sam\reflection_testing\segmented\second_reflection",
        "interactive": True,
        "show_graphs": True,
        "theta_external_deg": 45.0,
        "n_sio2": 1.95,
        "polarization": "s",
        "gates": {
            "first_reflection": (151.2, 158.5),
            "second_reflection": (159.5, 168.8),
        },
    }
    # import acquisition_editor
    # acquisition_editor.process_directory(config["file_dir"])

    # --- LOAD  ---
    dataset = DataSet(config['file_dir'])
    dataset.load_database()  # optional .db file with pre-parsed metadata; skip if you want to re-parse from the raw files
    dataset.load_all_data(case_insensitive=True)
    # dataset.load_state()  # optional .state file with pre-computed processing results; skip if you want to re-run the chain from scratch
    # segmented_root = segment(dataset, config)
    show_graphs = True

    # --- PROCESS + REPORT ---
    ds = process(dataset, config, show_graphs=show_graphs)
    report(ds)
    thz.result_viewer(ds)
    dataset.save_database()
