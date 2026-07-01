"""Process the CNT-21 polarization dataset through the new p-pol + dual-pol code.

Data: C:\\Users\\Samuel\\Data\\THz\\Sam\\Analysis\\CNT-21\\polarization  (s-pol/ and p-pol/ folders).

Filename convention  ..._<polAngle>-<fiberOrientation>_...  with polAngle 0=S, 90=P and fiber
orientation 0=vertical(||S), 90=horizontal(||P). References carry an explicit source-detection tag
(SS/SP/PS/PP). So the four samples probe an ANISOTROPY matrix:

    0-0   S-pol, E || fibers  -> n_parallel
    0-90  S-pol, E _|_ fibers -> n_perp
    90-0  P-pol, E _|_ fibers -> n_perp
    90-90 P-pol, E || fibers  -> n_parallel

The dual-pol joint fit assumes ONE material index explains both polarisations, so the valid pairs
(same axis seen by s AND p) are:
    n_parallel : 0-0 (S) + 90-90 (P)
    n_perp     : 0-90 (S) + 90-0 (P)

Each folder has two references (co-pol + cross-pol); we self-reference each sample against the
CO-POLARISED window reference (SS for S samples, PP for P samples) and drop the cross-pol one so the
auto-grouper is unambiguous.

Run:  PYTHONPATH=. ./.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/process_cnt21_polarization.py
"""

from __future__ import annotations

import os
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thz_core.thz_core as core
from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz

warnings.filterwarnings("ignore")
plt.show = lambda *a, **k: None  # headless

ROOT = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-21\polarization"

CONFIG = {
    "general": {"show_graph": False, "save_database": False},
    "geometry": {"theta_external_deg": 45.0, "n_sio2": 1.96},
    "regions": {"first_reflection": (152.5, 158.5), "second_reflection": (177.0, 183.5)},
    # First pulse sits ~2.2 ps from the trace start (155.2 ps, trace [153, 184.5]); a 4 ps
    # half-width clips it and corrupts the self-reference Y1. 2 ps keeps both pulses clean.
    "window": {"type": "hann", "alpha": 1.0, "half_width_ps": 2},
    "fft": {"n_fft": 2000},
    "transfer": {"self_reference": True, "apply_snr_mask": True, "min_ref_amp_rel": 1e-3,
                 "regularization_eps": 1e-30, "unwrap_phase": True},
    "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
    "derive": {"eps_background": 1},
    "dual_pol": {"fixed_gap_um": None, "initial_gap_um": 1.0, "gap_bounds_um": (0.0, 60.0)},
}

# which reference token identifies the co-pol reference to KEEP per folder
CO_POL_REFERENCE_TAG = {"s-pol": "_ss_", "p-pol": "_pp_"}
PROBE_BAND_THZ = (0.5, 2.0)


def process_folder(folder_name: str, polarization: str) -> DataSet:
    """Run the windowed self-referenced pipeline for one folder, co-pol reference only."""
    run_config = {**CONFIG, "geometry": {**CONFIG["geometry"], "polarization": polarization}}
    dataset = DataSet(os.path.join(ROOT, folder_name), config=run_config)
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)

    # Drop the cross-pol reference so grouping is unambiguous (keep only the co-pol one).
    keep_tag = CO_POL_REFERENCE_TAG[folder_name]
    all_files = list(dataset.data.keys())
    survivors = []
    for filename in all_files:
        low = filename.lower()
        if low.startswith("reference") and keep_tag not in low:
            dataset.remove_item(filename)
            print(f"  [{folder_name}] dropped cross-pol reference {filename}")
        else:
            survivors.append(filename)
    # remove_item only touches the data dict; the grouping service keeps its own filelist +
    # file_items. Reset them to the survivors so the (co-pol-only) reference pairing is
    # unambiguous when group_files rebuilds below.
    dataset.grouping.filelist = list(survivors)
    dataset.grouping.file_items = {}
    dataset.grouping.set_current_data_list(list(survivors))

    thz.build_full_trace_reflection(dataset)
    thz.taper_and_pad_traces(dataset)
    dataset.group_files(keywords=["type"])
    thz.define_reflection_regions(dataset, run_config)
    thz.subtract_baseline(dataset)
    thz.window_pulses_fixed_width(dataset, half_width_ps=run_config["window"]["half_width_ps"],
                                  show_graph=False)
    shared_n_fft = run_config["fft"]["n_fft"]
    thz.fft_spectrum(dataset, segment="second_reflection", n_fft=shared_n_fft)
    thz.fft_spectrum(dataset, segment="first_reflection", n_fft=shared_n_fft)
    thz.transfer_function(dataset, ref_type="reference")
    thz.selfref_quality(dataset)
    thz.invert_nk_reflection(dataset, geometry="window",
                             theta_deg=run_config["geometry"]["theta_external_deg"],
                             polarization=polarization,
                             n_window=run_config["geometry"]["n_sio2"])
    return dataset


def sample_diagnostics(dataset: DataSet):
    """Per-sample: |H|>1 fraction, |C| median, median n over the probe band."""
    rows = []
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        p = data_obj.processing_dict
        freq = np.asarray(p["fft_freq"]); mask = np.asarray(p["transfer_mask"], bool)
        H = np.asarray(p["transfer_H"], complex)
        r = np.asarray(p["reflection_r"], complex)
        C = p.get("selfref_correction")
        finite = np.isfinite(H)
        frac_gt1 = float(np.mean(np.abs(H[finite]) > 1.0)) if finite.any() else np.nan
        c_med = float(np.nanmedian(np.abs(C[np.isfinite(C)]))) if C is not None else np.nan
        band = mask & (freq*1e-12 >= PROBE_BAND_THZ[0]) & (freq*1e-12 <= PROBE_BAND_THZ[1]) & np.isfinite(r)
        # |1+r| is the distance to the r=-1 mirror pole = the true inversion conditioning
        # (larger is better). |H|>1 is a common-mode alignment artifact that p-pol does NOT fix,
        # so it is NOT the right p-pol metric — the pole distance is.
        rows.append(dict(name=filename, frac_H_gt1=frac_gt1, c_med=c_med,
                         pole_dist=float(np.nanmedian(np.abs(1 + r[band]))) if band.any() else np.nan,
                         r_mag=float(np.nanmedian(np.abs(r[band]))) if band.any() else np.nan))
    return rows


def find_sample(dataset: DataSet, pol_fiber_token: str) -> str:
    """Return the sample filename whose name contains the '<pol>-<fiber>' token (prefer _realign)."""
    matches = [fn for fn in dataset.data.keys()
               if not dataset.data.is_reference(fn) and pol_fiber_token in fn.lower()]
    if not matches:
        raise KeyError(f"no sample matching {pol_fiber_token}")
    realign = [fn for fn in matches if "realign" in fn.lower()]
    return realign[0] if realign else matches[0]


def joint_fit_axis(axis_label, dataset_s, name_s, dataset_p, name_p):
    """Run the core dual-pol Drude+gap fit for one material axis (a matched s/p pair)."""
    proc_s = dataset_s.data[name_s].processing_dict
    proc_p = dataset_p.data[name_p].processing_dict
    freq = np.asarray(proc_s["fft_freq"], float)
    mask = np.asarray(proc_s["transfer_mask"], bool) & np.asarray(proc_p["transfer_mask"], bool)
    theta_gap = np.deg2rad(CONFIG["geometry"]["theta_external_deg"])
    measured = {"s": np.asarray(proc_s["reflection_r"], complex),
                "p": np.asarray(proc_p["reflection_r"], complex)}
    front = {"s": np.asarray(proc_s["r_reference"], complex) * np.ones_like(freq),
             "p": np.asarray(proc_p["r_reference"], complex) * np.ones_like(freq)}
    fit = core.fit_reflection_gap_dual_pol(freq, mask, measured, front, theta_gap,
                                           initial_gap_um=1.0, gap_bounds_um=(0.0, 60.0))
    print(f"\n[{axis_label}]  s={os.path.basename(name_s)}  p={os.path.basename(name_p)}")
    print(f"   Drude: eps_inf={fit['eps_inf']:.2f}  plasma={fit['plasma_thz']:.2f} THz  "
          f"damping={fit['damping_thz']:.2f} THz  gap={fit['gap_um']:.2f} um")
    print(f"   residual RMS {fit['residual_rms']:.3e} over {int(mask.sum())} bins  "
          f"(success={fit['success']})")
    fit["freq"] = freq; fit["mask"] = mask
    return fit


def main():
    print("===== process s-pol folder =====")
    ds_s = process_folder("s-pol", "s")
    print("===== process p-pol folder =====")
    ds_p = process_folder("p-pol", "p")

    print("\n===== single-polarisation diagnostics =====")
    print("  pole_dist=|1+r| (distance to the r=-1 mirror; LARGER=better conditioned, the true "
          "p-pol metric)")
    print("  |H|>1 frac = common-mode ALIGNMENT artifact (p-pol not expected to fix); |C| med = "
          "front-spot drift")
    print(f"{'sample':46s} {'pole|1+r|':>9s} {'|r|':>6s} {'|H|>1':>6s} {'|C|':>6s}")
    for label, ds in (("s-pol", ds_s), ("p-pol", ds_p)):
        for r in sample_diagnostics(ds):
            print(f"  {os.path.basename(r['name']):44s} {r['pole_dist']:>9.3f} {r['r_mag']:>6.3f} "
                  f"{r['frac_H_gt1']:>6.2f} {r['c_med']:>6.3f}")

    print("\n===== dual-pol joint Drude+gap fit on matched material axes =====")
    fits = {}
    # n_parallel: 0-0 (S) + 90-90 (P);  n_perp: 0-90 (S) + 90-0 (P)
    fits["n_parallel"] = joint_fit_axis("n_parallel (E || fibers)",
                                        ds_s, find_sample(ds_s, "0-0"),
                                        ds_p, find_sample(ds_p, "90-90"))
    fits["n_perp"] = joint_fit_axis("n_perp (E _|_ fibers)",
                                    ds_s, find_sample(ds_s, "0-90"),
                                    ds_p, find_sample(ds_p, "90-0"))

    # figure: fitted n,k for both axes
    figure, (ax_n, ax_k) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    figure.suptitle("CNT-21 dual-pol joint fit — anisotropy axes")
    for label, style in (("n_parallel", dict(color="C0")), ("n_perp", dict(color="C3"))):
        fit = fits[label]; ft = fit["freq"]*1e-12; b = fit["mask"] & (ft >= 0.3) & (ft <= 2.2)
        ax_n.plot(ft[b], fit["n"][b], label=f"{label} (gap {fit['gap_um']:.1f}um)", **style)
        ax_k.plot(ft[b], fit["k"][b], label=label, **style)
    ax_n.set_xlabel("THz"); ax_n.set_ylabel("n"); ax_n.legend(); ax_n.grid(alpha=0.3)
    ax_k.set_xlabel("THz"); ax_k.set_ylabel("k"); ax_k.legend(); ax_k.grid(alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "process_cnt21_polarization.png")
    figure.savefig(out, dpi=130); plt.close(figure)
    print(f"\nSaved figure: {out}")
    return ds_s, ds_p, fits


if __name__ == "__main__":
    main()
