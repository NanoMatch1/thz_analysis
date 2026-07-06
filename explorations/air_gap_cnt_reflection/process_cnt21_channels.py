"""Process CNT-21 polarization channels with the CORRECTED per-channel reference pairing.

Naming (Samuel, 2026-07-02): label X-Y means X = THz polarization w.r.t. the plane of
reflection (0=S, 90=P) and Y = SAMPLE/MOUNT ORIENTATION (0 = fibers vertical, 90 = rotated
90 deg). The two-letter reference tags (SS/SP/PS/PP) are pol + orientation-slot, NOT
source-detection. The SiO2 window faces are not parallel, so rotating the mount precesses the
front reflection around the frequency-anisotropic EO detection cone -> each sample must pair
with the reference recorded at ITS OWN pol + orientation slot:

    sample_0-0   <-> reference_0-0  (SS)   S-pol, E || fibers  -> n_parallel
    sample_0-90  <-> reference_0-90 (SP)   S-pol, E _|_ fibers -> n_perp
    sample_90-0  <-> reference_90-0 (PS)   P-pol, E _|_ fibers -> n_perp
    sample_90-90 <-> reference_90-90(PP)   P-pol, E || fibers  -> n_parallel

(The F23 run paired both S samples to SS and both P samples to PP — mispaired for 0-90/90-0.
Two deliberately-mispaired control runs are included here to quantify what that cost.)

Each channel is processed as its own two-file mini-dataset (sample + its reference copied to a
scratch folder) so the auto-grouper is unambiguous. Results are saved per channel as .npz for
the downstream de-embed work.

Run from repo root:
  PYTHONPATH=. ./.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/process_cnt21_channels.py
"""

from __future__ import annotations

import os
import shutil
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz

warnings.filterwarnings("ignore")
plt.show = lambda *a, **k: None  # headless

DATA_ROOT = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-21\polarization"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cnt21_channel_results")
SCRATCH_ROOT = os.path.join(OUTPUT_DIR, "_scratch_pairs")

# regions for the shared CNT-21 grid (pulses at 155.20 / 180.05 ps, trace [153.0, 184.45])
REGIONS_CNT21 = {"first_reflection": (152.5, 158.5), "second_reflection": (177.0, 183.5)}
# the old silicon s-pol pair sits on its own grid (pulses ~152.6 / 177.4 ps)
REGIONS_SI_OLD = {"first_reflection": (150.0, 156.0), "second_reflection": (174.4, 180.9)}

PROBE_BAND_THZ = (0.5, 2.0)

# channel -> dict(sample, reference, polarization, regions)
CHANNELS = {
    "0-0": dict(sample="s-pol/sample_0-0_CNT-0-0_cam-c.acc",
                reference="s-pol/reference_0-0_sio2-0-0_SS_cam-colinear.acc",
                polarization="s", regions=REGIONS_CNT21),
    "0-90": dict(sample="s-pol/sample_0-90_CNT-0-90_cam-c.acc",
                 reference="s-pol/reference_0-90_sio2-0-90_SP_cam-colinear.acc",
                 polarization="s", regions=REGIONS_CNT21),
    "90-0": dict(sample="p-pol/sample_90-0_CNT-90-0_cam-c.acc",
                 reference="p-pol/reference_90-0_sio2-90-0_PS_cam-colinear.acc",
                 polarization="p", regions=REGIONS_CNT21),
    "90-0R": dict(sample="p-pol/sample_90-0_CNT-90-0_cam-c_realign.acc",
                  reference="p-pol/reference_90-0_sio2-90-0_PS_cam-colinear.acc",
                  polarization="p", regions=REGIONS_CNT21),
    "90-90": dict(sample="p-pol/sample_90-90_CNT-90-90_cam-c.acc",
                  reference="p-pol/reference_90-90_sio2-90-90_PP_cam-colinear.acc",
                  polarization="p", regions=REGIONS_CNT21),
    "Si-p": dict(sample="silicon/p-pol/sample_90-0_silicon-FZ-90-0_PS_cam-colinear.acc",
                 reference="silicon/p-pol/reference_90-0_sio2-90-0_PS_cam-colinear.acc",
                 polarization="p", regions=REGIONS_CNT21),
    "Si-s-old": dict(sample="silicon/s-pol/sample_0-0_si_old.acc",
                     reference="silicon/s-pol/reference_0-0_sio2_old.acc",
                     polarization="s", regions=REGIONS_SI_OLD),
    # deliberately MISPAIRED controls reproducing the F23 pairing (co-pol tag instead of
    # per-orientation reference) — quantifies what the mispairing cost.
    "0-90_MISPAIRED": dict(sample="s-pol/sample_0-90_CNT-0-90_cam-c.acc",
                           reference="s-pol/reference_0-0_sio2-0-0_SS_cam-colinear.acc",
                           polarization="s", regions=REGIONS_CNT21),
    "90-0_MISPAIRED": dict(sample="p-pol/sample_90-0_CNT-90-0_cam-c.acc",
                           reference="p-pol/reference_90-90_sio2-90-90_PP_cam-colinear.acc",
                           polarization="p", regions=REGIONS_CNT21),
}


def build_channel_config(polarization, regions):
    """Fresh config per channel (build_full_trace_reflection self-modifies config.regions)."""
    return {
        "general": {"show_graph": False, "save_database": False},
        "geometry": {"theta_external_deg": 45.0, "n_sio2": 1.96, "polarization": polarization},
        "regions": {key: tuple(value) for key, value in regions.items()},
        # first pulse sits ~2.2 ps from trace start; 2 ps half-width keeps both pulses clean (F23)
        "window": {"type": "hann", "alpha": 1.0, "half_width_ps": 2},
        "fft": {"n_fft": 2000},
        "transfer": {"self_reference": True, "apply_snr_mask": True, "min_ref_amp_rel": 1e-3,
                     "regularization_eps": 1e-30, "unwrap_phase": True},
        "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
        "derive": {"eps_background": 1},
    }


def repair_sparse_time_axis(path):
    """Zero-fill unrecorded gaps in a .acc so every scan sits on one uniform dt grid.

    Some acquisitions record two time windows and simply omit the samples between them
    (e.g. the Si-p sample: 153.00-159.65 + 177.00-184.45 ps with a 17.35 ps hole). The
    pipeline FFT assumes uniform sampling, so an omitted gap silently COLLAPSES the
    inter-pulse delay (Si-p: 24.85 -> 7.56 ps, a spurious +17.3 ps linear phase in H).
    Zero amplitude in the hole is exactly what build_full_trace_reflection would impose
    anyway (it zeros between the reflection regions).
    """
    with open(path, "r") as handle:
        lines = handle.read().splitlines()

    output_lines = []
    block = []          # accumulated (time, value) rows of the current scan block
    n_filled_total = 0

    def flush_block():
        nonlocal n_filled_total
        if not block:
            return
        times = np.array([row[0] for row in block])
        step = float(np.median(np.diff(times))) if len(times) > 1 else None
        previous_time = None
        for time_value, line in block:
            if previous_time is not None and step and (time_value - previous_time) > 1.5 * step:
                n_missing = int(round((time_value - previous_time) / step)) - 1
                for k in range(1, n_missing + 1):
                    output_lines.append(f"{previous_time + k * step:.18e} "
                                        f"{0.0:.18e}")
                n_filled_total += n_missing
            output_lines.append(line)
            previous_time = time_value
        block.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            flush_block()
            output_lines.append(line)
            continue
        time_value = float(stripped.split()[0])
        block.append((time_value, line))
    flush_block()

    if n_filled_total:
        with open(path, "w") as handle:
            handle.write("\n".join(output_lines) + "\n")
        print(f"  [repair_sparse_time_axis] {os.path.basename(path)}: zero-filled "
              f"{n_filled_total} missing samples across scan blocks")


def stage_pair_folder(channel_name, spec):
    """Copy the channel's sample + reference into a scratch folder; return its path."""
    pair_dir = os.path.join(SCRATCH_ROOT, channel_name)
    if os.path.isdir(pair_dir):
        shutil.rmtree(pair_dir)
    os.makedirs(pair_dir)
    for relative in (spec["sample"], spec["reference"]):
        destination = shutil.copy2(os.path.join(DATA_ROOT, relative), pair_dir)
        repair_sparse_time_axis(destination)
    return pair_dir


def process_channel(channel_name, spec):
    """Run the windowed self-referenced pipeline on one (sample, reference) pair."""
    config = build_channel_config(spec["polarization"], spec["regions"])
    pair_dir = stage_pair_folder(channel_name, spec)
    dataset = DataSet(pair_dir, config=config)
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)

    thz.build_full_trace_reflection(dataset)
    thz.taper_and_pad_traces(dataset)
    dataset.group_files(keywords=["type"])
    thz.define_reflection_regions(dataset, config)
    thz.subtract_baseline(dataset)
    thz.window_pulses_fixed_width(dataset, half_width_ps=config["window"]["half_width_ps"],
                                  show_graph=False)
    thz.fft_spectrum(dataset, segment="second_reflection", n_fft=config["fft"]["n_fft"])
    thz.fft_spectrum(dataset, segment="first_reflection", n_fft=config["fft"]["n_fft"])
    thz.transfer_function(dataset, ref_type="reference")
    thz.selfref_quality(dataset)
    thz.invert_nk_reflection(dataset, geometry="window",
                             theta_deg=config["geometry"]["theta_external_deg"],
                             polarization=spec["polarization"],
                             n_window=config["geometry"]["n_sio2"])
    thz.derive_eps_sigma(dataset, config={"derive": {"eps_background": 1}})

    sample_name = next(fn for fn in dataset.data.keys() if not dataset.data.is_reference(fn))
    return dataset, sample_name


def extract_results(dataset, sample_name, spec):
    """Pull the arrays the de-embed work needs out of the processing_dict."""
    processing = dataset.data[sample_name].processing_dict
    freq = np.asarray(processing["fft_freq"], float)
    return dict(
        freq=freq,
        mask=np.asarray(processing["transfer_mask"], bool),
        transfer_H=np.asarray(processing["transfer_H"], complex),
        reflection_r=np.asarray(processing["reflection_r"], complex),
        r_reference=np.complex128(processing["r_reference"]),
        selfref_C=np.asarray(processing["selfref_correction"], complex),
        n=np.asarray(processing["n"], float),
        k=np.asarray(processing["k"], float),
        sigma=np.asarray(processing["sigma"], complex),
        polarization=spec["polarization"],
        theta_external_deg=45.0,
        n_sio2=1.96,
    )


def channel_diagnostics(result):
    """Band-limited health metrics for one channel."""
    freq_thz = result["freq"] * 1e-12
    mask = result["mask"]
    band = mask & (freq_thz >= PROBE_BAND_THZ[0]) & (freq_thz <= PROBE_BAND_THZ[1])
    H = result["transfer_H"]
    r = result["reflection_r"]
    C = result["selfref_C"]
    finite_H = np.isfinite(H) & mask
    band_r = band & np.isfinite(r)
    band_C = band & np.isfinite(C)

    def n_at(target_thz):
        finite_n = band & np.isfinite(result["n"])
        if not finite_n.any():
            return np.nan
        index = np.argmin(np.abs(freq_thz[finite_n] - target_thz))
        return float(result["n"][finite_n][index])

    return dict(
        frac_H_gt1=float(np.mean(np.abs(H[finite_H]) > 1.0)) if finite_H.any() else np.nan,
        C_median=float(np.nanmedian(np.abs(C[band_C]))) if band_C.any() else np.nan,
        C_std=float(np.nanstd(np.abs(C[band_C]))) if band_C.any() else np.nan,
        pole_distance=float(np.nanmedian(np.abs(1 + r[band_r]))) if band_r.any() else np.nan,
        n_median=float(np.nanmedian(result["n"][band])) if band.any() else np.nan,
        k_median=float(np.nanmedian(result["k"][band])) if band.any() else np.nan,
        n_low=n_at(0.5), n_mid=n_at(1.0), n_high=n_at(1.8),
        n_bins=int(band.sum()),
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}
    for channel_name, spec in CHANNELS.items():
        print(f"\n===== channel {channel_name} =====")
        dataset, sample_name = process_channel(channel_name, spec)
        result = extract_results(dataset, sample_name, spec)
        results[channel_name] = result
        np.savez(os.path.join(OUTPUT_DIR, f"channel_{channel_name}.npz"), **result)

    print("\n\n===== channel diagnostics (band {}-{} THz) =====".format(*PROBE_BAND_THZ))
    print("  pole|1+r| = distance to r=-1 mirror (larger = better conditioned)")
    print(f"{'channel':16s} {'pol':>3s} {'pole|1+r|':>9s} {'|H|>1':>6s} {'|C|med':>7s} "
          f"{'|C|std':>7s} {'n_med':>6s} {'k_med':>6s} {'n@0.5':>6s} {'n@1.0':>6s} "
          f"{'n@1.8':>6s} {'bins':>5s}")
    for channel_name, result in results.items():
        diag = channel_diagnostics(result)
        print(f"{channel_name:16s} {result['polarization']:>3s} "
              f"{diag['pole_distance']:>9.3f} {diag['frac_H_gt1']:>6.2f} "
              f"{diag['C_median']:>7.3f} {diag['C_std']:>7.3f} "
              f"{diag['n_median']:>6.2f} {diag['k_median']:>6.2f} "
              f"{diag['n_low']:>6.2f} {diag['n_mid']:>6.2f} {diag['n_high']:>6.2f} "
              f"{diag['n_bins']:>5d}")

    # overview figure: n and k per channel (correctly paired only)
    plot_channels = [c for c in results if not c.endswith("_MISPAIRED")]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    figure.suptitle("CNT-21 corrected per-channel pairing — naive window inversion")
    panels = [("0-0", "0-90", "s-pol CNT"), ("90-0", "90-0R", "p-pol CNT (90-0 + realign)"),
              ("90-90", None, "p-pol CNT (90-90)"), ("Si-p", "Si-s-old", "silicon control")]
    for ax, (channel_a, channel_b, title) in zip(axes.flat, panels):
        for channel, color in ((channel_a, "C0"), (channel_b, "C3")):
            if channel is None:
                continue
            result = results[channel]
            freq_thz = result["freq"] * 1e-12
            band = result["mask"] & (freq_thz >= 0.3) & (freq_thz <= 2.5)
            ax.plot(freq_thz[band], result["n"][band], color=color, lw=1.5,
                    label=f"{channel} n")
            ax.plot(freq_thz[band], result["k"][band], color=color, lw=1.2, ls="--",
                    label=f"{channel} k")
        if title.startswith("silicon"):
            ax.axhline(3.418, color="0.6", ls=":", lw=1.0, label="n_Si = 3.418")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("THz"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    output_path = os.path.join(OUTPUT_DIR, "cnt21_channels_overview.png")
    figure.savefig(output_path, dpi=130)
    print(f"\nSaved figure: {output_path}")
    print(f"Saved per-channel arrays to {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    main()
