"""Compare the two reflection processing pathways on the latest CNT data.

Runs the same dataset (2026_06_19_CNT/test1/export: SiO2 reference + CNT-0deg + CNT-90deg)
through BOTH reflection pathways and overlays the results:

  - shared_axis  : self-referencing ON  (both reflections on one shared time axis)
  - segmented    : self-referencing OFF (each reflection cropped + processed separately)

Both: align on the FIRST reflection only (front-face window pulse = stable T0), and carry
the sub-sample timing correction all the way through (applied as a spectral phase ramp in
transfer_function). Headless — presets only, no SpanSelectors.

Outputs n, k, and conductivity (real + imag) per sample, both pathways overlaid.

Run:  PYTHONPATH=. ../.venv/Scripts/python.exe explorations/compare_reflection_pathways.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Headless: neutralise every blocking show in the pipeline.
plt.show = lambda *args, **kwargs: None

import numpy as np

from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz

# Make sure the sub-sample timing correction is on for the whole run.
thz.SUBSAMPLE_TIMING_CORRECTION = True

ROOT_DIR = r"C:\Users\Samuel\Data\THz\calibration\reflection\2026_06_19_CNT\test1\export"

# Reflection regions for the NEW (thicker-window) data: first pulse ~155.85 ps,
# second ~180.8 ps (inspected from the scan-averaged traces).
FIRST_REGION_PS = (153.5, 159.0)
SECOND_REGION_PS = (178.5, 183.5)

# ε∞ for the CNT conductivity (sigma = -i*omega*eps0*(eps - eps_inf)).
# NOTE: the pipeline default is 1.0 (vacuum).  This shifts only sigma_IMAG; sigma_REAL
# (= omega*eps0*2nk) is independent of eps_inf.  See the printed note / report.
CNT_EPS_INFINITY = 1.0

PIPELINE_CONFIG = {
    "root_dir": ROOT_DIR,
    "theta_external_deg": 45.0,
    "polarization": "s",
    "n_sio2": 1.95,
    "center_mode": "crop",
    "n_fft": 4096,
    "regions": {"first_reflection": FIRST_REGION_PS, "second_reflection": SECOND_REGION_PS},
    "gates": {"first_reflection": FIRST_REGION_PS, "second_reflection": SECOND_REGION_PS},
    "window": {"type": "hann", "alpha": 1.0},
    "centering": {"peak_mode": "auto", "taper_ps": 1.0},
    "pad": {"n_samples": 500},
}


def _fresh_dataset() -> DataSet:
    dataset = DataSet(ROOT_DIR)
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)
    return dataset


def run_shared_axis(config: dict, eps_infinity: float) -> DataSet:
    """Shared-axis pathway with self-referencing ON (headless)."""
    dataset = _fresh_dataset()
    thz.build_full_trace_reflection(dataset)
    dataset.group_files(keywords=["type"])

    thz.align_to_reference(
        dataset, timing_segment="first_reflection",
        roi=config["regions"]["first_reflection"],
        subsample_correction=True, show_graph=False,
    )
    thz.subtract_baseline(dataset, segment="second_reflection")
    thz.subtract_baseline(dataset, segment="first_reflection")
    thz.global_truncate(dataset, segment="second_reflection")
    thz.global_truncate(dataset, segment="first_reflection")

    thz.define_reflection_regions(dataset, config)
    thz.isolate_and_window(
        dataset, config={"window": config["window"]},
        center_mode=config["center_mode"], show_graph=False,
    )
    thz.fft_spectrum(dataset, segment="second_reflection", n_fft=config["n_fft"])
    thz.fft_spectrum(dataset, segment="first_reflection", n_fft=config["n_fft"])

    thz.transfer_function(
        dataset, config={"transfer": {"self_reference": True}}, ref_type="reference",
    )
    thz.invert_nk_reflection(
        dataset, geometry="window", theta_deg=config["theta_external_deg"],
        polarization=config["polarization"], n_window=config["n_sio2"],
    )
    thz.derive_eps_sigma(dataset, config={"derive": {"eps_background": eps_infinity}})
    return dataset


def run_segmented(config: dict, eps_infinity: float) -> DataSet:
    """Segmented pathway with self-referencing OFF (headless)."""
    dataset = _fresh_dataset()
    thz.build_reflection_dataset(dataset, config["gates"])
    dataset.group_files(keywords=["type"])

    thz.align_to_reference(
        dataset, timing_segment="first_reflection",
        subsample_correction=True, show_graph=False,
    )
    for segment in ("second_reflection", "first_reflection"):
        thz.subtract_baseline(dataset, segment=segment)
        thz.global_truncate(dataset, segment=segment)
        thz.centering_manual(
            dataset, segment=segment,
            auto_range_ps=config["gates"].get(segment), show_graph=False,
        )
        thz.center_pulse(dataset, segment=segment,
                         config={"centering": config["centering"]}, show_graph=False)
        thz.window_time(dataset, segment=segment,
                        config={"window": config["window"]}, show_graph=False)
        thz.zero_pad(dataset, segment=segment, config={"pad": config["pad"]}, show_graph=False)

    thz.fft_spectrum(dataset, segment="second_reflection")
    any_obj = next(iter(dataset.data.values()))
    second_freq = any_obj.processing_dict.get("fft_freq")
    shared_n_fft = (2 * (len(second_freq) - 1)) if second_freq is not None else None
    thz.fft_spectrum(dataset, segment="first_reflection", n_fft=shared_n_fft)

    thz.transfer_function(
        dataset, config={"transfer": {"self_reference": False}}, ref_type="reference",
    )
    thz.invert_nk_reflection(
        dataset, geometry="window", theta_deg=config["theta_external_deg"],
        polarization=config["polarization"], n_window=config["n_sio2"],
    )
    thz.derive_eps_sigma(dataset, config={"derive": {"eps_background": eps_infinity}})
    return dataset


def collect_sample_results(dataset: DataSet) -> dict:
    """Return {sample_name: dict(freq_thz, n, k, sigma)} for non-reference entries."""
    results = {}
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        processing = data_obj.processing_dict
        frequency_hz = processing.get("fft_freq")
        if frequency_hz is None or processing.get("n") is None:
            continue
        results[filename] = dict(
            freq_thz=frequency_hz * 1e-12,
            n=processing.get("n"),
            k=processing.get("k"),
            sigma=processing.get("sigma"),
        )
    return results


def short_name(filename: str) -> str:
    """Compact label, e.g. 'CNT-0-deg'."""
    for token in filename.split("_"):
        if "CNT" in token:
            return token
    return filename


def plot_comparison(shared: dict, segmented: dict, band_thz=(0.2, 3.0)) -> str:
    """Overlay n, k, sigma_real, sigma_imag for each sample, both pathways."""
    sample_names = sorted(set(shared) | set(segmented))
    quantities = [
        ("n", "refractive index n", lambda r: np.real(r["n"])),
        ("k", "extinction k", lambda r: np.real(r["k"])),
        ("sigma_real", "Re sigma (S/m)", lambda r: np.real(r["sigma"])),
        ("sigma_imag", "Im sigma (S/m)", lambda r: np.imag(r["sigma"])),
    ]
    y_limits = {"n": (0.0, 3.0), "k": (0.0, 6.0)}
    n_rows = len(sample_names)
    figure, axes = plt.subplots(
        n_rows, 4, figsize=(20, 5.2 * n_rows), squeeze=False, layout="constrained",
    )
    figure.suptitle(
        "Reflection pathway comparison — shared-axis (self-ref ON) vs segmented (self-ref OFF)\n"
        "first-reflection alignment, sub-sample correction carried through",
        fontsize=13,
    )
    for row, sample_name in enumerate(sample_names):
        for col, (key, ylabel, getter) in enumerate(quantities):
            axis = axes[row][col]
            for results, style, label in (
                (shared, dict(color="C0", lw=1.6), "shared (self-ref ON)"),
                (segmented, dict(color="C3", lw=1.6, ls="--"), "segmented (self-ref OFF)"),
            ):
                if sample_name not in results:
                    continue
                record = results[sample_name]
                frequency = record["freq_thz"]
                value = getter(record)
                in_band = (frequency >= band_thz[0]) & (frequency <= band_thz[1])
                axis.plot(frequency[in_band], value[in_band], label=label, **style)
            if key == "n":
                axis.axhline(1.0, color="0.5", ls=":", lw=1.0)  # n = 1 reference line
            if key in y_limits:
                axis.set_ylim(*y_limits[key])
            if col == 0:
                axis.set_ylabel(f"{short_name(sample_name)}", fontsize=11, fontweight="bold")
            axis.set_title(ylabel, fontsize=11)
            axis.set_xlabel("Frequency (THz)", fontsize=9)
            axis.grid(alpha=0.3)
            if row == 0 and col == 0:
                axis.legend(fontsize=9)
    output_path = os.path.join(os.path.dirname(__file__), "compare_reflection_pathways.png")
    figure.savefig(output_path, dpi=130)
    return output_path


def report_band(name: str, results: dict, band_thz=(0.5, 2.5)) -> None:
    for sample_name, record in results.items():
        frequency = record["freq_thz"]
        in_band = (frequency >= band_thz[0]) & (frequency <= band_thz[1]) & np.isfinite(record["n"])
        if not in_band.any():
            continue
        n_band = np.real(record["n"])[in_band]
        k_band = np.real(record["k"])[in_band]
        print(f"  [{name}] {short_name(sample_name)}: "
              f"n {np.nanmean(n_band):.3f} ({np.nanmin(n_band):.3f}-{np.nanmax(n_band):.3f}), "
              f"k {np.nanmean(k_band):.3f}, "
              f"n<1 in {100*np.mean(n_band < 1.0):.0f}% of band")


def main() -> None:
    print(f"Reflection pathway comparison on {ROOT_DIR}")
    print(f"  regions: first {FIRST_REGION_PS} ps, second {SECOND_REGION_PS} ps; "
          f"eps_inf={CNT_EPS_INFINITY}")

    shared_dataset = run_shared_axis(PIPELINE_CONFIG, CNT_EPS_INFINITY)
    segmented_dataset = run_segmented(PIPELINE_CONFIG, CNT_EPS_INFINITY)

    shared_results = collect_sample_results(shared_dataset)
    segmented_results = collect_sample_results(segmented_dataset)

    print("\nBand-mean (0.5-2.5 THz):")
    report_band("shared", shared_results)
    report_band("segmented", segmented_results)

    output_path = plot_comparison(shared_results, segmented_results)
    print(f"\nSaved comparison figure: {output_path}")


if __name__ == "__main__":
    main()
