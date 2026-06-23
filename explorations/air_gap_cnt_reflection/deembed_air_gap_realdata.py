"""Apply the air-gap de-embedding to REAL CNT reflection data (first real-data test).

Background: the prototype `explore_air_gap_deembedding.py` proved the exact Fabry-Perot
de-embed recovers n,k when the gap thickness d is known, on synthetic data.  This is the
parked "next step 1": run it on the measured CNT reflection and see whether n lifts back
above 1 (ANALYSIS_NOTES §9/§9b, [[air_gap_deembedding]]).

Chain (all quantities come straight from the reflection pipeline's processing_dict):
  r_meas  = processing_dict['reflection_r']   (= r_reference * H, the measured reflection)
  r_front = processing_dict['r_reference']    (= r_{SiO2->air}, the known window back face)
  x       = (r_meas - r_front) / (1 - r_front * r_meas)          -> = r_back * e^{-i 2beta}
  r_back  = x * e^{+i 2beta(d)}                                   (strip the gap delay)
  invert r_back with AIR incidence (n=1) at theta_gap = 45 deg   -> de-embedded n, k

The gap thickness d here is estimated from the linear slope of arg(x) (biased by the
sample's own dispersion, per the prototype) AND swept, so we can see (a) whether a
plausible d lifts n above 1 and (b) how sensitive n is to d.  A robust d from the
second-reflection pulse round-trip delay is the follow-up refinement.

Run:  PYTHONPATH=. ../.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/deembed_air_gap_realdata.py
"""

from __future__ import annotations

import os

import numpy as np

import thz_core.thz_core as core
from compare_reflection_pathways import run_shared_axis, PIPELINE_CONFIG, short_name

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
EXTERNAL_ANGLE_DEG = 45.0
REFRACTIVE_INDEX_SIO2 = 1.95
FIT_BAND_HZ = (0.4e12, 2.5e12)
PROBE_BAND_HZ = (1.0e12, 2.0e12)   # band over which we summarise n vs d


def internal_and_gap_angles() -> tuple[float, float]:
    """(theta_in_SiO2, theta_in_gap) for a 45 deg external beam; gap returns to 45 deg."""
    external = np.deg2rad(EXTERNAL_ANGLE_DEG)
    theta_sio2 = np.arcsin(np.sin(external) / REFRACTIVE_INDEX_SIO2)
    theta_gap = np.arcsin(REFRACTIVE_INDEX_SIO2 * np.sin(theta_sio2) / 1.0)
    return float(theta_sio2), float(theta_gap)


def gap_round_trip_phase(frequency_hz, gap_thickness_m, gap_angle_rad):
    """2*beta = (omega/c) * 2 d cos(theta_gap)."""
    omega = 2.0 * np.pi * frequency_hz
    return omega / SPEED_OF_LIGHT_M_PER_S * 2.0 * gap_thickness_m * np.cos(gap_angle_rad)


def deembed_gap(reflection_measured, reflection_front):
    """x = (r_meas - r_front)/(1 - r_front r_meas) = r_back * e^{-i 2beta}."""
    return (reflection_measured - reflection_front) / (
        1.0 - reflection_front * reflection_measured
    )


def estimate_gap_thickness(frequency_hz, x_complex, mask, gap_angle_rad):
    """Gap d from the linear slope of arg(x) in the fit band (biased; first estimate)."""
    in_band = (
        (frequency_hz >= FIT_BAND_HZ[0]) & (frequency_hz <= FIT_BAND_HZ[1])
        & mask & np.isfinite(x_complex)
    )
    omega = 2.0 * np.pi * frequency_hz[in_band]
    phase = np.unwrap(np.angle(x_complex[in_band]))
    slope, _ = np.polyfit(omega, phase, 1)
    return float(-slope * SPEED_OF_LIGHT_M_PER_S / (2.0 * np.cos(gap_angle_rad)))


def invert_air_incidence(frequency_hz, reflection_back, mask, gap_angle_rad):
    """Invert r_back = r_{air->CNT} with air incidence at the gap angle."""
    n, k, _ = core.invert_nk_reflection(
        frequency_hz, reflection_back, mask, theta_rad=gap_angle_rad, n_incident=1.0,
    )
    return n, k


def band_statistic(frequency_hz, values, mask, band_hz, reducer):
    in_band = (frequency_hz >= band_hz[0]) & (frequency_hz <= band_hz[1]) & mask & np.isfinite(values)
    return float(reducer(values[in_band])) if in_band.any() else np.nan


def main() -> None:
    theta_sio2, theta_gap = internal_and_gap_angles()
    print(f"angles: SiO2 internal {np.rad2deg(theta_sio2):.2f} deg, gap {np.rad2deg(theta_gap):.2f} deg "
          f"(cos {np.cos(theta_gap):.3f})")

    dataset = run_shared_axis(PIPELINE_CONFIG, eps_infinity=1.0)

    samples = []
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        processing = data_obj.processing_dict
        if processing.get("reflection_r") is None:
            continue
        samples.append((filename, processing))

    gap_sweep_m = np.linspace(0.0, 40e-6, 81)
    figure, axes = plt.subplots(len(samples), 3, figsize=(16, 4.6 * len(samples)), squeeze=False)
    figure.suptitle("Air-gap de-embedding on real CNT reflection data — n before/after, and d sensitivity", fontsize=12)

    for row, (filename, processing) in enumerate(samples):
        frequency = np.asarray(processing["fft_freq"], dtype=float)
        mask = np.asarray(processing["transfer_mask"], dtype=bool)
        r_meas = np.asarray(processing["reflection_r"], dtype=complex)
        r_front = complex(processing["r_reference"])
        n_naive = np.asarray(processing["n"], dtype=float)
        k_naive = np.asarray(processing["k"], dtype=float)

        x = deembed_gap(r_meas, r_front)
        gap_estimate_m = estimate_gap_thickness(frequency, x, mask, theta_gap)

        r_back_est = x * np.exp(1j * gap_round_trip_phase(frequency, gap_estimate_m, theta_gap))
        n_deembed, k_deembed = invert_air_incidence(frequency, r_back_est, mask, theta_gap)

        # d sweep: band-min and band-mean of the de-embedded n vs gap thickness.
        band_min_vs_d, band_mean_vs_d = [], []
        for gap_thickness_m in gap_sweep_m:
            r_back = x * np.exp(1j * gap_round_trip_phase(frequency, gap_thickness_m, theta_gap))
            n_sweep, _ = invert_air_incidence(frequency, r_back, mask, theta_gap)
            band_min_vs_d.append(band_statistic(frequency, n_sweep, mask, PROBE_BAND_HZ, np.nanmin))
            band_mean_vs_d.append(band_statistic(frequency, n_sweep, mask, PROBE_BAND_HZ, np.nanmean))
        band_min_vs_d = np.array(band_min_vs_d)
        # smallest d that lifts the whole probe band to n >= 1
        lifts = np.where(band_min_vs_d >= 1.0)[0]
        d_lift_m = gap_sweep_m[lifts[0]] if lifts.size else np.nan

        f_thz = frequency * 1e-12
        plot_band = (f_thz >= 0.3) & (f_thz <= 2.5) & mask

        ax = axes[row][0]
        ax.plot(f_thz[plot_band], n_naive[plot_band], "C3--", lw=1.5, label="naive (no de-embed)")
        ax.plot(f_thz[plot_band], n_deembed[plot_band], "C0", lw=1.6,
                label=f"de-embedded (d={gap_estimate_m*1e6:.1f} um, arg-x)")
        ax.axhline(1.0, color="0.5", ls=":", lw=1.0)
        ax.set_ylim(0, 4); ax.set_title(f"{short_name(filename)}: n", fontsize=10)
        ax.set_xlabel("THz"); ax.set_ylabel("n"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row][1]
        ax.plot(f_thz[plot_band], k_naive[plot_band], "C3--", lw=1.5, label="naive")
        ax.plot(f_thz[plot_band], k_deembed[plot_band], "C0", lw=1.6, label="de-embedded")
        ax.set_title(f"{short_name(filename)}: k", fontsize=10)
        ax.set_xlabel("THz"); ax.set_ylabel("k"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row][2]
        ax.plot(gap_sweep_m * 1e6, band_min_vs_d, "C0", label=f"min n over {PROBE_BAND_HZ[0]/1e12:.0f}-{PROBE_BAND_HZ[1]/1e12:.0f} THz")
        ax.plot(gap_sweep_m * 1e6, band_mean_vs_d, "C2", label="mean n")
        ax.axhline(1.0, color="0.5", ls=":", lw=1.0)
        ax.axvline(gap_estimate_m * 1e6, color="C3", ls="--", lw=1.0, label=f"arg-x d={gap_estimate_m*1e6:.1f} um")
        if np.isfinite(d_lift_m):
            ax.axvline(d_lift_m * 1e6, color="k", ls=":", lw=1.0, label=f"n>=1 at d={d_lift_m*1e6:.1f} um")
        ax.set_title(f"{short_name(filename)}: n vs gap d", fontsize=10)
        ax.set_xlabel("gap thickness d (um)"); ax.set_ylabel("n in band"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        n_naive_min = band_statistic(frequency, n_naive, mask, PROBE_BAND_HZ, np.nanmin)
        n_deembed_min = band_statistic(frequency, n_deembed, mask, PROBE_BAND_HZ, np.nanmin)
        print(f"\n{short_name(filename)}:")
        print(f"  r_front (SiO2->air) = {r_front:.3f}")
        print(f"  arg(x)-slope gap estimate = {gap_estimate_m*1e6:.1f} um")
        print(f"  min n in {PROBE_BAND_HZ[0]/1e12:.0f}-{PROBE_BAND_HZ[1]/1e12:.0f} THz: "
              f"naive {n_naive_min:.2f} -> de-embedded(arg-x d) {n_deembed_min:.2f}")
        print(f"  smallest d lifting whole band to n>=1: "
              f"{d_lift_m*1e6:.1f} um" if np.isfinite(d_lift_m) else "  no d in sweep lifts whole band to n>=1")

    output_path = os.path.join(os.path.dirname(__file__), "deembed_air_gap_realdata.png")
    figure.savefig(output_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure: {output_path}")


if __name__ == "__main__":
    main()
