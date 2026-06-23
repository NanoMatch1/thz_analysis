"""Build, validate, fit, and COMPARE the two rough-gap models on the CNT reflection data.

Route A = statistical-gap Fabry-Perot (flat CNT at a Gaussian-distributed distance d).
Route B = graded effective-medium layer (micro-rough air->CNT density gradient, Bruggeman).
Both share the Drude-Smith material model and are fit to the SAME measured reflection r_meas
(supplied by the thz_core pipeline). Methodology = synthetic-validate each route (plant ->
recover) BEFORE fitting the real CNT-0/CNT-90, per the conditioning report's warning that a
single measurement is degeneracy-limited.

Run:  PYTHONPATH=. .venv/Scripts/python.exe explorations/air_gap_cnt_reflection/compare_gap_models.py
"""

from __future__ import annotations

import os

import numpy as np
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explore_air_gap_deembedding import internal_angles
import air_gap_models as models

VACUUM_PERMITTIVITY = models.VACUUM_PERMITTIVITY

# Natural-unit parameter vector: [eps_inf, plasma_THz, tau_fs, c, gap_p1, gap_p2]
#   Route A gap = (d_mean_um, sigma_d_um);  Route B gap = (sigma_h_um, z0_um).
PARAM_LABELS_A = ["eps_inf", "plasma_THz", "tau_fs", "c", "d_mean_um", "sigma_d_um"]
PARAM_LABELS_B = ["eps_inf", "plasma_THz", "tau_fs", "c", "sigma_h_um", "z0_um"]
LOWER_A = [1.0, 0.5, 3.0, -1.0, 0.0, 0.1]
UPPER_A = [20.0, 40.0, 300.0, 0.0, 80.0, 40.0]
LOWER_B = [1.0, 0.5, 3.0, -1.0, 0.5, 0.0]
UPPER_B = [20.0, 40.0, 300.0, 0.0, 40.0, 80.0]
INIT_A = [4.0, 8.0, 30.0, -0.5, 20.0, 8.0]
INIT_B = [4.0, 8.0, 30.0, -0.5, 10.0, 10.0]


def material_from_vector(vector):
    return dict(eps_inf=vector[0], plasma_omega=2.0 * np.pi * vector[1] * 1e12,
                scattering_time_s=vector[2] * 1e-15, persistence_c=vector[3])


def forward_reflection(route, vector, frequency_hz, r_front, gap_angle_rad, sio2_angle_rad):
    material = material_from_vector(vector)
    if route == "A":
        return models.rough_gap_reflection(
            frequency_hz, material, vector[4] * 1e-6, vector[5] * 1e-6, r_front, gap_angle_rad)
    return models.graded_emt_reflection(
        frequency_hz, material, vector[4] * 1e-6, vector[5] * 1e-6, sio2_angle_rad)


def drude_smith_conductivity(frequency_hz, vector):
    """Complex optical conductivity (S/m), sigma_real > 0."""
    material = material_from_vector(vector)
    omega = 2.0 * np.pi * frequency_hz
    drude = 1.0 / (1.0 - 1j * omega * material["scattering_time_s"])
    return (VACUUM_PERMITTIVITY * material["plasma_omega"]**2 * material["scattering_time_s"]
            * drude * (1.0 + material["persistence_c"] * drude))


def fit_route(route, frequency_hz, r_meas, mask, r_front, gap_angle_rad, sio2_angle_rad, init, bounds):
    band = mask & np.isfinite(r_meas)

    def residual(vector):
        model = forward_reflection(route, vector, frequency_hz, r_front, gap_angle_rad, sio2_angle_rad)
        diff = (model - r_meas)[band]
        return np.concatenate([diff.real, diff.imag])

    result = least_squares(residual, init, bounds=bounds, method="trf",
                           x_scale="jac", max_nfev=4000)
    rms = np.sqrt(np.mean(result.fun**2))
    return result.x, rms


def recovered_nk_sigma(vector, frequency_hz):
    material = material_from_vector(vector)
    n_hat = models.cnt_index_hat(frequency_hz, material)
    sigma = drude_smith_conductivity(frequency_hz, vector)
    return np.real(n_hat), -np.imag(n_hat), sigma


# ── Synthetic validation (plant -> recover) ──────────────────────────────────


def synthetic_validation(route, frequency_hz, r_front, gap_angle_rad, sio2_angle_rad,
                         true_vector, init, bounds):
    r_true = forward_reflection(route, true_vector, frequency_hz, r_front, gap_angle_rad, sio2_angle_rad)
    rng = np.random.default_rng(0)
    noise = 2e-3 * (rng.standard_normal(r_true.shape) + 1j * rng.standard_normal(r_true.shape))
    r_noisy = r_true + noise
    mask = np.ones(frequency_hz.size, dtype=bool)
    recovered, rms = fit_route(route, frequency_hz, r_noisy, mask, r_front, gap_angle_rad, sio2_angle_rad, init, bounds)
    labels = PARAM_LABELS_A if route == "A" else PARAM_LABELS_B
    print(f"\n--- Route {route} synthetic recovery (planted vs fit) ---")
    for lab, t, f in zip(labels, true_vector, recovered):
        print(f"    {lab:11s}: true {t:8.2f}   fit {f:8.2f}   ({100*(f-t)/(abs(t)+1e-9):+6.1f}%)")
    print(f"    residual RMS = {rms:.2e}")
    return recovered


def main():
    sio2_angle_rad, gap_angle_rad = internal_angles()
    r_front = models.reflection_sio2_to_air(sio2_angle_rad)
    frequency_synth = np.linspace(0.3e12, 2.5e12, 200)

    print("=== SYNTHETIC VALIDATION (plant -> recover, 2e-3 noise) ===")
    true_A = [4.0, 8.0, 30.0, -0.5, 17.0, 8.0]
    true_B = [4.0, 8.0, 30.0, -0.5, 9.0, 6.0]
    synthetic_validation("A", frequency_synth, r_front, gap_angle_rad, sio2_angle_rad, true_A, INIT_A, (LOWER_A, UPPER_A))
    synthetic_validation("B", frequency_synth, r_front, gap_angle_rad, sio2_angle_rad, true_B, INIT_B, (LOWER_B, UPPER_B))

    # --- Real data ---
    print("\n=== REAL DATA: fit both routes to CNT-0 and CNT-90 ===")
    samples = models.load_measured_reflection()
    results = {}
    for name, record in samples.items():
        freq = record["frequency_hz"]; mask = record["mask"]; r_meas = record["r_meas"]
        vec_A, rms_A = fit_route("A", freq, r_meas, mask, r_front, gap_angle_rad, sio2_angle_rad, INIT_A, (LOWER_A, UPPER_A))
        vec_B, rms_B = fit_route("B", freq, r_meas, mask, r_front, gap_angle_rad, sio2_angle_rad, INIT_B, (LOWER_B, UPPER_B))
        results[name] = dict(record=record, vec_A=vec_A, rms_A=rms_A, vec_B=vec_B, rms_B=rms_B)
        band = mask
        print(f"\n  {name}:")
        print(f"    Route A (stat-gap): RMS {rms_A:.2e}  d_mean {vec_A[4]:.1f} um, sigma_d {vec_A[5]:.1f} um, "
              f"eps_inf {vec_A[0]:.1f}, wp {vec_A[1]:.1f} THz, tau {vec_A[2]:.0f} fs, c {vec_A[3]:.2f}")
        print(f"    Route B (graded EMT): RMS {rms_B:.2e}  sigma_h {vec_B[4]:.1f} um, z0 {vec_B[5]:.1f} um, "
              f"eps_inf {vec_B[0]:.1f}, wp {vec_B[1]:.1f} THz, tau {vec_B[2]:.0f} fs, c {vec_B[3]:.2f}")
        winner = "A" if rms_A < rms_B else "B"
        print(f"    -> better fit: Route {winner} (RMS {min(rms_A,rms_B):.2e} vs {max(rms_A,rms_B):.2e})")

    make_figure(results, r_front, gap_angle_rad, sio2_angle_rad)
    print("\nReminder: single-measurement material is degeneracy-limited (conditioning report). "
          "These are exploratory; the pressure series is the trustworthy path.")


def make_figure(results, r_front, gap_angle_rad, sio2_angle_rad):
    names = list(results)
    figure, axes = plt.subplots(len(names), 4, figsize=(18, 4.6 * len(names)), squeeze=False)
    figure.suptitle("Rough-gap models compared — Route A (statistical-gap FP) vs Route B (graded EMT)", fontsize=13)
    for row, name in enumerate(names):
        res = results[name]; record = res["record"]
        freq = record["frequency_hz"]; mask = record["mask"]; f_thz = freq * 1e-12
        band = mask & (f_thz >= 0.3) & (f_thz <= 2.5)
        n_A, k_A, sig_A = recovered_nk_sigma(res["vec_A"], freq)
        n_B, k_B, sig_B = recovered_nk_sigma(res["vec_B"], freq)
        model_A = models.rough_gap_reflection(freq, material_from_vector(res["vec_A"]),
                                              res["vec_A"][4]*1e-6, res["vec_A"][5]*1e-6, r_front, gap_angle_rad)
        model_B = models.graded_emt_reflection(freq, material_from_vector(res["vec_B"]),
                                               res["vec_B"][4]*1e-6, res["vec_B"][5]*1e-6, sio2_angle_rad)

        ax = axes[row][0]
        ax.plot(f_thz[band], record["n_naive"][band], "0.6", ls=":", label="naive (no model)")
        ax.plot(f_thz[band], n_A[band], "C0", label="A stat-gap")
        ax.plot(f_thz[band], n_B[band], "C3--", label="B graded-EMT")
        ax.axhline(1.0, color="0.5", ls=":", lw=0.8); ax.set_ylim(0, 6)
        ax.set_title(f"{name}: n"); ax.set_xlabel("THz"); ax.set_ylabel("n"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row][1]
        ax.plot(f_thz[band], record["k_naive"][band], "0.6", ls=":", label="naive")
        ax.plot(f_thz[band], k_A[band], "C0"); ax.plot(f_thz[band], k_B[band], "C3--")
        ax.set_title(f"{name}: k"); ax.set_xlabel("THz"); ax.set_ylabel("k"); ax.grid(alpha=0.3)

        ax = axes[row][2]
        ax.plot(f_thz[band], np.real(sig_A)[band], "C0", label="A")
        ax.plot(f_thz[band], np.real(sig_B)[band], "C3--", label="B")
        ax.set_title(f"{name}: Re sigma (S/m)"); ax.set_xlabel("THz"); ax.set_ylabel("S/m"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[row][3]
        ax.plot(f_thz[band], np.abs(record["r_meas"])[band], "k", lw=1.6, label="measured |r|")
        ax.plot(f_thz[band], np.abs(model_A)[band], "C0", label=f"A fit (RMS {res['rms_A']:.1e})")
        ax.plot(f_thz[band], np.abs(model_B)[band], "C3--", label=f"B fit (RMS {res['rms_B']:.1e})")
        ax.set_title(f"{name}: |r| fit quality"); ax.set_xlabel("THz"); ax.set_ylabel("|r|"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "compare_gap_models.png")
    figure.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
