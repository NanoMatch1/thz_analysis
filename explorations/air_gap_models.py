"""Shared ad-hoc models for rough-contact CNT reflection + a CONDITIONING REPORT.

Script 1 of the gap/roughness modelling plan (reports/gap_roughness_modelling_implementation_plan.md).
This module holds the pieces both fitting routes will reuse, and its main() answers the
gating question BEFORE we build the fitters: is the inverse problem well-posed for OUR
sample, and where does it become degenerate?

It does three things on SYNTHETIC ground truth (Drude-Smith CNT + a rough gap), so the
answer is interpretable:
  1. Parameter conditioning of a SINGLE measurement — Jacobian/Fisher analysis: which of
     {eps_inf, omega_p, tau, c, d_mean, sigma_d} are well determined, and how strongly the
     GAP parameters correlate with the MATERIAL parameters (the degeneracy we worry about).
  2. The same for a synthetic pressure SERIES (shared material, several gaps) — does the
     global fit lift the material out of the degeneracy?
  3. The Kramers-Kronig / minimum-phase PREMISE check: how fast does surface roughness
     (sigma_d) corrupt |x| away from the true |r_back|?  This is where KK (arXiv 2412.18662)
     stops being valid for us.

Pipeline-as-machinery: the fitters import `load_measured_reflection` (runs the thz_core
reflection pipeline); the conditioning report itself is pure synthetic so it needs no data.

Sign convention: thz_core n_hat = n - i k (k>=0), exp(-i omega t).
Run:  PYTHONPATH=. ../.venv/Scripts/python.exe explorations/air_gap_models.py
"""

from __future__ import annotations

import numpy as np

import thz_core.thz_core as core
from explore_air_gap_deembedding import (
    SPEED_OF_LIGHT_M_PER_S, REFRACTIVE_INDEX_SIO2, REFRACTIVE_INDEX_AIR, internal_angles,
)

VACUUM_PERMITTIVITY = 8.8541878128e-12


# ── Material model: Drude-Smith CNT permittivity -> air->CNT reflection ──────


def drude_smith_permittivity(frequency_hz, eps_inf, plasma_omega, scattering_time_s, persistence_c):
    """Drude-Smith complex permittivity (thz_core n_hat = n - i k convention).

    sigma(w) = eps0 wp^2 tau / (1 - i w tau) * [1 + c/(1 - i w tau)]
    eps(w)   = eps_inf + i sigma(w) / (w eps0)
    Returns eps with the sign so that sqrt gives n + i k_phys; the caller conjugates.
    """
    omega = 2.0 * np.pi * frequency_hz
    drude = 1.0 / (1.0 - 1j * omega * scattering_time_s)
    sigma = VACUUM_PERMITTIVITY * plasma_omega**2 * scattering_time_s * drude * (1.0 + persistence_c * drude)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = eps_inf + 1j * sigma / (omega * VACUUM_PERMITTIVITY)
    return eps


def cnt_index_hat(frequency_hz, material_params):
    """Complex index n_hat = n - i k (thz_core convention) from Drude-Smith params."""
    eps = drude_smith_permittivity(
        frequency_hz, material_params["eps_inf"], material_params["plasma_omega"],
        material_params["scattering_time_s"], material_params["persistence_c"],
    )
    n_hat_plus = np.sqrt(eps)                 # principal branch -> n + i k
    return np.conj(n_hat_plus)                # -> n - i k (thz_core)


def reflection_air_to_cnt(frequency_hz, material_params, gap_angle_rad):
    """r_back = r_{air->CNT}(omega), s-pol, at the in-gap angle."""
    return core.fresnel_reflection_s(REFRACTIVE_INDEX_AIR, cnt_index_hat(frequency_hz, material_params), gap_angle_rad)


def reflection_sio2_to_air(sio2_angle_rad):
    """r_front = r_{SiO2->air} (real scalar), s-pol."""
    return complex(core.fresnel_reflection_s(REFRACTIVE_INDEX_SIO2, REFRACTIVE_INDEX_AIR, sio2_angle_rad))


# ── Gap forward models ──────────────────────────────────────────────────────


def gap_round_trip_phase(frequency_hz, gap_thickness_m, gap_angle_rad):
    omega = 2.0 * np.pi * frequency_hz
    return omega / SPEED_OF_LIGHT_M_PER_S * 2.0 * gap_thickness_m * np.cos(gap_angle_rad)


def single_gap_reflection(frequency_hz, material_params, gap_thickness_m, r_front, gap_angle_rad):
    """Exact single-gap Fabry-Perot reflection SiO2|air d|CNT."""
    r_back = reflection_air_to_cnt(frequency_hz, material_params, gap_angle_rad)
    bounce = np.exp(-1j * gap_round_trip_phase(frequency_hz, gap_thickness_m, gap_angle_rad))
    return (r_front + r_back * bounce) / (1.0 + r_front * r_back * bounce)


def rough_gap_reflection(frequency_hz, material_params, gap_mean_m, gap_sigma_m, r_front, gap_angle_rad,
                         n_quadrature=41):
    """Spot-averaged reflection over a Gaussian gap distribution truncated at d>=0.

    Numerical average (keeps all bounces).  sigma=0 reduces to the single gap.
    """
    if gap_sigma_m <= 0:
        return single_gap_reflection(frequency_hz, material_params, gap_mean_m, r_front, gap_angle_rad)
    # Gauss-Hermite-like grid on a truncated Gaussian.
    grid = np.linspace(-4.0, 4.0, n_quadrature)
    gap_values = gap_mean_m + gap_sigma_m * grid
    weights = np.exp(-0.5 * grid**2)
    keep = gap_values >= 0.0
    gap_values, weights = gap_values[keep], weights[keep]
    weights = weights / weights.sum()
    accumulator = np.zeros(frequency_hz.size, dtype=complex)
    for gap_thickness_m, weight in zip(gap_values, weights):
        accumulator += weight * single_gap_reflection(frequency_hz, material_params, gap_thickness_m, r_front, gap_angle_rad)
    return accumulator


def deembed_gap(reflection_measured, r_front):
    return (reflection_measured - r_front) / (1.0 - r_front * reflection_measured)


# ── Pipeline loader (used by the fitter scripts, not the conditioning report) ─


def load_measured_reflection():
    """Run the thz_core shared-axis reflection pipeline; return per-sample reflection data."""
    from compare_reflection_pathways import run_shared_axis, PIPELINE_CONFIG, short_name
    dataset = run_shared_axis(PIPELINE_CONFIG, eps_infinity=1.0)
    out = {}
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        processing = data_obj.processing_dict
        if processing.get("reflection_r") is None:
            continue
        out[short_name(filename)] = dict(
            frequency_hz=np.asarray(processing["fft_freq"], float),
            mask=np.asarray(processing["transfer_mask"], bool),
            r_meas=np.asarray(processing["reflection_r"], complex),
            r_front=complex(processing["r_reference"]),
            n_naive=np.asarray(processing["n"], float),
            k_naive=np.asarray(processing["k"], float),
        )
    return out


# ── Conditioning machinery ──────────────────────────────────────────────────

# Nominal synthetic ground truth (a plausible thick conductive CNT + rough ~17 um gap).
NOMINAL_PARAMS = {
    "eps_inf": 4.0,
    "plasma_omega": 2.0 * np.pi * 8e12,      # 8 THz plasma frequency (metallic)
    "scattering_time_s": 30e-15,             # 30 fs
    "persistence_c": -0.5,                   # Drude-Smith back-scattering
    "gap_mean_m": 17e-6,
    "gap_sigma_m": 8e-6,
}
PARAM_ORDER = ["eps_inf", "plasma_omega", "scattering_time_s", "persistence_c", "gap_mean_m", "gap_sigma_m"]


def _split_params(vector):
    material = dict(zip(["eps_inf", "plasma_omega", "scattering_time_s", "persistence_c"], vector[:4]))
    return material, vector[4], vector[5]


def model_reflection(frequency_hz, param_vector, r_front, gap_angle_rad):
    material, gap_mean_m, gap_sigma_m = _split_params(param_vector)
    return rough_gap_reflection(frequency_hz, material, gap_mean_m, gap_sigma_m, r_front, gap_angle_rad)


def residual_jacobian(frequency_hz, param_vector, r_front, gap_angle_rad, rel_step=1e-4):
    """Numerical Jacobian d[Re,Im r_meas]/d[param], stacked over the frequency band."""
    base = model_reflection(frequency_hz, param_vector, r_front, gap_angle_rad)
    base_stack = np.concatenate([base.real, base.imag])
    columns = []
    for j, value in enumerate(param_vector):
        step = rel_step * (abs(value) if value != 0 else 1.0)
        perturbed = np.array(param_vector, dtype=float); perturbed[j] += step
        model = model_reflection(frequency_hz, perturbed, r_front, gap_angle_rad)
        columns.append((np.concatenate([model.real, model.imag]) - base_stack) / step)
    return np.column_stack(columns)


def conditioning_report(frequency_hz, param_vectors, r_front, gap_angle_rad, noise_amplitude=2e-3):
    """Fisher analysis over one or more measurements that SHARE the material params.

    param_vectors: list of full 6-vectors (material identical across them; gaps may differ).
    Returns dict with per-parameter relative sigma, the gap<->material correlation, and the
    SVD condition number / poorly-determined directions of the (shared-material) design.
    """
    # Build the joint design matrix: shared material columns (0..3) + per-measurement gap
    # columns (4,5).  Total params = 4 + 2*N.
    n_meas = len(param_vectors)
    n_freq2 = 2 * frequency_hz.size
    total_params = 4 + 2 * n_meas
    design = np.zeros((n_meas * n_freq2, total_params))
    for i, vector in enumerate(param_vectors):
        jac = residual_jacobian(frequency_hz, vector, r_front, gap_angle_rad)
        rows = slice(i * n_freq2, (i + 1) * n_freq2)
        design[rows, 0:4] = jac[:, 0:4]            # shared material
        design[rows, 4 + 2 * i] = jac[:, 4]        # this measurement's gap_mean
        design[rows, 4 + 2 * i + 1] = jac[:, 5]    # this measurement's gap_sigma
    # Normalise columns by the nominal parameter scale so sigmas are RELATIVE.
    scales = np.ones(total_params)
    scales[0:4] = [abs(v) if v else 1.0 for v in param_vectors[0][:4]]
    for i, vector in enumerate(param_vectors):
        scales[4 + 2 * i] = abs(vector[4]) or 1.0
        scales[4 + 2 * i + 1] = abs(vector[5]) or 1.0
    design_scaled = design * scales[None, :]

    fisher = design_scaled.T @ design_scaled / noise_amplitude**2
    try:
        covariance = np.linalg.inv(fisher)
        rel_sigma = np.sqrt(np.clip(np.diag(covariance), 0, None))
    except np.linalg.LinAlgError:
        covariance = None
        rel_sigma = np.full(total_params, np.inf)
    singular_values = np.linalg.svd(design_scaled, compute_uv=False)
    condition_number = singular_values[0] / singular_values[-1] if singular_values[-1] > 0 else np.inf

    # gap_mean <-> material correlations for the first measurement.
    correlations = {}
    if covariance is not None:
        def corr(a, b):
            denom = np.sqrt(covariance[a, a] * covariance[b, b])
            return covariance[a, b] / denom if denom > 0 else np.nan
        for mi, mname in enumerate(["eps_inf", "plasma_omega", "scattering_time_s", "persistence_c"]):
            correlations[mname] = corr(mi, 4)   # vs first gap_mean
    return dict(rel_sigma=rel_sigma, condition_number=condition_number,
                singular_values=singular_values, correlations=correlations,
                total_params=total_params, n_meas=n_meas)


def kk_premise_check(frequency_hz, material_params, gap_mean_m, r_front, gap_angle_rad, sigma_list):
    """How far does |x| drift from the true |r_back| as roughness sigma grows?

    KK / minimum-phase (arXiv 2412.18662) rebuilds phase from |x| assuming |x|=|r_back|.
    This returns, per sigma, the max relative |x|/|r_back| deviation over the band.
    """
    r_back_true = reflection_air_to_cnt(frequency_hz, material_params, gap_angle_rad)
    out = []
    for sigma in sigma_list:
        r_meas = rough_gap_reflection(frequency_hz, material_params, gap_mean_m, sigma, r_front, gap_angle_rad)
        x = deembed_gap(r_meas, r_front)
        ratio = np.abs(x) / np.abs(r_back_true)
        out.append((sigma, float(np.min(ratio)), float(np.max(np.abs(ratio - 1.0)))))
    return out


def main():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 220)
    _, gap_angle_rad = internal_angles()
    r_front = reflection_sio2_to_air(internal_angles()[0])
    nominal_vector = np.array([NOMINAL_PARAMS[k] for k in PARAM_ORDER], dtype=float)

    print("=== CONDITIONING REPORT (synthetic Drude-Smith CNT + rough gap) ===")
    print(f"r_front (SiO2->air) = {r_front:.3f}; gap angle {np.rad2deg(gap_angle_rad):.1f} deg")
    material0, gap_mean0, gap_sigma0 = _split_params(nominal_vector)
    r_back0 = reflection_air_to_cnt(frequency_hz, material0, gap_angle_rad)
    n0 = cnt_index_hat(frequency_hz, material0)
    print(f"planted CNT: n {np.real(n0).mean():.2f}, k {(-np.imag(n0)).mean():.2f}, "
          f"|r_back| {np.abs(r_back0).mean():.2f}  (sensitivity proxy)")
    print(f"planted gap: d_mean {gap_mean0*1e6:.1f} um, sigma_d {gap_sigma0*1e6:.1f} um")

    # 1. Single measurement.
    single = conditioning_report(frequency_hz, [nominal_vector], r_front, gap_angle_rad)
    labels = ["eps_inf", "omega_p", "tau", "c", "d_mean", "sigma_d"]
    print("\n[1] SINGLE measurement — relative 1-sigma uncertainty per parameter "
          "(noise 2e-3 of |r|):")
    for lab, rs in zip(labels, single["rel_sigma"]):
        flag = "  <-- poorly determined" if rs > 0.5 else ""
        print(f"     {lab:9s} : {rs*100:8.1f} %{flag}")
    print(f"     condition number (design) = {single['condition_number']:.1e}")
    print("     gap_mean <-> material correlations:")
    for mname, cval in single["correlations"].items():
        print(f"        d_mean ~ {mname:18s}: r = {cval:+.2f}")

    # 2. Pressure series: 4 measurements, shared material, different gaps.
    series_vectors = []
    for d_um, s_um in [(25, 12), (17, 8), (11, 5), (6, 3)]:
        v = nominal_vector.copy(); v[4] = d_um * 1e-6; v[5] = s_um * 1e-6
        series_vectors.append(v)
    series = conditioning_report(frequency_hz, series_vectors, r_front, gap_angle_rad)
    print(f"\n[2] PRESSURE SERIES ({series['n_meas']} runs, shared material) — material "
          "relative 1-sigma:")
    for mi, lab in enumerate(["eps_inf", "omega_p", "tau", "c"]):
        rs_single = single["rel_sigma"][mi]
        rs_series = series["rel_sigma"][mi]
        print(f"     {lab:9s} : single {rs_single*100:7.1f} %  ->  series {rs_series*100:7.1f} %  "
              f"({rs_single/max(rs_series,1e-9):.1f}x better)")
    print(f"     condition number: single {single['condition_number']:.1e} -> "
          f"series {series['condition_number']:.1e}")

    # 3. KK premise check.
    print("\n[3] KK / minimum-phase premise — |x| vs true |r_back| as roughness grows:")
    for sigma, min_ratio, max_dev in kk_premise_check(
            frequency_hz, material0, gap_mean0, r_front, gap_angle_rad,
            [0e-6, 3e-6, 6e-6, 10e-6, 15e-6]):
        verdict = "OK" if max_dev < 0.05 else ("marginal" if max_dev < 0.15 else "BROKEN")
        print(f"     sigma_d {sigma*1e6:5.1f} um : |x|/|r_back| in band drops to {min_ratio:.2f}, "
              f"max dev {max_dev*100:4.1f}%  -> KK premise {verdict}")

    _try_plot(frequency_hz, nominal_vector, series_vectors, r_front, gap_angle_rad)
    print("\nInterpretation guidance printed; see explorations/air_gap_conditioning.png.")


def _try_plot(frequency_hz, nominal_vector, series_vectors, r_front, gap_angle_rad):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exception:  # pragma: no cover
        print(f"(plot skipped: {exception})")
        return
    import os
    f_thz = frequency_hz * 1e-12
    material0, gap_mean0, gap_sigma0 = _split_params(nominal_vector)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")

    ax = axes[0]
    r_back = reflection_air_to_cnt(frequency_hz, material0, gap_angle_rad)
    for s_um in [0, 6, 12]:
        r_meas = rough_gap_reflection(frequency_hz, material0, gap_mean0, s_um * 1e-6, r_front, gap_angle_rad)
        ax.plot(f_thz, np.abs(deembed_gap(r_meas, r_front)) / np.abs(r_back), label=f"sigma_d={s_um} um")
    ax.axhline(1.0, color="k", ls=":"); ax.set_ylim(0, 1.2)
    ax.set_title("KK premise: |x|/|r_back| (roughness corrupts magnitude)")
    ax.set_xlabel("THz"); ax.set_ylabel("|x|/|r_back|"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for v in series_vectors:
        _, gm, gs = _split_params(v)
        r_meas = rough_gap_reflection(frequency_hz, material0, gm, gs, r_front, gap_angle_rad)
        ax.plot(f_thz, np.abs(r_meas), label=f"d={gm*1e6:.0f},s={gs*1e6:.0f} um")
    ax.set_title("Pressure series: |r_meas| (shared material, varying gap)")
    ax.set_xlabel("THz"); ax.set_ylabel("|r_meas|"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    jac = residual_jacobian(frequency_hz, nominal_vector, r_front, gap_angle_rad)
    labels = ["eps_inf", "omega_p", "tau", "c", "d_mean", "sigma_d"]
    half = frequency_hz.size
    for j, lab in enumerate(labels):
        sens = np.sqrt(jac[:half, j]**2 + jac[half:, j]**2)
        sens = sens / (np.max(sens) + 1e-30)
        ax.plot(f_thz, sens, label=lab)
    ax.set_title("Normalised |dr/dparam| vs frequency (where each param speaks)")
    ax.set_xlabel("THz"); ax.set_ylabel("norm sensitivity"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "air_gap_conditioning.png")
    figure.savefig(out, dpi=130)
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
