"""Decisive residual-budget test: Drude-Smith + gap + PER-CHANNEL NUISANCE (flat amplitude
scale + delay) fits of the CNT-21 reflections.

Motivation: the realign repeat shows the data reproduces to ~0.5% in r-space, but every
material+gap model leaves ~10% residual. The self-ref |C| ~ 0.85 (flat) says the front-pulse
coupling differs ~15% between the sample and reference mounts (front spot moves on the
frequency-anisotropic detection cone when the pressed window deforms). If the front/back
spots couple differently, the self-referenced H — and so r_meas — carries a near-flat
spurious complex scale. Model it:

    r_model(omega) = scale * exp(-i omega tau) * FP(n_DS(omega), d)

with scale (real, per channel) and tau (per channel) as nuisance parameters. If the residual
collapses toward the reproducibility floor, the error budget is closed: coupling scale +
timing + gap + Drude-Smith material explains the measurement, and the material extraction is
analytic despite the mount systematics.

Run from repo root:
  PYTHONPATH=".;./explorations/air_gap_cnt_reflection" ./.venv/Scripts/python.exe \
      explorations/air_gap_cnt_reflection/fit_cnt_nuisance_scale.py
"""

from __future__ import annotations

import os

import numpy as np
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thz_core.thz_core as core
from thz_core.thz_core.reflection_gap import modelled_windowed_reflection
from fit_cnt_drude_smith import drude_smith_material_model, load_channel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "cnt21_channel_results")
THETA_GAP = np.deg2rad(45.0)

# parameter layout: [eps_inf, plasma_thz, damping_thz, c] + [gap_um] + per channel [scale, tau_fs]
MATERIAL_INITIAL = (5.0, 30.0, 10.0, -0.7)
MATERIAL_LOWER = (1.0, 0.5, 0.2, -1.0)
MATERIAL_UPPER = (60.0, 300.0, 100.0, 0.0)
GAP_BOUNDS_UM = (0.0, 30.0)
SCALE_BOUNDS = (0.5, 2.0)
TAU_BOUNDS_FS = (-200.0, 200.0)

AXES = {
    "n_parallel": ("0-0", "90-90"),
    "n_perp": ("0-90", "90-0"),
}


def fit_with_nuisance(freq, mask, channels, fixed_gap_um=None, use_nuisance=True):
    """channels: list of dicts (r, r_front, polarization, name). Shared material + gap;
    per-channel scale/tau nuisance."""
    n_channels = len(channels)
    fit_gap = fixed_gap_um is None

    initial = list(MATERIAL_INITIAL)
    lower = list(MATERIAL_LOWER)
    upper = list(MATERIAL_UPPER)
    if fit_gap:
        initial += [2.0]; lower += [GAP_BOUNDS_UM[0]]; upper += [GAP_BOUNDS_UM[1]]
    if use_nuisance:
        for _ in range(n_channels):
            initial += [1.0, 0.0]
            lower += [SCALE_BOUNDS[0], TAU_BOUNDS_FS[0]]
            upper += [SCALE_BOUNDS[1], TAU_BOUNDS_FS[1]]

    def unpack(params):
        material = params[:4]
        cursor = 4
        gap_um = params[cursor] if fit_gap else float(fixed_gap_um)
        cursor += 1 if fit_gap else 0
        nuisance = []
        for index in range(n_channels):
            if use_nuisance:
                nuisance.append((params[cursor], params[cursor + 1]))
                cursor += 2
            else:
                nuisance.append((1.0, 0.0))
        return material, gap_um, nuisance

    def residuals(params):
        material, gap_um, nuisance = unpack(params)
        n_sample = drude_smith_material_model(freq, material)
        parts = []
        for channel, (scale, tau_fs) in zip(channels, nuisance):
            model = modelled_windowed_reflection(
                freq, n_sample, channel["r_front"], gap_um * 1e-6, THETA_GAP,
                channel["polarization"])
            model = scale * np.exp(-1j * 2 * np.pi * freq * tau_fs * 1e-15) * model
            difference = (model - channel["r"])[mask]
            parts.extend([difference.real, difference.imag])
        return np.concatenate(parts)

    solution = least_squares(residuals, initial, bounds=(lower, upper),
                             xtol=1e-12, ftol=1e-12, max_nfev=20000)
    material, gap_um, nuisance = unpack(solution.x)
    rms = float(np.sqrt(np.mean(solution.fun**2)))
    n_sample = drude_smith_material_model(freq, material)
    return dict(material=material, gap_um=gap_um, nuisance=nuisance, rms=rms,
                n=n_sample.real, k=np.abs(n_sample.imag), success=solution.success)


def describe(label, fit, channel_names):
    eps_inf, plasma, damping, c = fit["material"]
    nuisance_text = "  ".join(
        f"{name}: scale={scale:.3f} tau={tau:+.1f}fs"
        for name, (scale, tau) in zip(channel_names, fit["nuisance"]))
    print(f"    {label:26s}: eps_inf={eps_inf:6.2f} plasma={plasma:7.2f} damping={damping:6.2f} "
          f"c={c:+.3f} gap={fit['gap_um']:5.2f}um rms={fit['rms']:.4f}")
    print(f"    {'':26s}  {nuisance_text}")


def main():
    channels = {name: load_channel(name) for name in ("0-0", "0-90", "90-0", "90-90")}

    figure, axes_grid = plt.subplots(2, 3, figsize=(16, 8.4), layout="constrained")
    figure.suptitle("Drude-Smith + gap + per-channel nuisance (scale, tau): residual budget test")

    for row, (axis_label, (s_channel, p_channel)) in enumerate(AXES.items()):
        s_data = channels[s_channel]; p_data = channels[p_channel]
        freq = s_data["freq"]
        joint_mask = s_data["mask"] & p_data["mask"]
        channel_list = [dict(name=s_channel, **s_data), dict(name=p_channel, **p_data)]
        names = [s_channel, p_channel]

        print(f"\n===== axis {axis_label} (s={s_channel}, p={p_channel}) =====")
        fits = {}
        fits["no nuisance"] = fit_with_nuisance(freq, joint_mask, channel_list,
                                                use_nuisance=False)
        describe("joint, NO nuisance", fits["no nuisance"], names)
        fits["nuisance"] = fit_with_nuisance(freq, joint_mask, channel_list,
                                             use_nuisance=True)
        describe("joint, nuisance", fits["nuisance"], names)
        fits["nuisance gap2.9"] = fit_with_nuisance(freq, joint_mask, channel_list,
                                                    fixed_gap_um=2.9, use_nuisance=True)
        describe("joint, nuisance, gap 2.9", fits["nuisance gap2.9"], names)
        # single-channel nuisance fits (does each channel alone close its budget?)
        for channel_dict in channel_list:
            single = fit_with_nuisance(freq, channel_dict["mask"], [channel_dict],
                                       use_nuisance=True)
            describe(f"single {channel_dict['name']}, nuisance", single, [channel_dict["name"]])
            fits[f"single {channel_dict['name']}"] = single

        freq_thz = freq * 1e-12
        band = joint_mask & (freq_thz >= 0.3) & (freq_thz <= 2.4)
        ax_n, ax_k, ax_s = axes_grid[row]
        for key, color in (("no nuisance", "0.6"), ("nuisance", "C0"),
                           ("nuisance gap2.9", "C2"),
                           (f"single {p_channel}", "C3")):
            fit = fits[key]
            eps, sigma, _ = core.derive_eps_sigma(freq, fit["n"], fit["k"], {"derive": {}})
            style = dict(color=color, lw=1.5, ls="--" if key == "no nuisance" else "-")
            ax_n.plot(freq_thz[band], fit["n"][band],
                      label=f"{key} (rms {fit['rms']:.3f}, gap {fit['gap_um']:.1f}um)", **style)
            ax_k.plot(freq_thz[band], fit["k"][band], label=key, **style)
            ax_s.plot(freq_thz[band], sigma[band].real * 1e-2, label=key, **style)
        ax_n.set_title(f"{axis_label}: n", fontsize=10)
        ax_k.set_title(f"{axis_label}: k", fontsize=10)
        ax_s.set_title(f"{axis_label}: sigma1 [S/cm]", fontsize=10)
        for ax in (ax_n, ax_k, ax_s):
            ax.set_xlabel("THz"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    out = os.path.join(RESULTS_DIR, "fit_cnt_nuisance_scale.png")
    figure.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
