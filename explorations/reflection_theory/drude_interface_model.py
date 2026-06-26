"""Reflection physics tutorial-model: a Drude conductor seen through different media.

Builds intuition for WHY coupling a highly reflective (conductive) sample through a
high-index medium (SiO2, Si) pulls its reflection off the r = -1 singularity and reopens
the low-frequency window — the practical reason behind the window geometry.

The chain modelled:  Drude ε(ω)  ->  n̂ = √ε  ->  Fresnel r(ω; N1)  ->  inversion r -> n̂.
Everything is self-contained (numpy only) so it reads as a learning artifact.

Run:  PYTHONPATH=. .venv/Scripts/python.exe explorations/reflection_theory/drude_interface_model.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Physical constants ──────────────────────────────────────────────────────
SPEED_OF_LIGHT = 299_792_458.0
VACUUM_PERMITTIVITY = 8.8541878128e-12

# ── Knobs ───────────────────────────────────────────────────────────────────
THETA_DEG = 45.0                       # s-pol incidence angle (the rig geometry)
EPS_INF = 11.0                         # high-frequency permittivity (CNT-ish)
PLASMA_FREQUENCY_THZ = 25.0            # Drude plasma frequency (sets how metallic)
SCATTERING_TIME_FS = 40.0             # Drude scattering time (carrier damping)
NOISE_AMPLITUDE = 3e-3                 # fixed additive noise on the measured r (|r|~1)
INCIDENT_MEDIA = {"air (n=1)": 1.0, "SiO2 (n=1.95)": 1.95, "Si (n=3.4)": 3.4}
FREQ_THZ = np.linspace(0.1, 3.0, 600)


# ── Material: Drude free-carrier permittivity ───────────────────────────────
def drude_permittivity(frequency_hz, eps_inf, plasma_omega, scattering_time_s):
    """ε(ω) = ε∞ − ω_p² / (ω² + iω/τ).  Free electrons, no restoring force.

    The conductivity σ(ω) = ε0 ω_p² τ / (1 − iωτ) has its peak at ω = 0 (the Drude
    peak): a free-carrier metal is MOST reflective at low frequency. That is exactly
    the regime where the reflection inversion is hardest.
    """
    omega = 2.0 * np.pi * frequency_hz
    return eps_inf - plasma_omega**2 / (omega**2 + 1j * omega / scattering_time_s)


def conductivity(frequency_hz, plasma_omega, scattering_time_s):
    omega = 2.0 * np.pi * frequency_hz
    return VACUUM_PERMITTIVITY * plasma_omega**2 * scattering_time_s / (1.0 - 1j * omega * scattering_time_s)


def index_from_eps(eps):
    """n̂ = n − i k (k ≥ 0 absorption, exp(−iωt) convention)."""
    return np.conj(np.sqrt(eps))


# ── Interface: Fresnel reflection (s-pol) and its closed-form inverse ────────
def fresnel_r_s(n_incident, n_sample, theta_incident_rad):
    """r_s for a wave in medium n_incident hitting a semi-infinite n_sample."""
    cos_i = np.cos(theta_incident_rad)
    root = np.sqrt(n_sample**2 - (n_incident * np.sin(theta_incident_rad)) ** 2)
    return (n_incident * cos_i - root) / (n_incident * cos_i + root)


def invert_r_to_index(r, n_incident, theta_incident_rad):
    """Closed-form inverse of fresnel_r_s -> n̂.  Singular as r -> -1 (perfect mirror)."""
    q = n_incident * np.cos(theta_incident_rad) * (1.0 - r) / (1.0 + r)
    n_hat = np.sqrt(q**2 + (n_incident * np.sin(theta_incident_rad)) ** 2)
    return np.where(n_hat.real < 0, -n_hat, n_hat)


def main():
    theta = np.deg2rad(THETA_DEG)
    freq_hz = FREQ_THZ * 1e12
    plasma_omega = 2.0 * np.pi * PLASMA_FREQUENCY_THZ * 1e12
    tau = SCATTERING_TIME_FS * 1e-15

    eps = drude_permittivity(freq_hz, EPS_INF, plasma_omega, tau)
    n_hat = index_from_eps(eps)                       # n − ik
    n_true, k_true = n_hat.real, -n_hat.imag
    sigma = conductivity(freq_hz, plasma_omega, tau)

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), layout="constrained")
    fig.suptitle(
        f"A Drude conductor through different interfaces  (ω_p={PLASMA_FREQUENCY_THZ} THz, "
        f"τ={SCATTERING_TIME_FS} fs, θ={THETA_DEG}°)", fontsize=13)

    # (0,0) material n,k
    ax = axes[0][0]
    ax.plot(FREQ_THZ, n_true, label="n"); ax.plot(FREQ_THZ, k_true, label="k")
    ax.set_title("Material optical constants (Drude)"); ax.set_xlabel("THz"); ax.legend(); ax.grid(alpha=0.3)

    # (0,1) conductivity (the Drude peak at 0)
    ax = axes[0][1]
    ax.plot(FREQ_THZ, np.real(sigma), label="Re σ"); ax.plot(FREQ_THZ, np.imag(sigma), label="Im σ")
    ax.set_title("Conductivity σ(ω) — Drude peak at ω=0"); ax.set_xlabel("THz"); ax.set_ylabel("S/m")
    ax.legend(); ax.grid(alpha=0.3)

    # (0,2) |r| vs frequency per incident medium
    ax = axes[0][2]
    r_by_medium = {}
    for label, n1 in INCIDENT_MEDIA.items():
        r = fresnel_r_s(n1, n_hat, theta)
        r_by_medium[label] = r
        ax.plot(FREQ_THZ, np.abs(r), label=label)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_title("|r|: higher-index incidence => more contrast (lower |r|)")
    ax.set_xlabel("THz"); ax.set_ylabel("|r|"); ax.set_ylim(0.5, 1.02); ax.legend(); ax.grid(alpha=0.3)

    # (1,0) Argand: r(ω) trajectory vs the r=-1 singularity (zoomed near -1)
    ax = axes[1][0]
    circle = np.exp(1j * np.linspace(0, 2 * np.pi, 400))
    ax.plot(circle.real, circle.imag, "0.7", lw=0.8)
    for label, r in r_by_medium.items():
        ax.plot(r.real, r.imag, lw=1.6, label=label)
        ax.plot(r.real[0], r.imag[0], "o", ms=4)   # mark the low-frequency end
    ax.plot(-1, 0, "kx", ms=11, mew=2, label="r = -1 (singular)")
    ax.set_title("r(ω) near the singularity (dot = 0.1 THz end)")
    ax.set_xlabel("Re r"); ax.set_ylabel("Im r"); ax.set_aspect("equal")
    ax.set_xlim(-1.05, -0.55); ax.set_ylim(-0.45, 0.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,1) conditioning: |1+r| (distance to the singularity)
    ax = axes[1][1]
    for label, r in r_by_medium.items():
        ax.semilogy(FREQ_THZ, np.abs(1.0 + r), label=label)
    ax.set_title("|1+r| — proximity to the singularity (bigger = better)")
    ax.set_xlabel("THz"); ax.set_ylabel("|1+r|"); ax.legend(); ax.grid(alpha=0.3, which="both")

    # (1,2) the punchline: error amplification. Monte-Carlo the SAME measurement noise
    # through the inversion and report the fractional uncertainty in n per medium.
    ax = axes[1][2]
    n_trials = 300
    for label, n1 in INCIDENT_MEDIA.items():
        r = r_by_medium[label]
        recovered = np.empty((n_trials, r.size))
        for t in range(n_trials):
            noise = NOISE_AMPLITUDE * (rng.standard_normal(r.shape) + 1j * rng.standard_normal(r.shape))
            recovered[t] = invert_r_to_index(r + noise, n1, theta).real
        fractional_uncertainty = np.std(recovered, axis=0) / np.abs(n_true)
        ax.semilogy(FREQ_THZ, fractional_uncertainty, lw=1.6, label=label)
    ax.set_title(f"Error amplification: σ(n)/n from the SAME r-noise ({NOISE_AMPLITUDE:.0e})")
    ax.set_xlabel("THz"); ax.set_ylabel("fractional uncertainty in n")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    out = os.path.join(HERE, "drude_interface_model.png")
    fig.savefig(out, dpi=120)
    print(f"Saved figure: {out}")

    # printed summary: low-frequency conditioning gain
    print(f"\n|1+r| at 0.3 THz (distance to singularity; bigger = recoverable):")
    i = int(np.argmin(np.abs(FREQ_THZ - 0.3)))
    for label, r in r_by_medium.items():
        print(f"  {label:16s}: |r|={abs(r[i]):.3f}  |1+r|={abs(1+r[i]):.3f}")


if __name__ == "__main__":
    main()
