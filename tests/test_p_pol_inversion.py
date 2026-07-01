"""Round-trip tests for the p-polarisation reflection inversion in thz_core.invert.

The closed form must recover a planted index from its own forward p-Fresnel reflection across
angles/incident media, resolving the p-pol root ambiguity (physical passive root, not its gain
twin). s-pol is checked alongside to confirm it is unchanged.
"""

import warnings

import numpy as np

import thz_core.thz_core as core
from thz_core.thz_core.multilayer import fresnel_reflection_s, fresnel_reflection_p


def _planted_drude(frequency_hz):
    omega = 2.0 * np.pi * frequency_hz
    eps = 4.0 - (2 * np.pi * 6e12) ** 2 / (omega**2 + 1j * 2 * np.pi * 3e12 * omega)
    return np.conj(np.sqrt(eps))  # n - i k, k > 0


def _roundtrip_error(polarization, n_incident, theta_deg, planted):
    frequency_hz = np.linspace(0.3e12, 2.5e12, 300)
    mask = np.ones(frequency_hz.size, dtype=bool)
    theta = np.deg2rad(theta_deg)
    fresnel = fresnel_reflection_p if polarization == "p" else fresnel_reflection_s
    r = fresnel(n_incident, planted, theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n, k, _ = core.invert_nk_reflection(
            frequency_hz, r, mask, theta_rad=theta, n_incident=n_incident,
            polarization=polarization)
    return np.nanmax(np.abs(n - planted.real)), np.nanmax(np.abs(k + planted.imag))


def test_p_pol_roundtrip_conductor_various_geometry():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 300)
    planted = _planted_drude(frequency_hz)
    for n_incident, theta_deg in [(1.0, 0.0), (1.0, 45.0), (1.0, 63.0), (1.95, 21.3)]:
        dn, dk = _roundtrip_error("p", n_incident, theta_deg, planted)
        assert dn < 1e-9 and dk < 1e-9, (n_incident, theta_deg, dn, dk)


def test_p_pol_roundtrip_low_loss_dielectric():
    # Near-lossless: both roots are nearly real, so the residual (not the gain penalty) must
    # break the tie correctly.
    frequency_hz = np.linspace(0.3e12, 2.5e12, 300)
    planted = 3.4 - 0.001j * np.ones_like(frequency_hz)
    dn, dk = _roundtrip_error("p", 1.0, 45.0, planted)
    assert dn < 1e-6 and dk < 1e-6


def test_s_pol_still_exact():
    frequency_hz = np.linspace(0.3e12, 2.5e12, 300)
    planted = _planted_drude(frequency_hz)
    for n_incident, theta_deg in [(1.0, 0.0), (1.0, 45.0), (1.95, 21.3)]:
        dn, dk = _roundtrip_error("s", n_incident, theta_deg, planted)
        assert dn < 1e-9 and dk < 1e-9


def test_p_pol_recovers_passive_not_gain_twin():
    # The spurious root has Im(N2) > 0 (gain); the recovered k must be >= 0 everywhere.
    frequency_hz = np.linspace(0.3e12, 2.5e12, 300)
    planted = _planted_drude(frequency_hz)
    mask = np.ones(frequency_hz.size, dtype=bool)
    theta = np.deg2rad(45.0)
    r = fresnel_reflection_p(1.0, planted, theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, k, _ = core.invert_nk_reflection(
            frequency_hz, r, mask, theta_rad=theta, n_incident=1.0, polarization="p")
    assert np.all(k[np.isfinite(k)] >= -1e-9)
