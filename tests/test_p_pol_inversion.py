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


def test_p_pol_no_grazing_collapse_low_frequency():
    # Regression for the F25 failure (Si/CNT p-pol window geometry). The measured back reflection is
    # small and near-real-positive (r = r_front * H, |r| ~ 0.2) with a phase that sweeps through
    # zero across the band, so Im(r) flips sign. The two exact roots are the physical high index
    # (~3.3) and a spurious twin that degenerates to the grazing value n1 sin(theta). The old
    # per-bin passivity picker collapsed n onto that twin wherever Im(r) made the physical root read
    # as gain (the "hard cut and drop" the user saw at ~0.87 THz); the global branch vote must hold
    # the physical branch across the whole band.
    frequency_hz = np.linspace(0.3e12, 2.5e12, 300)
    n_incident = 1.96
    theta = np.deg2rad(21.15)  # internal angle for a 45 deg external SiO2 window
    phase = np.linspace(0.25, -0.25, frequency_hz.size)  # sweeps Im(r) through zero
    r = 0.24 * np.exp(1j * phase)
    mask = np.ones(frequency_hz.size, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n, k, _ = core.invert_nk_reflection(
            frequency_hz, r, mask, theta_rad=theta, n_incident=n_incident, polarization="p")
    grazing = n_incident * np.sin(theta)
    finite = np.isfinite(n)
    assert np.mean(np.abs(n[finite] - grazing) < 0.02) < 0.02  # not collapsed onto the grazing twin
    assert np.nanmedian(n) > 3.0                                # holds the physical high-index branch
    assert np.all(k[finite] >= -1e-9)                          # still passive


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


def test_p_pol_grazing_override_when_physical_root_reads_as_gain():
    # F30 (MINTS doped-Si p-pol): front-pulse phase referencing (self_phase / time alignment)
    # rotates arg(H) a hair PAST the real-negative reflection point, so Im(r) tips positive and
    # the PHYSICAL high-index root reads as slight gain. The passivity vote alone would then
    # collapse the whole band onto the grazing twin (n -> n1 sin(theta) ~ 0.707 -- the "rails to
    # 0.7" failure). The physicality override must hold the high-index branch instead.
    frequency_hz = np.linspace(0.3e12, 2.5e12, 300)
    n_incident = 1.96
    theta = np.deg2rad(21.15)  # internal angle, 45 deg external SiO2 window
    planted = 3.3 - 0.02j * np.ones_like(frequency_hz)  # high index, near-lossless
    r = np.asarray(fresnel_reflection_p(n_incident, planted, theta), dtype=np.complex128)
    r = r * np.exp(1j * 0.12)  # over-rotate the phase -> Im(r) tips, physical root looks like gain
    mask = np.ones(frequency_hz.size, dtype=bool)
    grazing = n_incident * np.sin(theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n, k, _ = core.invert_nk_reflection(
            frequency_hz, r, mask, theta_rad=theta, n_incident=n_incident, polarization="p")
    finite = np.isfinite(n)
    assert np.nanmedian(n) > 2.5                                  # held the physical high-index branch
    assert np.mean(np.abs(n[finite] - grazing) < 0.05) < 0.05    # did NOT collapse to the grazing twin
