"""Regression test: the reflection inversion must not spike to infinity near r = -1.

A highly reflective sample (|H| ~ 1, e.g. conductive CNT referenced to gold) sits on the
r -> -1 singularity of the closed-form inversion q = N1 cosθ (1-r)/(1+r); n spikes toward
infinity wherever the measured reflection passes through r = -1. The configurable
``min_one_plus_r`` floor masks those ill-conditioned bins (NaN) instead of emitting a spike.
These pin: default (1e-12) preserves legacy behaviour; a finite floor caps/masks the blow-up
and reports the count.

Run with:
    .venv/Scripts/python.exe tests/test_reflection_singularity_guard.py
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import thz_core.thz_core as core

FREQ = np.linspace(0.3e12, 2.5e12, 300)
MASK = np.ones(FREQ.size, dtype=bool)
GAP_ANGLE = np.deg2rad(45.0)


def _near_mirror_reflection():
    """Real r(f) that crosses -1 periodically (the conductive-sample-vs-gold case).

    The user's H is ~real and ~1 (r = -H ~ real near -1), so r approaches -1 along the
    REAL axis -> q = cosθ(1-r)/(1+r) is real and large -> the blow-up lands in n (not k).
    r = -1 + 0.3 cos(2π f τ), τ = 1 ps: |1+r| sweeps 0 -> 0.3, crossing -1 ~every 0.5 THz.
    """
    tau = 1e-12
    return (-1.0 + 0.3 * np.cos(2 * np.pi * FREQ * tau)).astype(complex)


def test_default_floor_spikes_toward_infinity():
    """With the legacy 1e-12 floor, n blows up near the r=-1 crossings."""
    r = _near_mirror_reflection()
    n, k, metrics = core.invert_nk_reflection(
        FREQ, r, MASK, {"invert": {"min_one_plus_r": 1e-12}},
        theta_rad=GAP_ANGLE, n_incident=1.0)
    assert np.nanmax(n) > 100.0, "expected a near-singular spike with the tight floor"


def test_floor_masks_the_blowup():
    """A finite floor masks the ill-conditioned bins (NaN) and caps n."""
    r = _near_mirror_reflection()
    n, k, metrics = core.invert_nk_reflection(
        FREQ, r, MASK, {"invert": {"min_one_plus_r": 0.1}},
        theta_rad=GAP_ANGLE, n_incident=1.0)
    finite = np.isfinite(n)
    assert np.any(~finite), "near-singular bins should be masked to NaN"
    assert np.nanmax(n[finite]) < 30.0, "remaining n should be capped, not infinite"
    assert metrics["values"]["n_singular_bins"] > 0


def test_normal_sample_unaffected_by_floor():
    """A normal dielectric (r well away from -1) is untouched by a 0.1 floor."""
    n_true = 3.4
    r = np.asarray(core.fresnel_reflection_s(1.0, n_true - 0j, GAP_ANGLE)) * np.ones(FREQ.size)
    n, k, metrics = core.invert_nk_reflection(
        FREQ, r, MASK, {"invert": {"min_one_plus_r": 0.1}},
        theta_rad=GAP_ANGLE, n_incident=1.0)
    assert np.allclose(n, n_true, atol=1e-6), "floor must not disturb a normal sample"
    assert metrics["values"]["n_singular_bins"] == 0


_TESTS = [
    test_default_floor_spikes_toward_infinity,
    test_floor_masks_the_blowup,
    test_normal_sample_unaffected_by_floor,
]


def main() -> int:
    passed = failed = 0
    for test in _TESTS:
        try:
            test()
            passed += 1
            print(f"PASS  {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed ({len(_TESTS)} total)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
