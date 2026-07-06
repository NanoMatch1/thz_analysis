"""Unit tests for thz_adapter.deembed_air_gap_reflection (Route-A pipeline de-embed step).

Plant a known CNT index behind a known single gap, forward-model the SiO2-side reflection,
feed it through the adapter step, and check it recovers the planted n,k. Also pin the
bookkeeping: window-geometry n,k preserved under 'n_window', de-embedded values overwrite 'n'.

Run with:
    .venv/Scripts/python.exe tests/test_deembed_air_gap_reflection.py
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "explorations", "air_gap_cnt_reflection"))

import thz_core.thz_core as core
from dataset_core.adapters.thz_adapter import deembed_air_gap_reflection
from explore_air_gap_deembedding import internal_angles

SIO2_ANGLE_RAD, GAP_ANGLE_RAD = internal_angles()
FREQ_HZ = np.linspace(0.3e12, 2.5e12, 200)
SPEED_OF_LIGHT = 299_792_458.0
R_FRONT = complex(core.fresnel_reflection_s(1.95, 1.0, SIO2_ANGLE_RAD))  # SiO2 -> air


def _planted_back_reflection(n_value, k_value):
    n_hat = n_value - 1j * k_value
    return np.asarray(core.fresnel_reflection_s(1.0, n_hat, GAP_ANGLE_RAD), complex) * np.ones(FREQ_HZ.size)


def _single_gap_forward(r_back, gap_m):
    bounce = np.exp(-1j * (2 * np.pi * FREQ_HZ) / SPEED_OF_LIGHT * 2 * gap_m * np.cos(GAP_ANGLE_RAD))
    return (R_FRONT + r_back * bounce) / (1.0 + R_FRONT * r_back * bounce)


class _FakeObj:
    def __init__(self, processing):
        self.processing_dict = processing


class _FakeDataAccess:
    def __init__(self, items, references):
        self._items = items
        self._references = set(references)

    def items(self):
        return list(self._items.items())

    def is_reference(self, name):
        return name in self._references


class _FakeDataset:
    def __init__(self, items, references, config):
        self.data = _FakeDataAccess(items, references)
        self.config = config


def _dataset_with_gap(n_true, k_true, gap_um, position_um, width_um=0.0):
    r_back = _planted_back_reflection(n_true, k_true)
    r_meas = _single_gap_forward(r_back, gap_um * 1e-6)
    sample = _FakeObj({
        "fft_freq": FREQ_HZ,
        "transfer_mask": np.ones(FREQ_HZ.size, bool),
        "reflection_r": r_meas,
        "r_reference": R_FRONT,
        "n": np.full(FREQ_HZ.size, 99.0),   # bogus window-geometry values to be preserved
        "k": np.full(FREQ_HZ.size, 99.0),
    })
    config = {"geometry": {"theta_external_deg": 45.0},
              "air_gap": {"position_um": position_um, "width_um": width_um}}
    ds = _FakeDataset({"sample_x.acc": sample, "reference_x.acc": _FakeObj({})},
                      references={"reference_x.acc"}, config=config)
    return ds, sample


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_recovers_planted_nk_at_correct_gap():
    """De-embedding at the planted gap recovers the planted n,k (sigma_d = 0)."""
    n_true, k_true, gap_um = 3.0, 1.5, 12.0
    ds, sample = _dataset_with_gap(n_true, k_true, gap_um, position_um=gap_um)
    deembed_air_gap_reflection(ds)
    band = sample.processing_dict["transfer_mask"]
    assert np.allclose(sample.processing_dict["n"][band], n_true, atol=1e-6)
    assert np.allclose(sample.processing_dict["k"][band], k_true, atol=1e-6)


def test_preserves_window_geometry_nk():
    """The pre-de-embed n,k are stashed under n_window/k_window, not lost."""
    ds, sample = _dataset_with_gap(3.0, 1.0, 10.0, position_um=10.0)
    deembed_air_gap_reflection(ds)
    assert np.allclose(sample.processing_dict["n_window"], 99.0)
    assert np.allclose(sample.processing_dict["k_window"], 99.0)
    assert "reflection_r_deembedded" in sample.processing_dict
    assert sample.processing_dict["air_gap_deembed"]["position_um"] == 10.0


def test_wrong_gap_does_not_recover_truth():
    """De-embedding at the WRONG gap must NOT return the planted n (sensitivity check)."""
    n_true = 3.0
    ds, sample = _dataset_with_gap(n_true, 0.5, gap_um=20.0, position_um=20.0 + 8.0)
    deembed_air_gap_reflection(ds)
    band = sample.processing_dict["transfer_mask"]
    assert not np.allclose(sample.processing_dict["n"][band], n_true, atol=0.05)


def test_zero_gap_is_noop():
    """position_um = 0 AND width_um = 0 is a no-op: the window-geometry n,k are left untouched.

    A zero gap is not a gap. The front-interface strip assumes a SiO2 | air | sample stack, so
    applying it to a gap-free (well-contacted) sample re-inverts at the ill-conditioned air
    incidence and collapses n to the grazing floor (~n1 sin theta). With both gap parameters zero
    there is nothing to de-embed, so the step must leave the window inversion in place and not
    stash an 'n_window' (it never touched n).
    """
    ds, sample = _dataset_with_gap(3.0, 1.0, gap_um=0.0, position_um=0.0, width_um=0.0)
    sample.processing_dict["n"] = np.full(FREQ_HZ.size, 3.418)   # a valid window-geometry result
    sample.processing_dict["k"] = np.full(FREQ_HZ.size, 0.2)
    deembed_air_gap_reflection(ds)
    band = sample.processing_dict["transfer_mask"]
    assert np.allclose(sample.processing_dict["n"][band], 3.418)
    assert np.allclose(sample.processing_dict["k"][band], 0.2)
    assert "n_window" not in sample.processing_dict  # no-op did not run the de-embed


_TESTS = [
    test_recovers_planted_nk_at_correct_gap,
    test_preserves_window_geometry_nk,
    test_wrong_gap_does_not_recover_truth,
    test_zero_gap_is_noop,
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
