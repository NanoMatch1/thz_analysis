"""Unit tests for detrend_transfer_phase — experimental linear-phase (group-delay) removal.

It fits a line to arg(H) over a trusted band and multiplies H by exp(-i·slope·f) so the
linear-in-frequency phase is flattened. Magnitude is untouched. These pin: a planted delay
is recovered and removed (phase slope -> ~0), |H| unchanged, and the original H is preserved.

Run with:
    .venv/Scripts/python.exe tests/test_detrend_transfer_phase.py
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.adapters.thz_adapter import detrend_transfer_phase

FREQ = np.linspace(0.1e12, 3.0e12, 400)


class _Obj:
    def __init__(self, processing):
        self.processing_dict = processing


class _DataAccess:
    def __init__(self, items, refs):
        self._items = items
        self._refs = set(refs)

    def items(self):
        return list(self._items.items())

    def is_reference(self, name):
        return name in self._refs


class _DS:
    def __init__(self, items, refs, config=None):
        self.data = _DataAccess(items, refs)
        self.config = config or {}


def _sample_with_delay(delay_s, magnitude=0.9):
    H = magnitude * np.exp(-1j * 2 * np.pi * FREQ * delay_s)   # |H| flat, linear phase
    return _Obj({"fft_freq": FREQ, "transfer_H": H,
                 "transfer_mask": np.ones(FREQ.size, bool)})


def _slope_rad_per_hz(H, band=(0.3e12, 2.0e12)):
    m = (FREQ >= band[0]) & (FREQ <= band[1])
    return np.polyfit(FREQ[m], np.unwrap(np.angle(H[m])), 1)[0]


# ---------------------------------------------------------------------------

def test_removes_planted_delay():
    obj = _sample_with_delay(1.0e-12)   # 1 ps group delay
    before_slope = _slope_rad_per_hz(obj.processing_dict["transfer_H"])
    detrend_transfer_phase(_DS({"sample_x.acc": obj, "reference_x.acc": _Obj({})},
                               {"reference_x.acc"}))
    after_slope = _slope_rad_per_hz(obj.processing_dict["transfer_H"])
    assert abs(before_slope) > 1e-13           # there was a real slope
    assert abs(after_slope) < 1e-3 * abs(before_slope)  # flattened to ~0
    assert abs(obj.processing_dict["phase_detrend_metrics"]["delay_seconds"] - 1.0e-12) < 1e-14


def test_magnitude_unchanged():
    obj = _sample_with_delay(0.7e-12, magnitude=0.85)
    before_mag = np.abs(obj.processing_dict["transfer_H"]).copy()
    detrend_transfer_phase(_DS({"sample_x.acc": obj, "reference_x.acc": _Obj({})},
                               {"reference_x.acc"}))
    assert np.allclose(np.abs(obj.processing_dict["transfer_H"]), before_mag)


def test_original_H_preserved():
    obj = _sample_with_delay(1.0e-12)
    original = obj.processing_dict["transfer_H"].copy()
    detrend_transfer_phase(_DS({"sample_x.acc": obj, "reference_x.acc": _Obj({})},
                               {"reference_x.acc"}))
    assert np.allclose(obj.processing_dict["transfer_H_pre_detrend"], original)


_TESTS = [test_removes_planted_delay, test_magnitude_unchanged, test_original_H_preserved]


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
