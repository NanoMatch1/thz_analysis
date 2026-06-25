"""Unit tests for thz_adapter.selfref_quality — the front-pulse alignment gate.

The self-reference correction C = Y1_r/Y1_s should be flat (|C| ~ 1) when the front
reflection spots are well aligned. selfref_quality flags acquisitions whose |C| is
structured (front spot drifted) WITHOUT modifying any data. These tests pin: flat C
passes, structured C is flagged, and a non-self-referenced run returns nothing.

Run with:
    .venv/Scripts/python.exe tests/test_selfref_quality.py
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.adapters.thz_adapter import selfref_quality, _HZ_TO_THZ


class _FakeHolder:
    def __init__(self, processing_dict):
        self.processing_dict = processing_dict


class _FakeDataAccess:
    """Mimics dataset.data: .items() + .is_reference(name)."""

    def __init__(self, items, references):
        self._items = items
        self._references = set(references)

    def items(self):
        return list(self._items.items())

    def is_reference(self, name):
        return name in self._references


class _FakeDataset:
    def __init__(self, items, references):
        self.data = _FakeDataAccess(items, references)


_FREQ_HZ = np.linspace(0.05e12, 3.0e12, 400)


def _make_sample(correction, mask=None):
    processing = {"fft_freq": _FREQ_HZ, "selfref_correction": correction}
    if mask is not None:
        processing["transfer_mask"] = mask
    return _FakeHolder(processing)


def _dataset_with(correction, mask=None, config=None):
    sample = _make_sample(correction, mask)
    reference = _FakeHolder({})  # references are skipped before processing_dict is read
    ds = _FakeDataset({"sample_x.acc": sample, "reference_x.acc": reference},
                      references={"reference_x.acc"})
    ds.config = config or {}
    return ds


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_flat_correction_passes():
    """|C| ~ 1 with tiny ripple -> not flagged (good alignment)."""
    rng = np.random.default_rng(0)
    flat = (1.0 + 0.01 * rng.standard_normal(_FREQ_HZ.size)).astype(complex)
    results = selfref_quality(_dataset_with(flat))
    assert "sample_x.acc" in results
    assert results["sample_x.acc"]["flagged"] is False
    assert results["sample_x.acc"]["std"] < 0.1


def test_structured_correction_is_flagged():
    """A |C| with a strong low-frequency rise -> flagged (front spot drifted)."""
    fthz = _FREQ_HZ * _HZ_TO_THZ
    structured = (1.0 + 2.5 * np.exp(-((fthz - 0.3) ** 2) / (2 * 0.15 ** 2))).astype(complex)
    results = selfref_quality(_dataset_with(structured))
    assert results["sample_x.acc"]["flagged"] is True
    assert results["sample_x.acc"]["std"] > 0.1


def test_median_dev_threshold_flags_offset_unity():
    """A flat but offset |C| (e.g. 1.3) trips the median-deviation threshold."""
    offset = np.full(_FREQ_HZ.size, 1.3, dtype=complex)
    results = selfref_quality(_dataset_with(offset))
    assert results["sample_x.acc"]["flagged"] is True
    assert results["sample_x.acc"]["median_dev"] > 0.15


def test_non_selfref_run_returns_empty():
    """No selfref_correction present -> nothing to assess."""
    sample = _FakeHolder({"fft_freq": _FREQ_HZ})  # no 'selfref_correction'
    reference = _FakeHolder({})
    ds = _FakeDataset({"sample_x.acc": sample, "reference_x.acc": reference},
                      references={"reference_x.acc"})
    ds.config = {}
    results = selfref_quality(ds)
    assert results == {}


def test_snr_mask_restricts_band():
    """When use_snr_mask, only trusted bins count -> masking out the spike un-flags it."""
    fthz = _FREQ_HZ * _HZ_TO_THZ
    structured = (1.0 + 3.0 * np.exp(-((fthz - 0.3) ** 2) / (2 * 0.1 ** 2))).astype(complex)
    # mask excludes the spike region (< 0.6 THz), keeping the flat tail only
    mask = fthz >= 0.6
    results = selfref_quality(_dataset_with(structured, mask=mask))
    assert results["sample_x.acc"]["flagged"] is False


def test_threshold_is_configurable():
    """A stricter std_threshold flags a mildly structured C that default would pass."""
    rng = np.random.default_rng(1)
    mild = (1.0 + 0.05 * rng.standard_normal(_FREQ_HZ.size)).astype(complex)
    default_pass = selfref_quality(_dataset_with(mild))
    assert default_pass["sample_x.acc"]["flagged"] is False
    strict = selfref_quality(_dataset_with(
        mild, config={"selfref_quality": {"std_threshold": 0.02}}))
    assert strict["sample_x.acc"]["flagged"] is True


_TESTS = [
    test_flat_correction_passes,
    test_structured_correction_is_flagged,
    test_median_dev_threshold_flags_offset_unity,
    test_non_selfref_run_returns_empty,
    test_snr_mask_restricts_band,
    test_threshold_is_configurable,
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
