"""Regression tests: dataset.config is the single source of truth for pipeline stages.

These pin the carry-through bug Samuel hit — a config value defined once on the DataSet
(here eps_background) must reach the core layer even when the stage function is called with
NO explicit config; and an explicit config must still override (explicit wins, else dataset).

Run with:
    .venv/Scripts/python.exe tests/test_config_single_source.py
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.adapters import thz_adapter as thz


class _FakeDataObj:
    def __init__(self, n, k, freq):
        self.processing_dict = {"n": n, "k": k, "fft_freq": freq}
        self.data = None
        self.filename = "sample.acc"


class _FakeDataMap:
    def __init__(self, items):
        self._items = items

    def items(self):
        return list(self._items.items())

    def is_reference(self, filename):
        return False


class _FakeDataset:
    """Minimal stand-in for DataSet: just .config and .data (items + is_reference)."""

    def __init__(self, data_obj, config):
        self.data = _FakeDataMap({data_obj.filename: data_obj})
        self.config = config


def _make_dataset(eps_background_in_config):
    freq = np.linspace(0.3e12, 2.5e12, 64)
    n = 3.0 * np.ones_like(freq)
    k = 1.0 * np.ones_like(freq)
    data_obj = _FakeDataObj(n, k, freq)
    config = {"derive": {"eps_background": eps_background_in_config}}
    return _FakeDataset(data_obj, config), data_obj, freq, n, k


def _expected_sigma(freq, n, k, eps_background):
    omega = 2.0 * np.pi * freq
    eps = (n + 1j * k) ** 2
    return -1j * omega * 8.8541878128e-12 * (eps - eps_background)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_eps_background_flows_from_dataset_config():
    """derive_eps_sigma(dataset) with NO explicit config must use dataset.config's value."""
    dataset, data_obj, freq, n, k = _make_dataset(11.7)
    thz.derive_eps_sigma(dataset)  # no config arg -> single source of truth
    sigma = data_obj.processing_dict["sigma"]
    assert np.allclose(sigma, _expected_sigma(freq, n, k, 11.7))
    # and it must NOT be the vacuum default (proves the value actually carried)
    assert not np.allclose(sigma, _expected_sigma(freq, n, k, 1.0))


def test_explicit_config_overrides_dataset_config():
    """An explicit config wins over dataset.config (explicit, else dataset, else default)."""
    dataset, data_obj, freq, n, k = _make_dataset(11.7)
    thz.derive_eps_sigma(dataset, config={"derive": {"eps_background": 1.0}})
    sigma = data_obj.processing_dict["sigma"]
    assert np.allclose(sigma, _expected_sigma(freq, n, k, 1.0))


def test_missing_config_falls_back_to_vacuum_default():
    """No 'derive' key anywhere -> core default eps_background = 1.0, no crash."""
    freq = np.linspace(0.3e12, 2.5e12, 64)
    n = 3.0 * np.ones_like(freq)
    k = 1.0 * np.ones_like(freq)
    data_obj = _FakeDataObj(n, k, freq)
    dataset = _FakeDataset(data_obj, config={})  # empty config
    thz.derive_eps_sigma(dataset)
    sigma = data_obj.processing_dict["sigma"]
    assert np.allclose(sigma, _expected_sigma(freq, n, k, 1.0))


def test_dataset_without_config_attribute_does_not_crash():
    """getattr fallback: a minimal mock with no .config must still work (-> default)."""
    freq = np.linspace(0.3e12, 2.5e12, 64)
    n = 3.0 * np.ones_like(freq)
    k = 1.0 * np.ones_like(freq)
    data_obj = _FakeDataObj(n, k, freq)

    class _NoConfigDataset:
        data = _FakeDataMap({"sample.acc": data_obj})

    thz.derive_eps_sigma(_NoConfigDataset())
    assert np.allclose(data_obj.processing_dict["sigma"], _expected_sigma(freq, n, k, 1.0))


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

_TESTS = [
    test_eps_background_flows_from_dataset_config,
    test_explicit_config_overrides_dataset_config,
    test_missing_config_falls_back_to_vacuum_default,
    test_dataset_without_config_attribute_does_not_crash,
]


def main() -> int:
    passed = 0
    failed = 0
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
