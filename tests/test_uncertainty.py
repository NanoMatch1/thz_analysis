"""Tests for Monte-Carlo uncertainty propagation."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_core.adapters import uncertainty


class _FakeData(dict):
    def is_reference(self, name):
        return name.startswith("reference")


class _FakeObj:
    def __init__(self, pd):
        self.processing_dict = pd


class _FakeDataset:
    def __init__(self, samples):
        self.data = _FakeData(samples)


def _reinvert_identity(dataset):
    """Deterministic 'inversion': n = |H|, k = arg(H). So sigma_n<-sigma_|H|, sigma_k<-sigma_phase."""
    for name, obj in dataset.data.items():
        if dataset.data.is_reference(name):
            continue
        H = obj.processing_dict["transfer_H"]
        obj.processing_dict["n"] = np.abs(H)
        obj.processing_dict["k"] = np.angle(H)


class TestPropagateUncertainty(unittest.TestCase):
    def _dataset(self, sigma_rel=0.05, sigma_phase=0.05):
        freq = np.linspace(0.1e12, 2.0e12, 60)
        H = 0.5 * np.exp(-1j * 0.2 * freq / 1e12)
        pd = {
            "fft_freq": freq,
            "transfer_H": H,
            "transfer_H_sigma": np.full(freq.shape, np.abs(H).mean() * sigma_rel),
            "transfer_phase_sigma": np.full(freq.shape, sigma_phase),
        }
        return _FakeDataset({"sample_a.acc": _FakeObj(pd)}), pd

    def test_writes_sigma_keys_and_restores_originals(self):
        dataset, pd = self._dataset()
        H0 = pd["transfer_H"].copy()
        uncertainty.propagate_uncertainty(dataset, _reinvert_identity, n_draws=300, verbose=False)
        self.assertIn("n_sigma", pd)
        self.assertIn("k_sigma", pd)
        # transfer_H restored to the original after resampling
        np.testing.assert_allclose(pd["transfer_H"], H0)

    def test_sigma_scales_with_input(self):
        # n = |H| => sigma_n should approximately equal sigma_|H|
        dataset, pd = self._dataset(sigma_rel=0.05, sigma_phase=0.02)
        uncertainty.propagate_uncertainty(dataset, _reinvert_identity, n_draws=2000, verbose=False)
        expected_n = pd["transfer_H_sigma"]
        # k = arg(H) => sigma_k should approximately equal sigma_phase
        np.testing.assert_allclose(np.nanmedian(pd["n_sigma"]),
                                   np.nanmedian(expected_n), rtol=0.15)
        np.testing.assert_allclose(np.nanmedian(pd["k_sigma"]),
                                   np.nanmedian(pd["transfer_phase_sigma"]), rtol=0.15)

    def test_larger_input_gives_larger_bars(self):
        small, pd_small = self._dataset(sigma_rel=0.02)
        big, pd_big = self._dataset(sigma_rel=0.10)
        uncertainty.propagate_uncertainty(small, _reinvert_identity, n_draws=800, verbose=False)
        uncertainty.propagate_uncertainty(big, _reinvert_identity, n_draws=800, verbose=False)
        self.assertLess(np.nanmedian(pd_small["n_sigma"]), np.nanmedian(pd_big["n_sigma"]))

    def test_clean_restore_recovers_extra_keys(self):
        # A reflection-style inversion also writes reflection_r; the finally-block clean
        # re-inversion must restore it (not leave it at the last draw's perturbed value).
        dataset, pd = self._dataset()
        H0 = pd["transfer_H"].copy()

        def reinvert_reflection(ds):
            for name, obj in ds.data.items():
                if ds.data.is_reference(name):
                    continue
                H = obj.processing_dict["transfer_H"]
                obj.processing_dict["n"] = np.abs(H)
                obj.processing_dict["reflection_r"] = 2.0 * H   # extra downstream product

        uncertainty.propagate_uncertainty(dataset, reinvert_reflection, n_draws=100, verbose=False)
        np.testing.assert_allclose(pd["reflection_r"], 2.0 * H0)  # restored from true H
        self.assertIn("n_sigma", pd)

    def test_skips_without_transfer_sigma(self):
        freq = np.linspace(0.1e12, 2e12, 20)
        pd = {"fft_freq": freq, "transfer_H": np.ones(20, dtype=complex)}  # no *_sigma
        dataset = _FakeDataset({"sample_a.acc": _FakeObj(pd)})
        uncertainty.propagate_uncertainty(dataset, _reinvert_identity, n_draws=10, verbose=False)
        self.assertNotIn("n_sigma", pd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
