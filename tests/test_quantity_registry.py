"""Tests for the quantity registry, registry-driven export, and the new ResultsViewer."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_core.adapters import quantity_registry as registry
from dataset_core.adapters import results_viewer
from dataset_core.adapters import conductivity_fitting as cfit
from thz_core.thz_core.fitting import drude_conductivity


# ── synthetic processing_dict + fake dataset ─────────────────────────────────


def _synthetic_processing_dict(with_fit=True):
    freq = np.linspace(0.0, 2.5e12, 200)
    omega = 2 * np.pi * freq
    sigma = drude_conductivity(omega, 320.0, 1.8e-13)
    n = np.full_like(freq, 3.4)
    k = np.linspace(0.4, 0.02, freq.size)
    eps = (n + 1j * k) ** 2
    transfer_H = 0.5 * np.exp(-1j * 0.3 * freq / 1e12)
    fft_spectrum = np.exp(-((freq - 0.6e12) / 3e11) ** 2) * np.exp(1j * freq / 1e12)
    mask = (freq > 0.3e12) & (freq < 2.0e12)
    pd = {
        "fft_freq": freq, "fft_spectrum": fft_spectrum,
        "transfer_H": transfer_H, "transfer_mask": mask, "snr_mask": mask,
        "n": n, "k": k, "eps": eps, "sigma": sigma,
    }
    if with_fit:
        result, _ = cfit.joint_drude_fit(freq, sigma, mask, fit_band_thz=(0.3, 2.0))
        pd["fit_result"] = result
    return pd


class _FakeData(dict):
    def is_reference(self, name):
        return name.startswith("reference")


class _FakeObj:
    def __init__(self, pd):
        self.processing_dict = pd


class _FakeDataset:
    def __init__(self, samples, file_dir):
        self.data = _FakeData(samples)
        self.config = {"resolution": {"limit_to_instrument_resolution": False}}
        self.file_dir = file_dir


def _make_dataset(tmp):
    return _FakeDataset(
        {
            "sample_a.acc": _FakeObj(_synthetic_processing_dict(with_fit=True)),
            "reference.acc": _FakeObj(_synthetic_processing_dict(with_fit=False)),
        },
        tmp,
    )


# ── registry extraction ──────────────────────────────────────────────────────


class TestQuantityExtraction(unittest.TestCase):
    def setUp(self):
        self.pd = _synthetic_processing_dict()

    def test_all_builtin_quantities_extract(self):
        names = registry.available_names(self.pd)
        for expected in ["fft_mag", "transfer_mag", "n", "k", "eps_real", "sigma_real", "sigma_imag"]:
            self.assertIn(expected, names)

    def test_complex_components(self):
        freq, sigma_r = registry.QUANTITY_REGISTRY["sigma_real"].extract(self.pd)
        self.assertTrue(np.allclose(sigma_r, np.real(self.pd["sigma"])))
        _, sigma_i = registry.QUANTITY_REGISTRY["sigma_imag"].extract(self.pd)
        self.assertTrue(np.allclose(sigma_i, np.imag(self.pd["sigma"])))

    def test_missing_input_returns_none(self):
        self.assertIsNone(registry.QUANTITY_REGISTRY["sigma_real"].extract({"fft_freq": np.arange(5)}))

    def test_fit_overlay_matches_drude(self):
        overlay = registry.QUANTITY_REGISTRY["sigma_real"].overlay(self.pd)
        self.assertIsNotNone(overlay)
        freq, y = overlay
        omega = 2 * np.pi * freq
        fit = self.pd["fit_result"]
        params = {**fit.param_values, **fit.fixed_params}
        expected = np.real(drude_conductivity(omega, params["sigma_dc"], params["tau"]))
        self.assertTrue(np.allclose(y, expected))

    def test_overlay_none_without_fit(self):
        pd = _synthetic_processing_dict(with_fit=False)
        self.assertIsNone(registry.QUANTITY_REGISTRY["sigma_real"].overlay(pd))


# ── registry-driven export ───────────────────────────────────────────────────


class TestExport(unittest.TestCase):
    def test_export_headers_match_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(tmp)
            written = results_viewer.export_quantities(dataset, tmp)
            results_csv = [p for p in written if p.endswith("_results.csv")]
            self.assertEqual(len(results_csv), 1)  # only the sample, not the reference
            with open(results_csv[0]) as f:
                header = f.readline().strip().lstrip("# ").split(",")
            expected = ["freq_THz"] + [q.export_header for q in registry.export_quantities_list()]
            self.assertEqual(header, expected)
            data = np.loadtxt(results_csv[0], delimiter=",", skiprows=1)
            self.assertEqual(data.shape[1], len(expected))

    def test_missing_quantity_is_nan_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            pd = {"fft_freq": np.linspace(0, 2e12, 50)}  # only freq, nothing else
            dataset = _FakeDataset({"sample_a.acc": _FakeObj(pd)}, tmp)
            written = results_viewer.export_quantities(dataset, tmp)
            results_csv = [p for p in written if p.endswith("_results.csv")][0]
            data = np.loadtxt(results_csv, delimiter=",", skiprows=1)
            # freq column finite, all quantity columns NaN
            self.assertTrue(np.all(np.isfinite(data[:, 0])))
            self.assertTrue(np.all(np.isnan(data[:, 1:])))


# ── collaborator handoff: fit summary + README ───────────────────────────────


class TestFitSummaryAndReadme(unittest.TestCase):
    def test_fit_summary_has_scattering_rate_and_uncertainty_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(tmp)
            written = results_viewer.export_quantities(dataset, tmp)
            fit_csv = [p for p in written if p.endswith("fit_summary.csv")]
            self.assertEqual(len(fit_csv), 1)
            with open(fit_csv[0]) as f:
                rows = f.read().strip().splitlines()
            header = rows[0].split(",")
            for expected in ["filename", "model", "r_squared", "tau [s]", "tau [s] uncertainty",
                              "scattering_rate_Hz", "crossover_THz"]:
                self.assertIn(expected, header)
            # only the fitted, non-reference sample gets a row
            self.assertEqual(len(rows), 2)
            self.assertIn("sample_a.acc", rows[1])

    def test_readme_documents_columns_and_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(tmp)
            written = results_viewer.export_quantities(dataset, tmp)
            readme = [p for p in written if p.endswith("README.md")]
            self.assertEqual(len(readme), 1)
            text = open(readme[0]).read()
            # column glossary pulled from the registry (not hand-duplicated)
            for quantity in registry.export_quantities_list():
                self.assertIn(quantity.export_header, text)
            self.assertIn("sample_a.acc", text)
            self.assertIn("scattering rate", text)

    def test_no_fits_skips_fit_summary_and_says_so_in_readme(self):
        with tempfile.TemporaryDirectory() as tmp:
            pd = _synthetic_processing_dict(with_fit=False)
            dataset = _FakeDataset({"sample_a.acc": _FakeObj(pd)}, tmp)
            written = results_viewer.export_quantities(dataset, tmp)
            self.assertFalse(any(p.endswith("fit_summary.csv") for p in written))
            readme = [p for p in written if p.endswith("README.md")][0]
            self.assertIn("No stored fits", open(readme).read())

    def test_readme_false_skips_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(tmp)
            written = results_viewer.export_quantities(dataset, tmp, readme=False)
            self.assertFalse(any(p.endswith("fit_summary.csv") for p in written))
            self.assertFalse(any(p.endswith("README.md") for p in written))
            self.assertFalse(os.path.exists(os.path.join(tmp, "README.md")))


# ── new viewer (headless) ────────────────────────────────────────────────────


class TestResultsViewerHeadless(unittest.TestCase):
    def test_builds_and_switches_quantities(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _make_dataset(tmp)
            viewer = results_viewer.launch_results_viewer(dataset, show=False)
            # widgets pinned to the figure (GC guard)
            self.assertTrue(hasattr(viewer._fig, "_results_viewer_widgets"))
            # switching to every quantity must not raise (drives the registry-driven draw)
            for label in viewer._labels:
                viewer._on_quantity(label)
            # sample toggle + freq range must not raise
            viewer._on_toggle(viewer._short("sample_a.acc"))
            viewer._on_fmin("0.2")
            viewer._on_fmax("2.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
