"""Phase B: fit_interactive launcher plumbing (GUI injected, no display needed)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_core.adapters import fitting
from dataset_core.adapters import conductivity_fitting as cfit
from thz_core.thz_core.fitting import drude_conductivity


class _FakeData(dict):
    def is_reference(self, name):
        return name.startswith("reference")


class _FakeObj:
    def __init__(self, pd):
        self.processing_dict = pd


class _FakeDataset:
    def __init__(self, samples):
        self.data = _FakeData(samples)
        self.recipe = []


def _sample_pd(with_error=True):
    freq = np.linspace(0.1e12, 2.5e12, 120)
    omega = 2 * np.pi * freq
    sigma = drude_conductivity(omega, 300.0, 1.6e-13)
    mask = (freq > 0.3e12) & (freq < 2.0e12)
    pd = {"fft_freq": freq, "sigma": sigma, "transfer_mask": mask}
    if with_error:
        pd["sigma_real_sigma"] = np.abs(sigma) * 0.05
        pd["sigma_imag_sigma"] = np.abs(sigma) * 0.07
    return pd


class _FakeLauncher:
    """Stand-in for launch_fit_gui: records the kwargs, returns real FitResults."""
    def __init__(self):
        self.captured = {}

    def __call__(self, freq, data_dict=None, mask_dict=None, error_dict=None,
                 initial_model=None, config=None):
        self.captured = dict(freq=freq, data_dict=data_dict, mask_dict=mask_dict,
                             error_dict=error_dict, initial_model=initial_model, config=config)
        results = {}
        for name, data in data_dict.items():
            mask = (mask_dict or {}).get(name)
            result, _ = cfit.joint_drude_fit(freq, data, mask, fit_band_thz=(0.35, 1.6))
            results[name] = result
        return results


class TestFitInteractive(unittest.TestCase):
    def test_builds_error_dict_from_mc_sigmas(self):
        pd = _sample_pd(with_error=True)
        dataset = _FakeDataset({"sample_a.acc": _FakeObj(pd)})
        launcher = _FakeLauncher()
        fitting.fit_interactive(dataset, _gui_launcher=launcher)
        error = launcher.captured["error_dict"]["sample_a.acc"]
        expected = pd["sigma_real_sigma"] + 1j * pd["sigma_imag_sigma"]
        np.testing.assert_allclose(error, expected)

    def test_writes_fit_result_back_and_provenance(self):
        dataset = _FakeDataset({
            "sample_a.acc": _FakeObj(_sample_pd()),
            "reference.acc": _FakeObj(_sample_pd()),  # must be ignored
        })
        results = fitting.fit_interactive(dataset, _gui_launcher=_FakeLauncher())
        self.assertIn("sample_a.acc", results)
        self.assertNotIn("reference.acc", results)  # references excluded
        self.assertIn("fit_result", dataset.data["sample_a.acc"].processing_dict)
        # provenance recorded, flagged non-replayable
        entry = dataset.recipe[-1]
        self.assertEqual(entry["stage"], "fit_interactive")
        self.assertTrue(entry["provenance_only"])
        self.assertIn("sample_a.acc", entry["results"])
        self.assertIn("sigma_dc", entry["results"]["sample_a.acc"])

    def test_shared_freq_axis_enforced(self):
        pd_a = _sample_pd()
        pd_b = _sample_pd()
        pd_b["fft_freq"] = np.linspace(0.1e12, 2.5e12, 90)  # different length
        pd_b["sigma"] = pd_b["sigma"][:90]
        dataset = _FakeDataset({"a.acc": _FakeObj(pd_a), "b.acc": _FakeObj(pd_b)})
        with self.assertRaises(ValueError):
            fitting.fit_interactive(dataset, _gui_launcher=_FakeLauncher())

    def test_no_matching_samples_returns_empty(self):
        pd = _sample_pd()
        del pd["sigma"]  # nothing to fit
        dataset = _FakeDataset({"a.acc": _FakeObj(pd)})
        self.assertEqual(fitting.fit_interactive(dataset, _gui_launcher=_FakeLauncher()), {})

    def test_n_quantity_real_error(self):
        freq = np.linspace(0.1e12, 2.5e12, 100)
        pd = {"fft_freq": freq, "n": np.full(100, 3.4), "n_sigma": np.full(100, 0.01),
              "transfer_mask": np.ones(100, bool)}
        dataset = _FakeDataset({"a.acc": _FakeObj(pd)})
        launcher = _FakeLauncher()
        fitting.fit_interactive(dataset, quantity="n", model="drude_conductivity",
                                _gui_launcher=launcher)
        err = launcher.captured["error_dict"]["a.acc"]
        np.testing.assert_allclose(np.real(err), 0.01)   # real error present
        np.testing.assert_allclose(np.imag(err), 0.0)    # no imag error for a real quantity


class _ReportData(dict):
    @property
    def data_dict(self):
        return self

    def is_reference(self, name):
        return name.startswith("reference")


class _ReportDataset:
    def __init__(self, samples):
        self.data = _ReportData(samples)
        self.recipe = []
        self.seriesname = "t"
        self.file_dir = "x"


class TestReportFitsSection(unittest.TestCase):
    def test_report_includes_fit_params(self):
        import tempfile
        from dataset_core.adapters import session_bundle
        pd = _sample_pd()
        result, _ = cfit.joint_drude_fit(pd["fft_freq"], pd["sigma"], pd["transfer_mask"],
                                         fit_band_thz=(0.35, 1.6))
        pd["fit_result"] = result
        dataset = _ReportDataset({"sample_a.acc": _FakeObj(pd)})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.md")
            session_bundle.write_processing_report(dataset, path)
            text = open(path, encoding="utf-8").read()
        self.assertIn("## Fits", text)
        self.assertIn("drude_conductivity", text)
        self.assertIn("sigma_dc", text)
        self.assertIn("R²", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
