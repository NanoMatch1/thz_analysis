"""Tests for the three bundle front-ends: open_session (view), extract_session_data (dicts),
display_cookbook (custom figures).

Two forms, per the project's testing philosophy:
  * unit tests on a synthetic dataset — extraction shapes, trusted-band masking, cookbook figures
  * a headless workflow test — open_session.main / extract_session_data on a real saved bundle,
    exactly as a user would run them (skipped if the data root is absent)
"""

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stdout

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import display_cookbook
import extract_session_data
import open_session
from dataset_core.adapters.catalog import Catalog
from test_display import _misalignment_dataset

# A gold-mirror misalignment bundle known to be in the data root (8 files, reflection).
_REAL_BUNDLE_ID = "49ad3cc4"


class TestExtractSeries(unittest.TestCase):
    def setUp(self):
        self.dataset = _misalignment_dataset()

    def test_quantity_then_filename_nesting(self):
        data = extract_session_data.extract_series(self.dataset, ["n", "k"])
        self.assertEqual(set(data), {"n", "k"})
        self.assertEqual(len(data["n"]), 4)                   # samples only for n
        arrays = next(iter(data["n"].values()))
        self.assertEqual(set(arrays), {"freq_hz", "freq_thz", "y", "error", "mask"})

    def test_files_selector_passes_through(self):
        data = extract_session_data.extract_series(self.dataset, ["n"], files="minus")
        self.assertEqual(len(data["n"]), 2)

    def test_trusted_applies_mask(self):
        arrays = extract_session_data.extract_series(self.dataset, ["n"])["n"]
        one = next(iter(arrays.values()))
        freq_thz, values = extract_session_data.trusted(one)
        self.assertEqual(len(freq_thz), int(np.sum(one["mask"])))
        self.assertTrue(freq_thz.min() > 0.3 and freq_thz.max() < 2.0)
        _, errors = extract_session_data.trusted(one, key="error")
        self.assertEqual(errors.shape, values.shape)

    def test_trusted_without_mask_returns_everything(self):
        arrays = {"freq_thz": np.arange(5.0), "y": np.ones(5), "mask": None}
        freq_thz, _ = extract_session_data.trusted(arrays)
        self.assertEqual(len(freq_thz), 5)

    def test_summarise_and_example_plot_run(self):
        with redirect_stdout(io.StringIO()) as captured:
            extract_session_data.summarise(self.dataset)
        self.assertIn("Extractable quantities", captured.getvalue())
        figures_before = len(plt.get_fignums())
        extract_session_data.example_custom_plot(self.dataset)
        self.assertEqual(len(plt.get_fignums()) - figures_before, 1)
        plt.close("all")


class TestCookbookHeadless(unittest.TestCase):
    def setUp(self):
        plt.close("all")  # figure counts below must not see figures other test files left open

    def tearDown(self):
        plt.close("all")

    def test_misalignment_figures_build(self):
        dataset = _misalignment_dataset()
        display_cookbook.show_misalignment_figures(dataset, displacement_key="_gold_", show=False)
        self.assertEqual(len(plt.get_fignums()), 7)            # time + fft mag/phase + n,k + sigma x2

    def test_cookbook_builds(self):
        with redirect_stdout(io.StringIO()):
            display_cookbook.display_adapter_cookbook(_misalignment_dataset(), show=False)
        self.assertGreaterEqual(len(plt.get_fignums()), 5)


def _real_bundle_available() -> bool:
    try:
        return len(Catalog().match(_REAL_BUNDLE_ID)) == 1
    except Exception:
        return False


@unittest.skipUnless(_real_bundle_available(), "real gold misalignment bundle not in the data root")
class TestWorkflowOnRealBundle(unittest.TestCase):
    """Run the front-ends end to end on a real saved bundle, headless."""

    def test_open_session_main_extract_without_viewer(self):
        with redirect_stdout(io.StringIO()) as captured:
            dataset = open_session.main([_REAL_BUNDLE_ID, "--no-viewer", "--extract"])
        self.assertEqual(len(dataset.data.data_dict), 8)
        self.assertIn("Registered quantities with data", captured.getvalue())
        # Saved on Windows, so the recorded source dir must have been mapped onto this machine.
        self.assertTrue(os.path.isdir(dataset.file_dir))

    def test_open_session_accepts_explicit_bundle_path(self):
        bundle_dir = Catalog().resolve_bundle_dir(_REAL_BUNDLE_ID)
        with redirect_stdout(io.StringIO()):
            dataset = open_session.main([bundle_dir, "--no-viewer"])
        self.assertEqual(len(dataset.data.data_dict), 8)

    def test_extract_session_data_on_real_bundle(self):
        with redirect_stdout(io.StringIO()):
            dataset = extract_session_data.load_dataset(_REAL_BUNDLE_ID)
            data = extract_session_data.extract_series(dataset, ["n", "k", "fft_mag"])
        self.assertGreater(len(data["n"]), 0)
        self.assertGreater(len(data["fft_mag"]), len(data["n"]))   # fft includes the reference
        traces = extract_session_data.display.get_time_series(dataset)
        self.assertEqual(len(traces), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
